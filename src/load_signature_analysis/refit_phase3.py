"""Generate REFIT event features with the same schema as UK-DALE Phase 2."""

from __future__ import annotations

from .data_paths import resolve_project_path

from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile
import tomllib

import numpy as np
import pandas as pd

from .data_adapters import ApplianceChannel, RefitCsvAdapter
from .event_detection import (
    DetectionResult,
    EventDetectionConfig,
    detect_on_intervals_array,
)
from .event_features import FEATURE_COLUMNS, MASK_COLUMNS, EventWindowConfig
from .ukdale_phase2 import (
    METADATA_COLUMNS,
    _assign_window_overlap_clusters,
    _build_feature_tables,
    _json_safe,
    _prefilter_events,
)


def _resolve_project_path(project_root: Path, value: str | Path) -> Path:
    return resolve_project_path(project_root, str(value))


def _stable_event_id(house_id: int, channel_id: int, timestamp: float) -> str:
    milliseconds = int(round(timestamp * 1_000.0))
    return f"refit_h{house_id:02d}_c{channel_id:02d}_t{milliseconds}"


def _detection_config(
    channel: ApplianceChannel,
    thresholds: dict[str, dict[str, object]],
) -> EventDetectionConfig:
    if channel.canonical_label not in thresholds:
        raise ValueError(f"缺少 {channel.canonical_label} 的 REFIT 事件阈值")
    values = thresholds[channel.canonical_label]
    return EventDetectionConfig(
        on_threshold_w=float(values["on_threshold_w"]),
        off_threshold_w=float(values["off_threshold_w"]),
        minimum_on_seconds=float(values["minimum_on_seconds"]),
        merge_gap_seconds=float(values["merge_gap_seconds"]),
        maximum_sample_gap_seconds=float(values["maximum_sample_gap_seconds"]),
    )


def _detect_house_events(
    house_data: object,
    channels: list[ApplianceChannel],
    thresholds: dict[str, dict[str, object]],
) -> tuple[pd.DataFrame, dict[int, DetectionResult], dict[int, EventDetectionConfig]]:
    rows: list[dict[str, object]] = []
    results: dict[int, DetectionResult] = {}
    configs: dict[int, EventDetectionConfig] = {}
    for channel in channels:
        config = _detection_config(channel, thresholds)
        configs[channel.channel_id] = config
        result = detect_on_intervals_array(
            house_data.timestamp_s,
            house_data.appliance_power_w[channel.channel_id],
            house_data.appliance_valid_mask,
            config,
        )
        results[channel.channel_id] = result
        for interval in result.intervals:
            timestamp = float(interval["start_timestamp"])
            rows.append(
                {
                    "event_id": _stable_event_id(
                        house_data.house_id, channel.channel_id, timestamp
                    ),
                    "dataset": "refit",
                    "house_id": int(house_data.house_id),
                    "channel_id": channel.channel_id,
                    "source_appliance": channel.canonical_label,
                    "semantic_class": channel.semantic_class,
                    "label": channel.binary_label,
                    "event_timestamp": timestamp,
                    "event_utc": pd.to_datetime(
                        timestamp, unit="s", utc=True
                    ).isoformat(),
                    "end_timestamp": float(interval["end_timestamp"]),
                    "end_utc": pd.to_datetime(
                        float(interval["end_timestamp"]), unit="s", utc=True
                    ).isoformat(),
                    "onset_uncertainty_seconds": float(
                        interval["onset_uncertainty_seconds"]
                    ),
                    "session_duration_seconds": float(
                        interval["session_duration_seconds"]
                    ),
                    "active_duration_seconds": float(
                        interval["active_duration_seconds"]
                    ),
                    "mean_active_power_w": float(interval["mean_active_power_w"]),
                    "maximum_active_power_w": float(
                        interval["maximum_active_power_w"]
                    ),
                    "touches_data_boundary": bool(
                        interval["touches_data_boundary"]
                    ),
                    "end_reason": str(interval["end_reason"]),
                    "merged_segment_count": int(interval["merged_segment_count"]),
                }
            )
    if not rows:
        raise ValueError(f"REFIT House {house_data.house_id} 没有检出候选事件")
    events = pd.DataFrame(rows).sort_values(
        ["event_timestamp", "channel_id"], kind="stable"
    )
    if events["event_id"].duplicated().any():
        raise ValueError("REFIT 事件 ID 不唯一")
    return events.reset_index(drop=True), results, configs


def _extract_array_windows(
    timestamps: np.ndarray,
    aggregate_power_w: np.ndarray,
    valid_mask: np.ndarray,
    event_timestamps: np.ndarray,
    config: EventWindowConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Extract aggregate medians on the event-relative half-open grid without interpolating across gaps."""

    timestamps = np.asarray(timestamps, dtype=np.float64)
    aggregate_power_w = np.asarray(aggregate_power_w, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    event_timestamps = np.asarray(event_timestamps, dtype=np.float64)
    if np.any(np.diff(timestamps[np.isfinite(timestamps)]) <= 0.0):
        raise ValueError("REFIT Unix 时间戳必须严格递增")

    medians = np.full(
        (event_timestamps.size, config.total_bin_count), np.nan, dtype=np.float64
    )
    counts = np.zeros(
        (event_timestamps.size, config.total_bin_count), dtype=np.int32
    )
    maximum_raw_gaps = np.zeros(event_timestamps.size, dtype=np.float64)
    quality_invalid_counts = np.zeros(event_timestamps.size, dtype=np.int32)
    for event_index, event_timestamp in enumerate(event_timestamps):
        window_start = event_timestamp - config.pre_event_seconds
        window_end = event_timestamp + config.post_event_seconds
        left = int(np.searchsorted(timestamps, window_start, side="left"))
        right = int(np.searchsorted(timestamps, window_end, side="left"))
        selected_timestamps = timestamps[left:right]
        selected_power = aggregate_power_w[left:right]
        selected_valid = valid_mask[left:right]
        quality_invalid_counts[event_index] = int(np.count_nonzero(~selected_valid))
        valid_timestamps = selected_timestamps[selected_valid]
        valid_power = selected_power[selected_valid]
        if valid_timestamps.size >= 2:
            maximum_raw_gaps[event_index] = float(
                np.max(np.diff(valid_timestamps))
            )
        if valid_timestamps.size == 0:
            continue
        relative = valid_timestamps - window_start
        bin_indices = np.floor(
            relative / config.resample_interval_seconds
        ).astype(np.int64)
        inside = (bin_indices >= 0) & (bin_indices < config.total_bin_count)
        for bin_index in np.unique(bin_indices[inside]):
            values = valid_power[inside & (bin_indices == bin_index)]
            medians[event_index, int(bin_index)] = float(np.median(values))
            counts[event_index, int(bin_index)] = int(values.size)

    finite_timestamps = timestamps[np.isfinite(timestamps)]
    audit = {
        "raw_record_count": int(timestamps.size),
        "valid_active_record_count": int(np.count_nonzero(valid_mask)),
        "quality_invalid_record_count": int(np.count_nonzero(~valid_mask)),
        "coverage_start_timestamp": float(finite_timestamps[0]),
        "coverage_end_timestamp": float(finite_timestamps[-1]),
        "nonmonotonic_count": 0,
    }
    return medians, counts, maximum_raw_gaps, quality_invalid_counts, audit


def _build_house(
    adapter: RefitCsvAdapter,
    house_id: int,
    taxonomy_roles: list[str],
    thresholds: dict[str, dict[str, object]],
    window_config: EventWindowConfig,
    event_merge_tolerance_seconds: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    channels = adapter.appliance_channels(house_id, taxonomy_roles)
    if not any(channel.binary_label == 1 for channel in channels):
        raise ValueError(f"REFIT House {house_id} 没有可靠正类通道")
    house_data = adapter.load_house(house_id, channels)
    events, detection_results, detection_configs = _detect_house_events(
        house_data, channels, thresholds
    )
    events = _prefilter_events(
        events,
        detection_results,
        {channel.channel_id: channel.binary_label for channel in channels},
        window_config,
        event_merge_tolerance_seconds,
        maximum_events_per_device=None,
    )
    selected = events[events["selected_for_feature_extraction"]].copy()
    selected = selected.sort_values(["event_timestamp", "channel_id"], kind="stable")
    if selected.empty:
        raise ValueError(f"REFIT House {house_id} 预过滤后没有事件")
    medians, counts, maximum_gaps, invalid_counts, aggregate_audit = (
        _extract_array_windows(
            house_data.timestamp_s,
            house_data.aggregate_power_w,
            house_data.aggregate_valid_mask,
            selected["event_timestamp"].to_numpy(dtype=np.float64),
            window_config,
        )
    )
    aggregate_audit["source_file"] = house_data.source_path.as_posix()
    features, windows, feature_rejections = _build_feature_tables(
        selected,
        medians,
        counts,
        maximum_gaps,
        aggregate_audit,
        window_config,
        aggregate_quality_invalid_counts=invalid_counts,
    )
    events["feature_rejection_reasons"] = (
        events["event_id"].map(feature_rejections).fillna("")
    )
    events["feature_accepted"] = (
        events["selected_for_feature_extraction"]
        & events["feature_rejection_reasons"].eq("")
    )
    if not features.empty:
        features["window_overlap_cluster_id"] = _assign_window_overlap_clusters(
            features, window_config, house_id
        )
        features = features[[*METADATA_COLUMNS, *FEATURE_COLUMNS, *MASK_COLUMNS]]
        cluster_by_event = features.set_index("event_id")[
            "window_overlap_cluster_id"
        ]
        events["window_overlap_cluster_id"] = (
            events["event_id"].map(cluster_by_event).fillna("")
        )
    else:
        events["window_overlap_cluster_id"] = ""

    rejection_counter: Counter[str] = Counter()
    for column in ("prefilter_rejection_reasons", "feature_rejection_reasons"):
        for value in events[column]:
            rejection_counter.update(
                reason for reason in str(value).split(";") if reason
            )
    channel_by_id = {channel.channel_id: channel for channel in channels}
    house_summary = {
        "house_id": house_id,
        "source_audit": house_data.audit,
        "aggregate_audit": aggregate_audit,
        "event_count_detected": int(len(events)),
        "event_count_selected": int(
            events["selected_for_feature_extraction"].sum()
        ),
        "event_count_feature_accepted": int(events["feature_accepted"].sum()),
        "accepted_counts": {
            f"label_{int(label)}:{appliance}": int(count)
            for (label, appliance), count in features.groupby(
                ["label", "source_appliance"]
            ).size().items()
        }
        if not features.empty
        else {},
        "rejection_counts": dict(sorted(rejection_counter.items())),
        "channels": {
            str(channel_id): {
                "source_appliance": channel_by_id[channel_id].canonical_label,
                "label": channel_by_id[channel_id].binary_label,
                "detection_config": asdict(detection_configs[channel_id]),
                "interval_count": len(result.intervals),
                "raw_activity_interval_count": len(result.activity_intervals),
                "trusted_coverage_segment_count": len(
                    result.trusted_coverage_segments
                ),
            }
            for channel_id, result in detection_results.items()
        },
    }
    return events, features, windows, house_summary


def build_refit_phase3_features(
    project_root: Path,
    config_path: Path,
    output_override: Path | None = None,
    houses_override: list[int] | None = None,
) -> dict[str, object]:
    """Process configured REFIT households and atomically write the common feature table."""

    project_root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    with config_path.open("rb") as handle:
        raw_config = tomllib.load(handle)
    paths = raw_config["paths"]
    run = raw_config["refit"]
    houses = (
        [int(value) for value in houses_override]
        if houses_override is not None
        else [int(value) for value in run["houses"]]
    )
    if len(houses) != len(set(houses)):
        raise ValueError("REFIT houses 配置不能重复")
    output_dir = (
        _resolve_project_path(project_root, output_override)
        if output_override is not None
        else _resolve_project_path(project_root, paths["refit_output_dir"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter = RefitCsvAdapter(
        project_root,
        Path(paths["appliance_inventory"]),
        Path(paths["aggregate_source_inventory"]),
        chunksize=int(run["chunksize"]),
    )
    unknown_houses = sorted(set(houses) - set(adapter.list_houses()))
    if unknown_houses:
        raise ValueError(f"REFIT 主池中不存在住宅：{unknown_houses}")
    window_config = EventWindowConfig(**raw_config["window"])
    thresholds = {
        str(name): dict(values) for name, values in raw_config["thresholds"].items()
    }

    event_tables: list[pd.DataFrame] = []
    feature_tables: list[pd.DataFrame] = []
    window_tables: list[pd.DataFrame] = []
    house_summaries: dict[str, object] = {}
    for house_id in houses:
        events, features, windows, summary = _build_house(
            adapter,
            house_id,
            [str(value) for value in run["taxonomy_roles"]],
            thresholds,
            window_config,
            float(run["event_merge_tolerance_seconds"]),
        )
        event_tables.append(events)
        feature_tables.append(features)
        window_tables.append(windows)
        house_summaries[str(house_id)] = summary

    events = pd.concat(event_tables, ignore_index=True)
    features = pd.concat(feature_tables, ignore_index=True)
    windows = pd.concat(window_tables, ignore_index=True)
    if events["event_id"].duplicated().any() or features["event_id"].duplicated().any():
        raise ValueError("合并后的 REFIT 事件 ID 不唯一")
    manifest = {
        "schema_version": "0.2.0-refit-phase3",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "REFIT event-level aggregate feature table；尚未划分或训练。",
        "dataset": "refit",
        "houses": houses,
        "taxonomy_roles": list(run["taxonomy_roles"]),
        "window_config": asdict(window_config),
        "event_count_detected": int(len(events)),
        "event_count_feature_accepted": int(len(features)),
        "accepted_counts": {
            f"label_{int(label)}:{appliance}": int(count)
            for (label, appliance), count in features.groupby(
                ["label", "source_appliance"]
            ).size().items()
        },
        "house_summaries": house_summaries,
        "feature_columns": list(FEATURE_COLUMNS),
        "mask_columns": list(MASK_COLUMNS),
        "quality_policy": {
            "unix_is_time_authority": True,
            "aggregate_active_power_unit": "W",
            "issues_rows_are_invalid": True,
            "aggregate_nonpositive_is_invalid": True,
            "house_1_has_no_issues_field": True,
            "interpolation": "none",
        },
        "output_files": {
            "events": "events.csv",
            "features": "event_features.csv",
            "windows": "event_windows.csv",
            "config_snapshot": "config_snapshot.toml",
        },
    }
    with tempfile.TemporaryDirectory(
        prefix="refit_phase3_staging_", dir=output_dir.parent
    ) as temporary_directory:
        staging = Path(temporary_directory)
        events.to_csv(staging / "events.csv", index=False, encoding="utf-8-sig")
        features.to_csv(
            staging / "event_features.csv", index=False, encoding="utf-8-sig"
        )
        windows.to_csv(
            staging / "event_windows.csv", index=False, encoding="utf-8-sig"
        )
        shutil.copyfile(config_path, staging / "config_snapshot.toml")
        with (staging / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(_json_safe(manifest), handle, ensure_ascii=False, indent=2)
        for filename in (
            "events.csv",
            "event_features.csv",
            "event_windows.csv",
            "config_snapshot.toml",
            "manifest.json",
        ):
            (staging / filename).replace(output_dir / filename)
    return manifest
