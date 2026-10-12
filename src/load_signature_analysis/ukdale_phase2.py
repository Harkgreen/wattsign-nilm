"""Assemble a single-household UK-DALE Phase 2 pilot with events and aggregate features."""

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
from sklearn.metrics import roc_auc_score, roc_curve

from .event_detection import DetectionResult, EventDetectionConfig, detect_on_intervals
from .event_features import (
    FEATURE_COLUMNS,
    GROUP_FEATURES,
    MASK_COLUMNS,
    EventWindowConfig,
    assess_window_quality,
    extract_event_window_features,
)


METADATA_COLUMNS = (
    "event_id",
    "window_overlap_cluster_id",
    "dataset",
    "house_id",
    "channel_id",
    "source_appliance",
    "semantic_class",
    "label",
    "event_timestamp",
    "event_utc",
    "onset_uncertainty_seconds",
    "session_duration_seconds",
    "active_duration_seconds",
    "pre_coverage",
    "post_coverage",
    "aggregate_maximum_raw_gap_seconds",
)


def _resolve_project_path(project_root: Path, value: str | Path) -> Path:
    return resolve_project_path(project_root, str(value))


def _stable_event_id(house_id: int, channel_id: int, timestamp: float) -> str:
    milliseconds = int(round(timestamp * 1_000.0))
    return f"uk_dale_h{house_id:02d}_c{channel_id:02d}_t{milliseconds}"


def _read_configuration(config_path: Path) -> dict[str, object]:
    with Path(config_path).open("rb") as handle:
        return tomllib.load(handle)


def _validate_active_power_source(
    project_root: Path,
    aggregate_inventory_path: Path,
    house_id: int,
    mains_path: Path,
) -> dict[str, object]:
    inventory = pd.read_csv(aggregate_inventory_path, encoding="utf-8-sig")
    candidates = inventory[
        (inventory["dataset"] == "uk_dale")
        & (inventory["house_id"] == house_id)
        & (inventory["source_kind"] == "alternate_high_frequency_mains")
    ]
    if len(candidates) != 1:
        raise ValueError(f"House {house_id} 缺少唯一的 UK-DALE mains.dat 来源登记")
    row = candidates.iloc[0]
    if not bool(row["active_power_physically_compatible"]):
        raise ValueError("所选总表来源不是与 REFIT 兼容的有功功率来源")
    if "active_power" not in str(row["measurement_type"]):
        raise ValueError("所选总表来源没有登记 active_power")
    registered_path = _resolve_project_path(project_root, str(row["source_file"])).resolve()
    if registered_path != mains_path.resolve():
        raise ValueError("配置中的 mains.dat 与 Phase 1 来源登记不一致")
    return row.to_dict()


def _load_device_specs(
    appliance_inventory_path: Path,
    house_id: int,
    configured_devices: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Read labels only from the Phase 1 inventory; configuration supplies detection thresholds only."""

    inventory = pd.read_csv(appliance_inventory_path, encoding="utf-8-sig")
    house_rows = inventory[
        (inventory["dataset"] == "uk_dale") & (inventory["house_id"] == house_id)
    ].set_index("channel_id")
    specs: list[dict[str, object]] = []
    for configured in configured_devices:
        channel_id = int(configured["channel_id"])
        if channel_id not in house_rows.index:
            raise ValueError(f"House {house_id} channel {channel_id} 不在 appliance inventory 中")
        taxonomy = house_rows.loc[channel_id]
        if isinstance(taxonomy, pd.DataFrame):
            raise ValueError(f"House {house_id} channel {channel_id} 在 inventory 中重复")
        if not bool(taxonomy["include_in_main_mvp"]):
            raise ValueError(
                f"House {house_id} channel {channel_id} 未通过 main + high-confidence 过滤"
            )
        if pd.isna(taxonomy["binary_label"]):
            raise ValueError(f"House {house_id} channel {channel_id} 没有二分类标签")
        if str(taxonomy["measurement_unit"]) != "W":
            raise ValueError(f"House {house_id} channel {channel_id} 不是 W 单位")
        detection = EventDetectionConfig(
            on_threshold_w=float(configured["on_threshold_w"]),
            off_threshold_w=float(configured["off_threshold_w"]),
            minimum_on_seconds=float(configured.get("minimum_on_seconds", 20.0)),
            merge_gap_seconds=float(configured.get("merge_gap_seconds", 10.0)),
            maximum_sample_gap_seconds=float(
                configured.get("maximum_sample_gap_seconds", 30.0)
            ),
        )
        specs.append(
            {
                "channel_id": channel_id,
                "source_appliance": str(taxonomy["canonical_label"]),
                "semantic_class": str(taxonomy["semantic_class"]),
                "label": int(taxonomy["binary_label"]),
                "detection": detection,
            }
        )
    if not any(spec["label"] == 1 for spec in specs):
        raise ValueError("试点至少需要一个可靠正类通道")
    if not any(spec["label"] == 0 for spec in specs):
        raise ValueError("试点至少需要一个可靠负类通道")
    return specs


def _detect_events(
    house_path: Path,
    house_id: int,
    specs: list[dict[str, object]],
) -> tuple[pd.DataFrame, dict[int, DetectionResult]]:
    rows: list[dict[str, object]] = []
    results: dict[int, DetectionResult] = {}
    for spec in specs:
        channel_id = int(spec["channel_id"])
        channel_path = house_path / f"channel_{channel_id}.dat"
        if not channel_path.exists():
            raise FileNotFoundError(channel_path)
        detection = detect_on_intervals(channel_path, spec["detection"])
        results[channel_id] = detection
        for interval in detection.intervals:
            timestamp = float(interval["start_timestamp"])
            rows.append(
                {
                    "event_id": _stable_event_id(house_id, channel_id, timestamp),
                    "dataset": "uk_dale",
                    "house_id": house_id,
                    "channel_id": channel_id,
                    "source_appliance": spec["source_appliance"],
                    "semantic_class": spec["semantic_class"],
                    "label": int(spec["label"]),
                    "event_timestamp": timestamp,
                    "event_utc": pd.to_datetime(timestamp, unit="s", utc=True).isoformat(),
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
                    "active_duration_seconds": float(interval["active_duration_seconds"]),
                    "mean_active_power_w": float(interval["mean_active_power_w"]),
                    "maximum_active_power_w": float(interval["maximum_active_power_w"]),
                    "touches_data_boundary": bool(interval["touches_data_boundary"]),
                    "end_reason": str(interval["end_reason"]),
                    "merged_segment_count": int(interval["merged_segment_count"]),
                }
            )
    if not rows:
        raise ValueError("配置通道没有检出任何满足最短持续时间的事件")
    events = pd.DataFrame(rows).sort_values(
        ["event_timestamp", "channel_id"], kind="stable"
    )
    if events["event_id"].duplicated().any():
        raise ValueError("事件 ID 不唯一")
    return events.reset_index(drop=True), results


def _append_reason(reasons: dict[int, list[str]], index: int, reason: str) -> None:
    if reason not in reasons[index]:
        reasons[index].append(reason)


def _prefilter_events(
    events: pd.DataFrame,
    detection_results: dict[int, DetectionResult],
    channel_labels: dict[int, int],
    window_config: EventWindowConfig,
    event_merge_tolerance_seconds: float,
    maximum_events_per_device: int | None,
) -> pd.DataFrame:
    """Apply trusted label coverage, concurrent onset and positive-activity overlap rules."""

    events = events.copy()
    reasons = {index: [] for index in events.index}
    channel_labels = {
        int(channel): int(label) for channel, label in channel_labels.items()
    }
    positive_channels = sorted(
        channel for channel, label in channel_labels.items() if label == 1
    )
    if not positive_channels:
        raise ValueError("事件表中没有正类通道")
    for channel in positive_channels:
        if not detection_results[channel].trusted_coverage_segments:
            raise ValueError(f"正类 channel {channel} 没有可用的可信覆盖区间")
    positive_coverage_start = max(
        detection_results[channel].trusted_coverage_segments[0][0]
        for channel in positive_channels
    )
    positive_coverage_end = min(
        detection_results[channel].trusted_coverage_segments[-1][1]
        for channel in positive_channels
    )

    for index, row in events.iterrows():
        if bool(row["touches_data_boundary"]):
            _append_reason(reasons, index, "submeter_data_boundary")
        window_start = float(row["event_timestamp"]) - window_config.pre_event_seconds
        window_end = float(row["event_timestamp"]) + window_config.post_event_seconds
        if window_start < positive_coverage_start or window_end > positive_coverage_end:
            _append_reason(reasons, index, "outside_positive_label_coverage")
            continue
        # Overall start/end times cannot express internal gaps; the window must fit within a trusted segment of each positive channel.
        for channel in positive_channels:
            segments = detection_results[channel].trusted_coverage_segments
            if not any(
                window_start >= segment_start and window_end <= segment_end
                for segment_start, segment_end in segments
            ):
                _append_reason(reasons, index, "positive_label_gap")
                break

    timestamps = events["event_timestamp"].to_numpy(dtype=np.float64)
    channels = events["channel_id"].to_numpy(dtype=np.int64)
    for left in range(len(events)):
        right = left + 1
        while right < len(events) and timestamps[right] - timestamps[left] <= event_merge_tolerance_seconds:
            if channels[left] != channels[right]:
                _append_reason(reasons, left, "concurrent_device_onset")
                _append_reason(reasons, right, "concurrent_device_onset")
            right += 1

    activity_by_label: dict[int, np.ndarray] = {}
    for label in (0, 1):
        interval_rows = []
        for channel, channel_label in channel_labels.items():
            if channel_label != label:
                continue
            for interval in detection_results[channel].activity_intervals:
                # Submeters observe onset only at sample times, so extend its uncertainty backward.
                interval_rows.append(
                    (
                        float(interval["start_timestamp"])
                        - float(interval.get("onset_uncertainty_seconds", 0.0)),
                        float(interval["end_timestamp"]),
                    )
                )
        activity_by_label[label] = np.asarray(interval_rows, dtype=np.float64).reshape(-1, 2)

    for index, row in events.iterrows():
        window_start = float(row["event_timestamp"]) - window_config.pre_event_seconds
        window_end = float(row["event_timestamp"]) + window_config.post_event_seconds
        opposite_intervals = activity_by_label[1 - int(row["label"])]
        overlap = (opposite_intervals[:, 0] < window_end) & (
            opposite_intervals[:, 1] > window_start
        )
        if np.any(overlap):
            _append_reason(reasons, index, "opposite_class_activity_overlap")

    events["prefilter_rejection_reasons"] = [";".join(reasons[index]) for index in events.index]
    events["selected_for_feature_extraction"] = events[
        "prefilter_rejection_reasons"
    ].eq("")

    if maximum_events_per_device is not None:
        for channel_id, group in events[events["selected_for_feature_extraction"]].groupby(
            "channel_id", sort=True
        ):
            if len(group) <= maximum_events_per_device:
                continue
            positions = np.linspace(
                0,
                len(group) - 1,
                maximum_events_per_device,
                dtype=np.int64,
            )
            keep = set(group.iloc[np.unique(positions)].index.tolist())
            for index in group.index:
                if index not in keep:
                    events.loc[index, "selected_for_feature_extraction"] = False
                    events.loc[index, "prefilter_rejection_reasons"] = "deterministic_device_cap"

    events.attrs["positive_label_coverage_start"] = positive_coverage_start
    events.attrs["positive_label_coverage_end"] = positive_coverage_end
    return events


def _extract_relative_mains_windows(
    mains_path: Path,
    event_timestamps: np.ndarray,
    config: EventWindowConfig,
    chunksize: int,
    invalid_zero_active_when_apparent_above_va: float,
    maximum_nonmonotonic_records: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Build half-open ten-second bins relative to each event t0 and aggregate raw samples by their median."""

    if maximum_nonmonotonic_records < 0:
        raise ValueError("非递增记录允许上限不能为负数")
    event_timestamps = np.asarray(event_timestamps, dtype=np.float64)
    bin_values: list[list[list[float]]] = [
        [[] for _ in range(config.total_bin_count)] for _ in event_timestamps
    ]
    maximum_raw_gaps = np.zeros(event_timestamps.size, dtype=np.float64)
    previous_event_timestamps = np.full(event_timestamps.size, np.nan, dtype=np.float64)
    aggregate_start: float | None = None
    aggregate_end: float | None = None
    previous_file_timestamp: float | None = None
    raw_record_count = 0
    valid_active_record_count = 0
    invalid_zero_active_count = 0
    nonmonotonic_count = 0

    reader = pd.read_csv(
        mains_path,
        sep=r"\s+",
        header=None,
        names=["timestamp", "active_power_w", "apparent_power_va", "rms_voltage_v"],
        usecols=[0, 1, 2],
        chunksize=chunksize,
    )
    for chunk in reader:
        timestamps = pd.to_numeric(chunk["timestamp"], errors="coerce").to_numpy(dtype=float)
        active = pd.to_numeric(chunk["active_power_w"], errors="coerce").to_numpy(dtype=float)
        apparent = pd.to_numeric(chunk["apparent_power_va"], errors="coerce").to_numpy(dtype=float)
        finite_timestamp = np.isfinite(timestamps)
        timestamps = timestamps[finite_timestamp]
        active = active[finite_timestamp]
        apparent = apparent[finite_timestamp]
        if timestamps.size == 0:
            continue
        raw_record_count += int(timestamps.size)

        previous_limit = (
            previous_file_timestamp
            if previous_file_timestamp is not None
            else -np.inf
        )
        prior_maximum = np.maximum.accumulate(
            np.concatenate(([previous_limit], timestamps[:-1]))
        )
        monotonic_keep = timestamps > prior_maximum
        nonmonotonic_count += int(np.count_nonzero(~monotonic_keep))
        if nonmonotonic_count > maximum_nonmonotonic_records:
            raise ValueError(
                "mains.dat 非递增记录超过配置上限；不能静默排序或连接时间段"
            )
        timestamps = timestamps[monotonic_keep]
        active = active[monotonic_keep]
        apparent = apparent[monotonic_keep]
        if timestamps.size == 0:
            continue
        previous_file_timestamp = float(timestamps[-1])
        aggregate_start = float(timestamps[0]) if aggregate_start is None else aggregate_start
        aggregate_end = float(timestamps[-1])

        zero_invalid = (
            np.isfinite(active)
            & np.isfinite(apparent)
            & (active == 0.0)
            & (apparent > invalid_zero_active_when_apparent_above_va)
        )
        invalid_zero_active_count += int(np.count_nonzero(zero_invalid))
        valid_active = np.isfinite(active) & ~zero_invalid
        valid_active_record_count += int(np.count_nonzero(valid_active))

        chunk_start = float(timestamps[0])
        chunk_end = float(timestamps[-1])
        relevant = np.flatnonzero(
            (event_timestamps + config.post_event_seconds > chunk_start)
            & (event_timestamps - config.pre_event_seconds <= chunk_end)
        )
        for event_index in relevant:
            event_timestamp = event_timestamps[event_index]
            window_start = event_timestamp - config.pre_event_seconds
            window_end = event_timestamp + config.post_event_seconds
            left = int(np.searchsorted(timestamps, window_start, side="left"))
            right = int(np.searchsorted(timestamps, window_end, side="left"))
            if left >= right:
                continue
            selected_timestamps = timestamps[left:right]
            selected_power = active[left:right]
            selected_valid = valid_active[left:right]

            valid_timestamps = selected_timestamps[selected_valid]
            if valid_timestamps.size:
                if np.isfinite(previous_event_timestamps[event_index]):
                    maximum_raw_gaps[event_index] = max(
                        maximum_raw_gaps[event_index],
                        float(
                            valid_timestamps[0]
                            - previous_event_timestamps[event_index]
                        ),
                    )
                if valid_timestamps.size >= 2:
                    maximum_raw_gaps[event_index] = max(
                        maximum_raw_gaps[event_index],
                        float(np.max(np.diff(valid_timestamps))),
                    )
                previous_event_timestamps[event_index] = float(valid_timestamps[-1])

            relative = valid_timestamps - window_start
            bin_indices = np.floor(relative / config.resample_interval_seconds).astype(np.int64)
            selected_power = selected_power[selected_valid]
            inside = (bin_indices >= 0) & (bin_indices < config.total_bin_count)
            for bin_index in np.unique(bin_indices[inside]):
                values = selected_power[inside & (bin_indices == bin_index)]
                bin_values[event_index][int(bin_index)].extend(values.tolist())

    medians = np.full(
        (event_timestamps.size, config.total_bin_count), np.nan, dtype=np.float64
    )
    counts = np.zeros((event_timestamps.size, config.total_bin_count), dtype=np.int32)
    for event_index, event_bins in enumerate(bin_values):
        for bin_index, values in enumerate(event_bins):
            if values:
                medians[event_index, bin_index] = float(np.median(values))
                counts[event_index, bin_index] = len(values)

    audit = {
        "source_file": mains_path.as_posix(),
        "raw_record_count": raw_record_count,
        "valid_active_record_count": valid_active_record_count,
        "invalid_zero_active_count": invalid_zero_active_count,
        "nonmonotonic_count": nonmonotonic_count,
        "coverage_start_timestamp": aggregate_start,
        "coverage_end_timestamp": aggregate_end,
    }
    return medians, counts, maximum_raw_gaps, audit


def _build_feature_tables(
    selected_events: pd.DataFrame,
    medians: np.ndarray,
    counts: np.ndarray,
    maximum_raw_gaps: np.ndarray,
    aggregate_audit: dict[str, object],
    config: EventWindowConfig,
    aggregate_quality_invalid_counts: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    feature_rows: list[dict[str, object]] = []
    window_rows: list[dict[str, object]] = []
    rejection_reasons: dict[str, str] = {}
    aggregate_start = float(aggregate_audit["coverage_start_timestamp"])
    aggregate_end = float(aggregate_audit["coverage_end_timestamp"])

    for position, (_, event) in enumerate(selected_events.iterrows()):
        pre = medians[position, : config.pre_bin_count]
        post = medians[position, config.pre_bin_count :]
        quality = assess_window_quality(pre, post, config)
        reasons = list(quality["rejection_reasons"])
        window_start = float(event["event_timestamp"]) - config.pre_event_seconds
        window_end = float(event["event_timestamp"]) + config.post_event_seconds
        if window_start < aggregate_start or window_end > aggregate_end:
            reasons.append("aggregate_window_boundary")
        if maximum_raw_gaps[position] > config.maximum_raw_gap_seconds:
            reasons.append("aggregate_raw_gap")
        if (
            aggregate_quality_invalid_counts is not None
            and int(aggregate_quality_invalid_counts[position]) > 0
        ):
            reasons.append("aggregate_quality_flag")
        reasons = list(dict.fromkeys(reasons))
        rejection_reasons[str(event["event_id"])] = ";".join(reasons)

        for bin_index in range(config.total_bin_count):
            relative_start = (
                -config.pre_event_seconds
                + bin_index * config.resample_interval_seconds
            )
            window_rows.append(
                {
                    "event_id": event["event_id"],
                    "relative_start_seconds": relative_start,
                    "relative_end_seconds": relative_start
                    + config.resample_interval_seconds,
                    "segment": "pre" if bin_index < config.pre_bin_count else "post",
                    "aggregate_active_power_w": medians[position, bin_index],
                    "raw_sample_count": int(counts[position, bin_index]),
                    "feature_window_accepted": not reasons,
                }
            )
        if reasons:
            continue

        row = {column: event[column] for column in METADATA_COLUMNS if column in event.index}
        row.update(
            {
                "pre_coverage": quality["pre_coverage"],
                "post_coverage": quality["post_coverage"],
                "aggregate_maximum_raw_gap_seconds": maximum_raw_gaps[position],
            }
        )
        row.update(extract_event_window_features(pre, post, config))
        feature_rows.append(row)

    features = pd.DataFrame(feature_rows)
    if not features.empty:
        ordered = [*METADATA_COLUMNS, *FEATURE_COLUMNS, *MASK_COLUMNS]
        features = features[[column for column in ordered if column in features.columns]]
        for feature_column, mask_column in zip(FEATURE_COLUMNS, MASK_COLUMNS, strict=True):
            observed = features[feature_column].notna().astype(int)
            if not observed.equals(features[mask_column].astype(int)):
                raise ValueError(f"{feature_column} 的值与 mask 不一致")
    return features, pd.DataFrame(window_rows), rejection_reasons


def _assign_window_overlap_clusters(
    features: pd.DataFrame,
    config: EventWindowConfig,
    house_id: int,
) -> pd.Series:
    """Group events with overlapping raw 180-second windows into connected clusters."""

    cluster_ids = pd.Series("", index=features.index, dtype="object")
    datasets = features["dataset"].astype(str).unique()
    if len(datasets) != 1:
        raise ValueError("重叠簇必须在单个 dataset-house 内计算")
    dataset = datasets[0]
    ordered = features.sort_values(
        ["event_timestamp", "channel_id", "event_id"], kind="stable"
    )
    cluster_number = 0
    current_right: float | None = None
    current_cluster = ""
    for index, event in ordered.iterrows():
        left = float(event["event_timestamp"]) - config.pre_event_seconds
        right = float(event["event_timestamp"]) + config.post_event_seconds
        # Adjacent but nonoverlapping half-open windows may belong to different clusters.
        if current_right is None or left >= current_right:
            cluster_number += 1
            current_cluster = f"{dataset}_h{house_id:02d}_overlap_{cluster_number:06d}"
            current_right = right
        else:
            current_right = max(current_right, right)
        cluster_ids.at[index] = current_cluster
    return cluster_ids


def _build_pilot_diagnostics(features: pd.DataFrame) -> dict[str, object]:
    """Quantify single-household class bias; these values are not generalization metrics."""

    if features.empty:
        return {}
    labels = features["label"].to_numpy(dtype=np.int64)
    counts = features["label"].value_counts().sort_index()
    diagnostics: dict[str, object] = {
        "class_counts": {str(int(label)): int(count) for label, count in counts.items()},
        "positive_fraction": float(np.mean(labels == 1)),
        "majority_class_fraction": float(counts.max() / len(features)),
        "observed_feature_fraction": float(
            features[list(MASK_COLUMNS)].to_numpy(dtype=np.float64).mean()
        ),
        "all_feature_masks_observed": bool(
            features[list(MASK_COLUMNS)].to_numpy(dtype=np.int64).all()
        ),
        "shortcut_diagnostic_scope": "同一单户样本上的分布可分性检查，不是验证集性能。",
    }
    shortcut: dict[str, dict[str, float]] = {}
    if len(np.unique(labels)) == 2:
        for feature in (
            "power__pre_mean",
            "power__post_mean",
            "power__delta_mean",
            "power__area_above_baseline",
        ):
            values = features[feature].to_numpy(dtype=np.float64)
            valid = np.isfinite(values)
            if np.unique(labels[valid]).size != 2:
                continue
            fpr, tpr, _ = roc_curve(labels[valid], values[valid])
            shortcut[feature] = {
                "roc_auc": float(roc_auc_score(labels[valid], values[valid])),
                "best_in_sample_balanced_accuracy": float(
                    np.max((tpr + 1.0 - fpr) / 2.0)
                ),
            }
    diagnostics["single_feature_shortcut"] = shortcut
    positive_delta = features.loc[
        features["label"] == 1, "power__delta_mean"
    ].to_numpy(dtype=np.float64)
    diagnostics["positive_alignment_audit"] = {
        "event_count": int(positive_delta.size),
        "negative_delta_mean_count": int(np.count_nonzero(positive_delta < 0.0)),
        "delta_mean_below_500w_count": int(np.count_nonzero(positive_delta < 500.0)),
        "note": "较小或为负的增量可能由分表采样时刻、未监测负载变化和 10 秒中位数聚合共同造成。",
    }
    return diagnostics


def _write_pilot_report(
    output_path: Path,
    manifest: dict[str, object],
) -> None:
    """Summarize event extraction, data quality and single-house feature diagnostics."""

    device_rows = []
    for channel, item in manifest["detection_summary"].items():
        device_rows.append(
            "| {channel} | {appliance} | {label} | {on:.0f}/{off:.0f} | {detected} | {accepted} |".format(
                channel=channel,
                appliance=item["source_appliance"],
                label=item["label"],
                on=item["detection_config"]["on_threshold_w"],
                off=item["detection_config"]["off_threshold_w"],
                detected=item["interval_count"],
                accepted=item["feature_accepted_count"],
            )
        )
    rejection_rows = [
        f"| `{reason}` | {count} |"
        for reason, count in manifest["rejection_counts"].items()
    ] or ["| 无 | 0 |"]
    window = manifest["window_config"]
    diagnostics = manifest["pilot_diagnostics"]
    shortcut_rows = [
        f"| `{feature}` | {values['roc_auc']:.3f} | {values['best_in_sample_balanced_accuracy']:.3f} |"
        for feature, values in diagnostics["single_feature_shortcut"].items()
    ]
    alignment = diagnostics["positive_alignment_audit"]
    lines = [
        "# UK-DALE House 5 Phase 2 试点报告",
        "",
        "报告覆盖事件级样本提取与输入特征检查，不包含分类模型性能。",
        "设备分表只用于生成标签；模型侧只读取 `mains.dat` 第 2 列有功功率 W，",
        "不使用第 3 列视在功率 VA，也不把设备名称、通道号或真实持续时间送入模型。",
        "",
        "## 事件提取结果",
        "",
        f"- 共检测 {manifest['event_count_detected']} 个候选事件。",
        f"- 预过滤后进入总表窗口提取 {manifest['event_count_selected_before_aggregate_qc']} 个。",
        f"- 总表质量检查后保留 {manifest['event_count_feature_accepted']} 个。",
        f"- 最终事件形成 {manifest['overlap_cluster_audit']['cluster_count']} 个互不重叠的时间簇；后续划分必须以簇为单位。",
        f"- 每个事件以分表首次观测到 ON 的 t0 为原点（并保存启动不确定度），前 {window['pre_event_seconds']} 秒、后 {window['post_event_seconds']} 秒，按 {window['resample_interval_seconds']} 秒半开区间取中位数。",
        "- 输入包含27个特征值和27个缺失mask，共54个标量；标准化统计量在训练户上拟合。",
        "",
        "| 通道 | 设备 | 标签 | ON/OFF 阈值(W) | 检出 | 最终保留 |",
        "|---:|---|---:|---:|---:|---:|",
        *device_rows,
        "",
        "## 拒绝原因",
        "",
        "| 原因代码 | 数量 |",
        "|---|---:|",
        *rejection_rows,
        "",
        "## 数据偏差与对齐检查",
        "",
        f"- 正类占 {diagnostics['positive_fraction']:.1%}，只猜多数类的表观准确率为 {diagnostics['majority_class_fraction']:.1%}。",
        f"- 真实 H5 样本的特征观测率为 {diagnostics['observed_feature_fraction']:.1%}；当前 mask 是否全部为 1：{diagnostics['all_feature_masks_observed']}。",
        f"- {alignment['event_count']} 个正类中，{alignment['negative_delta_mean_count']} 个 `delta_mean < 0`，{alignment['delta_mean_below_500w_count']} 个低于 500 W；这可能由分表采样时刻、其他负载变化和 10 秒中位数聚合共同造成。",
        "- 下表仅检查同一批数据是否存在单特征捷径，不是验证集性能：",
        "",
        "| 单特征 | ROC-AUC | 样本内最佳平衡准确率 |",
        "|---|---:|---:|",
        *shortcut_rows,
        "",
        "## 适用范围与局限",
        "",
        "- 单户特征试点未建立独立住宅的训练、验证和测试划分；样本内诊断不构成泛化性能。",
        "- 事件阈值按分表功率分布设定。跨户评测需要在使用测试结果前固定检测配置。",
        "- 异类活动排歧目前只覆盖配置中的 5 个通道；H5 其余设备及未分表负载仍可能进入总表窗口。",
        "- 正负类存在功率幅值差异，识别结果可能依赖幅值捷径。全特征与去幅值/形状特征对照可用于检查这一影响。",
        "- H5的mask可能全部为1，缺失处理主要由合成缺口测试覆盖。真实数据中的时间戳倒序、长缺口和异常零功率需单独检查。",
        "",
        "## 产物",
        "",
        "- `events.csv`：所有候选事件及拒绝原因。",
        "- `event_windows.csv`：逐事件 18 个相对时间 bin，便于画图核查。",
        "- `event_features.csv`：27 维特征、27 维 mask 和非模型元数据。",
        "- `manifest.json`：配置、字段顺序、统计与来源审计。",
        "- `config_snapshot.toml`：事件提取的配置快照。",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def build_ukdale_phase2_pilot(
    project_root: Path,
    config_path: Path,
    output_override: Path | None = None,
) -> dict[str, object]:
    """Run the House 5 pilot and write events, windows, features and audit manifests."""

    project_root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    raw_config = _read_configuration(config_path)
    run_config = raw_config["run"]
    path_config = raw_config["paths"]
    house_id = int(run_config["house_id"])
    house_path = _resolve_project_path(
        project_root, path_config["ukdale_root"]
    ) / f"house_{house_id}"
    mains_path = house_path / "mains.dat"
    appliance_inventory_path = _resolve_project_path(
        project_root, path_config["appliance_inventory"]
    )
    aggregate_inventory_path = _resolve_project_path(
        project_root, path_config["aggregate_source_inventory"]
    )
    output_dir = (
        _resolve_project_path(project_root, output_override)
        if output_override is not None
        else _resolve_project_path(project_root, path_config["output_dir"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    source_record = _validate_active_power_source(
        project_root, aggregate_inventory_path, house_id, mains_path
    )
    specs = _load_device_specs(
        appliance_inventory_path,
        house_id,
        list(raw_config["devices"]),
    )
    events, detection_results = _detect_events(house_path, house_id, specs)
    window_config = EventWindowConfig(**raw_config["window"])
    events = _prefilter_events(
        events,
        detection_results,
        {int(spec["channel_id"]): int(spec["label"]) for spec in specs},
        window_config,
        float(run_config["event_merge_tolerance_seconds"]),
        int(run_config["maximum_events_per_device"])
        if int(run_config["maximum_events_per_device"]) > 0
        else None,
    )
    positive_coverage = {
        "start_timestamp": events.attrs["positive_label_coverage_start"],
        "end_timestamp": events.attrs["positive_label_coverage_end"],
    }

    selected = events[events["selected_for_feature_extraction"]].copy()
    selected = selected.sort_values(["event_timestamp", "channel_id"], kind="stable")
    if selected.empty:
        raise ValueError("预过滤后没有可提取特征的事件")
    medians, counts, maximum_raw_gaps, aggregate_audit = _extract_relative_mains_windows(
        mains_path,
        selected["event_timestamp"].to_numpy(dtype=np.float64),
        window_config,
        chunksize=int(run_config["mains_chunksize"]),
        invalid_zero_active_when_apparent_above_va=float(
            run_config["invalid_zero_active_when_apparent_above_va"]
        ),
        maximum_nonmonotonic_records=int(
            run_config.get("maximum_nonmonotonic_records", 0)
        ),
    )
    features, windows, feature_rejections = _build_feature_tables(
        selected,
        medians,
        counts,
        maximum_raw_gaps,
        aggregate_audit,
        window_config,
    )
    if features.empty:
        raise ValueError("所有候选事件都未通过总表窗口质量检查")
    events["feature_rejection_reasons"] = events["event_id"].map(feature_rejections).fillna("")
    events["feature_accepted"] = (
        events["selected_for_feature_extraction"]
        & events["feature_rejection_reasons"].eq("")
    )
    if not features.empty:
        features["window_overlap_cluster_id"] = _assign_window_overlap_clusters(
            features, window_config, house_id
        )
        ordered_columns = [*METADATA_COLUMNS, *FEATURE_COLUMNS, *MASK_COLUMNS]
        features = features[ordered_columns]
        cluster_by_event = features.set_index("event_id")[
            "window_overlap_cluster_id"
        ]
        events["window_overlap_cluster_id"] = events["event_id"].map(cluster_by_event).fillna("")
    else:
        events["window_overlap_cluster_id"] = ""

    rejection_counter: Counter[str] = Counter()
    for value in events["prefilter_rejection_reasons"]:
        rejection_counter.update(reason for reason in str(value).split(";") if reason)
    for value in events["feature_rejection_reasons"]:
        rejection_counter.update(reason for reason in str(value).split(";") if reason)
    accepted_counts = (
        features.groupby(["label", "source_appliance"]).size().to_dict()
        if not features.empty
        else {}
    )
    cluster_sizes = (
        features.groupby("window_overlap_cluster_id").size()
        if not features.empty
        else pd.Series(dtype="int64")
    )
    overlap_cluster_audit = {
        "cluster_count": int(len(cluster_sizes)),
        "multi_event_cluster_count": int((cluster_sizes > 1).sum()),
        "events_in_multi_event_clusters": int(cluster_sizes[cluster_sizes > 1].sum()),
        "maximum_cluster_size": int(cluster_sizes.max()) if len(cluster_sizes) else 0,
        "split_rule": "训练、验证和测试必须按 window_overlap_cluster_id 整簇分配。",
    }
    pilot_diagnostics = _build_pilot_diagnostics(features)
    specs_by_channel = {int(spec["channel_id"]): spec for spec in specs}
    accepted_by_channel = (
        features.groupby("channel_id").size().to_dict() if not features.empty else {}
    )
    detection_summary = {}
    for channel, result in detection_results.items():
        spec = specs_by_channel[channel]
        detection_summary[str(channel)] = {
            "source_appliance": spec["source_appliance"],
            "semantic_class": spec["semantic_class"],
            "label": int(spec["label"]),
            "detection_config": asdict(spec["detection"]),
            "interval_count": len(result.intervals),
            "raw_activity_interval_count": len(result.activity_intervals),
            "feature_accepted_count": int(accepted_by_channel.get(channel, 0)),
            "coverage_start_timestamp": result.coverage_start_timestamp,
            "coverage_end_timestamp": result.coverage_end_timestamp,
            "trusted_coverage_segment_count": len(result.trusted_coverage_segments),
            "valid_record_count": result.valid_record_count,
            "gap_count": result.gap_count,
            "duplicate_or_backward_count": result.duplicate_or_backward_count,
            "short_interval_count": result.short_interval_count,
        }
    manifest = {
        "schema_version": "0.1.0-phase2-pilot",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "UK-DALE 单户事件级特征试点；未训练模型。",
        "dataset": "uk_dale",
        "house_id": house_id,
        "aggregate_source": source_record,
        "aggregate_audit": aggregate_audit,
        "positive_label_coverage": positive_coverage,
        "window_config": asdict(window_config),
        "event_count_detected": int(len(events)),
        "event_count_selected_before_aggregate_qc": int(
            events["selected_for_feature_extraction"].sum()
        ),
        "event_count_feature_accepted": int(events["feature_accepted"].sum()),
        "accepted_counts": {
            f"label_{int(label)}:{appliance}": int(count)
            for (label, appliance), count in accepted_counts.items()
        },
        "overlap_cluster_audit": overlap_cluster_audit,
        "pilot_diagnostics": pilot_diagnostics,
        "rejection_counts": dict(sorted(rejection_counter.items())),
        "detection_summary": detection_summary,
        "feature_schema": {
            "groups": {group: list(names) for group, names in GROUP_FEATURES.items()},
            "feature_columns": list(FEATURE_COLUMNS),
            "mask_columns": list(MASK_COLUMNS),
            "feature_count": len(FEATURE_COLUMNS),
            "model_scalar_input_count_with_masks": len(FEATURE_COLUMNS) * 2,
            "metadata_columns_not_model_inputs": list(METADATA_COLUMNS),
            "missing_rule": "原始特征保留 NaN 且 mask=0；标准化与填零留到训练阶段。",
        },
        "output_files": {
            "events": "events.csv",
            "features": "event_features.csv",
            "windows": "event_windows.csv",
            "config_snapshot": "config_snapshot.toml",
            "report": "pilot_report.md",
        },
    }
    # After all computations succeed, replace staged files individually and commit the manifest last.
    with tempfile.TemporaryDirectory(
        prefix="phase2_ukdale_staging_", dir=output_dir.parent
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
        _write_pilot_report(staging / "pilot_report.md", manifest)
        with (staging / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(_json_safe(manifest), handle, ensure_ascii=False, indent=2)
        for filename in (
            "events.csv",
            "event_features.csv",
            "event_windows.csv",
            "config_snapshot.toml",
            "pilot_report.md",
            "manifest.json",
        ):
            (staging / filename).replace(output_dir / filename)
    return manifest
