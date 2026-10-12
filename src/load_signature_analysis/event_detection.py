"""Detect ON events in individual appliance submeters with hysteresis and gap protection."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EventDetectionConfig:
    """Event-detection parameters for one appliance channel."""

    on_threshold_w: float
    off_threshold_w: float
    minimum_on_seconds: float = 20.0
    merge_gap_seconds: float = 10.0
    maximum_sample_gap_seconds: float = 30.0

    def __post_init__(self) -> None:
        values = (
            self.on_threshold_w,
            self.off_threshold_w,
            self.minimum_on_seconds,
            self.merge_gap_seconds,
            self.maximum_sample_gap_seconds,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("事件检测配置不能包含 NaN 或无穷值")
        if self.on_threshold_w <= self.off_threshold_w:
            raise ValueError("ON 阈值必须高于 OFF 阈值")
        if self.off_threshold_w < 0:
            raise ValueError("OFF 阈值不能为负数")
        if self.minimum_on_seconds < 0 or self.merge_gap_seconds < 0:
            raise ValueError("持续时间和合并间隔不能为负数")
        if self.maximum_sample_gap_seconds <= 0:
            raise ValueError("最大采样缺口必须大于 0")


@dataclass(frozen=True)
class DetectionResult:
    """Detected intervals and channel-coverage quality statistics."""

    intervals: tuple[dict[str, object], ...]
    activity_intervals: tuple[dict[str, object], ...]
    trusted_coverage_segments: tuple[tuple[float, float], ...]
    coverage_start_timestamp: float | None
    coverage_end_timestamp: float | None
    valid_record_count: int
    gap_count: int
    duplicate_or_backward_count: int
    short_interval_count: int


def _close_interval(
    active: dict[str, object],
    end_timestamp: float,
    end_reason: str,
) -> dict[str, object]:
    interval = dict(active)
    interval["end_timestamp"] = float(end_timestamp)
    interval["end_reason"] = end_reason
    interval["active_duration_seconds"] = max(
        0.0,
        float(interval["last_active_timestamp"]) - float(interval["start_timestamp"]),
    )
    interval["session_duration_seconds"] = max(
        0.0,
        float(end_timestamp) - float(interval["start_timestamp"]),
    )
    sample_count = int(interval["sample_count"])
    interval["mean_active_power_w"] = (
        float(interval["power_sum"]) / sample_count if sample_count else 0.0
    )
    return interval


def _merge_intervals(
    intervals: list[dict[str, object]],
    maximum_gap_seconds: float,
) -> list[dict[str, object]]:
    """Merge only short normal-OFF gaps; never merge across missing data or file boundaries."""

    merged: list[dict[str, object]] = []
    for interval in intervals:
        if not merged:
            merged.append(dict(interval))
            continue
        previous = merged[-1]
        gap = float(interval["start_timestamp"]) - float(previous["end_timestamp"])
        can_merge = (
            gap <= maximum_gap_seconds
            and previous["end_reason"] == "off_threshold"
            and not bool(interval["start_after_gap"])
        )
        if not can_merge:
            merged.append(dict(interval))
            continue

        previous_count = int(previous["sample_count"])
        current_count = int(interval["sample_count"])
        total_count = previous_count + current_count
        total_power = float(previous["power_sum"]) + float(interval["power_sum"])
        previous["end_timestamp"] = interval["end_timestamp"]
        previous["last_active_timestamp"] = interval["last_active_timestamp"]
        previous["end_reason"] = interval["end_reason"]
        previous["active_duration_seconds"] = float(previous["active_duration_seconds"]) + float(
            interval["active_duration_seconds"]
        )
        previous["session_duration_seconds"] = float(previous["end_timestamp"]) - float(
            previous["start_timestamp"]
        )
        previous["sample_count"] = total_count
        previous["power_sum"] = total_power
        previous["mean_active_power_w"] = total_power / total_count
        previous["maximum_active_power_w"] = max(
            float(previous["maximum_active_power_w"]),
            float(interval["maximum_active_power_w"]),
        )
        previous["merged_segment_count"] = int(previous["merged_segment_count"]) + 1
    return merged


def detect_on_intervals(
    channel_path: Path,
    config: EventDetectionConfig,
) -> DetectionResult:
    """Detect OFF-to-ON intervals as a stream; unknown state must observe OFF before producing an event."""

    channel_path = Path(channel_path)
    intervals: list[dict[str, object]] = []
    active: dict[str, object] | None = None
    state = "unknown"
    previous_timestamp: float | None = None
    coverage_start: float | None = None
    coverage_end: float | None = None
    valid_record_count = 0
    gap_count = 0
    duplicate_or_backward_count = 0
    gap_before_current = False
    trusted_segment_start: float | None = None
    trusted_coverage_segments: list[tuple[float, float]] = []

    with channel_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                timestamp = float(parts[0])
                power = float(parts[1])
            except ValueError:
                continue
            if not (math.isfinite(timestamp) and math.isfinite(power)):
                continue

            if previous_timestamp is not None and timestamp <= previous_timestamp:
                duplicate_or_backward_count += 1
                continue

            valid_record_count += 1
            if coverage_start is None:
                coverage_start = timestamp
            coverage_end = timestamp
            gap_before_current = False

            if previous_timestamp is not None:
                sample_gap = timestamp - previous_timestamp
                if sample_gap > config.maximum_sample_gap_seconds:
                    gap_count += 1
                    gap_before_current = True
                    if trusted_segment_start is not None:
                        trusted_coverage_segments.append(
                            (trusted_segment_start, previous_timestamp)
                        )
                    trusted_segment_start = None
                    if active is not None:
                        intervals.append(
                            _close_interval(active, previous_timestamp, "data_gap")
                        )
                    active = None
                    state = "unknown"

            if state == "unknown":
                if power <= config.off_threshold_w:
                    state = "off"
                    # Observing OFF after a gap makes subsequent positive labels trustworthy again.
                    trusted_segment_start = timestamp
            elif state == "off":
                if power >= config.on_threshold_w:
                    onset_uncertainty = (
                        timestamp - previous_timestamp
                        if previous_timestamp is not None
                        else config.maximum_sample_gap_seconds
                    )
                    active = {
                        "start_timestamp": timestamp,
                        "last_active_timestamp": timestamp,
                        "onset_uncertainty_seconds": float(onset_uncertainty),
                        "start_after_gap": gap_before_current,
                        "power_sum": power,
                        "sample_count": 1,
                        "maximum_active_power_w": power,
                        "merged_segment_count": 1,
                    }
                    state = "on"
            elif state == "on":
                if power <= config.off_threshold_w:
                    if active is not None:
                        intervals.append(_close_interval(active, timestamp, "off_threshold"))
                    active = None
                    state = "off"
                elif active is not None:
                    active["last_active_timestamp"] = timestamp
                    active["power_sum"] = float(active["power_sum"]) + power
                    active["sample_count"] = int(active["sample_count"]) + 1
                    active["maximum_active_power_w"] = max(
                        float(active["maximum_active_power_w"]), power
                    )
            previous_timestamp = timestamp

    if active is not None and previous_timestamp is not None:
        intervals.append(_close_interval(active, previous_timestamp, "file_boundary"))
    if trusted_segment_start is not None and previous_timestamp is not None:
        trusted_coverage_segments.append((trusted_segment_start, previous_timestamp))

    merged = _merge_intervals(intervals, config.merge_gap_seconds)
    accepted: list[dict[str, object]] = []
    short_interval_count = 0
    for interval in merged:
        interval["touches_data_boundary"] = interval["end_reason"] in {
            "data_gap",
            "file_boundary",
        }
        if float(interval["active_duration_seconds"]) < config.minimum_on_seconds:
            short_interval_count += 1
            continue
        accepted.append(interval)

    return DetectionResult(
        intervals=tuple(accepted),
        # Excluding negative contamination must include short pulses and observed boundary activity.
        activity_intervals=tuple(merged),
        trusted_coverage_segments=tuple(trusted_coverage_segments),
        coverage_start_timestamp=coverage_start,
        coverage_end_timestamp=coverage_end,
        valid_record_count=valid_record_count,
        gap_count=gap_count,
        duplicate_or_backward_count=duplicate_or_backward_count,
        short_interval_count=short_interval_count,
    )


def detect_on_intervals_array(
    timestamps: np.ndarray,
    power_w: np.ndarray,
    valid_mask: np.ndarray,
    config: EventDetectionConfig,
) -> DetectionResult:
    """Apply the file detector hysteresis rules to arrays, explicitly separating invalid regions."""

    timestamps = np.asarray(timestamps, dtype=np.float64)
    power_w = np.asarray(power_w, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if timestamps.ndim != 1 or power_w.ndim != 1 or valid_mask.ndim != 1:
        raise ValueError("事件检测输入必须是一维数组")
    if not (timestamps.size == power_w.size == valid_mask.size):
        raise ValueError("时间戳、功率和有效 mask 的长度必须一致")

    finite = valid_mask & np.isfinite(timestamps) & np.isfinite(power_w)
    valid_indices = np.flatnonzero(finite)
    if valid_indices.size == 0:
        return DetectionResult((), (), (), None, None, 0, 0, 0, 0)

    timestamp_differences = np.diff(timestamps[valid_indices])
    row_differences = np.diff(valid_indices)
    duplicate_or_backward_count = int(
        np.count_nonzero((row_differences == 1) & (timestamp_differences <= 0.0))
    )
    break_positions = np.flatnonzero(
        (row_differences != 1)
        | (timestamp_differences <= 0.0)
        | (timestamp_differences > config.maximum_sample_gap_seconds)
    ) + 1
    segments = np.split(valid_indices, break_positions)

    raw_intervals: list[dict[str, object]] = []
    trusted_segments: list[tuple[float, float]] = []
    for segment in segments:
        if segment.size == 0:
            continue
        segment_timestamps = timestamps[segment]
        segment_power = power_w[segment]
        threshold_positions = np.flatnonzero(
            (segment_power <= config.off_threshold_w)
            | (segment_power >= config.on_threshold_w)
        )
        if threshold_positions.size == 0:
            continue
        threshold_states = (
            segment_power[threshold_positions] >= config.on_threshold_w
        ).astype(np.int8)
        state_changes = np.concatenate(
            ([True], threshold_states[1:] != threshold_states[:-1])
        )
        compressed_positions = threshold_positions[state_changes]
        compressed_states = threshold_states[state_changes]
        off_states = np.flatnonzero(compressed_states == 0)
        if off_states.size == 0:
            # A high-power first segment after a gap has an unknown true onset.
            continue
        first_off_position = int(compressed_positions[off_states[0]])
        trusted_segments.append(
            (
                float(segment_timestamps[first_off_position]),
                float(segment_timestamps[-1]),
            )
        )

        on_transitions = np.flatnonzero(
            (compressed_states[:-1] == 0) & (compressed_states[1:] == 1)
        ) + 1
        for transition in on_transitions:
            start_position = int(compressed_positions[transition])
            has_observed_off = transition + 1 < compressed_positions.size
            if has_observed_off:
                end_position = int(compressed_positions[transition + 1])
                end_timestamp = float(segment_timestamps[end_position])
                end_reason = "off_threshold"
                active_slice = slice(start_position, end_position)
            else:
                end_timestamp = float(segment_timestamps[-1])
                end_reason = (
                    "file_boundary"
                    if int(segment[-1]) == timestamps.size - 1
                    else "data_gap"
                )
                active_slice = slice(start_position, segment_timestamps.size)

            active_power = segment_power[active_slice]
            active_timestamps = segment_timestamps[active_slice]
            if active_power.size == 0:
                continue
            onset_uncertainty = (
                float(segment_timestamps[start_position] - segment_timestamps[start_position - 1])
                if start_position > 0
                else config.maximum_sample_gap_seconds
            )
            raw_intervals.append(
                {
                    "start_timestamp": float(segment_timestamps[start_position]),
                    "last_active_timestamp": float(active_timestamps[-1]),
                    "onset_uncertainty_seconds": onset_uncertainty,
                    "start_after_gap": False,
                    "power_sum": float(np.sum(active_power)),
                    "sample_count": int(active_power.size),
                    "maximum_active_power_w": float(np.max(active_power)),
                    "merged_segment_count": 1,
                    "end_timestamp": end_timestamp,
                    "end_reason": end_reason,
                    "active_duration_seconds": float(
                        active_timestamps[-1] - active_timestamps[0]
                    ),
                    "session_duration_seconds": float(
                        end_timestamp - active_timestamps[0]
                    ),
                    "mean_active_power_w": float(np.mean(active_power)),
                }
            )

    merged = _merge_intervals(raw_intervals, config.merge_gap_seconds)
    accepted: list[dict[str, object]] = []
    short_interval_count = 0
    for interval in merged:
        interval["touches_data_boundary"] = interval["end_reason"] in {
            "data_gap",
            "file_boundary",
        }
        if float(interval["active_duration_seconds"]) < config.minimum_on_seconds:
            short_interval_count += 1
            continue
        accepted.append(interval)

    return DetectionResult(
        intervals=tuple(accepted),
        activity_intervals=tuple(merged),
        trusted_coverage_segments=tuple(trusted_segments),
        coverage_start_timestamp=float(timestamps[valid_indices[0]]),
        coverage_end_timestamp=float(timestamps[valid_indices[-1]]),
        valid_record_count=int(valid_indices.size),
        gap_count=max(0, len(segments) - 1),
        duplicate_or_backward_count=duplicate_or_backward_count,
        short_interval_count=short_interval_count,
    )
