"Build online causal candidate-event samples using only whole-house aggregate measurements.\n\nThis causal sample builder has a separate entry point from offline event extraction and grouped-MLP training. Submeters supply ground truth, coverage audits and auxiliary labels only; candidate detection and all model inputs never read device submeters."

from __future__ import annotations

from .data_paths import resolve_project_path

from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import tomllib
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from .event_detection import DetectionResult, EventDetectionConfig, detect_on_intervals
from .mvp_preprocessing import ExplicitHouseSplit, HouseKey, assert_no_split_leakage
from .online_candidate_detection import (
    AggregateBin,
    AggregateCandidate,
    CandidateDetectionConfig,
    detect_aggregate_candidates,
)
from .online_features import (
    FEATURE_REGISTRY_VERSION,
    ONLINE_FEATURE_COLUMNS,
    ONLINE_MASK_COLUMNS,
    SEQUENCE_FEATURE_NAMES,
    OnlineFeatureConfig,
    assess_online_horizon_quality,
    build_causal_sequence_inputs,
    extract_online_features,
    post_bin_count_for_horizon,
    relative_bin_indices_for_horizon,
)


SCHEMA_VERSION = "online_causal_v1_1"
SEQUENCE_CONTRACT_VERSION = "online_sequence_v1_1"
CORE_POSITIVE_ROLES = frozenset({"core_positive"})
HARD_NEGATIVE_ROLES = frozenset({"hard_negative"})
TRANSFER_POSITIVE_ROLES = frozenset({"extended_clean_positive"})
QUALITY_BREAK_FLAGS = frozenset(
    {
        "invalid_raw_record",
        "raw_gap_gt_30s",
        "starts_after_long_gap",
        "nonmonotonic_timestamp",
        "refit_issues",
        "refit_nonpositive_aggregate",
        "ukdale_invalid_active_zero",
        "no_valid_raw_sample",
    }
)


@dataclass(frozen=True)
class OnlineBinSeries:
    "Store one house on a compressed fixed UTC grid with its original coverage boundaries."

    bins_by_left_s: Mapping[int, AggregateBin]
    first_bin_left_s: int | None
    last_bin_left_s: int | None
    first_raw_timestamp_s: float | None
    last_raw_timestamp_s: float | None
    nonmonotonic_timestamp_count: int = 0
    source_quality_flags: tuple[str, ...] = ()

    def bin_at(self, bin_left_s: int, bin_seconds: int = 10) -> AggregateBin:
        "Return the requested bin; missing bins explicitly contain no valid raw observations."

        existing = self.bins_by_left_s.get(int(bin_left_s))
        if existing is not None:
            return existing
        return AggregateBin(
            bin_left_s=int(bin_left_s),
            bin_right_s=int(bin_left_s + bin_seconds),
            aggregate_power_w=float("nan"),
            raw_valid_sample_count=0,
            quality_valid=False,
            maximum_raw_gap_seconds=float("nan"),
            quality_flags=("no_valid_raw_sample",),
        )


@dataclass(frozen=True)
class CandidateWindow:
    "A candidate window containing only fixed-grid observations that have arrived by its horizon."

    relative_bin_indices: np.ndarray
    bins: tuple[AggregateBin, ...]
    power_w: np.ndarray
    quality_valid_mask: np.ndarray
    quality_flags: tuple[tuple[str, ...], ...]
    touches_file_boundary: bool


@dataclass(frozen=True)
class GroundTruthEvent:
    "A true submeter onset with a stable audit identifier."

    event_id: str
    dataset: str
    house_id: str
    channel_id: str
    source_appliance: str
    taxonomy_role: str
    start_timestamp_s: float
    end_timestamp_s: float
    touches_boundary: bool


@dataclass(frozen=True)
class GroundTruthChannel:
    "Store events, activity intervals and coverage intervals for one trusted device channel."

    dataset: str
    house_id: str
    channel_id: str
    source_appliance: str
    taxonomy_role: str
    result: DetectionResult


@dataclass(frozen=True)
class _IntervalQueryIndex:
    "Index intervals for logarithmic-time queries of timestamps and windows."

    starts: tuple[float, ...]
    prefix_max_ends: tuple[float, ...]

    def contains_point_half_open(self, timestamp_s: float) -> bool:
        "Determine whether any half-open activity interval contains the timestamp."

        index = bisect_right(self.starts, float(timestamp_s)) - 1
        return index >= 0 and self.prefix_max_ends[index] > float(timestamp_s)

    def contains_point_closed_end(self, timestamp_s: float) -> bool:
        "Determine whether trusted coverage contains a point, preserving its existing closed-endpoint semantics."

        index = bisect_right(self.starts, float(timestamp_s)) - 1
        return index >= 0 and self.prefix_max_ends[index] >= float(timestamp_s)


@dataclass(frozen=True)
class _CoreChannelLabelIndex:
    "Activity and coverage query indices for one reliable primary-positive channel."

    activity: _IntervalQueryIndex
    coverage: _IntervalQueryIndex


@dataclass(frozen=True)
class CandidateLabelAudit:
    "A candidate label decision for the largest window, shared by every horizon."

    matched_event_id: str | None
    matched_source_appliance: str | None
    onset_match_error_seconds: float | None
    detector_report_delay_seconds: float | None
    label: int | None
    label_type: str
    label_status: str
    ignore_reasons: tuple[str, ...]
    label_confidence: str = "none"
    sample_weight: float = 0.0
    source_quality_warnings: tuple[str, ...] = ()


@dataclass
class _MutableBin:
    "Internal mutable state during fixed-grid aggregation, excluded from the output contract."

    values: list[float] = field(default_factory=list)
    raw_valid_sample_count: int = 0
    maximum_raw_gap_seconds: float = float("nan")
    flags: set[str] = field(default_factory=set)


class FixedUtcBinner:
    "Aggregate active-W readings into fixed UTC 10-second bins in their original arrival order."

    def __init__(self, *, bin_seconds: int, maximum_raw_gap_seconds: float) -> None:
        "Initialize the streaming aggregator without retaining per-second or per-row raw data in memory."

        if bin_seconds <= 0:
            raise ValueError("bin_seconds 必须为正数")
        self.bin_seconds = int(bin_seconds)
        self.maximum_raw_gap_seconds = float(maximum_raw_gap_seconds)
        # Keep raw values only in the current UTC bin; compress the preceding completed bin and release its lists.
        self._frozen_bins: dict[int, AggregateBin] = {}
        self._current_bin_left_s: int | None = None
        self._current_state: _MutableBin | None = None
        self._previous_timestamp_s: float | None = None
        self._previous_valid_timestamp_s: float | None = None
        self._first_raw_timestamp_s: float | None = None
        self._last_raw_timestamp_s: float | None = None
        self._nonmonotonic_timestamp_count = 0

    def ingest(
        self,
        timestamp_s: float,
        power_w: float,
        *,
        valid: bool,
        invalid_flags: Sequence[str] = (),
    ) -> None:
        "Accept one raw record; invalid readings still mark a quality break in their bin."

        if not math.isfinite(timestamp_s):
            return
        timestamp_s = float(timestamp_s)
        bin_left_s = int(math.floor(timestamp_s / self.bin_seconds) * self.bin_seconds)
        if self._previous_timestamp_s is not None and timestamp_s <= self._previous_timestamp_s:
            # Timestamp reversals must not enter the median and must cause an auditable quality failure for their windows.
            self._mark_bin_flag(bin_left_s, "nonmonotonic_timestamp")
            self._nonmonotonic_timestamp_count += 1
            return
        state = self._state_for_current_bin(bin_left_s)
        self._previous_timestamp_s = timestamp_s
        if self._first_raw_timestamp_s is None:
            self._first_raw_timestamp_s = timestamp_s
        self._last_raw_timestamp_s = timestamp_s

        if not valid or not math.isfinite(power_w):
            state.flags.update(invalid_flags or ("invalid_raw_record",))
            state.flags.add("invalid_raw_record")
            return

        power_w = float(power_w)
        if self._previous_valid_timestamp_s is not None:
            raw_gap = timestamp_s - self._previous_valid_timestamp_s
            state.maximum_raw_gap_seconds = _finite_max(state.maximum_raw_gap_seconds, raw_gap)
            if raw_gap > self.maximum_raw_gap_seconds:
                # Assign a long gap only to the first bin after it. Marking the last bin before the gap
                # would let arrivals beyond the horizon boundary change short-window membership retroactively.
                # Bins with no records inside the gap remain explicit invalid bins supplied by bin_at().
                state.flags.update({"raw_gap_gt_30s", "starts_after_long_gap"})
        self._previous_valid_timestamp_s = timestamp_s
        state.values.append(power_w)
        state.raw_valid_sample_count += 1

    def finalize(self) -> OnlineBinSeries:
        "Freeze a compressed grid with explicit quality; bin_at supplies missing bins on demand."

        self._freeze_current_bin()
        ordered = sorted(self._frozen_bins)
        return OnlineBinSeries(
            bins_by_left_s=dict(self._frozen_bins),
            first_bin_left_s=ordered[0] if ordered else None,
            last_bin_left_s=ordered[-1] if ordered else None,
            first_raw_timestamp_s=self._first_raw_timestamp_s,
            last_raw_timestamp_s=self._last_raw_timestamp_s,
            nonmonotonic_timestamp_count=self._nonmonotonic_timestamp_count,
        )

    def _state_for_current_bin(self, bin_left_s: int) -> _MutableBin:
        "Compress the preceding bin on transition; monotonic input never revisits completed bins."

        if self._current_bin_left_s is None:
            self._current_bin_left_s = bin_left_s
            self._current_state = _MutableBin()
        elif bin_left_s > self._current_bin_left_s:
            self._freeze_current_bin()
            self._current_bin_left_s = bin_left_s
            self._current_state = _MutableBin()
        elif bin_left_s < self._current_bin_left_s:
            self._mark_bin_flag(bin_left_s, "nonmonotonic_timestamp")
            raise RuntimeError("单调原始时间不应回到已完成的固定 bin")
        if self._current_state is None:
            raise RuntimeError("固定网格当前状态未初始化")
        return self._current_state

    def _freeze_current_bin(self) -> None:
        "Freeze the current median and quality, then release the raw-value list."

        if self._current_bin_left_s is None or self._current_state is None:
            return
        self._frozen_bins[self._current_bin_left_s] = self._freeze_state(
            self._current_bin_left_s, self._current_state
        )
        self._current_state = None

    def _freeze_state(self, bin_left_s: int, state: _MutableBin) -> AggregateBin:
        "Convert one mutable aggregation state into an immutable output bin."

        flags = set(state.flags)
        if state.raw_valid_sample_count == 0:
            flags.add("no_valid_raw_sample")
        quality_valid = state.raw_valid_sample_count > 0 and not bool(flags & QUALITY_BREAK_FLAGS)
        aggregate_power_w = (
            float(np.median(np.asarray(state.values, dtype=np.float64)))
            if quality_valid
            else float("nan")
        )
        return AggregateBin(
            bin_left_s=int(bin_left_s),
            bin_right_s=int(bin_left_s + self.bin_seconds),
            aggregate_power_w=aggregate_power_w,
            raw_valid_sample_count=int(state.raw_valid_sample_count),
            quality_valid=bool(quality_valid),
            maximum_raw_gap_seconds=(
                float(state.maximum_raw_gap_seconds)
                if math.isfinite(state.maximum_raw_gap_seconds)
                else 0.0
            ),
            quality_flags=tuple(sorted(flags)),
        )

    def _mark_bin_flag(self, bin_left_s: int, flag: str) -> None:
        "Append a quality flag to a current or completed bin without expanding its raw samples again."

        if bin_left_s == self._current_bin_left_s and self._current_state is not None:
            self._current_state.flags.add(flag)
            return
        existing = self._frozen_bins.get(bin_left_s)
        if existing is None:
            self._frozen_bins[bin_left_s] = AggregateBin(
                bin_left_s=bin_left_s,
                bin_right_s=bin_left_s + self.bin_seconds,
                aggregate_power_w=float("nan"),
                raw_valid_sample_count=0,
                quality_valid=False,
                maximum_raw_gap_seconds=0.0,
                quality_flags=(flag, "no_valid_raw_sample"),
            )
            return
        self._frozen_bins[bin_left_s] = replace(
            existing,
            aggregate_power_w=float("nan"),
            quality_valid=False,
            quality_flags=tuple(sorted({*existing.quality_flags, flag})),
        )

def _finite_max(current: float, value: float) -> float:
    "Compute a maximum while allowing an initial NaN."

    if not math.isfinite(current):
        return float(value)
    return max(float(current), float(value))


def build_fixed_utc_bins_from_records(
    records: Iterable[tuple[float, float, bool, Sequence[str]]],
    *,
    bin_seconds: int = 10,
    maximum_raw_gap_seconds: float = 30.0,
) -> OnlineBinSeries:
    "Pure entry point for building a fixed UTC grid from synthetic or test records."

    binner = FixedUtcBinner(
        bin_seconds=bin_seconds,
        maximum_raw_gap_seconds=maximum_raw_gap_seconds,
    )
    for timestamp_s, power_w, valid, flags in records:
        binner.ingest(timestamp_s, power_w, valid=valid, invalid_flags=flags)
    return binner.finalize()


def extract_candidate_window(
    series: OnlineBinSeries,
    candidate_onset_s: int,
    horizon_seconds: int,
    *,
    bin_seconds: int = 10,
    pre_seconds: int = 60,
) -> CandidateWindow:
    "Extract a causal candidate window using half-open intervals, without reading beyond the horizon right boundary."

    if candidate_onset_s % bin_seconds != 0:
        raise ValueError("candidate_onset_s 必须落在固定 bin 左边界")
    if pre_seconds % bin_seconds or horizon_seconds % bin_seconds:
        raise ValueError("窗口长度必须是 bin_seconds 的整数倍")
    pre_bin_count = pre_seconds // bin_seconds
    post_bin_count = horizon_seconds // bin_seconds
    relative_indices = np.arange(-pre_bin_count, post_bin_count, dtype=np.int64)
    bins = tuple(
        series.bin_at(int(candidate_onset_s + relative * bin_seconds), bin_seconds)
        for relative in relative_indices
    )
    power_w = np.asarray([item.aggregate_power_w for item in bins], dtype=np.float64)
    quality = np.asarray([item.quality_valid for item in bins], dtype=bool)
    left_s = int(candidate_onset_s - pre_seconds)
    # Check only whether the left history reaches the data origin. Later file records beyond the right edge
    # are information after the horizon decision boundary; local missing bins, coverage and raw gaps suffice for QC.
    touches_boundary = (
        series.first_bin_left_s is None
        or left_s <= int(series.first_bin_left_s)
    )
    return CandidateWindow(
        relative_bin_indices=relative_indices,
        bins=bins,
        power_w=power_w,
        quality_valid_mask=quality,
        quality_flags=tuple(item.quality_flags for item in bins),
        touches_file_boundary=touches_boundary,
    )


def assess_candidate_horizon(
    window: CandidateWindow,
    *,
    horizon_seconds: int,
    feature_config: OnlineFeatureConfig,
    maximum_raw_gap_seconds: float = 30.0,
    minimum_pre_coverage: float = 5.0 / 6.0,
    minimum_post_coverage: float = 0.8,
    maximum_consecutive_missing_bins: int = 1,
) -> tuple[bool, tuple[str, ...]]:
    "Combine fixed-window coverage, raw gaps and file boundaries into an interpretable quality decision."

    pre_bin_count = feature_config.pre_bin_count
    basic = assess_online_horizon_quality(
        window.power_w[:pre_bin_count],
        window.power_w[pre_bin_count:],
        window.quality_valid_mask[:pre_bin_count],
        window.quality_valid_mask[pre_bin_count:],
        config=feature_config,
        minimum_pre_coverage=minimum_pre_coverage,
        minimum_post_coverage=minimum_post_coverage,
        maximum_consecutive_missing_bins=maximum_consecutive_missing_bins,
    )
    reasons = list(basic.rejection_reasons)
    if window.touches_file_boundary:
        reasons.append("aggregate_file_boundary")
    for item in window.bins:
        if math.isfinite(item.maximum_raw_gap_seconds) and (
            item.maximum_raw_gap_seconds > maximum_raw_gap_seconds
        ):
            reasons.append("raw_gap_gt_30s")
        if "nonmonotonic_timestamp" in item.quality_flags:
            reasons.append("nonmonotonic_timestamp")
        if "refit_issues" in item.quality_flags:
            reasons.append("refit_issues")
        if "refit_nonpositive_aggregate" in item.quality_flags:
            reasons.append("refit_nonpositive_aggregate")
        if "ukdale_invalid_active_zero" in item.quality_flags:
            reasons.append("ukdale_invalid_active_zero")
    stable_reasons = tuple(sorted(set(reasons)))
    return not stable_reasons, stable_reasons


class StreamingOnIntervalDetector:
    "Track one submeter hysteresis state during a single REFIT CSV scan."

    def __init__(self, config: EventDetectionConfig) -> None:
        "Create a streaming state machine with the existing submeter-detector semantics, without accumulating raw arrays."

        self.config = config
        self._raw_intervals: list[dict[str, object]] = []
        self._active: dict[str, object] | None = None
        self._state = "unknown"
        self._previous_timestamp_s: float | None = None
        self._coverage_start_s: float | None = None
        self._coverage_end_s: float | None = None
        self._trusted_segment_start_s: float | None = None
        self._trusted_segments: list[tuple[float, float]] = []
        self._valid_record_count = 0
        self._gap_count = 0
        self._duplicate_or_backward_count = 0

    def _close_active(self, end_timestamp_s: float, reason: str) -> None:
        "Close the active interval while retaining statistics needed for later merging."

        if self._active is None:
            return
        active = dict(self._active)
        active["end_timestamp"] = float(end_timestamp_s)
        active["end_reason"] = reason
        active["active_duration_seconds"] = max(
            0.0,
            float(active["last_active_timestamp"]) - float(active["start_timestamp"]),
        )
        active["session_duration_seconds"] = max(
            0.0,
            float(end_timestamp_s) - float(active["start_timestamp"]),
        )
        sample_count = int(active["sample_count"])
        active["mean_active_power_w"] = (
            float(active["power_sum"]) / sample_count if sample_count else 0.0
        )
        self._raw_intervals.append(active)
        self._active = None

    def _break_coverage(self, reason: str) -> None:
        "Close activity on invalid readings or long gaps and return to the unknown state."

        if self._previous_timestamp_s is not None:
            self._close_active(self._previous_timestamp_s, reason)
        if self._trusted_segment_start_s is not None and self._previous_timestamp_s is not None:
            self._trusted_segments.append(
                (self._trusted_segment_start_s, self._previous_timestamp_s)
            )
        self._trusted_segment_start_s = None
        self._state = "unknown"

    def consume(self, timestamp_s: float, power_w: float, valid: bool) -> None:
        "Accept a submeter reading; shared aggregate quality also determines validity."

        if not (math.isfinite(timestamp_s) and math.isfinite(power_w)):
            if self._previous_timestamp_s is not None:
                self._break_coverage("invalid_record")
            return
        timestamp_s = float(timestamp_s)
        power_w = float(power_w)
        if self._previous_timestamp_s is not None and timestamp_s <= self._previous_timestamp_s:
            self._duplicate_or_backward_count += 1
            return
        if not valid:
            self._break_coverage("invalid_record")
            self._previous_timestamp_s = timestamp_s
            return
        gap_before_current = False
        if self._previous_timestamp_s is not None:
            gap_seconds = timestamp_s - self._previous_timestamp_s
            if gap_seconds > self.config.maximum_sample_gap_seconds:
                self._gap_count += 1
                self._break_coverage("data_gap")
                gap_before_current = True
        self._valid_record_count += 1
        if self._coverage_start_s is None:
            self._coverage_start_s = timestamp_s
        self._coverage_end_s = timestamp_s

        if self._state == "unknown":
            if power_w <= self.config.off_threshold_w:
                self._state = "off"
                self._trusted_segment_start_s = timestamp_s
        elif self._state == "off":
            if power_w >= self.config.on_threshold_w:
                onset_uncertainty_s = (
                    timestamp_s - self._previous_timestamp_s
                    if self._previous_timestamp_s is not None
                    else self.config.maximum_sample_gap_seconds
                )
                self._active = {
                    "start_timestamp": timestamp_s,
                    "last_active_timestamp": timestamp_s,
                    "onset_uncertainty_seconds": float(onset_uncertainty_s),
                    "start_after_gap": bool(gap_before_current),
                    "power_sum": power_w,
                    "sample_count": 1,
                    "maximum_active_power_w": power_w,
                    "merged_segment_count": 1,
                }
                self._state = "on"
        elif self._state == "on":
            if power_w <= self.config.off_threshold_w:
                self._close_active(timestamp_s, "off_threshold")
                self._state = "off"
            elif self._active is not None:
                self._active["last_active_timestamp"] = timestamp_s
                self._active["power_sum"] = float(self._active["power_sum"]) + power_w
                self._active["sample_count"] = int(self._active["sample_count"]) + 1
                self._active["maximum_active_power_w"] = max(
                    float(self._active["maximum_active_power_w"]), power_w
                )
        self._previous_timestamp_s = timestamp_s

    def finalize(self) -> DetectionResult:
        "Close the state at end-of-file and return an object matching the existing detector contract."

        if self._active is not None and self._previous_timestamp_s is not None:
            self._close_active(self._previous_timestamp_s, "file_boundary")
        if self._trusted_segment_start_s is not None and self._previous_timestamp_s is not None:
            self._trusted_segments.append(
                (self._trusted_segment_start_s, self._previous_timestamp_s)
            )
        merged = _merge_streamed_intervals(self._raw_intervals, self.config.merge_gap_seconds)
        accepted: list[dict[str, object]] = []
        short_interval_count = 0
        for interval in merged:
            interval["touches_data_boundary"] = interval.get("end_reason") in {
                "data_gap",
                "file_boundary",
                "invalid_record",
            }
            if float(interval["active_duration_seconds"]) < self.config.minimum_on_seconds:
                short_interval_count += 1
            else:
                accepted.append(interval)
        return DetectionResult(
            intervals=tuple(accepted),
            activity_intervals=tuple(merged),
            trusted_coverage_segments=tuple(self._trusted_segments),
            coverage_start_timestamp=self._coverage_start_s,
            coverage_end_timestamp=self._coverage_end_s,
            valid_record_count=self._valid_record_count,
            gap_count=self._gap_count,
            duplicate_or_backward_count=self._duplicate_or_backward_count,
            short_interval_count=short_interval_count,
        )


def _merge_streamed_intervals(
    intervals: Sequence[Mapping[str, object]], maximum_gap_seconds: float
) -> list[dict[str, object]]:
    "Merge short gaps only after normal OFF transitions, never across gaps, invalid readings or file boundaries."

    merged: list[dict[str, object]] = []
    for raw_interval in intervals:
        interval = dict(raw_interval)
        if not merged:
            merged.append(interval)
            continue
        previous = merged[-1]
        gap_seconds = float(interval["start_timestamp"]) - float(previous["end_timestamp"])
        can_merge = (
            gap_seconds <= maximum_gap_seconds
            and previous.get("end_reason") == "off_threshold"
            and not bool(interval.get("start_after_gap"))
        )
        if not can_merge:
            merged.append(interval)
            continue
        prior_count = int(previous["sample_count"])
        current_count = int(interval["sample_count"])
        total_count = prior_count + current_count
        total_power = float(previous["power_sum"]) + float(interval["power_sum"])
        previous.update(
            {
                "end_timestamp": interval["end_timestamp"],
                "last_active_timestamp": interval["last_active_timestamp"],
                "end_reason": interval["end_reason"],
                "active_duration_seconds": float(previous["active_duration_seconds"])
                + float(interval["active_duration_seconds"]),
                "session_duration_seconds": float(interval["end_timestamp"])
                - float(previous["start_timestamp"]),
                "sample_count": total_count,
                "power_sum": total_power,
                "mean_active_power_w": total_power / total_count,
                "maximum_active_power_w": max(
                    float(previous["maximum_active_power_w"]),
                    float(interval["maximum_active_power_w"]),
                ),
                "merged_segment_count": int(previous["merged_segment_count"]) + 1,
            }
        )
    return merged


def _interval_overlaps(
    left_s: float, right_s: float, interval_left_s: float, interval_right_s: float
) -> bool:
    "Determine whether two half-open intervals have a positive-length intersection."

    return float(left_s) < float(interval_right_s) and float(interval_left_s) < float(right_s)


def _interval_contains(segments: Sequence[tuple[float, float]], left_s: float, right_s: float) -> bool:
    "Determine whether a trusted coverage interval spans the complete half-open window."

    return any(float(start) <= left_s and float(end) >= right_s for start, end in segments)


def _numeric_channel_key(channel_id: str) -> tuple[int, str]:
    "Provide a stable channel-id sort key across Appliance7 and channel_7 representations."

    digits = "".join(character for character in str(channel_id) if character.isdigit())
    return (int(digits) if digits else sys.maxsize, str(channel_id))


def ground_truth_events_from_channels(
    channels: Sequence[GroundTruthChannel],
) -> list[GroundTruthEvent]:
    "Convert detector results into stable true onsets eligible for one-to-one matching."

    events: list[GroundTruthEvent] = []
    for channel in channels:
        for event_index, interval in enumerate(channel.result.intervals):
            start_s = float(interval["start_timestamp"])
            events.append(
                GroundTruthEvent(
                    event_id=(
                        f"gt_{channel.dataset}_h{channel.house_id}_"
                        f"c{channel.channel_id}_t{int(round(start_s * 1000.0))}_{event_index}"
                    ),
                    dataset=channel.dataset,
                    house_id=channel.house_id,
                    channel_id=channel.channel_id,
                    source_appliance=channel.source_appliance,
                    taxonomy_role=channel.taxonomy_role,
                    start_timestamp_s=start_s,
                    end_timestamp_s=float(interval["end_timestamp"]),
                    touches_boundary=bool(interval.get("touches_data_boundary", False)),
                )
            )
    return sorted(
        events,
        key=lambda item: (
            item.start_timestamp_s,
            _numeric_channel_key(item.channel_id),
            item.event_id,
        ),
    )


def _build_interval_query_index(
    intervals: Iterable[tuple[float, float]],
) -> _IntervalQueryIndex:
    "Convert possibly unordered intervals into a prefix-maximum-end query index."

    ordered = sorted(
        (float(start_s), float(end_s))
        for start_s, end_s in intervals
        if math.isfinite(float(start_s)) and math.isfinite(float(end_s))
    )
    starts: list[float] = []
    prefix_max_ends: list[float] = []
    maximum_end = -math.inf
    for start_s, end_s in ordered:
        starts.append(start_s)
        maximum_end = max(maximum_end, end_s)
        prefix_max_ends.append(maximum_end)
    return _IntervalQueryIndex(tuple(starts), tuple(prefix_max_ends))


def _build_core_label_index(
    core_channels: Sequence[GroundTruthChannel],
) -> tuple[_CoreChannelLabelIndex, ...]:
    "Precompile primary-positive activity and coverage indices instead of rescanning submeter intervals for every candidate."

    indexes: list[_CoreChannelLabelIndex] = []
    for channel in core_channels:
        indexes.append(
            _CoreChannelLabelIndex(
                activity=_build_interval_query_index(
                    (
                        (
                            float(interval["start_timestamp"]),
                            float(interval["end_timestamp"]),
                        )
                        for interval in channel.result.activity_intervals
                    )
                ),
                coverage=_build_interval_query_index(channel.result.trusted_coverage_segments),
            )
        )
    return tuple(indexes)


def assign_target_active_labels(
    bin_left_s: Sequence[int],
    *,
    bin_seconds: int,
    core_channels: Sequence[GroundTruthChannel],
    core_label_index: Sequence[_CoreChannelLabelIndex] | None = None,
) -> np.ndarray:
    "Generate reliable primary-positive auxiliary activity labels 1/0/-1 at bin centers."

    indexes = (
        tuple(core_label_index)
        if core_label_index is not None
        else _build_core_label_index(core_channels)
    )
    if len(indexes) != len(core_channels):
        raise ValueError("core_label_index 与 core_channels 长度不一致")
    labels = np.full(len(bin_left_s), -1, dtype=np.int8)
    for index, left_s in enumerate(bin_left_s):
        center_s = float(left_s) + bin_seconds / 2.0
        active = any(item.activity.contains_point_half_open(center_s) for item in indexes)
        all_covered = bool(indexes) and all(
            item.coverage.contains_point_closed_end(center_s) for item in indexes
        )
        if active:
            labels[index] = 1
        elif all_covered:
            labels[index] = 0
    return labels


def _channel_covers_window(channel: GroundTruthChannel, left_s: float, right_s: float) -> bool:
    "Determine whether a primary-positive submeter reliably covers the largest label window."

    return _interval_contains(channel.result.trusted_coverage_segments, left_s, right_s)


def _activity_intervals(
    channels: Sequence[GroundTruthChannel],
) -> Iterator[tuple[GroundTruthChannel, Mapping[str, object]]]:
    "Iterate activity intervals in stable order to reproduce label reasons and outputs."

    for channel in sorted(channels, key=lambda item: _numeric_channel_key(item.channel_id)):
        for interval in channel.result.activity_intervals:
            yield channel, interval


def assign_candidate_labels(
    candidate_rows: Sequence[Mapping[str, object]],
    *,
    core_channels: Sequence[GroundTruthChannel],
    hard_channels: Sequence[GroundTruthChannel],
    transfer_channels: Sequence[GroundTruthChannel] = (),
    onset_match_tolerance_seconds: float = 20.0,
    pre_seconds: int = 60,
    maximum_horizon_seconds: int = 120,
    aggregate_max_window_trusted: Mapping[str, bool] | None = None,
    core_positive_sample_weight: float = 1.0,
    hard_negative_sample_weight: float = 1.0,
    background_negative_sample_weight: float = 0.25,
) -> tuple[dict[str, CandidateLabelAudit], list[GroundTruthEvent], list[GroundTruthEvent]]:
    "Audit candidate labels using largest-window purity and stable one-to-one matching.\n\nThe candidate_rows input needs only candidate_id, candidate_onset_s and candidate_detected_at_s, which simplifies tests and avoids leaking submeter fields into input sequences."

    # v1.0 used H120 aggregate QC to qualify labels at every shorter horizon, causing future leakage
    # through sample selection. Retain this argument for old callers, but v1.1 explicitly does not read it.
    del aggregate_max_window_trusted
    weight_values = {
        "core_positive": core_positive_sample_weight,
        "hard_negative": hard_negative_sample_weight,
        "background_negative": background_negative_sample_weight,
    }
    for name, value in weight_values.items():
        if not math.isfinite(float(value)) or not 0.0 < float(value) <= 1.0:
            raise ValueError(f"{name} sample weight 必须位于 (0, 1]")
    all_channels = [*core_channels, *hard_channels, *transfer_channels]
    events = ground_truth_events_from_channels(all_channels)
    core_events = [event for event in events if event.taxonomy_role in CORE_POSITIVE_ROLES]
    pair_rows: list[tuple[float, float, tuple[int, str], str, float, str]] = []
    nearby_by_candidate: dict[str, list[GroundTruthEvent]] = defaultdict(list)
    event_by_id = {event.event_id: event for event in events}
    for candidate in candidate_rows:
        candidate_id = str(candidate["candidate_id"])
        onset_s = float(candidate["candidate_onset_s"])
        for event in events:
            error_s = onset_s - event.start_timestamp_s
            if abs(error_s) <= onset_match_tolerance_seconds:
                nearby_by_candidate[candidate_id].append(event)
                pair_rows.append(
                    (
                        abs(error_s),
                        event.start_timestamp_s,
                        _numeric_channel_key(event.channel_id),
                        event.event_id,
                        onset_s,
                        candidate_id,
                    )
                )
    # Sort globally before greedy matching so each aggregate candidate and true onset is used at most once.
    pair_rows.sort()
    matched_event_by_candidate: dict[str, GroundTruthEvent] = {}
    used_events: set[str] = set()
    for _, _, _, event_id, _, candidate_id in pair_rows:
        if candidate_id in matched_event_by_candidate or event_id in used_events:
            continue
        matched_event_by_candidate[candidate_id] = event_by_id[event_id]
        used_events.add(event_id)

    audits: dict[str, CandidateLabelAudit] = {}
    for candidate in candidate_rows:
        candidate_id = str(candidate["candidate_id"])
        onset_s = float(candidate["candidate_onset_s"])
        detected_at_s = float(candidate["candidate_detected_at_s"])
        left_s = onset_s - float(pre_seconds)
        right_s = onset_s + float(maximum_horizon_seconds)
        candidate_events = nearby_by_candidate.get(candidate_id, [])
        matched = matched_event_by_candidate.get(candidate_id)
        reasons: list[str] = []
        if not core_channels:
            reasons.append("no_reliable_core_positive_channel")
        elif not all(_channel_covers_window(channel, left_s, right_s) for channel in core_channels):
            reasons.append("positive_coverage_gap")
        if transfer_channels and not all(
            _channel_covers_window(channel, left_s, right_s)
            for channel in transfer_channels
        ):
            # Transfer positives do not enter primary supervision; unknown coverage still cannot prove a candidate negative.
            reasons.append("transfer_positive_coverage_gap")

        # Kettle and toaster both belong to core_positive, but simultaneous nearby onsets
        # cannot become a unique source through stable tie-breaking; different canonical appliances also require ignore.
        nearby_categories = {
            (event.taxonomy_role, event.source_appliance) for event in candidate_events
        }
        if len(nearby_categories) > 1:
            reasons.append("multiple_different_category_onsets")
        if matched is not None and matched.touches_boundary:
            reasons.append("matched_event_touches_submeter_boundary")
        if matched is not None:
            matched_channels = [
                channel
                for channel in all_channels
                if channel.taxonomy_role == matched.taxonomy_role
                and channel.channel_id == matched.channel_id
                and channel.source_appliance == matched.source_appliance
            ]
            if len(matched_channels) != 1 or not _channel_covers_window(
                matched_channels[0], left_s, right_s
            ):
                # Hard-negative and transfer channels must not misrepresent unknown coverage as device OFF.
                reasons.append("matched_submeter_coverage_gap")
        unmatched_core_onsets_in_window = [
            event
            for event in core_events
            if left_s <= event.start_timestamp_s < right_s
            and (matched is None or event.event_id != matched.event_id)
        ]
        if unmatched_core_onsets_in_window:
            # Label purity always uses the largest window: a valid match to the first kettle
            # does not explain a second or different primary-positive onset in the same window.
            reasons.append("target_onset_unmatched_or_nonunique")

        core_active_before_onset = False
        core_active_in_window = False
        hard_active_in_window = False
        transfer_active_in_window = False
        for channel, interval in _activity_intervals(core_channels):
            interval_left = float(interval["start_timestamp"])
            interval_right = float(interval["end_timestamp"])
            is_matched_core_interval = (
                matched is not None
                and matched.taxonomy_role in CORE_POSITIVE_ROLES
                and channel.dataset == matched.dataset
                and channel.house_id == matched.house_id
                and channel.channel_id == matched.channel_id
                and channel.source_appliance == matched.source_appliance
                and math.isclose(
                    interval_left,
                    matched.start_timestamp_s,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            )
            if (
                interval_left < onset_s
                and _interval_overlaps(left_s, right_s, interval_left, interval_right)
                and not is_matched_core_interval
            ):
                # A submeter onset may precede the aggregate candidate by onset_match_tolerance_seconds;
                # the same uniquely matched event must not be counted again as another prior positive.
                core_active_before_onset = True
            if _interval_overlaps(left_s, right_s, interval_left, interval_right):
                core_active_in_window = True
            if bool(interval.get("touches_data_boundary", False)) and _interval_overlaps(
                left_s, right_s, interval_left, interval_right
            ):
                reasons.append("core_activity_touches_submeter_boundary")
        for channel, interval in _activity_intervals(hard_channels):
            if _interval_overlaps(
                left_s,
                right_s,
                float(interval["start_timestamp"]),
                float(interval["end_timestamp"]),
            ):
                hard_active_in_window = True
        for channel, interval in _activity_intervals(transfer_channels):
            if _interval_overlaps(
                left_s,
                right_s,
                float(interval["start_timestamp"]),
                float(interval["end_timestamp"]),
            ):
                transfer_active_in_window = True

        if core_active_before_onset:
            reasons.append("target_already_active_before_candidate")
        if core_active_in_window and hard_active_in_window:
            reasons.append("positive_negative_activity_overlap")
        if transfer_active_in_window and (
            matched is None or matched.taxonomy_role not in TRANSFER_POSITIVE_ROLES
        ):
            # Transfer positives are not primary training samples and cannot be merged with core/hard/background sources.
            reasons.append("transfer_positive_activity_in_window")

        label: int | None = None
        label_type = "ignore"
        label_status = "ignore"
        if matched is not None:
            if matched.taxonomy_role in CORE_POSITIVE_ROLES:
                label = 1
                label_type = "core_positive"
                label_status = "matched"
            elif matched.taxonomy_role in HARD_NEGATIVE_ROLES:
                label = 0
                label_type = "hard_negative"
                label_status = "matched"
            elif matched.taxonomy_role in TRANSFER_POSITIVE_ROLES:
                label_type = "transfer_positive"
                label_status = "transfer_only"
            else:
                reasons.append("unsupported_matched_taxonomy_role")
        elif not reasons and not core_active_in_window:
            label = 0
            label_type = "background_negative"
            label_status = "background_verified"
        elif not reasons and core_active_in_window:
            reasons.append("target_onset_unmatched_or_nonunique")

        # Any quality or mixture reason overrides a provisional primary training label; transfer labels remain audit-only.
        if reasons and label is not None:
            label = None
            label_type = "ignore"
            label_status = "ignore"
        if label_type == "transfer_positive":
            label = None
        if label_type == "core_positive":
            label_confidence = "high"
            sample_weight = float(core_positive_sample_weight)
        elif label_type == "hard_negative":
            label_confidence = "high"
            sample_weight = float(hard_negative_sample_weight)
        elif label_type == "background_negative":
            # Unmatched background excludes only reliably monitored positives; it cannot prove the absence of unsubmetered household loads.
            label_confidence = "weak_unmonitored_background"
            sample_weight = float(background_negative_sample_weight)
        else:
            label_confidence = "none"
            sample_weight = 0.0
        error_s = (
            onset_s - matched.start_timestamp_s if matched is not None else None
        )
        report_delay_s = (
            detected_at_s - matched.start_timestamp_s if matched is not None else None
        )
        audits[candidate_id] = CandidateLabelAudit(
            matched_event_id=matched.event_id if matched else None,
            matched_source_appliance=matched.source_appliance if matched else None,
            onset_match_error_seconds=error_s,
            detector_report_delay_seconds=report_delay_s,
            label=label,
            label_type=label_type,
            label_status=label_status,
            ignore_reasons=tuple(sorted(set(reasons))),
            label_confidence=label_confidence,
            sample_weight=sample_weight,
        )

    matched_core = [
        event
        for event in core_events
        if event.event_id in used_events
    ]
    unmatched_core = [
        event
        for event in core_events
        if event.event_id not in used_events
    ]
    return audits, matched_core, unmatched_core


def apply_source_quality_policy(
    audits: Mapping[str, CandidateLabelAudit],
    *,
    source_quality_flags: Sequence[str],
    allow_refit_missing_issues_matched_positive: bool,
    allow_refit_missing_issues_background_negative: bool,
    missing_issues_positive_sample_weight: float,
) -> dict[str, CandidateLabelAudit]:
    "Turn source-quality constraints into explicit confidence and supervision eligibility.\n\nThe complete official REFIT H1 processed file has no Issues column. v1.1 permits uniquely matched primary positives to enter training at medium confidence, while still rejecting unmatched background and hard negatives so unproven OFF states never become reliable negatives. This affects label eligibility only and reads no future aggregate data from any horizon."

    missing_issues = "refit_issues_flag_unavailable" in set(source_quality_flags)
    if not missing_issues:
        return dict(audits)

    warning = "refit_issues_flag_unavailable"
    result: dict[str, CandidateLabelAudit] = {}
    for candidate_id, audit in audits.items():
        warnings = tuple(sorted(set((*audit.source_quality_warnings, warning))))
        if (
            audit.label_type == "core_positive"
            and allow_refit_missing_issues_matched_positive
        ):
            result[candidate_id] = replace(
                audit,
                label_status="matched_source_quality_limited",
                label_confidence="medium_missing_refit_issues",
                sample_weight=float(missing_issues_positive_sample_weight),
                source_quality_warnings=warnings,
            )
            continue
        if (
            audit.label_type == "background_negative"
            and allow_refit_missing_issues_background_negative
        ):
            result[candidate_id] = replace(
                audit,
                label_confidence="weak_missing_refit_issues_background",
                source_quality_warnings=warnings,
            )
            continue
        if audit.label_type in {"core_positive", "hard_negative", "background_negative"}:
            result[candidate_id] = replace(
                audit,
                label=None,
                label_type="ignore",
                label_status="ignore",
                ignore_reasons=tuple(sorted(set((*audit.ignore_reasons, warning)))),
                label_confidence="none",
                sample_weight=0.0,
                source_quality_warnings=warnings,
            )
            continue
        if audit.label_type == "ignore":
            result[candidate_id] = replace(
                audit,
                ignore_reasons=tuple(sorted(set((*audit.ignore_reasons, warning)))),
                source_quality_warnings=warnings,
            )
        else:
            result[candidate_id] = replace(audit, source_quality_warnings=warnings)
    return result


def assign_overlap_clusters(candidate_rows: Sequence[Mapping[str, object]]) -> dict[str, str]:
    "Assign stable cluster IDs from overlapping largest windows within the same dataset and house."

    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in candidate_rows:
        grouped[(str(row["dataset"]), str(row["house_id"]))].append(row)
    result: dict[str, str] = {}
    for (dataset, house_id), rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda item: (float(item["candidate_onset_s"]), str(item["candidate_id"])))
        cluster_number = 0
        cluster_right_s = -math.inf
        for row in ordered:
            onset_s = float(row["candidate_onset_s"])
            left_s = onset_s - 60.0
            right_s = onset_s + 120.0
            if left_s >= cluster_right_s:
                cluster_number += 1
                cluster_right_s = right_s
            else:
                cluster_right_s = max(cluster_right_s, right_s)
            result[str(row["candidate_id"])] = f"{dataset}_h{house_id}_cluster_{cluster_number:05d}"
    return result


@dataclass(frozen=True)
class ChannelSpec:
    "A submeter channel eligible for label generation after taxonomy filtering."

    dataset: str
    house_id: str
    channel_id: str
    source_column: str
    source_file: Path
    source_appliance: str
    taxonomy_role: str


def _normalise_house_text(value: object) -> str:
    "Convert numeric or string house IDs from CSV into stable text without decimal suffixes."

    return HouseKey("placeholder", value).house_id


def _as_project_path(project_root: Path, value: object) -> Path:
    return resolve_project_path(project_root, str(value))


def _truthy_csv_value(value: object) -> bool:
    "Accept bool, 0/1 and true/false text representations in CSV."

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes"}


def select_label_channels(
    appliance_inventory: pd.DataFrame,
    *,
    project_root: Path,
    dataset: str,
    house_id: str,
) -> tuple[list[ChannelSpec], list[ChannelSpec], list[ChannelSpec]]:
    "Select primary-positive, hard-negative and audit-only transfer-positive channels under the two-filter contract."

    required = {
        "dataset",
        "house_id",
        "channel_id",
        "source_column",
        "source_file",
        "canonical_label",
        "taxonomy_confidence",
        "data_use",
        "include_in_main_mvp",
    }
    missing = required - set(appliance_inventory.columns)
    if missing:
        raise ValueError(f"appliance inventory 缺少列：{sorted(missing)}")
    role_column = (
        "taxonomy_role" if "taxonomy_role" in appliance_inventory else "semantic_class"
    )
    if role_column not in appliance_inventory:
        raise ValueError("appliance inventory 缺少 taxonomy_role 或 semantic_class")
    selected = appliance_inventory[
        (appliance_inventory["dataset"].astype(str) == dataset)
        & (appliance_inventory["house_id"].map(_normalise_house_text) == house_id)
    ]
    core: list[ChannelSpec] = []
    hard: list[ChannelSpec] = []
    transfer: list[ChannelSpec] = []
    for row in selected.to_dict(orient="records"):
        role = str(row.get(role_column, "")).strip()
        confidence = str(row.get("taxonomy_confidence", "")).strip()
        data_use = str(row.get("data_use", "")).strip()
        source_appliance = str(row.get("canonical_label", "")).strip()
        spec = ChannelSpec(
            dataset=dataset,
            house_id=house_id,
            channel_id=str(row["channel_id"]),
            source_column=str(row["source_column"]),
            source_file=_as_project_path(project_root, row["source_file"]),
            source_appliance=source_appliance,
            taxonomy_role=role,
        )
        main_eligible = (
            _truthy_csv_value(row.get("include_in_main_mvp"))
            and data_use == "main"
            and confidence == "high"
        )
        if main_eligible and role in CORE_POSITIVE_ROLES:
            core.append(spec)
        elif main_eligible and role in HARD_NEGATIVE_ROLES:
            hard.append(spec)
        elif (
            role in TRANSFER_POSITIVE_ROLES
            and confidence == "high"
            and data_use == "transfer_only"
        ):
            transfer.append(spec)
    stable_sort = lambda item: (_numeric_channel_key(item.channel_id), item.source_appliance)
    return sorted(core, key=stable_sort), sorted(hard, key=stable_sort), sorted(transfer, key=stable_sort)


def _threshold_config_for_channel(
    thresholds: Mapping[str, Mapping[str, object]], source_appliance: str
) -> EventDetectionConfig:
    "Convert explicitly frozen online submeter thresholds into the existing detector configuration."

    aliases = {
        "steam_iron": "iron",
        "hairdryer": "hair_dryer",
        "vacuum": "vacuum_cleaner",
        "running_machine": "treadmill",
    }
    lookup_name = source_appliance
    if lookup_name not in thresholds:
        lookup_name = aliases.get(lookup_name, lookup_name)
    values = thresholds.get(lookup_name)
    if values is None:
        raise ValueError(f"online config 没有 {source_appliance!r} 的分表检测阈值")
    return EventDetectionConfig(
        on_threshold_w=float(values["on_threshold_w"]),
        off_threshold_w=float(values["off_threshold_w"]),
        minimum_on_seconds=float(values["minimum_on_seconds"]),
        merge_gap_seconds=float(values["merge_gap_seconds"]),
        maximum_sample_gap_seconds=float(values["maximum_sample_gap_seconds"]),
    )


def _ground_truth_channel_from_result(
    spec: ChannelSpec, result: DetectionResult
) -> GroundTruthChannel:
    "Assemble a detected channel for primary-label and auxiliary-activity processing."

    return GroundTruthChannel(
        dataset=spec.dataset,
        house_id=spec.house_id,
        channel_id=spec.channel_id,
        source_appliance=spec.source_appliance,
        taxonomy_role=spec.taxonomy_role,
        result=result,
    )


def _load_refit_house_streaming(
    *,
    aggregate_path: Path,
    specs: Sequence[ChannelSpec],
    thresholds: Mapping[str, Mapping[str, object]],
    bin_seconds: int,
    maximum_raw_gap_seconds: float,
    chunksize: int,
) -> tuple[OnlineBinSeries, list[GroundTruthChannel], tuple[str, ...]]:
    "Scan a REFIT whole-house CSV once in chunks, producing aggregate bins and multiple submeter state machines."

    if not aggregate_path.is_file():
        raise FileNotFoundError(f"REFIT aggregate 文件不存在：{aggregate_path}")
    header = pd.read_csv(aggregate_path, nrows=0)
    required = {"Unix", "Aggregate"}
    missing = required - set(header.columns)
    if missing:
        raise ValueError(f"REFIT 文件缺少列：{sorted(missing)}")
    source_columns = {spec.source_column for spec in specs}
    missing_channels = source_columns - set(header.columns)
    if missing_channels:
        raise ValueError(f"REFIT 文件缺少已选设备列：{sorted(missing_channels)}")
    has_issues = "Issues" in header.columns
    use_columns = ["Unix", "Aggregate", *sorted(source_columns)]
    if has_issues:
        use_columns.append("Issues")
    binner = FixedUtcBinner(
        bin_seconds=bin_seconds,
        maximum_raw_gap_seconds=maximum_raw_gap_seconds,
    )
    detectors = {
        spec.channel_id: StreamingOnIntervalDetector(
            _threshold_config_for_channel(thresholds, spec.source_appliance)
        )
        for spec in specs
    }
    for chunk in pd.read_csv(
        aggregate_path,
        usecols=use_columns,
        chunksize=chunksize,
        low_memory=False,
    ):
        timestamps = pd.to_numeric(chunk["Unix"], errors="coerce").to_numpy(dtype=np.float64)
        aggregate = pd.to_numeric(chunk["Aggregate"], errors="coerce").to_numpy(dtype=np.float64)
        issues = (
            pd.to_numeric(chunk["Issues"], errors="coerce").to_numpy(dtype=np.float64)
            if has_issues
            else np.zeros(timestamps.size, dtype=np.float64)
        )
        appliance_power = {
            spec.channel_id: pd.to_numeric(chunk[spec.source_column], errors="coerce").to_numpy(
                dtype=np.float64
            )
            for spec in specs
        }
        for index, timestamp_s in enumerate(timestamps):
            aggregate_w = float(aggregate[index])
            row_flags: list[str] = []
            aggregate_valid = math.isfinite(float(timestamp_s)) and math.isfinite(aggregate_w)
            if has_issues and (not math.isfinite(float(issues[index])) or issues[index] == 1.0):
                aggregate_valid = False
                row_flags.append("refit_issues")
            if not math.isfinite(aggregate_w) or aggregate_w <= 0.0:
                aggregate_valid = False
                row_flags.append("refit_nonpositive_aggregate")
            binner.ingest(
                float(timestamp_s),
                aggregate_w,
                valid=aggregate_valid,
                invalid_flags=tuple(sorted(set(row_flags))),
            )
            for spec in specs:
                power_w = float(appliance_power[spec.channel_id][index])
                detectors[spec.channel_id].consume(
                    float(timestamp_s),
                    power_w,
                    bool(aggregate_valid and math.isfinite(power_w)),
                )
    flags = () if has_issues else ("refit_issues_flag_unavailable",)
    raw_series = binner.finalize()
    series = OnlineBinSeries(
        bins_by_left_s=raw_series.bins_by_left_s,
        first_bin_left_s=raw_series.first_bin_left_s,
        last_bin_left_s=raw_series.last_bin_left_s,
        first_raw_timestamp_s=raw_series.first_raw_timestamp_s,
        last_raw_timestamp_s=raw_series.last_raw_timestamp_s,
        nonmonotonic_timestamp_count=raw_series.nonmonotonic_timestamp_count,
        source_quality_flags=flags,
    )
    return (
        series,
        [_ground_truth_channel_from_result(spec, detectors[spec.channel_id].finalize()) for spec in specs],
        flags,
    )


def _load_ukdale_house_streaming(
    *,
    mains_path: Path,
    specs: Sequence[ChannelSpec],
    thresholds: Mapping[str, Mapping[str, object]],
    bin_seconds: int,
    maximum_raw_gap_seconds: float,
    chunksize: int,
    invalid_zero_apparent_threshold_va: float,
    maximum_nonmonotonic_records: int,
) -> tuple[OnlineBinSeries, list[GroundTruthChannel]]:
    "Stream active-W from UK-DALE mains.dat and detect selected submeters separately."

    if not mains_path.is_file():
        raise FileNotFoundError(f"UK-DALE mains 文件不存在：{mains_path}")
    binner = FixedUtcBinner(
        bin_seconds=bin_seconds,
        maximum_raw_gap_seconds=maximum_raw_gap_seconds,
    )
    for chunk in pd.read_csv(
        mains_path,
        sep=r"\s+",
        header=None,
        names=["timestamp", "active_power_w", "apparent_power_va"],
        usecols=[0, 1, 2],
        chunksize=chunksize,
    ):
        timestamps = pd.to_numeric(chunk["timestamp"], errors="coerce").to_numpy(dtype=np.float64)
        active = pd.to_numeric(chunk["active_power_w"], errors="coerce").to_numpy(dtype=np.float64)
        apparent = pd.to_numeric(chunk["apparent_power_va"], errors="coerce").to_numpy(dtype=np.float64)
        for timestamp_s, active_w, apparent_va in zip(timestamps, active, apparent, strict=True):
            invalid_zero = (
                math.isfinite(float(active_w))
                and math.isfinite(float(apparent_va))
                and active_w == 0.0
                and abs(apparent_va) > invalid_zero_apparent_threshold_va
            )
            valid = (
                math.isfinite(float(timestamp_s))
                and math.isfinite(float(active_w))
                and not invalid_zero
            )
            flags = ("ukdale_invalid_active_zero",) if invalid_zero else ()
            binner.ingest(
                float(timestamp_s),
                float(active_w),
                valid=valid,
                invalid_flags=flags,
            )
    series = binner.finalize()
    if series.nonmonotonic_timestamp_count > maximum_nonmonotonic_records:
        raise ValueError(
            "UK-DALE mains 非单调时间戳数超出既有清理上限："
            f"{series.nonmonotonic_timestamp_count} > {maximum_nonmonotonic_records}"
        )
    channels: list[GroundTruthChannel] = []
    for spec in specs:
        if not spec.source_file.is_file():
            raise FileNotFoundError(f"UK-DALE 分表文件不存在：{spec.source_file}")
        result = detect_on_intervals(
            spec.source_file,
            _threshold_config_for_channel(thresholds, spec.source_appliance),
        )
        channels.append(_ground_truth_channel_from_result(spec, result))
    return series, channels


def load_online_config(config_path: Path) -> dict[str, Any]:
    "Read online_causal_v1_1 TOML and validate every frozen parameter."

    with Path(config_path).open("rb") as handle:
        config = tomllib.load(handle)
    required_sections = {
        "schema",
        "paths",
        "streaming",
        "candidate_detector",
        "window",
        "matching",
        "features",
        "sample_weight",
        "label_policy",
        "split",
        "thresholds",
    }
    missing = required_sections - set(config)
    if missing:
        raise ValueError(f"online config 缺少段：{sorted(missing)}")
    for section in required_sections:
        if not isinstance(config[section], Mapping):
            raise ValueError(f"online config 的 [{section}] 必须是 TOML table")
    schema = config["schema"]
    expected_schema = {
        "schema_version": SCHEMA_VERSION,
        "feature_registry_version": FEATURE_REGISTRY_VERSION,
        "npz_contract_version": SEQUENCE_CONTRACT_VERSION,
    }
    for key, expected in expected_schema.items():
        if str(schema.get(key, "")) != expected:
            raise ValueError(f"schema.{key} 必须冻结为 {expected!r}")

    def require_integer(section: str, key: str, expected: int) -> None:
        value = config[section].get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            raise ValueError(f"{section}.{key} 必须冻结为 {expected}")

    def require_float(section: str, key: str, expected: float) -> None:
        raw_value = config[section].get(key)
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ValueError(f"{section}.{key} 必须冻结为 {expected}")
        value = float(raw_value)
        if not math.isfinite(value) or not math.isclose(
            value, expected, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"{section}.{key} 必须冻结为 {expected}")

    require_integer("streaming", "bin_seconds", 10)
    require_float("streaming", "maximum_raw_gap_seconds", 30.0)
    require_float("streaming", "ukdale_invalid_zero_apparent_threshold_va", 50.0)
    for chunk_key in ("refit_chunksize", "ukdale_chunksize"):
        chunk_value = config["streaming"].get(chunk_key, 0)
        if isinstance(chunk_value, bool) or not isinstance(chunk_value, int) or chunk_value <= 0:
            raise ValueError(f"streaming.{chunk_key} 必须为正整数")

    require_integer("candidate_detector", "baseline_bin_count", 6)
    require_integer("candidate_detector", "minimum_valid_baseline_bins", 5)
    require_float("candidate_detector", "positive_step_threshold_w", 300.0)
    require_integer("candidate_detector", "confirmation_bin_count", 2)
    require_float("candidate_detector", "cooldown_seconds", 30.0)
    require_float("candidate_detector", "maximum_raw_gap_seconds", 30.0)

    require_integer("window", "pre_seconds", 60)
    raw_horizons = config["window"]["horizons_seconds"]
    if not isinstance(raw_horizons, list) or any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in raw_horizons
    ):
        raise ValueError("window.horizons_seconds 只能包含整数秒")
    horizons = tuple(int(value) for value in raw_horizons)
    if horizons != (20, 40, 60, 120):
        raise ValueError("online 样本 horizon 必须固定为 [20, 40, 60, 120]")
    require_float("window", "minimum_pre_coverage", 5.0 / 6.0)
    require_float("window", "minimum_post_coverage", 0.8)
    require_integer("window", "maximum_consecutive_missing_bins", 1)
    require_float("window", "maximum_raw_gap_seconds", 30.0)

    require_float("matching", "onset_match_tolerance_seconds", 20.0)
    require_float("features", "activity_margin_w", 50.0)
    require_float("features", "step_threshold_w", 30.0)
    require_float("features", "relative_epsilon_w", 50.0)

    expected_weights = {
        "core_positive": 1.0,
        "hard_negative": 1.0,
        "background_negative": 0.25,
        "refit_missing_issues_matched_positive": 0.5,
    }
    for key, expected in expected_weights.items():
        require_float("sample_weight", key, expected)
    if config["label_policy"].get("allow_refit_missing_issues_matched_positive") is not True:
        raise ValueError(
            "label_policy.allow_refit_missing_issues_matched_positive 必须冻结为 true"
        )
    if config["label_policy"].get("allow_refit_missing_issues_background_negative") is not False:
        raise ValueError(
            "label_policy.allow_refit_missing_issues_background_negative 必须冻结为 false"
        )
    return config


def _split_lookup(config: Mapping[str, Any]) -> tuple[ExplicitHouseSplit, dict[HouseKey, str]]:
    "Build an unambiguous dataset/house_id lookup from frozen TOML split definitions."

    split = ExplicitHouseSplit.from_mapping(config["split"])
    lookup: dict[HouseKey, str] = {}
    for split_name, keys in split.to_dict().items():
        for value in keys:
            lookup[HouseKey.parse(value)] = split_name
    return split, lookup


def _parse_house_selection(
    values: Sequence[str] | None,
    split_lookup: Mapping[HouseKey, str],
) -> list[HouseKey]:
    "Parse CLI dataset:house filters; return all frozen houses when none are specified."

    if not values:
        return sorted(split_lookup)
    requested: list[HouseKey] = []
    for value in values:
        for token in str(value).split(","):
            token = token.strip()
            if token:
                requested.append(HouseKey.parse(token))
    missing = sorted(
        key.serialise() for key in set(requested) - set(split_lookup)
    )
    if missing:
        raise ValueError(f"--houses 包含不在冻结 split 中的住宅：{missing}")
    return sorted(set(requested))


def _candidate_identifier(dataset: str, house_id: str, onset_s: float) -> str:
    "Use integer milliseconds for stable candidate IDs without floating-point text differences."

    return f"{dataset}_h{house_id}_t{int(round(float(onset_s) * 1000.0))}"


def _candidate_row_sort_key(row: Mapping[str, object]) -> tuple[str, str, int, str]:
    "Return the stable ordering key shared by every candidate-related output.\n\nOutputs must not independently infer candidate_id text order. The candidate table contract explicitly includes dataset, house and onset; NPZ, QC, features and long tables reuse this key to align row-by-row with candidates.csv."

    return (
        str(row["dataset"]),
        _normalise_house_text(row["house_id"]),
        int(row["candidate_onset_s"]),
        str(row["candidate_id"]),
    )


def _sha256_file(path: Path) -> str:
    "Compute file SHA-256 in fixed blocks without loading a large input into memory."

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _describe_file(path: Path, project_root: Path) -> dict[str, object]:
    "Describe a manifest file with its relative path, size and SHA-256."

    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        relative = resolved.as_posix()
    return {
        "path": relative,
        "size_bytes": int(resolved.stat().st_size),
        "sha256": _sha256_file(resolved),
    }


def _describe_staged_output(path: Path, *, final_dir: Path, project_root: Path) -> dict[str, object]:
    "Hash staging files while recording their final committed paths in the manifest."

    try:
        final_relative = final_dir.resolve().relative_to(project_root.resolve()) / path.name
        display_path = final_relative.as_posix()
    except ValueError:
        display_path = (final_dir / path.name).as_posix()
    return {
        "path": display_path,
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }


def _json_default(value: object) -> object:
    "Keep NumPy scalars and paths JSON-readable in the manifest."

    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"无法 JSON 序列化：{type(value)!r}")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    "Write audit JSON with UTF-8 and stable indentation."

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> None:
    "Write CSV in explicit column order; empty tables retain their schema for downstream readers."

    frame = pd.DataFrame(list(rows), columns=list(columns))
    frame.to_csv(path, index=False, encoding="utf-8")


def _append_aggregate_bin_audit(
    path: Path,
    *,
    dataset: str,
    house_id: str,
    split: str,
    series: OnlineBinSeries,
    bin_seconds: int,
) -> int:
    "Append a complete fixed UTC grid for one house to the aggregate-bin audit table.\n\nThis table audits candidate-detector inputs; it is not a model input. Bins with no raw records inside long gaps are explicitly written as NaN/invalid rows, so gaps cannot be silently compressed or treated as true OFF. Write one house at a time rather than retaining all houses in memory."

    columns = (
        "dataset",
        "house_id",
        "split",
        "bin_left_s",
        "bin_right_s",
        "aggregate_power_w",
        "raw_valid_sample_count",
        "quality_valid",
        "maximum_raw_gap_seconds",
        "quality_flags",
    )
    write_header = not path.exists()
    row_count = 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if write_header:
            writer.writeheader()
        if series.first_bin_left_s is None or series.last_bin_left_s is None:
            return row_count
        for bin_left_s in range(
            int(series.first_bin_left_s),
            int(series.last_bin_left_s) + int(bin_seconds),
            int(bin_seconds),
        ):
            item = series.bin_at(bin_left_s, bin_seconds)
            writer.writerow(
                {
                    "dataset": dataset,
                    "house_id": house_id,
                    "split": split,
                    "bin_left_s": int(item.bin_left_s),
                    "bin_right_s": int(item.bin_right_s),
                    "aggregate_power_w": item.aggregate_power_w,
                    "raw_valid_sample_count": int(item.raw_valid_sample_count),
                    "quality_valid": int(item.quality_valid),
                    "maximum_raw_gap_seconds": item.maximum_raw_gap_seconds,
                    "quality_flags": ";".join(item.quality_flags),
                }
            )
            row_count += 1
    return row_count


def _unicode_array(values: Sequence[object]) -> np.ndarray:
    "Build a fixed Unicode dtype so NPZ loads with allow_pickle=False."

    text_values = [str(value) for value in values]
    width = max([1, *(len(value) for value in text_values)])
    return np.asarray(text_values, dtype=f"<U{width}")


def _copy_snapshot(source: Path, destination: Path) -> None:
    "Copy configuration and taxonomy snapshots into staging to isolate outputs from later edits."

    shutil.copy2(source, destination)


def _detector_config_from_mapping(values: Mapping[str, object]) -> CandidateDetectionConfig:
    "Convert the TOML detector section into an aggregate-only candidate-detector configuration."

    return CandidateDetectionConfig(
        bin_seconds=float(values["bin_seconds"]) if "bin_seconds" in values else 10.0,
        baseline_bin_count=int(values["baseline_bin_count"]),
        minimum_valid_baseline_bins=int(values["minimum_valid_baseline_bins"]),
        positive_step_threshold_w=float(values["positive_step_threshold_w"]),
        confirmation_bin_count=int(values["confirmation_bin_count"]),
        cooldown_seconds=float(values["cooldown_seconds"]),
        maximum_raw_gap_seconds=float(values["maximum_raw_gap_seconds"]),
    )


def _make_feature_config(config: Mapping[str, Any]) -> OnlineFeatureConfig:
    "Build online_feature_v1_1 configuration from the same window parameters used for bins."

    return OnlineFeatureConfig(
        bin_seconds=int(config["streaming"]["bin_seconds"]),
        pre_bin_count=int(config["window"]["pre_seconds"]) // int(config["streaming"]["bin_seconds"]),
        activity_margin_w=float(config["features"]["activity_margin_w"]),
        step_threshold_w=float(config["features"]["step_threshold_w"]),
        relative_epsilon_w=float(config["features"]["relative_epsilon_w"]),
    )


def _max_nonmonotonic_records_for_ukdale(house_id: str) -> int:
    "Retain existing H2/H5 mains-cleaning limits; never silently reorder or repair records beyond them."

    limits = {"2": 2, "5": 0}
    if house_id not in limits:
        raise ValueError(f"online v1 没有 UK-DALE H{house_id} 的 mains 清理契约")
    return limits[house_id]


def _select_aggregate_source_path(
    aggregate_inventory: pd.DataFrame,
    *,
    project_root: Path,
    dataset: str,
    house_id: str,
) -> Path:
    "Choose REFIT Aggregate or UK-DALE mains active-W under the Phase 1 measurement semantics."

    required = {"dataset", "house_id", "source_file", "source_kind"}
    missing = required - set(aggregate_inventory.columns)
    if missing:
        raise ValueError(f"aggregate source inventory 缺少列：{sorted(missing)}")
    rows = aggregate_inventory[
        (aggregate_inventory["dataset"].astype(str) == dataset)
        & (aggregate_inventory["house_id"].map(_normalise_house_text) == house_id)
    ]
    expected_kind = (
        "primary_inventory_source" if dataset == "refit" else "alternate_high_frequency_mains"
    )
    rows = rows[rows["source_kind"].astype(str) == expected_kind]
    if len(rows) != 1:
        raise ValueError(
            f"{dataset}:{house_id} 未找到唯一的 {expected_kind} aggregate source"
        )
    source_path = _as_project_path(project_root, rows.iloc[0]["source_file"])
    if not source_path.is_file():
        raise FileNotFoundError(f"aggregate source 文件不存在：{source_path}")
    return source_path


def _as_csv_scalar(value: object) -> object:
    "Write None/NaN as empty CSV cells so the literal text nan cannot be mistaken for a label."

    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


def _append_house_candidate_sample_artifacts(
    *,
    candidate_rows: Sequence[Mapping[str, object]],
    series: OnlineBinSeries,
    core_channels: Sequence[GroundTruthChannel],
    core_label_index: Sequence[_CoreChannelLabelIndex] | None = None,
    config: Mapping[str, Any],
    feature_config: OnlineFeatureConfig,
    horizons: Sequence[int],
    bin_seconds: int,
    horizon_qc_rows: list[dict[str, object]],
    sequence_bin_rows: list[dict[str, object]],
    feature_rows: list[dict[str, object]],
    horizon_npz_rows: dict[int, list[dict[str, object]]],
) -> None:
    "Generate QC, sequences and features while one house has resident raw bins, then release its memory."

    pre_seconds = int(config["window"]["pre_seconds"])
    maximum_raw_gap_seconds = float(config["window"]["maximum_raw_gap_seconds"])
    minimum_pre_coverage = float(config["window"]["minimum_pre_coverage"])
    minimum_post_coverage = float(config["window"]["minimum_post_coverage"])
    maximum_consecutive_missing_bins = int(
        config["window"]["maximum_consecutive_missing_bins"]
    )
    stable_core_label_index = (
        tuple(core_label_index)
        if core_label_index is not None
        else _build_core_label_index(core_channels)
    )
    for row in candidate_rows:
        candidate_id = str(row["candidate_id"])
        onset_s = int(row["candidate_onset_s"])
        max_window = extract_candidate_window(
            series,
            onset_s,
            120,
            bin_seconds=bin_seconds,
            pre_seconds=pre_seconds,
        )
        max_target_labels = assign_target_active_labels(
            [item.bin_left_s for item in max_window.bins],
            bin_seconds=bin_seconds,
            core_channels=core_channels,
            core_label_index=stable_core_label_index,
        )
        max_values = build_causal_sequence_inputs(
            max_window.power_w,
            max_window.quality_valid_mask,
            pre_bin_count=feature_config.pre_bin_count,
        )
        for index, item in enumerate(max_window.bins):
            sequence_bin_rows.append(
                {
                    "candidate_id": candidate_id,
                    "relative_bin_index": int(max_window.relative_bin_indices[index]),
                    "bin_left_s": int(item.bin_left_s),
                    "bin_right_s": int(item.bin_right_s),
                    "aggregate_power_w": max_values[index, 0],
                    "raw_valid_sample_count": int(item.raw_valid_sample_count),
                    "maximum_raw_gap_seconds": item.maximum_raw_gap_seconds,
                    "power_minus_pre_mean_w": max_values[index, 1],
                    "adjacent_power_difference_w": max_values[index, 2],
                    "pre_post_indicator": int(max_values[index, 3]),
                    "quality_valid_mask": int(max_window.quality_valid_mask[index]),
                    "target_active_label": int(max_target_labels[index]),
                    "quality_flags": ";".join(item.quality_flags),
                }
            )

        label_value = row["label"]
        binary_label = (
            int(label_value)
            if str(label_value) in {"0", "1"} or label_value in {0, 1}
            else None
        )
        for horizon_seconds in horizons:
            window = extract_candidate_window(
                series,
                onset_s,
                int(horizon_seconds),
                bin_seconds=bin_seconds,
                pre_seconds=pre_seconds,
            )
            quality = assess_online_horizon_quality(
                window.power_w[: feature_config.pre_bin_count],
                window.power_w[feature_config.pre_bin_count :],
                window.quality_valid_mask[: feature_config.pre_bin_count],
                window.quality_valid_mask[feature_config.pre_bin_count :],
                config=feature_config,
                minimum_pre_coverage=minimum_pre_coverage,
                minimum_post_coverage=minimum_post_coverage,
                maximum_consecutive_missing_bins=maximum_consecutive_missing_bins,
            )
            quality_accepted, raw_reasons = assess_candidate_horizon(
                window,
                horizon_seconds=int(horizon_seconds),
                feature_config=feature_config,
                maximum_raw_gap_seconds=maximum_raw_gap_seconds,
                minimum_pre_coverage=minimum_pre_coverage,
                minimum_post_coverage=minimum_post_coverage,
                maximum_consecutive_missing_bins=maximum_consecutive_missing_bins,
            )
            rejection_reasons = list(raw_reasons)
            if binary_label is None:
                rejection_reasons.append(f"label_status_{row['label_status']}")
                if row["ignore_reasons"]:
                    rejection_reasons.extend(
                        part for part in str(row["ignore_reasons"]).split(";") if part
                    )
            accepted = bool(quality_accepted and binary_label in {0, 1})
            stable_reasons = tuple(sorted(set(rejection_reasons)))
            horizon_qc_rows.append(
                {
                    "candidate_id": candidate_id,
                    "dataset": row["dataset"],
                    "house_id": row["house_id"],
                    "split": row["split"],
                    "horizon_seconds": int(horizon_seconds),
                    "accepted": int(accepted),
                    "quality_accepted": int(quality_accepted),
                    "label": _as_csv_scalar(binary_label),
                    "label_type": row["label_type"],
                    "label_confidence": row["label_confidence"],
                    "sample_weight": float(row["sample_weight"]),
                    "source_quality_warnings": row["source_quality_warnings"],
                    "pre_valid_bin_count": quality.pre_valid_bin_count,
                    "post_valid_bin_count": quality.post_valid_bin_count,
                    "required_post_valid_bin_count": quality.required_post_valid_bin_count,
                    "pre_coverage": quality.pre_coverage,
                    "post_coverage": quality.post_coverage,
                    "maximum_consecutive_missing_bins": quality.maximum_consecutive_missing_bins,
                    "rejection_reasons": ";".join(stable_reasons),
                }
            )
            if not accepted:
                continue
            target_labels = assign_target_active_labels(
                [item.bin_left_s for item in window.bins],
                bin_seconds=bin_seconds,
                core_channels=core_channels,
                core_label_index=stable_core_label_index,
            )
            values = build_causal_sequence_inputs(
                window.power_w,
                window.quality_valid_mask,
                pre_bin_count=feature_config.pre_bin_count,
            )
            horizon_npz_rows[int(horizon_seconds)].append(
                {
                    "candidate_id": candidate_id,
                    "values": values,
                    "quality_valid_mask": window.quality_valid_mask.astype(np.uint8),
                    "target_active_label": target_labels.astype(np.int8),
                    "label": int(binary_label),
                    "sample_weight": float(row["sample_weight"]),
                    "split": str(row["split"]),
                }
            )
            feature = extract_online_features(
                window.power_w[: feature_config.pre_bin_count],
                window.power_w[feature_config.pre_bin_count :],
                window.quality_valid_mask[: feature_config.pre_bin_count],
                window.quality_valid_mask[feature_config.pre_bin_count :],
                horizon_seconds=int(horizon_seconds),
                config=feature_config,
            )
            feature_rows.append(
                {
                    "candidate_id": candidate_id,
                    "dataset": row["dataset"],
                    "house_id": row["house_id"],
                    "split": row["split"],
                    "candidate_onset_s": onset_s,
                    "candidate_detected_at_s": row["candidate_detected_at_s"],
                    "horizon_seconds": int(horizon_seconds),
                    "label": int(binary_label),
                    "label_type": row["label_type"],
                    "label_confidence": row["label_confidence"],
                    "sample_weight": float(row["sample_weight"]),
                    "source_appliance": row["matched_source_appliance"],
                    "accepted": True,
                    **feature,
                }
            )


def build_online_causal_samples(
    *,
    project_root: Path,
    config_path: Path,
    output_dir: Path | None = None,
    houses: Sequence[str] | None = None,
    smoke_test: bool = False,
    maximum_candidates_per_house: int | None = None,
) -> dict[str, object]:
    "Build online_causal_v1_1 outputs and atomically commit the directory only after every step succeeds.\n\nFormal runs prohibit candidate caps; caps are allowed only for explicitly nonformal smoke runs. Read one house at a time and scan each REFIT CSV once, feeding aggregate and selected submeter state machines together instead of rereading multi-GB house files for each appliance."

    project_root = Path(project_root).resolve()
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = (project_root / config_path).resolve()
    config = load_online_config(config_path)
    split, split_lookup = _split_lookup(config)
    selected_houses = _parse_house_selection(houses, split_lookup)
    if maximum_candidates_per_house is not None:
        if maximum_candidates_per_house <= 0:
            raise ValueError("maximum_candidates_per_house 必须为正数")
        if not smoke_test:
            raise ValueError("正式模式不允许 maximum_candidates_per_house cap")
    paths_config = config["paths"]
    appliance_inventory_path = _as_project_path(project_root, paths_config["appliance_inventory"])
    aggregate_inventory_path = _as_project_path(
        project_root, paths_config["aggregate_source_inventory"]
    )
    taxonomy_path = _as_project_path(project_root, paths_config["taxonomy"])
    missing_phase1 = [
        path
        for path in (appliance_inventory_path, aggregate_inventory_path)
        if not path.is_file()
    ]
    if missing_phase1:
        formatted = ", ".join(str(path) for path in missing_phase1)
        raise FileNotFoundError(
            "Phase 1 inventory 尚未完整生成，不能使用 artifacts 替代；缺少："
            f"{formatted}。请先运行既有 Phase 1 四步脚本。"
        )
    if not taxonomy_path.is_file():
        raise FileNotFoundError(f"taxonomy 文件不存在：{taxonomy_path}")
    appliance_inventory = pd.read_csv(appliance_inventory_path, encoding="utf-8-sig")
    aggregate_inventory = pd.read_csv(aggregate_inventory_path, encoding="utf-8-sig")
    default_output = _as_project_path(project_root, paths_config["default_output_dir"])
    if smoke_test and output_dir is None:
        final_dir = default_output.parent / f"{default_output.name}_smoke"
    else:
        final_dir = Path(output_dir) if output_dir is not None else default_output
    if not final_dir.is_absolute():
        final_dir = (project_root / final_dir).resolve()
    if smoke_test and final_dir == default_output.resolve():
        raise ValueError("smoke-test 必须使用独立输出目录，不能指向正式 default_output_dir")
    if final_dir.exists():
        raise FileExistsError(
            f"目标输出目录已存在，拒绝覆盖：{final_dir}；请传入新的 --output-dir"
        )
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{final_dir.name}.staging-", dir=final_dir.parent))

    feature_config = _make_feature_config(config)
    detector_config = _detector_config_from_mapping(config["candidate_detector"])
    bin_seconds = int(config["streaming"]["bin_seconds"])
    horizons = tuple(int(item) for item in config["window"]["horizons_seconds"])
    all_candidate_rows: list[dict[str, object]] = []
    detector_audit_rows: list[dict[str, object]] = []
    unmatched_rows: list[dict[str, object]] = []
    matched_rows: list[dict[str, object]] = []
    horizon_qc_rows: list[dict[str, object]] = []
    sequence_bin_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    horizon_npz_rows: dict[int, list[dict[str, object]]] = {horizon: [] for horizon in horizons}
    input_paths: set[Path] = {config_path, appliance_inventory_path, aggregate_inventory_path, taxonomy_path}
    house_runtime_audit: dict[str, dict[str, object]] = {}
    aggregate_bin_audit_path = staging_dir / "aggregate_bins.csv"
    matched_positive_total = 0
    eligible_positive_total = 0

    try:
        for house_key in selected_houses:
            dataset = house_key.dataset
            house_id = house_key.house_id
            split_name = split_lookup[house_key]
            core_specs, hard_specs, transfer_specs = select_label_channels(
                appliance_inventory,
                project_root=project_root,
                dataset=dataset,
                house_id=house_id,
            )
            all_specs = [*core_specs, *hard_specs, *transfer_specs]
            aggregate_path = _select_aggregate_source_path(
                aggregate_inventory,
                project_root=project_root,
                dataset=dataset,
                house_id=house_id,
            )
            input_paths.add(aggregate_path)
            input_paths.update(spec.source_file for spec in all_specs)
            if dataset == "refit":
                series, channels, source_flags = _load_refit_house_streaming(
                    aggregate_path=aggregate_path,
                    specs=all_specs,
                    thresholds=config["thresholds"],
                    bin_seconds=bin_seconds,
                    maximum_raw_gap_seconds=float(config["streaming"]["maximum_raw_gap_seconds"]),
                    chunksize=int(config["streaming"]["refit_chunksize"]),
                )
            elif dataset == "uk_dale":
                series, channels = _load_ukdale_house_streaming(
                    mains_path=aggregate_path,
                    specs=all_specs,
                    thresholds=config["thresholds"],
                    bin_seconds=bin_seconds,
                    maximum_raw_gap_seconds=float(config["streaming"]["maximum_raw_gap_seconds"]),
                    chunksize=int(config["streaming"]["ukdale_chunksize"]),
                    invalid_zero_apparent_threshold_va=float(
                        config["streaming"]["ukdale_invalid_zero_apparent_threshold_va"]
                    ),
                    maximum_nonmonotonic_records=_max_nonmonotonic_records_for_ukdale(
                        house_id
                    ),
                )
                source_flags = ()
            else:
                raise ValueError(f"online v1.1 不支持的数据集：{dataset}")

            aggregate_bin_count = _append_aggregate_bin_audit(
                aggregate_bin_audit_path,
                dataset=dataset,
                house_id=house_id,
                split=split_name,
                series=series,
                bin_seconds=bin_seconds,
            )

            core_channels = [
                channel for channel in channels if channel.taxonomy_role in CORE_POSITIVE_ROLES
            ]
            core_label_index = _build_core_label_index(core_channels)
            hard_channels = [
                channel for channel in channels if channel.taxonomy_role in HARD_NEGATIVE_ROLES
            ]
            transfer_channels = [
                channel for channel in channels if channel.taxonomy_role in TRANSFER_POSITIVE_ROLES
            ]
            ordered_bins = [series.bins_by_left_s[key] for key in sorted(series.bins_by_left_s)]
            detection = detect_aggregate_candidates(
                ordered_bins,
                detector_config,
                audit_mode="transitions",
            )
            house_candidates = list(detection.candidates)
            cap_applied = False
            if maximum_candidates_per_house is not None and len(house_candidates) > maximum_candidates_per_house:
                house_candidates = house_candidates[:maximum_candidates_per_house]
                cap_applied = True
            house_rows: list[dict[str, object]] = []
            for candidate in house_candidates:
                candidate_id = _candidate_identifier(dataset, house_id, candidate.candidate_onset_s)
                house_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "dataset": dataset,
                        "house_id": house_id,
                        "split": split_name,
                        "candidate_onset_s": int(round(candidate.candidate_onset_s)),
                        "candidate_detected_at_s": int(round(candidate.candidate_detected_at_s)),
                        "detector_internal_delay_s": float(candidate.detector_internal_delay_s),
                        "detector_baseline_w": float(candidate.detector_baseline_w),
                        "first_step_w": float(candidate.first_step_w),
                        "confirmation_step_w": float(candidate.confirmation_step_w),
                    }
                )
            if len({row["candidate_id"] for row in house_rows}) != len(house_rows):
                raise RuntimeError(f"{dataset}:{house_id} 出现重复 candidate_id")

            label_audits, matched_core, unmatched_core = assign_candidate_labels(
                house_rows,
                core_channels=core_channels,
                hard_channels=hard_channels,
                transfer_channels=transfer_channels,
                onset_match_tolerance_seconds=float(config["matching"]["onset_match_tolerance_seconds"]),
                pre_seconds=int(config["window"]["pre_seconds"]),
                maximum_horizon_seconds=120,
                core_positive_sample_weight=float(
                    config["sample_weight"]["core_positive"]
                ),
                hard_negative_sample_weight=float(
                    config["sample_weight"]["hard_negative"]
                ),
                background_negative_sample_weight=float(
                    config["sample_weight"]["background_negative"]
                ),
            )
            label_audits = apply_source_quality_policy(
                label_audits,
                source_quality_flags=source_flags,
                allow_refit_missing_issues_matched_positive=bool(
                    config["label_policy"][
                        "allow_refit_missing_issues_matched_positive"
                    ]
                ),
                allow_refit_missing_issues_background_negative=bool(
                    config["label_policy"][
                        "allow_refit_missing_issues_background_negative"
                    ]
                ),
                missing_issues_positive_sample_weight=float(
                    config["sample_weight"][
                        "refit_missing_issues_matched_positive"
                    ]
                ),
            )
            matched_positive_total += len(matched_core)
            eligible_positive_total += len(ground_truth_events_from_channels(core_channels))
            for row in house_rows:
                audit = label_audits[str(row["candidate_id"])]
                row.update(
                    {
                        "matched_event_id": _as_csv_scalar(audit.matched_event_id),
                        "matched_source_appliance": _as_csv_scalar(
                            audit.matched_source_appliance
                        ),
                        "onset_match_error_seconds": _as_csv_scalar(
                            audit.onset_match_error_seconds
                        ),
                        "detector_report_delay_seconds": _as_csv_scalar(
                            audit.detector_report_delay_seconds
                        ),
                        "label": _as_csv_scalar(audit.label),
                        "label_type": audit.label_type,
                        "label_status": audit.label_status,
                        "ignore_reasons": ";".join(audit.ignore_reasons),
                        "label_confidence": audit.label_confidence,
                        "sample_weight": float(audit.sample_weight),
                        "source_quality_warnings": ";".join(
                            audit.source_quality_warnings
                        ),
                    }
                )
            for event in unmatched_core:
                unmatched_rows.append(
                    {
                        "event_id": event.event_id,
                        "dataset": event.dataset,
                        "house_id": event.house_id,
                        "split": split_name,
                        "channel_id": event.channel_id,
                        "source_appliance": event.source_appliance,
                        "ground_truth_onset_s": event.start_timestamp_s,
                        "ground_truth_end_s": event.end_timestamp_s,
                        "touches_submeter_boundary": int(event.touches_boundary),
                    }
                )
            for event in matched_core:
                matching_candidate = next(
                    row for row in house_rows if row["matched_event_id"] == event.event_id
                )
                matched_rows.append(
                    {
                        "event_id": event.event_id,
                        "candidate_id": matching_candidate["candidate_id"],
                        "dataset": event.dataset,
                        "house_id": event.house_id,
                        "split": split_name,
                        "channel_id": event.channel_id,
                        "source_appliance": event.source_appliance,
                        "ground_truth_onset_s": event.start_timestamp_s,
                        "candidate_onset_s": matching_candidate["candidate_onset_s"],
                        "onset_match_error_seconds": matching_candidate[
                            "onset_match_error_seconds"
                        ],
                        "detector_report_delay_seconds": matching_candidate[
                            "detector_report_delay_seconds"
                        ],
                    }
                )
            # The detector returns transitions and quality resets in transitions mode; counts of action/reason
            # across all scanned bins stay in result, avoiding vast repeated baseline rows for large houses.
            for audit_row in detection.audit_rows:
                detector_audit_rows.append(
                    {
                        "dataset": dataset,
                        "house_id": house_id,
                        "split": split_name,
                        **asdict(audit_row),
                    }
                )
            _append_house_candidate_sample_artifacts(
                candidate_rows=house_rows,
                series=series,
                core_channels=core_channels,
                core_label_index=core_label_index,
                config=config,
                feature_config=feature_config,
                horizons=horizons,
                bin_seconds=bin_seconds,
                horizon_qc_rows=horizon_qc_rows,
                sequence_bin_rows=sequence_bin_rows,
                feature_rows=feature_rows,
                horizon_npz_rows=horizon_npz_rows,
            )
            all_candidate_rows.extend(house_rows)
            house_runtime_audit[house_key.serialise()] = {
                "split": split_name,
                "aggregate_source": str(aggregate_path),
                "selected_core_channels": [spec.channel_id for spec in core_specs],
                "selected_hard_channels": [spec.channel_id for spec in hard_specs],
                "selected_transfer_channels": [spec.channel_id for spec in transfer_specs],
                "candidate_detector_candidates_before_cap": int(detection.candidate_count),
                "candidate_cap_applied": cap_applied,
                "candidate_detector_audit_scope": "state_transitions_and_quality_resets",
                "candidate_detector_action_counts": detection.action_count_mapping(),
                "candidate_detector_reason_counts": detection.reason_count_mapping(),
                "core_positive_ground_truth_onset_count": int(
                    len(ground_truth_events_from_channels(core_channels))
                ),
                "matched_core_positive_onset_count": int(len(matched_core)),
                "unmatched_core_positive_onset_count": int(len(unmatched_core)),
                "source_quality_flags": list(source_flags),
                "aggregate_nonmonotonic_timestamp_count": int(
                    series.nonmonotonic_timestamp_count
                ),
                "aggregate_bin_audit_count": int(aggregate_bin_count),
            }
            # This house now has lightweight materialized sample rows; release its compressed bins and submeter results.
            del (
                ordered_bins,
                detection,
                series,
                channels,
                core_channels,
                core_label_index,
                hard_channels,
                transfer_channels,
            )

        all_candidate_rows.sort(key=_candidate_row_sort_key)
        if len({str(row["candidate_id"]) for row in all_candidate_rows}) != len(all_candidate_rows):
            raise RuntimeError("全局 candidate_id 不唯一")
        overlap_clusters = assign_overlap_clusters(all_candidate_rows)
        for row in all_candidate_rows:
            row["window_overlap_cluster_id"] = overlap_clusters[str(row["candidate_id"])]
        candidate_sort_rank = {
            str(row["candidate_id"]): index
            for index, row in enumerate(all_candidate_rows)
        }
        candidate_frame_for_leakage = pd.DataFrame(all_candidate_rows)
        if not candidate_frame_for_leakage.empty:
            assert_no_split_leakage(candidate_frame_for_leakage)

        candidate_columns = (
            "candidate_id",
            "dataset",
            "house_id",
            "split",
            "candidate_onset_s",
            "candidate_detected_at_s",
            "detector_internal_delay_s",
            "detector_baseline_w",
            "first_step_w",
            "confirmation_step_w",
            "matched_event_id",
            "matched_source_appliance",
            "onset_match_error_seconds",
            "detector_report_delay_seconds",
            "label",
            "label_type",
            "label_status",
            "label_confidence",
            "sample_weight",
            "source_quality_warnings",
            "ignore_reasons",
            "window_overlap_cluster_id",
        )
        label_audit_columns = (
            *candidate_columns,
            "h120_aggregate_quality_accepted",
        )
        # Each house populated these lightweight sample rows before releasing its compressed bins.

        # Candidate windows were materialized in the per-house loop; never revisit released raw bins here.
        horizon_qc_rows.sort(
            key=lambda item: (
                candidate_sort_rank[str(item["candidate_id"])],
                int(item["horizon_seconds"]),
            )
        )
        sequence_bin_rows.sort(
            key=lambda item: (
                candidate_sort_rank[str(item["candidate_id"])],
                int(item["relative_bin_index"]),
            )
        )
        feature_rows.sort(
            key=lambda item: (
                candidate_sort_rank[str(item["candidate_id"])],
                int(item["horizon_seconds"]),
            )
        )
        unmatched_rows.sort(
            key=lambda item: (
                str(item["dataset"]),
                _normalise_house_text(item["house_id"]),
                float(item["ground_truth_onset_s"]),
                str(item["event_id"]),
            )
        )
        matched_rows.sort(
            key=lambda item: (
                str(item["dataset"]),
                _normalise_house_text(item["house_id"]),
                float(item["ground_truth_onset_s"]),
                str(item["event_id"]),
            )
        )
        h120_quality_by_candidate = {
            str(item["candidate_id"]): bool(item["quality_accepted"])
            for item in horizon_qc_rows
            if int(item["horizon_seconds"]) == 120
        }
        candidate_label_rows = [
            {
                **row,
                "h120_aggregate_quality_accepted": int(
                    h120_quality_by_candidate.get(str(row["candidate_id"]), False)
                ),
            }
            for row in all_candidate_rows
        ]
        for house_serialized, audit in house_runtime_audit.items():
            house_key = HouseKey.parse(house_serialized)
            house_candidates = [
                row
                for row in all_candidate_rows
                if str(row["dataset"]) == house_key.dataset
                and _normalise_house_text(row["house_id"]) == house_key.house_id
            ]
            audit["candidate_count_after_cap"] = len(house_candidates)
            audit["candidate_label_type_counts"] = dict(
                sorted(Counter(str(row["label_type"]) for row in house_candidates).items())
            )
            audit["horizon_statistics"] = {}
            candidate_ids = {str(row["candidate_id"]) for row in house_candidates}
            for horizon_seconds in horizons:
                rows = [
                    row
                    for row in horizon_qc_rows
                    if int(row["horizon_seconds"]) == horizon_seconds
                    and str(row["candidate_id"]) in candidate_ids
                ]
                accepted_rows = [row for row in rows if int(row["accepted"]) == 1]
                audit["horizon_statistics"][f"h{horizon_seconds}"] = {
                    "accepted": len(accepted_rows),
                    "rejected_or_ignored": len(rows) - len(accepted_rows),
                    "label_0": sum(int(row["label"]) == 0 for row in accepted_rows),
                    "label_1": sum(int(row["label"]) == 1 for row in accepted_rows),
                }

        _write_csv(staging_dir / "candidates.csv", all_candidate_rows, candidate_columns)
        _write_csv(
            staging_dir / "candidate_label_audit.csv",
            candidate_label_rows,
            label_audit_columns,
        )
        _write_csv(
            staging_dir / "candidate_horizon_qc.csv",
            horizon_qc_rows,
            (
                "candidate_id",
                "dataset",
                "house_id",
                "split",
                "horizon_seconds",
                "accepted",
                "quality_accepted",
                "label",
                "label_type",
                "label_confidence",
                "sample_weight",
                "source_quality_warnings",
                "pre_valid_bin_count",
                "post_valid_bin_count",
                "required_post_valid_bin_count",
                "pre_coverage",
                "post_coverage",
                "maximum_consecutive_missing_bins",
                "rejection_reasons",
            ),
        )
        _write_csv(
            staging_dir / "candidate_detector_audit.csv",
            detector_audit_rows,
            (
                "dataset",
                "house_id",
                "split",
                "bin_left_s",
                "bin_right_s",
                "state_before",
                "state_after",
                "action",
                "reason",
                "quality_valid",
                "baseline_w",
                "power_minus_baseline_w",
            ),
        )
        _write_csv(
            staging_dir / "sequence_bins.csv",
            sequence_bin_rows,
            (
                "candidate_id",
                "relative_bin_index",
                "bin_left_s",
                "bin_right_s",
                "aggregate_power_w",
                "raw_valid_sample_count",
                "maximum_raw_gap_seconds",
                "power_minus_pre_mean_w",
                "adjacent_power_difference_w",
                "pre_post_indicator",
                "quality_valid_mask",
                "target_active_label",
                "quality_flags",
            ),
        )
        _write_csv(
            staging_dir / "causal_features.csv",
            feature_rows,
            (
                "candidate_id",
                "dataset",
                "house_id",
                "split",
                "candidate_onset_s",
                "candidate_detected_at_s",
                "horizon_seconds",
                "label",
                "label_type",
                "label_confidence",
                "sample_weight",
                "source_appliance",
                "accepted",
                *ONLINE_FEATURE_COLUMNS,
                *ONLINE_MASK_COLUMNS,
            ),
        )
        _write_csv(
            staging_dir / "unmatched_positive_onsets.csv",
            unmatched_rows,
            (
                "event_id",
                "dataset",
                "house_id",
                "split",
                "channel_id",
                "source_appliance",
                "ground_truth_onset_s",
                "ground_truth_end_s",
                "touches_submeter_boundary",
            ),
        )
        _write_csv(
            staging_dir / "matched_positive_onsets.csv",
            matched_rows,
            (
                "event_id",
                "candidate_id",
                "dataset",
                "house_id",
                "split",
                "channel_id",
                "source_appliance",
                "ground_truth_onset_s",
                "candidate_onset_s",
                "onset_match_error_seconds",
                "detector_report_delay_seconds",
            ),
        )

        npz_counts: dict[int, int] = {}
        for horizon_seconds in horizons:
            # Do not write NPZ in house-scan order. After retaining accepted candidates, preserve the global
            # canonical candidates.csv order so CSV and NPZ rows can be joined unambiguously.
            rows = sorted(
                horizon_npz_rows[horizon_seconds],
                key=lambda item: candidate_sort_rank[str(item["candidate_id"])],
            )
            expected_length = feature_config.pre_bin_count + post_bin_count_for_horizon(
                horizon_seconds, feature_config
            )
            if rows:
                values = np.stack([item["values"] for item in rows]).astype(np.float64, copy=False)
                quality_masks = np.stack(
                    [item["quality_valid_mask"] for item in rows]
                ).astype(np.uint8, copy=False)
                target_labels = np.stack(
                    [item["target_active_label"] for item in rows]
                ).astype(np.int8, copy=False)
            else:
                values = np.empty((0, expected_length, len(SEQUENCE_FEATURE_NAMES)), dtype=np.float64)
                quality_masks = np.empty((0, expected_length), dtype=np.uint8)
                target_labels = np.empty((0, expected_length), dtype=np.int8)
            candidate_ids = _unicode_array([item["candidate_id"] for item in rows])
            splits = _unicode_array([item["split"] for item in rows])
            labels = np.asarray([item["label"] for item in rows], dtype=np.int8)
            sample_weights = np.asarray(
                [item["sample_weight"] for item in rows], dtype=np.float32
            )
            relative_indices = relative_bin_indices_for_horizon(horizon_seconds, feature_config)
            np.savez_compressed(
                staging_dir / f"sequences_h{horizon_seconds}.npz",
                candidate_ids=candidate_ids,
                values=values,
                quality_valid_mask=quality_masks,
                target_active_label=target_labels,
                labels=labels,
                sample_weights=sample_weights,
                split=splits,
                sequence_feature_names=_unicode_array(SEQUENCE_FEATURE_NAMES),
                relative_bin_indices=relative_indices.astype(np.int64),
                horizon_seconds=np.asarray(horizon_seconds, dtype=np.int64),
            )
            npz_counts[horizon_seconds] = int(len(rows))

        rejection_counter: Counter[str] = Counter()
        for row in horizon_qc_rows:
            for reason in str(row["rejection_reasons"]).split(";"):
                if reason:
                    rejection_counter[reason] += 1
        label_type_counts = Counter(str(row["label_type"]) for row in all_candidate_rows)
        label_confidence_counts = Counter(
            str(row["label_confidence"]) for row in all_candidate_rows
        )
        accepted_by_horizon_and_label: dict[str, dict[str, int]] = {}
        for horizon_seconds in horizons:
            accepted_rows = [
                row
                for row in horizon_qc_rows
                if int(row["horizon_seconds"]) == horizon_seconds and int(row["accepted"]) == 1
            ]
            accepted_by_horizon_and_label[f"h{horizon_seconds}"] = {
                "total": len(accepted_rows),
                "label_0": sum(int(row["label"]) == 0 for row in accepted_rows),
                "label_1": sum(int(row["label"]) == 1 for row in accepted_rows),
            }
        _write_json(
            staging_dir / "rejection_summary.json",
            {
                "schema_version": SCHEMA_VERSION,
                "candidate_label_type_counts": dict(sorted(label_type_counts.items())),
                "candidate_horizon_rejection_reason_counts": dict(
                    sorted(rejection_counter.items())
                ),
                "accepted_by_horizon_and_label": accepted_by_horizon_and_label,
            },
        )
        detector_metrics = {
            "schema_version": SCHEMA_VERSION,
            "candidate_detector_uses_only": "aggregate_10_second_active_power_w_bins",
            "core_positive_ground_truth_onsets": int(eligible_positive_total),
            "matched_positive_onsets": int(matched_positive_total),
            "unmatched_positive_onsets": int(eligible_positive_total - matched_positive_total),
            "positive_candidate_recall": (
                float(matched_positive_total / eligible_positive_total)
                if eligible_positive_total
                else None
            ),
            "accepted_positive_by_horizon": {
                key: int(value["label_1"])
                for key, value in accepted_by_horizon_and_label.items()
            },
            "accepted_positive_yield_by_horizon": {
                key: (
                    float(value["label_1"] / eligible_positive_total)
                    if eligible_positive_total
                    else None
                )
                for key, value in accepted_by_horizon_and_label.items()
            },
            "onset_match_tolerance_seconds": float(
                config["matching"]["onset_match_tolerance_seconds"]
            ),
            "per_house": house_runtime_audit,
        }
        _write_json(staging_dir / "detector_metrics.json", detector_metrics)

        _copy_snapshot(config_path, staging_dir / "config_snapshot.toml")
        _copy_snapshot(taxonomy_path, staging_dir / "taxonomy_snapshot.yaml")
        _write_sample_report(
            staging_dir / "sample_report.md",
            all_candidate_rows=all_candidate_rows,
            horizon_qc_rows=horizon_qc_rows,
            detector_metrics=detector_metrics,
            smoke_test=smoke_test,
            maximum_candidates_per_house=maximum_candidates_per_house,
            house_runtime_audit=house_runtime_audit,
        )

        implementation_paths = [
            project_root / "src" / "load_signature_analysis" / "online_candidate_detection.py",
            project_root / "src" / "load_signature_analysis" / "online_features.py",
            project_root / "src" / "load_signature_analysis" / "online_samples.py",
            project_root / "scripts" / "build_online_causal_samples.py",
        ]
        output_descriptions = {
            path.name: _describe_staged_output(
                path,
                final_dir=final_dir,
                project_root=project_root,
            )
            for path in sorted(staging_dir.iterdir())
            if path.is_file() and path.name != "manifest.json"
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "smoke" if smoke_test else "formal",
            "maximum_candidates_per_house": maximum_candidates_per_house,
            "candidate_cap_is_nonformal_sampling": bool(maximum_candidates_per_house is not None),
            "config_snapshot": config,
            "taxonomy_filter_contract": {
                "main": "include_in_main_mvp && data_use=main && taxonomy_confidence=high && role in {core_positive, hard_negative}",
                "transfer_audit_only": "taxonomy_role=extended_clean_positive && data_use=transfer_only && taxonomy_confidence=high",
            },
            "label_policy_contract": {
                "background_negative": "weak label: only monitored reliable positives are ruled out; use sample_weights",
                "maximum_window_submeter_purity": "[-60s,+120s) labels only; aggregate QC is horizon-local",
                "h120_aggregate_quality_field": "audit only; never controls H20/H40/H60 membership",
                "refit_missing_issues": "matched core positive may enter at medium confidence; background and hard negative are excluded",
            },
            "sample_weight_contract": dict(config["sample_weight"]),
            "split": split.to_dict(),
            "selected_houses": [key.serialise() for key in selected_houses],
            "candidate_detector_configuration": asdict(detector_config),
            "horizons_seconds": list(horizons),
            "online_feature_registry": list(ONLINE_FEATURE_COLUMNS),
            "feature_registry_version": FEATURE_REGISTRY_VERSION,
            "sequence_channel_order": list(SEQUENCE_FEATURE_NAMES),
            "npz_contract_version": SEQUENCE_CONTRACT_VERSION,
            "npz_keys": [
                "candidate_ids",
                "values",
                "quality_valid_mask",
                "target_active_label",
                "labels",
                "sample_weights",
                "split",
                "sequence_feature_names",
                "relative_bin_indices",
                "horizon_seconds",
            ],
            "input_files": [
                _describe_file(path, project_root) for path in sorted(input_paths)
            ],
            "input_hash_note": "SHA-256 audit pass is separate from the single semantic REFIT CSV scan.",
            "implementation_files": [
                _describe_file(path, project_root)
                for path in implementation_paths
                if path.is_file()
            ],
            "per_house_statistics": house_runtime_audit,
            "candidate_count": len(all_candidate_rows),
            "candidate_label_type_counts": dict(sorted(label_type_counts.items())),
            "candidate_label_confidence_counts": dict(
                sorted(label_confidence_counts.items())
            ),
            "horizon_sample_counts": {f"h{key}": value for key, value in npz_counts.items()},
            "horizon_class_counts": accepted_by_horizon_and_label,
            "output_files": output_descriptions,
            "manifest_self_hash": "not_applicable: manifest is the completion marker and excludes self-reference",
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
        }
        _write_json(staging_dir / "manifest.json", manifest)
        if final_dir.exists():
            raise FileExistsError(f"目标输出目录在构建期间出现，拒绝覆盖：{final_dir}")
        staging_dir.replace(final_dir)
        return {
            "output_dir": final_dir,
            "candidate_count": len(all_candidate_rows),
            "horizon_sample_counts": npz_counts,
            "detector_metrics": detector_metrics,
        }
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def _write_sample_report(
    path: Path,
    *,
    all_candidate_rows: Sequence[Mapping[str, object]],
    horizon_qc_rows: Sequence[Mapping[str, object]],
    detector_metrics: Mapping[str, object],
    smoke_test: bool,
    maximum_candidates_per_house: int | None,
    house_runtime_audit: Mapping[str, Mapping[str, object]],
) -> None:
    "Summarize candidate detection, horizon quality control and label coverage."

    label_counts = Counter(str(row["label_type"]) for row in all_candidate_rows)
    confidence_counts = Counter(
        str(row["label_confidence"]) for row in all_candidate_rows
    )
    horizon_rows: list[str] = []
    for horizon in (20, 40, 60, 120):
        rows = [
            row
            for row in horizon_qc_rows
            if int(row["horizon_seconds"]) == horizon
        ]
        accepted = [row for row in rows if int(row["accepted"]) == 1]
        positive = sum(int(row["label"]) == 1 for row in accepted)
        negative = sum(int(row["label"]) == 0 for row in accepted)
        horizon_rows.append(f"| H{horizon} | {len(accepted)} | {positive} | {negative} | {len(rows) - len(accepted)} |")
    house_rows: list[str] = []
    for house_key, audit in sorted(house_runtime_audit.items()):
        house_rows.append(
            "| {house} | {split} | {candidates} | {core} | {matched} | {flags} |".format(
                house=house_key,
                split=audit["split"],
                candidates=audit["candidate_detector_candidates_before_cap"],
                core=audit["core_positive_ground_truth_onset_count"],
                matched=audit["matched_core_positive_onset_count"],
                flags="; ".join(audit["source_quality_flags"]) or "—",
            )
        )
    report = f"""# {SCHEMA_VERSION} 样本报告

模式：**{'smoke（非正式抽样）' if smoke_test else 'formal（未抽样）'}**。
候选 cap：{maximum_candidates_per_house if maximum_candidates_per_house is not None else '无'}。
报告汇总候选样本、检测器召回和分户质量检查。候选检测阈值固定为300 W，独立于测试住宅的候选数量、召回和类别比例。

## 总表候选检测器

只读取已完成的 aggregate active-W 10 秒 bin：当前 bin 前的最近 6 个可信基线取 median，
当前功率相对基线 `>=300 W` 进入 PENDING；第二个连续有效高 bin 的右边界确认，随后冷却 30 秒。
无效 bin、长 raw gap 或固定网格不连续会重置 PENDING 与基线。

## 候选与真实启动

- aggregate candidates：{len(all_candidate_rows)}
- core positive ground-truth onsets：{detector_metrics['core_positive_ground_truth_onsets']}
- matched core onsets：{detector_metrics['matched_positive_onsets']}
- unmatched core onsets：{detector_metrics['unmatched_positive_onsets']}
- candidate detector recall：{detector_metrics['positive_candidate_recall']}
- accepted positive yield：{detector_metrics['accepted_positive_yield_by_horizon']}
- label types：{dict(sorted(label_counts.items()))}
- label confidence：{dict(sorted(confidence_counts.items()))}

`background_negative` 是低权重弱负类，不等价于“住宅中确认不存在任何未分表电阻设备”；
CSV/NPZ中的`sample_weight(s)`定义监督权重。各horizon独立通过总表质量检查；H120质量统计不决定
H20/H40/H60 是否入集。

| horizon | accepted | label=1 | label=0 | rejected/ignored |
|---|---:|---:|---:|---:|
{chr(10).join(horizon_rows)}

## 分户审计

| house | split | candidates before cap | core GT onsets | matched core | source limitations |
|---|---|---:|---:|---:|---|
{chr(10).join(house_rows)}

`candidate_horizon_qc.csv` 保存每个候选和 horizon 的完整拒绝原因；
`candidate_detector_audit.csv` 保存候选检测器的状态转换与质量重置（逐 action/reason
总计在 manifest）；`matched_positive_onsets.csv` 与 `unmatched_positive_onsets.csv`
保存召回审计；`aggregate_bins.csv` 保存每户完整固定网格输入审计；
`sequence_bins.csv` 保存每个候选的最大 H120 因果序列。
"""
    path.write_text(report, encoding="utf-8")
