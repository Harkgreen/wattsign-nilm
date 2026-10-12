"""Extract interpretable power and temporal features from fixed aggregate event windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


POWER_FEATURES = (
    "pre_mean",
    "pre_std",
    "pre_min",
    "pre_max",
    "post_mean",
    "post_std",
    "post_min",
    "post_max",
    "delta_mean",
    "delta_median",
    "delta_max",
    "delta_min",
    "max_positive_step",
    "max_negative_step",
    "post_range",
    "area_above_baseline",
    "relative_delta",
)

TEMPORAL_FEATURES = (
    "persistence_fraction",
    "time_to_peak",
    "positive_step_count",
    "negative_step_count",
    "sign_change_count",
    "post_linear_slope",
    "early_mean",
    "late_mean",
    "early_late_difference",
    "above_baseline_longest_run",
)

GROUP_FEATURES: Mapping[str, tuple[str, ...]] = {
    "power": POWER_FEATURES,
    "temporal": TEMPORAL_FEATURES,
}

FEATURE_COLUMNS = tuple(
    f"{group}__{feature}"
    for group, features in GROUP_FEATURES.items()
    for feature in features
)
MASK_COLUMNS = tuple(f"mask__{column}" for column in FEATURE_COLUMNS)


@dataclass(frozen=True)
class EventWindowConfig:
    """Configure event windows, coverage requirements and temporal-change thresholds."""

    resample_interval_seconds: int = 10
    pre_event_seconds: int = 60
    post_event_seconds: int = 120
    minimum_pre_coverage: float = 0.8
    minimum_post_coverage: float = 0.8
    activity_margin_w: float = 50.0
    step_threshold_w: float = 30.0
    relative_epsilon_w: float = 50.0
    maximum_consecutive_missing_bins: int = 1
    maximum_raw_gap_seconds: float = 30.0

    def __post_init__(self) -> None:
        numeric_values = (
            self.resample_interval_seconds,
            self.pre_event_seconds,
            self.post_event_seconds,
            self.minimum_pre_coverage,
            self.minimum_post_coverage,
            self.activity_margin_w,
            self.step_threshold_w,
            self.relative_epsilon_w,
            self.maximum_consecutive_missing_bins,
            self.maximum_raw_gap_seconds,
        )
        if not all(np.isfinite(value) for value in numeric_values):
            raise ValueError("事件窗口配置不能包含 NaN 或无穷值")
        if self.resample_interval_seconds <= 0:
            raise ValueError("重采样间隔必须大于 0")
        if self.pre_event_seconds <= 0 or self.post_event_seconds <= 0:
            raise ValueError("事件前后窗口长度必须大于 0")
        if self.pre_event_seconds % self.resample_interval_seconds != 0:
            raise ValueError("事件前窗口必须是重采样间隔的整数倍")
        if self.post_event_seconds % self.resample_interval_seconds != 0:
            raise ValueError("事件后窗口必须是重采样间隔的整数倍")
        if not 0.0 <= self.minimum_pre_coverage <= 1.0:
            raise ValueError("事件前覆盖率门槛必须位于 [0, 1]")
        if not 0.0 <= self.minimum_post_coverage <= 1.0:
            raise ValueError("事件后覆盖率门槛必须位于 [0, 1]")
        if self.activity_margin_w < 0 or self.step_threshold_w < 0:
            raise ValueError("活动与阶跃阈值不能为负数")
        if self.relative_epsilon_w <= 0:
            raise ValueError("相对变化分母下限必须大于 0")
        if self.maximum_consecutive_missing_bins < 0:
            raise ValueError("最大连续缺失 bin 数不能为负数")
        if self.maximum_raw_gap_seconds <= 0:
            raise ValueError("总表原始采样最大允许缺口必须大于 0")

    @property
    def pre_bin_count(self) -> int:
        return self.pre_event_seconds // self.resample_interval_seconds

    @property
    def post_bin_count(self) -> int:
        return self.post_event_seconds // self.resample_interval_seconds

    @property
    def total_bin_count(self) -> int:
        return self.pre_bin_count + self.post_bin_count


def _finite_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def _safe_stat(values: np.ndarray, operation: str) -> float:
    finite = _finite_values(values)
    if finite.size == 0:
        return np.nan
    if operation == "mean":
        return float(np.mean(finite))
    if operation == "median":
        return float(np.median(finite))
    if operation == "std":
        return float(np.std(finite))
    if operation == "min":
        return float(np.min(finite))
    if operation == "max":
        return float(np.max(finite))
    raise ValueError(f"不支持的统计操作：{operation}")


def _safe_difference(left: float, right: float) -> float:
    if not np.isfinite(left) or not np.isfinite(right):
        return np.nan
    return float(left - right)


def _adjacent_differences(pre: np.ndarray, post: np.ndarray) -> np.ndarray:
    """Compare only the final baseline bin with post-event bins; exclude internal baseline fluctuations from the event step."""

    combined = np.concatenate((pre[-1:], post)).astype(np.float64, copy=False)
    if combined.size < 2:
        return np.empty(0, dtype=np.float64)
    valid_pairs = np.isfinite(combined[:-1]) & np.isfinite(combined[1:])
    return np.diff(combined)[valid_pairs]


def _maximum_missing_run(values: np.ndarray) -> int:
    maximum = 0
    current = 0
    for missing in ~np.isfinite(values):
        if missing:
            current += 1
            maximum = max(maximum, current)
        else:
            current = 0
    return maximum


def _longest_true_run(values: np.ndarray) -> int:
    maximum = 0
    current = 0
    for value in values:
        if bool(value):
            current += 1
            maximum = max(maximum, current)
        else:
            current = 0
    return maximum


def _linear_slope(values: np.ndarray, interval_seconds: float) -> float:
    valid = np.isfinite(values)
    if np.count_nonzero(valid) < 2:
        return np.nan
    times = np.arange(values.size, dtype=np.float64) * interval_seconds
    centered_times = times[valid] - float(np.mean(times[valid]))
    centered_values = values[valid] - float(np.mean(values[valid]))
    denominator = float(np.dot(centered_times, centered_times))
    if denominator <= 0.0:
        return np.nan
    return float(np.dot(centered_times, centered_values) / denominator)


def assess_window_quality(
    pre: np.ndarray,
    post: np.ndarray,
    config: EventWindowConfig,
) -> dict[str, object]:
    """Check window coverage and consecutive missingness, returning explicit acceptance or rejection reasons."""

    pre = np.asarray(pre, dtype=np.float64)
    post = np.asarray(post, dtype=np.float64)
    if pre.size != config.pre_bin_count or post.size != config.post_bin_count:
        raise ValueError("窗口 bin 数与配置不一致")

    pre_coverage = float(np.mean(np.isfinite(pre)))
    post_coverage = float(np.mean(np.isfinite(post)))
    maximum_missing_run = _maximum_missing_run(np.concatenate((pre, post)))
    reasons: list[str] = []
    if pre_coverage < config.minimum_pre_coverage:
        reasons.append("insufficient_pre_coverage")
    if post_coverage < config.minimum_post_coverage:
        reasons.append("insufficient_post_coverage")
    if maximum_missing_run > config.maximum_consecutive_missing_bins:
        reasons.append("consecutive_missing_bins")
    return {
        "accepted": not reasons,
        "rejection_reasons": reasons,
        "pre_coverage": pre_coverage,
        "post_coverage": post_coverage,
        "maximum_consecutive_missing_bins": maximum_missing_run,
    }


def extract_event_window_features(
    pre: np.ndarray,
    post: np.ndarray,
    config: EventWindowConfig,
) -> dict[str, float | int]:
    """Convert an aggregate event window to 27 features and per-feature missingness masks."""

    pre = np.asarray(pre, dtype=np.float64)
    post = np.asarray(post, dtype=np.float64)
    if pre.size != config.pre_bin_count or post.size != config.post_bin_count:
        raise ValueError("窗口 bin 数与配置不一致")

    pre_mean = _safe_stat(pre, "mean")
    pre_median = _safe_stat(pre, "median")
    pre_std = _safe_stat(pre, "std")
    pre_min = _safe_stat(pre, "min")
    pre_max = _safe_stat(pre, "max")
    post_mean = _safe_stat(post, "mean")
    post_median = _safe_stat(post, "median")
    post_std = _safe_stat(post, "std")
    post_min = _safe_stat(post, "min")
    post_max = _safe_stat(post, "max")

    differences = _adjacent_differences(pre, post)
    positive_steps = differences[differences > 0]
    negative_steps = differences[differences < 0]
    max_positive_step = float(np.max(positive_steps)) if positive_steps.size else 0.0
    max_negative_step = float(np.min(negative_steps)) if negative_steps.size else 0.0

    complete_post_and_boundary = bool(
        pre.size > 0
        and np.isfinite(pre[-1])
        and np.all(np.isfinite(post))
        and np.isfinite(pre_mean)
    )
    if complete_post_and_boundary:
        area_above_baseline = float(
            np.sum(np.maximum(post - pre_mean, 0.0))
            * config.resample_interval_seconds
        )
    else:
        area_above_baseline = np.nan

    delta_mean = _safe_difference(post_mean, pre_mean)
    denominator = max(abs(pre_mean), config.relative_epsilon_w) if np.isfinite(pre_mean) else np.nan
    relative_delta = (
        float(delta_mean / denominator)
        if np.isfinite(delta_mean) and np.isfinite(denominator)
        else np.nan
    )

    power_features = {
        "pre_mean": pre_mean,
        "pre_std": pre_std,
        "pre_min": pre_min,
        "pre_max": pre_max,
        "post_mean": post_mean,
        "post_std": post_std,
        "post_min": post_min,
        "post_max": post_max,
        "delta_mean": delta_mean,
        "delta_median": _safe_difference(post_median, pre_median),
        "delta_max": _safe_difference(post_max, pre_mean),
        "delta_min": _safe_difference(post_min, pre_mean),
        "max_positive_step": max_positive_step,
        "max_negative_step": max_negative_step,
        "post_range": _safe_difference(post_max, post_min),
        "area_above_baseline": area_above_baseline,
        "relative_delta": relative_delta,
    }

    above_baseline = np.zeros(post.size, dtype=bool)
    if np.isfinite(pre_mean):
        above_baseline = np.isfinite(post) & (post > pre_mean + config.activity_margin_w)
    persistence_fraction = (
        float(np.count_nonzero(above_baseline) / post.size)
        if complete_post_and_boundary
        else np.nan
    )

    if complete_post_and_boundary:
        peak_index = int(np.nanargmax(post))
        time_to_peak = float(peak_index * config.resample_interval_seconds)
    else:
        time_to_peak = np.nan

    significant_signs = np.sign(differences[np.abs(differences) > config.step_threshold_w])
    if complete_post_and_boundary:
        sign_change_count = (
            int(np.count_nonzero(significant_signs[1:] != significant_signs[:-1]))
            if significant_signs.size >= 2
            else 0
        )
        positive_step_count = int(np.count_nonzero(differences > config.step_threshold_w))
        negative_step_count = int(np.count_nonzero(differences < -config.step_threshold_w))
    else:
        sign_change_count = np.nan
        positive_step_count = np.nan
        negative_step_count = np.nan
    early_slice = post[:4]
    late_slice = post[-4:]
    early_mean = _safe_stat(early_slice, "mean") if np.all(np.isfinite(early_slice)) else np.nan
    late_mean = _safe_stat(late_slice, "mean") if np.all(np.isfinite(late_slice)) else np.nan
    temporal_features = {
        "persistence_fraction": persistence_fraction,
        "time_to_peak": time_to_peak,
        "positive_step_count": positive_step_count,
        "negative_step_count": negative_step_count,
        "sign_change_count": sign_change_count,
        "post_linear_slope": _linear_slope(post, config.resample_interval_seconds),
        "early_mean": early_mean,
        "late_mean": late_mean,
        "early_late_difference": _safe_difference(early_mean, late_mean),
        "above_baseline_longest_run": (
            float(_longest_true_run(above_baseline) * config.resample_interval_seconds)
            if complete_post_and_boundary
            else np.nan
        ),
    }

    row: dict[str, float | int] = {}
    for group, values in (("power", power_features), ("temporal", temporal_features)):
        for feature_name in GROUP_FEATURES[group]:
            column = f"{group}__{feature_name}"
            value = float(values[feature_name])
            row[column] = value
            row[f"mask__{column}"] = int(np.isfinite(value))
    return row
