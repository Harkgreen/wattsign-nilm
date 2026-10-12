"""The ``online_feature_v1_1`` feature and sequence contract for online causal candidates.

Process only arrived fixed UTC ten-second bins, reading no raw files, submeter labels or future time steps. Do not reuse the offline 27-feature registry: online post-event statistics explicitly describe only observations available within the current horizon for H20/H40/H60/H120 models."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np


SUPPORTED_HORIZONS_SECONDS = (20, 40, 60, 120)
FEATURE_REGISTRY_VERSION = "online_feature_v1_1"


# Field order is the persisted online_feature_v1_1 contract; do not derive it from the old offline registry.
ONLINE_POWER_FEATURES = (
    "online_power__pre_mean",
    "online_power__pre_std",
    "online_power__pre_min",
    "online_power__pre_max",
    "online_power__observed_post_mean",
    "online_power__observed_post_std",
    "online_power__observed_post_min",
    "online_power__observed_post_max",
    "online_power__delta_mean",
    "online_power__delta_median",
    "online_power__delta_max",
    "online_power__delta_min",
    "online_power__max_positive_step",
    "online_power__max_negative_step",
    "online_power__observed_post_range",
    "online_power__mean_area_above_baseline",
    "online_power__relative_delta",
)

ONLINE_TEMPORAL_FEATURES = (
    "online_temporal__observed_persistence_fraction",
    "online_temporal__observed_time_to_peak",
    "online_temporal__positive_step_count",
    "online_temporal__negative_step_count",
    "online_temporal__sign_change_count",
    "online_temporal__observed_post_linear_slope",
    "online_temporal__initial_20s_mean",
    "online_temporal__recent_20s_mean",
    "online_temporal__initial_recent_difference",
    "online_temporal__longest_active_fraction",
)

ONLINE_GROUP_FEATURES: Mapping[str, tuple[str, ...]] = {
    "power": ONLINE_POWER_FEATURES,
    "temporal": ONLINE_TEMPORAL_FEATURES,
}
ONLINE_FEATURE_COLUMNS = (*ONLINE_POWER_FEATURES, *ONLINE_TEMPORAL_FEATURES)
ONLINE_MASK_COLUMNS = tuple(f"mask__{name}" for name in ONLINE_FEATURE_COLUMNS)

# This order is written to NPZ ``sequence_feature_names``; it is not a subset of online scalar features.
SEQUENCE_FEATURE_NAMES = (
    "aggregate_power_w",
    "power_minus_pre_mean_w",
    "adjacent_power_difference_w",
    "pre_post_indicator",
)


@dataclass(frozen=True)
class OnlineFeatureConfig:
    """Freeze the online_feature_v1_1 grid and feature-computation thresholds."""

    bin_seconds: int = 10
    pre_bin_count: int = 6
    activity_margin_w: float = 50.0
    step_threshold_w: float = 30.0
    relative_epsilon_w: float = 50.0

    def __post_init__(self) -> None:
        if self.bin_seconds != 10:
            raise ValueError("online_feature_v1_1 固定使用 10 秒 bin")
        if self.pre_bin_count != 6:
            raise ValueError("online_feature_v1_1 固定使用 6 个 pre bin")
        numeric_values = (
            self.activity_margin_w,
            self.step_threshold_w,
            self.relative_epsilon_w,
        )
        if not all(math.isfinite(float(value)) for value in numeric_values):
            raise ValueError("在线特征阈值不能包含 NaN 或无穷值")
        if self.activity_margin_w < 0.0 or self.step_threshold_w < 0.0:
            raise ValueError("活动和阶跃阈值不能为负数")
        if self.relative_epsilon_w <= 0.0:
            raise ValueError("相对增量分母下限必须大于 0")


@dataclass(frozen=True)
class OnlineHorizonQuality:
    """Basic coverage audit result for one candidate and horizon."""

    accepted: bool
    pre_valid_bin_count: int
    post_valid_bin_count: int
    pre_coverage: float
    post_coverage: float
    required_post_valid_bin_count: int
    maximum_consecutive_missing_bins: int
    rejection_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Convert to a stable scalar dictionary suitable for audit CSV or JSON."""

        return {
            "accepted": self.accepted,
            "pre_valid_bin_count": self.pre_valid_bin_count,
            "post_valid_bin_count": self.post_valid_bin_count,
            "pre_coverage": self.pre_coverage,
            "post_coverage": self.post_coverage,
            "required_post_valid_bin_count": self.required_post_valid_bin_count,
            "maximum_consecutive_missing_bins": self.maximum_consecutive_missing_bins,
            "rejection_reasons": list(self.rejection_reasons),
        }


def post_bin_count_for_horizon(
    horizon_seconds: int,
    config: OnlineFeatureConfig = OnlineFeatureConfig(),
) -> int:
    """Return the number of arrived post-event bins in a frozen horizon, rejecting undefined horizons."""

    try:
        horizon = int(horizon_seconds)
    except (TypeError, ValueError) as error:
        raise ValueError("horizon_seconds 必须是整数秒") from error
    if horizon != horizon_seconds:
        raise ValueError("horizon_seconds 必须是整数秒")
    if horizon not in SUPPORTED_HORIZONS_SECONDS:
        raise ValueError(
            "online_feature_v1_1 只支持 horizon："
            f"{list(SUPPORTED_HORIZONS_SECONDS)}"
        )
    if horizon % config.bin_seconds != 0:
        raise ValueError("horizon_seconds 必须是 bin_seconds 的整数倍")
    return horizon // config.bin_seconds


def expected_sequence_length_for_horizon(
    horizon_seconds: int,
    config: OnlineFeatureConfig = OnlineFeatureConfig(),
) -> int:
    """Return the actual sequence length for a horizon without padding for future H120 data."""

    return config.pre_bin_count + post_bin_count_for_horizon(horizon_seconds, config)


def relative_bin_indices_for_horizon(
    horizon_seconds: int,
    config: OnlineFeatureConfig = OnlineFeatureConfig(),
) -> np.ndarray:
    """Return the fixed relative bin indices ``[-6, ..., post_bin_count-1]``."""

    post_bin_count = post_bin_count_for_horizon(horizon_seconds, config)
    return np.arange(-config.pre_bin_count, post_bin_count, dtype=np.int64)


def _coerce_quality_mask(mask: np.ndarray, expected_size: int, name: str) -> np.ndarray:
    """Convert to a strict boolean quality mask; arbitrary nonzero values are not automatically valid."""

    raw = np.asarray(mask)
    if raw.ndim != 1 or raw.size != expected_size:
        raise ValueError(f"{name} 的长度必须为 {expected_size}")
    if raw.dtype == np.bool_:
        return raw.astype(bool, copy=True)
    try:
        numeric = raw.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} 只能包含 0 或 1") from error
    if not np.all(np.isfinite(numeric)) or not np.all((numeric == 0.0) | (numeric == 1.0)):
        raise ValueError(f"{name} 只能包含 0 或 1")
    return numeric.astype(bool)


def _effective_power(
    power_w: np.ndarray,
    quality_valid_mask: np.ndarray,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Represent invalid-quality or nonfinite power as NaN and return a validity mask."""

    power = np.asarray(power_w, dtype=np.float64)
    if power.ndim != 1:
        raise ValueError(f"{name} 必须是一维数组")
    quality = _coerce_quality_mask(quality_valid_mask, power.size, f"{name}_quality_valid_mask")
    valid = quality & np.isfinite(power)
    return np.where(valid, power, np.nan), valid


def _safe_stat(values: np.ndarray, operation: str) -> float:
    """Compute basic statistics on valid values, returning NaN when none are valid."""

    observed = values[np.isfinite(values)]
    if observed.size == 0:
        return float("nan")
    if operation == "mean":
        return float(np.mean(observed))
    if operation == "median":
        return float(np.median(observed))
    if operation == "std":
        return float(np.std(observed, ddof=0))
    if operation == "min":
        return float(np.min(observed))
    if operation == "max":
        return float(np.max(observed))
    raise ValueError(f"未知统计操作：{operation}")


def _safe_difference(left: float, right: float) -> float:
    """Subtract only when both operands are finite, never disguising missing values as zero."""

    return float(left - right) if np.isfinite(left) and np.isfinite(right) else float("nan")


def _adjacent_differences(pre: np.ndarray, post: np.ndarray) -> np.ndarray:
    """Compute valid adjacent differences from the final pre-event bin through all post-event bins."""

    if pre.size == 0:
        return np.full(post.size, np.nan, dtype=np.float64)
    joined = np.concatenate((pre[-1:], post))
    differences = np.diff(joined)
    valid_pairs = np.isfinite(joined[:-1]) & np.isfinite(joined[1:])
    return np.where(valid_pairs, differences, np.nan)


def _sign_change_count_within_observed_runs(
    differences: np.ndarray,
    *,
    threshold_w: float,
) -> float:
    """Count sign changes of significant steps only within continuously observed difference runs.
    
    A missing difference breaks a run. A finite difference below the threshold contributes no sign and does not break the run. This preserves the definition of counting significant steps without joining a false sign change across a gap."""

    total = 0
    current_signs: list[float] = []

    def flush() -> None:
        nonlocal total
        if len(current_signs) > 1:
            signs = np.asarray(current_signs, dtype=np.float64)
            total += int(np.count_nonzero(signs[1:] != signs[:-1]))
        current_signs.clear()

    for value in np.asarray(differences, dtype=np.float64):
        if not np.isfinite(value):
            flush()
            continue
        if abs(float(value)) > float(threshold_w):
            current_signs.append(float(np.sign(value)))
    flush()
    return float(total)


def _maximum_missing_run(valid_mask: np.ndarray) -> int:
    """Compute the longest consecutive run of invalid-quality bins."""

    maximum = 0
    current = 0
    for valid in np.asarray(valid_mask, dtype=bool):
        if valid:
            current = 0
        else:
            current += 1
            maximum = max(maximum, current)
    return maximum


def _linear_slope(post: np.ndarray, bin_seconds: int) -> float:
    """Fit a causal linear slope using only valid post-event points in the current horizon, in W/s."""

    valid = np.isfinite(post)
    if int(np.count_nonzero(valid)) < 2:
        return float("nan")
    times = np.flatnonzero(valid).astype(np.float64) * float(bin_seconds)
    values = post[valid]
    centered_times = times - float(np.mean(times))
    denominator = float(np.dot(centered_times, centered_times))
    if denominator <= 0.0:
        return float("nan")
    centered_values = values - float(np.mean(values))
    return float(np.dot(centered_times, centered_values) / denominator)


def _longest_true_run(values: np.ndarray) -> int:
    """Return the longest consecutive True run in a boolean array."""

    maximum = 0
    current = 0
    for value in np.asarray(values, dtype=bool):
        if value:
            current += 1
            maximum = max(maximum, current)
        else:
            current = 0
    return maximum


def assess_online_horizon_quality(
    pre_power_w: np.ndarray,
    post_power_w: np.ndarray,
    pre_quality_valid_mask: np.ndarray,
    post_quality_valid_mask: np.ndarray,
    *,
    config: OnlineFeatureConfig = OnlineFeatureConfig(),
    minimum_pre_coverage: float = 0.8,
    minimum_post_coverage: float = 0.8,
    maximum_consecutive_missing_bins: int = 1,
) -> OnlineHorizonQuality:
    """Check coverage and consecutive missingness in a truncated horizon, excluding file boundaries and raw gaps."""

    if not 0.0 <= float(minimum_pre_coverage) <= 1.0:
        raise ValueError("minimum_pre_coverage 必须位于 [0, 1]")
    if not 0.0 <= float(minimum_post_coverage) <= 1.0:
        raise ValueError("minimum_post_coverage 必须位于 [0, 1]")
    if int(maximum_consecutive_missing_bins) != maximum_consecutive_missing_bins:
        raise ValueError("maximum_consecutive_missing_bins 必须是整数")
    if maximum_consecutive_missing_bins < 0:
        raise ValueError("maximum_consecutive_missing_bins 不能为负数")

    pre, pre_valid = _effective_power(pre_power_w, pre_quality_valid_mask, "pre_power_w")
    post, post_valid = _effective_power(post_power_w, post_quality_valid_mask, "post_power_w")
    if pre.size != config.pre_bin_count:
        raise ValueError(f"pre_power_w 必须包含 {config.pre_bin_count} 个 bin")
    if post.size not in {
        post_bin_count_for_horizon(horizon, config)
        for horizon in SUPPORTED_HORIZONS_SECONDS
    }:
        raise ValueError("post_power_w 的长度必须对应 H20/H40/H60/H120")

    pre_valid_count = int(np.count_nonzero(pre_valid))
    post_valid_count = int(np.count_nonzero(post_valid))
    pre_coverage = float(pre_valid_count / pre.size)
    post_coverage = float(post_valid_count / post.size)
    required_post = int(math.ceil(float(minimum_post_coverage) * post.size))
    maximum_missing = _maximum_missing_run(np.concatenate((pre_valid, post_valid)))
    reasons: list[str] = []
    if pre_coverage < float(minimum_pre_coverage):
        reasons.append("insufficient_pre_coverage")
    if post_valid_count < required_post:
        reasons.append("insufficient_post_coverage")
    if maximum_missing > int(maximum_consecutive_missing_bins):
        reasons.append("consecutive_missing_bins")
    return OnlineHorizonQuality(
        accepted=not reasons,
        pre_valid_bin_count=pre_valid_count,
        post_valid_bin_count=post_valid_count,
        pre_coverage=pre_coverage,
        post_coverage=post_coverage,
        required_post_valid_bin_count=required_post,
        maximum_consecutive_missing_bins=maximum_missing,
        rejection_reasons=tuple(reasons),
    )


def build_causal_sequence_inputs(
    power_w: np.ndarray,
    quality_valid_mask: np.ndarray,
    *,
    pre_bin_count: int = 6,
) -> np.ndarray:
    """Build the four-channel input for the current horizon without adding future time steps.
    
    ``power_w`` must already be truncated to ``6 pre + current post`` bins. Invalid-quality measurements remain NaN in the first three measurement channels; the fourth pre/post indicator is always valid."""

    if pre_bin_count != 6:
        raise ValueError("online_feature_v1_1 固定使用 6 个 pre bin")
    power, valid = _effective_power(power_w, quality_valid_mask, "power_w")
    if power.size <= pre_bin_count:
        raise ValueError("序列必须至少包含一个已到达的 post bin")
    pre_mean = _safe_stat(power[:pre_bin_count], "mean")

    values = np.full((power.size, len(SEQUENCE_FEATURE_NAMES)), np.nan, dtype=np.float64)
    values[:, 0] = power
    if np.isfinite(pre_mean):
        values[:, 1] = np.where(valid, power - pre_mean, np.nan)
    adjacent_valid = valid[:-1] & valid[1:]
    values[1:, 2] = np.where(
        adjacent_valid,
        power[1:] - power[:-1],
        np.nan,
    )
    values[:pre_bin_count, 3] = 0.0
    values[pre_bin_count:, 3] = 1.0
    return values


def extract_online_features(
    pre_power_w: np.ndarray,
    post_power_w: np.ndarray,
    pre_quality_valid_mask: np.ndarray,
    post_quality_valid_mask: np.ndarray,
    *,
    horizon_seconds: int,
    config: OnlineFeatureConfig = OnlineFeatureConfig(),
) -> dict[str, float | int]:
    """Extract 27 online causal features and their masks for one candidate and horizon.
    
    ``post_power_w`` may contain only bins already arrived within the horizon. For area, persistence, peak and longest activity features that require the complete post-event trajectory, any invalid-quality post bin makes the feature NaN. This denotes unknown measurement quality, not missing or zero future data."""

    expected_post_count = post_bin_count_for_horizon(horizon_seconds, config)
    pre, _ = _effective_power(pre_power_w, pre_quality_valid_mask, "pre_power_w")
    post, _ = _effective_power(post_power_w, post_quality_valid_mask, "post_power_w")
    if pre.size != config.pre_bin_count:
        raise ValueError(f"pre_power_w 必须包含 {config.pre_bin_count} 个 bin")
    if post.size != expected_post_count:
        raise ValueError(
            f"H{horizon_seconds} 必须包含 {expected_post_count} 个 post bin"
        )

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
    finite_differences = differences[np.isfinite(differences)]
    if finite_differences.size:
        positive_steps = finite_differences[finite_differences > 0.0]
        negative_steps = finite_differences[finite_differences < 0.0]
        max_positive_step = (
            float(np.max(positive_steps)) if positive_steps.size else 0.0
        )
        max_negative_step = (
            float(np.min(negative_steps)) if negative_steps.size else 0.0
        )
        significant_signs = np.sign(
            finite_differences[
                np.abs(finite_differences) > config.step_threshold_w
            ]
        )
        positive_step_count = float(np.count_nonzero(significant_signs > 0.0))
        negative_step_count = float(np.count_nonzero(significant_signs < 0.0))
        sign_change_count = _sign_change_count_within_observed_runs(
            differences,
            threshold_w=config.step_threshold_w,
        )
    else:
        max_positive_step = float("nan")
        max_negative_step = float("nan")
        positive_step_count = float("nan")
        negative_step_count = float("nan")
        sign_change_count = float("nan")

    complete_post = bool(np.all(np.isfinite(post)))
    has_baseline = bool(np.isfinite(pre_mean))
    if complete_post and has_baseline:
        above_baseline = post > pre_mean + config.activity_margin_w
        persistence_fraction = float(np.mean(above_baseline))
        time_to_peak = float(int(np.argmax(post)) * config.bin_seconds)
        mean_area_above_baseline = float(
            np.sum(np.maximum(post - pre_mean, 0.0))
            * config.bin_seconds
            / float(horizon_seconds)
        )
        longest_active_fraction = float(
            _longest_true_run(above_baseline) / post.size
        )
    else:
        persistence_fraction = float("nan")
        time_to_peak = float("nan")
        mean_area_above_baseline = float("nan")
        longest_active_fraction = float("nan")

    initial_20s_mean = (
        float(np.mean(post[:2])) if np.all(np.isfinite(post[:2])) else float("nan")
    )
    recent_20s_mean = (
        float(np.mean(post[-2:])) if np.all(np.isfinite(post[-2:])) else float("nan")
    )

    feature_values = {
        "online_power__pre_mean": pre_mean,
        "online_power__pre_std": pre_std,
        "online_power__pre_min": pre_min,
        "online_power__pre_max": pre_max,
        "online_power__observed_post_mean": post_mean,
        "online_power__observed_post_std": post_std,
        "online_power__observed_post_min": post_min,
        "online_power__observed_post_max": post_max,
        "online_power__delta_mean": _safe_difference(post_mean, pre_mean),
        "online_power__delta_median": _safe_difference(post_median, pre_median),
        "online_power__delta_max": _safe_difference(post_max, pre_mean),
        "online_power__delta_min": _safe_difference(post_min, pre_mean),
        "online_power__max_positive_step": max_positive_step,
        "online_power__max_negative_step": max_negative_step,
        "online_power__observed_post_range": _safe_difference(post_max, post_min),
        "online_power__mean_area_above_baseline": mean_area_above_baseline,
        "online_power__relative_delta": (
            float(
                _safe_difference(post_mean, pre_mean)
                / max(abs(pre_mean), config.relative_epsilon_w)
            )
            if np.isfinite(post_mean) and np.isfinite(pre_mean)
            else float("nan")
        ),
        "online_temporal__observed_persistence_fraction": persistence_fraction,
        "online_temporal__observed_time_to_peak": time_to_peak,
        "online_temporal__positive_step_count": positive_step_count,
        "online_temporal__negative_step_count": negative_step_count,
        "online_temporal__sign_change_count": sign_change_count,
        "online_temporal__observed_post_linear_slope": _linear_slope(
            post, config.bin_seconds
        ),
        "online_temporal__initial_20s_mean": initial_20s_mean,
        "online_temporal__recent_20s_mean": recent_20s_mean,
        "online_temporal__initial_recent_difference": _safe_difference(
            initial_20s_mean, recent_20s_mean
        ),
        "online_temporal__longest_active_fraction": longest_active_fraction,
    }
    if tuple(feature_values) != ONLINE_FEATURE_COLUMNS:
        raise RuntimeError("online_feature_v1_1 内部字段顺序与 registry 不一致")

    result: dict[str, float | int] = {}
    for feature_name in ONLINE_FEATURE_COLUMNS:
        value = float(feature_values[feature_name])
        result[feature_name] = value
        result[f"mask__{feature_name}"] = int(np.isfinite(value))
    _validate_feature_masks(result)
    return result


def _validate_feature_masks(row: Mapping[str, float | int]) -> None:
    """Verify exact correspondence between each online feature mask and finiteness."""

    for feature_name, mask_name in zip(
        ONLINE_FEATURE_COLUMNS, ONLINE_MASK_COLUMNS, strict=True
    ):
        if feature_name not in row or mask_name not in row:
            raise ValueError(f"在线特征行缺少 {feature_name} 或 {mask_name}")
        expected = int(np.isfinite(float(row[feature_name])))
        if int(row[mask_name]) != expected:
            raise ValueError(f"{mask_name} 与 {feature_name} 的有限性不一致")
