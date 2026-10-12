"""Online causal candidate detection from household aggregate ten-second bins only.

This module reads no appliance submeters, appliance names or training labels. Completed aggregate bins enter an explicit state machine that produces candidate onsets for later label matching and causal feature construction. Audit all transitions so quality gaps are never mistaken for a real power drop or a new onset."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from statistics import median
from typing import Iterable


@dataclass(frozen=True)
class AggregateBin:
    """A completed aggregate power bin on a fixed UTC grid."""

    bin_left_s: float
    bin_right_s: float
    aggregate_power_w: float
    raw_valid_sample_count: int
    quality_valid: bool
    maximum_raw_gap_seconds: float
    quality_flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate one bin boundary and its counts, allowing NaN power for invalid measurements."""

        if not (
            math.isfinite(float(self.bin_left_s))
            and math.isfinite(float(self.bin_right_s))
        ):
            raise ValueError("bin 边界必须是有限秒数")
        if float(self.bin_right_s) <= float(self.bin_left_s):
            raise ValueError("bin_right_s 必须大于 bin_left_s")
        if int(self.raw_valid_sample_count) < 0:
            raise ValueError("raw_valid_sample_count 不能为负数")
        if not isinstance(self.quality_flags, tuple):
            object.__setattr__(
                self,
                "quality_flags",
                tuple(str(value) for value in self.quality_flags),
            )


@dataclass(frozen=True)
class CandidateDetectionConfig:
    """Frozen aggregate candidate detector parameters."""

    bin_seconds: float = 10.0
    baseline_bin_count: int = 6
    minimum_valid_baseline_bins: int = 5
    positive_step_threshold_w: float = 300.0
    confirmation_bin_count: int = 2
    cooldown_seconds: float = 30.0
    maximum_raw_gap_seconds: float = 30.0

    def __post_init__(self) -> None:
        """Reject configurations that violate the online state-machine semantics."""

        finite_values = (
            self.bin_seconds,
            self.positive_step_threshold_w,
            self.cooldown_seconds,
            self.maximum_raw_gap_seconds,
        )
        if not all(math.isfinite(float(value)) for value in finite_values):
            raise ValueError("候选检测配置不能包含 NaN 或无穷值")
        if self.bin_seconds <= 0.0:
            raise ValueError("bin_seconds 必须大于 0")
        if self.baseline_bin_count <= 0:
            raise ValueError("baseline_bin_count 必须大于 0")
        if not 1 <= self.minimum_valid_baseline_bins <= self.baseline_bin_count:
            raise ValueError("minimum_valid_baseline_bins 必须位于 [1, baseline_bin_count]")
        if self.positive_step_threshold_w < 0.0:
            raise ValueError("positive_step_threshold_w 不能为负数")
        if self.confirmation_bin_count < 2:
            raise ValueError("confirmation_bin_count 至少为 2")
        if self.cooldown_seconds < 0.0:
            raise ValueError("cooldown_seconds 不能为负数")
        if self.maximum_raw_gap_seconds <= 0.0:
            raise ValueError("maximum_raw_gap_seconds 必须大于 0")


@dataclass(frozen=True)
class AggregateCandidate:
    """An aggregate candidate event confirmed by the second consecutive high-power bin."""

    candidate_onset_s: float
    candidate_detected_at_s: float
    detector_internal_delay_s: float
    detector_baseline_w: float
    first_step_w: float
    confirmation_step_w: float


@dataclass(frozen=True)
class CandidateDetectionAudit:
    """One input-bin state-machine audit row for the detector audit table."""

    bin_left_s: float
    bin_right_s: float
    state_before: str
    state_after: str
    action: str
    reason: str
    quality_valid: bool
    baseline_w: float | None
    power_minus_baseline_w: float | None


@dataclass(frozen=True)
class CandidateDetectionResult:
    """Candidate detection results and per-bin audit records."""

    candidates: tuple[AggregateCandidate, ...]
    audit_rows: tuple[CandidateDetectionAudit, ...]
    total_bin_count: int
    action_counts: tuple[tuple[str, int], ...]
    reason_counts: tuple[tuple[str, int], ...]

    @property
    def candidate_count(self) -> int:
        """Return the confirmed candidate count without requiring callers to inspect tuple length."""

        return len(self.candidates)

    def action_count_mapping(self) -> dict[str, int]:
        """Return action counts for all scanned bins even when output retains only transition audits."""

        return dict(self.action_counts)

    def reason_count_mapping(self) -> dict[str, int]:
        """Return reason counts for all scanned bins for compact manifest auditing."""

        return dict(self.reason_counts)


@dataclass(frozen=True)
class _PendingCandidate:
    """Internal PENDING state; freeze the baseline at the first threshold crossing."""

    onset_s: float
    baseline_w: float
    first_step_w: float
    confirmation_count: int


def _is_grid_aligned(value: float, bin_seconds: float) -> bool:
    """Check fixed UTC-grid alignment, allowing only tiny floating-point representation errors."""

    grid_index = float(value) / float(bin_seconds)
    return math.isclose(grid_index, round(grid_index), rel_tol=0.0, abs_tol=1e-8)


def _validate_bin_sequence(
    bins: tuple[AggregateBin, ...], config: CandidateDetectionConfig
) -> None:
    """Verify time order and require every input item to be a fixed half-open ten-second grid bin."""

    previous: AggregateBin | None = None
    tolerance = 1e-8
    for current in bins:
        if not isinstance(current, AggregateBin):
            raise TypeError("bins 必须全部是 AggregateBin")
        duration = float(current.bin_right_s) - float(current.bin_left_s)
        if not math.isclose(
            duration,
            float(config.bin_seconds),
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            raise ValueError("aggregate 候选检测器只接受固定 bin_seconds 网格")
        if not _is_grid_aligned(float(current.bin_left_s), float(config.bin_seconds)):
            raise ValueError("bin_left_s 必须对齐到固定 UTC 网格")
        if previous is not None and float(current.bin_left_s) < float(previous.bin_right_s) - tolerance:
            raise ValueError("aggregate bins 不能倒序或重叠")
        previous = current


def _is_contiguous(previous: AggregateBin | None, current: AggregateBin) -> bool:
    """Check that two supplied bins have no omitted fixed-grid interval between them."""

    if previous is None:
        return True
    return math.isclose(
        float(previous.bin_right_s),
        float(current.bin_left_s),
        rel_tol=0.0,
        abs_tol=1e-8,
    )


def _trusted_bin_reason(
    value: AggregateBin, config: CandidateDetectionConfig
) -> str | None:
    """Return reasons a bin is untrusted; only trusted bins may contribute to baseline or confirmation."""

    if not bool(value.quality_valid):
        return "quality_invalid"
    if int(value.raw_valid_sample_count) <= 0:
        return "no_valid_raw_samples"
    if not math.isfinite(float(value.aggregate_power_w)):
        return "nonfinite_aggregate_power"
    if not math.isfinite(float(value.maximum_raw_gap_seconds)):
        return "unknown_raw_gap"
    if float(value.maximum_raw_gap_seconds) > float(config.maximum_raw_gap_seconds):
        return "raw_gap_exceeds_limit"
    return None


def _append_baseline(
    history: list[float], power_w: float, config: CandidateDetectionConfig
) -> None:
    """Retain the latest trusted baseline values within the configured window length."""

    history.append(float(power_w))
    excess = len(history) - int(config.baseline_bin_count)
    if excess > 0:
        del history[:excess]


def _state_name(
    pending: _PendingCandidate | None,
    cooldown_until_s: float | None,
    timestamp_s: float,
) -> str:
    """Return an auditable state name from internal variables."""

    if pending is not None:
        return "PENDING"
    if cooldown_until_s is not None and float(timestamp_s) < cooldown_until_s:
        return "COOLDOWN"
    return "IDLE"


def detect_aggregate_candidates(
    bins: Iterable[AggregateBin],
    config: CandidateDetectionConfig,
    *,
    audit_mode: str = "all",
) -> CandidateDetectionResult:
    """Scan aggregate bins online and generate candidate onsets confirmed by two consecutive bins.
    
    Each input bin has already reached its right edge, so confirmation occurs at the right edge of the second threshold-crossing bin. Read no future bins or appliance submeter information."""

    if audit_mode not in {"all", "transitions"}:
        raise ValueError("audit_mode 必须是 all 或 transitions")
    ordered_bins = tuple(bins)
    _validate_bin_sequence(ordered_bins, config)

    candidates: list[AggregateCandidate] = []
    audit_rows: list[CandidateDetectionAudit] = []
    action_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    baseline_history: list[float] = []
    pending: _PendingCandidate | None = None
    cooldown_until_s: float | None = None
    previous_bin: AggregateBin | None = None

    for current in ordered_bins:
        current_left = float(current.bin_left_s)
        current_right = float(current.bin_right_s)
        if cooldown_until_s is not None and current_left >= cooldown_until_s:
            # After cooldown, discard the old baseline and accumulate new trusted observations.
            cooldown_until_s = None
            baseline_history.clear()

        state_before = _state_name(pending, cooldown_until_s, current_left)
        contiguous = _is_contiguous(previous_bin, current)
        trusted_reason = _trusted_bin_reason(current, config)
        baseline_w: float | None = None
        residual_w: float | None = None

        if trusted_reason is not None:
            # Never cross invalid, empty or long-gap input; invalidate PENDING immediately as well.
            pending = None
            baseline_history.clear()
            action = "reset"
            reason = trusted_reason
        elif not contiguous:
            # Even if both endpoint bins are valid, omitted grid intervals break continuity and must not be crossed.
            pending = None
            baseline_history.clear()
            if _state_name(pending, cooldown_until_s, current_left) == "COOLDOWN":
                action = "cooldown"
                reason = "bin_discontinuity"
            else:
                _append_baseline(baseline_history, float(current.aggregate_power_w), config)
                action = "baseline_restarted"
                reason = "bin_discontinuity"
        elif _state_name(pending, cooldown_until_s, current_left) == "COOLDOWN":
            action = "cooldown"
            reason = "cooldown_active"
        elif pending is not None:
            baseline_w = float(pending.baseline_w)
            residual_w = float(current.aggregate_power_w) - baseline_w
            if residual_w >= float(config.positive_step_threshold_w):
                confirmation_count = pending.confirmation_count + 1
                if confirmation_count >= int(config.confirmation_bin_count):
                    candidate = AggregateCandidate(
                        candidate_onset_s=float(pending.onset_s),
                        candidate_detected_at_s=current_right,
                        detector_internal_delay_s=current_right - float(pending.onset_s),
                        detector_baseline_w=baseline_w,
                        first_step_w=float(pending.first_step_w),
                        confirmation_step_w=residual_w,
                    )
                    candidates.append(candidate)
                    pending = None
                    baseline_history.clear()
                    cooldown_until_s = current_right + float(config.cooldown_seconds)
                    action = "candidate_confirmed"
                    reason = "confirmation_threshold_met"
                else:
                    pending = _PendingCandidate(
                        onset_s=pending.onset_s,
                        baseline_w=baseline_w,
                        first_step_w=pending.first_step_w,
                        confirmation_count=confirmation_count,
                    )
                    action = "confirmation_observed"
                    reason = "confirmation_threshold_met"
            else:
                # After failed initial confirmation, the load may have changed the baseline; restart from the current low bin.
                pending = None
                baseline_history.clear()
                _append_baseline(baseline_history, float(current.aggregate_power_w), config)
                action = "pending_cancelled"
                reason = "confirmation_below_threshold"
        else:
            if len(baseline_history) >= int(config.minimum_valid_baseline_bins):
                baseline_w = float(median(baseline_history[-int(config.baseline_bin_count) :]))
                residual_w = float(current.aggregate_power_w) - baseline_w
                if residual_w >= float(config.positive_step_threshold_w):
                    pending = _PendingCandidate(
                        onset_s=current_left,
                        baseline_w=baseline_w,
                        first_step_w=residual_w,
                        confirmation_count=1,
                    )
                    action = "pending_started"
                    reason = "positive_step_threshold_met"
                else:
                    _append_baseline(
                        baseline_history, float(current.aggregate_power_w), config
                    )
                    action = "baseline_updated"
                    reason = "below_positive_step_threshold"
            else:
                _append_baseline(baseline_history, float(current.aggregate_power_w), config)
                action = "baseline_warmup"
                reason = "insufficient_baseline_coverage"

        state_after = _state_name(pending, cooldown_until_s, current_left)
        action_counts[action] += 1
        reason_counts[reason] += 1
        should_emit_audit = audit_mode == "all" or action in {
            "reset",
            "baseline_restarted",
            "pending_started",
            "pending_cancelled",
            "confirmation_observed",
            "candidate_confirmed",
        }
        if should_emit_audit:
            audit_rows.append(
                CandidateDetectionAudit(
                    bin_left_s=current_left,
                    bin_right_s=current_right,
                    state_before=state_before,
                    state_after=state_after,
                    action=action,
                    reason=reason,
                    quality_valid=bool(current.quality_valid),
                    baseline_w=baseline_w,
                    power_minus_baseline_w=residual_w,
                )
            )
        previous_bin = current

    return CandidateDetectionResult(
        candidates=tuple(candidates),
        audit_rows=tuple(audit_rows),
        total_bin_count=len(ordered_bins),
        action_counts=tuple(sorted(action_counts.items())),
        reason_counts=tuple(sorted(reason_counts.items())),
    )
