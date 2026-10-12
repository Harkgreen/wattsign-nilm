"""Synthetic regression tests for the online aggregate candidate state machine."""

from __future__ import annotations

import math
import unittest

from load_signature_analysis.online_candidate_detection import (
    AggregateBin,
    CandidateDetectionConfig,
    detect_aggregate_candidates,
)


class AggregateCandidateDetectionTests(unittest.TestCase):
    """Fix confirmation, gap and cooldown state transitions of the candidate detector."""

    def setUp(self) -> None:
        self.config = CandidateDetectionConfig()

    @staticmethod
    def _bin(
        index: int,
        power_w: float,
        *,
        quality_valid: bool = True,
        raw_gap_seconds: float = 1.0,
    ) -> AggregateBin:
        """Construct a 10-second half-open grid bin; invalid bins are not valid measurements."""

        return AggregateBin(
            bin_left_s=float(index * 10),
            bin_right_s=float((index + 1) * 10),
            aggregate_power_w=float(power_w),
            raw_valid_sample_count=1 if quality_valid else 0,
            quality_valid=quality_valid,
            maximum_raw_gap_seconds=float(raw_gap_seconds),
            quality_flags=() if quality_valid else ("synthetic_invalid",),
        )

    def _baseline_bins(self, start_index: int = 0) -> list[AggregateBin]:
        """Create a stable 100 W baseline with the required minimum coverage."""

        return [self._bin(start_index + offset, 100.0) for offset in range(6)]

    def test_two_consecutive_high_bins_confirm_one_candidate(self) -> None:
        """Two 450 W bins after a 100 W baseline confirm at the second right boundary."""

        result = detect_aggregate_candidates(
            [*self._baseline_bins(), self._bin(6, 450.0), self._bin(7, 450.0)],
            self.config,
        )

        self.assertEqual(result.candidate_count, 1)
        candidate = result.candidates[0]
        self.assertEqual(candidate.candidate_onset_s, 60.0)
        self.assertEqual(candidate.candidate_detected_at_s, 80.0)
        self.assertEqual(candidate.detector_internal_delay_s, 20.0)
        self.assertEqual(candidate.detector_baseline_w, 100.0)
        self.assertEqual(candidate.first_step_w, 350.0)
        self.assertEqual(candidate.confirmation_step_w, 350.0)

    def test_single_high_spike_does_not_confirm_candidate(self) -> None:
        """A single 450 W spike followed by a decrease must cancel PENDING."""

        result = detect_aggregate_candidates(
            [
                *self._baseline_bins(),
                self._bin(6, 450.0),
                self._bin(7, 100.0),
            ],
            self.config,
        )

        self.assertEqual(result.candidate_count, 0)
        self.assertEqual(result.audit_rows[-1].action, "pending_cancelled")
        self.assertEqual(result.audit_rows[-1].reason, "confirmation_below_threshold")

    def test_high_power_at_file_start_never_has_trusted_baseline(self) -> None:
        """The first high-power bin lacks an earlier baseline and cannot create a candidate directly."""

        result = detect_aggregate_candidates(
            [self._bin(0, 450.0), self._bin(1, 450.0)], self.config
        )

        self.assertEqual(result.candidate_count, 0)
        self.assertTrue(
            all(row.action == "baseline_warmup" for row in result.audit_rows)
        )

    def test_long_gap_then_first_high_bin_does_not_confirm(self) -> None:
        """The first high bin after a long gap must rebuild its baseline before entering PENDING."""

        result = detect_aggregate_candidates(
            [
                *self._baseline_bins(),
                self._bin(6, 450.0, raw_gap_seconds=31.0),
                self._bin(7, 450.0),
            ],
            self.config,
        )

        self.assertEqual(result.candidate_count, 0)
        self.assertEqual(result.audit_rows[6].action, "reset")
        self.assertEqual(result.audit_rows[6].reason, "raw_gap_exceeds_limit")
        self.assertEqual(result.audit_rows[7].action, "baseline_warmup")

    def test_invalid_bin_during_pending_resets_confirmation(self) -> None:
        """An invalid bin cancels PENDING; later high bins cannot confirm across the gap."""

        result = detect_aggregate_candidates(
            [
                *self._baseline_bins(),
                self._bin(6, 450.0),
                self._bin(7, math.nan, quality_valid=False),
                self._bin(8, 450.0),
            ],
            self.config,
        )

        self.assertEqual(result.candidate_count, 0)
        self.assertEqual(result.audit_rows[7].action, "reset")
        self.assertEqual(result.audit_rows[8].action, "baseline_warmup")

    def test_cooldown_prevents_duplicate_candidate_for_sustained_high_power(self) -> None:
        """The 30-second cooldown must prevent repeated candidates for the same sustained load."""

        bins = [*self._baseline_bins(), *[self._bin(index, 450.0) for index in range(6, 12)]]
        result = detect_aggregate_candidates(bins, self.config)

        self.assertEqual(result.candidate_count, 1)
        self.assertEqual(result.candidates[0].candidate_detected_at_s, 80.0)
        cooldown_rows = [
            row
            for row in result.audit_rows
            if row.bin_left_s in {80.0, 90.0, 100.0}
        ]
        self.assertEqual([row.action for row in cooldown_rows], ["cooldown"] * 3)

    def test_exactly_three_hundred_w_step_is_inclusive(self) -> None:
        """An increment of exactly 300 W satisfies the >= 300 W boundary and can confirm on the next bin."""

        result = detect_aggregate_candidates(
            [*self._baseline_bins(), self._bin(6, 400.0), self._bin(7, 400.0)],
            self.config,
        )

        self.assertEqual(result.candidate_count, 1)
        self.assertEqual(result.candidates[0].first_step_w, 300.0)
        self.assertEqual(result.candidates[0].confirmation_step_w, 300.0)

    def test_transition_audit_keeps_global_bin_counts_without_baseline_spam(self) -> None:
        """The compact audit retains transitions while counting every scanned bin."""

        bins = [*self._baseline_bins(), self._bin(6, 450.0), self._bin(7, 450.0)]
        result = detect_aggregate_candidates(bins, self.config, audit_mode="transitions")

        self.assertEqual(result.total_bin_count, len(bins))
        self.assertEqual(result.action_count_mapping()["candidate_confirmed"], 1)
        self.assertLess(len(result.audit_rows), result.total_bin_count)


if __name__ == "__main__":
    unittest.main()
