from __future__ import annotations

import unittest

import numpy as np

from load_signature_analysis.event_features import (
    FEATURE_COLUMNS,
    MASK_COLUMNS,
    EventWindowConfig,
    assess_window_quality,
    extract_event_window_features,
)


class EventFeatureTests(unittest.TestCase):
    def test_relative_epsilon_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            EventWindowConfig(relative_epsilon_w=0.0)

    def setUp(self) -> None:
        self.config = EventWindowConfig()

    def test_constant_step_has_expected_features(self) -> None:
        pre = np.full(self.config.pre_bin_count, 200.0)
        post = np.full(self.config.post_bin_count, 1_200.0)
        row = extract_event_window_features(pre, post, self.config)

        self.assertAlmostEqual(row["power__delta_mean"], 1_000.0)
        self.assertAlmostEqual(row["power__area_above_baseline"], 120_000.0)
        self.assertAlmostEqual(row["power__relative_delta"], 5.0)
        self.assertAlmostEqual(row["temporal__persistence_fraction"], 1.0)
        self.assertAlmostEqual(row["temporal__time_to_peak"], 0.0)
        self.assertAlmostEqual(row["temporal__above_baseline_longest_run"], 120.0)
        self.assertEqual(row["temporal__positive_step_count"], 1.0)
        self.assertTrue(all(row[column] == 1 for column in MASK_COLUMNS))

    def test_alternating_sequence_covers_temporal_directions(self) -> None:
        pre = np.zeros(self.config.pre_bin_count)
        post = np.tile(np.asarray([100.0, 200.0]), 6)
        row = extract_event_window_features(pre, post, self.config)

        expected_slope = np.polyfit(np.arange(12) * 10.0, post, 1)[0]
        self.assertAlmostEqual(row["power__max_negative_step"], -100.0)
        self.assertEqual(row["temporal__positive_step_count"], 7.0)
        self.assertEqual(row["temporal__negative_step_count"], 5.0)
        self.assertEqual(row["temporal__sign_change_count"], 10.0)
        self.assertAlmostEqual(row["temporal__post_linear_slope"], expected_slope)
        self.assertAlmostEqual(row["temporal__early_late_difference"], 0.0)

    def test_partial_post_window_sets_strict_masks_to_zero(self) -> None:
        pre = np.full(self.config.pre_bin_count, 200.0)
        post = np.full(self.config.post_bin_count, 1_200.0)
        post[5] = np.nan
        quality = assess_window_quality(pre, post, self.config)
        row = extract_event_window_features(pre, post, self.config)

        self.assertTrue(quality["accepted"])
        self.assertEqual(row["mask__power__post_mean"], 1)
        self.assertEqual(row["mask__power__area_above_baseline"], 0)
        self.assertEqual(row["mask__temporal__persistence_fraction"], 0)
        self.assertEqual(row["mask__temporal__positive_step_count"], 0)
        self.assertEqual(row["mask__temporal__early_mean"], 1)
        self.assertEqual(row["mask__temporal__late_mean"], 1)

    def test_two_consecutive_missing_bins_reject_window(self) -> None:
        pre = np.full(self.config.pre_bin_count, 200.0)
        post = np.full(self.config.post_bin_count, 1_200.0)
        post[4:6] = np.nan
        quality = assess_window_quality(pre, post, self.config)

        self.assertFalse(quality["accepted"])
        self.assertIn("consecutive_missing_bins", quality["rejection_reasons"])

    def test_feature_registry_does_not_contain_metadata(self) -> None:
        self.assertEqual(len(FEATURE_COLUMNS), 27)
        self.assertEqual(len(MASK_COLUMNS), 27)
        forbidden = {"dataset", "house_id", "channel_id", "source_appliance", "label"}
        self.assertFalse(forbidden.intersection(FEATURE_COLUMNS))


if __name__ == "__main__":
    unittest.main()
