from __future__ import annotations

import unittest

import numpy as np

from load_signature_analysis.online_features import (
    ONLINE_FEATURE_COLUMNS,
    ONLINE_MASK_COLUMNS,
    ONLINE_POWER_FEATURES,
    ONLINE_TEMPORAL_FEATURES,
    SEQUENCE_FEATURE_NAMES,
    SUPPORTED_HORIZONS_SECONDS,
    assess_online_horizon_quality,
    build_causal_sequence_inputs,
    expected_sequence_length_for_horizon,
    extract_online_features,
    post_bin_count_for_horizon,
    relative_bin_indices_for_horizon,
)


def _all_valid(length: int) -> np.ndarray:
    """Create fully valid quality masks for synthetic windows."""

    return np.ones(length, dtype=np.int8)


def _assert_feature_rows_equal(
    testcase: unittest.TestCase,
    left: dict[str, float | int],
    right: dict[str, float | int],
) -> None:
    """Compare two feature rows containing NaNs field by field."""

    testcase.assertEqual(tuple(left), tuple(right))
    for name in left:
        if isinstance(left[name], (int, np.integer)):
            testcase.assertEqual(left[name], right[name])
        else:
            testcase.assertAlmostEqual(
                float(left[name]), float(right[name]), places=12
            ) if np.isfinite(float(left[name])) else testcase.assertTrue(
                np.isnan(float(right[name]))
            )


class OnlineFeatureRegistryTests(unittest.TestCase):
    """Fix the independent online_feature_v1 registry to prevent mixing offline fields."""

    def test_registry_has_exactly_27_features_and_masks(self) -> None:
        self.assertEqual(len(ONLINE_POWER_FEATURES), 17)
        self.assertEqual(len(ONLINE_TEMPORAL_FEATURES), 10)
        self.assertEqual(len(ONLINE_FEATURE_COLUMNS), 27)
        self.assertEqual(len(ONLINE_MASK_COLUMNS), 27)
        self.assertEqual(
            ONLINE_MASK_COLUMNS,
            tuple(f"mask__{name}" for name in ONLINE_FEATURE_COLUMNS),
        )
        self.assertEqual(len(SEQUENCE_FEATURE_NAMES), 4)

    def test_registry_does_not_include_metadata_or_labels(self) -> None:
        forbidden = {
            "dataset",
            "house",
            "appliance",
            "duration",
            "channel",
            "label",
        }
        joined = " ".join(ONLINE_FEATURE_COLUMNS)
        self.assertFalse(any(token in joined for token in forbidden))


class OnlineHorizonAndSequenceTests(unittest.TestCase):
    """Verify actual horizon lengths, relative indices and four-channel causal sequences."""

    def test_supported_horizons_have_fixed_lengths_without_future_padding(self) -> None:
        expected = {20: (2, 8), 40: (4, 10), 60: (6, 12), 120: (12, 18)}
        self.assertEqual(SUPPORTED_HORIZONS_SECONDS, tuple(expected))
        for horizon, (post_count, length) in expected.items():
            self.assertEqual(post_bin_count_for_horizon(horizon), post_count)
            self.assertEqual(expected_sequence_length_for_horizon(horizon), length)
            indices = relative_bin_indices_for_horizon(horizon)
            self.assertEqual(indices.shape, (length,))
            self.assertEqual(indices[0], -6)
            self.assertEqual(indices[-1], post_count - 1)

    def test_h20_sequence_uses_six_pre_plus_two_post_and_preserves_nan_quality(self) -> None:
        power = np.asarray([200.0] * 6 + [1_200.0, np.nan])
        quality = np.asarray([1] * 7 + [0], dtype=np.int8)

        values = build_causal_sequence_inputs(power, quality)

        self.assertEqual(values.shape, (8, 4))
        np.testing.assert_allclose(values[:7, 0], power[:7])
        self.assertTrue(np.isnan(values[7, 0]))
        np.testing.assert_allclose(values[:6, 1], np.zeros(6))
        self.assertEqual(values[6, 1], 1_000.0)
        self.assertTrue(np.isnan(values[7, 1]))
        self.assertTrue(np.isnan(values[0, 2]))
        np.testing.assert_allclose(values[1:6, 2], np.zeros(5))
        self.assertEqual(values[6, 2], 1_000.0)
        self.assertTrue(np.isnan(values[7, 2]))
        np.testing.assert_array_equal(values[:, 3], [0.0] * 6 + [1.0, 1.0])

    def test_h20_half_open_contract_excludes_bin_starting_at_t0_plus_20(self) -> None:
        # If t0 has relative index 0, H20 post bins have indices 0 and 1; index 2 starts exactly at t0+20.
        full_h120 = np.asarray([200.0] * 6 + [1_200.0] * 12)
        h20_length = expected_sequence_length_for_horizon(20)
        h20_values = build_causal_sequence_inputs(
            full_h120[:h20_length], _all_valid(h20_length)
        )

        self.assertEqual(h20_values.shape, (8, 4))
        self.assertEqual(relative_bin_indices_for_horizon(20)[-1], 1)
        self.assertEqual(full_h120[h20_length], 1_200.0)

    def test_future_mutation_does_not_change_h20_or_h40_outputs(self) -> None:
        full = np.asarray([200.0] * 6 + [1_200.0] * 12, dtype=np.float64)

        for horizon in (20, 40):
            length = expected_sequence_length_for_horizon(horizon)
            original_sequence = build_causal_sequence_inputs(
                full[:length], _all_valid(length)
            )
            original_features = extract_online_features(
                full[:6],
                full[6:length],
                _all_valid(6),
                _all_valid(length - 6),
                horizon_seconds=horizon,
            )

            altered = full.copy()
            altered[length:] = np.linspace(9_999.0, 19_999.0, altered.size - length)
            altered_sequence = build_causal_sequence_inputs(
                altered[:length], _all_valid(length)
            )
            altered_features = extract_online_features(
                altered[:6],
                altered[6:length],
                _all_valid(6),
                _all_valid(length - 6),
                horizon_seconds=horizon,
            )

            np.testing.assert_allclose(
                original_sequence, altered_sequence, equal_nan=True
            )
            _assert_feature_rows_equal(self, original_features, altered_features)


class OnlineFeatureExtractionTests(unittest.TestCase):
    """Verify online_feature_v1 power, timing, missingness and normalization definitions."""

    def test_200_to_1200_step_has_consistent_direction_across_h20_h40_h60(self) -> None:
        pre = np.full(6, 200.0)
        for horizon in (20, 40, 60):
            post = np.full(post_bin_count_for_horizon(horizon), 1_200.0)
            row = extract_online_features(
                pre,
                post,
                _all_valid(pre.size),
                _all_valid(post.size),
                horizon_seconds=horizon,
            )

            self.assertEqual(row["online_power__pre_mean"], 200.0)
            self.assertEqual(row["online_power__observed_post_mean"], 1_200.0)
            self.assertEqual(row["online_power__delta_mean"], 1_000.0)
            self.assertEqual(row["online_power__relative_delta"], 5.0)
            self.assertEqual(row["online_power__max_positive_step"], 1_000.0)
            self.assertEqual(row["online_power__max_negative_step"], 0.0)
            self.assertEqual(
                row["online_temporal__observed_persistence_fraction"], 1.0
            )
            self.assertEqual(row["online_temporal__observed_time_to_peak"], 0.0)
            self.assertEqual(row["online_temporal__positive_step_count"], 1.0)
            self.assertEqual(row["online_temporal__negative_step_count"], 0.0)
            self.assertEqual(row["online_temporal__sign_change_count"], 0.0)
            self.assertEqual(row["online_temporal__observed_post_linear_slope"], 0.0)
            self.assertEqual(row["online_temporal__initial_recent_difference"], 0.0)
            self.assertEqual(row["online_temporal__longest_active_fraction"], 1.0)

    def test_mean_area_is_normalised_by_horizon_seconds(self) -> None:
        pre = np.full(6, 200.0)
        areas: list[float] = []
        for horizon in SUPPORTED_HORIZONS_SECONDS:
            post = np.full(post_bin_count_for_horizon(horizon), 1_200.0)
            row = extract_online_features(
                pre,
                post,
                _all_valid(6),
                _all_valid(post.size),
                horizon_seconds=horizon,
            )
            areas.append(float(row["online_power__mean_area_above_baseline"]))
        np.testing.assert_allclose(areas, [1_000.0] * 4)

    def test_initial_and_recent_20_seconds_use_their_exact_two_bin_regions(self) -> None:
        pre = np.full(6, 200.0)
        post = np.asarray([1_200.0, 1_000.0, 800.0, 600.0, 400.0, 200.0])
        row = extract_online_features(
            pre,
            post,
            _all_valid(6),
            _all_valid(6),
            horizon_seconds=60,
        )

        self.assertEqual(row["online_temporal__initial_20s_mean"], 1_100.0)
        self.assertEqual(row["online_temporal__recent_20s_mean"], 300.0)
        self.assertEqual(row["online_temporal__initial_recent_difference"], 800.0)
        self.assertLess(row["online_temporal__observed_post_linear_slope"], 0.0)

    def test_max_negative_step_retains_its_negative_sign(self) -> None:
        row = extract_online_features(
            np.full(6, 200.0),
            np.asarray([1_200.0, 700.0, 700.0, 700.0]),
            _all_valid(6),
            _all_valid(4),
            horizon_seconds=40,
        )

        self.assertEqual(row["online_power__max_positive_step"], 1_000.0)
        self.assertEqual(row["online_power__max_negative_step"], -500.0)
        self.assertEqual(row["online_temporal__negative_step_count"], 1.0)

    def test_accepted_h60_with_one_missing_post_keeps_feature_masks_honest(self) -> None:
        pre = np.full(6, 200.0)
        post = np.asarray([1_200.0, 1_200.0, 1_200.0, np.nan, 1_200.0, 1_200.0])
        pre_quality = _all_valid(6)
        post_quality = _all_valid(6)
        post_quality[3] = 0

        quality = assess_online_horizon_quality(
            pre,
            post,
            pre_quality,
            post_quality,
        )
        row = extract_online_features(
            pre,
            post,
            pre_quality,
            post_quality,
            horizon_seconds=60,
        )

        self.assertTrue(quality.accepted)
        self.assertEqual(quality.post_valid_bin_count, 5)
        self.assertTrue(np.isfinite(row["online_power__observed_post_mean"]))
        self.assertTrue(np.isnan(row["online_power__mean_area_above_baseline"]))
        self.assertTrue(
            np.isnan(row["online_temporal__observed_persistence_fraction"])
        )
        for feature_name, mask_name in zip(
            ONLINE_FEATURE_COLUMNS, ONLINE_MASK_COLUMNS, strict=True
        ):
            self.assertEqual(
                row[mask_name], int(np.isfinite(row[feature_name]))
            )

    def test_h20_and_h40_coverage_require_all_post_bins(self) -> None:
        pre = np.full(6, 200.0)
        for horizon in (20, 40):
            post_count = post_bin_count_for_horizon(horizon)
            post = np.full(post_count, 1_200.0)
            post_quality = _all_valid(post_count)
            post_quality[-1] = 0
            quality = assess_online_horizon_quality(
                pre,
                post,
                _all_valid(6),
                post_quality,
            )
            self.assertFalse(quality.accepted)
            self.assertIn("insufficient_post_coverage", quality.rejection_reasons)


if __name__ == "__main__":
    unittest.main()
