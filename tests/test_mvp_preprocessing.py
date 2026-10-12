from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from load_signature_analysis.mvp_preprocessing import (
    ExplicitHouseSplit,
    MaskedStandardizer,
    assert_no_split_leakage,
    compute_binary_metrics,
    select_validation_threshold,
)


class ExplicitHouseSplitTests(unittest.TestCase):
    def test_dataset_and_house_joint_key_prevents_id_collision(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "dataset": "refit",
                    "house_id": 5,
                    "label": 0,
                    "window_overlap_cluster_id": "refit_h05_a",
                },
                {
                    "dataset": "uk_dale",
                    "house_id": 5,
                    "label": 1,
                    "window_overlap_cluster_id": "uk_dale_h05_a",
                },
                {
                    "dataset": "refit",
                    "house_id": 9,
                    "label": 1,
                    "window_overlap_cluster_id": "refit_h09_a",
                },
            ]
        )
        split = ExplicitHouseSplit.from_mapping(
            {
                "train": ["refit:5"],
                "validation": [("refit", 9)],
                "test": ["uk_dale:5"],
            }
        )

        assigned = split.assign(events)

        self.assertEqual(assigned["split"].tolist(), ["train", "test", "validation"])
        self.assertEqual(split.to_dict()["test"], ["uk_dale:5"])

    def test_duplicate_or_missing_assignment_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "重复分配"):
            ExplicitHouseSplit.from_mapping(
                {
                    "train": ["refit:1"],
                    "validation": ["refit:1"],
                    "test": [],
                }
            )
        events = pd.DataFrame([{"dataset": "refit", "house_id": 2}])
        split = ExplicitHouseSplit.from_mapping(
            {"train": ["refit:1"], "validation": [], "test": []}
        )
        with self.assertRaisesRegex(ValueError, "没有 split 分配"):
            split.assign(events)

    def test_same_house_or_overlap_cluster_cannot_cross_split(self) -> None:
        same_house = pd.DataFrame(
            [
                {"dataset": "refit", "house_id": 1, "split": "train"},
                {"dataset": "refit", "house_id": 1, "split": "test"},
            ]
        )
        with self.assertRaisesRegex(ValueError, "house-level"):
            assert_no_split_leakage(same_house)

        same_cluster = pd.DataFrame(
            [
                {
                    "dataset": "refit",
                    "house_id": 1,
                    "split": "train",
                    "window_overlap_cluster_id": "shared",
                },
                {
                    "dataset": "refit",
                    "house_id": 2,
                    "split": "test",
                    "window_overlap_cluster_id": "shared",
                },
            ]
        )
        with self.assertRaisesRegex(ValueError, "重叠窗口簇"):
            assert_no_split_leakage(same_cluster)

    def test_optional_class_coverage_check_applies_to_each_split(self) -> None:
        events = pd.DataFrame(
            [
                {"dataset": "refit", "house_id": 1, "label": 0},
                {"dataset": "refit", "house_id": 1, "label": 1},
                {"dataset": "refit", "house_id": 2, "label": 0},
                {"dataset": "refit", "house_id": 3, "label": 1},
            ]
        )
        split = ExplicitHouseSplit.from_mapping(
            {
                "train": ["refit:1"],
                "validation": ["refit:2"],
                "test": ["refit:3"],
            }
        )
        with self.assertRaisesRegex(ValueError, "同时包含正负类"):
            split.assign(events, require_both_classes=True)


class MaskedStandardizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.names = ("varying", "constant", "train_missing")

    def test_fit_uses_only_observed_training_values(self) -> None:
        values = np.asarray(
            [
                [1.0, 10.0, np.nan],
                [3.0, 999.0, np.nan],
            ]
        )
        masks = np.asarray([[1, 1, 0], [1, 0, 0]])
        scaler = MaskedStandardizer(self.names).fit(values, masks)

        np.testing.assert_allclose(scaler.mean_, [2.0, 10.0, 0.0])
        np.testing.assert_allclose(scaler.scale_, [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(scaler.observed_count_, [2, 1, 0])
        np.testing.assert_array_equal(scaler.supported_, [True, True, False])

        transformed, effective_masks = scaler.transform(
            np.asarray([[2.0, 5000.0, 7.0]]),
            np.asarray([[1, 0, 1]]),
        )
        np.testing.assert_allclose(transformed, [[0.0, 0.0, 0.0]])
        np.testing.assert_array_equal(effective_masks, [[1, 0, 0]])

    def test_missing_zero_and_observed_standardised_zero_remain_distinguishable(self) -> None:
        scaler = MaskedStandardizer(("a", "b")).fit(
            np.asarray([[1.0, 5.0], [3.0, 7.0]]),
            np.ones((2, 2), dtype=int),
        )
        transformed, masks = scaler.transform(
            np.asarray([[2.0, np.nan]]),
            np.asarray([[1, 0]]),
        )

        np.testing.assert_allclose(transformed, [[0.0, 0.0]])
        np.testing.assert_array_equal(masks, [[1, 0]])

    def test_serialisation_round_trip_preserves_transform(self) -> None:
        values = np.asarray([[1.0, 2.0], [5.0, 2.0]])
        masks = np.ones_like(values, dtype=int)
        original = MaskedStandardizer(("a", "b")).fit(values, masks)
        restored = MaskedStandardizer.from_dict(original.to_dict())

        expected = original.transform(values, masks)
        actual = restored.transform(values, masks, feature_names=("a", "b"))
        np.testing.assert_allclose(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        with self.assertRaisesRegex(ValueError, "特征顺序"):
            restored.transform(values, masks, feature_names=("b", "a"))

    def test_observed_nonfinite_value_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "有限数"):
            MaskedStandardizer(("a",)).fit(
                np.asarray([[np.nan]]),
                np.asarray([[1]]),
            )


class ThresholdAndMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.probabilities = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    def test_threshold_tie_breaks_by_balanced_accuracy_then_recall(self) -> None:
        balanced_winner = select_validation_threshold(
            np.asarray([1, 0, 0, 0, 1, 0]), self.probabilities
        )
        self.assertAlmostEqual(balanced_winner.threshold, 0.5)

        recall_winner = select_validation_threshold(
            np.asarray([0, 0, 1, 0, 0, 1]), self.probabilities
        )
        self.assertAlmostEqual(recall_winner.threshold, 0.3)

    def test_threshold_final_tie_prefers_closest_to_half(self) -> None:
        result = select_validation_threshold(
            np.asarray([1, 0, 0, 0, 0, 0]), self.probabilities
        )
        self.assertAlmostEqual(result.threshold, 0.1)
        self.assertIn("closest_to_0.5", result.selection_rule)

    def test_single_class_metrics_are_safe_but_threshold_selection_rejects_it(self) -> None:
        labels = np.zeros(3, dtype=int)
        probabilities = np.asarray([0.1, 0.2, 0.3])
        metrics = compute_binary_metrics(labels, probabilities, threshold=0.5)

        self.assertIsNone(metrics["balanced_accuracy"])
        self.assertIsNone(metrics["roc_auc"])
        self.assertIsNone(metrics["pr_auc"])
        self.assertEqual(metrics["confusion_matrix"], [[3, 0], [0, 0]])
        with self.assertRaisesRegex(ValueError, "同时包含正负类"):
            select_validation_threshold(labels, probabilities)


if __name__ == "__main__":
    unittest.main()
