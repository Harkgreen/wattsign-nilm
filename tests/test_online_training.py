from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as torch_functional

import load_signature_analysis.online_training as online_training_module
from load_signature_analysis.grouped_mlp import GroupedFingerprintMLP
from load_signature_analysis.mvp_preprocessing import (
    ExplicitHouseSplit,
    MaskedStandardizer,
)
from load_signature_analysis.online_features import (
    ONLINE_FEATURE_COLUMNS,
    ONLINE_MASK_COLUMNS,
)
from load_signature_analysis.online_training import (
    OnlineFeatureTable,
    compute_train_weighted_pos_weight,
    compute_weighted_binary_metrics,
    is_high_confidence,
    load_online_feature_table,
    run_online_weighted_grouped_mlp,
    select_weighted_validation_threshold,
    weighted_bce_with_logits,
)


def _sha256(path: Path) -> str:
    """Compute fixture file digests to emulate a formal sample manifest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _online_row(
    *,
    candidate_id: str,
    dataset: str,
    house_id: int,
    split: str,
    horizon_seconds: int,
    label: int,
    label_type: str,
    label_confidence: str,
    sample_weight: float,
    source_appliance: str,
    feature_offset: float,
) -> dict[str, object]:
    """Create one record satisfying the online_causal_v1_1 scalar-feature contract."""

    row: dict[str, object] = {
        "candidate_id": candidate_id,
        "dataset": dataset,
        "house_id": house_id,
        "split": split,
        "candidate_onset_s": 1_700_000_000 + len(candidate_id),
        "candidate_detected_at_s": 1_700_000_020 + len(candidate_id),
        "horizon_seconds": horizon_seconds,
        "label": label,
        "label_type": label_type,
        "label_confidence": label_confidence,
        "sample_weight": sample_weight,
        "source_appliance": source_appliance,
        "accepted": True,
    }
    for feature_index, feature_name in enumerate(ONLINE_FEATURE_COLUMNS):
        row[feature_name] = feature_offset + feature_index / 100.0
        row[f"mask__{feature_name}"] = 1
    return row


class OnlineTrainingFixture:
    """Write samples with three house-level splits, four weight types and cross-horizon distractor rows."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.input_dir = root / "online_input"
        self.input_dir.mkdir(parents=True)
        self.rows = self._default_rows()
        self.manifest_overrides: dict[str, object] = {}
        self.write()

    @staticmethod
    def _default_rows() -> list[dict[str, object]]:
        definitions = (
            # Training weights deliberately correlate with features to detect accidental weighted scaler fitting.
            ("train_core", "refit", 1, "train", 1, "core_positive", "high", 1.0, "kettle", 0.0),
            (
                "train_h1_heater",
                "refit",
                1,
                "train",
                1,
                "core_positive",
                "medium_missing_refit_issues",
                0.5,
                "electric_heater",
                10.0,
            ),
            ("train_hard", "refit", 1, "train", 0, "hard_negative", "high", 1.0, "microwave", 20.0),
            (
                "train_background",
                "refit",
                1,
                "train",
                0,
                "background_negative",
                "weak_unmonitored_background",
                0.25,
                "",
                30.0,
            ),
            ("validation_core", "refit", 2, "validation", 1, "core_positive", "high", 1.0, "toaster", 1_000.0),
            ("validation_hard", "refit", 2, "validation", 0, "hard_negative", "high", 1.0, "microwave", 1_010.0),
            (
                "validation_background",
                "refit",
                2,
                "validation",
                0,
                "background_negative",
                "weak_unmonitored_background",
                0.25,
                "",
                1_020.0,
            ),
            ("validation_core_2", "refit", 2, "validation", 1, "core_positive", "high", 1.0, "kettle", 1_030.0),
            ("test_core", "refit", 19, "test", 1, "core_positive", "high", 1.0, "kettle", -1_000.0),
            ("test_hard", "refit", 19, "test", 0, "hard_negative", "high", 1.0, "microwave", -1_010.0),
            (
                "test_background",
                "refit",
                19,
                "test",
                0,
                "background_negative",
                "weak_unmonitored_background",
                0.25,
                "",
                -1_020.0,
            ),
            ("test_core_2", "refit", 19, "test", 1, "core_positive", "high", 1.0, "toaster", -1_030.0),
        )
        rows = [
            _online_row(
                candidate_id=candidate_id,
                dataset=dataset,
                house_id=house_id,
                split=split,
                horizon_seconds=20,
                label=label,
                label_type=label_type,
                label_confidence=confidence,
                sample_weight=weight,
                source_appliance=appliance,
                feature_offset=offset,
            )
            for (
                candidate_id,
                dataset,
                house_id,
                split,
                label,
                label_type,
                confidence,
                weight,
                appliance,
                offset,
            ) in definitions
        ]
        # An H40 row with the same candidate_id proves horizon belongs to the composite key.
        rows.append(
            _online_row(
                candidate_id="train_core",
                dataset="refit",
                house_id=1,
                split="train",
                horizon_seconds=40,
                label=1,
                label_type="core_positive",
                label_confidence="high",
                sample_weight=1.0,
                source_appliance="kettle",
                feature_offset=1_000_000.0,
            )
        )
        return rows

    @property
    def split(self) -> ExplicitHouseSplit:
        return ExplicitHouseSplit.from_mapping(
            {
                "train": ["refit:1"],
                "validation": ["refit:2"],
                "test": ["refit:19"],
            }
        )

    def write(self, *, reverse_columns: bool = False) -> None:
        """Write CSV and a minimal manifest with matching counts and digests."""

        table = pd.DataFrame(self.rows)
        if reverse_columns:
            table = table.loc[:, list(reversed(table.columns))]
        csv_path = self.input_dir / "causal_features.csv"
        table.to_csv(csv_path, index=False, encoding="utf-8-sig")

        horizon_counts = {
            f"h{horizon}": int(count)
            for horizon, count in table.groupby("horizon_seconds").size().items()
        }
        for horizon in (20, 40, 60, 120):
            horizon_counts.setdefault(f"h{horizon}", 0)
        manifest: dict[str, object] = {
            "schema_version": "online_causal_v1_1",
            "feature_registry_version": "online_feature_v1_1",
            "npz_contract_version": "online_sequence_v1_1",
            "online_feature_registry": list(ONLINE_FEATURE_COLUMNS),
            "horizons_seconds": [20, 40, 60, 120],
            "horizon_sample_counts": horizon_counts,
            "split": self.split.to_dict(),
            "output_files": {
                "causal_features.csv": {
                    "path": "causal_features.csv",
                    "size_bytes": csv_path.stat().st_size,
                    "sha256": _sha256(csv_path),
                }
            },
        }
        manifest.update(self.manifest_overrides)
        (self.input_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def write_training_config(self) -> Path:
        """Write a full configuration for two quick CPU training epochs."""

        (self.input_dir / "taxonomy_snapshot.yaml").write_text(
            "schema_version: online-training-test\n",
            encoding="utf-8",
        )
        config_path = self.root / "online_training.toml"
        config_path.write_text(
            """
[schema]
training_schema_version = "online_weighted_grouped_mlp_v1"
input_schema_version = "online_causal_v1_1"
feature_registry_version = "online_feature_v1_1"
npz_contract_version = "online_sequence_v1_1"

[paths]
online_input_dir = "online_input"
run_output_root = "runs"

[dataset]
horizon_seconds = 20
csv_chunksize = 5

[split]
train = ["refit:1"]
validation = ["refit:2"]
test = ["refit:19"]
known_constraint = "合成跨户划分只用于端到端测试。"

[model]
embedding_dimension = 4
encoder_hidden_dimension = 8
classifier_hidden_dimension = 4
dropout_probability = 0.0
feature_dropout_enabled = false
group_dropout_probability = 0.0
feature_dropout_probability = 0.0

[training]
seed = 20260825
epochs = 2
smoke_epochs = 2
batch_size = 4
learning_rate = 0.005
weight_decay = 0.0
early_stopping_patience = 4
num_workers = 0

[smoke]
maximum_rows_per_group = 8

[loss]
class_balance = "train_weighted_negative_mass_over_positive_mass"
sample_weight_normalization = "sum_sample_weight"
sampler = "shuffle_without_weighted_resampling"

[evaluation]
threshold_objective = "validation_all_candidates_sample_weighted_f1"
threshold_tie_break = "weighted_balanced_accuracy_then_weighted_recall_then_closest_to_0.5_then_lower_threshold"
high_confidence_definition = "label_type in {core_positive,hard_negative} and label_confidence=high and sample_weight=1.0"
""".strip()
            + "\n",
            encoding="utf-8",
        )
        return config_path


class OnlineFeatureTableLoadingTests(unittest.TestCase):
    """Verify the H20 scalar-table schema, ordering, splits and missingness contract."""

    def test_loader_selects_one_horizon_and_uses_registry_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = OnlineTrainingFixture(Path(temporary))
            table = load_online_feature_table(
                fixture.input_dir,
                horizon_seconds=20,
                expected_split=fixture.split,
            )

            self.assertIsInstance(table, OnlineFeatureTable)
            self.assertEqual(table.horizon_seconds, 20)
            self.assertEqual(table.values.shape, (12, len(ONLINE_FEATURE_COLUMNS)))
            self.assertEqual(table.masks.shape, (12, len(ONLINE_MASK_COLUMNS)))
            self.assertEqual(table.labels.shape, (12,))
            self.assertEqual(table.sample_weights.shape, (12,))
            self.assertEqual(table.split_values.shape, (12,))
            self.assertNotIn(1_000_000.0, table.values[:, 0])
            self.assertEqual(set(table.metadata["horizon_seconds"]), {20})

            position = int(
                np.flatnonzero(
                    table.metadata["candidate_id"].astype(str).to_numpy()
                    == "train_core"
                )[0]
            )
            expected = np.asarray(
                [index / 100.0 for index in range(len(ONLINE_FEATURE_COLUMNS))]
            )
            np.testing.assert_allclose(table.values[position], expected)
            np.testing.assert_array_equal(
                table.masks[position], np.ones(len(ONLINE_MASK_COLUMNS), dtype=np.int8)
            )

    def test_physical_csv_column_order_does_not_change_model_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = OnlineTrainingFixture(Path(temporary))
            normal = load_online_feature_table(fixture.input_dir, horizon_seconds=20)
            fixture.write(reverse_columns=True)
            reversed_columns = load_online_feature_table(
                fixture.input_dir, horizon_seconds=20
            )

            np.testing.assert_allclose(reversed_columns.values, normal.values)
            np.testing.assert_array_equal(reversed_columns.masks, normal.masks)
            np.testing.assert_array_equal(reversed_columns.labels, normal.labels)
            self.assertEqual(
                reversed_columns.metadata["candidate_id"].tolist(),
                normal.metadata["candidate_id"].tolist(),
            )

    def test_expected_house_split_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = OnlineTrainingFixture(Path(temporary))
            wrong_split = ExplicitHouseSplit.from_mapping(
                {
                    "train": ["refit:1"],
                    "validation": ["refit:4"],
                    "test": ["refit:19"],
                }
            )

            with self.assertRaises(ValueError):
                load_online_feature_table(
                    fixture.input_dir,
                    horizon_seconds=20,
                    expected_split=wrong_split,
                )

    def test_missing_requested_horizon_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = OnlineTrainingFixture(Path(temporary))

            with self.assertRaises(ValueError):
                load_online_feature_table(fixture.input_dir, horizon_seconds=60)

    def test_manifest_registry_counts_and_feature_hash_are_mandatory(self) -> None:
        """Formal inputs require the registry, horizon counts and feature digest together."""

        def remove_registry(manifest: dict[str, object]) -> None:
            manifest.pop("online_feature_registry")

        def remove_horizon_counts(manifest: dict[str, object]) -> None:
            manifest.pop("horizon_sample_counts")

        def remove_feature_hash(manifest: dict[str, object]) -> None:
            output_files = manifest["output_files"]
            assert isinstance(output_files, dict)
            description = output_files["causal_features.csv"]
            assert isinstance(description, dict)
            description.pop("sha256")

        cases = {
            "缺少 online_feature_registry": remove_registry,
            "缺少 horizon_sample_counts": remove_horizon_counts,
            "缺少 causal_features SHA-256": remove_feature_hash,
        }
        for case_name, mutate in cases.items():
            with self.subTest(case_name=case_name), tempfile.TemporaryDirectory() as temporary:
                fixture = OnlineTrainingFixture(Path(temporary))
                manifest_path = fixture.input_dir / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                mutate(manifest)
                manifest_path.write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                with self.assertRaises(ValueError):
                    load_online_feature_table(fixture.input_dir, horizon_seconds=20)

    def test_frozen_label_confidence_weight_combinations_are_enforced(self) -> None:
        """Category, confidence and supervision weight cannot be recombined arbitrarily."""

        def weak_hard_negative(fixture: OnlineTrainingFixture) -> None:
            fixture.rows[2]["sample_weight"] = 0.25

        def downweighted_high_core(fixture: OnlineTrainingFixture) -> None:
            fixture.rows[0]["sample_weight"] = 0.5

        def unknown_confidence(fixture: OnlineTrainingFixture) -> None:
            fixture.rows[0]["label_confidence"] = "unknown_future_policy"

        cases = {
            "hard negative 错用背景权重": weak_hard_negative,
            "high core 错误降权": downweighted_high_core,
            "未知置信度": unknown_confidence,
        }
        for case_name, mutate in cases.items():
            with self.subTest(case_name=case_name), tempfile.TemporaryDirectory() as temporary:
                fixture = OnlineTrainingFixture(Path(temporary))
                mutate(fixture)
                fixture.write()

                with self.assertRaises(ValueError):
                    load_online_feature_table(fixture.input_dir, horizon_seconds=20)

    def test_invalid_contract_variants_are_rejected(self) -> None:
        def remove_feature(fixture: OnlineTrainingFixture) -> None:
            for row in fixture.rows:
                row.pop(ONLINE_FEATURE_COLUMNS[0])

        def duplicate_selected_id(fixture: OnlineTrainingFixture) -> None:
            fixture.rows.append(dict(fixture.rows[0]))

        def nonbinary_mask(fixture: OnlineTrainingFixture) -> None:
            fixture.rows[0][ONLINE_MASK_COLUMNS[0]] = 2

        def observed_nan(fixture: OnlineTrainingFixture) -> None:
            fixture.rows[0][ONLINE_FEATURE_COLUMNS[0]] = np.nan
            fixture.rows[0][ONLINE_MASK_COLUMNS[0]] = 1

        def nonpositive_weight(fixture: OnlineTrainingFixture) -> None:
            fixture.rows[0]["sample_weight"] = 0.0

        def unknown_split(fixture: OnlineTrainingFixture) -> None:
            fixture.rows[0]["split"] = "future"

        def old_schema(fixture: OnlineTrainingFixture) -> None:
            fixture.manifest_overrides["schema_version"] = "online_causal_v1"

        def wrong_registry(fixture: OnlineTrainingFixture) -> None:
            fixture.manifest_overrides["feature_registry_version"] = "online_feature_v1"

        cases = {
            "缺少冻结特征": remove_feature,
            "同一 horizon 候选重复": duplicate_selected_id,
            "mask 非二值": nonbinary_mask,
            "观测值非有限": observed_nan,
            "样本权重非正": nonpositive_weight,
            "未知 split": unknown_split,
            "旧样本 schema": old_schema,
            "错误特征 registry": wrong_registry,
        }
        for case_name, mutate in cases.items():
            with self.subTest(case_name=case_name), tempfile.TemporaryDirectory() as temporary:
                fixture = OnlineTrainingFixture(Path(temporary))
                mutate(fixture)
                fixture.write()
                with self.assertRaises(ValueError):
                    load_online_feature_table(fixture.input_dir, horizon_seconds=20)

    def test_scaler_uses_unweighted_training_observations_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = OnlineTrainingFixture(Path(temporary))
            table = load_online_feature_table(fixture.input_dir, horizon_seconds=20)
            train_rows = table.split_values == "train"

            scaler = MaskedStandardizer(ONLINE_FEATURE_COLUMNS).fit(
                table.values[train_rows], table.masks[train_rows]
            )

            # Training values 0/10/20/30 have mean 15; incorrectly using supervision weights changes that value.
            self.assertAlmostEqual(float(scaler.mean_[0]), 15.0)
            self.assertNotAlmostEqual(
                float(scaler.mean_[0]),
                float(
                    np.average(
                        table.values[train_rows, 0],
                        weights=table.sample_weights[train_rows],
                    )
                ),
            )
            self.assertLess(float(np.max(table.values[train_rows, 0])), 100.0)


class OnlineWeightedTrainingMathTests(unittest.TestCase):
    """Fix the mathematical semantics of confidence weights, class weights, thresholds and metrics."""

    def test_pos_weight_uses_training_effective_class_mass_only(self) -> None:
        labels = np.asarray([0, 0, 1, 1, 0, 1, 0, 1], dtype=np.int8)
        sample_weights = np.asarray(
            [0.25, 1.0, 0.5, 1.0, 100.0, 0.01, 0.01, 100.0],
            dtype=np.float32,
        )
        split_values = np.asarray(
            ["train"] * 4 + ["validation"] * 2 + ["test"] * 2
        )

        result = compute_train_weighted_pos_weight(
            labels, sample_weights, split_values
        )

        self.assertAlmostEqual(result, 5.0 / 6.0)

    def test_pos_weight_rejects_a_training_split_without_both_classes(self) -> None:
        with self.assertRaises(ValueError):
            compute_train_weighted_pos_weight(
                np.asarray([1, 1, 0]),
                np.asarray([1.0, 0.5, 1.0]),
                np.asarray(["train", "train", "validation"]),
            )

    def test_weighted_bce_matches_the_frozen_manual_formula(self) -> None:
        logits = torch.tensor([-1.0, 0.5, 2.0, -0.25], dtype=torch.float64)
        labels = torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=torch.float64)
        sample_weights = torch.tensor([0.25, 0.5, 1.0, 1.0], dtype=torch.float64)
        pos_weight = 5.0 / 6.0
        expected_per_sample = torch_functional.binary_cross_entropy_with_logits(
            logits,
            labels,
            reduction="none",
            pos_weight=torch.tensor(pos_weight, dtype=torch.float64),
        )
        expected = torch.sum(expected_per_sample * sample_weights) / torch.sum(
            sample_weights
        )

        actual = weighted_bce_with_logits(
            logits, labels, sample_weights, pos_weight
        )

        torch.testing.assert_close(actual, expected)
        self.assertEqual(actual.ndim, 0)

    def test_weighted_threshold_can_differ_from_unweighted_optimum(self) -> None:
        # The second positive has low confidence, so weighted F1 favors accepting only the first confident positive.
        labels = np.asarray([1, 1, 0], dtype=np.int8)
        probabilities = np.asarray([0.9, 0.4, 0.5], dtype=np.float64)
        sample_weights = np.asarray([1.0, 0.1, 1.0], dtype=np.float64)

        result = select_weighted_validation_threshold(
            labels, probabilities, sample_weights
        )

        self.assertAlmostEqual(result.threshold, 0.9)

    def test_weighted_metrics_use_fractional_confusion_mass(self) -> None:
        labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
        probabilities = np.asarray([0.1, 0.8, 0.7, 0.2], dtype=np.float64)
        sample_weights = np.asarray([0.25, 1.0, 0.5, 1.0], dtype=np.float64)

        metrics = compute_weighted_binary_metrics(
            labels, probabilities, sample_weights, threshold=0.5
        )

        self.assertEqual(metrics["sample_count"], 4)
        self.assertAlmostEqual(metrics["sample_weight_sum"], 2.75)
        self.assertAlmostEqual(metrics["negative_weight"], 1.25)
        self.assertAlmostEqual(metrics["positive_weight"], 1.5)
        self.assertAlmostEqual(metrics["accuracy"], 0.75 / 2.75)
        self.assertAlmostEqual(metrics["precision"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["recall"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["f1"], 1.0 / 3.0)
        np.testing.assert_allclose(
            metrics["confusion_matrix"], [[0.25, 1.0], [1.0, 0.5]]
        )

    def test_high_confidence_requires_type_confidence_and_full_weight(self) -> None:
        label_types = np.asarray(
            [
                "core_positive",
                "hard_negative",
                "background_negative",
                "core_positive",
                "core_positive",
                "hard_negative",
            ]
        )
        label_confidences = np.asarray(
            [
                "high",
                "high",
                "high",
                "medium_missing_refit_issues",
                "high",
                "high",
            ]
        )
        sample_weights = np.asarray([1.0, 1.0, 1.0, 1.0, 0.5, 0.25])

        selected = is_high_confidence(
            label_types, label_confidences, sample_weights
        )

        self.assertEqual(selected.dtype, np.dtype(bool))
        np.testing.assert_array_equal(
            selected, np.asarray([True, True, False, False, False, False])
        )




if __name__ == "__main__":
    unittest.main()
