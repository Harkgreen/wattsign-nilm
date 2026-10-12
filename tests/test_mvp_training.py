from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import os

import numpy as np
import pandas as pd
import torch

from load_signature_analysis.event_features import FEATURE_COLUMNS, MASK_COLUMNS
from load_signature_analysis.grouped_mlp import GroupedFingerprintMLP
from load_signature_analysis.mvp_training import (
    _metrics_by_slice,
    _predict,
    fit_grouped_mlp,
    prepare_combined_features,
    run_grouped_mlp_mvp,
)


def _feature_row(
    *,
    event_id: str,
    dataset: str,
    house_id: int,
    label: int,
    feature_offset: float,
) -> dict[str, object]:
    """Create one synthetic event satisfying the MVP input contract."""

    row: dict[str, object] = {
        "event_id": event_id,
        "dataset": dataset,
        "house_id": house_id,
        "channel_id": 1,
        "source_appliance": "kettle" if label else "microwave",
        "label": label,
        "event_timestamp": 1_700_000_000 + len(event_id),
        "event_utc": "2023-11-14T22:13:20Z",
        "window_overlap_cluster_id": f"cluster_{event_id}",
    }
    for feature_index, feature_name in enumerate(FEATURE_COLUMNS):
        row[feature_name] = feature_offset + feature_index / 100.0
        row[f"mask__{feature_name}"] = 1
    return row


def _write_synthetic_sources(project_root: Path) -> dict[str, object]:
    """Write three tiny feature tables and return a preprocessing configuration."""

    refit_directory = project_root / "prepared_refit"
    refit_directory.mkdir(parents=True)
    prepared_ukdale = project_root / "prepared_ukdale"
    prepared_ukdale.mkdir(parents=True)

    source_definitions = (
        (
            refit_directory / "event_features.csv",
            "refit",
            1,
            "train",
        ),
        (
            prepared_ukdale / "house5.csv",
            "uk_dale",
            5,
            "validation",
        ),
        (
            prepared_ukdale / "house2.csv",
            "uk_dale",
            2,
            "test",
        ),
    )
    for path, dataset, house_id, split_name in source_definitions:
        rows = [
            _feature_row(
                event_id=f"{dataset}_{house_id}_{label}",
                dataset=dataset,
                house_id=house_id,
                label=label,
                feature_offset=float(label * 2 + house_id / 10),
            )
            for label in (0, 1)
        ]
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")

    return {
        "paths": {
            "refit_output_dir": "prepared_refit",
            "ukdale_house5_features": "prepared_ukdale/house5.csv",
            "ukdale_house2_features": "prepared_ukdale/house2.csv",
            "combined_output_dir": "combined",
        },
        "split": {
            "train": ["refit:1"],
            "validation": ["uk_dale:5"],
            "test": ["uk_dale:2"],
            "known_constraint": (
                "验证集只有 UK-DALE；此字段是说明文字，不是第四种 split"
            ),
        },
    }


def _model_config() -> dict[str, object]:
    """Return a minimal model configuration that runs quickly."""

    return {
        "encoder_hidden_dimension": 8,
        "embedding_dimension": 4,
        "classifier_hidden_dimension": 4,
        "dropout_probability": 0.0,
        "feature_dropout_enabled": False,
        "group_dropout_probability": 0.0,
        "feature_dropout_probability": 0.0,
    }


def _training_config() -> dict[str, object]:
    """Return a deterministic training configuration for focused tests."""

    return {
        "seed": 20260820,
        "epochs": 2,
        "smoke_epochs": 2,
        "batch_size": 4,
        "learning_rate": 0.005,
        "weight_decay": 0.0,
        "early_stopping_patience": 4,
    }


class CombinedFeaturePreparationTests(unittest.TestCase):
    """Verify configuration boundaries for merged features and the fixed house split."""

    def test_known_constraint_is_metadata_not_an_extra_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            raw_config = _write_synthetic_sources(project_root)

            combined, split = prepare_combined_features(project_root, raw_config)

            self.assertEqual(
                combined["split"].tolist(),
                ["train", "train", "validation", "validation", "test", "test"],
            )
            self.assertEqual(split.to_dict()["test"], ["uk_dale:2"])
            manifest_path = project_root / "combined" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertIn("验证集只有 UK-DALE", manifest["known_split_constraint"])
            self.assertNotIn("known_constraint", manifest["house_split"])


class GroupedMlpTrainingTests(unittest.TestCase):
    """Verify the complete synthetic-data training path of a small grouped MLP."""

    @unittest.skipUnless(os.environ.get("LSA_RUN_TRAINING_TESTS") == "1", "Explicit opt-in required for synthetic optimizer tests")
    def test_two_epoch_training_returns_runnable_best_model(self) -> None:
        rng = np.random.default_rng(42)
        labels = np.tile(np.asarray([0, 1], dtype=np.int64), 12)
        values = rng.normal(0.0, 0.15, size=(labels.size, len(FEATURE_COLUMNS)))
        values[:, :3] += np.where(labels[:, None] == 1, 1.5, -1.5)
        masks = np.ones_like(values, dtype=np.int8)
        masks[::4, -5:] = 0
        values[masks == 0] = 0.0
        split_values = np.asarray(
            ["train"] * 12 + ["validation"] * 8 + ["test"] * 4
        )

        fitted = fit_grouped_mlp(
            values,
            masks,
            labels,
            split_values,
            _model_config(),
            _training_config(),
            epochs_override=2,
        )
        probabilities, gates = _predict(fitted.model, values, masks)

        self.assertEqual(len(fitted.history), 2)
        self.assertIn(fitted.best_epoch, (1, 2))
        self.assertTrue(np.isfinite(fitted.best_validation_loss))
        self.assertTrue(all(np.isfinite(row["train_loss"]) for row in fitted.history))
        self.assertEqual(probabilities.shape, (24,))
        self.assertEqual(gates.shape, (24, 2))
        self.assertTrue(np.all((probabilities >= 0.0) & (probabilities <= 1.0)))
        np.testing.assert_allclose(gates.sum(axis=1), np.ones(24), atol=1e-6)

    @unittest.skipUnless(os.environ.get("LSA_RUN_TRAINING_TESTS") == "1", "Explicit opt-in required for synthetic optimizer tests")
    def test_multi_layer_width_config_trains_without_casting_to_scalar(self) -> None:
        rng = np.random.default_rng(314)
        labels = np.tile(np.asarray([0, 1], dtype=np.int64), 12)
        values = rng.normal(0.0, 0.15, size=(labels.size, len(FEATURE_COLUMNS)))
        values[:, :3] += np.where(labels[:, None] == 1, 1.2, -1.2)
        masks = np.ones_like(values, dtype=np.int8)
        split_values = np.asarray(
            ["train"] * 12 + ["validation"] * 8 + ["test"] * 4
        )
        model_config = _model_config()
        model_config["encoder_hidden_dimension"] = [8, 6]
        model_config["classifier_hidden_dimension"] = [6, 4]

        fitted = fit_grouped_mlp(
            values,
            masks,
            labels,
            split_values,
            model_config,
            _training_config(),
            epochs_override=2,
        )

        self.assertEqual(fitted.model.encoder_hidden_dimensions, (8, 6))
        self.assertEqual(fitted.model.classifier_hidden_dimensions, (6, 4))
        self.assertEqual(len(fitted.history), 2)


class MetricSliceTests(unittest.TestCase):
    """Verify grouped metrics align by row position rather than external DataFrame indices."""

    def test_non_contiguous_metadata_index_keeps_array_alignment(self) -> None:
        metadata = pd.DataFrame(
            {
                "dataset": ["alpha", "beta", "alpha", "beta"],
                "house_id": [1, 2, 1, 3],
                "source_appliance": ["heater", "microwave", "heater", "kettle"],
            },
            index=[101, 17, 303, 42],
        )
        labels = np.asarray([1, 0, 1, 0])
        probabilities = np.asarray([0.9, 0.1, 0.8, 0.2])

        metrics = _metrics_by_slice(metadata, labels, probabilities, threshold=0.5)

        self.assertEqual(metrics["overall"]["accuracy"], 1.0)
        self.assertEqual(metrics["per_dataset"]["alpha"]["sample_count"], 2)
        self.assertEqual(metrics["per_dataset"]["beta"]["negative_count"], 2)
        self.assertEqual(metrics["per_house"]["beta:3"]["sample_count"], 1)
        self.assertEqual(metrics["per_appliance"]["heater"]["positive_count"], 2)


class MvpRunHelperTests(unittest.TestCase):
    """Verify the smoke run writes and reloads a minimal checkpoint."""

    def test_smoke_run_writes_reloadable_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            _write_synthetic_sources(project_root)
            (project_root / "taxonomy.yaml").write_text(
                "schema_version: test\n", encoding="utf-8"
            )
            config_path = project_root / "mvp.toml"
            config_path.write_text(
                """
[paths]
refit_output_dir = "prepared_refit"
ukdale_house5_features = "prepared_ukdale/house5.csv"
ukdale_house2_features = "prepared_ukdale/house2.csv"
combined_output_dir = "combined"
run_output_root = "runs"
taxonomy = "taxonomy.yaml"

[split]
train = ["refit:1"]
validation = ["uk_dale:5"]
test = ["uk_dale:2"]
known_constraint = "测试说明字段"

[model]
encoder_hidden_dimension = [8, 6]
embedding_dimension = 4
classifier_hidden_dimension = [6, 4]
dropout_probability = 0.0
feature_dropout_enabled = false
group_dropout_probability = 0.0
feature_dropout_probability = 0.0

[training]
seed = 20260820
epochs = 2
smoke_epochs = 2
batch_size = 2
learning_rate = 0.005
weight_decay = 0.0
early_stopping_patience = 4

[evaluation]
missing_feature_seed = 20260820
missing_feature_fractions = [0.25]
""".strip()
                + "\n",
                encoding="utf-8",
            )

            result = run_grouped_mlp_mvp(
                project_root,
                config_path,
                smoke_test=True,
                run_id="smoke_test",
            )
            run_directory = Path(result["run_dir"])
            checkpoint = torch.load(
                run_directory / "best_checkpoint.pt",
                map_location="cpu",
                weights_only=False,
            )
            checkpoint_config = checkpoint["model_config"]
            model = GroupedFingerprintMLP(
                checkpoint_config["group_dimensions"],
                encoder_hidden_dimension=checkpoint_config[
                    "encoder_hidden_dimension"
                ],
                embedding_dimension=checkpoint_config["embedding_dimension"],
                classifier_hidden_dimension=checkpoint_config[
                    "classifier_hidden_dimension"
                ],
                dropout_probability=checkpoint_config["dropout_probability"],
                feature_dropout_enabled=False,
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            with torch.no_grad():
                logits, gates = model(
                    torch.zeros(1, len(FEATURE_COLUMNS)),
                    torch.ones(1, len(MASK_COLUMNS)),
                )

            self.assertEqual(tuple(logits.shape), (1,))
            self.assertEqual(tuple(gates.shape), (1, 2))
            self.assertTrue((run_directory / "metrics.json").is_file())
            self.assertTrue((run_directory / "predictions.csv").is_file())
            self.assertFalse(result["manifest"]["test_used_for_selection"])


if __name__ == "__main__":
    unittest.main()
