"""Feature-cache construction and grouped PLAID baseline evaluation."""

from __future__ import annotations

import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FULL_FEATURES, SHAPE_FEATURES, extract_electrical_features
from .plaid import load_plaid_records, read_waveform_tail


def build_plaid_feature_cache(
    plaid_root: Path,
    output_csv: Path,
    max_rows: int = 30_000,
    limit: int | None = None,
) -> pd.DataFrame:
    records = load_plaid_records(plaid_root)
    if limit is not None:
        records = records[:limit]
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for index, record in enumerate(records, start=1):
        current, voltage = read_waveform_tail(record.csv_path, max_rows=max_rows)
        features = extract_electrical_features(
            current=current,
            voltage=voltage,
            sample_rate_hz=record.sampling_frequency_hz,
        )
        rows.append(
            {
                "sample_id": record.sample_id,
                "release": record.release,
                "appliance_type": record.appliance_type,
                "load_type": record.load_type,
                "status": record.status,
                "location": record.location,
                "brand": record.brand,
                "model_number": record.model_number,
                "resistive_label": record.resistive_label,
                "heating_active_proxy_label": record.heating_active_proxy_label,
                **features,
            }
        )
        if index % 100 == 0 or index == len(records):
            elapsed = time.perf_counter() - started
            print(f"features: {index}/{len(records)} samples ({elapsed:.1f}s)", flush=True)
    frame = pd.DataFrame(rows)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    return frame


def _model_definitions(random_state: int) -> dict[str, tuple[tuple[str, ...], object]]:
    logistic = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=5_000, class_weight="balanced", random_state=random_state),
    )
    forest = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
    )
    return {
        "full_logistic": (FULL_FEATURES, logistic),
        "full_random_forest": (FULL_FEATURES, forest),
        "shape_random_forest": (SHAPE_FEATURES, clone(forest)),
    }


def _binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    predictions = (probabilities >= 0.5).astype(np.int8)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    return {
        "average_precision": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "confusion_matrix_tn_fp_fn_tp": [
            int(matrix[0, 0]),
            int(matrix[0, 1]),
            int(matrix[1, 0]),
            int(matrix[1, 1]),
        ],
    }


def evaluate_grouped_baselines(
    frame: pd.DataFrame,
    output_dir: Path,
    target_column: str = "resistive_label",
    random_state: int = 42,
    folds: int = 5,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = frame[target_column].to_numpy(dtype=np.int8)
    groups = frame["location"].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=folds)
    results: dict[str, object] = {
        "target": target_column,
        "samples": int(len(frame)),
        "positive_samples": int(labels.sum()),
        "positive_fraction": float(labels.mean()),
        "groups": int(pd.Series(groups).nunique()),
        "split": f"GroupKFold({folds}) by location",
        "models": {},
    }

    for model_name, (feature_names, model) in _model_definitions(random_state).items():
        design = frame.loc[:, list(feature_names)].replace([np.inf, -np.inf], np.nan)
        probabilities = np.full(len(frame), np.nan, dtype=np.float64)
        fold_summaries: list[dict[str, object]] = []
        for fold_number, (train_indices, test_indices) in enumerate(
            splitter.split(design, labels, groups), start=1
        ):
            fitted = clone(model)
            fitted.fit(design.iloc[train_indices], labels[train_indices])
            fold_probabilities = fitted.predict_proba(design.iloc[test_indices])[:, 1]
            probabilities[test_indices] = fold_probabilities
            fold_metrics = _binary_metrics(labels[test_indices], fold_probabilities)
            fold_metrics.update(
                {
                    "fold": fold_number,
                    "train_samples": int(len(train_indices)),
                    "test_samples": int(len(test_indices)),
                    "test_groups": sorted(set(groups[test_indices].tolist())),
                }
            )
            fold_summaries.append(fold_metrics)
        metrics = _binary_metrics(labels, probabilities)
        metrics["features"] = list(feature_names)
        metrics["folds"] = fold_summaries
        results["models"][model_name] = metrics
        frame[f"oof_probability_{model_name}"] = probabilities

        final_model = clone(model)
        final_model.fit(design, labels)
        joblib.dump(
            {"model": final_model, "features": feature_names, "target": target_column},
            output_dir / f"{model_name}.joblib",
        )

    best_model_name = max(
        results["models"],
        key=lambda name: results["models"][name]["average_precision"],
    )
    results["best_model"] = best_model_name

    prediction_columns = [
        "sample_id",
        "release",
        "location",
        "appliance_type",
        "load_type",
        "status",
        target_column,
        *[f"oof_probability_{name}" for name in results["models"]],
    ]
    frame.loc[:, prediction_columns].to_csv(output_dir / "oof_predictions.csv", index=False)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    return results
