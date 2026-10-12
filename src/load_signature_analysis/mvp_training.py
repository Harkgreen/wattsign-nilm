"""Train and evaluate the missingness-aware grouped MLP on UK-DALE and REFIT."""

from __future__ import annotations

from .data_paths import resolve_project_path

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import random
import re
import shutil
import tempfile
import tomllib
from typing import Mapping

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .event_features import FEATURE_COLUMNS, GROUP_FEATURES, MASK_COLUMNS
from .grouped_mlp import GroupedFingerprintMLP
from .mvp_preprocessing import (
    ExplicitHouseSplit,
    MaskedStandardizer,
    compute_binary_metrics,
    select_validation_threshold,
)


def _resolve_project_path(project_root: Path, value: str | Path) -> Path:
    return resolve_project_path(project_root, str(value))


def _json_dump(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _sha256_file(path: Path) -> str:
    """Stream a file hash instead of loading a large feature table into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _describe_file(path: Path, project_root: Path) -> dict[str, object]:
    resolved = path.resolve()
    try:
        display_path = resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        display_path = str(resolved)
    return {
        "path": display_path,
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _load_prepared_sources(
    project_root: Path,
    paths: Mapping[str, object],
) -> pd.DataFrame:
    sources = (
        ("refit", paths["refit_output_dir"], "event_features.csv"),
        ("uk_dale", paths["ukdale_house5_features"], None),
        ("uk_dale", paths["ukdale_house2_features"], None),
    )
    tables: list[pd.DataFrame] = []
    for expected_dataset, configured_path, child in sources:
        path = _resolve_project_path(project_root, str(configured_path))
        if child is not None:
            path = path / child
        if not path.exists():
            raise FileNotFoundError(path)
        table = pd.read_csv(path, encoding="utf-8-sig")
        if set(table["dataset"].astype(str).unique()) != {expected_dataset}:
            raise ValueError(f"{path} 的 dataset 与配置不一致")
        tables.append(table)
    combined = pd.concat(tables, ignore_index=True)
    if combined["event_id"].duplicated().any():
        duplicates = combined.loc[
            combined["event_id"].duplicated(), "event_id"
        ].head().tolist()
        raise ValueError(f"合并特征表存在重复事件 ID：{duplicates}")
    missing_columns = set((*FEATURE_COLUMNS, *MASK_COLUMNS)) - set(combined.columns)
    if missing_columns:
        raise ValueError(f"合并特征表缺少模型字段：{sorted(missing_columns)}")
    values = combined[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    masks = combined[list(MASK_COLUMNS)].to_numpy(dtype=np.int8)
    if not np.isin(masks, (0, 1)).all():
        raise ValueError("原始 feature mask 只能包含 0 和 1")
    if not np.array_equal(np.isfinite(values), masks.astype(bool)):
        raise ValueError("原始特征的有限性与 feature mask 不一致")
    return combined


def _prepared_source_paths(
    project_root: Path,
    paths: Mapping[str, object],
) -> tuple[Path, ...]:
    return (
        _resolve_project_path(project_root, str(paths["refit_output_dir"]))
        / "event_features.csv",
        _resolve_project_path(project_root, str(paths["ukdale_house5_features"])),
        _resolve_project_path(project_root, str(paths["ukdale_house2_features"])),
    )


def _prepared_input_artifact_paths(
    project_root: Path,
    paths: Mapping[str, object],
) -> tuple[Path, ...]:
    """Return training inputs and available upstream manifests for byte-level run provenance."""

    feature_paths = _prepared_source_paths(project_root, paths)
    manifests = tuple(
        manifest
        for feature_path in feature_paths
        if (manifest := feature_path.parent / "manifest.json").exists()
    )
    return (*feature_paths, *manifests)


def prepare_combined_features(
    project_root: Path,
    raw_config: Mapping[str, object],
) -> tuple[pd.DataFrame, ExplicitHouseSplit]:
    """Merge both datasets and apply the frozen household split without fitting statistics."""

    paths = raw_config["paths"]
    combined = _load_prepared_sources(project_root, paths)
    split_config = raw_config["split"]
    # Split tables may contain explanatory fields; the allocator accepts only the three contract-defined sets.
    split = ExplicitHouseSplit.from_mapping(
        {
            name: split_config[name]
            for name in ("train", "validation", "test")
        }
    )
    assigned = split.assign(
        combined,
        require_all_houses=True,
        require_both_classes=True,
    )
    output_dir = _resolve_project_path(project_root, str(paths["combined_output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    split_counts = (
        assigned.groupby(["split", "dataset", "house_id", "label"])
        .size()
        .rename("event_count")
        .reset_index()
    )
    generation_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    manifest = {
        "schema_version": "0.3.0-combined-mvp",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generation_id": generation_id,
        "feature_count": len(FEATURE_COLUMNS),
        "mask_count": len(MASK_COLUMNS),
        "event_count": int(len(assigned)),
        "house_split": split.to_dict(),
        "known_split_constraint": str(raw_config["split"].get("known_constraint", "")),
        "split_counts": split_counts.to_dict(orient="records"),
        "input_artifacts": [
            _describe_file(path, project_root)
            for path in _prepared_input_artifact_paths(project_root, paths)
        ],
        "model_input_contract": {
            "feature_columns": list(FEATURE_COLUMNS),
            "mask_columns": list(MASK_COLUMNS),
            "metadata_columns_are_excluded": True,
        },
    }
    with tempfile.TemporaryDirectory(
        prefix="combined_mvp_staging_", dir=output_dir.parent
    ) as temporary_directory:
        staging = Path(temporary_directory)
        assigned.to_csv(
            staging / "event_features_with_split.csv",
            index=False,
            encoding="utf-8-sig",
        )
        _json_dump(staging / "splits.json", split.to_dict())
        manifest["output_artifacts"] = [
            _describe_file(staging / filename, staging)
            for filename in ("event_features_with_split.csv", "splits.json")
        ]
        _json_dump(staging / "manifest.json", manifest)
        # Commit the manifest last; interruption while replacing the first two files makes the old hashes mismatch.
        for filename in (
            "event_features_with_split.csv",
            "splits.json",
            "manifest.json",
        ):
            (staging / filename).replace(output_dir / filename)
    return assigned, split


def _make_loader(
    values: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(values[indices].astype(np.float32, copy=False)),
        torch.from_numpy(masks[indices].astype(np.float32, copy=False)),
        torch.from_numpy(labels[indices].astype(np.float32, copy=False)),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


@dataclass
class FittedModel:
    """Save the model at the best validation loss and its training audit."""

    model: GroupedFingerprintMLP
    history: list[dict[str, float | int]]
    best_epoch: int
    best_validation_loss: float
    pos_weight: float


def fit_grouped_mlp(
    values: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    split_values: np.ndarray,
    model_config: Mapping[str, object],
    training_config: Mapping[str, object],
    *,
    epochs_override: int | None = None,
) -> FittedModel:
    """Update parameters only on train data and select checkpoints only by validation loss."""

    seed = int(training_config["seed"])
    _set_reproducible_seed(seed)
    split_values = np.asarray(split_values).astype(str)
    train_indices = np.flatnonzero(split_values == "train")
    validation_indices = np.flatnonzero(split_values == "validation")
    if train_indices.size == 0 or validation_indices.size == 0:
        raise ValueError("训练和验证 split 都必须非空")
    train_labels = labels[train_indices].astype(np.int64)
    negative_count = int(np.count_nonzero(train_labels == 0))
    positive_count = int(np.count_nonzero(train_labels == 1))
    if negative_count == 0 or positive_count == 0:
        raise ValueError("训练集必须同时包含正负类")
    pos_weight = negative_count / positive_count

    group_dimensions = {
        group: len(features) for group, features in GROUP_FEATURES.items()
    }
    model = GroupedFingerprintMLP(
        group_dimensions,
        encoder_hidden_dimension=model_config["encoder_hidden_dimension"],
        embedding_dimension=int(model_config["embedding_dimension"]),
        classifier_hidden_dimension=model_config["classifier_hidden_dimension"],
        dropout_probability=float(model_config["dropout_probability"]),
        feature_dropout_enabled=bool(model_config["feature_dropout_enabled"]),
        group_dropout_probability=float(model_config["group_dropout_probability"]),
        feature_dropout_probability=float(model_config["feature_dropout_probability"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(float(pos_weight), dtype=torch.float32)
    )
    batch_size = int(training_config["batch_size"])
    train_loader = _make_loader(
        values,
        masks,
        labels,
        train_indices,
        batch_size,
        True,
        seed,
    )
    validation_loader = _make_loader(
        values,
        masks,
        labels,
        validation_indices,
        batch_size,
        False,
        seed,
    )
    epochs = (
        int(epochs_override)
        if epochs_override is not None
        else int(training_config["epochs"])
    )
    patience_limit = int(training_config["early_stopping_patience"])
    history: list[dict[str, float | int]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_validation_loss = np.inf
    best_epoch = 0
    patience = 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch_values, batch_masks, batch_labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(batch_values, batch_masks)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * int(batch_labels.numel())
            train_count += int(batch_labels.numel())

        model.eval()
        validation_loss_sum = 0.0
        validation_count = 0
        with torch.no_grad():
            for batch_values, batch_masks, batch_labels in validation_loader:
                logits, _ = model(batch_values, batch_masks)
                loss = criterion(logits, batch_labels)
                validation_loss_sum += float(loss.item()) * int(batch_labels.numel())
                validation_count += int(batch_labels.numel())
        train_loss = train_loss_sum / train_count
        validation_loss = validation_loss_sum / validation_count
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_validation_loss - 1e-8:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
            if patience >= patience_limit:
                break
    if best_state is None:
        raise RuntimeError("训练未产生有效 checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return FittedModel(
        model=model,
        history=history,
        best_epoch=best_epoch,
        best_validation_loss=float(best_validation_loss),
        pos_weight=float(pos_weight),
    )


def _predict(
    model: GroupedFingerprintMLP,
    values: np.ndarray,
    masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        logits, gates = model(
            torch.from_numpy(values.astype(np.float32, copy=False)),
            torch.from_numpy(masks.astype(np.float32, copy=False)),
        )
        probabilities = torch.sigmoid(logits)
    return probabilities.cpu().numpy(), gates.cpu().numpy()


def _metrics_by_slice(
    metadata: pd.DataFrame,
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    metadata = metadata.reset_index(drop=True)
    if len(metadata) != len(labels) or len(labels) != len(probabilities):
        raise ValueError("metadata、labels 与 probabilities 必须等长")
    result: dict[str, object] = {
        "overall": compute_binary_metrics(labels, probabilities, threshold),
        "per_dataset": {},
        "per_house": {},
        "per_appliance": {},
    }
    for dataset, group in metadata.groupby("dataset", sort=True):
        positions = group.index.to_numpy(dtype=np.int64)
        result["per_dataset"][str(dataset)] = compute_binary_metrics(
            labels[positions], probabilities[positions], threshold
        )
    for (dataset, house_id), group in metadata.groupby(
        ["dataset", "house_id"], sort=True
    ):
        positions = group.index.to_numpy(dtype=np.int64)
        key = f"{dataset}:{int(house_id)}"
        result["per_house"][key] = compute_binary_metrics(
            labels[positions], probabilities[positions], threshold
        )
    for appliance, group in metadata.groupby("source_appliance", sort=True):
        positions = group.index.to_numpy(dtype=np.int64)
        result["per_appliance"][str(appliance)] = compute_binary_metrics(
            labels[positions], probabilities[positions], threshold
        )
    return result


def _summarise_gate_weights(
    gates: np.ndarray,
    masks: np.ndarray,
    group_order: tuple[str, ...],
) -> dict[str, object]:
    """Summarize gate weights and naturally available groups for debugging, without causal interpretation."""

    if gates.ndim != 2 or gates.shape[1] != len(group_order):
        raise ValueError("gate 权重形状与 group_order 不一致")
    if masks.ndim != 2 or masks.shape[0] != gates.shape[0]:
        raise ValueError("mask 与 gate 权重样本数不一致")
    boundaries = np.cumsum(
        [0, *(len(GROUP_FEATURES[group]) for group in group_order)]
    )
    group_available = np.column_stack(
        [
            masks[:, boundaries[index] : boundaries[index + 1]].any(axis=1)
            for index in range(len(group_order))
        ]
    )
    return {
        "interpretation": "仅用于调试缺失路由，不代表因果特征贡献。",
        "groups": {
            group: {
                "mean_weight": float(np.mean(gates[:, index])),
                "median_weight": float(np.median(gates[:, index])),
                "mean_weight_when_available": (
                    float(np.mean(gates[group_available[:, index], index]))
                    if group_available[:, index].any()
                    else None
                ),
                "median_weight_when_available": (
                    float(np.median(gates[group_available[:, index], index]))
                    if group_available[:, index].any()
                    else None
                ),
                "available_count": int(np.count_nonzero(group_available[:, index])),
                "missing_count": int(np.count_nonzero(~group_available[:, index])),
            }
            for index, group in enumerate(group_order)
        },
    }


def _render_run_report(
    run_manifest: Mapping[str, object],
    metrics: Mapping[str, object],
) -> str:
    """Summarize the grouped-MLP experiment, held-out metrics and missing-feature tests."""

    overall = metrics["test"]["overall"]
    threshold = metrics["validation_threshold"]
    train_counts = metrics["train_class_counts"]
    lines = [
        "# UK-DALE + REFIT 缺失感知分组 MLP MVP 报告",
        "",
        "## 实验设置",
        "",
        f"- 运行 ID：`{run_manifest['run_id']}`",
        f"- 事件数：{run_manifest['event_count']}",
        "- 模型输入：27 个 aggregate 派生特征值 + 27 个逐特征 mask；元数据与分表真实持续时间不入模。",
        "- scaler 只在训练户拟合；checkpoint 只按验证损失选择；分类阈值只在验证集选择。",
        "- 测试集未参与 scaler、checkpoint 或阈值选择。",
        f"- 最佳 epoch：{int(metrics['best_epoch'])}；训练负/正样本：{int(train_counts['negative'])}/{int(train_counts['positive'])}。",
        "",
        "## 测试集结果",
        "",
        f"- 验证集冻结阈值：{float(threshold['threshold']):.6f}",
        f"- 样本数：{int(overall['sample_count'])}",
        f"- Precision：{float(overall['precision']):.4f}",
        f"- Recall：{float(overall['recall']):.4f}",
        f"- F1：{float(overall['f1']):.4f}",
        f"- Balanced accuracy：{float(overall['balanced_accuracy']):.4f}",
        f"- ROC-AUC：{float(overall['roc_auc']):.4f}",
        f"- PR-AUC：{float(overall['pr_auc']):.4f}",
        f"- Confusion matrix [[TN, FP], [FN, TP]]：`{overall['confusion_matrix']}`",
        "",
        "### 分数据集",
        "",
        "| 数据集 | n | Precision | Recall | F1 | Balanced accuracy | PR-AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, values in metrics["test"]["per_dataset"].items():
        lines.append(
            f"| {dataset} | {int(values['sample_count'])} | "
            f"{float(values['precision']):.4f} | {float(values['recall']):.4f} | "
            f"{float(values['f1']):.4f} | {float(values['balanced_accuracy']):.4f} | "
            f"{float(values['pr_auc']):.4f} |"
        )
    lines.extend(
        [
            "",
            "### 缺失消融",
            "",
            "| 模式 | n | F1 | Balanced accuracy | PR-AUC | 全组缺失排除 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for values in metrics["missing_feature_evaluation"]:
        if int(values["sample_count"]) == 0:
            lines.append(
                f"| {values['mode']} | 0 | — | — | — | "
                f"{int(values['excluded_all_groups_missing'])} |"
            )
            continue
        lines.append(
            f"| {values['mode']} | {int(values['sample_count'])} | "
            f"{float(values['f1']):.4f} | "
            f"{float(values['balanced_accuracy']):.4f} | "
            f"{float(values['pr_auc']):.4f} | "
            f"{int(values['excluded_all_groups_missing'])} |"
        )
    lines.extend(
        [
            "",
        "## 已知限制",
        "",
        "- 严格有功 W 主池中，可靠 UK-DALE 正类户主要只有 H2/H5，因此 UK-DALE 只出现在 train/test，validation 由 REFIT 构成。",
        "- 标签由分表通道阈值事件构造，仍会受到采样不确定度、未纳入通道活动和公开数据清洗方式影响。",
        "- gate 权重只用于检查缺失组路由是否正确，不能解释为因果贡献。",
        "- 模型使用手工统计特征。与时序模型比较时，可复用事件、住宅划分和评测口径，以分离模型结构与数据准备的影响。",
        "",
        ]
    )
    return "\n".join(lines)


def _masked_evaluation(
    model: GroupedFingerprintMLP,
    values: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    mode: str,
    seed: int,
) -> dict[str, object]:
    evaluation_masks = masks.astype(bool, copy=True)
    if mode == "power_only":
        evaluation_masks[:, len(GROUP_FEATURES["power"]) :] = False
    elif mode == "temporal_only":
        evaluation_masks[:, : len(GROUP_FEATURES["power"])] = False
    elif mode.startswith("random_"):
        fraction = float(mode.split("_", maxsplit=1)[1])
        rng = np.random.default_rng(seed)
        evaluation_masks &= rng.random(evaluation_masks.shape) >= fraction
    elif mode != "full":
        raise ValueError(f"未知缺失评估模式：{mode}")

    available = evaluation_masks.any(axis=1)
    excluded_negative = int(np.count_nonzero(labels[~available] == 0))
    excluded_positive = int(np.count_nonzero(labels[~available] == 1))
    if not available.any():
        return {
            "mode": mode,
            "sample_count": 0,
            "excluded_all_groups_missing": int(len(labels)),
            "excluded_negative_count": excluded_negative,
            "excluded_positive_count": excluded_positive,
            "reason": "遮挡后所有样本均为全组缺失，模型契约不允许推理。",
        }
    evaluation_values = values[available].copy()
    evaluation_masks = evaluation_masks[available]
    evaluation_values[~evaluation_masks] = 0.0
    probabilities, gates = _predict(model, evaluation_values, evaluation_masks)
    metrics = compute_binary_metrics(labels[available], probabilities, threshold)
    metrics["mode"] = mode
    metrics["excluded_all_groups_missing"] = int(np.count_nonzero(~available))
    metrics["excluded_negative_count"] = excluded_negative
    metrics["excluded_positive_count"] = excluded_positive
    metrics["mean_power_gate_weight"] = float(np.mean(gates[:, 0]))
    metrics["mean_temporal_gate_weight"] = float(np.mean(gates[:, 1]))
    return metrics


def _smoke_subset(features: pd.DataFrame, maximum_per_split_label: int = 256) -> pd.DataFrame:
    """Apply stable sampling only for end-to-end smoke training, preserving household split membership."""

    keep: list[int] = []
    for _, group in features.groupby(["split", "label"], sort=True):
        if len(group) <= maximum_per_split_label:
            keep.extend(group.index.tolist())
            continue
        positions = np.linspace(
            0, len(group) - 1, maximum_per_split_label, dtype=np.int64
        )
        keep.extend(group.iloc[np.unique(positions)].index.tolist())
    return features.loc[sorted(keep)].reset_index(drop=True)


def _validate_run_id(run_id: str) -> str:
    """Restrict run-directory names to prevent path traversal or unintended nested directories."""

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id):
        raise ValueError(
            "run_id 必须以字母或数字开头，且只能包含字母、数字、点、下划线和连字符"
        )
    if run_id in {".", ".."}:
        raise ValueError("run_id 不能是相对路径标记")
    return run_id


def run_grouped_mlp_mvp(
    project_root: Path,
    config_path: Path,
    *,
    smoke_test: bool = False,
    run_id: str | None = None,
) -> dict[str, object]:
    """Run merging, household splits, training-only standardization, training and testing."""

    project_root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    with config_path.open("rb") as handle:
        raw_config = tomllib.load(handle)
    features, split = prepare_combined_features(project_root, raw_config)
    if smoke_test:
        features = _smoke_subset(features)

    raw_values = features[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    raw_masks = features[list(MASK_COLUMNS)].to_numpy(dtype=np.int8)
    labels = features["label"].to_numpy(dtype=np.int64)
    split_values = features["split"].astype(str).to_numpy()
    train_rows = split_values == "train"
    scaler = MaskedStandardizer(FEATURE_COLUMNS).fit(
        raw_values[train_rows], raw_masks[train_rows]
    )
    values, masks = scaler.transform(
        raw_values, raw_masks, feature_names=FEATURE_COLUMNS
    )
    if not masks.any(axis=1).all():
        raise ValueError("标准化后存在所有特征组均缺失的样本")

    training_config = raw_config["training"]
    fitted = fit_grouped_mlp(
        values,
        masks,
        labels,
        split_values,
        raw_config["model"],
        training_config,
        epochs_override=(
            int(training_config["smoke_epochs"]) if smoke_test else None
        ),
    )
    validation_rows = split_values == "validation"
    validation_probabilities, _ = _predict(
        fitted.model, values[validation_rows], masks[validation_rows]
    )
    threshold = select_validation_threshold(
        labels[validation_rows], validation_probabilities
    )
    test_rows = split_values == "test"
    test_metadata = features.loc[test_rows].reset_index(drop=True)
    test_labels = labels[test_rows]
    test_values = values[test_rows]
    test_masks = masks[test_rows]
    test_probabilities, test_gates = _predict(
        fitted.model, test_values, test_masks
    )
    test_metrics = _metrics_by_slice(
        test_metadata,
        test_labels,
        test_probabilities,
        threshold.threshold,
    )

    ablations = [
        _masked_evaluation(
            fitted.model,
            test_values,
            test_masks,
            test_labels,
            threshold.threshold,
            mode,
            int(raw_config["evaluation"]["missing_feature_seed"]),
        )
        for mode in (
            "full",
            "power_only",
            "temporal_only",
            *(
                f"random_{float(fraction):.2f}"
                for fraction in raw_config["evaluation"][
                    "missing_feature_fractions"
                ]
            ),
        )
    ]
    predicted_labels = (test_probabilities >= threshold.threshold).astype(np.int64)
    predictions = test_metadata[
        [
            "event_id",
            "dataset",
            "house_id",
            "event_timestamp",
            "event_utc",
            "source_appliance",
            "label",
        ]
    ].copy()
    predictions = predictions.rename(columns={"label": "true_label"})
    predictions["predicted_probability"] = test_probabilities
    predictions["predicted_label"] = predicted_labels
    for group_index, group_name in enumerate(fitted.model.group_order):
        predictions[f"{group_name}_gate_weight"] = test_gates[:, group_index]

    parameter_count = int(
        sum(parameter.numel() for parameter in fitted.model.parameters())
    )
    metrics = {
        "schema_version": "0.4.0-grouped-mlp",
        "smoke_test": smoke_test,
        "best_epoch": fitted.best_epoch,
        "best_validation_loss": fitted.best_validation_loss,
        "training_history": fitted.history,
        "train_class_counts": {
            "negative": int(np.count_nonzero(labels[train_rows] == 0)),
            "positive": int(np.count_nonzero(labels[train_rows] == 1)),
            "pos_weight": fitted.pos_weight,
        },
        "validation_threshold": threshold.to_dict(),
        "test": test_metrics,
        "test_gate_statistics": _summarise_gate_weights(
            test_gates,
            test_masks,
            fitted.model.group_order,
        ),
        "missing_feature_evaluation": ablations,
    }
    feature_schema = {
        "feature_columns": list(FEATURE_COLUMNS),
        "mask_columns": list(MASK_COLUMNS),
        "groups": {
            group: [f"{group}__{feature}" for feature in group_features]
            for group, group_features in GROUP_FEATURES.items()
        },
        "metadata_excluded_from_model": [
            column
            for column in features.columns
            if column not in {*FEATURE_COLUMNS, *MASK_COLUMNS}
        ],
    }
    model_config = {
        **dict(raw_config["model"]),
        "group_dimensions": {
            group: len(group_features)
            for group, group_features in GROUP_FEATURES.items()
        },
        "parameter_count": parameter_count,
        "group_order": list(fitted.model.group_order),
    }
    generated_id = (
        ("smoke_" if smoke_test else "mvp_")
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    run_id = _validate_run_id(run_id or generated_id)
    run_root = _resolve_project_path(
        project_root, str(raw_config["paths"]["run_output_root"])
    )
    run_root.mkdir(parents=True, exist_ok=True)
    final_dir = (run_root / run_id).resolve()
    if final_dir.parent != run_root.resolve():
        raise ValueError("run_id 解析后逃出了运行输出目录")
    if final_dir.exists():
        raise FileExistsError(f"运行目录已存在：{final_dir}")

    run_manifest = {
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "smoke_test": smoke_test,
        "datasets": ["uk_dale", "refit"],
        "event_count": int(len(features)),
        "house_split": split.to_dict(),
        "known_split_constraint": str(
            raw_config["split"].get("known_constraint", "")
        ),
        "model_parameter_count": parameter_count,
        "input_dimension_values": len(FEATURE_COLUMNS),
        "input_dimension_masks": len(MASK_COLUMNS),
        "checkpoint_selection": "minimum validation BCEWithLogitsLoss",
        "threshold_selection": threshold.selection_rule,
        "test_used_for_selection": False,
        "checkpoint_purpose": "inference_only，不包含 optimizer/RNG 状态，不能精确续训。",
        "format_note": "CSV + JSON + joblib/pt；当前项目未引入 pyarrow。",
        "input_artifacts": [
            _describe_file(path, project_root)
            for path in _prepared_input_artifact_paths(
                project_root, raw_config["paths"]
            )
        ],
        "configuration_artifacts": [
            _describe_file(config_path, project_root),
            _describe_file(
                _resolve_project_path(
                    project_root, str(raw_config["paths"]["taxonomy"])
                ),
                project_root,
            ),
        ],
        "implementation_artifacts": [
            _describe_file(path, project_root)
            for path in (
                Path(__file__),
                Path(__file__).with_name("grouped_mlp.py"),
                Path(__file__).with_name("mvp_preprocessing.py"),
                Path(__file__).with_name("event_features.py"),
                Path(__file__).with_name("refit_phase3.py"),
                Path(__file__).with_name("data_adapters.py"),
                Path(__file__).with_name("event_detection.py"),
                Path(__file__).with_name("ukdale_phase2.py"),
                *((project_root / "uv.lock",) if (project_root / "uv.lock").exists() else ()),
            )
        ],
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    staging = Path(tempfile.mkdtemp(prefix=f"{run_id}_staging_", dir=run_root))
    try:
        predictions.to_csv(
            staging / "predictions.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(ablations).to_csv(
            staging / "missing_feature_evaluation.csv",
            index=False,
            encoding="utf-8-sig",
        )
        _json_dump(staging / "metrics.json", metrics)
        _json_dump(staging / "model_config.json", model_config)
        _json_dump(staging / "feature_schema.json", feature_schema)
        _json_dump(staging / "scaler.json", scaler.to_dict())
        _json_dump(staging / "splits.json", split.to_dict())
        (staging / "run_report.md").write_text(
            _render_run_report(run_manifest, metrics),
            encoding="utf-8",
        )
        joblib.dump(scaler, staging / "scaler.joblib")
        torch.save(
            {
                "model_state_dict": fitted.model.state_dict(),
                "model_config": model_config,
                "feature_schema": feature_schema,
                "validation_threshold": threshold.to_dict(),
                "pos_weight": fitted.pos_weight,
            },
            staging / "best_checkpoint.pt",
        )
        shutil.copyfile(config_path, staging / "config_snapshot.toml")
        shutil.copyfile(
            _resolve_project_path(project_root, str(raw_config["paths"]["taxonomy"])),
            staging / "taxonomy_snapshot.yaml",
        )
        output_filenames = sorted(
            path.name for path in staging.iterdir() if path.is_file()
        )
        run_manifest["output_artifacts"] = [
            _describe_file(staging / filename, staging)
            for filename in output_filenames
        ]
        # Write the manifest last so its hashes describe one complete generation of run artifacts.
        _json_dump(staging / "run_manifest.json", run_manifest)
        staging.replace(final_dir)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return {"run_dir": final_dir, "manifest": run_manifest, "metrics": metrics}
