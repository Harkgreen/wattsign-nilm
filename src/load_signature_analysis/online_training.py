"Weighted grouped-MLP training, evaluation and artifact persistence for online_causal_v1_1.\n\nThis entry point trains the causal online classifier rather than the offline event classifier. Training reads only manifest.json, causal_features.csv and the taxonomy snapshot from validated sample packages; it never loads multi-GB aggregate/candidate audit CSVs or modifies the upstream sample directory."

from __future__ import annotations

from .data_paths import resolve_project_path

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import platform
from pathlib import Path
import random
import re
import shutil
import tempfile
import tomllib
from typing import Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .grouped_mlp import GroupedFingerprintMLP
from .mvp_preprocessing import (
    ExplicitHouseSplit,
    HouseKey,
    MaskedStandardizer,
    compute_binary_metrics,
)
from .online_features import (
    FEATURE_REGISTRY_VERSION,
    ONLINE_FEATURE_COLUMNS,
    ONLINE_GROUP_FEATURES,
    ONLINE_MASK_COLUMNS,
    SUPPORTED_HORIZONS_SECONDS,
)
from .online_samples import SCHEMA_VERSION as ONLINE_SAMPLE_SCHEMA_VERSION
from .online_samples import SEQUENCE_CONTRACT_VERSION


TRAINING_SCHEMA_VERSION = "online_weighted_grouped_mlp_v1"
THRESHOLD_SELECTION_RULE = (
    "max_sample_weighted_f1_then_weighted_balanced_accuracy_then_weighted_recall_"
    "then_closest_to_0.5_then_lower_threshold"
)
HIGH_CONFIDENCE_LABEL_TYPES = frozenset({"core_positive", "hard_negative"})
SPLIT_NAMES = ("train", "validation", "test")

ONLINE_METADATA_COLUMNS = (
    "candidate_id",
    "dataset",
    "house_id",
    "split",
    "candidate_onset_s",
    "candidate_detected_at_s",
    "horizon_seconds",
    "label",
    "label_type",
    "label_confidence",
    "sample_weight",
    "source_appliance",
    "accepted",
)


@dataclass(frozen=True)
class OnlineFeatureTable:
    "Validated scalar features, masks, labels and audit metadata for one horizon."

    metadata: pd.DataFrame
    values: np.ndarray
    masks: np.ndarray
    labels: np.ndarray
    sample_weights: np.ndarray
    split_values: np.ndarray
    horizon_seconds: int
    input_manifest: Mapping[str, object]

    def __len__(self) -> int:
        return int(self.labels.size)

    def subset(self, positions: np.ndarray | Sequence[int]) -> OnlineFeatureTable:
        "Select a stable positional subset and reset its metadata index."

        indices = np.asarray(positions, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError("positions 必须是一维位置索引")
        if indices.size and (indices.min() < 0 or indices.max() >= len(self)):
            raise IndexError("positions 超出 OnlineFeatureTable 范围")
        return OnlineFeatureTable(
            metadata=self.metadata.iloc[indices].reset_index(drop=True),
            values=self.values[indices].copy(),
            masks=self.masks[indices].copy(),
            labels=self.labels[indices].copy(),
            sample_weights=self.sample_weights[indices].copy(),
            split_values=self.split_values[indices].copy(),
            horizon_seconds=self.horizon_seconds,
            input_manifest=self.input_manifest,
        )


@dataclass(frozen=True)
class WeightedThresholdSelection:
    "A threshold selected only from per-sample validation weights."

    threshold: float
    f1: float
    balanced_accuracy: float
    recall: float
    candidate_count: int
    sample_count: int
    sample_weight_sum: float
    selection_rule: str = THRESHOLD_SELECTION_RULE

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "threshold": self.threshold,
            "f1": self.f1,
            "balanced_accuracy": self.balanced_accuracy,
            "recall": self.recall,
            "candidate_count": self.candidate_count,
            "sample_count": self.sample_count,
            "sample_weight_sum": self.sample_weight_sum,
            "selection_rule": self.selection_rule,
        }


@dataclass
class FittedOnlineModel:
    "The trained model state after restoring the best validation checkpoint."

    model: GroupedFingerprintMLP
    history: list[dict[str, float | int]]
    best_epoch: int
    best_validation_loss: float
    pos_weight: float
    train_positive_weight: float
    train_negative_weight: float


def _json_dump(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _sha256_file(path: Path) -> str:
    "Stream a file digest rather than loading the approximately 92 MiB feature table at once."

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _display_path(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _describe_file(path: Path, project_root: Path) -> dict[str, object]:
    resolved = path.resolve()
    return {
        "path": _display_path(resolved, project_root),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _describe_captured_bytes(
    path: Path,
    payload: bytes,
    project_root: Path,
) -> dict[str, object]:
    "Describe configuration/snapshot bytes already read into memory to remove read/hash races."

    return {
        "path": _display_path(path, project_root),
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _assert_lineage_unchanged(
    snapshots: Sequence[tuple[Path, Mapping[str, object]]],
) -> None:
    "Before atomic commit, verify inputs, configuration and implementation files stayed unchanged throughout training."

    changed: list[str] = []
    for path, description in snapshots:
        resolved = path.resolve()
        if not resolved.is_file():
            changed.append(f"{resolved}（已消失）")
            continue
        if (
            resolved.stat().st_size != int(description["size_bytes"])
            or _sha256_file(resolved) != str(description["sha256"])
        ):
            changed.append(str(resolved))
    if changed:
        raise RuntimeError(
            "训练期间以下 lineage 文件发生变化，拒绝提交可能错误归因的运行："
            f"{changed}"
        )


def _resolve_project_path(project_root: Path, configured: str | Path) -> Path:
    return resolve_project_path(project_root, str(configured))


def _verify_preregistration_for_run(project_root, preregistration_path, horizon_seconds, config_path):
    """Reject retired experiment protocols; ordinary H20 fitting remains explicit."""
    if preregistration_path is not None or horizon_seconds in {40, 60, 120}:
        raise ValueError("Historical multi-horizon fitting is not a public entry point. Use an explicit new experiment configuration.")
    return None, None



def _coerce_binary_vector(values: Sequence[object], *, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError(f"{name} 必须是非空一维数组")
    try:
        numeric = raw.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} 只能包含 0 或 1") from error
    if not np.all(np.isfinite(numeric)) or not np.all(
        (numeric == 0.0) | (numeric == 1.0)
    ):
        raise ValueError(f"{name} 只能包含 0 或 1")
    return numeric.astype(np.int64)


def _coerce_weights(values: Sequence[object], *, name: str = "sample_weights") -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError(f"{name} 必须是非空一维数组")
    try:
        numeric = raw.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} 必须是有限正数") from error
    if not np.all(np.isfinite(numeric)) or np.any(numeric <= 0.0):
        raise ValueError(f"{name} 必须是有限正数")
    return numeric


def _validate_probability_vector(
    probabilities: Sequence[object], expected_size: int
) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or values.size != expected_size:
        raise ValueError("probabilities 必须与 labels 等长且为一维数组")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("probabilities 必须是位于 [0, 1] 的有限数")
    return values


def _manifest_version(manifest: Mapping[str, object], key: str) -> str:
    "Accept direct fields or config_snapshot.schema, but require one unambiguous version."

    direct = manifest.get(key)
    snapshot = manifest.get("config_snapshot")
    nested = None
    if isinstance(snapshot, Mapping):
        schema = snapshot.get("schema")
        if isinstance(schema, Mapping):
            nested = schema.get(key)
    values = {str(value) for value in (direct, nested) if value is not None}
    if not values:
        raise ValueError(f"上游 manifest 缺少 {key}")
    if len(values) != 1:
        raise ValueError(f"上游 manifest 的 {key} 自相矛盾：{sorted(values)}")
    return values.pop()


def _validate_input_manifest(manifest: Mapping[str, object]) -> None:
    expected_versions = {
        "schema_version": ONLINE_SAMPLE_SCHEMA_VERSION,
        "feature_registry_version": FEATURE_REGISTRY_VERSION,
        "npz_contract_version": SEQUENCE_CONTRACT_VERSION,
    }
    for key, expected in expected_versions.items():
        actual = _manifest_version(manifest, key)
        if actual != expected:
            raise ValueError(
                f"上游 manifest 的 {key} 不兼容：期望 {expected!r}，实际 {actual!r}"
            )
    split = manifest.get("split")
    if not isinstance(split, Mapping):
        raise ValueError("上游 manifest 缺少冻结 split")
    ExplicitHouseSplit.from_mapping(split)

    registry = manifest.get("online_feature_registry")
    if not isinstance(registry, Sequence) or isinstance(registry, str):
        raise ValueError("上游 manifest 缺少 online_feature_registry")
    if tuple(str(name) for name in registry) != tuple(ONLINE_FEATURE_COLUMNS):
        raise ValueError("上游 manifest 的 online_feature_registry 顺序不兼容")
    horizon_counts = manifest.get("horizon_sample_counts")
    if not isinstance(horizon_counts, Mapping):
        raise ValueError("上游 manifest 缺少 horizon_sample_counts")
    outputs = manifest.get("output_files")
    if not isinstance(outputs, Mapping):
        raise ValueError("上游 manifest 缺少 output_files")
    feature_description = outputs.get("causal_features.csv")
    if not isinstance(feature_description, Mapping) or not feature_description.get("sha256"):
        raise ValueError("上游 manifest 缺少 causal_features.csv 的 SHA-256")


def _expected_causal_feature_hash(manifest: Mapping[str, object]) -> str:
    outputs = manifest.get("output_files")
    if not isinstance(outputs, Mapping):
        raise ValueError("上游 manifest 缺少 output_files")
    description = outputs.get("causal_features.csv")
    if not isinstance(description, Mapping):
        raise ValueError("上游 manifest 缺少 causal_features.csv 工件描述")
    value = description.get("sha256")
    if not value or not re.fullmatch(r"[0-9a-fA-F]{64}", str(value)):
        raise ValueError("上游 manifest 的 causal_features.csv SHA-256 非法")
    return str(value).lower()


def _validate_frozen_label_weight_contract(
    labels: np.ndarray,
    label_types: np.ndarray,
    label_confidences: np.ndarray,
    sample_weights: np.ndarray,
) -> None:
    "Validate every frozen v1.1 combination of label source, confidence and supervision weight row-by-row."

    expected: dict[tuple[str, str], tuple[int, float]] = {
        ("core_positive", "high"): (1, 1.0),
        ("core_positive", "medium_missing_refit_issues"): (1, 0.5),
        ("hard_negative", "high"): (0, 1.0),
        ("background_negative", "weak_unmonitored_background"): (0, 0.25),
    }
    invalid_examples: list[dict[str, object]] = []
    for label, label_type, confidence, weight in zip(
        labels,
        label_types,
        label_confidences,
        sample_weights,
        strict=True,
    ):
        contract = expected.get((str(label_type), str(confidence)))
        if contract is None or int(label) != contract[0] or not math.isclose(
            float(weight), contract[1], rel_tol=0.0, abs_tol=1e-12
        ):
            invalid_examples.append(
                {
                    "label": int(label),
                    "label_type": str(label_type),
                    "label_confidence": str(confidence),
                    "sample_weight": float(weight),
                }
            )
            if len(invalid_examples) >= 5:
                break
    if invalid_examples:
        raise ValueError(
            "label_type × label_confidence × sample_weight 不符合 online_causal_v1_1："
            f"{invalid_examples}"
        )


def _parse_accepted(values: pd.Series) -> np.ndarray:
    normalised = values.astype(str).str.strip().str.lower()
    accepted = normalised.isin({"true", "1"}).to_numpy(dtype=bool)
    known = normalised.isin({"true", "false", "1", "0"}).to_numpy(dtype=bool)
    if not bool(np.all(known)):
        raise ValueError("accepted 只能包含 True/False 或 1/0")
    return accepted


def load_online_feature_table(
    input_dir: Path,
    horizon_seconds: int,
    expected_split: ExplicitHouseSplit | None = None,
    *,
    chunksize: int = 100_000,
    verify_hash: bool = True,
) -> OnlineFeatureTable:
    "Read one horizon by selected columns in chunks and strictly validate the v1.1 scalar-feature contract."

    input_dir = Path(input_dir).resolve()
    manifest_path = input_dir / "manifest.json"
    feature_path = input_dir / "causal_features.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not feature_path.is_file():
        raise FileNotFoundError(feature_path)
    if int(chunksize) != chunksize or chunksize <= 0:
        raise ValueError("chunksize 必须是正整数")
    if int(horizon_seconds) != horizon_seconds:
        raise ValueError("horizon_seconds 必须是整数秒")
    horizon = int(horizon_seconds)
    if horizon not in SUPPORTED_HORIZONS_SECONDS:
        raise ValueError(f"不支持的 online horizon：H{horizon}")

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, Mapping):
        raise ValueError("上游 manifest 顶层必须是对象")
    _validate_input_manifest(manifest)

    expected_hash = _expected_causal_feature_hash(manifest)
    if verify_hash:
        actual_hash = _sha256_file(feature_path)
        if actual_hash.lower() != expected_hash:
            raise ValueError(
                "causal_features.csv 的 SHA-256 与上游 manifest 不一致；"
                "拒绝使用可能被修改的正式输入"
            )

    required_columns = (
        *ONLINE_METADATA_COLUMNS,
        *ONLINE_FEATURE_COLUMNS,
        *ONLINE_MASK_COLUMNS,
    )
    header = pd.read_csv(feature_path, nrows=0, encoding="utf-8-sig")
    missing_columns = set(required_columns) - set(header.columns)
    if missing_columns:
        raise ValueError(f"causal_features.csv 缺少字段：{sorted(missing_columns)}")

    selected_chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        feature_path,
        usecols=list(required_columns),
        chunksize=int(chunksize),
        encoding="utf-8-sig",
    ):
        numeric_horizon = pd.to_numeric(chunk["horizon_seconds"], errors="raise")
        selected = chunk.loc[numeric_horizon == horizon]
        if not selected.empty:
            selected_chunks.append(selected)
    if not selected_chunks:
        raise ValueError(f"causal_features.csv 中没有 H{horizon} 样本")
    table = pd.concat(selected_chunks, ignore_index=True)

    expected_counts = manifest["horizon_sample_counts"]
    if f"h{horizon}" not in expected_counts:
        raise ValueError(f"上游 manifest 缺少 H{horizon} 行数")
    expected_count = int(expected_counts[f"h{horizon}"])
    if len(table) != expected_count:
        raise ValueError(
            f"H{horizon} 行数与 manifest 不一致："
            f"期望 {expected_count}，实际 {len(table)}"
        )

    if table["candidate_id"].isna().any() or table["candidate_id"].astype(str).str.strip().eq("").any():
        raise ValueError("candidate_id 不能为空")
    duplicated = table["candidate_id"].astype(str).duplicated(keep=False)
    if duplicated.any():
        examples = table.loc[duplicated, "candidate_id"].astype(str).head().tolist()
        raise ValueError(f"H{horizon} 存在重复 candidate_id：{examples}")
    if not bool(np.all(_parse_accepted(table["accepted"]))):
        raise ValueError("causal_features.csv 只能包含 accepted=True 的训练样本")

    for column in ("dataset", "house_id", "split", "label_type", "label_confidence"):
        values = table[column]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise ValueError(f"{column} 不能为空")
    split_values = table["split"].astype(str).str.strip().to_numpy(dtype=str)
    invalid_splits = set(split_values) - set(SPLIT_NAMES)
    if invalid_splits:
        raise ValueError(f"H{horizon} 包含未知 split：{sorted(invalid_splits)}")

    labels = _coerce_binary_vector(table["label"].to_numpy(), name="label")
    sample_weights = _coerce_weights(table["sample_weight"].to_numpy())
    # v1.1 confidence weights lie in [0,1]; larger values usually indicate class-balance weights incorrectly stored in the sample table.
    if np.any(sample_weights > 1.0):
        raise ValueError("sample_weight 不能大于 1；pos_weight 必须在训练时另行计算")

    try:
        values = table[list(ONLINE_FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
        raw_masks = table[list(ONLINE_MASK_COLUMNS)].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("在线 feature/mask 必须是数值") from error
    if not np.all(np.isfinite(raw_masks)) or not np.all(
        (raw_masks == 0.0) | (raw_masks == 1.0)
    ):
        raise ValueError("在线 feature mask 只能包含 0 或 1")
    masks = raw_masks.astype(np.int8)
    if not np.array_equal(np.isfinite(values), masks.astype(bool)):
        raise ValueError("在线特征有限性必须与逐特征 mask 完全一致")
    if not masks.any(axis=1).all():
        raise ValueError("存在 27 个在线特征全部缺失的样本")

    label_type = table["label_type"].astype(str).to_numpy()
    label_confidence = table["label_confidence"].astype(str).to_numpy()
    _validate_frozen_label_weight_contract(
        labels,
        label_type,
        label_confidence,
        sample_weights,
    )

    manifest_split = ExplicitHouseSplit.from_mapping(manifest["split"])
    if expected_split is not None and expected_split.to_dict() != manifest_split.to_dict():
        raise ValueError("训练配置 split 与上游 manifest 的冻结 split 不一致")
    frozen_split = expected_split or manifest_split
    assigned = frozen_split.assign(
        table[["dataset", "house_id", "label"]],
        require_all_houses=True,
        require_both_classes=True,
    )
    assigned_values = assigned["split"].astype(str).to_numpy()
    if not np.array_equal(split_values, assigned_values):
        raise ValueError("causal_features.csv 的逐行 split 与冻结 house split 不一致")

    metadata = table[list(ONLINE_METADATA_COLUMNS)].copy()
    metadata["candidate_id"] = metadata["candidate_id"].astype(str)
    metadata["dataset"] = metadata["dataset"].astype(str)
    metadata["split"] = split_values
    metadata["label"] = labels
    metadata["sample_weight"] = sample_weights
    metadata["source_appliance"] = (
        metadata["source_appliance"].fillna("").astype(str).str.strip()
    )
    return OnlineFeatureTable(
        metadata=metadata.reset_index(drop=True),
        values=values,
        masks=masks,
        labels=labels,
        sample_weights=sample_weights,
        split_values=split_values,
        horizon_seconds=horizon,
        input_manifest=manifest,
    )


def compute_train_weighted_pos_weight(
    labels: Sequence[object],
    sample_weights: Sequence[object],
    split_values: Sequence[object],
) -> float:
    "Compute BCE pos_weight using only valid positive and negative training weight mass."

    label_array = _coerce_binary_vector(labels, name="labels")
    weight_array = _coerce_weights(sample_weights)
    split_array = np.asarray(split_values).astype(str)
    if split_array.ndim != 1 or split_array.size != label_array.size:
        raise ValueError("split_values 必须与 labels 等长")
    if weight_array.size != label_array.size:
        raise ValueError("sample_weights 必须与 labels 等长")
    train = split_array == "train"
    if not train.any():
        raise ValueError("没有 train 样本，无法计算 pos_weight")
    positive_weight = float(weight_array[train & (label_array == 1)].sum())
    negative_weight = float(weight_array[train & (label_array == 0)].sum())
    if positive_weight <= 0.0 or negative_weight <= 0.0:
        raise ValueError("train 的正类和负类都必须具有正的有效权重")
    return negative_weight / positive_weight


def weighted_bce_with_logits(
    logits: Tensor,
    labels: Tensor,
    sample_weights: Tensor,
    pos_weight: float | Tensor,
) -> Tensor:
    "Binary cross-entropy weighted by sample confidence and normalized by total weight."

    if logits.ndim != 1 or labels.ndim != 1 or sample_weights.ndim != 1:
        raise ValueError("logits、labels、sample_weights 必须是一维张量")
    if logits.shape != labels.shape or logits.shape != sample_weights.shape:
        raise ValueError("logits、labels、sample_weights 的形状必须一致")
    if logits.numel() == 0:
        raise ValueError("不能计算空 batch 的 loss")
    labels = labels.to(device=logits.device, dtype=logits.dtype)
    sample_weights = sample_weights.to(device=logits.device, dtype=logits.dtype)
    if not bool(torch.isfinite(labels).all()) or not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError("labels 只能包含 0 或 1")
    if not bool(torch.isfinite(sample_weights).all()) or not bool((sample_weights > 0).all()):
        raise ValueError("sample_weights 必须是有限正数")
    positive_class_weight = torch.as_tensor(
        pos_weight, device=logits.device, dtype=logits.dtype
    )
    if positive_class_weight.numel() != 1 or not bool(torch.isfinite(positive_class_weight)):
        raise ValueError("pos_weight 必须是有限正标量")
    if not bool(positive_class_weight > 0):
        raise ValueError("pos_weight 必须大于 0")
    raw = F.binary_cross_entropy_with_logits(
        logits,
        labels,
        pos_weight=positive_class_weight,
        reduction="none",
    )
    return torch.sum(sample_weights * raw) / torch.sum(sample_weights)


def compute_weighted_binary_metrics(
    labels: Sequence[object],
    probabilities: Sequence[object],
    sample_weights: Sequence[object],
    threshold: float,
) -> dict[str, object]:
    "Compute confidence-weighted metrics; confusion-matrix units are weight mass rather than event counts."

    label_array = _coerce_binary_vector(labels, name="labels")
    probability_array = _validate_probability_vector(probabilities, label_array.size)
    weight_array = _coerce_weights(sample_weights)
    if weight_array.size != label_array.size:
        raise ValueError("sample_weights 必须与 labels 等长")
    if not np.isfinite(threshold) or not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold 必须位于 [0, 1]")
    predicted = (probability_array >= float(threshold)).astype(np.int64)

    negative = label_array == 0
    positive = ~negative
    predicted_negative = predicted == 0
    predicted_positive = ~predicted_negative
    true_negative = float(weight_array[negative & predicted_negative].sum())
    false_positive = float(weight_array[negative & predicted_positive].sum())
    false_negative = float(weight_array[positive & predicted_negative].sum())
    true_positive = float(weight_array[positive & predicted_positive].sum())
    negative_weight = true_negative + false_positive
    positive_weight = true_positive + false_negative
    total_weight = negative_weight + positive_weight

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    two_classes = negative_weight > 0.0 and positive_weight > 0.0
    specificity = true_negative / negative_weight if negative_weight else 0.0
    return {
        "sample_count": int(label_array.size),
        "sample_weight_sum": float(total_weight),
        "negative_count": int(np.count_nonzero(negative)),
        "positive_count": int(np.count_nonzero(positive)),
        "negative_weight": float(negative_weight),
        "positive_weight": float(positive_weight),
        "threshold": float(threshold),
        "accuracy": float((true_negative + true_positive) / total_weight),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "balanced_accuracy": float((specificity + recall) / 2.0) if two_classes else None,
        "roc_auc": (
            float(roc_auc_score(label_array, probability_array, sample_weight=weight_array))
            if two_classes
            else None
        ),
        "pr_auc": (
            float(
                average_precision_score(
                    label_array,
                    probability_array,
                    sample_weight=weight_array,
                )
            )
            if two_classes
            else None
        ),
        "pr_auc_method": "sample_weighted_average_precision",
        "confusion_matrix": [
            [true_negative, false_positive],
            [false_negative, true_positive],
        ],
        "confusion_matrix_unit": "sample_weight_mass",
    }


def select_weighted_validation_threshold(
    labels: Sequence[object],
    probabilities: Sequence[object],
    sample_weights: Sequence[object],
) -> WeightedThresholdSelection:
    "Select a threshold on all validation candidates using weighted F1 and the frozen tie-breaking rule."

    label_array = _coerce_binary_vector(labels, name="labels")
    probability_array = _validate_probability_vector(probabilities, label_array.size)
    weight_array = _coerce_weights(sample_weights)
    if weight_array.size != label_array.size:
        raise ValueError("sample_weights 必须与 labels 等长")
    if np.unique(label_array).size != 2:
        raise ValueError("阈值选择要求验证集同时包含正负类")

    candidates = np.unique(np.concatenate(([0.0, 1.0], probability_array)))
    best_ranking: tuple[float, float, float, float, float] | None = None
    best_result: WeightedThresholdSelection | None = None
    negative = label_array == 0
    positive = ~negative
    negative_weight = float(weight_array[negative].sum())
    positive_weight = float(weight_array[positive].sum())
    for threshold in candidates:
        predicted_positive = probability_array >= threshold
        true_positive = float(weight_array[positive & predicted_positive].sum())
        false_positive = float(weight_array[negative & predicted_positive].sum())
        false_negative = positive_weight - true_positive
        true_negative = negative_weight - false_positive
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = true_positive / (true_positive + false_negative)
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        specificity = true_negative / (true_negative + false_positive)
        balanced = (recall + specificity) / 2.0
        ranking = (
            f1,
            balanced,
            recall,
            -abs(float(threshold) - 0.5),
            -float(threshold),
        )
        if best_ranking is None or ranking > best_ranking:
            best_ranking = ranking
            best_result = WeightedThresholdSelection(
                threshold=float(threshold),
                f1=float(f1),
                balanced_accuracy=float(balanced),
                recall=float(recall),
                candidate_count=int(candidates.size),
                sample_count=int(label_array.size),
                sample_weight_sum=float(weight_array.sum()),
            )
    assert best_result is not None
    return best_result


def is_high_confidence(
    label_types: Sequence[object],
    label_confidences: Sequence[object],
    sample_weights: Sequence[object],
) -> np.ndarray:
    "Return the frozen reliable-label subset mask without including medium-confidence H1 heater samples."

    types = np.asarray(label_types).astype(str)
    confidences = np.asarray(label_confidences).astype(str)
    weights = _coerce_weights(sample_weights)
    if types.ndim != 1 or confidences.ndim != 1:
        raise ValueError("label_types 与 label_confidences 必须是一维数组")
    if not (types.size == confidences.size == weights.size):
        raise ValueError("高置信条件的三个数组必须等长")
    return (
        np.isin(types, list(HIGH_CONFIDENCE_LABEL_TYPES))
        & (confidences == "high")
        & np.isclose(weights, 1.0, rtol=0.0, atol=1e-12)
    )


def _set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _make_loader(
    values: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
    positions: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(values[positions].astype(np.float32, copy=False)),
        torch.from_numpy(masks[positions].astype(np.float32, copy=False)),
        torch.from_numpy(labels[positions].astype(np.float32, copy=False)),
        torch.from_numpy(sample_weights[positions].astype(np.float32, copy=False)),
    )
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )


def fit_online_weighted_grouped_mlp(
    values: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
    split_values: np.ndarray,
    model_config: Mapping[str, object],
    training_config: Mapping[str, object],
    *,
    epochs_override: int | None = None,
) -> FittedOnlineModel:
    "Update parameters on training samples and select the best checkpoint by weighted validation loss."

    values = np.asarray(values, dtype=np.float64)
    masks = np.asarray(masks)
    labels = _coerce_binary_vector(labels, name="labels")
    sample_weights = _coerce_weights(sample_weights)
    split_values = np.asarray(split_values).astype(str)
    expected_shape = (labels.size, len(ONLINE_FEATURE_COLUMNS))
    if values.shape != expected_shape or masks.shape != expected_shape:
        raise ValueError(f"values/masks 必须是 {expected_shape}")
    if sample_weights.size != labels.size or split_values.shape != labels.shape:
        raise ValueError("权重、split 与 labels 必须等长")
    if not np.all(np.isfinite(masks)) or not np.all((masks == 0) | (masks == 1)):
        raise ValueError("masks 只能包含 0 或 1")
    if not np.all(np.isfinite(values[masks.astype(bool)])):
        raise ValueError("mask=1 的 values 必须是有限数")
    if set(split_values) != set(SPLIT_NAMES):
        raise ValueError("训练输入必须同时包含 train/validation/test")
    if int(training_config.get("num_workers", 0)) != 0:
        raise ValueError("online weighted MLP v1 固定 num_workers=0 以保证可复现")

    seed = int(training_config["seed"])
    epochs = int(epochs_override or training_config["epochs"])
    batch_size = int(training_config["batch_size"])
    patience_limit = int(training_config["early_stopping_patience"])
    if epochs <= 0 or batch_size <= 0 or patience_limit <= 0:
        raise ValueError("epochs、batch_size、early_stopping_patience 必须为正整数")
    _set_reproducible_seed(seed)

    group_dimensions = {
        group: len(features) for group, features in ONLINE_GROUP_FEATURES.items()
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

    train_positions = np.flatnonzero(split_values == "train")
    validation_positions = np.flatnonzero(split_values == "validation")
    pos_weight = compute_train_weighted_pos_weight(labels, sample_weights, split_values)
    train_positive_weight = float(
        sample_weights[train_positions][labels[train_positions] == 1].sum()
    )
    train_negative_weight = float(
        sample_weights[train_positions][labels[train_positions] == 0].sum()
    )
    train_loader = _make_loader(
        values,
        masks,
        labels,
        sample_weights,
        train_positions,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    validation_loader = _make_loader(
        values,
        masks,
        labels,
        sample_weights,
        validation_positions,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )

    history: list[dict[str, float | int]] = []
    best_state: dict[str, Tensor] | None = None
    best_validation_loss = math.inf
    best_epoch = 0
    patience = 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_numerator = 0.0
        train_weight_sum = 0.0
        for batch_values, batch_masks, batch_labels, batch_weights in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(batch_values, batch_masks)
            loss = weighted_bce_with_logits(
                logits, batch_labels, batch_weights, pos_weight
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("训练 loss 出现 NaN 或无穷值")
            loss.backward()
            optimizer.step()
            batch_weight_sum = float(batch_weights.sum().item())
            train_numerator += float(loss.detach().item()) * batch_weight_sum
            train_weight_sum += batch_weight_sum

        model.eval()
        validation_numerator = 0.0
        validation_weight_sum = 0.0
        with torch.no_grad():
            for batch_values, batch_masks, batch_labels, batch_weights in validation_loader:
                logits, _ = model(batch_values, batch_masks)
                loss = weighted_bce_with_logits(
                    logits, batch_labels, batch_weights, pos_weight
                )
                batch_weight_sum = float(batch_weights.sum().item())
                validation_numerator += float(loss.item()) * batch_weight_sum
                validation_weight_sum += batch_weight_sum
        train_loss = train_numerator / train_weight_sum
        validation_loss = validation_numerator / validation_weight_sum
        history.append(
            {
                "epoch": epoch,
                "train_weighted_loss": float(train_loss),
                "validation_weighted_loss": float(validation_loss),
            }
        )
        if validation_loss < best_validation_loss - 1e-12:
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
    return FittedOnlineModel(
        model=model,
        history=history,
        best_epoch=best_epoch,
        best_validation_loss=float(best_validation_loss),
        pos_weight=float(pos_weight),
        train_positive_weight=train_positive_weight,
        train_negative_weight=train_negative_weight,
    )


def _predict(
    model: GroupedFingerprintMLP,
    values: np.ndarray,
    masks: np.ndarray,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(values, dtype=np.float32)),
        torch.from_numpy(np.asarray(masks, dtype=np.float32)),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    probability_batches: list[np.ndarray] = []
    gate_batches: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch_values, batch_masks in loader:
            logits, gates = model(batch_values, batch_masks)
            probability_batches.append(torch.sigmoid(logits).cpu().numpy())
            gate_batches.append(gates.cpu().numpy())
    return np.concatenate(probability_batches), np.concatenate(gate_batches)


def _metric_pair(
    labels: np.ndarray,
    probabilities: np.ndarray,
    sample_weights: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    return {
        "unweighted": compute_binary_metrics(labels, probabilities, threshold),
        "weighted": compute_weighted_binary_metrics(
            labels, probabilities, sample_weights, threshold
        ),
    }


def _evaluate_view(
    metadata: pd.DataFrame,
    labels: np.ndarray,
    probabilities: np.ndarray,
    sample_weights: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    "Compute overall and metadata-slice metrics by positional index rather than the original DataFrame index."

    metadata = metadata.reset_index(drop=True).copy()
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    sample_weights = np.asarray(sample_weights)
    if not (
        len(metadata) == labels.size == probabilities.size == sample_weights.size
    ):
        raise ValueError("metadata、labels、probabilities、sample_weights 必须等长")
    if len(metadata) == 0:
        return {"available": False, "sample_count": 0}
    metadata["source_appliance_group"] = (
        metadata["source_appliance"]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace("", "unmatched_background")
    )
    result: dict[str, object] = {
        "available": True,
        "overall": _metric_pair(labels, probabilities, sample_weights, threshold),
        "per_dataset": {},
        "per_house": {},
        "per_source_appliance": {},
        "per_label_type": {},
        "per_label_confidence": {},
    }

    groupings = (
        ("per_dataset", ["dataset"]),
        ("per_house", ["dataset", "house_id"]),
        ("per_source_appliance", ["source_appliance_group"]),
        ("per_label_type", ["label_type"]),
        ("per_label_confidence", ["label_confidence"]),
    )
    for result_key, columns in groupings:
        grouper: str | list[str] = columns[0] if len(columns) == 1 else columns
        for raw_key, group in metadata.groupby(grouper, sort=True, dropna=False):
            positions = group.index.to_numpy(dtype=np.int64)
            if result_key == "per_house":
                dataset, house_id = raw_key
                key = HouseKey(str(dataset), str(house_id)).serialise()
            else:
                key = str(raw_key)
            result[result_key][key] = _metric_pair(
                labels[positions],
                probabilities[positions],
                sample_weights[positions],
                threshold,
            )
    return result


def _evaluate_split(
    metadata: pd.DataFrame,
    labels: np.ndarray,
    probabilities: np.ndarray,
    sample_weights: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    reliable = is_high_confidence(
        metadata["label_type"].to_numpy(),
        metadata["label_confidence"].to_numpy(),
        sample_weights,
    )
    return {
        "all_candidates": _evaluate_view(
            metadata, labels, probabilities, sample_weights, threshold
        ),
        "high_confidence": _evaluate_view(
            metadata.loc[reliable].reset_index(drop=True),
            labels[reliable],
            probabilities[reliable],
            sample_weights[reliable],
            threshold,
        ),
    }


def _gate_statistics(
    gates: np.ndarray,
    masks: np.ndarray,
    group_order: Sequence[str],
) -> dict[str, object]:
    "Summarize missing-data routing without interpreting model gate weights as causal feature contributions."

    gates = np.asarray(gates, dtype=np.float64)
    masks = np.asarray(masks).astype(bool)
    if gates.ndim != 2 or gates.shape[1] != len(group_order):
        raise ValueError("gate 权重形状与 group_order 不一致")
    if masks.shape != (gates.shape[0], len(ONLINE_FEATURE_COLUMNS)):
        raise ValueError("mask 形状与在线特征 registry 不一致")
    boundaries = np.cumsum(
        [0, *(len(ONLINE_GROUP_FEATURES[name]) for name in group_order)]
    )
    return {
        "interpretation": "仅用于检查缺失路由，不代表因果贡献或特征重要性。",
        "groups": {
            name: {
                "mean_weight": float(np.mean(gates[:, index])),
                "median_weight": float(np.median(gates[:, index])),
                "available_sample_count": int(
                    masks[:, boundaries[index] : boundaries[index + 1]]
                    .any(axis=1)
                    .sum()
                ),
            }
            for index, name in enumerate(group_order)
        },
    }


def _smoke_subset(table: OnlineFeatureTable, maximum_rows_per_group: int) -> OnlineFeatureTable:
    if maximum_rows_per_group <= 0:
        raise ValueError("smoke maximum_rows_per_group 必须大于 0")
    audit = table.metadata[["split", "label", "label_type", "candidate_id"]].copy()
    audit["_position"] = np.arange(len(audit), dtype=np.int64)
    audit = audit.sort_values(
        ["split", "label", "label_type", "candidate_id"], kind="stable"
    )
    keep = (
        audit.groupby(["split", "label", "label_type"], sort=True, dropna=False)
        .head(maximum_rows_per_group)["_position"]
        .sort_values()
        .to_numpy(dtype=np.int64)
    )
    result = table.subset(keep)
    coverage = result.metadata.groupby("split")["label"].nunique()
    if set(coverage.index) != set(SPLIT_NAMES) or not bool((coverage == 2).all()):
        raise ValueError("smoke 子集必须在每个 split 同时保留正负类")
    return result


def _model_configuration(
    raw_model_config: Mapping[str, object],
    model: GroupedFingerprintMLP,
) -> dict[str, object]:
    return {
        **dict(raw_model_config),
        "group_dimensions": {
            name: len(features) for name, features in ONLINE_GROUP_FEATURES.items()
        },
        "group_order": list(model.group_order),
        "input_dimension": len(ONLINE_FEATURE_COLUMNS),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
    }


def _prediction_table(
    metadata: pd.DataFrame,
    probabilities: np.ndarray,
    gates: np.ndarray,
    threshold: float,
    group_order: Sequence[str],
) -> pd.DataFrame:
    columns = [
        "candidate_id",
        "dataset",
        "house_id",
        "split",
        "candidate_onset_s",
        "candidate_detected_at_s",
        "horizon_seconds",
        "label",
        "label_type",
        "label_confidence",
        "sample_weight",
        "source_appliance",
    ]
    result = metadata[columns].copy().rename(columns={"label": "true_label"})
    result["predicted_probability"] = probabilities
    result["predicted_label"] = (probabilities >= threshold).astype(np.int64)
    result["threshold"] = float(threshold)
    for index, name in enumerate(group_order):
        result[f"{name}_gate_weight"] = gates[:, index]
    return result


def _split_summary(table: OnlineFeatureTable) -> dict[str, object]:
    result: dict[str, object] = {}
    for split in SPLIT_NAMES:
        selected = table.split_values == split
        result[split] = {
            "sample_count": int(selected.sum()),
            "negative_count": int(np.count_nonzero(table.labels[selected] == 0)),
            "positive_count": int(np.count_nonzero(table.labels[selected] == 1)),
            "negative_weight": float(
                table.sample_weights[selected & (table.labels == 0)].sum()
            ),
            "positive_weight": float(
                table.sample_weights[selected & (table.labels == 1)].sum()
            ),
        }
    return result


def _mask_availability_summary(table: OnlineFeatureTable) -> dict[str, object]:
    "Summarize mask availability in each split from actual run samples without assuming any horizon result."

    result: dict[str, object] = {}
    for split in SPLIT_NAMES:
        selected = table.split_values == split
        split_masks = np.asarray(table.masks[selected], dtype=bool)
        sample_count = int(split_masks.shape[0])
        if sample_count == 0:
            raise ValueError(f"{split} 没有样本，无法生成 mask 审计")
        fully_available = split_masks.all(axis=1)
        result[split] = {
            "sample_count": sample_count,
            "mask_available_count": int(split_masks.sum()),
            "mask_total_count": int(split_masks.size),
            "mask_availability_rate": float(split_masks.mean()),
            "all_features_available_sample_count": int(fully_available.sum()),
            "all_features_available_rate": float(fully_available.mean()),
            "per_feature_availability_rate": {
                name: float(split_masks[:, index].mean())
                for index, name in enumerate(ONLINE_FEATURE_COLUMNS)
            },
        }
    return result


def _format_metric(value: object) -> str:
    return "—" if value is None else f"{float(value):.4f}"


def _format_feature_names(values: Sequence[object]) -> str:
    "Render field lists as stable report text."

    names = [str(value) for value in values]
    return "无" if not names else "、".join(f"`{name}`" for name in names)


def _render_run_report(
    *,
    run_id: str,
    smoke_test: bool,
    horizon_seconds: int,
    model_config: Mapping[str, object],
    metrics: Mapping[str, object],
    feature_schema: Mapping[str, object],
    split_constraint: str,
) -> str:
    validation = metrics["validation"]["all_candidates"]["overall"]
    test = metrics["test"]["all_candidates"]["overall"]
    test_reliable = metrics["test"]["high_confidence"]
    reliable_overall = test_reliable.get("overall", {}) if test_reliable.get("available") else {}
    mask_summary = feature_schema.get("mask_availability_by_split")
    if not isinstance(mask_summary, Mapping):
        raise ValueError("feature_schema 缺少 mask_availability_by_split")
    mask_rows: list[str] = []
    feature_mask_rows: list[str] = []
    for split in SPLIT_NAMES:
        detail = mask_summary.get(split)
        if not isinstance(detail, Mapping):
            raise ValueError(f"mask_availability_by_split 缺少 {split}")
        mask_rows.append(
            "| {split} | {sample_count} | {mask_rate:.2%} | {complete_count} | {complete_rate:.2%} |".format(
                split=split,
                sample_count=int(detail["sample_count"]),
                mask_rate=float(detail["mask_availability_rate"]),
                complete_count=int(detail["all_features_available_sample_count"]),
                complete_rate=float(detail["all_features_available_rate"]),
            )
        )
    for feature_name in ONLINE_FEATURE_COLUMNS:
        availability = []
        for split in SPLIT_NAMES:
            detail = mask_summary[split]
            assert isinstance(detail, Mapping)
            per_feature = detail.get("per_feature_availability_rate")
            if not isinstance(per_feature, Mapping) or feature_name not in per_feature:
                raise ValueError(f"{split} 缺少 {feature_name} 的 mask 可用率")
            availability.append(f"{float(per_feature[feature_name]):.2%}")
        feature_mask_rows.append(
            f"| `{feature_name}` | {availability[0]} | {availability[1]} | {availability[2]} |"
        )
    onset_delay = 20
    detector_extra_wait = max(0, horizon_seconds - 20)
    mask_section = f"""## 实际 mask 与 train 零方差审计

| split | 样本数 | 全部 mask 单元可用率 | 27 字段全可用样本 | 全可用样本率 |
|---|---:|---:|---:|---:|
{chr(10).join(mask_rows)}

下表列出当前horizon在各split中的逐特征mask可用率。

| 特征 | train | validation | test |
|---|---:|---:|---:|
{chr(10).join(feature_mask_rows)}

- train 零方差字段：{_format_feature_names(feature_schema['zero_variance_train_features'])}。字段仍保留在冻结 registry 中。
"""
    if smoke_test:
        main_section = """## Smoke检查范围

Smoke模式检查样本读取、加权loss、前向与反向计算、验证集选择、checkpoint重载、来源追踪及原子写入。
用于重载检查的预测和指标不构成模型性能评测；正式结果需要完整训练与留出测试。
"""
    else:
        main_section = f"""最佳 epoch：{metrics['best_epoch']}  
最佳加权 validation loss：{metrics['best_validation_loss']:.6f}  
train-only `pos_weight`：{metrics['train_class_weight']['pos_weight']:.10f}  
validation 加权阈值：{metrics['validation_threshold']['threshold']:.10f}

## 主要指标

| 数据口径 | n | F1 | weighted F1 | balanced accuracy | weighted balanced accuracy | PR-AUC | weighted PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation 全候选 | {validation['unweighted']['sample_count']} | {_format_metric(validation['unweighted']['f1'])} | {_format_metric(validation['weighted']['f1'])} | {_format_metric(validation['unweighted']['balanced_accuracy'])} | {_format_metric(validation['weighted']['balanced_accuracy'])} | {_format_metric(validation['unweighted']['pr_auc'])} | {_format_metric(validation['weighted']['pr_auc'])} |
| test 全候选 | {test['unweighted']['sample_count']} | {_format_metric(test['unweighted']['f1'])} | {_format_metric(test['weighted']['f1'])} | {_format_metric(test['unweighted']['balanced_accuracy'])} | {_format_metric(test['weighted']['balanced_accuracy'])} | {_format_metric(test['unweighted']['pr_auc'])} | {_format_metric(test['weighted']['pr_auc'])} |
| test 高置信 | {reliable_overall.get('unweighted', {}).get('sample_count', 0)} | {_format_metric(reliable_overall.get('unweighted', {}).get('f1'))} | {_format_metric(reliable_overall.get('weighted', {}).get('f1'))} | {_format_metric(reliable_overall.get('unweighted', {}).get('balanced_accuracy'))} | {_format_metric(reliable_overall.get('weighted', {}).get('balanced_accuracy'))} | {_format_metric(reliable_overall.get('unweighted', {}).get('pr_auc'))} | {_format_metric(reliable_overall.get('weighted', {}).get('pr_auc'))} |

阈值只由 validation 全候选的 `sample_weight` 加权 F1 决定；平局依次比较加权
balanced accuracy、加权 recall、距 0.5、较低阈值。`pos_weight` 只参与 BCE，未参与阈值。
test 没有参与 scaler、训练、checkpoint 或阈值选择。
"""
    return f"""# 在线加权分组 MLP 运行报告：{run_id}

模式：**{'smoke（流程检查）' if smoke_test else '正式'}**  
输入：`online_causal_v1_1 / online_feature_v1_1 / H{horizon_seconds}`  
参数量：{model_config['parameter_count']}  

{main_section}

## 输入与时延边界

- H{horizon_seconds} 输入是 Power-17 + Temporal-10 和逐特征 mask。
- 候选检测器在启动后约 20 秒确认；H{horizon_seconds} 相对启动可在约 {onset_delay + detector_extra_wait} 秒输出，
  相对候选确认额外等待 {detector_extra_wait} 秒。
- 分类指标以“已被 aggregate detector 找到且通过 H{horizon_seconds} QC”的候选为条件，
  不能补回候选检测器漏检或 QC 拒绝的真实启动。

{mask_section}

## 已知限制

- {split_constraint}
- background negative 是 0.25 权重弱标签，设备为空时分层名称为 `unmatched_background`。
- 高置信子集严格要求可靠 label type、`label_confidence=high` 且权重为 1。
- hair dryer/iron迁移样本未包含在主causal feature/NPZ中；此数据包不支持这两类迁移评测。
- gate 统计只用于缺失路由调试，不代表因果贡献或特征重要性。
"""


def _validate_training_config(raw_config: Mapping[str, object]) -> None:
    for section in (
        "schema",
        "paths",
        "dataset",
        "split",
        "model",
        "training",
        "smoke",
        "loss",
        "evaluation",
    ):
        if not isinstance(raw_config.get(section), Mapping):
            raise ValueError(f"训练配置缺少 [{section}] section")
    schema = raw_config["schema"]
    expected = {
        "training_schema_version": TRAINING_SCHEMA_VERSION,
        "input_schema_version": ONLINE_SAMPLE_SCHEMA_VERSION,
        "feature_registry_version": FEATURE_REGISTRY_VERSION,
        "npz_contract_version": SEQUENCE_CONTRACT_VERSION,
    }
    for key, value in expected.items():
        if str(schema.get(key)) != value:
            raise ValueError(f"训练配置 {key} 必须是 {value!r}")
    loss = raw_config["loss"]
    if loss.get("class_balance") != "train_weighted_negative_mass_over_positive_mass":
        raise ValueError("loss.class_balance 与冻结协议不一致")
    if loss.get("sample_weight_normalization") != "sum_sample_weight":
        raise ValueError("loss.sample_weight_normalization 与冻结协议不一致")
    if loss.get("sampler") != "shuffle_without_weighted_resampling":
        raise ValueError("初版禁止叠加 WeightedRandomSampler")
    evaluation = raw_config["evaluation"]
    if evaluation.get("threshold_objective") != "validation_all_candidates_sample_weighted_f1":
        raise ValueError("evaluation.threshold_objective 与冻结协议不一致")


def _validate_run_id(run_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id):
        raise ValueError(
            "run_id 必须以字母或数字开头，且只能包含字母、数字、点、下划线和连字符"
        )
    if run_id in {".", ".."}:
        raise ValueError("run_id 不能是相对路径标记")
    return run_id


def run_online_weighted_grouped_mlp(
    project_root: Path,
    config_path: Path,
    *,
    smoke_test: bool = False,
    run_id: str | None = None,
    preregistration_path: Path | None = None,
) -> dict[str, object]:
    "Run the weighted grouped MLP for one horizon and atomically persist complete immutable run artifacts."

    project_root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    config_snapshot_bytes = config_path.read_bytes()
    raw_config = tomllib.loads(config_snapshot_bytes.decode("utf-8-sig"))
    _validate_training_config(raw_config)
    horizon_seconds = int(raw_config["dataset"]["horizon_seconds"])
    preregistration_reference, resolved_preregistration_path = (
        _verify_preregistration_for_run(
            project_root,
            preregistration_path,
            horizon_seconds,
            config_path,
        )
    )

    generated_id = (
        ("smoke_online_mlp_h" if smoke_test else "online_mlp_h")
        + str(int(raw_config["dataset"]["horizon_seconds"]))
        + "_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    run_id = _validate_run_id(run_id or generated_id)
    run_root = _resolve_project_path(
        project_root, str(raw_config["paths"]["run_output_root"])
    )
    run_root.mkdir(parents=True, exist_ok=True)
    final_dir = (run_root / run_id).resolve()
    if final_dir.parent != run_root:
        raise ValueError("run_id 解析后逃出了运行输出目录")
    if final_dir.exists():
        raise FileExistsError(f"运行目录已存在，拒绝覆盖：{final_dir}")

    # The split section also includes the readable known_constraint; the split object accepts only its three formal sets.
    split = ExplicitHouseSplit.from_mapping(
        {name: raw_config["split"][name] for name in SPLIT_NAMES}
    )
    input_dir = _resolve_project_path(
        project_root, str(raw_config["paths"]["online_input_dir"])
    )
    taxonomy_path = input_dir / "taxonomy_snapshot.yaml"
    input_manifest_path = input_dir / "manifest.json"
    input_feature_path = input_dir / "causal_features.csv"
    for required_path in (taxonomy_path, input_manifest_path, input_feature_path):
        if not required_path.is_file():
            raise FileNotFoundError(required_path)
    taxonomy_snapshot_bytes = taxonomy_path.read_bytes()
    input_manifest_snapshot_bytes = input_manifest_path.read_bytes()
    implementation_paths = (
        Path(__file__),
        Path(__file__).with_name("grouped_mlp.py"),
        Path(__file__).with_name("mvp_preprocessing.py"),
        Path(__file__).with_name("online_features.py"),
        project_root / "scripts" / "run_online_weighted_mlp.py",
        *(
            (Path(__file__).with_name("online_multihorizon.py"),)
            if resolved_preregistration_path is not None
            else ()
        ),
    )
    input_artifacts = [
        _describe_captured_bytes(
            input_manifest_path, input_manifest_snapshot_bytes, project_root
        ),
        _describe_file(input_feature_path, project_root),
        _describe_captured_bytes(taxonomy_path, taxonomy_snapshot_bytes, project_root),
    ]
    configuration_artifacts = [
        _describe_captured_bytes(config_path, config_snapshot_bytes, project_root)
    ]
    if preregistration_reference is not None:
        configuration_artifacts.append(preregistration_reference)
    implementation_artifacts = [
        _describe_file(path, project_root)
        for path in implementation_paths
        if path.is_file()
    ]
    lineage_snapshots = [
        *((path, description) for path, description in zip(
            (input_manifest_path, input_feature_path, taxonomy_path),
            input_artifacts,
            strict=True,
        )),
        (config_path, configuration_artifacts[0]),
        *((path, description) for path, description in zip(
            (path for path in implementation_paths if path.is_file()),
            implementation_artifacts,
            strict=True,
        )),
    ]
    if (
        preregistration_reference is not None
        and resolved_preregistration_path is not None
    ):
        lineage_snapshots.append(
            (resolved_preregistration_path, preregistration_reference)
        )
    table = load_online_feature_table(
        input_dir,
        horizon_seconds,
        split,
        chunksize=int(raw_config["dataset"]["csv_chunksize"]),
    )
    if smoke_test:
        table = _smoke_subset(
            table, int(raw_config["smoke"]["maximum_rows_per_group"])
        )

    train_rows = table.split_values == "train"
    scaler = MaskedStandardizer(ONLINE_FEATURE_COLUMNS).fit(
        table.values[train_rows], table.masks[train_rows]
    )
    values, masks = scaler.transform(
        table.values, table.masks, feature_names=ONLINE_FEATURE_COLUMNS
    )
    if not masks.any(axis=1).all():
        raise ValueError("标准化后出现 27 个特征全部不可用的样本")

    observed_train_standard_deviations: dict[str, float | None] = {}
    zero_variance_train_features: list[str] = []
    for index, name in enumerate(ONLINE_FEATURE_COLUMNS):
        observed = table.values[train_rows, index][table.masks[train_rows, index] == 1]
        standard_deviation = float(np.std(observed, ddof=0)) if observed.size else None
        observed_train_standard_deviations[name] = standard_deviation
        if standard_deviation is not None and standard_deviation <= scaler.variance_epsilon:
            zero_variance_train_features.append(name)

    fitted = fit_online_weighted_grouped_mlp(
        values,
        masks,
        table.labels,
        table.sample_weights,
        table.split_values,
        raw_config["model"],
        raw_config["training"],
        epochs_override=(
            int(raw_config["training"]["smoke_epochs"]) if smoke_test else None
        ),
    )
    batch_size = int(raw_config["training"]["batch_size"])
    validation_rows = table.split_values == "validation"
    test_rows = table.split_values == "test"
    validation_probabilities, validation_gates = _predict(
        fitted.model,
        values[validation_rows],
        masks[validation_rows],
        batch_size=batch_size,
    )
    threshold = select_weighted_validation_threshold(
        table.labels[validation_rows],
        validation_probabilities,
        table.sample_weights[validation_rows],
    )
    test_probabilities, test_gates = _predict(
        fitted.model,
        values[test_rows],
        masks[test_rows],
        batch_size=batch_size,
    )
    validation_metadata = table.metadata.loc[validation_rows].reset_index(drop=True)
    test_metadata = table.metadata.loc[test_rows].reset_index(drop=True)

    model_config = _model_configuration(raw_config["model"], fitted.model)
    feature_schema = {
        "input_schema_version": ONLINE_SAMPLE_SCHEMA_VERSION,
        "feature_registry_version": FEATURE_REGISTRY_VERSION,
        "horizon_seconds": horizon_seconds,
        "feature_columns": list(ONLINE_FEATURE_COLUMNS),
        "mask_columns": list(ONLINE_MASK_COLUMNS),
        "groups": {
            group: list(features) for group, features in ONLINE_GROUP_FEATURES.items()
        },
        "metadata_excluded_from_model": list(ONLINE_METADATA_COLUMNS),
        "train_observed_count": scaler.observed_count_.astype(int).tolist(),
        "train_supported": scaler.supported_.astype(bool).tolist(),
        "train_observed_standard_deviation": observed_train_standard_deviations,
        "zero_variance_train_features": zero_variance_train_features,
        "mask_availability_by_split": _mask_availability_summary(table),
    }
    metrics = {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "smoke_test": bool(smoke_test),
        "horizon_seconds": horizon_seconds,
        "best_epoch": fitted.best_epoch,
        "best_validation_loss": fitted.best_validation_loss,
        "training_history": fitted.history,
        "train_class_weight": {
            "positive_weight": fitted.train_positive_weight,
            "negative_weight": fitted.train_negative_weight,
            "pos_weight": fitted.pos_weight,
            "formula": "train_weighted_negative_mass / train_weighted_positive_mass",
        },
        "validation_threshold": threshold.to_dict(),
        "validation": _evaluate_split(
            validation_metadata,
            table.labels[validation_rows],
            validation_probabilities,
            table.sample_weights[validation_rows],
            threshold.threshold,
        ),
        "test": _evaluate_split(
            test_metadata,
            table.labels[test_rows],
            test_probabilities,
            table.sample_weights[test_rows],
            threshold.threshold,
        ),
        "gate_statistics": {
            "validation": _gate_statistics(
                validation_gates, masks[validation_rows], fitted.model.group_order
            ),
            "test": _gate_statistics(
                test_gates, masks[test_rows], fitted.model.group_order
            ),
        },
    }
    validation_predictions = _prediction_table(
        validation_metadata,
        validation_probabilities,
        validation_gates,
        threshold.threshold,
        fitted.model.group_order,
    )
    test_predictions = _prediction_table(
        test_metadata,
        test_probabilities,
        test_gates,
        threshold.threshold,
        fitted.model.group_order,
    )

    # Training may take minutes; recheck protected inputs and all lineage before commit to prevent concurrent edits from creating a false snapshot.
    if preregistration_reference is not None:
        final_preregistration_reference, _ = _verify_preregistration_for_run(
            project_root,
            preregistration_path,
            horizon_seconds,
            config_path,
        )
        if final_preregistration_reference != preregistration_reference:
            raise RuntimeError("训练期间多 horizon 预注册引用发生变化，拒绝提交")
    _assert_lineage_unchanged(lineage_snapshots)
    run_manifest: dict[str, object] = {
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_schema_version": TRAINING_SCHEMA_VERSION,
        "input_schema_version": ONLINE_SAMPLE_SCHEMA_VERSION,
        "feature_registry_version": FEATURE_REGISTRY_VERSION,
        "npz_contract_version": SEQUENCE_CONTRACT_VERSION,
        "horizon_seconds": horizon_seconds,
        "smoke_test": bool(smoke_test),
        "preregistration": preregistration_reference,
        "sample_counts": _split_summary(table),
        "house_split": split.to_dict(),
        "known_split_constraint": str(raw_config["split"].get("known_constraint", "")),
        "model_parameter_count": model_config["parameter_count"],
        "input_dimension_values": len(ONLINE_FEATURE_COLUMNS),
        "input_dimension_masks": len(ONLINE_MASK_COLUMNS),
        "loss": {
            "raw": "BCEWithLogits(logit_i,label_i,pos_weight=train_weighted_neg/train_weighted_pos)",
            "reduction": "sum(sample_weight_i * raw_i) / sum(sample_weight_i)",
            "weighted_sampler_used": False,
            "pos_weight": fitted.pos_weight,
        },
        "checkpoint_selection": "minimum validation sample-weighted BCE using train-only pos_weight",
        "threshold_selection": threshold.to_dict(),
        "high_confidence_definition": (
            "label_type in {core_positive,hard_negative} AND "
            "label_confidence=high AND sample_weight=1.0"
        ),
        "test_used_for_selection": False,
        "checkpoint_purpose": "inference_only；不包含 optimizer/RNG，不能精确断点续训。",
        "input_artifacts": input_artifacts,
        "configuration_artifacts": configuration_artifacts,
        "implementation_artifacts": implementation_artifacts,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "device": "cpu",
        },
    }

    staging = Path(tempfile.mkdtemp(prefix=f".{run_id}.staging-", dir=run_root))
    try:
        validation_predictions.to_csv(
            staging / "validation_predictions.csv", index=False, encoding="utf-8-sig"
        )
        test_predictions.to_csv(
            staging / "predictions.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(fitted.history).to_csv(
            staging / "training_history.csv", index=False, encoding="utf-8-sig"
        )
        _json_dump(staging / "metrics.json", metrics)
        _json_dump(staging / "model_config.json", model_config)
        _json_dump(staging / "feature_schema.json", feature_schema)
        _json_dump(staging / "scaler.json", scaler.to_dict())
        _json_dump(staging / "splits.json", split.to_dict())
        (staging / "input_manifest_snapshot.json").write_bytes(
            input_manifest_snapshot_bytes
        )
        joblib.dump(scaler, staging / "scaler.joblib")
        torch.save(
            {
                "training_schema_version": TRAINING_SCHEMA_VERSION,
                "input_schema_version": ONLINE_SAMPLE_SCHEMA_VERSION,
                "feature_registry_version": FEATURE_REGISTRY_VERSION,
                "horizon_seconds": horizon_seconds,
                "model_state_dict": fitted.model.state_dict(),
                "model_config": model_config,
                "feature_schema": feature_schema,
                "scaler": scaler.to_dict(),
                "validation_threshold": threshold.to_dict(),
                "pos_weight": fitted.pos_weight,
            },
            staging / "best_checkpoint.pt",
        )
        (staging / "config_snapshot.toml").write_bytes(config_snapshot_bytes)
        (staging / "taxonomy_snapshot.yaml").write_bytes(taxonomy_snapshot_bytes)
        (staging / "run_report.md").write_text(
            _render_run_report(
                run_id=run_id,
                smoke_test=smoke_test,
                horizon_seconds=horizon_seconds,
                model_config=model_config,
                metrics=metrics,
                feature_schema=feature_schema,
                split_constraint=str(raw_config["split"].get("known_constraint", "")),
            ),
            encoding="utf-8",
        )
        run_manifest["output_artifacts"] = [
            _describe_file(path, staging)
            for path in sorted(staging.iterdir())
            if path.is_file()
        ]
        run_manifest["manifest_self_hash"] = (
            "not_applicable: run_manifest is the completion marker and excludes self-reference"
        )
        _json_dump(staging / "run_manifest.json", run_manifest)
        if final_dir.exists():
            raise FileExistsError(f"运行目录在训练期间出现，拒绝覆盖：{final_dir}")
        staging.replace(final_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "run_dir": final_dir,
        "metrics": metrics,
        "manifest": run_manifest,
    }
