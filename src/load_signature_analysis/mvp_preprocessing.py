"""Household splits, missingness-aware standardization and binary evaluation for the MVP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


SPLIT_NAMES = ("train", "validation", "test")


def _normalise_house_id(value: object) -> str:
    """Normalize numeric household identifiers to strings without a decimal suffix."""

    if value is None or value is pd.NA:
        raise ValueError("家庭编号不能为空")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value) or not float(value).is_integer():
            raise ValueError(f"家庭编号不是有限整数：{value!r}")
        return str(int(value))
    text = str(value).strip()
    if not text:
        raise ValueError("家庭编号不能为空")
    # After a CSV round trip, an integer column may contain strings such as ``5.0``.
    try:
        numeric = float(text)
    except ValueError:
        return text
    if np.isfinite(numeric) and numeric.is_integer():
        return str(int(numeric))
    return text


@dataclass(frozen=True, order=True)
class HouseKey:
    """Identify a household by both dataset and household number, avoiding collisions across datasets."""

    dataset: str
    house_id: str

    def __post_init__(self) -> None:
        if self.dataset is None or self.dataset is pd.NA:
            raise ValueError("dataset 不能为空")
        dataset = str(self.dataset).strip()
        if not dataset or ":" in dataset:
            raise ValueError("dataset 不能为空且不能包含冒号")
        house_id = _normalise_house_id(self.house_id)
        if ":" in house_id:
            raise ValueError("house_id 不能包含冒号")
        object.__setattr__(self, "dataset", dataset)
        object.__setattr__(self, "house_id", house_id)

    @classmethod
    def parse(cls, value: object) -> HouseKey:
        """Accept a ``dataset:house`` string or a two-element sequence."""

        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            parts = value.split(":")
            if len(parts) != 2:
                raise ValueError(f"家庭键必须写成 dataset:house_id：{value!r}")
            return cls(parts[0], parts[1])
        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            if len(value) != 2:
                raise ValueError(f"家庭键序列必须恰好包含两个元素：{value!r}")
            return cls(str(value[0]), _normalise_house_id(value[1]))
        raise TypeError(f"无法解析家庭键：{value!r}")

    def serialise(self) -> str:
        return f"{self.dataset}:{self.house_id}"


@dataclass(frozen=True)
class ExplicitHouseSplit:
    """Frozen explicit household-level split; never randomly split event rows."""

    train: tuple[HouseKey, ...]
    validation: tuple[HouseKey, ...]
    test: tuple[HouseKey, ...]

    @classmethod
    def from_mapping(cls, assignments: Mapping[str, Sequence[object]]) -> ExplicitHouseSplit:
        unknown = set(assignments) - set(SPLIT_NAMES)
        missing = set(SPLIT_NAMES) - set(assignments)
        if unknown:
            raise ValueError(f"存在未知 split：{sorted(unknown)}")
        if missing:
            raise ValueError(f"缺少 split：{sorted(missing)}")
        parsed = {
            split: tuple(HouseKey.parse(value) for value in assignments[split])
            for split in SPLIT_NAMES
        }
        split_object = cls(**parsed)
        split_object._assignment_lookup()
        return split_object

    def _assignment_lookup(self) -> dict[HouseKey, str]:
        lookup: dict[HouseKey, str] = {}
        for split in SPLIT_NAMES:
            for key in getattr(self, split):
                if key in lookup:
                    raise ValueError(
                        f"家庭 {key.serialise()} 被重复分配到 "
                        f"{lookup[key]} 与 {split}"
                    )
                lookup[key] = split
        return lookup

    def to_dict(self) -> dict[str, list[str]]:
        """Return a stable representation suitable for JSON/TOML manifests."""

        return {
            split: [key.serialise() for key in getattr(self, split)]
            for split in SPLIT_NAMES
        }

    def assign(
        self,
        events: pd.DataFrame,
        *,
        require_all_houses: bool = True,
        require_both_classes: bool = False,
        overlap_cluster_column: str = "window_overlap_cluster_id",
    ) -> pd.DataFrame:
        """Attach splits to an event table and check household/overlapping-window leakage."""

        required = {"dataset", "house_id"}
        missing_columns = required - set(events.columns)
        if missing_columns:
            raise ValueError(f"事件表缺少列：{sorted(missing_columns)}")
        lookup = self._assignment_lookup()
        observed_keys = {
            HouseKey(str(dataset), _normalise_house_id(house_id))
            for dataset, house_id in events[["dataset", "house_id"]].itertuples(
                index=False, name=None
            )
        }
        assigned_keys = set(lookup)
        missing_assignments = observed_keys - assigned_keys
        unused_assignments = assigned_keys - observed_keys
        if missing_assignments:
            formatted = sorted(key.serialise() for key in missing_assignments)
            raise ValueError(f"以下事件家庭没有 split 分配：{formatted}")
        if require_all_houses and unused_assignments:
            formatted = sorted(key.serialise() for key in unused_assignments)
            raise ValueError(f"split 清单中的家庭没有事件：{formatted}")

        result = events.copy()
        result["split"] = [
            lookup[HouseKey(str(dataset), _normalise_house_id(house_id))]
            for dataset, house_id in result[["dataset", "house_id"]].itertuples(
                index=False, name=None
            )
        ]
        assert_no_split_leakage(
            result,
            overlap_cluster_column=overlap_cluster_column,
        )
        if require_both_classes:
            if "label" not in result.columns:
                raise ValueError("检查 split 类别覆盖时事件表必须包含 label")
            _validate_binary_labels(result["label"].to_numpy())
            missing_class_splits = [
                split
                for split in SPLIT_NAMES
                if result.loc[result["split"] == split, "label"].nunique() != 2
            ]
            if missing_class_splits:
                raise ValueError(
                    "以下 split 没有同时包含正负类："
                    f"{missing_class_splits}"
                )
        return result


def assert_no_split_leakage(
    events: pd.DataFrame,
    *,
    split_column: str = "split",
    overlap_cluster_column: str = "window_overlap_cluster_id",
) -> None:
    """Verify that neither a household nor an overlapping-window cluster crosses splits."""

    required = {"dataset", "house_id", split_column}
    missing_columns = required - set(events.columns)
    if missing_columns:
        raise ValueError(f"事件表缺少列：{sorted(missing_columns)}")
    invalid_splits = set(events[split_column].dropna().astype(str)) - set(SPLIT_NAMES)
    if invalid_splits:
        raise ValueError(f"事件表包含未知 split：{sorted(invalid_splits)}")
    if events[split_column].isna().any():
        raise ValueError("事件表存在未分配 split 的行")

    house_split_counts = events.groupby(["dataset", "house_id"], dropna=False)[
        split_column
    ].nunique(dropna=False)
    leaking_houses = house_split_counts[house_split_counts > 1]
    if not leaking_houses.empty:
        raise ValueError(f"检测到 house-level split 泄漏：{list(leaking_houses.index)}")

    if overlap_cluster_column not in events.columns:
        return
    cluster_values = events[overlap_cluster_column]
    has_cluster = cluster_values.notna() & cluster_values.astype(str).str.strip().ne("")
    clustered = events.loc[has_cluster]
    if clustered.empty:
        return
    cluster_audit = clustered.groupby(overlap_cluster_column, dropna=False).agg(
        dataset_count=("dataset", "nunique"),
        house_count=("house_id", "nunique"),
        split_count=(split_column, "nunique"),
    )
    leaking_clusters = cluster_audit[
        (cluster_audit["dataset_count"] != 1)
        | (cluster_audit["house_count"] != 1)
        | (cluster_audit["split_count"] != 1)
    ]
    if not leaking_clusters.empty:
        raise ValueError(
            "检测到重叠窗口簇跨数据集、家庭或 split："
            f"{list(leaking_clusters.index)}"
        )


def _validate_matrix_pair(
    values: np.ndarray,
    masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values_array = np.asarray(values, dtype=np.float64)
    masks_array = np.asarray(masks)
    if values_array.ndim != 2 or masks_array.ndim != 2:
        raise ValueError("values 与 masks 必须都是二维矩阵")
    if values_array.shape != masks_array.shape:
        raise ValueError("values 与 masks 的形状必须一致")
    if masks_array.dtype == np.bool_:
        boolean_masks = masks_array.copy()
    else:
        try:
            numeric_masks = masks_array.astype(np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("masks 只能包含 0 或 1") from error
        if not np.all(np.isfinite(numeric_masks)) or not np.all(
            (numeric_masks == 0.0) | (numeric_masks == 1.0)
        ):
            raise ValueError("masks 只能包含 0 或 1")
        boolean_masks = numeric_masks.astype(bool)
    if not np.all(np.isfinite(values_array[boolean_masks])):
        raise ValueError("mask=1 的观测值必须是有限数")
    return values_array, boolean_masks


class MaskedStandardizer:
    """Fit feature-wise standardization using only training observations with mask=1."""

    def __init__(
        self,
        feature_names: Sequence[str],
        *,
        variance_epsilon: float = 1e-12,
    ) -> None:
        names = tuple(str(name) for name in feature_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("feature_names 必须非空且不能重复")
        if not np.isfinite(variance_epsilon) or variance_epsilon < 0.0:
            raise ValueError("variance_epsilon 必须是有限非负数")
        self.feature_names = names
        self.variance_epsilon = float(variance_epsilon)

    def fit(self, values: np.ndarray, masks: np.ndarray) -> MaskedStandardizer:
        values_array, masks_array = _validate_matrix_pair(values, masks)
        if values_array.shape[1] != len(self.feature_names):
            raise ValueError("训练矩阵列数与 feature_names 不一致")
        if values_array.shape[0] == 0:
            raise ValueError("不能用空训练集拟合标准化器")

        observed_count = masks_array.sum(axis=0, dtype=np.int64)
        supported = observed_count > 0
        means = np.zeros(values_array.shape[1], dtype=np.float64)
        scales = np.ones(values_array.shape[1], dtype=np.float64)
        for feature_index in np.flatnonzero(supported):
            observed = values_array[masks_array[:, feature_index], feature_index]
            means[feature_index] = float(np.mean(observed))
            standard_deviation = float(np.std(observed, ddof=0))
            if standard_deviation > self.variance_epsilon:
                scales[feature_index] = standard_deviation

        self.mean_ = means
        self.scale_ = scales
        self.observed_count_ = observed_count
        self.supported_ = supported
        self.n_features_in_ = values_array.shape[1]
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "mean_"):
            raise RuntimeError("MaskedStandardizer 尚未拟合")

    def transform(
        self,
        values: np.ndarray,
        masks: np.ndarray,
        *,
        feature_names: Sequence[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return standardized values and valid masks, with every missing value exactly zero."""

        self._check_fitted()
        if feature_names is not None and tuple(feature_names) != self.feature_names:
            raise ValueError("transform 的特征顺序与拟合时不一致")
        values_array, masks_array = _validate_matrix_pair(values, masks)
        if values_array.shape[1] != self.n_features_in_:
            raise ValueError("transform 矩阵列数与拟合统计量不一致")

        effective_masks = masks_array & self.supported_[None, :]
        transformed = np.zeros_like(values_array, dtype=np.float64)
        observed_rows, observed_columns = np.nonzero(effective_masks)
        transformed[observed_rows, observed_columns] = (
            values_array[observed_rows, observed_columns] - self.mean_[observed_columns]
        ) / self.scale_[observed_columns]
        if not np.all(np.isfinite(transformed)):
            raise ValueError("标准化结果包含 NaN 或无穷值")
        return transformed, effective_masks.astype(np.int8)

    def fit_transform(
        self,
        values: np.ndarray,
        masks: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.fit(values, masks).transform(values, masks)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible frozen statistics."""

        self._check_fitted()
        return {
            "schema_version": "1.0",
            "feature_names": list(self.feature_names),
            "variance_epsilon": self.variance_epsilon,
            "mean": self.mean_.tolist(),
            "scale": self.scale_.tolist(),
            "observed_count": self.observed_count_.astype(int).tolist(),
            "supported": self.supported_.astype(bool).tolist(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> MaskedStandardizer:
        """Restore a standardizer from ``to_dict`` output and validate all statistics."""

        if str(payload.get("schema_version")) != "1.0":
            raise ValueError("不支持的 MaskedStandardizer schema_version")
        feature_names = payload.get("feature_names")
        if not isinstance(feature_names, Sequence) or isinstance(feature_names, str):
            raise ValueError("序列化数据缺少 feature_names")
        instance = cls(
            [str(name) for name in feature_names],
            variance_epsilon=float(payload.get("variance_epsilon", 1e-12)),
        )
        expected_length = len(instance.feature_names)
        mean = np.asarray(payload.get("mean"), dtype=np.float64)
        scale = np.asarray(payload.get("scale"), dtype=np.float64)
        observed_count = np.asarray(payload.get("observed_count"), dtype=np.int64)
        supported = np.asarray(payload.get("supported"), dtype=bool)
        arrays = (mean, scale, observed_count, supported)
        if any(array.ndim != 1 or array.size != expected_length for array in arrays):
            raise ValueError("序列化统计量长度与 feature_names 不一致")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
            raise ValueError("序列化统计量包含 NaN 或无穷值")
        if np.any(scale <= 0.0) or np.any(observed_count < 0):
            raise ValueError("序列化 scale/count 非法")
        if not np.array_equal(supported, observed_count > 0):
            raise ValueError("序列化 supported 与 observed_count 不一致")
        instance.mean_ = mean
        instance.scale_ = scale
        instance.observed_count_ = observed_count
        instance.supported_ = supported
        instance.n_features_in_ = expected_length
        return instance


@dataclass(frozen=True)
class ThresholdSelection:
    """Validation-set threshold selection result."""

    threshold: float
    f1: float
    balanced_accuracy: float
    recall: float
    candidate_count: int
    selection_rule: str = (
        "max_f1_then_balanced_accuracy_then_recall_then_closest_to_0.5"
    )

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "threshold": self.threshold,
            "f1": self.f1,
            "balanced_accuracy": self.balanced_accuracy,
            "recall": self.recall,
            "candidate_count": self.candidate_count,
            "selection_rule": self.selection_rule,
        }


def _validate_binary_labels(labels: np.ndarray) -> np.ndarray:
    labels_array = np.asarray(labels)
    if labels_array.ndim != 1:
        raise ValueError("labels 必须是一维数组")
    if labels_array.size == 0:
        raise ValueError("labels 不能为空")
    try:
        numeric = labels_array.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("labels 只能包含 0 或 1") from error
    if not np.all(np.isfinite(numeric)) or not np.all(
        (numeric == 0.0) | (numeric == 1.0)
    ):
        raise ValueError("labels 只能包含 0 或 1")
    return numeric.astype(np.int64)


def _validate_probabilities(probabilities: np.ndarray, expected_size: int) -> np.ndarray:
    probabilities_array = np.asarray(probabilities, dtype=np.float64)
    if probabilities_array.ndim != 1 or probabilities_array.size != expected_size:
        raise ValueError("probabilities 必须与 labels 等长且为一维数组")
    if not np.all(np.isfinite(probabilities_array)):
        raise ValueError("probabilities 不能包含 NaN 或无穷值")
    if np.any(probabilities_array < 0.0) or np.any(probabilities_array > 1.0):
        raise ValueError("probabilities 必须位于 [0, 1]")
    return probabilities_array


def select_validation_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> ThresholdSelection:
    """Select a threshold from validation data only, using fixed reproducible tie rules."""

    labels_array = _validate_binary_labels(labels)
    probabilities_array = _validate_probabilities(probabilities, labels_array.size)
    if np.unique(labels_array).size != 2:
        raise ValueError("阈值选择要求验证集同时包含正负类")

    candidates = np.unique(np.concatenate(([0.0, 1.0], probabilities_array)))
    best_values: tuple[float, float, float, float, float] | None = None
    best_result: ThresholdSelection | None = None
    for threshold in candidates:
        predicted = (probabilities_array >= threshold).astype(np.int64)
        f1 = float(f1_score(labels_array, predicted, zero_division=0))
        balanced = float(balanced_accuracy_score(labels_array, predicted))
        recall = float(recall_score(labels_array, predicted, zero_division=0))
        # The final criterion only breaks an otherwise symmetric tie, choosing the lower threshold.
        ranking = (f1, balanced, recall, -abs(float(threshold) - 0.5), -float(threshold))
        if best_values is None or ranking > best_values:
            best_values = ranking
            best_result = ThresholdSelection(
                threshold=float(threshold),
                f1=f1,
                balanced_accuracy=balanced,
                recall=recall,
                candidate_count=int(candidates.size),
            )
    assert best_result is not None
    return best_result


def compute_binary_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    """Compute JSON-compatible binary metrics; safely return ``None`` for AUC in single-class subgroups."""

    labels_array = _validate_binary_labels(labels)
    probabilities_array = _validate_probabilities(probabilities, labels_array.size)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold 必须位于 [0, 1]")
    predicted = (probabilities_array >= threshold).astype(np.int64)
    matrix = confusion_matrix(labels_array, predicted, labels=[0, 1])
    two_classes = np.unique(labels_array).size == 2
    return {
        "sample_count": int(labels_array.size),
        "negative_count": int(np.count_nonzero(labels_array == 0)),
        "positive_count": int(np.count_nonzero(labels_array == 1)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels_array, predicted)),
        "precision": float(precision_score(labels_array, predicted, zero_division=0)),
        "recall": float(recall_score(labels_array, predicted, zero_division=0)),
        "f1": float(f1_score(labels_array, predicted, zero_division=0)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(labels_array, predicted))
            if two_classes
            else None
        ),
        "roc_auc": (
            float(roc_auc_score(labels_array, probabilities_array))
            if two_classes
            else None
        ),
        "pr_auc": (
            float(average_precision_score(labels_array, probabilities_array))
            if two_classes
            else None
        ),
        "pr_auc_method": "average_precision",
        "confusion_matrix": matrix.astype(int).tolist(),
    }
