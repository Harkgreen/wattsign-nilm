"""A small missingness-aware MLP with grouped encoders.

This module handles the model input contract only: flat values and per-feature masks from one fixed feature space are divided into configured groups. Dataset, household, appliance and true-duration metadata must never enter the model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeAlias

import torch
from torch import Tensor, nn


GroupIndex: TypeAlias = slice | Sequence[int] | Tensor
HiddenDimensions: TypeAlias = int | Sequence[int]


def _as_boolean_mask(mask: Tensor, *, name: str = "feature_mask") -> Tensor:
    """Validate a binary mask and convert it to a boolean tensor."""

    if mask.dtype == torch.bool:
        return mask
    if not torch.is_floating_point(mask) and mask.dtype not in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise TypeError(f"{name} 必须是布尔或数值张量")
    if not bool(torch.isfinite(mask).all()):
        raise ValueError(f"{name} 不能包含 NaN 或无穷值")
    if not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError(f"{name} 只能包含 0 和 1")
    return mask.bool()


def _indices_from_specification(specification: GroupIndex) -> tuple[int, ...]:
    """Convert a slice or explicit indices to an auditable immutable index tuple."""

    if isinstance(specification, slice):
        if specification.stop is None:
            raise ValueError("group slice 必须提供有限的 stop")
        start = 0 if specification.start is None else specification.start
        step = 1 if specification.step is None else specification.step
        if step <= 0:
            raise ValueError("group slice 的 step 必须大于 0")
        return tuple(range(start, specification.stop, step))
    if isinstance(specification, Tensor):
        if specification.ndim != 1:
            raise ValueError("group 索引张量必须是一维")
        return tuple(int(value) for value in specification.detach().cpu().tolist())
    return tuple(int(value) for value in specification)


def _build_group_indices(
    group_dimensions: Mapping[str, int],
    group_slices: Mapping[str, GroupIndex] | None,
) -> tuple[tuple[str, ...], dict[str, tuple[int, ...]], int]:
    """Freeze group order and indices, requiring every input feature to occur in exactly one group."""

    if not group_dimensions:
        raise ValueError("至少需要配置一个特征组")

    group_order = tuple(group_dimensions)
    if any(not isinstance(name, str) or not name for name in group_order):
        raise ValueError("特征组名称必须是非空字符串")
    if any("." in name for name in group_order):
        raise ValueError("特征组名称不能包含句点")

    dimensions: dict[str, int] = {}
    for name, raw_dimension in group_dimensions.items():
        dimension = int(raw_dimension)
        if dimension <= 0 or dimension != raw_dimension:
            raise ValueError(f"特征组 {name!r} 的维度必须是正整数")
        dimensions[name] = dimension

    if group_slices is None:
        indices: dict[str, tuple[int, ...]] = {}
        start = 0
        for name in group_order:
            stop = start + dimensions[name]
            indices[name] = tuple(range(start, stop))
            start = stop
        return group_order, indices, start

    if set(group_slices) != set(group_order):
        missing = sorted(set(group_order) - set(group_slices))
        unexpected = sorted(set(group_slices) - set(group_order))
        raise ValueError(
            f"group_slices 与 group_dimensions 不一致；缺少={missing}，额外={unexpected}"
        )

    indices = {}
    flattened: list[int] = []
    for name in group_order:
        group_indices = _indices_from_specification(group_slices[name])
        if len(group_indices) != dimensions[name]:
            raise ValueError(
                f"特征组 {name!r} 声明维度为 {dimensions[name]}，"
                f"但索引数量为 {len(group_indices)}"
            )
        if any(index < 0 for index in group_indices):
            raise ValueError(f"特征组 {name!r} 不能使用负索引")
        if len(set(group_indices)) != len(group_indices):
            raise ValueError(f"特征组 {name!r} 内存在重复索引")
        indices[name] = group_indices
        flattened.extend(group_indices)

    if len(set(flattened)) != len(flattened):
        raise ValueError("不同特征组不能重复使用同一输入特征")
    input_dimension = max(flattened) + 1
    expected = set(range(input_dimension))
    if set(flattened) != expected:
        missing_indices = sorted(expected - set(flattened))
        raise ValueError(f"group_slices 没有覆盖全部输入特征：{missing_indices}")
    return group_order, indices, input_dimension


def _normalise_hidden_dimensions(
    dimensions: HiddenDimensions,
    *,
    name: str,
) -> tuple[int, ...]:
    """Normalize a single width or multiple widths to a nonempty tuple of positive integers."""

    if isinstance(dimensions, bool):
        raise ValueError(f"{name} 必须是正整数或正整数序列")
    if isinstance(dimensions, int):
        values: tuple[object, ...] = (dimensions,)
    elif isinstance(dimensions, Sequence) and not isinstance(dimensions, str):
        values = tuple(dimensions)
    else:
        raise ValueError(f"{name} 必须是正整数或正整数序列")

    if not values:
        raise ValueError(f"{name} 不能为空")

    normalised: list[int] = []
    for raw_dimension in values:
        if isinstance(raw_dimension, bool):
            raise ValueError(f"{name} 只能包含正整数")
        try:
            dimension = int(raw_dimension)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} 只能包含正整数") from error
        if dimension <= 0 or dimension != raw_dimension:
            raise ValueError(f"{name} 只能包含正整数")
        normalised.append(dimension)
    return tuple(normalised)


class GroupEncoder(nn.Module):
    """Encode group values and per-feature masks into a fixed-dimensional embedding.
    
    ``hidden_dimension`` accepts one width or a sequence of layer widths from the input to the embedding."""

    def __init__(
        self,
        feature_dimension: int,
        hidden_dimension: HiddenDimensions = 32,
        embedding_dimension: int = 16,
        dropout_probability: float = 0.1,
    ) -> None:
        super().__init__()
        if feature_dimension <= 0:
            raise ValueError("feature_dimension 必须大于 0")
        hidden_dimensions = _normalise_hidden_dimensions(
            hidden_dimension,
            name="Encoder 隐藏维度",
        )
        if embedding_dimension <= 0:
            raise ValueError("Encoder embedding 维度必须大于 0")
        if not 0.0 <= dropout_probability < 1.0:
            raise ValueError("dropout_probability 必须位于 [0, 1)")

        self.feature_dimension = int(feature_dimension)
        self.embedding_dimension = int(embedding_dimension)
        self.hidden_dimensions = hidden_dimensions
        layers: list[nn.Module] = []
        input_dimension = 2 * self.feature_dimension
        for dimension in self.hidden_dimensions:
            layers.extend(
                (
                    nn.Linear(input_dimension, dimension),
                    nn.GELU(),
                    nn.Dropout(dropout_probability),
                )
            )
            input_dimension = dimension
        layers.extend(
            (
                nn.Linear(input_dimension, self.embedding_dimension),
                nn.LayerNorm(self.embedding_dimension),
            )
        )
        self.network = nn.Sequential(*layers)

    def forward(self, values: Tensor, feature_mask: Tensor) -> Tensor:
        """Encode ``[values * mask, mask]`` without propagating NaN from naturally missing positions."""

        if values.ndim != 2 or feature_mask.ndim != 2:
            raise ValueError("GroupEncoder 输入必须是 [batch, features] 二维张量")
        if values.shape != feature_mask.shape:
            raise ValueError("values 与 feature_mask 的形状必须相同")
        if values.device != feature_mask.device:
            raise ValueError("values 与 feature_mask 必须位于同一设备")
        if values.shape[1] != self.feature_dimension:
            raise ValueError(
                f"Encoder 期望 {self.feature_dimension} 个特征，实际为 {values.shape[1]}"
            )

        mask = _as_boolean_mask(feature_mask)
        if not bool(torch.isfinite(values[mask]).all()):
            raise ValueError("mask=1 的已观测特征不能包含 NaN 或无穷值")

        parameter = next(self.parameters())
        values = values.to(dtype=parameter.dtype)
        # Unlike mask multiplication, torch.where safely zeroes missing entries even when their raw value is NaN.
        observed_values = torch.where(mask, values, torch.zeros_like(values))
        encoder_input = torch.cat((observed_values, mask.to(values.dtype)), dim=-1)
        return self.network(encoder_input)


class MaskedGatedFusion(nn.Module):
    """Apply softmax gating only among available feature groups."""

    def __init__(self, group_names: Sequence[str], embedding_dimension: int = 16) -> None:
        super().__init__()
        self.group_order = tuple(group_names)
        if not self.group_order:
            raise ValueError("融合层至少需要一个特征组")
        if len(set(self.group_order)) != len(self.group_order):
            raise ValueError("融合层特征组名称不能重复")
        if embedding_dimension <= 0:
            raise ValueError("embedding_dimension 必须大于 0")

        self.embedding_dimension = int(embedding_dimension)
        self.gates = nn.ModuleDict(
            {name: nn.Linear(self.embedding_dimension, 1) for name in self.group_order}
        )

    def _stack_embeddings(self, embeddings: Mapping[str, Tensor] | Tensor) -> Tensor:
        if isinstance(embeddings, Tensor):
            stacked = embeddings
        else:
            if set(embeddings) != set(self.group_order):
                raise ValueError("embedding 字典的组名称与融合层配置不一致")
            stacked = torch.stack([embeddings[name] for name in self.group_order], dim=1)
        if stacked.ndim != 3:
            raise ValueError("group embeddings 必须是 [batch, groups, embedding] 三维张量")
        expected_tail = (len(self.group_order), self.embedding_dimension)
        if tuple(stacked.shape[1:]) != expected_tail:
            raise ValueError(
                f"group embeddings 末两维应为 {expected_tail}，实际为 {tuple(stacked.shape[1:])}"
            )
        return stacked

    def forward(
        self,
        embeddings: Mapping[str, Tensor] | Tensor,
        group_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return the fused embedding and gate weights in ``group_order``."""

        stacked = self._stack_embeddings(embeddings)
        if group_mask.ndim != 2:
            raise ValueError("group_mask 必须是 [batch, groups] 二维张量")
        if tuple(group_mask.shape) != tuple(stacked.shape[:2]):
            raise ValueError("group_mask 与 group embeddings 的 batch/group 维度不一致")
        if group_mask.device != stacked.device:
            raise ValueError("group_mask 与 group embeddings 必须位于同一设备")
        available = _as_boolean_mask(group_mask, name="group_mask")

        available_counts = available.sum(dim=1)
        invalid_rows = torch.nonzero(available_counts == 0, as_tuple=False).flatten()
        if invalid_rows.numel():
            rows = invalid_rows.detach().cpu().tolist()
            raise ValueError(f"每个样本至少需要一个可用特征组；全组缺失样本索引={rows}")

        scores = torch.cat(
            [
                self.gates[name](stacked[:, group_index, :])
                for group_index, name in enumerate(self.group_order)
            ],
            dim=1,
        )
        scores = scores.masked_fill(~available, -torch.inf)
        weights = torch.softmax(scores, dim=1)

        # Explicitly overwrite once more so missing groups are exactly 0 and a sole available group is exactly 1.
        weights = torch.where(available, weights, torch.zeros_like(weights))
        single_group_rows = available_counts == 1
        weights = torch.where(
            single_group_rows.unsqueeze(1),
            available.to(weights.dtype),
            weights,
        )
        fused = torch.sum(stacked * weights.unsqueeze(-1), dim=1)
        return fused, weights


class GroupedFingerprintMLP(nn.Module):
    """Build grouped encoders, fusion and classification layers from flat values and masks.
    
    ``encoder_hidden_dimension`` and ``classifier_hidden_dimension`` accept one integer width or a sequence of positive widths. Single-layer configurations preserve the existing checkpoint state-dict structure."""

    def __init__(
        self,
        group_dimensions: Mapping[str, int],
        group_slices: Mapping[str, GroupIndex] | None = None,
        *,
        encoder_hidden_dimension: HiddenDimensions = 32,
        embedding_dimension: int = 16,
        classifier_hidden_dimension: HiddenDimensions = 16,
        dropout_probability: float = 0.1,
        feature_dropout_enabled: bool = True,
        group_dropout_probability: float = 0.15,
        feature_dropout_probability: float = 0.05,
    ) -> None:
        super().__init__()
        group_order, group_indices, input_dimension = _build_group_indices(
            group_dimensions,
            group_slices,
        )
        encoder_hidden_dimensions = _normalise_hidden_dimensions(
            encoder_hidden_dimension,
            name="encoder_hidden_dimension",
        )
        classifier_hidden_dimensions = _normalise_hidden_dimensions(
            classifier_hidden_dimension,
            name="classifier_hidden_dimension",
        )
        for name, probability in (
            ("group_dropout_probability", group_dropout_probability),
            ("feature_dropout_probability", feature_dropout_probability),
        ):
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{name} 必须位于 [0, 1]")

        self.group_order = group_order
        self.group_dimensions = {
            name: int(group_dimensions[name]) for name in self.group_order
        }
        self.group_indices = group_indices
        self.input_dimension = input_dimension
        self.feature_dropout_enabled = bool(feature_dropout_enabled)
        self.group_dropout_probability = float(group_dropout_probability)
        self.feature_dropout_probability = float(feature_dropout_probability)
        self.encoder_hidden_dimensions = encoder_hidden_dimensions
        self.classifier_hidden_dimensions = classifier_hidden_dimensions

        self.encoders = nn.ModuleDict(
            {
                name: GroupEncoder(
                    feature_dimension=self.group_dimensions[name],
                    hidden_dimension=self.encoder_hidden_dimensions,
                    embedding_dimension=embedding_dimension,
                    dropout_probability=dropout_probability,
                )
                for name in self.group_order
            }
        )
        self.fusion = MaskedGatedFusion(self.group_order, embedding_dimension)
        classifier_layers: list[nn.Module] = []
        classifier_input_dimension = int(embedding_dimension)
        for dimension in self.classifier_hidden_dimensions:
            classifier_layers.extend(
                (
                    nn.Linear(classifier_input_dimension, dimension),
                    nn.GELU(),
                    nn.Dropout(dropout_probability),
                )
            )
            classifier_input_dimension = dimension
        classifier_layers.append(nn.Linear(classifier_input_dimension, 1))
        self.classifier = nn.Sequential(*classifier_layers)

    def _index_tensor(self, group_name: str, device: torch.device) -> Tensor:
        return torch.tensor(self.group_indices[group_name], dtype=torch.long, device=device)

    def _validate_flat_inputs(self, values: Tensor, feature_mask: Tensor) -> Tensor:
        if values.ndim != 2 or feature_mask.ndim != 2:
            raise ValueError("values 与 feature_mask 必须是 [batch, features] 二维张量")
        if values.shape != feature_mask.shape:
            raise ValueError("values 与 feature_mask 的形状必须相同")
        if values.device != feature_mask.device:
            raise ValueError("values 与 feature_mask 必须位于同一设备")
        if values.shape[1] != self.input_dimension:
            raise ValueError(
                f"模型期望 {self.input_dimension} 个扁平特征，实际为 {values.shape[1]}"
            )
        mask = _as_boolean_mask(feature_mask)
        invalid_rows = torch.nonzero(~mask.any(dim=1), as_tuple=False).flatten()
        if invalid_rows.numel():
            rows = invalid_rows.detach().cpu().tolist()
            raise ValueError(f"每个样本至少需要一个已观测特征；全组缺失样本索引={rows}")
        if not bool(torch.isfinite(values[mask]).all()):
            raise ValueError("mask=1 的已观测特征不能包含 NaN 或无穷值")
        return mask

    def apply_structured_feature_dropout(
        self,
        feature_mask: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Randomly mask groups and features during training, preserving at least one naturally observed group per sample.
        
        The returned mask is always a subset of the original mask and never restores naturally missing positions. In evaluation mode, when disabled, or when both probabilities are zero, return an unchanged boolean mask copy."""

        natural_mask = _as_boolean_mask(feature_mask).clone()
        invalid_rows = torch.nonzero(~natural_mask.any(dim=1), as_tuple=False).flatten()
        if invalid_rows.numel():
            rows = invalid_rows.detach().cpu().tolist()
            raise ValueError(f"不能对全组自然缺失样本执行 dropout；样本索引={rows}")
        if (
            not self.training
            or not self.feature_dropout_enabled
            or (
                self.group_dropout_probability == 0.0
                and self.feature_dropout_probability == 0.0
            )
        ):
            return natural_mask

        effective_mask = natural_mask.clone()
        batch_size = natural_mask.shape[0]

        if self.group_dropout_probability > 0.0:
            for group_name in self.group_order:
                indices = self._index_tensor(group_name, natural_mask.device)
                drop_group = torch.rand(
                    batch_size,
                    device=natural_mask.device,
                    generator=generator,
                ) < self.group_dropout_probability
                if bool(drop_group.any()):
                    rows = torch.nonzero(drop_group, as_tuple=False).flatten()
                    effective_mask[rows.unsqueeze(1), indices.unsqueeze(0)] = False

        if self.feature_dropout_probability > 0.0:
            drop_features = torch.rand(
                effective_mask.shape,
                device=effective_mask.device,
                generator=generator,
            ) < self.feature_dropout_probability
            effective_mask &= ~drop_features

        emptied_rows = torch.nonzero(~effective_mask.any(dim=1), as_tuple=False).flatten()
        for row_tensor in emptied_rows:
            row = int(row_tensor.item())
            naturally_available_groups = [
                name
                for name in self.group_order
                if bool(
                    natural_mask[
                        row,
                        self._index_tensor(name, natural_mask.device),
                    ].any()
                )
            ]
            # Restore only groups originally observed in this sample; never create new observed positions.
            choice = int(
                torch.randint(
                    len(naturally_available_groups),
                    (1,),
                    device=natural_mask.device,
                    generator=generator,
                ).item()
            )
            selected_group = naturally_available_groups[choice]
            indices = self._index_tensor(selected_group, natural_mask.device)
            effective_mask[row, indices] = natural_mask[row, indices]

        if bool((effective_mask & ~natural_mask).any()):
            raise RuntimeError("structured dropout 错误恢复了自然缺失特征")
        if not bool(effective_mask.any(dim=1).all()):
            raise RuntimeError("structured dropout 清空了有效样本")
        return effective_mask

    def forward(self, values: Tensor, masks: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``logits[batch]`` and ``gate_weights[batch, groups]``."""

        natural_mask = self._validate_flat_inputs(values, masks)
        effective_mask = self.apply_structured_feature_dropout(natural_mask)

        embeddings: dict[str, Tensor] = {}
        group_masks: list[Tensor] = []
        for group_name in self.group_order:
            indices = self._index_tensor(group_name, values.device)
            group_values = values.index_select(1, indices)
            group_feature_mask = effective_mask.index_select(1, indices)
            embeddings[group_name] = self.encoders[group_name](
                group_values,
                group_feature_mask,
            )
            group_masks.append(group_feature_mask.any(dim=1))

        group_mask = torch.stack(group_masks, dim=1)
        fused, gate_weights = self.fusion(embeddings, group_mask)
        logits = self.classifier(fused).squeeze(-1)
        return logits, gate_weights
