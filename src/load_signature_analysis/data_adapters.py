"""Adapt public dataset storage formats to a common household power representation."""

from __future__ import annotations

from .data_paths import resolve_project_path

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ApplianceChannel:
    """Describe an appliance channel used only for event localization and label construction."""

    dataset: str
    house_id: int
    channel_id: int
    source_field: str
    canonical_label: str
    semantic_class: str
    binary_label: int
    taxonomy_role: str


@dataclass
class HousePowerData:
    """Common household timeline, aggregate power and selected label channels."""

    dataset: str
    house_id: int
    source_path: Path
    timestamp_s: np.ndarray
    aggregate_power_w: np.ndarray
    aggregate_valid_mask: np.ndarray
    appliance_power_w: dict[int, np.ndarray]
    appliance_valid_mask: np.ndarray
    quality_field_available: bool
    audit: dict[str, object]


class DatasetAdapter(Protocol):
    """Minimal reader interface for an additional dataset."""

    def list_houses(self) -> list[int]: ...

    def appliance_channels(
        self, house_id: int, taxonomy_roles: Sequence[str]
    ) -> list[ApplianceChannel]: ...

    def load_house(
        self, house_id: int, channels: Sequence[ApplianceChannel]
    ) -> HousePowerData: ...


def _resolve_project_path(project_root: Path, value: str | Path) -> Path:
    return resolve_project_path(project_root, str(value))


class RefitCsvAdapter:
    """Read REFIT wide-table CSV; keep dataset-specific rules inside this adapter."""

    dataset = "refit"

    def __init__(
        self,
        project_root: Path,
        appliance_inventory_path: Path,
        aggregate_inventory_path: Path,
        chunksize: int = 500_000,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.appliance_inventory_path = _resolve_project_path(
            self.project_root, appliance_inventory_path
        )
        self.aggregate_inventory_path = _resolve_project_path(
            self.project_root, aggregate_inventory_path
        )
        self.chunksize = int(chunksize)
        if self.chunksize <= 0:
            raise ValueError("REFIT chunksize 必须大于 0")
        self._appliance_inventory = pd.read_csv(
            self.appliance_inventory_path, encoding="utf-8-sig"
        )
        self._aggregate_inventory = pd.read_csv(
            self.aggregate_inventory_path, encoding="utf-8-sig"
        )

    def list_houses(self) -> list[int]:
        rows = self._aggregate_inventory[
            (self._aggregate_inventory["dataset"] == self.dataset)
            & self._aggregate_inventory["house_policy_primary_eligible"].astype(bool)
        ]
        return sorted(int(value) for value in rows["house_id"].unique())

    def appliance_channels(
        self,
        house_id: int,
        taxonomy_roles: Sequence[str] = ("core_positive", "hard_negative"),
    ) -> list[ApplianceChannel]:
        roles = set(str(role) for role in taxonomy_roles)
        rows = self._appliance_inventory[
            (self._appliance_inventory["dataset"] == self.dataset)
            & (self._appliance_inventory["house_id"] == int(house_id))
            & self._appliance_inventory["include_in_main_mvp"].astype(bool)
            & self._appliance_inventory["taxonomy_role"].isin(roles)
        ].sort_values("channel_id")
        channels: list[ApplianceChannel] = []
        for row in rows.itertuples(index=False):
            if str(row.taxonomy_confidence) != "high":
                raise ValueError("主池通道必须是 high-confidence")
            if str(row.measurement_unit) != "W":
                raise ValueError("REFIT 主池通道必须是有功功率 W")
            if pd.isna(row.binary_label):
                raise ValueError("REFIT 主池通道缺少二分类标签")
            channels.append(
                ApplianceChannel(
                    dataset=self.dataset,
                    house_id=int(house_id),
                    channel_id=int(row.channel_id),
                    source_field=str(row.source_column),
                    canonical_label=str(row.canonical_label),
                    semantic_class=str(row.semantic_class),
                    binary_label=int(row.binary_label),
                    taxonomy_role=str(row.taxonomy_role),
                )
            )
        if not channels:
            raise ValueError(f"REFIT House {house_id} 没有符合配置的可靠通道")
        return channels

    def _aggregate_source(self, house_id: int) -> tuple[Path, dict[str, object]]:
        rows = self._aggregate_inventory[
            (self._aggregate_inventory["dataset"] == self.dataset)
            & (self._aggregate_inventory["house_id"] == int(house_id))
            & (self._aggregate_inventory["source_kind"] == "primary_inventory_source")
        ]
        if len(rows) != 1:
            raise ValueError(f"REFIT House {house_id} 缺少唯一的主总表来源")
        row = rows.iloc[0]
        if not bool(row["active_power_physically_compatible"]):
            raise ValueError("REFIT 总表不是有功功率兼容来源")
        if not bool(row["house_policy_primary_eligible"]):
            raise ValueError(f"REFIT House {house_id} 被 house policy 排除")
        source_path = _resolve_project_path(
            self.project_root, str(row["source_file"])
        ).resolve()
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        return source_path, row.to_dict()

    def load_house(
        self,
        house_id: int,
        channels: Sequence[ApplianceChannel],
    ) -> HousePowerData:
        """Read chunks into compact arrays, keeping only one household in memory at a time."""

        source_path, source_record = self._aggregate_source(house_id)
        header = pd.read_csv(source_path, nrows=0).columns.tolist()
        required = ["Unix", "Aggregate", *(channel.source_field for channel in channels)]
        missing = sorted(set(required) - set(header))
        if missing:
            raise ValueError(f"REFIT House {house_id} 缺少字段：{missing}")
        has_issues = "Issues" in header
        usecols = [*required, *(["Issues"] if has_issues else [])]
        dtype = {column: "float32" for column in usecols if column != "Unix"}
        dtype["Unix"] = "float64"

        pieces: dict[str, list[np.ndarray]] = {column: [] for column in usecols}
        for chunk in pd.read_csv(
            source_path,
            usecols=usecols,
            dtype=dtype,
            chunksize=self.chunksize,
        ):
            for column in usecols:
                pieces[column].append(chunk[column].to_numpy(copy=True))
        arrays = {
            column: np.concatenate(column_pieces)
            for column, column_pieces in pieces.items()
        }
        timestamps = arrays["Unix"].astype(np.float64, copy=False)
        aggregate = arrays["Aggregate"].astype(np.float32, copy=False)
        timestamp_finite = np.isfinite(timestamps)
        quality_valid = (
            arrays["Issues"] == 0.0
            if has_issues
            else np.ones(timestamps.size, dtype=bool)
        )
        # Public CLEAN data uses 0 for long gaps; real household power should not stay exactly zero for long periods.
        aggregate_valid = (
            timestamp_finite
            & np.isfinite(aggregate)
            & (aggregate > 0.0)
            & quality_valid
        )
        appliance_valid = timestamp_finite & quality_valid & (aggregate > 0.0)
        appliance_power = {
            channel.channel_id: arrays[channel.source_field].astype(
                np.float32, copy=False
            )
            for channel in channels
        }
        audit = {
            "source_file": source_path.as_posix(),
            "source_record": source_record,
            "record_count": int(timestamps.size),
            "quality_field_available": has_issues,
            "issues_record_count": int(np.count_nonzero(~quality_valid)),
            "aggregate_nonpositive_count": int(
                np.count_nonzero(np.isfinite(aggregate) & (aggregate <= 0.0))
            ),
            "aggregate_valid_record_count": int(np.count_nonzero(aggregate_valid)),
        }
        return HousePowerData(
            dataset=self.dataset,
            house_id=int(house_id),
            source_path=source_path,
            timestamp_s=timestamps,
            aggregate_power_w=aggregate,
            aggregate_valid_mask=aggregate_valid,
            appliance_power_w=appliance_power,
            appliance_valid_mask=appliance_valid,
            quality_field_available=has_issues,
            audit=audit,
        )
