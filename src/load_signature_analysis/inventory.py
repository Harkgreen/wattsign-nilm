"""Read-only inventory helpers for UK-DALE and REFIT.

The functions in this module scan public dataset files without modifying them and
produce compact, auditable summaries suitable for Phase 1 data governance.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd

from .ukdale import parse_channel_labels


CORE_POSITIVE = {"kettle", "toaster", "electric_heater"}
EXTENDED_POSITIVE = {"hair_dryer", "hairdryer", "straighteners", "iron", "steam_iron"}
HARD_NEGATIVE = {
    "microwave",
    "hoover",
    "vacuum_cleaner",
    "treadmill",
    "running_machine",
}


REFIT_ANOMALIES: dict[tuple[int, int | None], tuple[str, str]] = {
    (1, None): (
        "clean_copy_truncated",
        "CLEAN_House1.csv stops at exactly 1,048,575 data rows; use the full Processed copy.",
    ),
    (3, 9): ("device_replaced_same_class", "Kettle was replaced during data collection."),
    (4, 3): ("device_replaced_same_class", "Fridge-Freezer changed on 2014-08-17."),
    (4, 6): ("signature_change", "Computer Site signature changed on 2015-02-01."),
    (4, 7): ("signature_change", "Television Site signature changed on 2014-12-19."),
    (5, 2): (
        "mixed_channel_after_change",
        "Dehumidifier was added to the Tumble Dryer channel on 2014-11-21.",
    ),
    (5, 5): ("signature_change", "Desktop Computer signature changed on 2015-03-17."),
    (7, 3): ("signature_change", "Freezer signature changed on 2013-11-24."),
    (7, 6): ("signature_change", "Dishwasher signature changed on 2014-05-20."),
    (8, 1): ("device_replaced", "Fridge changed on November 6; year is not explicit in README."),
    (8, 5): (
        "mixed_target_and_nontarget",
        "README notes that Toaster and DAB Radio share this IAM.",
    ),
    (10, 1): (
        "channel_repurposed",
        "Channel changed from Fridge to Blender on 2014-06-17.",
    ),
    (10, 2): (
        "metadata_conflict",
        "README gives the Freezer-to-Toaster change as 2014-06-25, while the metadata "
        "workbook gives 2014-10-17.",
    ),
    (10, 3): (
        "multiple_devices_same_channel",
        "A second freezer was added after April 2015.",
    ),
    (10, 4): ("device_replaced", "Fridge-Freezer was replaced in March 2015."),
    (11, None): ("aggregate_affected_by_solar", "House aggregate is affected by solar panels."),
    (11, 4): ("possible_device_replacement", "README notes a possible dishwasher replacement."),
    (12, 2): ("metadata_conflict", "README and metadata workbook assign different devices."),
    (12, 3): ("metadata_conflict", "README and metadata workbook assign different devices."),
    (12, 4): ("metadata_conflict", "README says Kettle; workbook says Computer Site."),
    (12, 5): ("metadata_conflict", "README says Toaster; workbook says Microwave."),
    (12, 6): ("metadata_conflict", "README says Television Site; workbook says Kettle."),
    (12, 7): ("metadata_conflict", "README says Not Used; workbook says Toaster/Nothing Recorded."),
    (12, 8): ("metadata_conflict", "README and workbook disagree; workbook says Nothing Recorded."),
    (12, 9): ("metadata_conflict", "README and workbook disagree; both indicate unusable data."),
    (13, 2): ("unknown_device", "An unrecorded device is attached to this channel."),
    (13, 5): ("signature_change", "Tumble Dryer signature changes without device metadata."),
    (13, 6): ("metadata_conflict", "README says Television Site; workbook says Network Site."),
    (13, 7): ("metadata_conflict", "README says Computer Site; workbook says Microwave."),
    (15, 8): ("metadata_conflict", "README says Kettle; metadata workbook says Hi-Fi."),
    (16, 3): ("uncertain_label", "Electric Heater (1) is marked with a question mark."),
    (16, 9): ("mixed_device_label", "Channel is labelled Dehumidifier/Heater."),
    (17, 1): ("device_replaced", "Freezer was replaced about one month before recording stopped."),
    (17, 2): ("device_replaced", "Fridge-Freezer was replaced in mid-May 2015."),
    (17, 8): (
        "mixed_target_and_unknown",
        "Kettle channel notes also mention toaster and miscellaneous plug-in devices.",
    ),
    (17, 9): (
        "mixed_plug_site",
        "Bedroom plug site may contain TV, laptop, hair dryer, or phone charger.",
    ),
    (19, 1): ("multiple_devices_same_channel", "A fridge and a freezer share this IAM."),
    (19, 8): ("metadata_conflict", "README says 80 W Lamp; metadata workbook says Games Console."),
    (21, None): ("aggregate_affected_by_solar", "House aggregate is affected by solar panels."),
    (21, 7): ("combined_positive_label", "Kettle and toaster share one IAM channel."),
}


UKDALE_ANOMALIES: dict[tuple[int, int], tuple[str, str]] = {
    (1, 10): (
        "mixed_appliance_meter",
        "Meter metadata includes kettle, food processor, and toasted sandwich maker; "
        "channels 10/11 were also swapped during 2013-06-24 through 2013-06-28.",
    ),
    (1, 11): (
        "mixed_appliance_meter",
        "Meter metadata includes toaster, kitchen aid, and food processor; channels "
        "10/11 were also swapped during 2013-06-24 through 2013-06-28.",
    ),
    (1, 21): ("composite_site", "GigE and USB hub share one channel."),
    (1, 26): ("duplicate_semantic_label", "Second living-room lamp channel."),
    (1, 32): ("composite_site", "Kitchen phone and stereo share one channel."),
    (2, 19): ("broad_cooking_label", "Cooker may contain several heating stages."),
    (4, 2): ("composite_site", "TV, DVD, digibox, and lamp share one channel."),
    (4, 3): ("mixed_target_and_nontarget", "Kettle and radio share one channel."),
    (4, 6): (
        "multiple_appliances_same_channel",
        "Washing machine, microwave, and breadmaker share one channel.",
    ),
    (5, 2): ("composite_site", "Stereo and bedroom speakers share one channel."),
    (5, 20): ("complex_heating_appliance", "Oven has multiple operating stages."),
    (5, 21): ("complex_heating_appliance", "Electric hob has multiple operating stages."),
    (5, 24): ("complex_heating_appliance", "Washer dryer has multiple heating stages."),
}


@dataclass
class TimeAccumulator:
    record_count: int = 0
    valid_timestamp_count: int = 0
    start_timestamp: float | None = None
    end_timestamp: float | None = None
    previous_timestamp: float | None = None
    positive_gap_histogram: Counter[int] = field(default_factory=Counter)
    gap_over_60_count: int = 0
    maximum_gap_seconds: float = 0.0
    nonpositive_gap_count: int = 0

    def update(self, raw_timestamps: np.ndarray) -> None:
        values = np.asarray(raw_timestamps, dtype=float)
        self.record_count += int(values.size)
        values = values[np.isfinite(values)]
        self.valid_timestamp_count += int(values.size)
        if not values.size:
            return

        if self.start_timestamp is None:
            self.start_timestamp = float(values[0])
        self.end_timestamp = float(values[-1])
        if self.previous_timestamp is not None:
            values_for_diff = np.concatenate(([self.previous_timestamp], values))
        else:
            values_for_diff = values
        diffs = np.diff(values_for_diff)
        if diffs.size:
            positive = diffs[diffs > 0]
            self.nonpositive_gap_count += int(np.count_nonzero(diffs <= 0))
            if positive.size:
                rounded = np.rint(positive).astype(np.int64)
                unique, counts = np.unique(rounded, return_counts=True)
                self.positive_gap_histogram.update(
                    {int(value): int(count) for value, count in zip(unique, counts, strict=True)}
                )
                self.gap_over_60_count += int(np.count_nonzero(positive > 60.0))
                self.maximum_gap_seconds = max(self.maximum_gap_seconds, float(np.max(positive)))
        self.previous_timestamp = float(values[-1])

    def _quantile(self, q: float) -> float | None:
        total = sum(self.positive_gap_histogram.values())
        if total == 0:
            return None
        target = max(1, int(np.ceil(q * total)))
        cumulative = 0
        for value, count in sorted(self.positive_gap_histogram.items()):
            cumulative += count
            if cumulative >= target:
                return float(value)
        return float(max(self.positive_gap_histogram))

    def summary(self) -> dict[str, object]:
        return {
            "record_count": self.record_count,
            "valid_timestamp_count": self.valid_timestamp_count,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "start_utc": timestamp_to_iso(self.start_timestamp),
            "end_utc": timestamp_to_iso(self.end_timestamp),
            "sampling_interval_median_seconds": self._quantile(0.5),
            "sampling_interval_p95_seconds": self._quantile(0.95),
            "gap_over_60_seconds_count": self.gap_over_60_count,
            "maximum_gap_seconds": self.maximum_gap_seconds,
            "nonpositive_gap_count": self.nonpositive_gap_count,
        }


@dataclass
class ValueAccumulator:
    record_count: int = 0
    observed_count: int = 0
    missing_count: int = 0
    nonzero_count: int = 0
    negative_count: int = 0
    total: float = 0.0
    minimum: float | None = None
    maximum: float | None = None

    def update(self, raw_values: np.ndarray) -> None:
        values = np.asarray(raw_values, dtype=float)
        self.record_count += int(values.size)
        finite = values[np.isfinite(values)]
        self.observed_count += int(finite.size)
        self.missing_count += int(values.size - finite.size)
        if not finite.size:
            return
        self.nonzero_count += int(np.count_nonzero(finite))
        self.negative_count += int(np.count_nonzero(finite < 0))
        self.total += float(np.sum(finite))
        chunk_minimum = float(np.min(finite))
        chunk_maximum = float(np.max(finite))
        self.minimum = chunk_minimum if self.minimum is None else min(self.minimum, chunk_minimum)
        self.maximum = chunk_maximum if self.maximum is None else max(self.maximum, chunk_maximum)

    def summary(self) -> dict[str, object]:
        return {
            "value_record_count": self.record_count,
            "observed_value_count": self.observed_count,
            "missing_value_count": self.missing_count,
            "missing_fraction": (
                self.missing_count / self.record_count if self.record_count else None
            ),
            "nonzero_value_count": self.nonzero_count,
            "negative_value_count": self.negative_count,
            "mean_raw_value": self.total / self.observed_count if self.observed_count else None,
            "minimum_raw_value": self.minimum,
            "maximum_raw_value": self.maximum,
        }


def timestamp_to_iso(value: float | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()


def normalize_label(value: str) -> str:
    normalized = value.strip().lower().replace("&", " and ")
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    normalized = re.sub(r"_+", "_", normalized)
    return normalized


def canonicalize_label(value: str) -> str:
    label = normalize_label(value)
    replacements = {
        "hairdryer": "hair_dryer",
        "steam_iron": "iron",
        "fridge_and_freezer": "fridge_freezer",
        "fridge_freezer_1": "fridge_freezer",
        "fridge_freezer_2": "fridge_freezer",
        "electric_heater_1": "electric_heater",
        "electric_heater_2": "electric_heater",
        "television": "tv",
        "television_site": "tv_site",
        "television_site_lounge": "tv_site",
        "television_site_bedroom": "tv_site",
    }
    return replacements.get(label, label)


def classify_appliance(
    dataset: str,
    house_id: int,
    channel_id: int,
    source_label: str,
) -> dict[str, object]:
    normalized = normalize_label(source_label)
    canonical = canonicalize_label(source_label)
    role = "ignore"
    confidence = "low"
    reason = "Not yet mapped to a reliable MVP class."

    if canonical == "aggregate":
        role, confidence, reason = "metadata", "high", "Aggregate input, never a target label."
    elif canonical in CORE_POSITIVE:
        role, confidence, reason = (
            "core_positive",
            "high",
            "Dedicated channel for a core resistive-heating appliance.",
        )
    elif canonical in EXTENDED_POSITIVE:
        role, confidence, reason = (
            "extended_clean_positive",
            "high",
            "Dedicated channel for a deployment-relevant resistive-dominant appliance.",
        )
    elif canonical in HARD_NEGATIVE:
        role, confidence, reason = (
            "hard_negative",
            "high",
            "High-power or motor load useful for detecting power-magnitude shortcuts.",
        )
    elif "combination_microwave" in normalized:
        role, confidence, reason = (
            "ignore",
            "low",
            "Combination microwave may contain a resistive grill or oven stage.",
        )
    elif any(
        token in normalized
        for token in (
            "washing_machine",
            "dishwasher",
            "dish_washer",
            "tumble_dryer",
            "washer_dryer",
            "dryer",
            "oven",
            "cooker",
            "electric_hob",
            "breadmaker",
            "bread_maker",
            "rice_cooker",
            "coffee_machine",
            "nespresso",
            "boiler",
            "dehumidifier",
            "soldering_iron",
        )
    ):
        role, confidence, reason = (
            "ignore",
            "high",
            "Complex, mixed, cyclic, or out-of-scope heating behaviour.",
        )
    elif any(token in normalized for token in ("unknown", "not_used", "plug_site")):
        role, confidence, reason = "ignore", "high", "Unknown, unused, or mixed plug-site label."
    elif any(
        token in normalized
        for token in (
            "fridge",
            "freezer",
            "laptop",
            "computer",
            "desktop",
            "monitor",
            "router",
            "modem",
            "tv",
            "television",
            "hifi",
            "hi_fi",
            "radio",
            "speaker",
            "server",
            "playstation",
            "ps4",
            "xbox",
            "fan",
            "pump",
            "treadmill",
            "running_machine",
            "network_attached_storage",
            "hub",
            "charger",
            "phone",
            "tablet",
            "ipad",
            "amp",
            "projector",
            "data_logger_pc",
            "htpc",
            "subwoofer",
            "food_mixer",
            "blender",
            "sky_hd_box",
        )
    ):
        role, confidence, reason = (
            "negative",
            "medium" if any(token in normalized for token in ("site", "and")) else "high",
            "Clearly non-heating electrical load for the first MVP.",
        )
    elif any(token in normalized for token in ("lamp", "light")):
        role, confidence, reason = (
            "negative",
            "medium",
            "Lighting is non-target for the dormitory-oriented heating-appliance definition.",
        )
    elif any(token in normalized for token in ("printer", "vivarium")):
        role, confidence, reason = (
            "ignore",
            "medium",
            "Low-power resistive or potentially mixed load excluded from the first target definition.",
        )

    anomaly = None
    if dataset == "refit":
        anomaly = REFIT_ANOMALIES.get((house_id, channel_id))
    elif dataset == "uk_dale":
        anomaly = UKDALE_ANOMALIES.get((house_id, channel_id))
    if anomaly is not None:
        anomaly_code, anomaly_note = anomaly
        if anomaly_code in {
            "channel_repurposed",
            "mixed_channel_after_change",
            "unknown_device",
            "uncertain_label",
            "mixed_device_label",
            "mixed_plug_site",
            "mixed_target_and_unknown",
            "mixed_target_and_nontarget",
            "multiple_appliances_same_channel",
            "complex_heating_appliance",
            "broad_cooking_label",
            "mixed_appliance_meter",
            "metadata_conflict",
        }:
            role = "ignore"
            confidence = "high"
            reason = f"Excluded because of anomaly: {anomaly_note}"

    if dataset == "uk_dale" and (house_id, channel_id) == (4, 3):
        role, confidence = "ignore", "high"
    if dataset == "refit" and (house_id, channel_id) == (21, 7):
        role, confidence, reason = (
            "core_positive",
            "medium",
            "Both possible appliances are core positive, but the channel is combined and house has solar.",
        )

    if role == "metadata":
        data_use = "metadata"
    elif role == "ignore":
        data_use = "ignore"
    elif role == "extended_clean_positive":
        data_use = "transfer_only"
    else:
        data_use = "main"
    aggregate_quality = "normal"
    if dataset == "refit" and house_id in {11, 21}:
        aggregate_quality = "pv_affected"
        if role != "metadata":
            data_use = "robustness_only"
    if dataset == "refit" and house_id == 10 and role != "metadata":
        data_use = "robustness_only"
    weak_label_candidate = ""
    if dataset == "refit" and (house_id, channel_id) == (17, 9):
        data_use = "weak_label_only"
        weak_label_candidate = "hair_dryer"
    if dataset == "uk_dale" and (house_id, channel_id) in {(1, 10), (1, 11)}:
        data_use = "weak_label_only"
        weak_label_candidate = "kettle" if channel_id == 10 else "toaster"

    binary_label = None
    if role in {"core_positive", "extended_clean_positive"}:
        binary_label = 1
    elif role in {"negative", "hard_negative"}:
        binary_label = 0

    return {
        "normalized_label": normalized,
        "canonical_label": canonical,
        "semantic_class": role,
        "taxonomy_role": role,
        "taxonomy_confidence": confidence,
        "binary_label": binary_label,
        "data_use": data_use,
        "aggregate_quality": aggregate_quality,
        "weak_label_candidate": weak_label_candidate,
        "include_in_main_mvp": (
            data_use == "main"
            and confidence == "high"
            and role in {"core_positive", "negative", "hard_negative"}
        ),
        "taxonomy_reason": reason,
        "anomaly_code": anomaly[0] if anomaly else "",
        "anomaly_note": anomaly[1] if anomaly else "",
    }


def parse_refit_appliance_labels(readme_path: Path) -> dict[int, dict[int, str]]:
    lines = Path(readme_path).read_text(encoding="utf-8").splitlines()
    mapping: dict[int, dict[int, str]] = {}
    current_house: int | None = None
    appliance_text: list[str] = []

    def flush() -> None:
        nonlocal appliance_text
        if current_house is None or not appliance_text:
            appliance_text = []
            return
        text = " ".join(appliance_text)
        matches = list(re.finditer(r"(?<![A-Za-z0-9])(\d+)\.\s*", text))
        house_mapping: dict[int, str] = {}
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            label = text[match.end() : end].strip(" ,\t")
            house_mapping[int(match.group(1))] = label
        if house_mapping:
            mapping[current_house] = house_mapping
        appliance_text = []

    for line in lines:
        house_match = re.fullmatch(r"House\s+(\d+)", line.strip())
        if house_match:
            flush()
            current_house = int(house_match.group(1))
            continue
        if current_house is None:
            continue
        if line.strip().startswith("!NOTES"):
            flush()
            current_house = None
            continue
        if re.search(r"(?<![A-Za-z0-9])0\.\s*Aggregate", line):
            appliance_text.append(line.strip())
        elif appliance_text and re.search(r"(?<![A-Za-z0-9])\d+\.\s*", line):
            appliance_text.append(line.strip())
    flush()
    return mapping


def parse_ukdale_meter_models(metadata_path: Path) -> dict[int, str]:
    """Extract channel -> device model from a UK-DALE building metadata YAML.

    This intentionally parses only the small ``elec_meters`` subset, avoiding a
    runtime YAML dependency for the Phase 1 scanner.
    """
    mapping: dict[int, str] = {}
    in_elec_meters = False
    current_channel: int | None = None
    for line in Path(metadata_path).read_text(encoding="utf-8").splitlines():
        if line == "elec_meters:":
            in_elec_meters = True
            continue
        if not in_elec_meters:
            continue
        meter_match = re.fullmatch(r"  (\d+):", line)
        if meter_match:
            current_channel = int(meter_match.group(1))
            continue
        if line and not line.startswith(" "):
            break
        model_match = re.fullmatch(r"    device_model:\s*(.+)", line)
        if model_match and current_channel is not None:
            mapping[current_channel] = model_match.group(1).strip()
    return mapping


def scan_refit_house(
    csv_path: Path,
    labels: dict[int, str],
    chunksize: int = 500_000,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    house_match = re.search(r"House_?(\d+)", csv_path.stem, flags=re.IGNORECASE)
    if house_match is None:
        raise ValueError(f"Cannot determine REFIT house id from {csv_path}")
    house_id = int(house_match.group(1))
    available_columns = set(pd.read_csv(csv_path, nrows=0).columns)
    measurement_columns = ["Aggregate", *[f"Appliance{i}" for i in range(1, 10)]]
    columns = ["Unix", *measurement_columns]
    has_issues_column = "Issues" in available_columns
    if has_issues_column:
        columns.append("Issues")
    time_accumulator = TimeAccumulator()
    value_accumulators = {column: ValueAccumulator() for column in columns if column != "Unix"}

    for chunk in pd.read_csv(
        csv_path,
        usecols=columns,
        chunksize=chunksize,
        low_memory=False,
    ):
        timestamps = pd.to_numeric(chunk["Unix"], errors="coerce").to_numpy(dtype=float)
        time_accumulator.update(timestamps)
        for column, accumulator in value_accumulators.items():
            values = pd.to_numeric(chunk[column], errors="coerce").to_numpy(dtype=float)
            accumulator.update(values)

    time_summary = time_accumulator.summary()
    issue_summary = value_accumulators["Issues"].summary() if has_issues_column else None
    aggregate_summary = value_accumulators["Aggregate"].summary()
    house_anomaly = REFIT_ANOMALIES.get((house_id, None))
    house_row = {
        "dataset": "refit",
        "house_id": house_id,
        "source_file": csv_path.as_posix(),
        **time_summary,
        "aggregate_missing_fraction": aggregate_summary["missing_fraction"],
        "aggregate_negative_value_count": aggregate_summary["negative_value_count"],
        "issue_row_count": issue_summary["nonzero_value_count"] if issue_summary else None,
        "issues_column_available": has_issues_column,
        "house_anomaly_code": house_anomaly[0] if house_anomaly else "",
        "house_anomaly_note": house_anomaly[1] if house_anomaly else "",
        "recommended_main_split_pool": house_id not in {10, 11, 21},
    }

    channel_rows: list[dict[str, object]] = []
    for channel_id in range(10):
        column = "Aggregate" if channel_id == 0 else f"Appliance{channel_id}"
        source_label = labels.get(channel_id, "UNMAPPED")
        classification = classify_appliance("refit", house_id, channel_id, source_label)
        value_summary = value_accumulators[column].summary()
        channel_rows.append(
            {
                "dataset": "refit",
                "house_id": house_id,
                "channel_id": channel_id,
                "source_column": column,
                "source_label": source_label,
                "source_file": csv_path.as_posix(),
                **classification,
                **time_summary,
                **value_summary,
                "coverage_start_offset_days": 0.0,
                "coverage_end_offset_days": 0.0,
            }
        )
    return house_row, channel_rows


def scan_ukdale_channel(
    channel_path: Path,
    chunksize: int = 500_000,
) -> tuple[dict[str, object], dict[str, object]]:
    time_accumulator = TimeAccumulator()
    value_accumulator = ValueAccumulator()
    for chunk in pd.read_csv(
        channel_path,
        sep=r"\s+",
        header=None,
        names=["timestamp", "power"],
        usecols=[0, 1],
        chunksize=chunksize,
    ):
        timestamps = pd.to_numeric(chunk["timestamp"], errors="coerce").to_numpy(dtype=float)
        powers = pd.to_numeric(chunk["power"], errors="coerce").to_numpy(dtype=float)
        time_accumulator.update(timestamps)
        value_accumulator.update(powers)
    return time_accumulator.summary(), value_accumulator.summary()


def scan_ukdale_house(
    house_path: Path,
    chunksize: int = 500_000,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    house_id = int(house_path.name.split("_")[-1])
    labels = dict(parse_channel_labels(house_path / "labels.dat"))
    channel_rows: list[dict[str, object]] = []
    aggregate_time: dict[str, object] | None = None
    aggregate_value: dict[str, object] | None = None

    for channel_id, source_label in sorted(labels.items()):
        channel_path = house_path / f"channel_{channel_id}.dat"
        if channel_path.exists():
            time_summary, value_summary = scan_ukdale_channel(channel_path, chunksize=chunksize)
        else:
            time_summary = TimeAccumulator().summary()
            value_summary = ValueAccumulator().summary()
        classification = classify_appliance("uk_dale", house_id, channel_id, source_label)
        row = {
            "dataset": "uk_dale",
            "house_id": house_id,
            "channel_id": channel_id,
            "source_column": f"channel_{channel_id}",
            "source_label": source_label,
            "source_file": channel_path.as_posix(),
            "source_file_exists": channel_path.exists(),
            **classification,
            **time_summary,
            **value_summary,
        }
        channel_rows.append(row)
        if channel_id == 1:
            aggregate_time = time_summary
            aggregate_value = value_summary

    if aggregate_time is None or aggregate_value is None:
        raise ValueError(f"UK-DALE house {house_id} has no aggregate channel 1")
    aggregate_start = aggregate_time["start_timestamp"]
    aggregate_end = aggregate_time["end_timestamp"]
    for row in channel_rows:
        start = row["start_timestamp"]
        end = row["end_timestamp"]
        row["coverage_start_offset_days"] = (
            (float(start) - float(aggregate_start)) / 86400.0
            if start is not None and aggregate_start is not None
            else None
        )
        row["coverage_end_offset_days"] = (
            (float(aggregate_end) - float(end)) / 86400.0
            if end is not None and aggregate_end is not None
            else None
        )

    house_anomalies = {
        1: (
            "submeters_switched_off",
            "Metadata states that many individual appliance meters were switched off for "
            "significant periods; missing coverage must not be interpreted as OFF.",
        ),
        4: ("limited_clean_labels", "Several House 4 channels combine multiple appliances."),
    }
    house_anomaly = house_anomalies.get(house_id)

    house_row = {
        "dataset": "uk_dale",
        "house_id": house_id,
        "source_file": (house_path / "channel_1.dat").as_posix(),
        **aggregate_time,
        "aggregate_missing_fraction": aggregate_value["missing_fraction"],
        "aggregate_negative_value_count": aggregate_value["negative_value_count"],
        "issue_row_count": None,
        "house_anomaly_code": house_anomaly[0] if house_anomaly else "",
        "house_anomaly_note": house_anomaly[1] if house_anomaly else "",
        "recommended_main_split_pool": True,
    }
    return house_row, channel_rows


def build_automatic_anomalies(
    houses: Iterable[dict[str, object]],
    channels: Iterable[dict[str, object]],
) -> list[dict[str, object]]:
    anomalies: list[dict[str, object]] = []
    for house in houses:
        house_anomaly_code = house.get("house_anomaly_code")
        if (
            house_anomaly_code is not None
            and pd.notna(house_anomaly_code)
            and str(house_anomaly_code).strip()
        ):
            action_by_code = {
                "clean_copy_truncated": "Use Processed_Data_CSV/House_1.csv and retain the Issues limitation.",
                "aggregate_affected_by_solar": "Use only in a named solar robustness slice.",
                "submeters_switched_off": "Generate labels only inside continuous observed submeter coverage.",
                "limited_clean_labels": "Use only explicitly approved channels; ignore mixed channels.",
            }
            anomalies.append(
                {
                    "dataset": house["dataset"],
                    "house_id": house["house_id"],
                    "channel_id": None,
                    "source_label": "aggregate/house",
                    "severity": "high",
                    "anomaly_code": house_anomaly_code,
                    "evidence": house["house_anomaly_note"],
                    "recommended_action": action_by_code.get(
                        str(house_anomaly_code), "Apply the explicit house policy before event generation."
                    ),
                }
            )
        if float(house.get("maximum_gap_seconds") or 0.0) > 7 * 86400:
            anomalies.append(
                {
                    "dataset": house["dataset"],
                    "house_id": house["house_id"],
                    "channel_id": None,
                    "source_label": "aggregate",
                    "severity": "medium",
                    "anomaly_code": "aggregate_gap_over_7_days",
                    "evidence": f"Longest aggregate gap is {house['maximum_gap_seconds']} seconds.",
                    "recommended_action": "Never interpolate across this gap; reject crossing event windows.",
                }
            )
        issue_fraction = house.get("issue_fraction")
        if issue_fraction is not None and pd.notna(issue_fraction) and float(issue_fraction) > 0.02:
            anomalies.append(
                {
                    "dataset": house["dataset"],
                    "house_id": house["house_id"],
                    "channel_id": None,
                    "source_label": "aggregate/house",
                    "severity": "medium",
                    "anomaly_code": "high_issues_fraction",
                    "evidence": f"Issues flag fraction is {float(issue_fraction):.4%}.",
                    "recommended_action": "Reject primary event windows touching Issues=1 rows.",
                }
            )
        zero_fraction = house.get("aggregate_zero_fraction")
        if zero_fraction is not None and pd.notna(zero_fraction) and float(zero_fraction) > 0.001:
            anomalies.append(
                {
                    "dataset": house["dataset"],
                    "house_id": house["house_id"],
                    "channel_id": None,
                    "source_label": "aggregate",
                    "severity": "medium",
                    "anomaly_code": "aggregate_zero_fill_candidate",
                    "evidence": f"Aggregate zero fraction is {float(zero_fraction):.4%}.",
                    "recommended_action": "Treat zero runs as missing-data candidates, not true household OFF.",
                }
            )
        if house.get("issues_column_available") is False or house.get("issues_column_available") == False:
            anomalies.append(
                {
                    "dataset": house["dataset"],
                    "house_id": house["house_id"],
                    "channel_id": None,
                    "source_label": "aggregate/house",
                    "severity": "medium",
                    "anomaly_code": "issues_flag_not_available_for_full_source",
                    "evidence": "The selected full source has no Issues column.",
                    "recommended_action": "Use timestamp gaps/zero runs and keep the limitation explicit.",
                }
            )

    for channel in channels:
        if channel.get("anomaly_code"):
            anomalies.append(
                {
                    "dataset": channel["dataset"],
                    "house_id": channel["house_id"],
                    "channel_id": channel["channel_id"],
                    "source_label": channel["source_label"],
                    "severity": "high" if channel["taxonomy_role"] == "ignore" else "medium",
                    "anomaly_code": channel["anomaly_code"],
                    "evidence": channel["anomaly_note"],
                    "recommended_action": (
                        "Ignore for primary labels." if channel["taxonomy_role"] == "ignore"
                        else "Keep with an explicit caveat and evaluate separately."
                    ),
                }
            )
        start_offset = channel.get("coverage_start_offset_days")
        end_offset = channel.get("coverage_end_offset_days")
        if channel["dataset"] == "uk_dale" and (
            (start_offset is not None and abs(float(start_offset)) > 1.0)
            or (end_offset is not None and abs(float(end_offset)) > 1.0)
        ):
            anomalies.append(
                {
                    "dataset": channel["dataset"],
                    "house_id": channel["house_id"],
                    "channel_id": channel["channel_id"],
                    "source_label": channel["source_label"],
                    "severity": "low",
                    "anomaly_code": "submeter_coverage_differs_from_aggregate",
                    "evidence": (
                        f"Start offset={start_offset:.2f} days; end offset={end_offset:.2f} days."
                    ),
                    "recommended_action": "Generate events only inside the channel's observed coverage.",
                }
            )
    return anomalies


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
