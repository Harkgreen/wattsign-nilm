"""Generate appliance activity intervals from UK-DALE submeter data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd


@dataclass(frozen=True)
class DeviceSpec:
    canonical_name: str
    aliases: tuple[str, ...]
    threshold_w: float
    minimum_active_seconds: float = 6.0
    merge_gap_seconds: float = 60.0


DEVICE_SPECS = (
    DeviceSpec("kettle", ("kettle", "kettle_radio"), 2_000.0, 6.0, 30.0),
    DeviceSpec("toaster", ("toaster",), 1_000.0, 6.0, 30.0),
    DeviceSpec("hair_dryer", ("hair_dryer", "hairdryer"), 300.0, 6.0, 60.0),
    DeviceSpec("hair_straighteners", ("straighteners",), 50.0, 6.0, 120.0),
    DeviceSpec("clothes_iron", ("iron", "steam_iron"), 300.0, 6.0, 120.0),
    DeviceSpec("electric_heater", ("electric_heater",), 300.0, 12.0, 120.0),
    DeviceSpec("soldering_iron", ("soldering_iron",), 20.0, 6.0, 120.0),
    DeviceSpec("rice_cooker", ("rice_cooker",), 100.0, 12.0, 120.0),
    DeviceSpec("electric_hob", ("electric_hob",), 300.0, 12.0, 120.0),
)


def _normalize_name(value: str) -> str:
    value = value.strip().lower().replace("&", "and")
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_")


def parse_channel_labels(labels_path: Path) -> list[tuple[int, str]]:
    labels: list[tuple[int, str]] = []
    with Path(labels_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            channel_text, name = line.strip().split(maxsplit=1)
            labels.append((int(channel_text), _normalize_name(name)))
    return labels


def match_device_spec(channel_name: str) -> DeviceSpec | None:
    for spec in DEVICE_SPECS:
        if channel_name in spec.aliases:
            return spec
    return None


def _close_interval(active: dict[str, float], end_timestamp: float) -> dict[str, float]:
    interval = dict(active)
    interval["end_timestamp"] = float(end_timestamp)
    interval["active_duration_seconds"] = max(
        0.0, float(interval["last_active_timestamp"] - interval["start_timestamp"])
    )
    interval["mean_active_power_w"] = float(interval["power_sum"] / interval["sample_count"])
    return interval


def _merge_intervals(
    intervals: list[dict[str, float]],
    maximum_gap_seconds: float,
) -> list[dict[str, float]]:
    merged: list[dict[str, float]] = []
    for interval in intervals:
        if not merged or interval["start_timestamp"] - merged[-1]["end_timestamp"] > maximum_gap_seconds:
            merged.append(dict(interval))
            continue
        previous = merged[-1]
        total_samples = previous["sample_count"] + interval["sample_count"]
        total_power = previous["power_sum"] + interval["power_sum"]
        previous["end_timestamp"] = interval["end_timestamp"]
        previous["last_active_timestamp"] = interval["last_active_timestamp"]
        previous["active_duration_seconds"] += interval["active_duration_seconds"]
        previous["sample_count"] = total_samples
        previous["power_sum"] = total_power
        previous["mean_active_power_w"] = total_power / total_samples
        previous["max_active_power_w"] = max(
            previous["max_active_power_w"], interval["max_active_power_w"]
        )
    return merged


def extract_channel_intervals(
    channel_path: Path,
    spec: DeviceSpec,
    maximum_sample_gap_seconds: float = 120.0,
) -> list[dict[str, float]]:
    raw_intervals: list[dict[str, float]] = []
    active: dict[str, float] | None = None
    previous_timestamp: float | None = None
    with Path(channel_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 2:
                continue
            timestamp = float(parts[0])
            power = float(parts[1])
            if (
                active is not None
                and previous_timestamp is not None
                and timestamp - previous_timestamp > maximum_sample_gap_seconds
            ):
                raw_intervals.append(_close_interval(active, previous_timestamp))
                active = None

            is_on = power >= spec.threshold_w
            if is_on and active is None:
                active = {
                    "start_timestamp": timestamp,
                    "last_active_timestamp": timestamp,
                    "power_sum": power,
                    "sample_count": 1.0,
                    "max_active_power_w": power,
                }
            elif is_on and active is not None:
                active["last_active_timestamp"] = timestamp
                active["power_sum"] += power
                active["sample_count"] += 1.0
                active["max_active_power_w"] = max(active["max_active_power_w"], power)
            elif not is_on and active is not None:
                raw_intervals.append(_close_interval(active, timestamp))
                active = None
            previous_timestamp = timestamp
    if active is not None and previous_timestamp is not None:
        raw_intervals.append(_close_interval(active, previous_timestamp))

    filtered = [
        interval
        for interval in raw_intervals
        if interval["active_duration_seconds"] >= spec.minimum_active_seconds
    ]
    return _merge_intervals(filtered, spec.merge_gap_seconds)


def build_ukdale_interval_table(
    ukdale_root: Path,
    output_csv: Path,
    houses: set[int] | None = None,
    devices: set[str] | None = None,
) -> pd.DataFrame:
    ukdale_root = Path(ukdale_root)
    rows: list[dict[str, object]] = []
    for house_path in sorted(ukdale_root.glob("house_*")):
        house_number = int(house_path.name.split("_")[-1])
        if houses is not None and house_number not in houses:
            continue
        for channel, channel_name in parse_channel_labels(house_path / "labels.dat"):
            spec = match_device_spec(channel_name)
            if spec is None or (devices is not None and spec.canonical_name not in devices):
                continue
            channel_path = house_path / f"channel_{channel}.dat"
            if not channel_path.exists():
                continue
            intervals = extract_channel_intervals(channel_path, spec)
            print(
                f"UK-DALE house {house_number} channel {channel} {channel_name}: "
                f"{len(intervals)} intervals",
                flush=True,
            )
            for interval in intervals:
                start = pd.to_datetime(interval["start_timestamp"], unit="s", utc=True)
                end = pd.to_datetime(interval["end_timestamp"], unit="s", utc=True)
                session_duration = float(interval["end_timestamp"] - interval["start_timestamp"])
                rows.append(
                    {
                        "house": house_number,
                        "channel": channel,
                        "source_name": channel_name,
                        "device": spec.canonical_name,
                        "threshold_w": spec.threshold_w,
                        "start_timestamp": interval["start_timestamp"],
                        "end_timestamp": interval["end_timestamp"],
                        "start_utc": start.isoformat(),
                        "end_utc": end.isoformat(),
                        "session_duration_seconds": session_duration,
                        "active_duration_seconds": interval["active_duration_seconds"],
                        "active_fraction": (
                            interval["active_duration_seconds"] / session_duration
                            if session_duration > 0
                            else 1.0
                        ),
                        "mean_active_power_w": interval["mean_active_power_w"],
                        "max_active_power_w": interval["max_active_power_w"],
                        "sample_count": int(interval["sample_count"]),
                    }
                )
    frame = pd.DataFrame(rows)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    return frame
