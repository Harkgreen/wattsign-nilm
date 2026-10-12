"""PLAID metadata loading and memory-efficient waveform access."""

from __future__ import annotations

from dataclasses import dataclass
import io
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PlaidRecord:
    sample_id: str
    release: str
    csv_path: Path
    appliance_type: str
    load_type: str
    status: str
    location: str
    brand: str
    model_number: str
    sampling_frequency_hz: float

    @property
    def resistive_label(self) -> int:
        return int(self.load_type.upper() == "R")

    @property
    def heating_active_proxy_label(self) -> int:
        # PLAID marks a hair dryer as an R-load even in its fan-only mode.
        return int(self.resistive_label == 1 and self.status.lower() != "noheat")


def _parse_frequency(value: str) -> float:
    normalized = value.strip().lower().replace("khz", "000").replace("hz", "")
    return float(normalized)


def load_plaid_records(plaid_root: Path) -> list[PlaidRecord]:
    plaid_root = Path(plaid_root)
    waveform_dir = plaid_root / "2017"
    records: list[PlaidRecord] = []
    for metadata_path in sorted(plaid_root.glob("meta_*.json")):
        release = metadata_path.stem.removeprefix("meta_")
        with metadata_path.open("r", encoding="utf-8") as handle:
            entries = json.load(handle)
        for entry in entries:
            meta = entry["meta"]
            appliance = meta["appliance"]
            sample_id = str(entry["id"])
            csv_path = waveform_dir / f"{sample_id}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"missing PLAID waveform: {csv_path}")
            records.append(
                PlaidRecord(
                    sample_id=sample_id,
                    release=release,
                    csv_path=csv_path,
                    appliance_type=appliance.get("type", ""),
                    load_type=appliance.get("load", ""),
                    status=appliance.get("status", ""),
                    location=meta.get("location", "unknown"),
                    brand=appliance.get("brand", ""),
                    model_number=appliance.get("model_number", ""),
                    sampling_frequency_hz=_parse_frequency(meta["header"]["sampling_frequency"]),
                )
            )
    if not records:
        raise FileNotFoundError(f"no PLAID metadata found under {plaid_root}")
    return records


def read_waveform_tail(csv_path: Path, max_rows: int = 30_000) -> tuple[np.ndarray, np.ndarray]:
    """Read the last rows of a two-column current/voltage CSV.

    PLAID files can be several megabytes. Seeking backwards avoids parsing the
    off-state portion when the steady-state feature extractor only needs the
    final cycles.
    """

    csv_path = Path(csv_path)
    with csv_path.open("rb") as handle:
        handle.seek(0, 2)
        file_size = handle.tell()
        block_size = min(file_size, 1_048_576)
        while True:
            handle.seek(file_size - block_size)
            payload = handle.read(block_size)
            if payload.count(b"\n") >= max_rows + 1 or block_size == file_size:
                break
            block_size = min(file_size, block_size * 2)

    lines = payload.splitlines()
    if block_size < file_size and lines:
        lines = lines[1:]
    lines = lines[-max_rows:]
    array = np.loadtxt(io.BytesIO(b"\n".join(lines)), delimiter=",", dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.shape[1] != 2:
        raise ValueError(f"expected two columns in {csv_path}, found {array.shape[1]}")
    current = array[:, 0]
    voltage = array[:, 1]
    return current, voltage

