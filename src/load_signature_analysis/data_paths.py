"""Explicit dataset paths and bounded, read-only inspection; never builds caches."""
from __future__ import annotations
from dataclasses import dataclass
import csv
import io
import os
from pathlib import Path
import tomllib
from typing import Mapping

DATASETS = {
    "ukdale": ("UK-DALE/ukdale", "NILM_UKDALE_ROOT"),
    "refit": ("REFIT", "NILM_REFIT_ROOT"),
    "plaid": ("PLAID", "NILM_PLAID_ROOT"),
    "tdha": ("TDHA/V2", "NILM_TDHA_ROOT"),
}

class DataPathError(ValueError):
    """The explicit input path is missing or outside its declared dataset."""


def project_root(value: str | Path | None = None) -> Path:
    if value is not None:
        return Path(value).expanduser().resolve()
    if os.environ.get("LSA_PROJECT_ROOT"):
        return Path(os.environ["LSA_PROJECT_ROOT"]).expanduser().resolve()
    source_checkout = Path(__file__).resolve().parents[2]
    if (source_checkout / "pyproject.toml").is_file():
        return source_checkout
    return Path.cwd().resolve()


def _resolve(value: str | Path, base: Path) -> Path:
    candidate = Path(value).expanduser()
    return (candidate if candidate.is_absolute() else base / candidate).resolve()


@dataclass(frozen=True)
class DataPaths:
    project: Path
    data_root: Path
    output_root: Path
    datasets: Mapping[str, Path]

    def as_dict(self) -> dict:
        return {"project_root": str(self.project), "data_root": str(self.data_root),
                "output_root": str(self.output_root),
                "datasets": {key: {"path": str(path), "exists": path.is_dir()}
                             for key, path in self.datasets.items()}}


def resolve_data_paths(*, project: str | Path | None = None,
                       config: str | Path | None = None,
                       data_root: str | Path | None = None,
                       output_root: str | Path | None = None,
                       dataset_overrides: Mapping[str, str | Path] | None = None,
                       environ: Mapping[str, str] | None = None) -> DataPaths:
    """CLI arguments > environment > TOML > new-checkout defaults; no data reads.

    TOML uses [paths] data_root/output_root and [datasets] ukdale/refit/plaid/tdha.
    Relative TOML paths and defaults are relative to the project root.
    """
    root = project_root(project)
    env = os.environ if environ is None else environ
    cfg_path = _resolve(config, root) if config is not None else root / "configs/local.toml"
    settings = {}
    if cfg_path.is_file():
        if cfg_path.stat().st_size > 64 * 1024:
            raise DataPathError("Local path configuration exceeds 64 KiB")
        with cfg_path.open("rb") as handle:
            settings = tomllib.load(handle)
    elif config is not None:
        raise FileNotFoundError(f"Path configuration missing: {cfg_path}")
    paths = settings.get("paths", {})
    configured = settings.get("datasets", {})
    if not isinstance(paths, dict) or not isinstance(configured, dict):
        raise DataPathError("[paths] and [datasets] must be TOML tables")
    overrides = dict(dataset_overrides or {})
    if set(overrides) - DATASETS.keys() or set(configured) - DATASETS.keys():
        raise DataPathError("Dataset must be one of: " + ", ".join(DATASETS))
    data = _resolve(data_root or env.get("NILM_DATA_ROOT") or paths.get("data_root", "datasets"), root)
    output = _resolve(output_root or env.get("LSA_OUTPUT_ROOT") or paths.get("output_root", "outputs"), root)
    datasets = {}
    for key, (relative, variable) in DATASETS.items():
        explicit = overrides.get(key) or env.get(variable) or configured.get(key)
        datasets[key] = _resolve(explicit, root) if explicit else (data / relative).resolve()
    if output == data or output.is_relative_to(data):
        raise DataPathError("Output root must be separate from the raw-data root")
    for path in datasets.values():
        if output == path or output.is_relative_to(path) or path.is_relative_to(output):
            raise DataPathError("Output root must not overlap a dataset directory")
    return DataPaths(root, data, output, datasets)


def resolve_project_path(project: str | Path, value: str | Path) -> Path:
    """Resolve scientific-config and inventory paths using the local path settings.

    Dataset and output prefixes honor the same environment/TOML overrides as the
    CLI. Other relative paths (for example configs/) stay within the checkout.
    Absolute paths remain explicit. Resolving paths does not create directories.
    """
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    root = project_root(project)
    parts = candidate.parts
    if not parts:
        return root
    prefix = parts[0].casefold()
    if prefix not in {"datasets", "outputs"}:
        return (root / candidate).resolve()
    paths = resolve_data_paths(project=root)
    if prefix == "outputs":
        base, tail = paths.output_root, parts[1:]
    else:
        base, tail = paths.data_root, parts[1:]
        for key, (relative, _) in DATASETS.items():
            canonical = Path(relative).parts
            if tuple(p.casefold() for p in tail[:len(canonical)]) == tuple(p.casefold() for p in canonical):
                base, tail = paths.datasets[key], tail[len(canonical):]
                break
    target = base.joinpath(*tail).resolve()
    if not target.is_relative_to(base):
        raise DataPathError("Configured path escapes its dataset or output root")
    return target


def inspect_file(paths: DataPaths, dataset: str, relative_file: str | Path, *,
                 max_rows: int = 8, max_bytes: int = 65536) -> dict:
    """Read only a bounded prefix of one explicit text file. Never enumerates data.

    Archives, binary arrays and implicit dataset scans are intentionally unsupported.
    Existing feature caches and raw files are never created, changed or refreshed.
    """
    if dataset not in paths.datasets:
        raise DataPathError("Unknown dataset: " + dataset)
    if not 1 <= max_rows <= 200 or not 1 <= max_bytes <= 1024 * 1024:
        raise DataPathError("Require 1..200 rows and 1..1048576 bytes")
    relative = Path(relative_file)
    if relative.is_absolute() or ".." in relative.parts:
        raise DataPathError("Use a relative filename inside the declared dataset")
    root = paths.datasets[dataset].resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise DataPathError("File resolves outside its declared dataset")
    if target.suffix.lower() not in {".csv", ".dat", ".txt", ".tsv"}:
        raise DataPathError("Bounded inspection supports .csv/.dat/.txt/.tsv only")
    if not target.is_file():
        raise FileNotFoundError(f"Explicit data file missing: {target}. Nothing was prepared or trained.")
    before = target.stat()
    with target.open("rb") as handle:
        raw = handle.read(max_bytes)
    after = target.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise DataPathError("Source changed while being inspected")
    text = raw.decode("utf-8-sig", errors="replace")
    lines = text.splitlines()
    if len(raw) < before.st_size and raw and raw[-1:] not in {b"\n", b"\r"}:
        lines = lines[:-1]  # Never present an incomplete trailing row as complete.
    selected = lines[:max_rows]
    delimiter = "," if target.suffix.lower() == ".csv" else "\t" if target.suffix.lower() == ".tsv" else None
    rows = list(csv.reader(selected, delimiter=delimiter)) if delimiter else [line.split() for line in selected]
    return {"dataset": dataset, "relative_file": relative.as_posix(), "size_bytes": before.st_size,
            "bytes_read": len(raw), "rows_read": len(rows), "truncated": len(raw) < before.st_size or len(lines) > max_rows,
            "column_counts": [len(row) for row in rows], "preview": rows, "read_only": True}
