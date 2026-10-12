"""Common explicit entry point for preserved research recipes; import is inert."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
from .data_paths import resolve_data_paths

def configured_data_root() -> Path:
    return resolve_data_paths().data_root

def configured_dataset_root(dataset: str) -> Path:
    return resolve_data_paths().datasets[dataset]

def prepare_research() -> Path:
    parser = argparse.ArgumentParser(description="Optional historical research; may train, select thresholds or regenerate reports.")
    parser.add_argument("--run-research", action="store_true")
    args = parser.parse_args()
    if not args.run_research:
        parser.error("Nothing was run. Explicit --run-research is required; use the notebooks for saved results.")
    paths = resolve_data_paths()
    if not paths.data_root.is_dir() and not any(p.is_dir() for p in paths.datasets.values()):
        raise FileNotFoundError("Configured historical data is unavailable. No cold-start download or training fallback exists.")
    local_config = paths.project / "configs/local.toml"
    if not os.environ.get("LSA_OUTPUT_ROOT") and not local_config.is_file():
        raise RuntimeError("Explicit LSA_OUTPUT_ROOT or local path configuration is required for research output.")
    os.chdir(paths.project)
    return paths.output_root
