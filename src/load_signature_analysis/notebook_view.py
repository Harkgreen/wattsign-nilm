"""Render experiment summaries, metrics and figures in Jupyter."""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any

def repository_root(root: str | Path | None = None) -> Path:
    if root is not None:
        candidate = Path(root).expanduser().resolve()
        if not (candidate / "results" / "index.json").is_file():
            raise FileNotFoundError("The selected folder has no results/index.json.")
        return candidate
    configured = os.environ.get("NILM_REPO_ROOT")
    if configured:
        return repository_root(configured)
    candidates = [Path.cwd(), *Path.cwd().parents, Path(__file__).resolve().parents[2]]
    for candidate in candidates:
        if (candidate / "results" / "index.json").is_file():
            return candidate
    raise FileNotFoundError("Open the notebook inside the WattSign repository or set NILM_REPO_ROOT to its root directory.")

def load_experiment(experiment_id: str, root: str | Path | None = None) -> dict[str, Any]:
    base = repository_root(root)
    from .cli import load_result_index
    experiments = load_result_index(base)
    selected = next((item for item in experiments if item.get("id") == experiment_id), None)
    if selected is None:
        raise KeyError(f"Experiment {experiment_id!r} is not listed in results/index.json.")
    return selected

def _items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]

def experiment_markdown(experiment: dict[str, Any]) -> str:
    title = str(experiment.get("title") or experiment["id"])
    summary = str(experiment.get("summary") or experiment.get("description") or "")
    lines = [f"## {title}", "", summary, ""]
    limitations = _items(experiment.get("limitations"))
    if limitations:
        lines += ["**Scope and limitations**", ""]
        lines += ["- " + str(item) for item in limitations]
        lines.append("")
    path = str(experiment.get("path", "results/" + experiment["id"])).replace("\\", "/")
    target = path if Path(path).suffix else path + "/README.md"
    lines += [f"[Case files](../{target})", "", "Experiment configuration, metrics and provenance are listed in the case files.", ""]
    reports = _items(experiment.get("reports"))
    for report in reports:
        if isinstance(report, dict):
            location = report.get("path") or report.get("file")
            label = report.get("title") or report.get("name") or location
        else:
            location = report
            label = Path(str(report)).name
        if location:
            location = str(location).replace("\\", "/")
            if location.startswith("results/"):
                lines.append(f"- [{label}](../{location})")
    assets = _items(experiment.get("assets"))
    if assets:
        lines += ["", "**Models and predictions**", "", "[Artifact access](../docs/en/runtime.md)", ""]
        configured = os.environ.get("NILM_RESULTS_ROOT")
        for asset in assets:
            if not isinstance(asset, dict):
                continue
            name = str(asset.get("filename", ""))
            local = Path(configured).expanduser() if configured else None
            if not name or Path(name).name != name:
                raise ValueError("Asset filename must be a filename, without directories")
            content = Path(str(asset.get("content_path", "results/" + experiment["id"])))
            if content.is_absolute() or ".." in content.parts:
                raise ValueError("Asset content_path must remain inside its extraction directory")
            available = bool(local and (local / name).is_file())
            extracted = bool(local and (local / Path(name).stem / content).is_dir())
            if available:
                state = "Archive available locally"
            elif extracted:
                state = "Extracted files available locally"
            else:
                state = "Not found locally"
            channel = asset.get("channel", "github_release")
            url = asset.get("url")
            if isinstance(url, str) and url.startswith("https://"):
                label = f"[{name}]({url})"
            else:
                label = name
                state += " · [Request artifacts](mailto:ruizhang.eric@gmail.com)" if not url else ""
            lines.append(f"- {label} ({channel}): {state}")
    return "\n".join(lines)

def _read_case_json(base: Path, relative: str) -> tuple[Path, dict]:
    path = (base / relative).resolve()
    if not path.is_relative_to(base.resolve()) or path.suffix != ".json":
        raise ValueError("Case JSON must remain inside the repository")
    if not path.is_file():
        raise FileNotFoundError(f"Case evidence file is missing: {relative}.")
    if path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("Case JSON exceeds the display size limit")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Expected a saved case JSON object")
    return path, payload


def _cell(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    return str(value).replace("|", "\\|").replace("\n", " ")


def case_markdown(experiment: dict[str, Any], summary: dict[str, Any], source: dict[str, Any]) -> str:
    lines = [experiment_markdown(experiment), "", "### Results", ""]
    for field, label in (("task", "Task"), ("method", "Method"), ("input_resolution", "Input resolution"), ("history_seconds", "History (s)")):
        if field in summary:
            lines.append(f"- **{label}:** {_cell(summary[field])}")
    rows = summary.get("metrics", [])
    if isinstance(rows, list) and rows and all(isinstance(row, dict) for row in rows):
        columns = list(dict.fromkeys(key for row in rows for key in row))
        lines += ["", "| " + " | ".join(map(_cell, columns)) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
        lines += ["| " + " | ".join(_cell(row.get(key)) for key in columns) + " |" for row in rows]
        lines += ["", "Metric values use the 0–1 scale; missing values indicate an inapplicable metric. AP measures ranking quality."]
    limitations = _items(summary.get("limitations"))
    if limitations:
        lines += ["", "**Limitations**", ""] + ["- " + str(item) for item in limitations]
    source_path = str(experiment.get("source", ""))
    if source_path:
        count = len(source.get("selected_files", []))
        lines += ["", f"[Provenance and checksums](../{source_path}): {count} artifact records."]
    return "\n".join(lines)


def show_experiment(experiment_id: str, root: str | Path | None = None) -> None:
    base = repository_root(root)
    experiment = load_experiment(experiment_id, base)
    relative = str(experiment["path"])
    figure = None
    if Path(relative).suffix == ".json":
        path, summary = _read_case_json(base, relative)
        if summary.get("case_id") != experiment_id:
            raise ValueError("Saved summary case identity does not match the catalog")
        source = {}
        if experiment.get("source"):
            _, source = _read_case_json(base, experiment["source"])
            if source.get("case_id") != experiment_id:
                raise ValueError("Saved source case identity does not match the catalog")
        rendered = case_markdown(experiment, summary, source)
        figure = (path.parent / "figures/main.png").resolve()
        if not figure.is_relative_to(base.resolve()):
            raise ValueError("Figure must remain inside the repository")
        if not figure.is_file():
            rendered += "\n\n> Figure file missing."
            figure = None
    else:
        rendered = experiment_markdown(experiment)
    try:
        from IPython.display import Image, Markdown, display
    except ImportError:
        print(rendered)
        if figure:
            print(f"Saved figure: {figure}")
    else:
        display(Markdown(rendered))
        if figure:
            display(Image(filename=str(figure)))
