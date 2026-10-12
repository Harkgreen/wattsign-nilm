"""Commands for experiment results, dataset paths and bounded file inspection."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from .data_paths import DATASETS, DataPathError, project_root, resolve_data_paths, inspect_file


def load_result_index(root: Path) -> list[dict]:
    path = root / "results/index.json"
    if not path.is_file():
        raise FileNotFoundError(f"Result index not found: {path}. Set --project-root to the WattSign checkout.")
    if path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("Saved result index exceeds 2 MiB")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1 or not isinstance(payload.get("experiments"), list):
        raise ValueError("Expected result index schema_version=1 and experiments=[]")
    seen = set()
    for entry in payload["experiments"]:
        if not isinstance(entry, dict) or not all(isinstance(entry.get(key), str) for key in ("id", "title", "status", "summary", "path")):
            raise ValueError("Result entry requires id, title, status, summary and path strings")
        if not entry["id"] or entry["id"] in seen:
            raise ValueError("Result identifiers must be nonempty and unique")
        seen.add(entry["id"])
        artifact = (root / entry["path"]).resolve()
        if Path(entry["path"]).is_absolute() or not artifact.is_relative_to(root.resolve()):
            raise ValueError("Result paths must remain inside the WattSign checkout")
    return payload["experiments"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="WattSign: saved results and read-only data checks")
    parser.add_argument("--project-root", type=Path, help="WattSign checkout containing results/index.json")
    sub = parser.add_subparsers(dest="command", required=True)
    results = sub.add_parser("results", help="List experiment records and inspect a case")
    result_sub = results.add_subparsers(dest="action", required=True)
    listing = result_sub.add_parser("list")
    listing.add_argument("--json", action="store_true")
    showing = result_sub.add_parser("show")
    showing.add_argument("id")
    data = sub.add_parser("data", help="Resolve configured paths or inspect one explicit text file")
    data_sub = data.add_subparsers(dest="action", required=True)
    for action in ("paths", "inspect"):
        command = data_sub.add_parser(action)
        command.add_argument("--config", type=Path)
        command.add_argument("--data-root", type=Path)
        command.add_argument("--output-root", type=Path)
        if action == "inspect":
            command.add_argument("--dataset", choices=DATASETS, required=True)
            command.add_argument("--file", required=True, help="Exact relative filename; no glob or recursive scan")
            command.add_argument("--dataset-root", type=Path, help="Override only the selected dataset root")
            command.add_argument("--max-rows", type=int, default=8)
            command.add_argument("--max-bytes", type=int, default=65536)
    args = parser.parse_args(argv)
    root = project_root(args.project_root)
    try:
        if args.command == "results":
            entries = load_result_index(root)
            if args.action == "list":
                if args.json:
                    print(json.dumps(entries, ensure_ascii=False, indent=2))
                else:
                    for entry in entries:
                        print(f"{entry['id']} | {entry['status']} | {entry['title']}")
            else:
                entry = next((item for item in entries if item["id"] == args.id), None)
                if entry is None:
                    raise ValueError("Unknown saved result: " + args.id)
                if not (root / entry["path"]).exists():
                    raise FileNotFoundError("Result file not found: " + entry["path"])
                print(json.dumps(entry, ensure_ascii=False, indent=2))
        else:
            overrides = {args.dataset: args.dataset_root} if args.action == "inspect" and args.dataset_root else None
            paths = resolve_data_paths(project=root, config=args.config, data_root=args.data_root,
                                       output_root=args.output_root, dataset_overrides=overrides)
            result = paths.as_dict() if args.action == "paths" else inspect_file(
                paths, args.dataset, args.file, max_rows=args.max_rows, max_bytes=args.max_bytes)
            print(json.dumps(result, ensure_ascii=False, indent=2))
    except (OSError, ValueError, DataPathError) as error:
        print(str(error), file=sys.stderr)
        return 2
    return 0
