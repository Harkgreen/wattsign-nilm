"""Original code stays executable, with raw access and training disabled by default."""
from __future__ import annotations

import ast
from contextlib import ExitStack
import importlib.util
import json
import os
from pathlib import Path
import sys
import subprocess
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
CASES = {"ukdale-10s", "ukdale-1s", "tdha-waveform", "plaid-resistive", "tdha-npu"}


class NotebookDisplaySafetyTests(unittest.TestCase):
    def test_original_code_and_disabled_training_are_present(self):
        paths = sorted((ROOT / "notebooks").glob("*.ipynb"))
        self.assertEqual(len(paths), 3)
        for path in paths:
            notebook = json.loads(path.read_text(encoding="utf-8"))
            publication = notebook["metadata"]["publication"]
            self.assertEqual(publication["display_mode"], "code_workflow")
            flags = set(publication["disabled_operations"])
            self.assertIn("RUN_TRAINING", flags)
            definitions = 0
            defaults = {}
            for cell in notebook["cells"]:
                if cell["cell_type"] != "code":
                    continue
                tree = ast.parse("".join(cell["source"]))
                definitions += sum(isinstance(n, (ast.FunctionDef, ast.ClassDef)) for n in ast.walk(tree))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name) and target.id in flags:
                                self.assertIsInstance(node.value, ast.Constant, target.id)
                                self.assertIs(node.value.value, False, target.id)
                                defaults[target.id] = node.value.value
            self.assertGreaterEqual(definitions, 10, path.name)
            self.assertEqual(set(defaults), flags, path.name)

    def test_default_execution_reads_no_raw_data_and_never_fits_or_writes(self):
        # Import dependencies before forbidding work: library initialization may
        # inspect its own files, while the notebook itself must stay read-only.
        import joblib
        import numpy as np
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        import wattsign.notebook_view as view
        try:
            import torch
        except ImportError:
            torch = None
        shown = set()

        def forbidden(*args, **kwargs):
            raise AssertionError("Default notebook execution attempted raw access, fitting, model IO or writes")

        original_cwd = Path.cwd()
        original_bytecode = sys.dont_write_bytecode
        try:
            os.chdir(ROOT)
            sys.dont_write_bytecode = True
            with ExitStack() as stack:
                stack.enter_context(patch.object(view, "show_experiment", side_effect=lambda case_id: shown.add(case_id)))
                targets = [(joblib, "load"), (joblib, "dump"), (pd, "read_csv"), (pd, "read_excel"),
                           (pd.DataFrame, "to_csv"), (pd.DataFrame, "to_parquet"),
                           (subprocess, "run"), (subprocess, "Popen"),
                           (np, "load"), (np, "save"), (np, "savez"), (np, "savez_compressed"), (np, "savetxt"),
                           (Path, "write_text"), (Path, "write_bytes"), (Path, "mkdir"),
                           (Pipeline, "fit"), (RandomForestClassifier, "fit"),
                           (SimpleImputer, "fit"), (StandardScaler, "fit")]
                if torch is not None:
                    targets.extend([(torch, "load"), (torch, "save"),
                                    (torch.optim.Adam, "step"), (torch.optim.AdamW, "step")])
                for obj, name in targets:
                    stack.enter_context(patch.object(obj, name, side_effect=forbidden))
                for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
                    notebook = json.loads(path.read_text(encoding="utf-8"))
                    name = "_wattsign_notebook_" + path.stem
                    module = types.ModuleType(name)
                    module.__file__ = str(path)
                    stack.enter_context(patch.dict(sys.modules, {name: module}))
                    for index, cell in enumerate(notebook["cells"]):
                        if cell["cell_type"] == "code":
                            exec(compile("".join(cell["source"]), f"{path}:cell{index}", "exec"), module.__dict__)
                    for flag in notebook["metadata"]["publication"]["disabled_operations"]:
                        self.assertIs(module.__dict__[flag], False, f"{path.name}:{flag}")
            self.assertEqual(shown, CASES)
        finally:
            os.chdir(original_cwd)
            sys.dont_write_bytecode = original_bytecode

    def test_inspection_entry_import_does_not_load_models(self):
        path = ROOT / "research/inspect_model.py"
        spec = importlib.util.spec_from_file_location("saved_model_inspection", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertTrue(callable(module.main))
        self.assertNotIn("load_model", vars(module))

    def test_missing_result_index_fails_without_computation(self):
        from load_signature_analysis.notebook_view import repository_root
        with self.assertRaises(FileNotFoundError):
            repository_root(ROOT / "tests")
