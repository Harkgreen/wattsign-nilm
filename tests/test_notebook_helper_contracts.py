"""Notebook helper contracts using AST inspection and synthetic feature arrays."""
from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def notebook_tree(name):
    notebook = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    return ast.parse(source)


def named_node(tree, name, kind=ast.FunctionDef):
    matches = [node for node in ast.walk(tree) if isinstance(node, kind) and node.name == name]
    if len(matches) != 1:
        raise AssertionError(f"Expected one definition for {name}; found {len(matches)}")
    return matches[0]


def load_helpers(tree, names, namespace):
    module = ast.Module(body=[named_node(tree, name) for name in names], type_ignores=[])
    exec(compile(module, "<notebook helpers>", "exec"), namespace)
    return namespace


class NotebookHelperContracts(unittest.TestCase):
    def test_historical_threshold_has_explicit_recall_policy(self):
        # Inspect the helper without running a threshold search or calibration.
        node = named_node(notebook_tree("01_low_frequency.ipynb"), "historical_threshold_10s")
        self.assertEqual([arg.arg for arg in node.args.args], ["y", "s", "target_recall"])
        self.assertEqual(ast.literal_eval(node.args.defaults[-1]), 0.95)
        loaded_names = {child.id for child in ast.walk(node) if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)}
        self.assertIn("target_recall", loaded_names)
        self.assertNotIn("RSTAR", loaded_names)

    def test_unsupported_waveform_representations_fail_before_data_access(self):
        namespace = load_helpers(notebook_tree("02_high_frequency.ipynb"), ("window_arrays", "feature_timeline"), {})
        for representation in ("F", "S", "H", "W0", "P", "unknown"):
            with self.subTest(representation=representation):
                with self.assertRaisesRegex(ValueError, "supports W1"):
                    namespace["window_arrays"]("missing", 999, [], representation)
                with self.assertRaisesRegex(ValueError, "supports W1"):
                    namespace["feature_timeline"]("missing", 999, representation)

    def test_selected_waveform_feature_layout_matches_window_and_timeline(self):
        base = np.arange(5 * 36, dtype=np.float32).reshape(5, 36)
        mask = np.array([1, 0, 1, 0, 1], dtype=np.uint8)
        namespace = {
            "np": np, "M": 242, "CHANNELS": 520,
            "FOLD_FEATURES": {"A": {1: {"base": base, "start": np.zeros(5, dtype=np.int64), "segment": np.zeros(5, dtype=np.int64)}}},
            "HF_RAW": {1: {"fm": mask[:, None]}},
        }
        load_helpers(notebook_tree("02_high_frequency.ipynb"), ("window_arrays", "feature_timeline"), namespace)
        timeline = namespace["feature_timeline"]("A", 1, "W1")
        window = namespace["window_arrays"]("A", 1, [4], "W1", length=5)
        self.assertEqual(window.shape, (1, 520, 5))
        self.assertEqual(window.dtype, np.float32)
        np.testing.assert_array_equal(window[0], timeline)
        np.testing.assert_array_equal(timeline[:36], base.T)
        np.testing.assert_array_equal(timeline[278:294], np.tile(mask, (16, 1)))
        self.assertFalse(timeline[36:278].any())
        self.assertFalse(timeline[294:].any())

    def test_unsupported_models_fail_before_constructing_layers(self):
        tree = notebook_tree("02_high_frequency.ipynb")
        model = named_node(tree, "AggregateModel", ast.ClassDef)
        # A minimal Module is enough to exercise validation before any layer or tensor creation.
        namespace = {"nn": SimpleNamespace(Module=object)}
        exec(compile(ast.Module(body=[model], type_ignores=[]), "<model contract>", "exec"), namespace)
        for family, representation, length in (("GRU", "F", 2560), ("GRU", "S", 2560), ("TCN", "W1", 2560), ("GRU", "W1", 640)):
            with self.subTest(family=family, representation=representation, length=length):
                with self.assertRaisesRegex(ValueError, "selected GRU/W1/2560"):
                    namespace["AggregateModel"](family, representation, length, 1.0, 1.0)


if __name__ == "__main__":
    unittest.main()
