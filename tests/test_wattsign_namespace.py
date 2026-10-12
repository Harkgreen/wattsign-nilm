"""Verify the public namespace stays read-only and preserves object identities."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class WattSignNamespaceTests(unittest.TestCase):
    def environment(self):
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTHONPATH"] = str(ROOT / "src")
        return environment

    def test_safe_import_preserves_implementation_identity(self):
        source = """
import sys
import wattsign
import wattsign.cli as public_cli
import wattsign.data_paths as public_paths
import wattsign.notebook_view as public_view
import load_signature_analysis.cli as implementation_cli
import load_signature_analysis.data_paths as implementation_paths
import load_signature_analysis.notebook_view as implementation_view
assert public_cli.main is implementation_cli.main
assert public_cli.load_result_index is implementation_cli.load_result_index
assert public_paths.DataPaths is implementation_paths.DataPaths
assert public_paths.DataPathError is implementation_paths.DataPathError
assert public_paths.resolve_data_paths is implementation_paths.resolve_data_paths
assert public_view.show_experiment is implementation_view.show_experiment
assert not any(name == 'torch' or name.startswith('torch.') for name in sys.modules)
assert not any(name.startswith('load_signature_analysis.') and 'training' in name for name in sys.modules)
"""
        result = subprocess.run([sys.executable, "-B", "-c", source],
                                env=self.environment(), capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_module_entrypoint_missing_results_creates_nothing(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                [sys.executable, "-B", "-m", "wattsign", "--project-root", temporary, "results", "list"],
                env=self.environment(), capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("results/index.json", result.stderr.replace("\\", "/"))
            self.assertEqual(list(Path(temporary).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
