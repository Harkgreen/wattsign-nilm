"""Statistical parity and safety contracts; no models, training or NPU access."""
from __future__ import annotations
import ast
import importlib.util
import json
import time
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np

WORK = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(WORK / 'tools'))
sys.path.insert(0, str(WORK))
spec = importlib.util.spec_from_file_location('local_final_analysis', WORK / 'tools' / 'final_analysis.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
from tdha_v3.metrics import block_bootstrap_pair as reference_pair


def reference_absent(records, iterations):
    # Isolate the copied pure reference function without importing training.
    tree = ast.parse((WORK / 'tdha_v3' / 'analysis.py').read_text(encoding='utf-8'))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'absent_bootstrap')
    class Constants(ast.NodeTransformer):
        def visit_Constant(self, node):
            if node.value == 2000:
                return ast.copy_location(ast.Constant(iterations), node)
            if node.value == 40000:
                return ast.copy_location(ast.Constant(iterations * 20), node)
            return node
    isolated = Constants().visit(ast.Module(body=[function], type_ignores=[]))
    ast.fix_missing_locations(isolated)
    namespace = {'np': np}
    exec(compile(isolated, 'frozen_analysis_absent_pure', 'exec'), namespace)
    return namespace['absent_bootstrap'](records)


def records(seed=77, sparse=False, tie=True, third=False):
    rng = np.random.default_rng(seed)
    result = []
    for fold in ('A', 'B'):
        for record in ([1] if third else [2, 3]):
            n = 151
            labels = rng.integers(2, size=(n, 11), dtype=np.int8)
            mask = rng.random((n, 11)) > .1
            if sparse:
                labels[:, 0] = 0
                labels[2, 0] = 1
                mask[:, 0] = True
            present = [True] * 6 + [False] * 5
            labels[:, 6:] = 0
            scores = []
            for sign in (0., .13):
                z = rng.normal(size=(3, n, 11)).astype(np.float32) + labels[None] * .9 + sign
                if tie:
                    z = np.round(z, 1)
                scores.append(z)
            threshold = rng.normal(size=(3, 11)) * .1
            threshold[:, 9:] = np.nan
            result.append({'fold': fold, 'record': record, 'y': labels, 'mask': mask,
                           'time': np.arange(n, dtype=np.int64) * 3_000_000,
                           'near': np.full(n, 2_000_000), 'a': scores[0], 'b': scores[1],
                           'ta': threshold, 'tb': threshold + .05,
                           'trained': [True] * 9 + [False] * 2, 'present': present})
    return result


class FrozenAnalysisContracts(unittest.TestCase):
    def assert_parity(self, expected, actual):
        self.assertEqual(expected['draws'], actual['draws'])
        self.assertEqual(expected['valid'], actual['valid'])
        self.assertEqual(expected['invalid'], actual['invalid'])
        np.testing.assert_allclose(expected['samples'], actual['samples'], rtol=0, atol=2e-15)
        if expected['CI'] is None:
            self.assertIsNone(actual['CI'])
        else:
            np.testing.assert_allclose(expected['CI'], actual['CI'], rtol=0, atol=2e-15)

    def test_paired_ties_and_unknown_masks(self):
        sample = records()
        self.assert_parity(reference_pair(sample, iterations=37),
                           adapter.block_bootstrap_pair_fast(sample, iterations=37, batch_size=8))

    def test_invalid_draws_keep_identical_rng_order(self):
        sample = records(sparse=True, tie=False)
        expected = reference_pair(sample, iterations=43)
        actual = adapter.block_bootstrap_pair_fast(sample, iterations=43, batch_size=11)
        self.assertGreater(expected['invalid'], 0)
        self.assert_parity(expected, actual)

    def test_cumsum_fallback_preserves_weighted_ties(self):
        sample = records(sparse=True)
        # Force the bounded-memory fallback by setting the prepared prefixes to None.
        original = adapter._prepare_class
        def fallback(*args):
            value = original(*args)
            for heads in value['heads'].values():
                for head in heads:
                    head['prefix'] = None
            return value
        adapter._prepare_class = fallback
        try:
            self.assert_parity(reference_pair(sample, iterations=31),
                               adapter.block_bootstrap_pair_fast(sample, iterations=31, batch_size=7))
        finally:
            adapter._prepare_class = original

    def test_absent_uses_all_eligible_heads_and_per_class_intervals(self):
        sample = records(third=True)
        expected, actual = reference_absent(sample, 61), adapter.absent_bootstrap_fast(sample, iterations=61, batch_size=9)
        self.assert_parity(expected, actual)
        self.assertEqual(set(actual['per_class_point']), {f'{fold}_{c}' for fold in ('A', 'B') for c in (6, 7, 8)})
        self.assertEqual(expected['per_class_point'], actual['per_class_point'])
        for key in expected['per_class_CI']:
            np.testing.assert_allclose(expected['per_class_CI'][key], actual['per_class_CI'][key], rtol=0, atol=2e-15)

    def test_absent_invalid_samples_keep_original_attempts(self):
        sample = records(third=True)
        for r in sample:
            r['mask'][:, 6:] = False
            r['mask'][2, 6:9] = True
        expected = reference_absent(sample, 41)
        self.assertGreater(expected['invalid'], 0)
        self.assert_parity(expected, adapter.absent_bootstrap_fast(sample, iterations=41, batch_size=17))

    def test_absent_head_without_known_off_support_is_not_silently_dropped(self):
        sample = records(third=True)
        for r in sample:
            r['mask'][:, 6] = False
        expected = reference_absent(sample, 7)
        self.assertEqual(expected['valid'], 0)
        self.assert_parity(expected, adapter.absent_bootstrap_fast(sample, iterations=7, batch_size=9))

    def test_training_or_npu_completion_claims_are_rejected_without_writes(self):
        marker = {'completed': True, 'completed_tasks': 32, 'trained': True,
                  'thresholds_changed': False, 'backend': 'cuda', 'original_npu_status': 'partial'}
        from tdha_v3.common import Blocked
        with patch.object(adapter, 'read', return_value=marker), patch.object(adapter, 'dump') as writer:
            with self.assertRaises(Blocked):
                adapter.validate_local_completion(WORK / 'run')
            writer.assert_not_called()
        marker['trained'] = False
        marker['backend'] = 'npu'
        with patch.object(adapter, 'read', return_value=marker), patch.object(adapter, 'dump') as writer:
            with self.assertRaises(Blocked):
                adapter.validate_local_completion(WORK / 'run')
            writer.assert_not_called()

    def test_original_protocol_constants_remain_frozen(self):
        self.assertEqual((adapter.ITERATIONS, adapter.BOOTSTRAP_SEED, adapter.BLOCK_US, adapter.SEEDS),
                         (2000, 42, 60_000_000, (11, 42, 73)))

    def test_adapter_does_not_import_training_or_npu_or_call_threshold_selection(self):
        tree = ast.parse((WORK / 'tools' / 'final_analysis.py').read_text(encoding='utf-8-sig'))
        imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        self.assertFalse(any(name and ('training' in name or 'torch' in name or 'runtime' in name) for name in imports))
        imports += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
        self.assertFalse(any(name and ('torch' in name or 'training' in name) for name in imports))
        calls = [node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
        self.assertFalse(set(calls) & {'threshold', 'fit', 'backend', 'doctor', 'benchmark', 'schedule'})


if __name__ == '__main__':
    unittest.main()
