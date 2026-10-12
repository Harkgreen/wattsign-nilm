"""Portable NPU artifact tooling: CPU-only contracts, no model training."""
from pathlib import Path
from collections import OrderedDict
from contextlib import redirect_stdout
from unittest.mock import patch
import importlib.util
import io
import json
import pickle
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1] / 'npu' / 'tdha_multilabel_v3'
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT))

def load(name):
    spec=importlib.util.spec_from_file_location('wattsign_' + name, ROOT / 'tools' / (name + '.py'))
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result

convert=load('convert_weights')
readout=load('local_readout')
statistics=load('final_analysis')
from paths import output_separate, task_path
from tdha_v3.common import digest
import torch


class NpuPortabilityTests(unittest.TestCase):
    def test_inputs_and_outputs_cannot_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for output, source in [(root,root),(root/'new',root),(root,root/'source')]:
                with self.assertRaises(ValueError):output_separate(output,source)
            self.assertEqual(output_separate(root/'output',root/'source'),(root/'output').resolve())

    def test_task_path_rejects_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ('../outside','one/two','',str(Path(tmp).anchor)):
                with self.assertRaises(ValueError):task_path(tmp,name)

    def test_unreviewed_pickle_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'bad.pt'
            with zipfile.ZipFile(file,'w') as archive:archive.writestr('checkpoint/data.pkl',pickle.dumps(eval,protocol=2))
            with patch.object(torch,'load') as loader:
                with self.assertRaises(ValueError):convert.static_review(file)
                loader.assert_not_called()

    def test_cpu_storage_bounds_and_hooks_are_checked(self):
        tensor=torch.arange(6,dtype=torch.float32)
        storage=tensor._typed_storage()
        result=convert.rebuild_cpu_tensor(storage,1,(2,),(2,),False,OrderedDict(),convert.CPUFormat.ND)
        torch.testing.assert_close(result,torch.tensor([1.,3.]))
        with self.assertRaisesRegex(ValueError,'bounds'):
            convert.rebuild_cpu_tensor(storage,5,(2,),(1,),False,OrderedDict(),convert.CPUFormat.ND)
        with self.assertRaisesRegex(ValueError,'hooks'):
            convert.rebuild_cpu_tensor(storage,0,(2,),(1,),False,OrderedDict(callback='untrusted'),convert.CPUFormat.ND)

    def test_readout_plan_is_really_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);archive=root/'archive';weights=root/'weights';output=root/'output'
            tid='N0-W1-A-s42';task={'id':tid,'net':'N0','rep':'W1','fold':'A','seed':42}
            entry=archive/'tasks'/tid;entry.mkdir(parents=True)
            wp=weights/tid;wp.mkdir(parents=True)
            threshold={'logit':[0.0]*11,'active':[True]*11,'high_power':[0.0]*11}
            (entry/'thresholds.json').write_text(json.dumps(threshold))
            (entry/'best.pt').write_bytes(b'read-only plan does not deserialize')
            (wp/'best.cpu.pt').write_bytes(b'read-only plan does not deserialize')
            model='0'*64
            (entry/'done.json').write_text(json.dumps({'task':task,'hashes':{'model':model}}))
            (archive/'freeze.json').write_text(json.dumps({'models':{tid:{'model':model,'thresholds':digest(threshold['logit'])}}}))
            (archive/'selection.json').write_text('{}')
            before={p.relative_to(root).as_posix():p.read_bytes() for p in root.rglob('*') if p.is_file()}
            with redirect_stdout(io.StringIO()), patch.object(torch,'load') as loader:
                readout.main(['--archive',str(archive),'--inputs',str(root/'prepared'),
                    '--weights',str(weights),'--output',str(output),'--tasks',tid,'--plan-only'])
                loader.assert_not_called()
            self.assertFalse(output.exists())
            self.assertEqual(before,{p.relative_to(root).as_posix():p.read_bytes() for p in root.rglob('*') if p.is_file()})

    def test_converter_missing_input_never_creates_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);archive=root/'archive';archive.mkdir();output=root/'output'
            (archive/'freeze.json').write_text('{"models":{}}')
            with self.assertRaises(ValueError):convert.main(['--archive',str(archive),'--output',str(output),'--tasks','missing'])
            self.assertFalse(output.exists())

    def test_cli_help_works_outside_repository(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ('convert_weights','local_readout','final_analysis'):
                result=subprocess.run([sys.executable,str(ROOT/'tools'/(name+'.py')),'--help'],
                    cwd=tmp,capture_output=True,text=True,encoding='utf-8')
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertIn('--execute',result.stdout)
            self.assertEqual(list(Path(tmp).iterdir()),[])

    def test_inference_requires_explicit_cpu_or_cuda_device(self):
        with self.assertRaises(SystemExit) as raised, redirect_stdout(io.StringIO()):
            readout.main(['--archive','archive','--inputs','input','--weights','weights',
                '--output','output','--tasks','N0-W1-A-s42','--execute'])
        self.assertEqual(raised.exception.code,2)

    def test_compact_metadata_preserves_masks_and_integer_endpoints(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'evaluation-metadata';folder.mkdir()
            meta=dict(y=np.zeros((3,11),np.int8),mask=np.ones((3,11),bool),strict=np.ones((3,11),bool),
                time=np.arange(3,dtype=np.int64)*100000,near=np.full(3,2000000,np.int64),
                record=np.full(3,2,np.int64),endpoint=np.arange(3,dtype=np.int64)*5+2609)
            np.savez_compressed(folder/'foldA_cross.npz',**meta)
            with patch.object(statistics,'ARTIFACT',root):
                result=statistics._compact_metadata('A','cross')
                for key in meta:np.testing.assert_array_equal(result[key],meta[key])
                np.testing.assert_array_equal(statistics._pairs(root,'A','cross'),np.column_stack((meta['record'],meta['endpoint'])))

    def test_compact_metadata_rejects_duplicate_endpoints(self):
        import numpy as np
        from tdha_v3.common import Blocked
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'evaluation-metadata';folder.mkdir()
            np.savez_compressed(folder/'foldA_cross.npz',y=np.zeros((2,11),np.int8),mask=np.ones((2,11),bool),strict=np.ones((2,11),bool),
                time=np.arange(2,dtype=np.int64),near=np.ones(2,np.int64),record=np.full(2,2,np.int64),endpoint=np.full(2,2609,np.int64))
            with patch.object(statistics,'ARTIFACT',root),self.assertRaisesRegex(Blocked,'Duplicate'):
                statistics._compact_metadata('A','cross')

    def test_compact_manifest_rejects_changed_file_without_writes(self):
        from tdha_v3.common import Blocked
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);payload=b'{}';(root/'config.json').write_bytes(payload)
            (root/'manifest.json').write_text(json.dumps({'files':[{'path':'config.json','bytes':len(payload),'sha256':'0'*64}]}))
            before={p.name:p.read_bytes() for p in root.iterdir()}
            with patch.object(statistics,'ARTIFACT',root),self.assertRaisesRegex(Blocked,'checksum'):
                statistics.validate_compact_artifacts()
            self.assertEqual(before,{p.name:p.read_bytes() for p in root.iterdir()})


if __name__=='__main__':
    unittest.main()
