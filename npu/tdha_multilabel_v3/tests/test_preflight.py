import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import torch
from tdha_v3.common import Blocked, code_hash, dump, read
from tdha_v3.npu_preflight import CASES, exercise, preflight


class PreflightTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("LSA_RUN_TRAINING_TESTS") == "1", "Explicit opt-in required for synthetic optimizer tests")
    def test_all_synthetic_training_paths_on_cpu(self):
        torch.set_num_threads(4)
        self.assertEqual(len(CASES),13)
        for n,r,kind in CASES:
            with self.subTest(net=n,rep=r,kind=kind):
                result=exercise(n,r,kind,torch.device('cpu'),tokens=8)
                self.assertEqual(len(result['checks']),2)

    def test_subprocess_failure_timeout_stale_receipt_and_continuation(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)/'audit/operator_preflight'
            dump(folder/'main-N0-P.json',{'passed':True,'code':code_hash()})
            calls=[]
            def launch(command,**kw):
                index=len(calls);calls.append(kw)
                dest=command[command.index('--output')+1]
                if index==0: return subprocess.CompletedProcess(command,1)
                if index==1: raise subprocess.TimeoutExpired(command,2)
                dump(dest,{'passed':True,'code':code_hash()})
                return subprocess.CompletedProcess(command,0)
            with patch('tdha_v3.runtime.available',return_value=True),patch('tdha_v3.npu_preflight.subprocess.run',side_effect=launch):
                with self.assertRaises(Blocked):preflight(temp,[2],timeout=2)
            result=read(Path(temp)/'operator_preflight.json')
            self.assertEqual(len(calls),13)
            self.assertTrue(result['complete']);self.assertFalse(result['passed'])
            self.assertFalse(result['rows'][0]['passed'])
            self.assertFalse(result['rows'][1]['passed'])
            self.assertTrue(result['rows'][-1]['passed'])
            self.assertEqual(calls[0]['env']['ASCEND_RT_VISIBLE_DEVICES'],'2')
            self.assertEqual(calls[0]['env']['ASCEND_LAUNCH_BLOCKING'],'1')

    def test_dependency_check_reports_missing_reportlab(self):
        from tdha_v3.runtime import dependency_imports
        def fake_import(name):
            if name.startswith('reportlab'):raise ModuleNotFoundError('reportlab')
        with patch('tdha_v3.runtime.importlib.import_module',side_effect=fake_import):
            self.assertEqual(dependency_imports(),{})
            self.assertEqual(list(dependency_imports(include_report=True)),['reportlab.pdfbase.pdfmetrics'])

    def test_failed_report_preserves_primary_error_and_pack_contains_diagnostics(self):
        from tdha_v3.cli import main
        from tdha_v3.reporting import pack
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);primary={'command':'preflight','reason':'original NPU failure'}
            dump(root/'blocker.json',primary)
            font=root/'report.ttf';font.write_bytes(b'mocked font')
            with patch.dict(os.environ,{'TDHA_REPORT_FONT':''}),patch('sys.argv',['tdha_v3','report','--run',temp,'--font',str(font)]),patch('tdha_v3.reporting.validate_report_font',return_value=font),patch('tdha_v3.reporting.report',side_effect=ModuleNotFoundError('reportlab')):
                self.assertEqual(main(),1)
            self.assertEqual(read(root/'blocker.json'),primary)
            self.assertEqual(len(list((root/'audit/errors').glob('report-*.json'))),1)
            dump(root/'operator_preflight.json',{'passed':False})
            pack(root)
            with zipfile.ZipFile(root/'downloads/TDHA_V3_报告与汇总.zip') as z:
                self.assertIn('operator_preflight.json',z.namelist())
                self.assertTrue(any(n.startswith('audit/errors/report-') for n in z.namelist()))
