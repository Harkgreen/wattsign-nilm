import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import zipfile
import numpy as np
import torch

from tdha_v3.common import Blocked,PROTOCOL,array_hash,code_hash,digest,dump,file_hash,read
from tdha_v3.data import labels,project
from tdha_v3.npu_preflight import record_skip
from tdha_v3.runtime import stage_a,task
from tdha_v3.training import state_hash


def label_zip(events):
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w') as z:
        text='id,Time,event\n'+''.join(f'{i},2023/05/18 16:01:{i*10:02d}.000000,{e}\n' for i,e in enumerate(events))
        z.writestr('sign.csv',text)
    buffer.seek(0)
    return zipfile.ZipFile(buffer)


class LabelUpgradeTests(unittest.TestCase):
    def test_invalid_interval_and_both_boundary_masks(self):
        with label_zip(['1-0','1-1','1)0','1-1','1-0']) as z:
            lab=labels(z,'sign.csv',{},single_class=3)
        origin=lab['time'][0]
        seconds=np.array([19.849999,19.85,20,25,29.999999,30,30.15,30.150001,30.5,30.500001])
        t=origin+np.rint(seconds*1e6).astype(np.int64)
        result=project(lab,t)
        np.testing.assert_array_equal(result['known'][:,3],[1,0,0,0,0,0,0,1,1,1])
        np.testing.assert_array_equal(result['strict'][:,3],[0,0,0,0,0,0,0,0,0,1])
        self.assertTrue(result['known'][:,5].all())
        self.assertEqual(lab['label_anomalies'][0]['event'],'1)0')
        self.assertEqual(lab['label_anomalies'][0]['unknown_until'],'2023/05/18 16:01:30.000000')
        # An unknown interval between identical ON states still has supervision boundaries; masking only numerical state changes is insufficient.
        self.assertEqual(result['switch_distance'][5,3],0)

    def test_leading_trailing_consecutive_invalid_and_unsupported(self):
        for events in (['1)0','1-1'],['1-1','1)0'],['1-1','1)0','1)0','1-1']):
            with label_zip(events) as z:lab=labels(z,'sign.csv',{},single_class=3)
            t=np.arange(lab['time'][0],lab['time'][-1]+1,1000000)
            out=project(lab,t)
            ix=np.searchsorted(lab['time'],t,side='right')-1
            self.assertFalse(out['known'][~lab['state_valid'][ix,3],3].any())
        with label_zip(['1-1','unknown']) as z:
            with self.assertRaisesRegex(Blocked,'CSV行3'):labels(z,'sign.csv',{},single_class=3)

    def test_legal_single_and_aggregate_keep_original_interface(self):
        with label_zip(['1-0','1-1','1-1','1-0']) as z:single=labels(z,'sign.csv',{},single_class=5)
        self.assertEqual(set(single),{'time','state','any_switch'})
        np.testing.assert_array_equal(single['state'][:,5],[0,1,1,0])
        np.testing.assert_array_equal(single['any_switch'],single['time'][[1,3]])
        with label_zip(['Socket0: 0 Socket1: 1 Socket2: 0 Socket3: 0 Socket4: 0',
                        'Socket0: 0 Socket1: 1 Socket2: 0 Socket3: 0 Socket4: 1']) as z:
            agg=labels(z,'sign.csv',{5:{'socket':4},0:{'socket':1}})
        self.assertEqual(set(agg),{'time','state','any_switch'})
        np.testing.assert_array_equal(agg['state'][:,5],[0,1])
        out=project(agg,agg['time'][0]+np.array([0,9850000,10000000]))
        np.testing.assert_array_equal(out['known'][:,5],[1,0,0])
        self.assertTrue(out['known'][:,0].all())

    def test_skip_preserves_actual_receipt_without_calling_device(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'operator_preflight.json';dump(path,{'passed':True,'code':'old'})
            original=path.read_bytes()
            with patch('tdha_v3.npu_preflight.subprocess.run') as call:record_skip(tmp,[0,1,2])
            call.assert_not_called();self.assertEqual(path.read_bytes(),original)
            receipt=read(next((Path(tmp)/'audit/operator_preflight_skips').glob('*.json')))
            self.assertIsNone(receipt['passed']);self.assertFalse(receipt['complete']);self.assertTrue(receipt['skipped'])
            self.assertTrue(receipt['previous']['passed'])

