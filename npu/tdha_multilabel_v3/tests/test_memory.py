import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import torch
from tdha_v3.memory import ArrayCache,BatchReader,encoding_windows,device_encoding,settings
from tdha_v3.models import Model
from tdha_v3.training import state_hash,rng_hash,rng_state


class Samples:
    def __getitem__(self,i):
        return {'power':np.full((8,108),i/10,np.float32),'y':np.full(11,i%2,np.float32)}


class MemoryTests(unittest.TestCase):
    def test_ram_readonly_reuse_budget_and_low_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);value=np.arange(64,dtype=np.float32).reshape(8,8)
            np.save(root/'a.npy',value);value.tofile(root/'raw.bin');np.save(root/'b.npy',value+1)
            c=ArrayCache(value.nbytes,100,lambda:100000)
            a=c.get(root/'a.npy');self.assertNotIsInstance(a,np.memmap)
            np.testing.assert_array_equal(a,value);self.assertIs(c.get(root/'a.npy'),a)
            with self.assertRaises(ValueError):a[0,0]=7
            self.assertIsInstance(c.get(root/'b.npy'),np.memmap)
            self.assertEqual(c.bytes,value.nbytes)
            low=ArrayCache(100000,100,lambda:100)
            self.assertIsInstance(low.get(root/'a.npy'),np.memmap)
            raw=ArrayCache(100000,0,lambda:100000).get(root/'raw.bin',np.float32,(8,8))
            np.testing.assert_array_equal(raw,value)
            self.assertIsInstance(ArrayCache(0,0,lambda:100000).get(root/'a.npy'),np.memmap)

    @unittest.skipUnless(os.environ.get("LSA_RUN_TRAINING_TESTS") == "1", "Explicit opt-in required for synthetic optimizer tests")
    def test_prefetch_preserves_batches_rng_and_model_update(self):
        results=[]
        for enabled in (False,True):
            model=Model('N0','P',42);opt=torch.optim.AdamW(model.parameters(),foreach=False)
            torch.manual_seed(42);seen=[]
            with BatchReader(enabled) as reader:
                for batch in reader.batches(Samples(),[[3,1],[1,2],[0]],torch.device('cpu')):
                    seen.append(batch['power'][:,0,0].tolist())
                    loss=torch.nn.functional.binary_cross_entropy_with_logits(model(batch),batch['y'])
                    loss.backward()
                opt.step()
            results.append((seen,state_hash(model.state_dict()),rng_hash(rng_state(torch.device('cpu')))))
        self.assertEqual(results[0],results[1])

    def test_prefetch_exception_is_not_hidden(self):
        class Broken:
            def __getitem__(self,i):raise ValueError('bad sample')
        with BatchReader(True) as reader:
            with self.assertRaisesRegex(ValueError,'bad sample'):
                list(reader.batches(Broken(),[[0]],torch.device('cpu')))

    def test_resident_window_gather_matches_reference(self):
        a=np.arange(320*128,dtype=np.float32).reshape(320,128)
        ends=np.array([255,319,260,255]);device=torch.device('cpu')
        reference=encoding_windows(a,ends,device)
        gathered=encoding_windows(a,ends,device,torch.from_numpy(a))
        self.assertTrue(torch.equal(reference,gathered))
        self.assertIsNone(device_encoding(a,device))

    def test_invalid_limits_fail(self):
        for variable,value in [('TDHA_RAM_CACHE_GB','nan'),('TDHA_EVAL_CACHE_GB','-1'),('TDHA_PREFETCH','2')]:
            with patch.dict(os.environ,{variable:value}):
                with self.assertRaises(ValueError):settings()

    def test_device_cache_respects_budget_and_free_memory(self):
        a=np.zeros((300,128),np.float32);device=SimpleNamespace(type='npu')
        cfg={'eval_bytes':a.nbytes,'hbm_reserve_bytes':100}
        npu=SimpleNamespace(mem_get_info=Mock(return_value=(a.nbytes+99,1000000)))
        with patch('tdha_v3.memory.settings',return_value=cfg), patch.object(torch,'npu',npu,create=True):
            with patch('tdha_v3.memory.torch.from_numpy') as transfer:
                self.assertIsNone(device_encoding(a,device));transfer.assert_not_called()
                npu.mem_get_info.return_value=(a.nbytes+100,1000000)
                self.assertIs(device_encoding(a,device),transfer.return_value.to.return_value)
                transfer.return_value.to.assert_called_once_with(device)
                transfer.reset_mock();cfg['eval_bytes']=a.nbytes-1
                self.assertIsNone(device_encoding(a,device));transfer.assert_not_called()
