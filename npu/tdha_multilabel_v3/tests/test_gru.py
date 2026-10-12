import unittest
from unittest.mock import patch
import torch
from torch import nn
from tdha_v3.gru import FP32GRU
from tdha_v3.models import Model
from tdha_v3.npu_preflight import batch_for
from tdha_v3.cost import macs


class GRUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def pair(self, dropout=0.):
        torch.manual_seed(917)
        native=nn.GRU(32,48,2,batch_first=True,dropout=dropout)
        torch.manual_seed(917)
        explicit=FP32GRU(32,48,2,batch_first=True,dropout=dropout)
        self.assertEqual(list(native.state_dict()),list(explicit.state_dict()))
        for key,value in native.state_dict().items():
            self.assertTrue(torch.equal(value,explicit.state_dict()[key]))
        return native,explicit

    def test_forward_hidden_input_and_parameter_gradients(self):
        native,explicit=self.pair()
        for tokens in (1,17,256):
            with self.subTest(tokens=tokens):
                native.zero_grad();explicit.zero_grad()
                x=torch.randn(2,tokens,32,requires_grad=True)
                xx=x.detach().clone().requires_grad_()
                h=torch.randn(2,2,48,requires_grad=True)
                hh=h.detach().clone().requires_grad_()
                a,ha=native(x,h);b,hb=explicit(xx,hh)
                torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
                torch.testing.assert_close(ha,hb,atol=2e-6,rtol=2e-5)
                grad=torch.randn_like(a);gh=torch.randn_like(ha)
                ((a*grad).sum()+(ha*gh).sum()).backward()
                ((b*grad).sum()+(hb*gh).sum()).backward()
                for left,right in [(x.grad,xx.grad),(h.grad,hh.grad)]+[(p.grad,q.grad) for p,q in zip(native.parameters(),explicit.parameters())]:
                    torch.testing.assert_close(left,right,atol=3e-5,rtol=2e-4)

    def test_eval_dropout_checkpoint_and_no_fused_dispatch(self):
        native,explicit=self.pair(.1)
        explicit.load_state_dict(native.state_dict(),strict=True)
        x=torch.randn(2,16,32)
        native.eval();explicit.eval()
        rng=torch.get_rng_state().clone()
        a=native(x)
        # This is the entry point used by nn.GRU; the replacement must bypass it.
        with patch.object(torch._VF,'gru',side_effect=AssertionError('fused GRU called')):
            b=explicit(x)
        for left,right in zip(a,b):torch.testing.assert_close(left,right,atol=2e-6,rtol=2e-5)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        explicit.train();rng=torch.get_rng_state().clone();a=explicit(x)[0]
        self.assertFalse(torch.equal(rng,torch.get_rng_state()))
        torch.set_rng_state(rng);b=explicit(x)[0]
        self.assertTrue(torch.equal(a,b))

    def test_model_parameter_and_mac_count_unchanged(self):
        model=Model('N0','P',42);batch=batch_for('P',2,16)
        count=sum(p.numel() for p in model.parameters())
        current=macs(model,batch)
        old=nn.GRU(32,48,2,batch_first=True,dropout=.1)
        old.load_state_dict(model.backbone.rnn.state_dict())
        model.backbone.rnn=old
        self.assertEqual(count,sum(p.numel() for p in model.parameters()))
        self.assertEqual(current,macs(model,batch))

    def test_reject_half_precision(self):
        _,explicit=self.pair()
        with self.assertRaisesRegex(ValueError,'FP32'):
            explicit.half()(torch.zeros(1,2,32,dtype=torch.float16))
