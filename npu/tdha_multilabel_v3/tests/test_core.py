import unittest
import os
import tempfile
from pathlib import Path
import numpy as np
import torch
from tdha_v3.common import *
from tdha_v3.physics import *
from tdha_v3.data import project,segments
from tdha_v3.metrics import threshold,weighted_ap,conditional
from tdha_v3.models import Model,Backbone
from tdha_v3.training import state_hash,rng_state,rng_hash
from unittest.mock import patch

torch.set_num_threads(1)

def wave(phase=0):
    t=np.arange(1024)*2*np.pi/1024+phase
    return np.rint(np.stack((325*np.sin(t)*12.90,(np.sin(t)+.2*np.sin(3*t))*32768/28.47))).astype(np.int16)

class Tests(unittest.TestCase):
    def test_adc(self):
        raw=wave()[None];b,m=quantities(raw)
        self.assertTrue(m.all());self.assertAlmostEqual(float(b[0,0]),162.5,delta=.1)
        f,fm=harmonics(raw);self.assertTrue(fm.all());self.assertAlmostEqual(float(np.hypot(f[0,4],f[0,5])),.2/np.sqrt(2),delta=.001)
        self.assertFalse(harmonics(np.zeros_like(raw))[1].any())
    def test_nonactive_and_phase(self):
        raw=wave();raw[1]=np.rint(raw[0]/12.90/325*32768/28.47).astype(np.int16)
        self.assertLess(np.abs(physical(raw[None])[2]).max(),.002)
        raw=wave()[None];a,_=aligned(raw);b,_=aligned(np.roll(raw,137,-1));np.testing.assert_array_equal(a,b)
        # Compare the original V2 torch path with the new CPU preprocessing under identical scaling.
        t=torch.from_numpy(raw);u=t[:,0].float()/12.90;i=t[:,1].float()*(28.47/32768.)
        f=i-((u*i).mean(-1)/(u*u).mean(-1))[:,None]*u
        shifts=torch.round(torch.angle(torch.fft.rfft(u)[:,1])*1024/(2*np.pi)).long()
        positions=(torch.arange(1024)[None]-shifts[:,None])%1024
        expected=torch.gather(torch.stack((u,i,f),1),2,positions[:,None].expand(-1,3,-1)).numpy()
        np.testing.assert_allclose(a,expected,rtol=2e-5,atol=2e-5)
    def test_trajectory(self):
        raw=np.repeat(wave()[None],5,0);d,v,c=raster5(raw);reverse,_,_=raster5(raw[:,:,::-1])
        self.assertTrue(v);self.assertEqual(d.shape,(6,32,32));self.assertEqual(c,0)
        np.testing.assert_allclose(d[[0,3]],reverse[[0,3]],atol=1e-6)
        np.testing.assert_allclose(d[[1,2,4,5]],-reverse[[1,2,4,5]],atol=1e-6)
        self.assertFalse(raster5(np.zeros_like(raw))[1])
    def test_baseline_no_future(self):
        x=np.arange(100,dtype=np.float32)[:,None];m=np.ones_like(x,bool);seg=np.zeros(100,int)
        a,am=changes(x,m,seg);self.assertFalse(am[41,2]);self.assertTrue(am[42,2]);self.assertAlmostEqual(a[50,2],25.5)
        xx=x.copy();xx[70:]=999;np.testing.assert_array_equal(a[:70],changes(xx,m,seg)[0][:70])
        seg[60:]=1;self.assertFalse(changes(x,m,seg)[1][60,1:].any())
    def test_token(self):
        x=np.arange(10,dtype=np.float32)[:,None];m=np.ones_like(x,bool);seg=np.zeros(10,int)
        v,k=pool5(x,m,seg);np.testing.assert_array_equal(v,[[4,2,4],[9,7,9]])
        m[:2]=False;v,k=pool5(x,m,seg);self.assertFalse(k[0,1]);self.assertEqual(v[0,1],0)
    def test_individual_label_boundaries(self):
        state=np.zeros((3,11),np.uint8);state[1:,5]=1
        lab={'time':np.array([0,1000000,2000000]),'state':state,'any_switch':np.array([1000000])}
        a=project(lab,np.array([500000,850000,1000000,1150000,1150001,2500000]))
        np.testing.assert_array_equal(a['known'][:,5],[1,0,0,0,1,0])
        self.assertTrue(a['known'][:5,0].all());self.assertFalse(a['y'][:,0].any())
    def test_threshold(self):
        y=np.array([1]*20+[0]*4);z=np.array([.1]*2+[.9]*18+[.2,.5,.95,1],np.float32)
        t=threshold(y,z);self.assertEqual(t,float(np.float32(.1)));self.assertGreaterEqual((z[y==1]>=t).mean(),.95)
        self.assertIsNone(threshold(np.zeros(4),np.arange(4)))
        self.assertEqual(np.float32(t).tobytes(),np.array(t,np.float32).tobytes())
    def test_weighted_bootstrap(self):
        from sklearn.metrics import average_precision_score
        y=np.array([0,1,0,1,1]);z=np.array([.2,.5,.5,.7,.7]);w=np.array([2,1,3,0,2])
        ii=np.repeat(np.arange(5),w)
        self.assertAlmostEqual(weighted_ap(y,z,w),average_precision_score(y[ii],z[ii]))
    def test_power_bins(self):
        y=np.array([1,0,1,0]);p=np.array([49.9,50,50,50]);z=np.array([1,0,1,1])
        score,cov,_=conditional(y,z,p);self.assertEqual(cov,.5);self.assertEqual(score,.75)
    def test_models_and_initialization(self):
        for n in ('N0','N1','N2'):
            hashes=[]
            for rep in ('P','F','W1','DVI'):
                model=Model(n,rep,42);hashes.append(state_hash(model.backbone.state_dict()))
                batch={'power':torch.randn(2,4,108)}
                if rep=='F':batch['extra']=torch.randn(2,4,1200);batch['extra'][...,600:]=1
                if rep=='W1':batch.update(extra=torch.randn(2,4,5,3,1024),valid=torch.ones(2,4,5))
                if rep=='DVI':batch.update(extra=torch.randn(2,4,6,32,32),valid=torch.ones(2,4))
                out=model(batch);self.assertEqual(out.shape,(2,11));out.sum().backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
                model.eval();before=rng_hash(rng_state(torch.device('cpu')))
                with torch.inference_mode():a=model(batch);b=model(batch)
                self.assertTrue(torch.equal(a,b));self.assertEqual(before,rng_hash(rng_state(torch.device('cpu'))))
                batch['y']=torch.full((2,11),999.);batch['mask']=torch.zeros(2,11)
                with torch.inference_mode():c=model(batch)
                self.assertTrue(torch.equal(a,c),'监督标签进入模型输入')
            self.assertEqual(len(set(hashes)),1)
    def test_causal_backbones(self):
        for n in ('N0','N1','N2'):
            model=Backbone(n).eval();x=torch.randn(2,16,128);x2=x.clone();x2[:,8:]=999
            with torch.inference_mode():
                a=model(x[:,:8]);b=model(x2[:,:8])
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        m=Backbone('N2').eval();x=torch.randn(2,8,192);y=x.clone();y[:,4:]=100
        with torch.inference_mode():a=m.blocks(x);b=m.blocks(y)
        torch.testing.assert_close(a[:,:4],b[:,:4],rtol=0,atol=0)

    def test_pairing_and_missing(self):
        import zipfile,csv,io
        from tdha_v3.data import extract
        fields=['HomeID','RecvTime','InternalIndex','Channel','Rate','Value']
        def row(ch,idx):return dict(zip(fields,['1',f'2023/01/01 00:00:00.{idx*20000:06d}',str(idx),str(ch),'1024','|'.join(map(str,wave()[ch-1]))]))
        with tempfile.TemporaryDirectory() as td:
            td=Path(td)
            for missing in (False,True):
                stream=io.StringIO();writer=csv.DictWriter(stream,fields);writer.writeheader();writer.writerows([row(2,0),row(1,0)]+([row(1,1)] if missing else [row(1,1),row(2,1)]))
                zpath=td/('missing.zip' if missing else 'good.zip')
                with zipfile.ZipFile(zpath,'w') as z:z.writestr('x.csv',stream.getvalue())
                if missing:
                    with self.assertRaises(Blocked):extract(zpath,'x.csv',td/'bad')
                else:
                    a=extract(zpath,'x.csv',td/'good');self.assertEqual(len(a['time']),2);self.assertEqual(a['continuity'][-1],0)

    def test_block_bootstrap_repeatable(self):
        from tdha_v3.metrics import block_bootstrap_pair
        y=np.tile(np.array([0,1,0,1])[:,None],(1,11));z=np.array([-.5,.4,.1,.8],np.float32)
        r={'fold':'A','record':2,'time':np.arange(4)*60000000,'y':y,'mask':np.ones_like(y,bool),'a':np.tile(z[None,:,None],(3,1,11)),
           'b':np.tile((z+y[:,0]*.3)[None,:,None],(3,1,11)),'ta':np.zeros((3,11)),'tb':np.zeros((3,11)),
           'trained':[True]*11,'present':[True]*11}
        a=block_bootstrap_pair([r],iterations=12);b=block_bootstrap_pair([r],iterations=12)
        self.assertTrue(a['complete']);np.testing.assert_array_equal(a['samples'],b['samples']);self.assertEqual(a['draws'],b['draws'])

    @unittest.skipUnless(os.environ.get("LSA_RUN_TRAINING_TESTS") == "1", "Explicit opt-in required for synthetic optimizer tests")
    def test_fit_resume_and_reference_inference(self):
        "Small synthetic training validates the execution path only; it is not an experimental performance result."
        from tdha_v3 import training as tr
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);fold=root/'cache/foldA';fold.mkdir(parents=True)
            n=3000;label=np.zeros((n,11),np.uint8);label[::2]=1
            for r in range(1,8):
                dest=root/'cache'/f'record{r}';dest.mkdir()
                for k,a in {'time':np.arange(n,dtype=np.int64)*20000,'y':label,'known':np.ones_like(label,bool),'strict':np.ones_like(label,bool),'near':np.full(n,2000000),'basic':np.ones((n,6),np.float32)}.items():save_array(dest/f'{k}.npy',a)
                save_array(fold/f'r{r}_power.npy',np.zeros((n//5,108),np.float32));save_array(fold/f'r{r}_F.npy',np.zeros((n//5,1200),np.float32))
            pairs=np.array([[2,2699],[2,2704],[2,2749],[2,2754]],np.int64)
            save_array(fold/'pairs.npy',pairs);save_array(fold/'order42.npy',np.tile(np.arange(4),(40,1)))
            ep={f'{r}_{part}':([2699,2704] if r==2 else []) for r in range(1,8) for part in ('time','calibration','train_natural','cross')}
            dump(fold/'endpoints.json',ep);dump(fold/'normalizer.json',{'wave_scale':[230,1]});dump(root/'benchmark.json',{'microbatch':2})
            dump(root/'audit/mapping.json',{str(r):{str(c):{'socket':c%5} for c in range(11)} for r in range(1,8)})
            task={'id':'synthetic-reference','net':'N0','rep':'P','fold':'A','seed':42}
            # A fixed calibration loss should early-stop at epoch 6; changing this behavior requires an explicit protocol change.
            with patch.object(tr,'calibration_loss',return_value=.7):
                a=tr.fit(root,task,torch.device('cpu'))
                original=tr.checkpoint;calls=[0]
                def interrupted(path,state):
                    original(path,state)
                    if Path(path).name=='last.pt':
                        calls[0]+=1
                        if calls[0]==3:raise RuntimeError('模拟进程中断')
                task2=task|{'id':'synthetic-resume'}
                with patch.object(tr,'checkpoint',side_effect=interrupted):
                    with self.assertRaisesRegex(RuntimeError,'模拟'):tr.fit(root,task2,torch.device('cpu'))
                b=tr.fit(root,task2,torch.device('cpu'))
            self.assertEqual(a['hashes'],b['hashes'])

if __name__=='__main__':unittest.main()
