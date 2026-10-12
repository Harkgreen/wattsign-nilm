"Pure CPU causal physical transforms; no labels or record identity are accepted."
import numpy as np
import pandas as pd

def physical(raw):
    assert raw.ndim==3 and raw.shape[1:]==(2,1024)
    u=raw[:,0].astype(np.float32)/12.90
    i=raw[:,1].astype(np.float32)*(28.47/32768.)
    p=np.mean(u*i,axis=1,dtype=np.float64)
    u2=np.mean(u*u,axis=1,dtype=np.float64)
    good=np.isfinite(u2)&(u2>1e-12)
    non=i-(np.divide(p,u2,out=np.zeros_like(p),where=good)[:,None]*u).astype(np.float32)
    return u,i,non,good

def quantities(raw):
    u,i,_,_=physical(raw)
    ur=np.sqrt(np.mean(u*u,axis=1,dtype=np.float64)); ir=np.sqrt(np.mean(i*i,axis=1,dtype=np.float64))
    p=np.mean(u*i,axis=1,dtype=np.float64); s=ur*ir; rad=s*s-p*p
    nv=rad>=-1e-5*np.maximum(s*s,1.); pv=s>1e-5
    pf=np.divide(p,s,out=np.zeros_like(p),where=pv)
    a=np.stack((p,s,np.sqrt(np.maximum(rad,0)),ur,ir,pf),1).astype(np.float32)
    m=np.stack((np.isfinite(p),np.isfinite(s),nv,np.isfinite(ur),np.isfinite(ir),pv),1)&np.isfinite(a)
    a[~m]=0
    return a,m.astype(np.uint8)

def harmonics(raw):
    u,i,_,good=physical(raw)
    ur=np.sqrt(np.mean(u*u,axis=1,dtype=np.float64)); ir=np.sqrt(np.mean(i*i,axis=1,dtype=np.float64))
    uf=np.fft.rfft(u,axis=1)[:,1]; fi=np.fft.rfft(i,axis=1)[:,1:26]
    a=np.sqrt(2.)/1024*fi*np.exp(-1j*np.angle(uf)[:,None]*np.arange(1,26))
    ab=np.stack((a.real,a.imag),-1).reshape(-1,50)
    valid=np.isfinite(ab).all(1)&np.isfinite(uf)&(ur>1.)&(np.abs(uf)>1e-3)&good
    relvalid=valid&(ir>1e-6)
    val=np.concatenate((ab,ab/np.maximum(ir[:,None],1e-12)),1).astype(np.float32)
    mask=np.concatenate((np.repeat(valid[:,None],50,1),np.repeat(relvalid[:,None],50,1)),1)
    return np.where(mask,val,0),mask

def aligned(raw):
    u,i,f,g=physical(raw)
    phase=np.angle(np.fft.rfft(u,axis=1)[:,1]); shifts=np.rint(phase*1024/(2*np.pi)).astype(int)
    ix=(np.arange(1024)[None,:]-shifts[:,None])%1024
    out=np.stack([np.take_along_axis(x,ix,1) for x in (u,i,f)],1)
    out[~g]=0
    return out,g

def changes(values,mask,seg):
    n,k=values.shape; out=np.zeros((n,k*3),np.float32); valid=np.zeros_like(out,bool)
    for j in range(k):
        v=values[:,j].astype(np.float64); m=mask[:,j].astype(bool)
        dm=np.r_[False,(seg[1:]==seg[:-1])&m[1:]&m[:-1]]
        baseline=pd.Series(np.where(m,v,np.nan)).groupby(seg,sort=False).transform(lambda a:a.shift(1).rolling(50,min_periods=42).median()).to_numpy()
        bm=m&np.isfinite(baseline)
        out[:,3*j]=np.where(m,v,0)
        out[1:,3*j+1]=np.where(dm[1:],np.diff(v),0)
        out[:,3*j+2]=np.where(bm,v-baseline,0)
        valid[:,3*j:3*j+3]=np.stack((m,dm,bm),1)
    return out,valid

def frequency_changes(v,m,seg):
    n=len(v); x=np.zeros((n,200),np.float32); mask=np.zeros((n,200),bool)
    x[:,:100]=v; mask[:,:100]=m
    for lag,slot in ((5,100),(50,150)):
        ok=m[lag:,:50]&m[:-lag,:50]&(seg[lag:]==seg[:-lag])[:,None]
        x[lag:,slot:slot+50]=np.where(ok,v[lag:,:50]-v[:-lag,:50],0)
        mask[lag:,slot:slot+50]=ok
    return x,mask

def pool5(v,m,seg):
    n=len(v)//5; k=v.shape[1]
    v=v[:n*5].reshape(n,5,k);m=m[:n*5].reshape(n,5,k).astype(bool)
    same=seg[:n*5].reshape(n,5); okseg=np.all(same==same[:,-1:],1)
    count=m.sum(1); enough=(count>=4)&okseg[:,None]
    last=m[:,-1]&okseg[:,None]
    mean=np.divide(np.where(m,v,0).sum(1,dtype=np.float64),count,out=np.zeros((n,k)),where=count>0)
    maximum=np.where(m,v,-np.inf).max(1)
    out=np.stack((v[:,-1],mean,maximum),-1).reshape(n,3*k).astype(np.float32)
    masks=np.stack((last,enough,enough),-1).reshape(n,3*k)
    out[~masks]=0
    return out,masks

def raster5(raw):
    "Directed trajectories over five cycles; deposit each cycle separately without connecting cycle endpoints."
    assert raw.shape==(5,2,1024)
    u,i,f,g=physical(raw)
    ur=np.sqrt(np.mean(u.astype(np.float64)**2)); ir=np.sqrt(np.mean(i.astype(np.float64)**2))
    out=np.zeros((6,32,32),np.float32)
    if ur<1e-6 or ir<1e-6 or not g.all(): return out,False,0.
    clipping=[]
    for branch,current in enumerate((i,f)):
        x=u/ur; y=current/ir
        good=np.isfinite(x)&np.isfinite(y)
        clipping.append(np.mean((np.abs(x)>4)|(np.abs(y)>4)))
        xc=np.clip(x,-4,4); yc=np.clip(y,-4,4)
        # Deposit each valid adjacent edge at its midpoint; never cross cycle boundaries or missing samples.
        for c in range(5):
            edge=good[c,:-1]&good[c,1:]
            xx=np.clip(((xc[c,:-1]+xc[c,1:])/2+4)*4,0,31).astype(int)
            yy=np.clip(((yc[c,:-1]+yc[c,1:])/2+4)*4,0,31).astype(int)
            dx=np.diff(xc[c]);dy=np.diff(yc[c])
            for ch,w in enumerate((np.ones(1023),dx,dy)):
                np.add.at(out[branch*3+ch],(yy[edge],xx[edge]),w[edge]/(5*1023))
    return out,True,float(np.mean(clipping))
