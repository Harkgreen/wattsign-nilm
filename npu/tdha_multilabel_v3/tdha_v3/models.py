import math
import torch
from torch import nn
from torch.nn import functional as F
from .gru import FP32GRU

class CausalConv(nn.Conv1d):
    def forward(self,x):return super().forward(F.pad(x,((self.kernel_size[0]-1)*self.dilation[0],0)))

class WaveEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv1d(3,8,15,4,7),nn.ReLU(),nn.Conv1d(8,16,9,4,4),nn.ReLU(),nn.Conv1d(16,16,5,2,2),nn.ReLU())
        self.head=nn.Linear(256,16);self.token=nn.Linear(32,64)
    def cycles(self,x):
        h=self.conv(x).reshape(-1,16,8,4)
        return self.head(torch.cat((h.mean(-1),h.amax(-1)),1).flatten(1))
    def forward(self,x,valid):
        b,t,_,_,_=x.shape
        h=self.cycles(x.reshape(-1,3,1024)).reshape(b,t,5,16)
        h=h*valid.unsqueeze(-1)
        mean=h.sum(2)/valid.sum(2).clamp_min(1).unsqueeze(-1)
        maximum=h.masked_fill(~valid.bool().unsqueeze(-1),-1e20).amax(2)
        ok=valid.sum(2)>=4
        return self.token(torch.cat((mean,maximum),-1))*ok.unsqueeze(-1)

class DVIEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv2d(6,24,3,2,1),nn.ReLU(),nn.Conv2d(24,48,3,2,1),nn.ReLU(),nn.Conv2d(48,64,3,2,1),nn.ReLU())
        self.fc=nn.Linear(128,64)
    def forward(self,x,valid):
        b,t=x.shape[:2];h=self.conv(x.reshape(-1,6,32,32)).flatten(2)
        return self.fc(torch.cat((h.mean(-1),h.amax(-1)),1)).reshape(b,t,64)*valid.unsqueeze(-1)

class Front(nn.Module):
    def __init__(self,rep):
        super().__init__();self.rep=rep;self.power=nn.Linear(108,64)
        if rep=='F':self.hf=nn.Sequential(nn.Linear(1200,128),nn.ReLU(),nn.Linear(128,64))
        elif rep=='W1':self.hf=WaveEncoder()
        elif rep=='DVI':self.hf=DVIEncoder()
    def forward(self,batch):
        p=self.power(batch['power'])
        if self.rep=='P':h=torch.zeros_like(p)
        elif self.rep=='F':h=self.hf(batch['extra'])*(batch['extra'][...,600:].sum(-1)>0).unsqueeze(-1)
        else:h=self.hf(batch['extra'],batch['valid'])
        return torch.cat((p,h),-1)

class LargeBlock(nn.Module):
    def __init__(self,d):
        super().__init__();self.large=CausalConv(192,192,31,dilation=d,groups=192);self.small=CausalConv(192,192,3,dilation=d,groups=192)
        self.norm=nn.LayerNorm(192);self.ff=nn.Sequential(nn.Linear(192,768),nn.GELU(),nn.Dropout(.1),nn.Linear(768,192),nn.Dropout(.1))
    def forward(self,x):
        h=(self.large(x.transpose(1,2))+self.small(x.transpose(1,2))).transpose(1,2)
        return x+self.ff(self.norm(h))

class AttentionBlock(nn.Module):
    "Implement attention with explicit standard operators; do not call FlashAttention or Triton."
    def __init__(self):
        super().__init__();self.ln1=nn.LayerNorm(192);self.qkv=nn.Linear(192,576);self.proj=nn.Linear(192,192)
        self.ln2=nn.LayerNorm(192);self.drop=nn.Dropout(.1)
        self.ff=nn.Sequential(nn.Linear(192,768),nn.GELU(),nn.Dropout(.1),nn.Linear(768,192),nn.Dropout(.1))
    def forward(self,x):
        b,t,_=x.shape;q,k,v=self.qkv(self.ln1(x)).reshape(b,t,3,6,32).permute(2,0,3,1,4).unbind(0)
        a=q@k.transpose(-1,-2)/math.sqrt(32)
        causal=torch.ones((t,t),dtype=torch.bool,device=x.device).triu(1)
        a=self.drop(a.masked_fill(causal,-float('inf')).softmax(-1))
        x=x+self.drop(self.proj((a@v).transpose(1,2).reshape(b,t,192)))
        return x+self.ff(self.ln2(x))

class Backbone(nn.Module):
    def __init__(self,name):
        super().__init__();self.name=name
        if name=='N0':
            self.conv=CausalConv(128,32,5);self.rnn=FP32GRU(32,48,2,batch_first=True,dropout=.1)
            self.head=nn.Sequential(nn.Linear(48,32),nn.ReLU(),nn.Dropout(.1),nn.Linear(32,11))
        elif name=='N1':
            self.proj=nn.Linear(128,192);self.blocks=nn.Sequential(*[LargeBlock(d) for d in (1,1,2,2,4,4)])
            self.head=nn.Linear(192,11)
        elif name=='N2':
            self.proj=nn.Linear(512,192);self.blocks=nn.Sequential(*[AttentionBlock() for _ in range(4)])
            self.head=nn.Linear(192,11)
            pe=torch.zeros(64,192);pos=torch.arange(64).float()[:,None];freq=torch.exp(torch.arange(0,192,2).float()*(-math.log(10000)/192))
            pe[:,0::2]=torch.sin(pos*freq);pe[:,1::2]=torch.cos(pos*freq);self.register_buffer('pe',pe)
        else:raise ValueError(name)
    def forward(self,x):
        if self.name=='N0':h,_=self.rnn(F.relu(self.conv(x.transpose(1,2))).transpose(1,2))
        elif self.name=='N1':h=self.blocks(self.proj(x))
        else:
            b,t,c=x.shape;assert t%4==0
            h=self.blocks(self.proj(x.reshape(b,t//4,c*4))+self.pe[None,:t//4])
        return self.head(h[:,-1])

class Model(nn.Module):
    def __init__(self,network,rep,seed):
        super().__init__()
        # Use independent random substreams: representation changes must not alter initial temporal-backbone weights.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+10000);self.front=Front(rep)
            torch.manual_seed(seed+20000);self.backbone=Backbone(network)
    def forward(self,batch):return self.backbone(self.front(batch))

class PretrainModel(nn.Module):
    def __init__(self,rep,seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+10000);self.front=Front(rep)
            torch.manual_seed(seed+30000);self.head=nn.Linear(128,11)
    def forward(self,batch):return self.head(self.front(batch)[:,-1])
