import torch
from torch import nn

def macs(model,batch):
    "Count one multiply-add as one MAC; exclude nonlinearities, normalization, FFT, rasterization and data movement."
    total=[0];hooks=[]
    def hook(m,inputs,out):
        if isinstance(m,(nn.Conv1d,nn.Conv2d)):
            k=1
            for v in m.kernel_size:k*=v
            total[0]+=out.numel()*m.in_channels//m.groups*k
        elif isinstance(m,nn.Linear):total[0]+=out.numel()*m.in_features
        elif isinstance(m,nn.GRU):
            b,t,_=inputs[0].shape;h=m.hidden_size
            for layer in range(m.num_layers):total[0]+=b*t*3*h*((m.input_size if layer==0 else h)+h)
    for m in model.modules():
        if isinstance(m,(nn.Conv1d,nn.Conv2d,nn.Linear,nn.GRU)):hooks.append(m.register_forward_hook(hook))
    previous=model.training;model.eval()
    with torch.inference_mode():model(batch)
    for h in hooks:h.remove()
    b,t=batch['power'].shape[:2]
    if hasattr(model,'backbone') and model.backbone.name=='N2':total[0]+=4*b*2*(t//4)**2*192
    model.train(previous)
    return int(total[0]//b)
