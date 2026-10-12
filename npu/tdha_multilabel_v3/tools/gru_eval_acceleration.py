"""Evaluation-only CUDA Graph replay of the unchanged TDHA N0 backbone.

This helper leaves source modules and frozen weights unchanged. The serialized
FP32GRU already uses nn.GRU parameter names and PyTorch gate ordering. No hidden
state is carried between independent 256-token windows. Call only in eval mode.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import time
import copy
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import nn
from collections import OrderedDict
import weakref

TOLERANCE={"rtol":2e-5,"atol":2e-5}


def configure_fp32():
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.set_float32_matmul_precision("highest")


class CUDAGraphBackbone:
    """Replay the exact original N0 backbone operations at a fixed window shape.

    Graph captures the original FP32GRU, not nn.GRU/cuDNN. Every replay creates
    the original zero hidden state for each window. Outputs are cloned before
    returning so a later replay cannot overwrite returned predictions.
    """
    def __init__(self, backbone, max_shapes=3):
        self.backbone_ref=weakref.ref(backbone)
        self.graphs=OrderedDict()
        self.max_shapes=max_shapes

    def __call__(self, windows):
        backbone=self.backbone_ref()
        if backbone is None or backbone.training:
            raise ValueError("CUDA Graph requires the existing eval-mode backbone")
        if torch.is_grad_enabled():
            raise ValueError("CUDA Graph is restricted to inference/no-grad execution")
        if windows.device.type!="cuda" or windows.dtype!=torch.float32 or windows.ndim!=3:
            raise ValueError("Expected CUDA FP32 [batch,256,128] independent windows")
        if tuple(windows.shape[1:])!=(256,128) or len(windows)==0:
            raise ValueError("Every held-out window must contain exactly 256 independent tokens")
        key=(tuple(windows.shape),windows.device.index)
        if key not in self.graphs:
            if len(self.graphs)>=self.max_shapes:
                self.graphs.popitem(last=False)
            static_input=torch.empty(windows.shape,dtype=windows.dtype,device=windows.device)
            static_input.copy_(windows)
            stream=torch.cuda.Stream(device=windows.device)
            stream.wait_stream(torch.cuda.current_stream(windows.device))
            with torch.cuda.stream(stream):
                for _ in range(2): backbone(static_input)
            torch.cuda.current_stream(windows.device).wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_output=backbone(static_input)
            self.graphs[key]=(graph,static_input,static_output)
        else:
            self.graphs.move_to_end(key)
        graph,static_input,static_output=self.graphs[key]
        static_input.copy_(windows)
        graph.replay()
        return static_output.clone()


def eval_backbone(model, windows):
    """Use exact-operation CUDA Graph for N0; other backbones keep normal calls.

    Does not modify modules, state_dict, source code, weights, or thresholds.
    Parent runner should retain its separate real full-window reference checks.
    """
    if model.training:
        raise ValueError("Only frozen eval-mode inference is supported")
    if model.backbone.name!="N0" or windows.device.type!="cuda":
        return model.backbone(windows)
    if type(model.backbone.rnn).__name__!="FP32GRU":
        raise ValueError("Original explicit FP32GRU must remain intact")
    cache=model.__dict__.get("_local_cuda_graph_backbone")
    if cache is None:
        cache=CUDAGraphBackbone(model.backbone)
        model.__dict__["_local_cuda_graph_backbone"]=cache
    return cache(windows)


def clear_backbone_graph_cache(model):
    model.__dict__.pop("_local_cuda_graph_backbone",None)


def clear_graph_cache(model):
    """Release all per-shape CUDA graph buffers after a frozen task."""
    clear_backbone_graph_cache(model)


def replace_gru_for_eval(model):
    raise RuntimeError("nn.GRU/cuDNN substitution failed strict 2e-5 checks and must not be used; use eval_backbone with the original FP32GRU")

