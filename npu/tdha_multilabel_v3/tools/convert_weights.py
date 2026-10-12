"""Convert frozen NPU best checkpoints to CPU with a restricted weights-only loader.

Only the reviewed NPU serialization globals are permitted. No torch_npu import,
training, forward pass, optimizer restore, or arbitrary pickle loading occurs.
The CPU storage representation follows Ascend v2.7.1 storage.py _reduce_ex:
https://github.com/Ascend/pytorch/blob/v2.7.1/torch_npu/utils/storage.py
The format enum and CPU storage reconstruction adapt Ascend/PyTorch v2.7.1.
Copyright (c) 2020 Huawei Technologies Co., Ltd and PyTorch contributors.
Adapted portions retain the upstream BSD terms in ../licenses/Ascend-PyTorch-LICENSE.txt.
Changes: CPU-only bounds checks, restricted globals, immutable-state verification.
"""
from __future__ import annotations
import argparse
from collections import OrderedDict
from enum import IntEnum
import hashlib
import json
import os
from pathlib import Path
import pickletools
import sys
sys.dont_write_bytecode = True
import zipfile
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paths import output_separate, task_path
ARCHIVE = None
SOURCE_TASKS = None
OUTPUT = None
ALLOWED_GLOBALS = {
    "collections OrderedDict", "torch FloatStorage",
    "torch_npu.utils.storage _rebuild_npu_tensor", "torch_npu.npu._format Format",
}

class CPUFormat(IntEnum):
    UNDEFINED=-1
    NCHW=0
    NHWC=1
    ND=2
    NC1HWC0=3
    FRACTAL_Z=4
    NC1HWC0_C04=12
    HWCN=16
    NDHWC=27
    FRACTAL_NZ=29
    NCDHW=30
    NDC1HWC0=32
    FRACTAL_Z_3D=33
    NC=35
    NCL=47
    FRACTAL_NZ_C0_16=50
    FRACTAL_NZ_C0_32=51
    FRACTAL_NZ_C0_2=52
    FRACTAL_NZ_C0_4=53
    FRACTAL_NZ_C0_8=54


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rebuild_cpu_tensor(storage, storage_offset, size, stride, requires_grad,
                       backward_hooks, npu_storage_info):
    require(isinstance(storage, torch.storage.TypedStorage), "Unexpected storage type")
    require(storage.dtype == torch.float32, "Only observed float32 weights are permitted")
    require(storage._untyped_storage.device.type == "cpu", "Storage must already be CPU")
    require(type(storage_offset) is int and storage_offset >= 0, "Invalid offset")
    require(isinstance(size, tuple) and isinstance(stride, tuple) and len(size)==len(stride), "Invalid shape/stride")
    require(len(size) <= 5, "Unexpected tensor rank")
    require(all(type(v) is int and 0 <= v <= 10000000 for v in (*size,*stride)), "Invalid dimensions")
    require(isinstance(requires_grad, bool), "Invalid requires_grad")
    require(isinstance(backward_hooks, OrderedDict) and not backward_hooks, "Unexpected serialized hooks")
    require(isinstance(npu_storage_info, CPUFormat) or type(npu_storage_info) is bool, "Unexpected format")
    n = int(np.prod(size, dtype=np.int64)) if size else 1
    require(n <= 10000000, "Unexpected tensor size")
    last = storage_offset + sum((s-1)*t for s,t in zip(size,stride)) if n else storage_offset
    storage_elements = storage._untyped_storage.nbytes()//4
    require(last < storage_elements if n else last <= storage_elements, "Storage bounds exceeded")
    tensor = torch.empty((0,), dtype=torch.float32, device="cpu")
    tensor.set_(storage, storage_offset, size, stride)
    tensor.requires_grad_(requires_grad)
    return tensor


def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def static_review(path):
    require(path.stat().st_size < 32*1024*1024, "Checkpoint exceeds reviewed size bound")
    with zipfile.ZipFile(path) as z:
        entries=z.infolist()
        require(len(entries)<1000, "Unexpected checkpoint member count")
        require(sum(i.file_size for i in entries)<64*1024*1024, "Unexpected uncompressed checkpoint size")
        names=[i.filename for i in entries if i.filename.endswith("/data.pkl")]
        require(len(names)==1, "Expected exactly one data.pkl")
        data=z.read(names[0])
    operations=list(pickletools.genops(data))
    require(bool(operations) and operations[0][0].name=="PROTO" and operations[0][1]==2, "Only observed protocol 2 is allowed")
    require(not any(op.name in {"STACK_GLOBAL","EXT1","EXT2","EXT4","INST","OBJ","NEWOBJ","NEWOBJ_EX"} for op,_,_ in operations), "Unreviewed pickle construction")
    globals_=sorted({a for op,a,_ in operations if op.name=="GLOBAL"})
    require(set(globals_)==ALLOWED_GLOBALS, f"Unreviewed globals: {globals_}")
    return {"pickle_protocol":2,"pickle_bytes":len(data),"globals":globals_,"source_bytes":path.stat().st_size,"source_sha256":sha256(path)}


def atomic_json(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_bytes(json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2).encode("utf-8"))
    os.replace(tmp,path)


def convert(task_id, freeze):
    task=task_path(SOURCE_TASKS, task_id)
    source=task/"best.pt"
    before=source.stat()
    audit=static_review(source)
    done=json.loads((task/"done.json").read_text(encoding="utf-8"))
    with torch.serialization.safe_globals([
        (rebuild_cpu_tensor, "torch_npu.utils.storage._rebuild_npu_tensor"),
        (CPUFormat, "torch_npu.npu._format.Format"),
    ]):
        checkpoint=torch.load(source,map_location="cpu",weights_only=True)
    require(isinstance(checkpoint,dict) and "model" in checkpoint,"Unexpected checkpoint container")
    state=checkpoint["model"]
    require(isinstance(state,(dict,OrderedDict)) and state,"Missing model state")
    require(all(isinstance(k,str) and isinstance(v,torch.Tensor) and v.device.type=="cpu" for k,v in state.items()),"Unexpected model state entry")
    from tdha_v3.common import array_hash, digest
    from tdha_v3.models import Model, PretrainModel
    tensor_hashes={k:array_hash(v.detach().cpu().numpy()) for k,v in state.items()}
    state_hash=digest(tensor_hashes)
    require(state_hash==done["hashes"]["model"],"Model hash differs from completed training record")
    frozen_hash=freeze.get("models",{}).get(task_id,{}).get("model")
    require(frozen_hash is None or state_hash==frozen_hash,"Model hash differs from frozen model record")
    config=done["task"]
    model=PretrainModel(config["rep"],config["seed"]) if config.get("kind")=="pretrain" or task_id.endswith("-pretrain") else Model(config["net"],config["rep"],config["seed"])
    incompatible=model.load_state_dict(state,strict=True)
    require(not incompatible.missing_keys and not incompatible.unexpected_keys,"State dictionary does not load strictly")
    actual_parameters=sum(p.numel() for p in model.parameters())
    require(actual_parameters==done["parameters"],"Parameter count differs")
    cpu_checkpoint={"model":OrderedDict((k,v.detach().cpu().contiguous()) for k,v in state.items()),"task":config,"frozen_model_hash":state_hash,"conversion":"restricted CPU storage reconstruction; no numerical changes"}
    dest=task_path(OUTPUT/"weights", task_id)/"best.cpu.pt"
    require(not dest.exists(), "Output checkpoint already exists; choose a fresh output directory")
    dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=dest.with_suffix(dest.suffix+".tmp")
    torch.save(cpu_checkpoint,tmp)
    restored=torch.load(tmp,map_location="cpu",weights_only=True)
    require(digest({k:array_hash(v.numpy()) for k,v in restored["model"].items()})==state_hash,"Converted checkpoint round-trip hash mismatch")
    after=source.stat()
    require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),"Source changed while reading")
    os.replace(tmp,dest)
    audit.update({"task":task_id,"state_hash":state_hash,"done_model_hash_match":True,"freeze_model_hash":frozen_hash,"freeze_model_hash_match":True if frozen_hash else None,"strict_load":True,"model_type":type(model).__name__,"parameters":actual_parameters,"tensors":len(state),"destination":str(dest.relative_to(OUTPUT)),"destination_bytes":dest.stat().st_size,"destination_sha256":sha256(dest),"roundtrip_hash_match":True,"tensor_hashes":tensor_hashes,"source_unchanged":True,"source_mtime_ns":before.st_mtime_ns})
    atomic_json(OUTPUT/"audit"/f"weight_conversion_{task_id}.json",audit)
    print(json.dumps({"task":task_id,"status":"verified","state_hash":state_hash,"strict_load":True}),flush=True)
    return audit


def main(argv=None):
    global ARCHIVE, OUTPUT, SOURCE_TASKS
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=Path, help="Original read-only freeze.json and tasks/<id>/best.pt, done.json")
    source.add_argument("--artifact-root", type=Path, help="Selected case with freeze.json and npu-original/<id>/best.pt, done.json")
    parser.add_argument("--output", type=Path, required=True, help="New directory for weights/ and audit/; separate from archive")
    parser.add_argument("--tasks", nargs="+", required=True, help="Explicit task IDs; no implicit full-history conversion")
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Actually convert; default only checks the plan")
    mode.add_argument("--plan-only", action="store_true", help="Read-only plan; also the default")
    args=parser.parse_args(argv)
    ARCHIVE=(args.artifact_root or args.archive).resolve()
    SOURCE_TASKS=ARCHIVE/("npu-original" if args.artifact_root else "tasks")
    OUTPUT=output_separate(args.output, ARCHIVE, ROOT)
    freeze=json.loads((ARCHIVE/"freeze.json").read_text(encoding="utf-8"))
    require(len(args.tasks)==len(set(args.tasks)), "Duplicate task IDs")
    for task in args.tasks:
        entry=task_path(SOURCE_TASKS,task)
        require((entry/"best.pt").is_file() and (entry/"done.json").is_file(), "Missing source checkpoint or done.json")
        require(not (OUTPUT/"weights"/task/"best.cpu.pt").exists(), "Output checkpoint already exists")
    if not args.execute:
        print(json.dumps({"planned_tasks":args.tasks,"writes":False,"training":False}))
        return
    reports=[convert(task,freeze) for task in args.tasks]
    aggregate={"status":"verified","scope":"explicit selected best checkpoints","task_count":len(reports),"torch_version":torch.__version__,"device":"cpu","weights_only":True,"arbitrary_pickle_loading":False,"training":False,"forward_passes":0,"original_archive_writes":False,"official_storage_source":"https://github.com/Ascend/pytorch/blob/v2.7.1/torch_npu/utils/storage.py","tasks":[{k:v for k,v in r.items() if k!="tensor_hashes"} for r in reports]}
    atomic_json(OUTPUT/"audit/weight_conversion.json",aggregate)
    print(json.dumps({"status":"verified","task_count":len(reports)}))


if __name__=="__main__":
    main()
