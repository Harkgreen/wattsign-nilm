"""Read-only loading and prepared-input adapters for selected saved models.

No function fits, calibrates, downloads, rebuilds missing inputs, or writes files.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np

CASES = ("ukdale-10s", "ukdale-1s", "tdha-waveform", "plaid-resistive")


def load_model(case_id: str, checkpoint: str | Path, *, expected_sha256: str):
    """Verify a trusted release hash, then load its selected model on CPU.

    The hash must come from the release manifest, not from an unknown file itself.
    PLAID uses joblib/pickle and therefore requires a trusted release artifact.
    NPU-converted checkpoints use the separate TDHA multilabel entry point.
    """
    if case_id not in CASES:
        raise ValueError(f"Unsupported selected case: {case_id}")
    if len(expected_sha256) != 64 or any(c not in "0123456789abcdefABCDEF" for c in expected_sha256):
        raise ValueError("Supply the checkpoint SHA-256 from the trusted release manifest")
    path = Path(checkpoint)
    with path.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual != expected_sha256.lower():
        raise ValueError("Checkpoint SHA-256 does not match the release manifest")
    if case_id == "plaid-resistive":
        import joblib
        from load_signature_analysis.features import FULL_FEATURES
        payload = joblib.load(path)
        if payload["target"] != "resistive_label" or tuple(payload["features"]) != FULL_FEATURES:
            raise ValueError("PLAID target or ordered feature contract differs from this case")
        return payload["model"]
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if case_id == "ukdale-10s":
        from ._ukdale_10s import RichCausalTCN
        model, state = RichCausalTCN(), payload["state_dict"]
    elif case_id == "ukdale-1s":
        from ._ukdale_1s import M0TCN
        model, state = M0TCN(), payload["state"]
    else:
        from ._tdha_waveform import AggregateModel
        if (payload.get("family"), payload.get("rep"), payload.get("length")) != ("GRU", "W1", 2560):
            raise ValueError("Checkpoint is not the selected GRU/W1/2560 model")
        state = payload["state_dict"]
        model = AggregateModel("GRU", "W1", 2560, state["vscale"].item(), state["iscale"].item())
    model.load_state_dict(state, strict=True)
    return model.eval()


def ukdale_10s_windows(features18, endpoints, normalizer):
    """Apply saved normalization to prepared 10-s features; return (N,18,64).

    The caller supplies only eligible endpoints within continuous records. Feature
    order is P, S, Q, each [value, difference, causal-baseline difference, 3 masks].
    PQ zeroes all six S channels. Raw-data preparation is a separate operation.
    """
    from ._ukdale_10s import norm_apply
    x = _matrix(features18, 18)
    e = _endpoints(endpoints, len(x), 64)
    normalized = norm_apply(x, normalizer)
    normalized[:, 6:12] = 0
    return np.stack([normalized[i-63:i+1].T for i in e]) if len(e) else np.empty((0,18,64), np.float32)


def ukdale_1s_windows(features18, endpoints, normalizer):
    """Return selected F0 input (N,42,640) from prepared native 1-s features.

    Endpoint indices address the 10-s grid; the rightmost second is e*10+9.
    Apply the saved nominal normalizer; S and all 24 optional shape channels
    remain zero. Caller is responsible for eligibility/continuity masks.
    """
    from ._ukdale_1s import apply_normalizer
    x = _matrix(features18, 18)
    e = np.asarray(endpoints)
    if e.ndim != 1 or not np.issubdtype(e.dtype, np.integer):
        raise ValueError("Endpoints must be a one-dimensional integer array")
    seconds = _endpoints(e.astype(np.int64)*10+9, len(x), 640)
    normalized = apply_normalizer(x, normalizer)
    normalized[:,6:12] = 0
    out = np.zeros((len(seconds),42,640),np.float32)
    for j,i in enumerate(seconds):
        out[j,:18] = normalized[i-639:i+1].T
    return out


def tdha_waveform_windows(normalized_base36, cycle_mask, endpoints):
    """Build W1's (N,520,2560) prepared feature input without raw-data reads.

    Base36 must already use the matching fold's saved normalizer. Pass the
    separately read raw ADC cycles as model(x, raw=raw) with shape
    (N,2560,2,1024). No waveform data or feature caches ship in the release.
    """
    x = _matrix(normalized_base36,36)
    mask = np.asarray(cycle_mask)
    if mask.shape != (len(x),) or not np.isin(mask,[0,1]).all():
        raise ValueError("cycle_mask must contain one binary value per cycle")
    e = _endpoints(endpoints,len(x),2560)
    out = np.zeros((len(e),520,2560),np.float32)
    for j,i in enumerate(e):
        out[j,:36] = x[i-2559:i+1].T
        out[j,278:294] = mask[i-2559:i+1]
    return out


def plaid_feature_frame(frame):
    """Select the saved pipeline's ordered columns; leave imputation to it."""
    from load_signature_analysis.features import FULL_FEATURES
    return frame.loc[:,list(FULL_FEATURES)].replace([np.inf,-np.inf],np.nan)


def _matrix(values, columns):
    x=np.asarray(values,dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != columns or not np.isfinite(x).all():
        raise ValueError(f"Expected finite prepared input with {columns} columns")
    return x


def _endpoints(values, count, history):
    e=np.asarray(values)
    if e.ndim != 1 or not np.issubdtype(e.dtype,np.integer):
        raise ValueError("Endpoints must be a one-dimensional integer array")
    if ((e < history-1) | (e >= count)).any():
        raise ValueError("Endpoint lacks the required complete past window")
    return e.astype(np.int64)
