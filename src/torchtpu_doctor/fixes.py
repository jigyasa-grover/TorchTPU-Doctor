from __future__ import annotations

import importlib
import importlib.util
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union


def is_tpu_available() -> bool:
    """Check for native `torch_tpu` via PyTorch's stable `torch.accelerator` API."""
    if importlib.util.find_spec("torch") is None:
        return False
    torch_mod = importlib.import_module("torch")

    if not hasattr(torch_mod, "tpu") and importlib.util.find_spec("torch_tpu") is not None:
        try:
            importlib.import_module("torch_tpu")
        except Exception:
            pass

    accel = getattr(torch_mod, "accelerator", None)
    if accel is not None and hasattr(accel, "is_available") and hasattr(accel, "current_accelerator"):
        try:
            if accel.is_available():
                curr = accel.current_accelerator()
                return curr is not None and getattr(curr, "type", str(curr)) == "tpu"
        except Exception:
            pass
    return False


def get_device() -> Any:
    """Resolve `torch.device` across Cloud TPU (`'tpu'`), CUDA, MPS, and CPU."""
    torch_mod = importlib.import_module("torch")
    if is_tpu_available():
        return torch_mod.device("tpu")

    accel = getattr(torch_mod, "accelerator", None)
    if accel is not None and hasattr(accel, "is_available"):
        try:
            if accel.is_available():
                curr = accel.current_accelerator()
                return torch_mod.device(getattr(curr, "type", str(curr)))
        except Exception:
            pass

    if hasattr(torch_mod, "cuda") and torch_mod.cuda.is_available():
        return torch_mod.device("cuda")
    if hasattr(getattr(torch_mod, "backends", None), "mps") and torch_mod.backends.mps.is_available():
        return torch_mod.device("mps")
    return torch_mod.device("cpu")


def enable_tpu_cache(cache_dir: Union[str, Path] = "/tmp/torchtpu_cache") -> Dict[str, str]:
    """Configure persistent Tier-3 (disk) and Tier-2 (multi-host) XLA compilation caches before TPU init."""
    path = Path(cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_TPU_TIER2_COMPILATION_CACHE"] = "1"
    os.environ["TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT"] = str(path)
    return {
        "TORCH_TPU_TIER2_COMPILATION_CACHE": "1",
        "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT": str(path),
    }


def patch_hf_tpu() -> List[str]:
    """Patch `peft`, `sentence-transformers`, and `transformers.masking_utils` for native `'tpu'` execution."""
    patched: List[str] = []

    def _get_mod(name: str) -> Optional[Any]:
        try:
            return importlib.import_module(name)
        except Exception:
            return None

    peft_other = _get_mod("peft.utils.other")
    if peft_other is not None and hasattr(peft_other, "infer_device"):
        orig = peft_other.infer_device
        if not getattr(orig, "_tpu_patched", False):
            def _infer(*a: Any, **kw: Any) -> str:
                return "tpu" if is_tpu_available() else orig(*a, **kw)

            setattr(_infer, "_tpu_patched", True)
            peft_other.infer_device = _infer
            for alias_mod in ("peft", "peft.utils"):
                m = _get_mod(alias_mod)
                if m is not None and hasattr(m, "infer_device"):
                    m.infer_device = _infer
            patched.append("peft.utils.other.infer_device")

    st_env = _get_mod("sentence_transformers.util.environment")
    if st_env is not None and hasattr(st_env, "get_device_name"):
        orig_st = st_env.get_device_name
        if not getattr(orig_st, "_tpu_patched", False):
            def _dev_name(*a: Any, **kw: Any) -> str:
                if is_tpu_available():
                    rank = os.environ.get("LOCAL_RANK")
                    return f"tpu:{rank}" if rank is not None else "tpu"
                return orig_st(*a, **kw)

            setattr(_dev_name, "_tpu_patched", True)
            st_env.get_device_name = _dev_name
            st_util = _get_mod("sentence_transformers.util")
            if st_util is not None and hasattr(st_util, "get_device_name"):
                st_util.get_device_name = _dev_name
            patched.append("sentence_transformers.util.environment.get_device_name")

    tf_mask = _get_mod("transformers.masking_utils")
    if tf_mask is not None and hasattr(tf_mask, "_ignore_causal_mask_sdpa"):
        orig_ignore = tf_mask._ignore_causal_mask_sdpa
        if not getattr(orig_ignore, "_tpu_patched", False):
            def _no_sync_ignore(padding_mask: Any, *a: Any, **kw: Any) -> bool:
                # Avoid synchronous fast_all(padding_mask) TPU->CPU bool readbacks on every forward pass
                if padding_mask is not None and getattr(getattr(padding_mask, "device", None), "type", None) == "tpu":
                    return False
                return bool(orig_ignore(padding_mask, *a, **kw))

            setattr(_no_sync_ignore, "_tpu_patched", True)
            tf_mask._ignore_causal_mask_sdpa = _no_sync_ignore
            patched.append("transformers.masking_utils._ignore_causal_mask_sdpa")

    return patched


def pad_to_multiple(tensor: Any, multiple: int = 128, value: Union[int, float] = 0, dim: int = -1) -> Any:
    """Pad `tensor` along `dim` to a multiple of `multiple` (128 matches TPU MXU tile lanes)."""
    import torch.nn.functional as F

    if multiple <= 1 or tensor.dim() == 0:
        return tensor
    d = dim if dim >= 0 else tensor.dim() + dim
    rem = int(tensor.shape[d]) % multiple
    if rem == 0:
        return tensor
    pad = [0] * (2 * (tensor.dim() - d))
    pad[-1] = multiple - rem
    return F.pad(tensor, tuple(pad), value=value)


class TileAlignedCollator:
    """Collator that buckets sequence lengths to multiples of 128 to avoid XLA recompiles."""

    def __init__(
        self,
        pad_to_multiple_of: int = 128,
        buckets: Sequence[int] = (128, 256, 512, 1024, 2048),
        pad_token_id: int = 0,
        label_pad_id: int = -100,
        pad_batch_to_multiple_of: Optional[int] = None,
    ) -> None:
        self.multiple = pad_to_multiple_of
        self.buckets = tuple(sorted(buckets))
        self.pad_token_id = pad_token_id
        self.label_pad_id = label_pad_id
        self.batch_multiple = pad_batch_to_multiple_of

    def _bucket_len(self, max_len: int) -> int:
        for b in self.buckets:
            if max_len <= b and b % self.multiple == 0:
                return b
        return max(self.multiple, int(math.ceil(max_len / self.multiple) * self.multiple))

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        import torch
        import torch.nn.functional as F

        if not features:
            return {}

        bsz = len(features)
        target_bsz = bsz
        if self.batch_multiple and bsz % self.batch_multiple != 0:
            target_bsz = bsz + (self.batch_multiple - (bsz % self.batch_multiple))

        seq_keys: List[str] = []
        scalar_keys: List[str] = []
        max_len = 0

        for k, sample in features[0].items():
            if isinstance(sample, (list, tuple)):
                seq_keys.append(k)
                max_len = max(max_len, max(len(f[k]) for f in features))
            elif isinstance(sample, torch.Tensor) and sample.dim() >= 1:
                seq_keys.append(k)
                max_len = max(max_len, max(int(f[k].shape[-1]) for f in features))
            else:
                scalar_keys.append(k)

        target_len = self._bucket_len(max_len) if seq_keys else 0
        batch: Dict[str, Any] = {}

        for k in seq_keys:
            fill = self.label_pad_id if k == "labels" else (0 if "mask" in k else self.pad_token_id)
            first = features[0][k]
            dtype = first.dtype if isinstance(first, torch.Tensor) else torch.long
            rows = []
            for f in features:
                t = f[k].detach() if isinstance(f[k], torch.Tensor) else torch.tensor(f[k], dtype=dtype)
                diff = target_len - int(t.shape[-1])
                rows.append(F.pad(t, (0, diff), value=fill) if diff > 0 else t)
            for _ in range(target_bsz - bsz):
                pad_row = torch.full((target_len,), fill, dtype=dtype)
                if "mask" in k and target_len > 0:
                    # Keep token 0 unmasked on dummy rows so SDPA softmax backward stays finite
                    pad_row[0] = 1
                rows.append(pad_row)
            batch[k] = torch.stack(rows, dim=0)

        for k in scalar_keys:
            vals = [f[k] for f in features]
            stacked = torch.stack(vals) if isinstance(vals[0], torch.Tensor) else torch.tensor(vals)
            if target_bsz > bsz:
                fill_s = self.label_pad_id if k in ("label", "labels") else 0
                extra = torch.full((target_bsz - bsz, *stacked.shape[1:]), fill_s, dtype=stacked.dtype)
                stacked = torch.cat([stacked, extra], dim=0)
            batch[k] = stacked

        return batch
