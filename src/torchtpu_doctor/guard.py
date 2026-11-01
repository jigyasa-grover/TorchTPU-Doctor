from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import sys
import time
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

_MATMUL_OPS = {"aten::mm", "aten::addmm", "aten::bmm", "aten::linear"}
_SHAPE_OPS = _MATMUL_OPS | {
    "aten::scaled_dot_product_attention",
    "aten::_scaled_dot_product_flash_attention",
    "aten::native_layer_norm",
    "aten::embedding",
}
_VIEW_OPS = {
    "aten::view",
    "aten::_unsafe_view",
    "aten::reshape",
    "aten::t",
    "aten::transpose",
    "aten::permute",
    "aten::expand",
    "aten::slice",
    "aten::select",
    "aten::unsqueeze",
    "aten::squeeze",
    "aten::as_strided",
    "aten::alias",
    "aten::detach",
}
_SYNC_OPS = {"aten::_local_scalar_dense", "aten::item", "aten::nonzero", "aten::masked_select"}


@dataclass
class DispatchSummary:
    label: str = "session"
    steps: int = 0
    aten_ops: int = 0
    host_syncs: int = 0
    slice_writes: int = 0
    wasted_copy_bytes: int = 0
    strided_views: int = 0
    float64_ops: int = 0
    matmuls: int = 0
    tile_aligned_matmuls: int = 0
    mxu_lane_fill_pct: float = 100.0
    dynamic_ops: Dict[str, List[str]] = field(default_factory=dict)
    elapsed_ms: float = 0.0

    @property
    def has_bottlenecks(self) -> bool:
        return bool(
            self.host_syncs
            or self.slice_writes
            or self.strided_views
            or self.float64_ops
            or self.dynamic_ops
            or (self.matmuls > 0 and self.mxu_lane_fill_pct < 75.0)
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def format_text(self) -> str:
        kb = self.wasted_copy_bytes / 1024.0
        lines = [
            f"torchtpu-doctor [{self.label}] ({self.steps} steps, {self.aten_ops} ops, {self.elapsed_ms:.1f} ms)",
            f"  host syncs (.item/.cpu) : {self.host_syncs}",
            f"  dynamic shape ops       : {len(self.dynamic_ops)}",
            f"  partial-slice writes    : {self.slice_writes}" + (f" (~{kb:.0f} KB storage copy risk)" if kb >= 1.0 else ""),
            f"  strided 3D+ view ops    : {self.strided_views}",
            f"  128-aligned matmuls     : {self.tile_aligned_matmuls}/{self.matmuls} ({self.mxu_lane_fill_pct:.1f}% MXU lane fill)",
        ]
        if self.float64_ops:
            lines.append(f"  float64 ops             : {self.float64_ops}")
        if not self.has_bottlenecks:
            lines.append("  status: clean (ready for torch.compile(backend='tpu'))")
        return "\n".join(lines)


def _walk_tensors(obj: Any, tensor_cls: type) -> Iterable[Any]:
    if isinstance(obj, tensor_cls):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            yield from _walk_tensors(x, tensor_cls)
    elif isinstance(obj, dict):
        for x in obj.values():
            yield from _walk_tensors(x, tensor_cls)


def _inside_torch_optim() -> bool:
    try:
        frame = sys._getframe(2)
        for _ in range(10):
            if frame is None:
                break
            if "/torch/optim/" in frame.f_code.co_filename.replace("\\", "/"):
                return True
            frame = frame.f_back
    except Exception:
        pass
    return False


def _mutates_arg0(func: Any, op_name: str) -> bool:
    schema = getattr(func, "_schema", None)
    if schema is not None and getattr(schema, "arguments", None):
        alias = getattr(schema.arguments[0], "alias_info", None)
        if alias is not None and getattr(alias, "is_write", False):
            return True
    short = op_name.split("::")[-1]
    return short.endswith("_") and not short.startswith("__")


class tpu_doctor:
    """`TorchDispatchMode` context manager that spots TPU syncs, recompiles, and tile waste."""

    def __init__(self, label: str = "session", tile_multiple: int = 128, shape_threshold: int = 3) -> None:
        self.tile_multiple = tile_multiple
        self.shape_threshold = shape_threshold
        self.result = DispatchSummary(label=label)

        self._mode: Any = None
        self._t0: int = 0
        self._steps = 0
        self._shapes: Dict[str, Set[str]] = {}
        self._cur_step_shapes: Dict[str, Set[str]] = {}
        self._step_profiles: Dict[str, Set[FrozenSet[str]]] = {}
        self._weighted_eff_sum = 0.0
        self._total_flops = 0.0
        self._busy = False

    def _flush_step_shapes(self) -> None:
        for op, sigs in self._cur_step_shapes.items():
            self._step_profiles.setdefault(op, set()).add(frozenset(sigs))
        self._cur_step_shapes.clear()

    def step(self) -> None:
        self._steps += 1
        self._flush_step_shapes()

    def __enter__(self) -> "tpu_doctor":
        import torch
        from torch.utils._python_dispatch import TorchDispatchMode

        parent = self
        self._t0 = time.perf_counter_ns()

        class _Mode(TorchDispatchMode):
            def __torch_dispatch__(self, func: Any, types: Any, args: Tuple[Any, ...] = (), kwargs: Optional[Dict[str, Any]] = None) -> Any:
                if kwargs is None:
                    kwargs = {}
                if not parent._busy:
                    parent._busy = True
                    try:
                        parent._record(torch, func, args, kwargs)
                    finally:
                        parent._busy = False
                return func(*args, **kwargs)

        self._mode = _Mode()
        self._mode.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._mode is not None:
            self._mode.__exit__(exc_type, exc_val, exc_tb)
            self._mode = None

        if self._cur_step_shapes:
            self._flush_step_shapes()

        self.result.elapsed_ms = round((time.perf_counter_ns() - self._t0) / 1e6, 2)
        self.result.steps = max(1, self._steps)
        if self._total_flops > 0:
            self.result.mxu_lane_fill_pct = round(100.0 * self._weighted_eff_sum / self._total_flops, 1)

        if self._steps >= self.shape_threshold:
            for op, profiles in sorted(self._step_profiles.items()):
                if len(profiles) >= self.shape_threshold:
                    self.result.dynamic_ops[op] = sorted(self._shapes.get(op, set()))[:5]
        else:
            for op, sigs in sorted(self._shapes.items()):
                if len(sigs) >= self.shape_threshold:
                    self.result.dynamic_ops[op] = sorted(sigs)[:5]

    def _record(self, torch_mod: Any, func: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
        self.result.aten_ops += 1
        op = getattr(getattr(func, "_schema", None), "name", str(func)).split(".")[0]

        tensors = list(_walk_tensors(args, torch_mod.Tensor))
        if kwargs:
            tensors.extend(_walk_tensors(kwargs, torch_mod.Tensor))

        if op in _SYNC_OPS:
            cpu_only = bool(tensors) and all(t.device.type == "cpu" for t in tensors)
            if not (cpu_only and _inside_torch_optim()):
                self.result.host_syncs += 1
        elif op == "aten::_to_copy" and tensors:
            dev = kwargs.get("device")
            if dev is not None and str(dev).startswith("cpu") and tensors[0].device.type != "cpu":
                self.result.host_syncs += 1

        if not tensors:
            return

        if any(t.dtype == torch_mod.float64 for t in tensors):
            self.result.float64_ops += 1

        if args and isinstance(args[0], torch_mod.Tensor) and _mutates_arg0(func, op):
            dst = args[0]
            if dst._is_view():
                storage_b = int(dst.untyped_storage().nbytes())
                view_b = int(dst.numel() * dst.element_size())
                if storage_b > view_b > 0:
                    self.result.slice_writes += 1
                    self.result.wasted_copy_bytes += storage_b

        if op not in _VIEW_OPS and op != "aten::copy_":
            for t in tensors:
                if t.numel() <= 1 or t.is_contiguous():
                    continue
                strides = t.stride()
                if 0 in strides:
                    continue  # Broadcast view -> stablehlo.broadcast_in_dim (no copy)
                if op in ("aten::mm", "aten::addmm", "aten::linear") and t.dim() == 2 and 1 in strides:
                    continue  # 2D weight transpose handled natively on MXU
                if strides[-1] == 1:
                    continue  # Unit-strided minor lane (e.g. (B, H, S, D) multi-head attention)
                self.result.strided_views += 1
                break

        if op in _SHAPE_OPS:
            sig = ", ".join(str(tuple(int(d) for d in t.shape)) for t in tensors[:2])
            self._shapes.setdefault(op, set()).add(sig)
            self._cur_step_shapes.setdefault(op, set()).add(sig)

        if op in _MATMUL_OPS:
            m, k, n = _matmul_dims(op, tensors)
            if m > 0 and k > 0 and n > 0:
                self.result.matmuls += 1
                tile = self.tile_multiple
                m_fill = m / (math.ceil(m / 8) * 8)
                k_fill = k / (math.ceil(k / tile) * tile)
                n_fill = n / (math.ceil(n / tile) * tile)
                eff = min(1.0, math.sqrt(k_fill * n_fill) * (0.5 + 0.5 * m_fill))
                flops = 2.0 * m * k * n
                self._weighted_eff_sum += flops * eff
                self._total_flops += flops
                if m % 8 == 0 and k % tile == 0 and n % tile == 0:
                    self.result.tile_aligned_matmuls += 1


def _matmul_dims(op: str, tensors: List[Any]) -> Tuple[int, int, int]:
    try:
        if op == "aten::mm" and len(tensors) >= 2:
            return int(tensors[0].shape[0]), int(tensors[0].shape[1]), int(tensors[1].shape[1])
        if op == "aten::addmm" and len(tensors) >= 3:
            return int(tensors[1].shape[0]), int(tensors[1].shape[1]), int(tensors[2].shape[1])
        if op == "aten::bmm" and len(tensors) >= 2:
            a, b = tensors[0], tensors[1]
            return int(a.shape[0] * a.shape[1]), int(a.shape[2]), int(b.shape[2])
        if op == "aten::linear" and len(tensors) >= 2:
            inp, w = tensors[0], tensors[1]
            m = 1
            for d in inp.shape[:-1]:
                m *= int(d)
            return m, int(inp.shape[-1]), int(w.shape[0])
    except Exception:
        pass
    return 0, 0, 0
