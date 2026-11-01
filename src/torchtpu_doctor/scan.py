from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

_SYNC_CALLS = {"item", "cpu", "numpy", "tolist"}
_INTERNAL_TPU_HOOKS = {"is_available", "device_count", "synchronize"}
_SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "build", "dist", ".pytest_cache"}


@dataclass
class Issue:
    file: str
    line: int
    code: str
    message: str
    hint: str

    def format_line(self) -> str:
        return f"{self.file}:{self.line}: {self.code} {self.message} -> {self.hint}"

    def to_dict(self) -> dict:
        return asdict(self)


def _dotted(node: ast.AST) -> Optional[str]:
    parts: List[str] = []
    curr: Optional[ast.AST] = node
    while isinstance(curr, ast.Attribute):
        parts.append(curr.attr)
        curr = curr.value
    if isinstance(curr, ast.Name):
        parts.append(curr.id)
        return ".".join(reversed(parts))
    return None


def _is_slice_subscript(slice_node: ast.AST) -> bool:
    if isinstance(slice_node, ast.Slice):
        return True
    if isinstance(slice_node, ast.Tuple):
        return any(isinstance(e, ast.Slice) for e in slice_node.elts) or len(slice_node.elts) >= 2
    return False


class _Visitor(ast.NodeVisitor):
    def __init__(self, filename: str, lines: Sequence[str]) -> None:
        self.filename = filename
        self.lines = lines
        self.issues: List[Issue] = []
        self._loop_depth = 0
        self._in_forward = False

        text = "\n".join(lines)
        self._has_tpu_check = (
            "torch.accelerator" in text or '"tpu"' in text or "'tpu'" in text or "get_device" in text
        )
        self._has_hf_patch = "patch_hf_tpu" in text

    def _add(self, node: ast.AST, code: str, message: str, hint: str) -> None:
        lineno = getattr(node, "lineno", 1)
        if 1 <= lineno <= len(self.lines):
            raw = self.lines[lineno - 1]
            if f"noqa: {code}" in raw or "torchtpu-doctor: ignore" in raw:
                return
        self.issues.append(Issue(self.filename, lineno, code, message, hint))

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        prev = self._in_forward
        if node.name == "forward":
            self._in_forward = True
        self.generic_visit(node)
        self._in_forward = prev

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_For(self, node: ast.For) -> None:
        self._loop_depth += 1
        self.generic_visit(node)
        self._loop_depth -= 1

    visit_AsyncFor = visit_For

    def visit_While(self, node: ast.While) -> None:
        self._loop_depth += 1
        self.generic_visit(node)
        self._loop_depth -= 1

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "torch_xla" or alias.name.startswith("torch_xla."):
                self._add(
                    node,
                    "TPU006",
                    f"legacy `{alias.name}` import",
                    "native torch_tpu uses `device='tpu'` and `torch.compile(backend='tpu')` directly",
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        if mod == "torch_xla" or mod.startswith("torch_xla."):
            self._add(
                node,
                "TPU006",
                f"legacy `{mod}` import",
                "drop xm.xla_device()/xm.mark_step() when running on native torch_tpu",
            )
        if not self._has_hf_patch:
            for alias in node.names:
                if alias.name in ("infer_device", "get_device_name") and (
                    "peft" in mod or "sentence_transformers" in mod
                ):
                    self._add(
                        node,
                        "TPU001",
                        f"`{mod}.{alias.name}()` omits 'tpu' and falls back to 'cpu'",
                        "call `patch_hf_tpu()` at startup or use `get_device()`",
                    )
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if self._in_forward or self._loop_depth > 0:
            for target in node.targets:
                if isinstance(target, ast.Subscript) and _is_slice_subscript(target.slice):
                    self._add(
                        target,
                        "TPU004",
                        "in-place slice write inside loop/forward can break XLA buffer donation",
                        "prefer out-of-place `torch.where` / `torch.cat` under `torch.compile(backend='tpu')`",
                    )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if _dotted(node) in ("torch.float64", "torch.double"):
            self._add(
                node,
                "TPU005",
                "float64 is not accelerated on TPU MXU",
                "use bfloat16 or float32 (with `torch.tpu.Precision.HIGHEST` if needed)",
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        dotted = _dotted(node.func)
        method = node.func.attr if isinstance(node.func, ast.Attribute) else None

        if dotted in {f"torch.tpu.{m}" for m in _INTERNAL_TPU_HOOKS}:
            repl = dotted.replace("torch.tpu.", "torch.accelerator.")
            self._add(
                node,
                "TPU001",
                f"`{dotted}()` is an internal PrivateUse1 hook in torch_tpu",
                f"use `{repl}()` instead",
            )
        elif method == "cuda" and len(node.args) <= 1:
            self._add(node, "TPU001", "hardcoded `.cuda()` call", "use `.to(device)` via `get_device()`")
        elif dotted == "torch.cuda.is_available" and not self._has_tpu_check:
            self._add(
                node,
                "TPU001",
                "`torch.cuda.is_available()` without TPU check falls back to CPU on Cloud TPU",
                "use `torch.accelerator.is_available()` or `get_device()`",
            )

        if dotted in ("torch.device", "dist.init_process_group", "torch.distributed.init_process_group") or method == "to":
            for arg in list(node.args) + [kw.value for kw in node.keywords]:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    val = arg.value.lower()
                    if val == "cuda" or val.startswith("cuda:") or val == "nccl":
                        self._add(
                            node,
                            "TPU001",
                            f"hardcoded `'{arg.value}'` literal",
                            "resolve device/backend dynamically so 'tpu' works without edits",
                        )

        if (self._loop_depth > 0 or self._in_forward) and (
            method in _SYNC_CALLS or dotted in ("torch.cuda.synchronize", "torch.accelerator.synchronize")
        ):
            label = f".{method}()" if method in _SYNC_CALLS else f"{dotted}()"
            self._add(
                node,
                "TPU002",
                f"`{label}` in hot path flushes TorchTPU deferred fusion queue",
                "accumulate tensors on device and read `.item()` only on log steps",
            )

        if (self._in_forward or self._loop_depth > 0) and (
            dotted in ("torch.nonzero", "torch.unique", "torch.masked_select")
            or method in ("nonzero", "unique", "masked_select")
        ):
            self._add(
                node,
                "TPU003",
                f"data-dependent shape op `{dotted or method}` forces host sync / recompile",
                "use fixed-shape `torch.where` masking when possible",
            )

        kw_map = {kw.arg: kw.value for kw in node.keywords if kw.arg}
        if "padding" in kw_map and "pad_to_multiple_of" not in kw_map:
            pval = kw_map["padding"]
            if isinstance(pval, ast.Constant) and pval.value in (True, "longest"):
                self._add(
                    node,
                    "TPU003",
                    "`padding=True` without `pad_to_multiple_of=128` causes per-batch XLA recompiles",
                    "pass `pad_to_multiple_of=128` or use `TileAlignedCollator`",
                )

        if method == "double" and not node.args:
            self._add(node, "TPU005", "`.double()` casts to float64", "use bfloat16 or float32 on TPU")
        elif method == "as_strided" or dotted == "torch.as_strided":
            self._add(
                node,
                "TPU005",
                "`as_strided` view can force HBM relayout across (8, 128) TPU tiles",
                "prefer reshape/view or call `.contiguous()` once outside the loop",
            )

        self.generic_visit(node)


def scan_source(source: str, filename: str = "<string>") -> List[Issue]:
    lines = source.splitlines()
    tree = ast.parse(source, filename=filename)
    visitor = _Visitor(filename, lines)
    visitor.visit(tree)
    return visitor.issues


def scan_path(path: Union[str, Path]) -> List[Issue]:
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"path not found: {root}")

    files = [root] if root.is_file() else [
        p for p in sorted(root.rglob("*.py")) if not any(part in _SKIP_DIRS for part in p.parts)
    ]
    base = root.parent if root.is_file() else root
    found: List[Issue] = []
    for py in files:
        try:
            src = py.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        rel = str(py.relative_to(base)) if py.is_relative_to(base) else str(py)
        found.extend(scan_source(src, filename=rel))
    return found


def format_issues(issues: Iterable[Issue]) -> str:
    items = list(issues)
    if not items:
        return "No TorchTPU issues found."
    lines = [item.format_line() for item in items]
    lines.append(f"\nFound {len(items)} potential TorchTPU issue(s).")
    return "\n".join(lines)
