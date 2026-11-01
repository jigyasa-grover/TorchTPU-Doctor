from __future__ import annotations

from pathlib import Path

from torchtpu_doctor import scan_path, scan_source


def test_cuda_and_internal_tpu_checks() -> None:
    src = """
import torch

def setup(m):
    if torch.tpu.is_available():
        pass
    if torch.cuda.is_available():
        return m.cuda()
    return m.to("cuda:0")
"""
    issues = scan_source(src, filename="check.py")
    codes = [i.code for i in issues]
    assert codes.count("TPU001") == 4


def test_sync_and_slice_and_padding_checks() -> None:
    src = """
import torch

def tok_fn(tok, texts):
    return tok(texts, padding=True)

def loop(xs, buf):
    metrics = {}
    for x in xs:
        metrics["n"] = 1
        buf[:, :8] = x[:, :8]
        print(x.sum().item())
"""
    issues = scan_source(src, filename="train.py")
    codes = [i.code for i in issues]
    assert "TPU002" in codes
    assert "TPU003" in codes
    assert codes.count("TPU004") == 1


def test_inline_suppression_and_example_files() -> None:
    src = """
for x in [1]:
    x.item()  # noqa: TPU002
"""
    assert scan_source(src) == []

    root = Path(__file__).resolve().parents[1]
    assert len(scan_path(root / "examples" / "naive_lora_sft.py")) >= 4
    assert scan_path(root / "examples" / "tpu_lora_sft.py") == []
