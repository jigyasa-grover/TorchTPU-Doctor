from __future__ import annotations

import json
from pathlib import Path
import types

import torch

from torchtpu_doctor import (
    TileAlignedCollator,
    enable_tpu_cache,
    get_device,
    is_tpu_available,
    pad_to_multiple,
    patch_hf_tpu,
)
from torchtpu_doctor.cli import main


def test_tpu_device_and_hf_patch(monkeypatch) -> None:
    import sys

    fake_accel = types.SimpleNamespace(
        is_available=lambda: True,
        current_accelerator=lambda: types.SimpleNamespace(type="tpu"),
    )
    monkeypatch.setattr(torch, "accelerator", fake_accel, raising=False)
    monkeypatch.setattr(torch, "device", lambda name: f"dev:{name}")

    assert is_tpu_available() is True
    assert get_device() == "dev:tpu"

    fake_peft = types.SimpleNamespace(infer_device=lambda: "cpu")
    fake_st = types.SimpleNamespace(get_device_name=lambda: "cpu")
    fake_mask = types.SimpleNamespace(_ignore_causal_mask_sdpa=lambda pm, *a, **kw: True)
    monkeypatch.setitem(sys.modules, "peft", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "peft.utils", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "peft.utils.other", fake_peft)
    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "sentence_transformers.util", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "sentence_transformers.util.environment", fake_st)
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "transformers.masking_utils", fake_mask)

    patched = patch_hf_tpu()
    assert len(patched) == 3
    assert fake_peft.infer_device() == "tpu"
    assert fake_st.get_device_name() == "tpu"
    tpu_mask = types.SimpleNamespace(device=types.SimpleNamespace(type="tpu"))
    cpu_mask = types.SimpleNamespace(device=types.SimpleNamespace(type="cpu"))
    assert fake_mask._ignore_causal_mask_sdpa(tpu_mask, 128, 128, 0, 0) is False
    assert fake_mask._ignore_causal_mask_sdpa(cpu_mask, 128, 128, 0, 0) is True


def test_collator_and_pad_helper(tmp_path: Path) -> None:
    cfg = enable_tpu_cache(tmp_path / "cache")
    assert cfg["TORCH_TPU_TIER2_COMPILATION_CACHE"] == "1"

    t = torch.ones((3, 190))
    assert pad_to_multiple(t, 128).shape == (3, 256)

    collator = TileAlignedCollator(pad_to_multiple_of=128, buckets=(128, 256), pad_batch_to_multiple_of=8)
    batch = collator([{"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [1, 2, 3]} for _ in range(5)])
    assert batch["input_ids"].shape == (8, 128)
    assert (batch["labels"][5:] == -100).all().item() is True
    assert (batch["attention_mask"][5:, 0] == 1).all().item() is True
    assert (batch["attention_mask"][5:, 1:] == 0).all().item() is True


def test_cli_and_notebook_validity(capsys) -> None:
    root = Path(__file__).resolve().parents[1]
    assert main(["scan", str(root / "examples" / "naive_lora_sft.py"), "--strict"]) == 1
    assert main(["scan", str(root / "examples" / "tpu_lora_sft.py"), "--strict"]) == 0
    assert main(["demo", "--steps", "3", "--json"]) == 0
    _ = capsys.readouterr()

    nb = json.loads((root / "notebooks" / "torchtpu_doctor_demo.ipynb").read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "<nb>", "exec")
