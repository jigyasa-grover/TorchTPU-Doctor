"""TPU-ready Hugging Face + PEFT LoRA fine-tuning script for Qwen2.5-0.5B-Instruct."""

from __future__ import annotations

import time
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from torchtpu_doctor import (
    TileAlignedCollator,
    enable_tpu_cache,
    get_device,
    patch_hf_tpu,
    tpu_doctor,
)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

SFT_BATCHES = [
    [
        "User: How do you check if a TPU is active in native PyTorch?\nAssistant: Query torch.accelerator.current_accelerator() rather than internal hooks.",
        "User: What backend string does torch.compile take on TorchTPU?\nAssistant: Pass backend='tpu' with static shapes.",
        "User: Does TorchTPU need xm.mark_step()?\nAssistant: No, TorchTPU dispatches eagerly through PrivateUse1 without torch_xla.",
    ],
    [
        "User: Explain what XLA buffer donation does during an optimizer step.\nAssistant: Buffer donation lets XLA alias an input HBM allocation directly to an updated parameter output when no live references remain, avoiding an extra full-tensor copy.",
        "User: Why avoid float64 tensors on Cloud TPUs?\nAssistant: TPU MXU tiles natively run bfloat16 and float32, so float64 falls back to much slower emulation paths.",
        "User: When should you read training loss back to Python?\nAssistant: Accumulate detached scalar losses on device and call .item() only every N logging steps.",
    ],
    [
        "User: Why does tokenizer(batch, padding=True) cause latency spikes on Cloud TPUs?\nAssistant: Dynamic batch padding sets the sequence dimension to whatever example happens to be longest in that mini-batch. Every new (batch, seq_len) pair misses the XLA executable cache and forces a fresh StableHLO compilation.",
        "User: What is a clean fix for variable-length SFT datasets on TPUs?\nAssistant: Bucket sequence lengths to fixed multiples of 128 such as 128, 256, or 512 so XLA compiles each bucket once during warmup.",
        "User: Does padding labels with -100 change the cross-entropy gradient?\nAssistant: No, PyTorch CrossEntropyLoss ignores index -100 by default.",
    ],
    [
        "User: Walk through what happens under the hood when loss.item() runs inside every training step on TorchTPU.\nAssistant: TorchTPU queues tensor ops asynchronously so adjacent elementwise kernels can fuse before hitting HBM. Calling .item() forces an immediate device-to-host transfer, flushing pending work and blocking Python until the accelerator finishes.",
        "User: Why do partial slice assignments like history[step:step+1] = loss hurt XLA graphs?\nAssistant: Functionalization lowers slice writes into dynamic-update-slice ops over the full backing buffer, which can copy the whole tensor when buffer donation cannot prove uniqueness.",
        "User: Why is 128 the magic tile multiple on TPU v5e and v6e?\nAssistant: Matrix Multiply Units process 128-wide systolic lanes, so odd dimensions leave lanes idle.",
    ],
]


def tokenize_examples(tokenizer: AutoTokenizer, texts: list[str]) -> list[dict[str, list[int]]]:
    items = []
    for text in texts:
        ids = tokenizer.encode(text, truncation=True, max_length=256)
        items.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": list(ids)})
    return items


def main() -> None:
    enable_tpu_cache("/tmp/torchtpu_cache")
    patch_hf_tpu()  # Patches PEFT infer_device, SentenceTransformers, and Transformers SDPA mask syncs
    device = get_device()
    dtype = torch.bfloat16 if device.type in ("tpu", "cuda") else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype)
    base_model.config.use_cache = False
    lora_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(base_model, lora_cfg).to(device)

    collator = TileAlignedCollator(
        pad_to_multiple_of=128,
        buckets=(128, 256),
        pad_token_id=tokenizer.pad_token_id,
        pad_batch_to_multiple_of=4,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    # 1. Audit 4 steps under TorchDispatchMode before compiling
    aligned_batches = [
        {k: v.to(device) for k, v in collator(tokenize_examples(tokenizer, texts)).items()}
        for texts in SFT_BATCHES
    ]

    with tpu_doctor("lora_audit") as doc:
        for batch in aligned_batches:
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            doc.step()
    print(doc.result.format_text())

    # 2. Steady-state fine-tuning with static 128-tile buckets and deferred loss readback
    run_model = torch.compile(model, backend="tpu", dynamic=False) if device.type == "tpu" else model
    seen_shapes = set()
    for wb in aligned_batches:
        shape = tuple(wb["input_ids"].shape)
        if shape not in seen_shapes:
            seen_shapes.add(shape)
            optimizer.zero_grad(set_to_none=True)
            warmup_loss = run_model(**wb).loss
            warmup_loss.backward()
            optimizer.step()
    if hasattr(torch, "accelerator") and torch.accelerator.is_available():
        torch.accelerator.synchronize()

    loss_sum = torch.zeros((), device=device, dtype=torch.float32)
    steps = 8

    t0 = time.perf_counter()
    for step in range(steps):
        batch = aligned_batches[step % len(aligned_batches)]
        optimizer.zero_grad(set_to_none=True)
        loss = run_model(**batch).loss
        loss.backward()
        optimizer.step()
        loss_sum = loss_sum + loss.detach().float()

    if hasattr(torch, "accelerator") and torch.accelerator.is_available():
        torch.accelerator.synchronize()
    elapsed = time.perf_counter() - t0
    print(f"8 steps finished in {elapsed:.2f}s ({elapsed * 1000 / steps:.1f} ms/step) | mean_loss={(loss_sum / steps).item():.4f}")


if __name__ == "__main__":
    main()
