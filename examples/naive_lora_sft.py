"""Standard GPU-style Hugging Face + PEFT LoRA fine-tuning script.

Runs without raising any exceptions on a Cloud TPU VM, but hits several silent
performance traps. Check it with:
    torchtpu-doctor scan examples/naive_lora_sft.py
"""

from __future__ import annotations

import torch
from peft import LoraConfig, get_peft_model
from peft.utils.other import infer_device
from transformers import AutoModelForCausalLM, AutoTokenizer

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


def make_batch(tokenizer: AutoTokenizer, texts: list[str], device: str) -> dict[str, torch.Tensor]:
    # Pads each batch only to that batch's longest string -> new shape signature every step
    enc = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    return {
        "input_ids": enc["input_ids"].to(device),
        "attention_mask": enc["attention_mask"].to(device),
        "labels": labels.to(device),
    }


def main() -> None:
    # infer_device() checks cuda/mps/mlu/xpu/npu and returns 'cpu' on TorchTPU
    device = infer_device()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    lora_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(base_model, lora_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    loss_history = torch.zeros(16, device=device)

    for step in range(8):
        texts = SFT_BATCHES[step % len(SFT_BATCHES)]
        batch = make_batch(tokenizer, texts, device)

        optimizer.zero_grad()
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()

        loss_history[step : step + 1] = loss.detach()
        print(f"step={step} shape={tuple(batch['input_ids'].shape)} loss={loss.item():.4f}")


if __name__ == "__main__":
    main()
