from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

from .scan import format_issues, scan_path


def run_demo(steps: int = 5) -> tuple:
    import torch
    import torch.nn as nn

    from .fixes import TileAlignedCollator, get_device
    from .guard import tpu_doctor

    torch.manual_seed(42)
    device = get_device()
    lengths = [73, 119, 141, 187, 211, 95]

    # 1. Naive GPU-style loop: odd dims, variable seq_len, slice writes, per-step .item()
    naive_proj = nn.Linear(190, 190, bias=False).to(device)
    scratch = torch.zeros(7, 256, 190, device=device)

    with tpu_doctor(label="before:naive_loop") as before:
        for i in range(steps):
            seq_len = lengths[i % len(lengths)]
            x = torch.randn(7, seq_len, 190, device=device)
            y = x.transpose(1, 2).relu().transpose(1, 2)
            out = naive_proj(y.reshape(-1, 190)).reshape(7, seq_len, 190)
            scratch[:, :seq_len, :] = out.detach()  # noqa: TPU004
            _ = float((out**2).mean().item())  # noqa: TPU002
            before.step()

    # 2. TPU-friendly loop: 128-aligned shapes, out-of-place ops, single host read after loop
    tpu_proj = nn.Linear(256, 256, bias=False).to(device)
    emb = nn.Embedding(512, 256).to(device)
    collator = TileAlignedCollator(pad_to_multiple_of=128, buckets=(256,), pad_batch_to_multiple_of=8)
    batches = [
        collator([{"input_ids": list(range(1, lengths[i % len(lengths)] + 1))} for _ in range(7)])["input_ids"].to(device)
        for i in range(steps)
    ]

    loss_sum = torch.zeros((), device=device)
    with tpu_doctor(label="after:tpu_aligned") as after:
        for ids in batches:
            h = emb(ids).relu()
            b, s, d = h.shape
            out = tpu_proj(h.reshape(b * s, d)).reshape(b, s, d)
            loss_sum = loss_sum + (out**2).mean().detach()
            after.step()
    _ = float((loss_sum / max(1, steps)).item())

    return before.result, after.result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="torchtpu-doctor",
        description="Catch silent PyTorch portability & XLA bottlenecks on Cloud TPUs.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_scan = sub.add_parser("scan", help="Run static AST checks on a file or directory.")
    p_scan.add_argument("path", help="Python file or directory to scan.")
    p_scan.add_argument("--json", action="store_true", help="Output JSON.")
    p_scan.add_argument("--strict", action="store_true", help="Exit 1 if any issues are found.")

    p_demo = sub.add_parser("demo", help="Run a before/after TorchDispatchMode check.")
    p_demo.add_argument("--steps", type=int, default=5)
    p_demo.add_argument("--json", action="store_true", help="Output JSON.")

    args = parser.parse_args(argv)

    if args.cmd == "scan":
        try:
            issues = scan_path(args.path)
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps([i.to_dict() for i in issues], indent=2))
        else:
            print(format_issues(issues))
        return 1 if (args.strict and issues) else 0

    if args.cmd == "demo":
        before, after = run_demo(steps=args.steps)
        if args.json:
            print(json.dumps({"before": before.to_dict(), "after": after.to_dict()}, indent=2))
        else:
            print(before.format_text())
            print()
            print(after.format_text())
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
