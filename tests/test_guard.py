from __future__ import annotations

import torch
import torch.nn as nn

from torchtpu_doctor import tpu_doctor


def test_guard_catches_bottlenecks_and_clean_loop() -> None:
    fc = nn.Linear(190, 190, bias=False)
    buf = torch.zeros(7, 256, 190)

    with tpu_doctor("naive") as bad:
        for s in (73, 119, 141, 187):
            x = torch.randn(7, s, 190)
            y = x.transpose(1, 2).relu().transpose(1, 2)
            out = fc(y.reshape(-1, 190)).reshape(7, s, 190)
            buf[:, :s, :] = out.detach()
            _ = out.sum().item()
            bad.step()

    assert bad.result.steps == 4
    assert bad.result.host_syncs >= 4
    assert bad.result.slice_writes == 4
    assert bad.result.strided_views >= 4
    assert len(bad.result.dynamic_ops) >= 1
    assert bad.result.mxu_lane_fill_pct < 80.0
    assert bad.result.has_bottlenecks is True

    fc_aligned = nn.Linear(256, 256, bias=False)
    loss_sum = torch.zeros(())
    with tpu_doctor("clean") as good:
        for _ in range(4):
            x = torch.randn(8, 256, 256)
            out = fc_aligned(x.reshape(-1, 256))
            loss_sum = loss_sum + (out**2).mean().detach()
            good.step()

    assert good.result.host_syncs == 0
    assert good.result.slice_writes == 0
    assert good.result.strided_views == 0
    assert good.result.mxu_lane_fill_pct == 100.0
    assert good.result.has_bottlenecks is False


def test_guard_ignores_heterogeneous_transformer_layers_when_steps_are_static() -> None:
    q_proj = nn.Linear(256, 256, bias=False)
    k_proj = nn.Linear(256, 128, bias=False)
    up_proj = nn.Linear(256, 512, bias=False)
    down_proj = nn.Linear(512, 256, bias=False)
    lora_a = nn.Linear(256, 8, bias=False)
    lora_b = nn.Linear(8, 256, bias=False)
    opt = torch.optim.AdamW(
        [*q_proj.parameters(), *k_proj.parameters(), *up_proj.parameters(), *down_proj.parameters(), *lora_a.parameters(), *lora_b.parameters()],
        lr=1e-4,
    )

    loss_sum = torch.zeros(())
    with tpu_doctor("lora_transformer") as doc:
        for _ in range(4):
            opt.zero_grad(set_to_none=True)
            x = torch.randn(4, 128, 256)
            q = q_proj(x) + lora_b(lora_a(x))
            _ = k_proj(x)
            h = down_proj(up_proj(q).relu())
            loss = (h**2).mean()
            loss.backward()
            opt.step()
            loss_sum = loss_sum + loss.detach()
            doc.step()

    assert doc.result.dynamic_ops == {}
    assert doc.result.host_syncs == 0
    assert doc.result.mxu_lane_fill_pct >= 98.0
    assert doc.result.has_bottlenecks is False

