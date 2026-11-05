#!/usr/bin/env python3
"""给 neurocardio 打「二值双极性前端」补丁（幂等）。

改动两处，均为**新增**，不改既有行为：
  1. `neurocardio/ablation.py`
       + dual_polarity_binary_encode(x, theta)     二值双极性编码
       + NeuroCardioShallowPolarityBinary          3 块 + 上面的编码
  2. `neurocardio/factory.py`
       + 名字解析 `ab_shallow_pol_bin{θ×100}`，如 `ab_shallow_pol_bin050` → θ=0.50

原文件先备份为 `*.bak-20261105`。重复运行不会重复插入。

用法::

    python3 experiments/55_patch_binary_frontend.py
"""
from __future__ import annotations

from pathlib import Path

PKG = Path("/mnt/ECG-SNN-LowPower/third_party/neurocardio/neurocardio")
ABL = PKG / "ablation.py"
FAC = PKG / "factory.py"

MARK_A = "# ---- binary dual-polarity frontend (batch 012, 2026-11-05) ----"
MARK_F = "# ---- binary dual-polarity frontend (batch 012) ----"

ABL_CODE = '''

{mark}
def dual_polarity_binary_encode(x: torch.Tensor, theta: float = 0.5,
                                eps: float = 1e-5) -> torch.Tensor:
    """Per-lead z-score -> dual-polarity *binary* spikes.

    与 Sarhaddi 的连续整流编码同构（z-score 后按符号拆两路），区别只在最后一步：
    ReLU(±z) -> 1[±z >= theta]。于是前端输出是真正的 0/1 spike，
    theta 从「无意义的摆设」变成真正的**发放阈值**，SOP 也可直接按发放率计费。
    """
    mu = x.mean(dim=-1, keepdim=True)
    sd = x.std(dim=-1, keepdim=True).clamp_min(eps)
    z = (x - mu) / sd
    return torch.cat([(z >= theta).float(), (z <= -theta).float()], dim=1)


class NeuroCardioShallowPolarityBinary(NeuroCardio):
    """P1a（3 块 + 双极性前端）的二值化版本：训练阈值由 theta_bin 固定。

    训练脚本不会把 cfg.theta 传进 forward，所以这里自己带上，
    保证「训练用哪个阈值，推理就用哪个阈值」。
    """

    def __init__(self, num_classes: int = 5, beta: float = 0.9,
                 theta_bin: float = 0.5):
        super().__init__(num_classes=num_classes, beta=beta,
                         channel_plan=(64, 128, 256),
                         dilation_plan=(1, 2, 4),
                         pool_plan=(1, 2, 2))
        self.theta_bin = float(theta_bin)
        self.delta_encode = dual_polarity_binary_encode

    def forward(self, x: torch.Tensor, theta: float | None = None,
                return_traces: bool = False):
        return super().forward(x, theta=self.theta_bin,
                               return_traces=return_traces)
'''

FAC_CODE = '''    {mark}
    if name.startswith("ab_shallow_pol_bin"):
        from neurocardio.ablation import NeuroCardioShallowPolarityBinary
        suffix = name[len("ab_shallow_pol_bin"):]
        theta_bin = float(suffix) / 100.0 if suffix else 0.5
        return NeuroCardioShallowPolarityBinary(num_classes=num_classes,
                                                theta_bin=theta_bin)
'''


def patch(path: Path, mark: str, code: str, anchor: str, before: bool) -> bool:
    txt = path.read_text()
    if mark in txt:
        print(f"[SKIP] {path.name} 已打过补丁")
        return False
    bak = path.with_suffix(path.suffix + ".bak-20261105")
    if not bak.exists():
        bak.write_text(txt)
        print(f"  备份 → {bak.name}")
    if before:
        assert anchor in txt, f"锚点不存在：{anchor!r}"
        txt = txt.replace(anchor, code.strip("\n") + "\n\n" + anchor, 1)
    else:
        assert anchor in txt, f"锚点不存在：{anchor!r}"
        txt = txt.replace(anchor, code.rstrip("\n") + "\n\n" + anchor, 1)
    path.write_text(txt)
    print(f"  已写入 {path.name}")
    return True


def main() -> None:
    patch(ABL, MARK_A, ABL_CODE.replace("{mark}", MARK_A),
          anchor='if __name__ == "__main__":', before=True)
    patch(FAC, MARK_F, FAC_CODE.replace("{mark}", MARK_F),
          anchor='    raise ValueError(f"unknown model: {name}")', before=True)

    # ---- 自检 ----
    import sys
    sys.path.insert(0, "/mnt/ECG-SNN-LowPower/third_party/neurocardio")
    import torch
    from neurocardio.factory import make_model
    from neurocardio.model import count_parameters

    x = torch.randn(2, 12, 1000)
    for nm in ("ab_shallow_pol_bin030", "ab_shallow_pol_bin050",
               "ab_shallow_pol_bin100"):
        m = make_model(nm, num_classes=5).eval()
        with torch.no_grad():
            _logits, tr = m(x, return_traces=True)
        print(f"{nm:24s} theta_bin={m.theta_bin:<4g} "
              f"params={count_parameters(m):>9,} "
              f"input_rate={float(tr['input_spikes'].float().mean()):.4f}")
    ref = make_model("ab_shallow_pol", num_classes=5)
    print(f"对照组 ab_shallow_pol params={count_parameters(ref):>9,}")


if __name__ == "__main__":
    main()
