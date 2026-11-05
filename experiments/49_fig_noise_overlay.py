#!/usr/bin/env python3
"""图：θ 曲线的噪声底叠加（6 块非单调 vs 3 块单调）。

数据全部来自 `results/p1_supplement.json`（48 号产物，零推理）。

口径说明（必须进图注）
  - 上格曲线 = `pareto/theta*_T8.json` 的 SOP（部署权重 d01e7aea，fold9 验证集
    前 8 个 batch 标定）；阴影带 = b38 对 32 个标定 batch 重抽 8 个的 2000 次
    bootstrap 的 2.5–97.5 分位。bootstrap 分布的中心与曲线点差 0.3%~2.8%，
    故**带宽按 bootstrap 实测宽度平移、中心对齐曲线点**。
  - 下格 = 十折各自权重的均值 ± 1 sd（不是标定噪声，是折间方差）。

用法::

    python3 experiments/49_fig_noise_overlay.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/mnt/ECG-SNN-LowPower")
R = ROOT / "results"
OUT = R / "figs" / "p1_noise_overlay.png"

C6 = "#2a78d6"      # slot 1 blue
C3 = "#eb6834"      # slot 2 orange
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURF = "#fcfcfb"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.6,
    "figure.facecolor": SURF,
    "axes.facecolor": SURF,
    "savefig.facecolor": SURF,
})

d = json.loads((R / "p1_supplement.json").read_text())

# 6 块：曲线（8-batch）+ bootstrap 带（平移后可加性）
noise = d["noise_floor_6block"]
th6 = np.array([r["theta"] for r in noise["rows"]])
s6 = np.array([r["sop_curve"] for r in noise["rows"]], dtype=float) / 1e6
w_lo = np.array([r["sop8_boot_mean"] - r["sop8_p2_5"] for r in noise["rows"]]) / 1e6
w_hi = np.array([r["sop8_p97_5"] - r["sop8_boot_mean"] for r in noise["rows"]]) / 1e6

# 3 块：十折 mean ± sd
sh = d["theta_curve_3block_delta"]
th3 = np.array([r["theta"] for r in sh["rows"]])
s3 = np.array([r["sops"]["mean"] for r in sh["rows"]]) / 1e6
e3 = np.array([r["sops"]["sd"] for r in sh["rows"]]) / 1e6

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(3.5, 3.7), sharex=True,
    gridspec_kw={"height_ratios": [1.05, 1.0], "hspace": 0.32})

# ── 上格：6 块 ───────────────────────────────────────────────────────
ax1.fill_between(th6, s6 - w_lo, s6 + w_hi, color=C6, alpha=0.16, lw=0)
ax1.plot(th6, s6, color=C6, lw=1.6, marker="o", ms=3.2,
         mfc=C6, mec=SURF, mew=1.0, zorder=3)
imin = int(np.argmin(s6))
ax1.plot(th6[imin], s6[imin], marker="o", ms=6.0, mfc=C6, mec=SURF, mew=1.6, zorder=4)
ax1.annotate(f"{s6[imin]:.2f} M", (th6[imin], s6[imin]),
             xytext=(2, 9), textcoords="offset points", color=INK, fontsize=7.5)
ax1.text(th6[0], s6.max() + 2.4, "6-block, Δ encoder", color=C6,
         fontsize=7.8, fontweight="bold")
ax1.text(0.02, 0.03, f"min at θ={th6[imin]:g}; dip vs neighbours z=1.25/0.77 (<1.96)",
         transform=ax1.transAxes, color=INK2, fontsize=6.6)

# ── 下格：3 块 ───────────────────────────────────────────────────────
ax2.fill_between(th3, s3 - e3, s3 + e3, color=C3, alpha=0.16, lw=0)
ax2.plot(th3, s3, color=C3, lw=1.6, marker="o", ms=3.2,
         mfc=C3, mec=SURF, mew=1.0, zorder=3)
ax2.plot(th3[-1], s3[-1], marker="o", ms=6.0, mfc=C3, mec=SURF, mew=1.6, zorder=4)
ax2.annotate(f"{s3[-1]:.2f} M", (th3[-1], s3[-1]),
             xytext=(-30, -11), textcoords="offset points", color=INK, fontsize=7.5)
ax2.text(th3[0], s3.max() + 2.2, "3-block, Δ encoder", color=C3,
         fontsize=7.8, fontweight="bold")
ax2.text(0.02, 0.06, "strictly decreasing, −21.4% over θ=0.05→0.30",
         transform=ax2.transAxes, color=INK2, fontsize=6.6)

# ── 公共 ─────────────────────────────────────────────────────────────
for ax in (ax1, ax2):
    ax.grid(axis="y", color=GRID, lw=0.5, ls="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, length=2.5, width=0.6)
    ax.set_ylabel("SOP (M)", color=INK2)

ax1.set_ylim(44.5, 57.5)
ax2.set_ylim(12.2, 24.0)
ax2.set_xscale("linear")
ax2.set_xticks([0.05, 0.10, 0.15, 0.20, 0.30])
ax2.set_xticklabels(["0.05", "0.10", "0.15", "0.20", "0.30"])
ax2.set_xlabel("θ (inference threshold, zero-shot)", color=INK2)
ax2.set_xlim(0.04, 0.31)

fig.tight_layout(pad=0.4)
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"→ {OUT}")
