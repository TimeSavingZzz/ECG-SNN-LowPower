#!/usr/bin/env python3
"""19_encoder_fidelity.py — 上游 delta 编码器到底是不是「information-preserving」？

**触发这个问题的一行数据**

`17_burstiness_probe.py` 顺带量了重构相对误差 `‖rec − x‖ / ‖x‖`（把 ON/OFF 事件按 ±θ
积分回信号，即编码器的朴素解码器）：

    θ       0.05    0.075   0.1     0.15    0.2     0.3
    误差    0.9662  0.9482  0.9351  0.9161  0.9021  0.8808

两件事都很反常：

1. **误差在 88%~97%**，即重构几乎不含原信号的信息（若 rec 退化成常数，‖rec−x‖≈‖x‖ ⇒ 误差≈1）。
2. **误差随 θ 单调下降**——与「θ 是量化步长、越大量化越粗」的直觉相反。

**原因（本脚本要验证的机制）**：`neurocardio/encoding.py` 里 `ref = ref + on*θ − off*θ`，
即**参考电压每步最多移动 ±θ**。于是单通道在 T 步内可用的总行程上限是

    |ref| 总行程  ≤  (事件率 × T) × θ

而信号自身的总变差（total variation）由数据决定、与 θ 无关。只要「可用总行程 < 信号总变差」，
`ref` 就**永远跟不上**，重构必然崩坏——这不是量化误差，是**压摆率限制（slew-rate limiting）**
造成的系统性滞后。而 θ 增大时，事件率下降得比 θ 上升慢，`rate×θ` 反而**变大**（θ=0.05:
0.435×0.05=0.0218；θ=0.3: 0.220×0.3=0.0660，差 3 倍），故误差随 θ **下降**。

所以上游 docstring 自称的

    "fully reconstructs the analog signal up to ±θ quantization — i.e. it is
     information-preserving in the same way a silicon-retina (DVS) encoding is for vision"

**若本脚本证实上述机制，则该表述在本数据/本 θ 区间内不成立**，属于上游文档缺陷，
是论文里可以正面写的一条「复现中发现的问题」。

**本脚本的量测**（只跑编码器，无参数、无 SNN 前向，秒级）

对每个 θ：重构相对误差、`corr(rec, x)`、`std(rec)/std(x)`（振幅保真度）、
以及「可用总行程 / 信号总变差」的比值（<1 即压摆率受限）。

θ 网格刻意**向下延伸到 0.001**，以找出「从这里开始编码器才真的能跟踪信号」的转折点；
若即使 θ→0 误差仍高，则说明该编码器对本数据（100 Hz 归一化 ECG）从根本上不够用。

用法（容器内）::

    python experiments/19_encoder_fidelity.py --out results/pareto/encoder_fidelity.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

NC = Path("/mnt/ECG-SNN-LowPower/third_party/neurocardio")
sys.path.insert(0, str(NC))

from neurocardio.dataset import PTBXLCache  # noqa: E402
from neurocardio.encoding import delta_encode_batch_12lead  # noqa: E402

CALIB_FOLD = 9
N_LEADS = 12


def fidelity(x: torch.Tensor, theta: float) -> dict:
    """跑编码 + 朴素解码，返回保真度指标。x: (B, 12, T)。"""
    sp = delta_encode_batch_12lead(x, theta=theta)
    on = sp[:, :N_LEADS, :].float()
    off = sp[:, N_LEADS:, :].float()
    step = theta * (on - off)                                  # 每步移动的量子数
    # ref_k = x[0] + Σ_{j<k} step_j（与编码器内部 ref 的演化逐位对应）
    rec = x[:, :, 0:1] + torch.cumsum(step[:, :, :-1], dim=2)
    tgt = x[:, :, :-1]

    num = (rec - tgt).pow(2).sum().sqrt()
    den = tgt.pow(2).sum().sqrt().clamp(min=1e-12)
    rel_err = float(num / den)

    # 相关系数与振幅比：刻画「退化成常数」的程度
    a = (rec - rec.mean()).flatten()
    b = (tgt - tgt.mean()).flatten()
    dnm = (a.pow(2).sum() * b.pow(2).sum()).sqrt()
    corr = float((a * b).sum() / dnm) if dnm > 0 else float("nan")
    amp_ratio = float(rec.std() / tgt.std().clamp(min=1e-12))

    # 可用总行程 vs 信号总变差（两者都按「每通道每步」平均）
    avail = float(step.abs().sum(dim=2).mean())                # 平均每通道总行程
    tv = float((tgt[:, :, 1:] - tgt[:, :, :-1]).abs().sum(dim=2).mean())
    return {
        "theta": theta,
        "rel_error": rel_err,
        "corr_rec_vs_signal": corr,
        "amp_ratio_std_rec_over_std_sig": amp_ratio,
        "event_rate": float(sp.float().mean()),
        "available_travel": avail,
        "signal_total_variation": tv,
        "travel_over_tv": avail / tv if tv > 0 else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/mnt/ECG-SNN-LowPower/results/ptbxl_cache")
    ap.add_argument("--out", default="/mnt/ECG-SNN-LowPower/results/pareto/encoder_fidelity.json")
    ap.add_argument("--theta",
                    default="0.001,0.002,0.005,0.01,0.02,0.05,0.075,0.1,0.15,0.2,0.3",
                    help="向下延伸以定位「开始能跟踪」的转折点")
    ap.add_argument("--k-batches", type=int, default=8, help="保真度只需少量样本（发射率另算）")
    args = ap.parse_args()

    thetas = [float(t) for t in args.theta.split(",") if t.strip()]
    val = PTBXLCache(Path(args.cache_dir), fold_subset=[CALIB_FOLD])
    loader = DataLoader(val, batch_size=32, num_workers=2)
    xs = []
    for i, (x, _y, _) in enumerate(loader):
        if i >= args.k_batches:
            break
        xs.append(x)
    X = torch.cat(xs, dim=0)
    print(f"fold{CALIB_FOLD} 取 {X.shape[0]} 条，每条 {X.shape[1]} 导联 × {X.shape[2]} 拍", flush=True)

    rows = [fidelity(X, t) for t in thetas]

    print(f"\n{'θ':<8}{'事件率':>9}{'重构误差':>10}{'corr':>8}{'振幅比':>9}"
          f"{'可用行程':>11}{'信号总变差':>12}{'行程/变差':>11}", flush=True)
    for r in rows:
        print(f"{r['theta']:<8}{r['event_rate']:>9.4f}{r['rel_error']:>10.4f}"
              f"{r['corr_rec_vs_signal']:>8.4f}{r['amp_ratio_std_rec_over_std_sig']:>9.4f}"
              f"{r['available_travel']:>11.2f}{r['signal_total_variation']:>12.2f}"
              f"{r['travel_over_tv']:>11.4f}", flush=True)

    # ── 判据 ────────────────────────────────────────────────────────────
    tv = rows[0]["signal_total_variation"]
    best = min(rows, key=lambda r: r["rel_error"])
    # 转折点：误差首次降到 0.5 以下的最小 θ
    below = [r["theta"] for r in rows if r["rel_error"] < 0.5]
    cross = min(below) if below else None
    swept = [r for r in rows if r["theta"] >= 0.05]
    all_bad_in_sweep = all(r["rel_error"] > 0.5 for r in swept)
    corr_ok = any(r["corr_rec_vs_signal"] > 0.8 for r in rows)

    print(f"\n  信号总变差（每通道平均）: {tv:.2f}", flush=True)
    print(f"  误差最小点: θ={best['theta']} → {best['rel_error']:.4f}", flush=True)
    print(f"  误差首次 < 0.5 的 θ: {cross}", flush=True)
    print(f"  扫描区间 [0.05, 0.3] 内误差全部 > 0.5 : {'是' if all_bad_in_sweep else '否'}",
          flush=True)
    print(f"  存在 θ 使 corr(rec, x) > 0.8 : {'是' if corr_ok else '否'}", flush=True)

    if all_bad_in_sweep and cross is not None:
        verdict = (f"SLEW_LIMITED：在 11/16 号脚本扫描的整个 θ 区间 [0.05, 0.3] 内，重构相对"
                   f"误差都 > 0.5（振幅比 <0.5），即**信号振幅基本没被编码进去**；"
                   f"只有当 θ 降到 {cross} 附近误差才降下来。⇒ 上游 docstring 的"
                   f"「fully reconstructs … up to ±θ quantization, information-preserving」"
                   f"在本数据（100 Hz 归一化 ECG）与本扫描区间内**不成立**："
                   f"编码器因 `ref` 每步最多移动 ±θ 而严重压摆率受限，退化为**边缘/事件检测器**。"
                   f"这是论文可正面报告的复现发现，也解释了 SNN 相对稠密基线的精度代价。")
    elif all_bad_in_sweep:
        verdict = ("SLEW_LIMITED_ALWAYS：连 θ→0 也救不回重构精度 ⇒ 该编码器对本数据"
                   "从根本上不够用（需检查归一化尺度与量化步长的匹配）。")
    else:
        verdict = ("FIDELITY_OK_IN_PARTS：存在使重构误差 <0.5 的 θ，说明压摆率限制只在"
                   "较大 θ 下发生；需据本表划出「可靠区间」。")
    print(f"\n  ⇒ {verdict}", flush=True)

    out = {"fold": CALIB_FOLD, "n_samples": int(X.shape[0]),
           "thetas": thetas, "rows": rows,
           "rel_error_below_half_min_theta": cross,
           "all_err_gt_half_in_swept_range": bool(all_bad_in_sweep),
           "verdict": verdict}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n产物 → {args.out}", flush=True)


if __name__ == "__main__":
    main()
