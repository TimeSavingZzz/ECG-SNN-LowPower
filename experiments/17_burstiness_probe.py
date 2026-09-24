#!/usr/bin/env python3
"""17_burstiness_probe.py — 为什么「输入事件变少」却让深层 LIF 发放更多？

**要解释的实测事实（来自 16_calib_noise_floor.py，K=32 batch / 1024 样本，低噪）**

    θ          input     stem   block0   block1   block2   block3   block4   block5
    0.15       0.3220   0.1726   0.1116   0.0933   0.0750   0.1136   0.1445   0.0005
    0.3        0.2200   0.1946   0.1071   0.0952   0.0910   0.1274   0.1460   0.0004
    Δ(0.3-0.15) -0.1020  +0.0220  -0.0044  +0.0020  +0.0160  +0.0138  +0.0015  -0.0001

输入层事件率掉 32%，但 **stem / block2 / block3 反而增发**（符号随深度翻转，低 θ 端
方向相反）。这既非直觉，也非纯噪声（右侧抬升 z=8.84σ，见 16 号产物）。

**假设**：上游的 delta 编码器是**压摆率受限（slew-rate limited）**的——每步每导联最多
发一个事件，且参考值 `ref` 每步最多移动 ±θ（`encoding.py` 里 `ref += θ·(on − off)`）。
于是：

  · θ 小 → `ref` 紧跟信号，事件数多但**散落**（多为孤立单拍）
  · θ 大 → `ref` 跟不上快变段（QRS 上升沿最典型），同一方向上**连续多拍**被迫补发，
           形成**脉冲串（burst）**；事件总数下降，但事件高度**时间成簇**

LIF 神经元是泄漏积分器（β=0.9）：**串内相继到达的脉冲能累加膜电位**，而散落的孤立脉冲
会被泄漏吃掉。故「同样多的脉冲、成串的更有效」⇒ 稀疏但成簇的输入反而更强烈地驱动
下游，且效应随层深累积（正是实测的 block2/block3 增发）。

**本脚本只量测编码器本身**（无参数、不含 SNN 前向，跑得很快），输出每个 θ 下的：
  · 输入事件率（应与 16 号脚本的 input 逐位一致，作为复刻校验）
  · **平均游程长度** mean_run（孤立脉冲 ⇒ 1.0；成串 ⇒ >1）——假设的核心观测量
  · 游程长度分位数、lag-1 自相关
  · 编码重构误差（刻画压摆率受限造成的跟踪滞后）

**判据**：若 mean_run 随 θ 单调上升 ⇒ 假设成立，可以把机制写成
「压摆率受限编码 ⇒ 事件成簇 ⇒ LIF 积分放大」；若 mean_run 平或下降 ⇒ 假设被推翻，
论文里只能把逐层反转作为**经验现象**报告，不得附会机制。

用法（容器内）::

    python experiments/17_burstiness_probe.py \\
        --out results/pareto/burstiness_probe.json | tee logs/burstiness_probe.log
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

CALIB_BATCH_SIZE = 32   # 与 11 / 16 对齐
CALIB_FOLD = 9
N_LEADS = 12


def burst_stats(sp: torch.Tensor) -> dict:
    """sp: (B, 2C, T) 二值脉冲，[0:C]=ON 通道。

    返回时序成簇统计量。游程（run）= 同一通道上连续为 1 的最长片段。
        mean_run = 事件总数 / 游程条数
    θ 小时事件散落 ⇒ mean_run → 1；θ 大时被迫补发成串 ⇒ mean_run > 1。
    """
    on = sp[:, :N_LEADS, :].float()                       # (B, C, T)
    total = on.sum()
    # 游程起点：本拍为 1 且前一拍为 0（t=0 特殊处理）
    starts = (on[:, :, 1:] - on[:, :, :-1]).clamp(min=0).sum() + on[:, :, 0].sum()
    mean_run = float(total / starts.clamp(min=1))

    # 游程长度分布：把整批展开成 (B*C, T)，用 numpy 逐行找游程（12288 行，很快）
    flat = on.reshape(-1, on.shape[-1]).cpu().numpy().astype(np.int8)
    lengths = []
    for row in flat:
        if not row.any():
            continue
        padded = np.concatenate([[0], row, [0]])
        dd = np.diff(padded)
        ss = np.nonzero(dd == 1)[0]
        ee = np.nonzero(dd == -1)[0]
        lengths.append(ee - ss)
    if lengths:
        L = np.concatenate(lengths)
        p50, p95, mx = (float(np.percentile(L, 50)), float(np.percentile(L, 95)),
                        int(L.max()))
        frac_multi = float((L >= 2).sum() / len(L))       # 长度≥2 的游程占比
    else:
        p50 = p95 = float(0); mx = 0; frac_multi = 0.0

    # lag-1 自相关：成簇信号显著为正
    a = on[:, :, :-1].flatten()
    b_ = on[:, :, 1:].flatten()
    ma, mb = a.mean(), b_.mean()
    denom = ((a - ma).pow(2).sum() * (b_ - mb).pow(2).sum()).sqrt()
    ac1 = float(((a - ma) * (b_ - mb)).sum() / denom) if denom > 0 else float("nan")

    return {
        "event_rate": float(sp.float().mean()),
        "mean_run_len": mean_run,
        "median_run_len": p50,
        "p95_run_len": p95,
        "max_run_len": mx,
        "frac_runs_ge2": frac_multi,
        "autocorr_lag1": ac1,
    }


def reconstruction_error(x: torch.Tensor, sp: torch.Tensor, theta: float) -> float:
    """把 ON/OFF 事件按 ±θ 积分回信号（编码器的朴素解码器），算相对重构误差。

    `ref` 从 x[:, :, 0] 起步，每步按 (ON − OFF) 移动 ±θ。误差大 = 压摆率受限严重。
    """
    on = sp[:, :N_LEADS, :].float()
    off = sp[:, N_LEADS:, :].float()
    step = theta * (on - off)                              # (B, C, T)
    rec = x[:, :, 0:1] + torch.cumsum(step[:, :, :-1], dim=2)
    tgt = x[:, :, :-1]
    num = (rec - tgt).pow(2).sum().sqrt()
    den = tgt.pow(2).sum().sqrt().clamp(min=1e-12)
    return float(num / den)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/mnt/ECG-SNN-LowPower/results/ptbxl_cache")
    ap.add_argument("--out", default="/mnt/ECG-SNN-LowPower/results/pareto/burstiness_probe.json")
    ap.add_argument("--theta", default="0.05,0.075,0.1,0.15,0.2,0.3")
    ap.add_argument("--k-batches", type=int, default=32, help="与 16 号脚本同口径（1024 样本）")
    ap.add_argument("--noise-floor", default="/mnt/ECG-SNN-LowPower/results/pareto/calib_noise_floor.json",
                    help="16 号产物，用于 input 率的复刻校验与逐层对照")
    ap.add_argument("--hidden-keys", default="stem,block0,block1,block2,block3,block4,block5",
                    help="用于算「深层合计」的层名")
    args = ap.parse_args()

    thetas = [float(t) for t in args.theta.split(",") if t.strip()]
    val = PTBXLCache(Path(args.cache_dir), fold_subset=[CALIB_FOLD])
    loader = DataLoader(val, batch_size=CALIB_BATCH_SIZE, num_workers=2)

    # 攒够 K 个 batch 的输入（编码器无参数，可直接在 CPU 上做）
    xs = []
    for i, (x, _y, _) in enumerate(loader):
        if i >= args.k_batches:
            break
        xs.append(x)
    X = torch.cat(xs, dim=0)
    print(f"标定集 fold{CALIB_FOLD} 取前 {args.k_batches} 个 batch = {X.shape[0]} 条，"
          f"每条 {X.shape[1]} 导联 × {X.shape[2]} 拍", flush=True)

    nf = {}
    nf_path = Path(args.noise_floor)
    if nf_path.exists():
        nf = {r["theta"]: r for r in json.loads(nf_path.read_text())["rows"]}

    rows = []
    print(f"\n{'θ':<7}{'事件率':>9}{'mean_run':>10}{'≥2游程占比':>12}{'p95游程':>9}"
          f"{'lag1自相关':>12}{'重构误差':>10}", flush=True)
    for theta in thetas:
        sp = delta_encode_batch_12lead(X, theta=theta)
        st = burst_stats(sp)
        st["theta"] = theta
        st["reconstruction_rel_error"] = reconstruction_error(X, sp, theta)
        if theta in nf:
            st["input_rate_from_16"] = nf[theta]["input_k"]
            # 容差 1e-6：16 号的率在 GPU 上算、17 号在 CPU 上算，float32 归约顺序
            # 不同会带来 ~1e-8 的差（实测最大 1.7e-8），1e-9 会误判成「未复刻」。
            st["input_rate_replicated"] = abs(st["event_rate"] - nf[theta]["input_k"]) < 1e-6
            st["hidden_sum_from_16"] = nf[theta]["hidden_sum_k"]
        rows.append(st)
        print(f"{theta:<7}{st['event_rate']:>9.4f}{st['mean_run_len']:>10.4f}"
              f"{st['frac_runs_ge2']:>12.4f}{st['p95_run_len']:>9.1f}"
              f"{st['autocorr_lag1']:>12.4f}{st['reconstruction_rel_error']:>10.4f}",
              flush=True)

    # ── 判据 ────────────────────────────────────────────────────────────
    mr = [r["mean_run_len"] for r in rows]
    ac = [r["autocorr_lag1"] for r in rows]
    er = [r["event_rate"] for r in rows]

    mono_mr_up = all(mr[i] <= mr[i + 1] + 1e-12 for i in range(len(mr) - 1))
    mono_ac_up = all(ac[i] <= ac[i + 1] + 1e-12 for i in range(len(ac) - 1))
    mono_er_down = all(er[i] >= er[i + 1] - 1e-12 for i in range(len(er) - 1))
    repl = [r.get("input_rate_replicated") for r in rows]
    repl_all = all(v for v in repl if v is not None)

    # 与「深层合计」的对照：若成簇假设成立，两者应同向
    hs = [r.get("hidden_sum_from_16") for r in rows]
    has_hs = all(v is not None for v in hs)
    corr = None
    if has_hs:
        corr = float(np.corrcoef(np.array(mr), np.array(hs))[0, 1])

    print(f"\n  事件率单调下降 : {'是' if mono_er_down else '否'}", flush=True)
    print(f"  平均游程单调上升 : {'是' if mono_mr_up else '否'}"
          f"   （{mr[0]:.3f} → {mr[-1]:.3f}）", flush=True)
    print(f"  lag1 自相关单调上升 : {'是' if mono_ac_up else '否'}"
          f"   （{ac[0]:.4f} → {ac[-1]:.4f}）", flush=True)
    print(f"  与 16 号 input 率复刻一致 : {'是' if repl_all else '否'}", flush=True)
    if corr is not None:
        print(f"  corr(mean_run, 隐藏层合计发放率) = {corr:.4f}", flush=True)

    if mono_mr_up and mono_ac_up:
        verdict = ("MECHANISM_SUPPORTED：事件率单调下降的同时，平均游程长度与 lag-1 "
                   "自相关单调上升 ⇒ delta 编码器在大 θ 下把「更多事件」换成了"
                   "「更成串的事件」。LIF 是泄漏积分器，成串脉冲的膜电位累积效率远高于"
                   "等量散落脉冲，故深层反而增发。机制可以写成："
                   "「压摆率受限编码 ⇒ 事件时间成簇 ⇒ 深层 LIF 积分放大」。")
    elif mono_mr_up:
        verdict = ("MECHANISM_PARTIAL：平均游程随 θ 上升（支持成簇假设），但 lag-1 "
                   "自相关不单调，成簇的具体形式需再刻画。")
    else:
        verdict = ("MECHANISM_REJECTED：平均游程不随 θ 上升 ⇒ 「事件成簇」假设不成立，"
                   "深层增发另有原因。论文中只把逐层反转作为经验现象报告，"
                   "不得把成簇机制写成结论。")
    print(f"\n  ⇒ {verdict}", flush=True)

    out = {
        "fold": CALIB_FOLD, "batch_size": CALIB_BATCH_SIZE,
        "k_batches": args.k_batches, "n_samples": int(X.shape[0]),
        "thetas": [r["theta"] for r in rows], "rows": rows,
        "event_rate_monotone_decreasing": mono_er_down,
        "mean_run_len_monotone_increasing": mono_mr_up,
        "autocorr_lag1_monotone_increasing": mono_ac_up,
        "input_rate_replicated_from_16": repl_all,
        "corr_mean_run_vs_hidden_rate": corr,
        "verdict": verdict,
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n产物 → {args.out}", flush=True)


if __name__ == "__main__":
    main()
