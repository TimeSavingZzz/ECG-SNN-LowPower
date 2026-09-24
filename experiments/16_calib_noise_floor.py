#!/usr/bin/env python3
"""16_calib_noise_floor.py — 「θ → 总 SOP 非单调」是真机制，还是标定噪声？

**本文件要判决的问题**

`11_pareto_sweep.py` 在 PTB-XL 上（T_dense=8）测出 `total_sops` 对输入编码阈值 θ
**非单调**：

    θ        0.05    0.075    0.1     0.15    0.2     0.3
    SOP(M)   35.32   34.41   33.79   33.10   33.02   34.17   ← θ=0.2 触底后**回升**
    发放率   0.1072  0.1040  0.1025  0.1018  0.1033  0.1101   ← θ=0.15 触底后回升

若成立，机制解释是「**输入层省、隐藏层增发**」的对冲：θ↑ 让输入事件变稀（省），
但编码变稀疏后隐藏层的驱动电流模式改变、部分层反而发放更密（增），净效果在中间 θ
取最小。**本次文献检索中未找到任何一篇报告过这个现象**（所有已发表工作调的是神经元
阈值 V_th，无一篇调输入编码器阈值 θ）——所以它是候选的核心创新点。

**但它可能是噪声。** 因为发放率只由 `calibrate_spike_rates(n_batches=8)` 估计，即
fold-9 的**前 256 个样本**。而「回升」幅度（θ=0.2→0.3，+3.5%）与相邻 θ 之间的差异
（θ=0.05→0.075，−2.6%）**同一个量级**，完全可能只是这 256 个样本的抽样波动。

**判决方法**

1. 在同一份冻结快照上，为每个 θ 收集 K=32 个 batch（1024 样本）的**逐 batch** 发放率；
   先验证「前 8 个 batch 的均值」与前作 `theta{θ}_T8.json` 里记的 `spike_rates`
   **逐位一致**（复刻成功证明），否则本脚本的结论不可信。
2. **bootstrap**：从 K 个 batch 里有放回抽 8 个，重算 `total_sops`，得 8-batch 估计量
   的抽样分布（均值/标准差/分位）。
3. 用 K=32 全量算一个**低噪点估计**（噪声约降 √(32/8)=2 倍）。
4. 判据：
   - K=32 曲线仍非单调           ⇒ 发现是真的，可以写进论文
   - K=32 曲线变单调             ⇒ 8-batch 的非单调是噪声，**不得**作为贡献
   - 8-batch 估计量的 argmin 在内部 θ 的频率 ≈ 随机 ⇒ 同上（噪声）

用法（容器内）::

    CUDA_VISIBLE_DEVICES=3 python experiments/16_calib_noise_floor.py \\
        --out results/pareto/calib_noise_floor.json | tee logs/calib_noise_floor.log
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

NC = Path("/mnt/ECG-SNN-LowPower/third_party/neurocardio")
sys.path.insert(0, str(NC))

from neurocardio.compute import calibrate_spike_rates, cost_snn  # noqa: E402
from neurocardio.dataset import PTBXLCache  # noqa: E402
from neurocardio.model import NeuroCardio  # noqa: E402

# 与 11_pareto_sweep.py 完全对齐（否则复刻校验必然失败）
CALIB_BATCH_SIZE = 32
CALIB_FOLD = 9


def bind_theta(model, theta: float):
    """把 θ 固定到 model 实例上（与 11_pareto_sweep.py::bind_theta 同一手法）。

    `compute.calibrate_spike_rates` / `cost_snn` 内部只调 `model(x)`，不透传 theta
    （写死 0.15），所以必须覆盖实例属性。
    """
    orig = model.forward

    def bound(x, *a, **kw):
        kw.setdefault("theta", theta)
        return orig(x, *a, **kw)

    model.forward = bound
    return model


def batch_rates(model, loader, dev, k_batches: int):
    """跑前 k_batches 个 batch，返回**逐 batch** 的逐层发放率（与上游同口径）。

    上游 `calibrate_spike_rates` 是「每个 batch 先在自己内部取 mean，再对 batch 求均值」，
    即逐 batch 平均、**不做样本加权**。这里刻意复制这一点，才能与前作逐位对拍。
    """
    model.eval()
    out = []
    with torch.no_grad():
        for i, (x, _y, _) in enumerate(loader):
            if i >= k_batches:
                break
            x = x.to(dev)
            _logits, tr = model(x, return_traces=True)
            d = {
                "input": float(tr["input_spikes"].float().mean()),
                "stem": float(tr["stem"].float().mean()),
                "fc": float(tr["s_fc"].float().mean()),
            }
            for j, s in enumerate(tr["blocks"]):
                d[f"block{j}"] = float(s.float().mean())
            out.append(d)
    return out


def mean_rates(batch_list, idx=None) -> dict:
    """对给定 batch 子集求逐层平均（idx=None 表示全用）。"""
    use = batch_list if idx is None else [batch_list[i] for i in idx]
    n = len(use)
    keys = use[0].keys()
    return {k: sum(d[k] for d in use) / n for k in keys}


def hidden_sum(rates: dict) -> float:
    """隐藏层（stem + 6 个 block + fc）合计发放率，用于查「隐藏层增发」的机制。"""
    return (rates["stem"] + sum(rates[f"block{i}"] for i in range(6)) + rates["fc"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/mnt/ECG-SNN-LowPower/results/ptbxl_cache")
    ap.add_argument("--snapshot", default="/mnt/ECG-SNN-LowPower/results/pareto/_snapshot_best.pt",
                    help="11_pareto_sweep.py 冻结的那份快照（md5 会记进产物）")
    ap.add_argument("--pareto-dir", default="/mnt/ECG-SNN-LowPower/results/pareto",
                    help="11 的产物目录，用于逐位复刻校验")
    ap.add_argument("--out", default="/mnt/ECG-SNN-LowPower/results/pareto/calib_noise_floor.json")
    ap.add_argument("--theta", default="0.05,0.075,0.1,0.15,0.2,0.3")
    ap.add_argument("--t-dense", type=int, default=8)
    ap.add_argument("--k-batches", type=int, default=32,
                    help="每个 θ 收集多少 batch 用于 bootstrap（默认 32 → 1024 样本）")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    thetas = [float(t) for t in args.theta.split(",") if t.strip()]
    snap = Path(args.snapshot)
    if not snap.exists():
        raise SystemExit(f"找不到冻结快照 {snap}（先跑 11_pareto_sweep.py）")
    md5 = hashlib.md5(snap.read_bytes()).hexdigest()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(snap, map_location="cpu", weights_only=False)
    labels = ckpt.get("label_columns") or ["NORM", "MI", "STTC", "CD", "HYP"]
    state = ckpt.get("ema") or ckpt.get("state_dict")

    print(f"快照 {snap}\n  md5={md5}  T_dense={args.t_dense}  θ={thetas}")
    print(f"device={dev}  标定集=fold{CALIB_FOLD}  batch_size={CALIB_BATCH_SIZE}  "
          f"K={args.k_batches} 个 batch（{args.k_batches * CALIB_BATCH_SIZE} 样本）  "
          f"bootstrap={args.n_boot} 次", flush=True)

    val = PTBXLCache(Path(args.cache_dir), fold_subset=[CALIB_FOLD])
    loader = DataLoader(val, batch_size=CALIB_BATCH_SIZE, num_workers=2, pin_memory=True)
    print(f"标定集共 {len(val)} 条", flush=True)

    model = NeuroCardio(num_classes=len(labels), T_dense=args.t_dense)
    model.load_state_dict(state)
    model.to(dev)
    model.eval()

    rng = np.random.default_rng(args.seed)
    draws = rng.integers(0, args.k_batches, size=(args.n_boot, 8))  # 8-batch 有放回抽样

    # 每轮都要把 forward 还原回原始绑定方法——否则 6 个 θ 会链式套 wrapper
    orig_forward = model.forward

    rows = []
    sops_by_theta: dict[float, np.ndarray] = {}
    print("\n=== 逐 θ 收集逐 batch 发放率 ===", flush=True)
    for theta in thetas:
        t0 = time.time()
        m = bind_theta(model, theta)
        try:
            bl = batch_rates(m, loader, dev, args.k_batches)
        finally:
            model.forward = orig_forward

        # (1) 复刻校验：前 8 个 batch 的均值必须等于前作记录的 spike_rates
        r8 = mean_rates(bl[:8])
        ref_path = Path(args.pareto_dir) / f"theta{theta}_T{args.t_dense}.json"
        repl_ok, repl_diff = None, None
        if ref_path.exists():
            ref = json.loads(ref_path.read_text())["spike_rates"]
            diffs = [abs(r8[k] - ref[k]) for k in r8 if k in ref]
            repl_diff = max(diffs) if diffs else None
            repl_ok = (repl_diff is not None and repl_diff < 1e-9)

        # (2) K-batch 低噪点估计
        rK = mean_rates(bl)
        costK = cost_snn(m, rK, T_in=1000)

        # (3) bootstrap：8-batch 估计量的抽样分布
        sops = np.empty(args.n_boot)
        rate_means = np.empty(args.n_boot)
        for b in range(args.n_boot):
            rb = mean_rates(bl, idx=draws[b])
            cb = cost_snn(m, rb, T_in=1000)
            sops[b] = cb.total_sops
            rate_means[b] = cb.spike_rate_mean
        sops_by_theta[theta] = sops

        rows.append({
            "theta": theta,
            "sop_k": costK.total_sops,
            "spike_rate_mean_k": costK.spike_rate_mean,
            "input_k": rK["input"],
            "hidden_sum_k": hidden_sum(rK),
            "per_layer_k": rK,
            "sop8_boot_mean": float(sops.mean()),
            "sop8_boot_std": float(sops.std(ddof=1)),
            "sop8_boot_p2_5": float(np.percentile(sops, 2.5)),
            "sop8_boot_p97_5": float(np.percentile(sops, 97.5)),
            "rate_mean8_boot_std": float(rate_means.std(ddof=1)),
            "replicated_from_11": repl_ok,
            "replicate_max_abs_diff": repl_diff,
            "seconds": round(time.time() - t0, 1),
        })
        print(f"  θ={theta:<6} SOP(K={args.k_batches})={costK.total_sops:>12.0f}  "
              f"SOP(8-batch)={sops.mean():>12.0f} ± {sops.std(ddof=1):>9.0f}  "
              f"[{np.percentile(sops, 2.5):.0f}, {np.percentile(sops, 97.5):.0f}]  "
              f"复刻={'OK' if repl_ok else ('跳过' if repl_ok is None else f'差异 {repl_diff:.2e}')}"
              f"  ({time.time()-t0:.0f}s)", flush=True)

    # ── 判决 ────────────────────────────────────────────────────────────
    rows.sort(key=lambda r: r["theta"])
    th = [r["theta"] for r in rows]
    sopK = [r["sop_k"] for r in rows]
    sop8 = [r["sop8_boot_mean"] for r in rows]
    inpK = [r["input_k"] for r in rows]
    hidK = [r["hidden_sum_k"] for r in rows]

    print("\n=== 判决表 ===", flush=True)
    print(f"  {'θ':<7}{'input':>9}{'隐藏合计':>10}{'SOP(K)':>14}{'SOP(8-batch)':>16}{'±std':>12}",
          flush=True)
    for r in rows:
        print(f"  {r['theta']:<7}{r['input_k']:>9.4f}{r['hidden_sum_k']:>10.4f}"
              f"{r['sop_k']:>14.0f}{r['sop8_boot_mean']:>16.0f}"
              f"{r['sop8_boot_std']:>12.0f}", flush=True)

    logn = np.log(np.array(sopK))
    mono_dec = bool(np.all(np.diff(logn) <= 1e-12))
    mono_inc = bool(np.all(np.diff(logn) >= -1e-12))
    iK = int(np.argmin(sopK))
    interior_K = 0 < iK < len(rows) - 1

    # 「若是纯噪声，8-batch 估计量给出内部 argmin 的概率」——直接把每个 θ 的
    # bootstrap 样本按行配对（第 b 次抽样在各 θ 下构成一次「假设的复测」），
    # 逐行取 argmin，统计落点分布。这是**非参数**的，不做正态假设。
    # 注意：各 θ 的 bootstrap 相互独立，故低估了跨 θ 的相关性 ⇒ 对「噪声」假设
    # 偏宽松（更容易判成 REAL），结论方向上保守。
    boot = np.stack([sops_by_theta[t] for t in th], axis=1)
    am = boot.argmin(axis=1)
    interior_frac = float(np.mean((am > 0) & (am < len(rows) - 1)))
    am_hist = {str(th[i]): float(np.mean(am == i)) for i in range(len(rows))}

    # 相邻比较的 z 值（8-batch 噪声尺度下的最小点显著性）
    z_vs_left = ((rows[iK - 1]["sop8_boot_mean"] - rows[iK]["sop8_boot_mean"])
                 / max(rows[iK]["sop8_boot_std"], 1.0)) if interior_K else None
    z_vs_right = ((rows[iK + 1]["sop8_boot_mean"] - rows[iK]["sop8_boot_mean"])
                  / max(rows[iK]["sop8_boot_std"], 1.0)) if interior_K else None

    print(f"\n  K={args.k_batches} 低噪曲线单调下降 : {'是' if mono_dec else '否'}"
          f"   单调上升 : {'是' if mono_inc else '否'}", flush=True)
    print(f"  K 曲线最小点 θ={th[iK]}  ({'内部' if interior_K else '端点'})", flush=True)
    print(f"  8-batch 估计量给出「内部最小点」的频率 : {interior_frac:.3f}"
          f"   （若 ≈{(len(rows)-2)/len(rows):.3f} 即与随机无异）", flush=True)
    print(f"  8-batch argmin 落点分布 : "
          + "  ".join(f"θ={k}:{v:.2f}" for k, v in am_hist.items()), flush=True)
    if interior_K:
        print(f"  最小点相对左右邻的 z 值 : 左 {z_vs_left:.2f}  右 {z_vs_right:.2f}", flush=True)

    if (not mono_dec) and interior_K:
        verdict = (f"REAL：K={args.k_batches} 的低噪曲线**仍非单调**且最小点在内部 "
                   f"θ={th[iK]} ⇒ 「输入层省、隐藏层增发」的对冲机制不是 8-batch 抽样噪声，"
                   f"可以作为贡献写入论文（并附本噪声分析作为稳健性证据）。")
    elif mono_dec:
        verdict = (f"NOISE：K={args.k_batches} 的低噪曲线**单调下降** ⇒ 11 号脚本在 "
                   f"n_batches=8 下看到的非单调是 256 样本的抽样噪声。"
                   f"**不得**把「θ 的净最小在中间」作为创新点写入论文；"
                   f"应改为如实报告 8-batch 标定的噪声量级。")
    else:
        verdict = (f"AMBIGUOUS：K 曲线非单调但最小点在端点 θ={th[iK]}，"
                   f"8-batch 给出内部最小点的频率 {interior_frac:.3f}。需增大 K 或改口径复核。")
    print(f"\n  ⇒ {verdict}", flush=True)

    out = {
        "snapshot": str(snap), "checkpoint_md5": md5, "t_dense": args.t_dense,
        "calib_fold": CALIB_FOLD, "calib_batch_size": CALIB_BATCH_SIZE,
        "k_batches": args.k_batches, "n_boot": args.n_boot, "seed": args.seed,
        "thetas": th, "rows": rows,
        "sop_k_monotone_decreasing": mono_dec,
        "sop_k_monotone_increasing": mono_inc,
        "argmin_sop_k_theta": th[iK],
        "argmin_sop_k_is_interior": bool(interior_K),
        "interior_argmin_freq_8batch": interior_frac,
        "argmin_hist_8batch": am_hist,
        "z_min_vs_left": z_vs_left, "z_min_vs_right": z_vs_right,
        "all_replicated_from_11": all(r["replicated_from_11"] for r in rows
                                      if r["replicated_from_11"] is not None),
        "verdict": verdict,
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n产物 → {args.out}", flush=True)


if __name__ == "__main__":
    main()
