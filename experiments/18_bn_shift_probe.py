#!/usr/bin/env python3
"""18_bn_shift_probe.py — 冻结的 BatchNorm 统计量失配，是不是「深层增发」的原因？

**前情（两个已被实测确立的事实）**

1. `16_calib_noise_floor.py`：θ 偏离训练值 0.15 时，隐藏层发放率**非单调**——stem /
   block2 / block3 在大 θ 下**反而增发**，而 block0 / block1 减发（符号随深度翻转）。
   θ=0.3 相对 0.15：input −0.1020，stem +0.0220，block2 +0.0160，block3 +0.0138。
   总 SOP 因此非单调（内部最小 θ=0.2，右侧抬升 8.84σ）。**这是候选的核心创新点。**

2. `17_burstiness_probe.py`：我曾假设「大 θ 下事件更成簇、LIF 积分放大」——
   **已被自己推翻**：平均游程随 θ 单调**下降**（5.80→2.64），lag-1 自相关也下降。
   事件在大 θ 下更**孤立**，不是更成簇。机制另有原因。

**本脚本检验的假设：冻结的 BN 统计量失配**

`NeuroCardio` 在 stem 和每个 S-TCN block 里都插了 `BatchNorm1d`（`stem_bn`、
`blocks[j].bn1/bn2`，另有 `fc1_bn`、`score_bn`）。**eval 模式下 BN 用的是训练期
（θ=0.15）累积的 running_mean/running_var，是冻结常量**。而

    post_bn = γ · (pre − run_mean) / sqrt(run_var + ε) + β

⇒ 若 θ≠0.15 让 *pre*（卷积输出）的均值整体平移，这个平移会被 BN **原样传递**到 LIF 的
净输入上（γ、run_mean、run_var、β 都是常量），从而直接改变该层发放率。这正好解释
「为什么 input 事件变少、深层却增发」——**不是脉冲时序的功劳，而是输入分布平移穿过
冻结归一化层后的净驱动**。

**判据**

对每一层，比较 Δ(post_bn 均值) 与 Δ(该层 LIF 发放率) 的**符号**（均相对 θ=0.15）：

  · 升发层（stem / block2 / block3）的 post_bn 均值也升，降发层也降 ⇒ 符号一致率高
    ⇒ **MECHANISM_SUPPORTED**：机制是「分布平移 × 冻结 BN」，可以写进论文。
  · 符号对不上 ⇒ **MECHANISM_REJECTED**：只把逐层反转作为经验现象报告。

同时记录「平移了多少个 running std」（`shift_in_run_std_units`），这是 BN 失配的
标准度量，可与文献里的 covariate-shift 讨论对齐。

用法（容器内）::

    CUDA_VISIBLE_DEVICES=3 python experiments/18_bn_shift_probe.py \\
        --out results/pareto/bn_shift_probe.json | tee logs/bn_shift_probe.log
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

NC = Path("/mnt/ECG-SNN-LowPower/third_party/neurocardio")
sys.path.insert(0, str(NC))

from neurocardio.dataset import PTBXLCache  # noqa: E402
from neurocardio.model import NeuroCardio  # noqa: E402

CALIB_BATCH_SIZE = 32   # 与 11 / 16 / 17 对齐
CALIB_FOLD = 9
TRAIN_THETA = 0.15      # running stats 的来处
N_BLOCKS = 6


def bind_theta(model, theta: float):
    """与 11/16 同一手法：把 θ 固定到实例的 forward 上。"""
    orig = model.forward

    def bound(x, *a, **kw):
        kw.setdefault("theta", theta)
        return orig(x, *a, **kw)

    model.forward = bound
    return model


def hooked_layers(model):
    """返回 [(层名, 该层的 BN 模块, 与之对应的发放率键)]。

    发放率键必须与 `tr["stem"]` / `tr["blocks"][j]` 对上：
      · stem 的 LIF 输出 = tr["stem"]，其驱动是 stem_bn 的输出
      · block j 的输出 = tr["blocks"][j]，其驱动主体是 blocks[j].bn2 的输出
    """
    pairs = [("stem", model.stem_bn, "stem")]
    for j in range(N_BLOCKS):
        blk = model.blocks[j]
        pairs.append((f"block{j}", blk.bn2, f"block{j}"))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/mnt/ECG-SNN-LowPower/results/ptbxl_cache")
    ap.add_argument("--snapshot", default="/mnt/ECG-SNN-LowPower/results/pareto/_snapshot_best.pt")
    ap.add_argument("--noise-floor", default="/mnt/ECG-SNN-LowPower/results/pareto/calib_noise_floor.json")
    ap.add_argument("--out", default="/mnt/ECG-SNN-LowPower/results/pareto/bn_shift_probe.json")
    ap.add_argument("--theta", default="0.05,0.075,0.1,0.15,0.2,0.3")
    ap.add_argument("--t-dense", type=int, default=8)
    ap.add_argument("--k-batches", type=int, default=32)
    args = ap.parse_args()

    thetas = [float(t) for t in args.theta.split(",") if t.strip()]
    snap = Path(args.snapshot)
    if not snap.exists():
        raise SystemExit(f"找不到冻结快照 {snap}（先跑 11_pareto_sweep.py）")
    md5 = hashlib.md5(snap.read_bytes()).hexdigest()
    ckpt = torch.load(snap, map_location="cpu", weights_only=False)
    labels = ckpt.get("label_columns") or ["NORM", "MI", "STTC", "CD", "HYP"]
    state = ckpt.get("ema") or ckpt.get("state_dict")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuroCardio(num_classes=len(labels), T_dense=args.t_dense)
    model.load_state_dict(state)
    model.to(dev)
    model.eval()

    val = PTBXLCache(Path(args.cache_dir), fold_subset=[CALIB_FOLD])
    loader = DataLoader(val, batch_size=CALIB_BATCH_SIZE, num_workers=2)

    pairs = hooked_layers(model)
    # running stats（冻结常量）——把平移换算成「几个 running std」
    run_stats = {name: {"run_mean": float(bn.running_mean.mean()),
                        "run_std": float(bn.running_var.sqrt().mean())}
                 for name, bn, _ in pairs}
    print(f"快照 md5={md5}  T_dense={args.t_dense}  θ={thetas}")
    print(f"device={dev}  标定集=fold{CALIB_FOLD}  K={args.k_batches} batch", flush=True)
    print("\n冻结的 running stats（训练期 θ=0.15 累积）:")
    for name, st in run_stats.items():
        print(f"  {name:<9} run_mean={st['run_mean']:+.4f}  run_std={st['run_std']:.4f}")
    assert abs(TRAIN_THETA - 0.15) < 1e-12

    orig_forward = model.forward
    rows = []
    print(f"\n{'θ':<7}" + "".join(f"{n:>12}" for n in
                                  [p[0] + ".bn_out" for p in pairs]), flush=True)
    for theta in thetas:
        # 挂钩子收集 post-BN 输出的均值
        acc: dict[str, list[float]] = {name: [] for name, _, _ in pairs}

        def mk(name):
            def hook(_mod, _inp, out):
                acc[name].append(float(out.float().mean()))
            return hook

        handles = [bn.register_forward_hook(mk(name)) for name, bn, _ in pairs]

        fires: dict[str, list[float]] = {name: [] for name, _, _ in pairs}
        m = bind_theta(model, theta)
        n = 0
        try:
            with torch.no_grad():
                for i, (x, _y, _) in enumerate(loader):
                    if i >= args.k_batches:
                        break
                    x = x.to(dev)
                    _logits, tr = m(x, return_traces=True)
                    fires["stem"].append(float(tr["stem"].float().mean()))
                    for j, s in enumerate(tr["blocks"]):
                        fires[f"block{j}"].append(float(s.float().mean()))
                    n += 1
        finally:
            model.forward = orig_forward
            for h in handles:
                h.remove()

        row = {"theta": theta, "layers": {}}
        for name, bn, key in pairs:
            post = float(np.mean(acc[name]))
            fire = float(np.mean(fires[key]))
            rm, rs = run_stats[name]["run_mean"], run_stats[name]["run_std"]
            row["layers"][name] = {
                "post_bn_mean": post,
                "firing_rate": fire,
                "shift_vs_run_mean": post - rm,
                "shift_in_run_std_units": (post - rm) / rs if rs > 0 else None,
            }
        rows.append(row)
        print(f"{theta:<7}" + "".join(f"{row['layers'][p[0]]['post_bn_mean']:>12.5f}"
                                      for p in pairs), flush=True)

    # ── 判据：Δ(post_bn) 与 Δ(firing) 的符号一致性（均相对 θ=0.15）──────
    base = [r for r in rows if abs(r["theta"] - TRAIN_THETA) < 1e-12]
    if not base:
        raise SystemExit(f"θ 列表里必须含训练阈值 {TRAIN_THETA} 作为基准")
    base = base[0]["layers"]

    print("\n=== Δ 相对 θ=0.15：BN 输出均值漂移 vs 该层发放率变化 ===")
    print(f"  {'层':<9}{'Δpost_bn':>12}{'Δ发放率':>12}{'符号一致':>10}"
          f"{'平移(running_std)':>20}")
    agree, total = 0, 0
    detail = {}
    for name, _bn, _key in pairs:
        dps, dfs = [], []
        for r in rows:
            L = r["layers"][name]
            if abs(r["theta"] - TRAIN_THETA) < 1e-12:
                continue
            dps.append(L["post_bn_mean"] - base[name]["post_bn_mean"])
            dfs.append(L["firing_rate"] - base[name]["firing_rate"])
        same = sum(1 for a, b in zip(dps, dfs)
                   if (a > 0 and b > 0) or (a < 0 and b < 0))
        agree += same
        total += len(dps)
        detail[name] = {
            "delta_post_bn_mean": dps, "delta_firing_rate": dfs,
            "sign_agree": same, "n": len(dps),
            "shift_in_run_std_units_at_0.3": (
                rows[-1]["layers"][name]["shift_in_run_std_units"]),
        }
        print(f"  {name:<9}{np.mean(dps):>+12.5f}{np.mean(dfs):>+12.5f}"
              f"{same}/{len(dps):>7}{rows[-1]['layers'][name]['shift_in_run_std_units']:>20.4f}")

    rate = agree / max(total, 1)
    print(f"\n  符号一致率 : {agree}/{total} = {rate:.3f}"
          f"   （随机基线 0.5）", flush=True)

    # 关键局部检验：升发三层的 post_bn 是否也升
    rising = ["stem", "block2", "block3"]
    rising_ok = []
    for name in rising:
        L03 = rows[-1]["layers"][name]
        rising_ok.append(L03["post_bn_mean"] > base[name]["post_bn_mean"])
    print(f"  「升发三层」(stem/block2/block3) 在 θ=0.3 的 post_bn 均值也上升 : "
          f"{sum(rising_ok)}/{len(rising)}", flush=True)

    if rate >= 0.75 and all(rising_ok):
        verdict = ("MECHANISM_SUPPORTED：Δ(BN 输出均值) 与 Δ(发放率) 的符号在三层升发层上"
                   "全部一致、整体一致率远超随机 ⇒ 深层增发的机制是"
                   "「θ 改变输入分布 → 平移穿过**冻结**的 BatchNorm 统计量 → 变成 LIF 的"
                   "净驱动」。可写成：推理期 θ 调节会与训练期归一化统计量失配，"
                   "这是零样本 θ 迁移的固有代价。")
    elif all(rising_ok):
        verdict = ("MECHANISM_PARTIAL：升发层的 BN 输出均值确实同向上升，但降发层符号对不齐。"
                   "BN 失配能解释增发，不足以解释全部层的变化。")
    else:
        verdict = ("MECHANISM_REJECTED：升发层的 BN 输出均值并未同向上升 ⇒ 冻结 BN 失配"
                   "不是主因。论文中只把逐层反转作为**经验现象**报告，不附会机制。")
    print(f"\n  ⇒ {verdict}", flush=True)

    out = {
        "snapshot": str(snap), "checkpoint_md5": md5, "t_dense": args.t_dense,
        "train_theta": TRAIN_THETA, "fold": CALIB_FOLD,
        "batch_size": CALIB_BATCH_SIZE, "k_batches": args.k_batches,
        "thetas": [r["theta"] for r in rows],
        "rows": rows, "running_stats": run_stats,
        "delta_detail": detail,
        "sign_agree_rate": rate,
        "rising_layers_bn_shift_ok": bool(all(rising_ok)),
        "verdict": verdict,
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n产物 → {args.out}", flush=True)


if __name__ == "__main__":
    main()
