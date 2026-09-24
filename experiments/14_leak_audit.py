#!/usr/bin/env python3
"""LIF 跨 batch 膜电位泄漏 —— 裁断实验。

## 为什么要有这个脚本

`results/comparison.json`（07 产出）里 SNN 的 `macro_auroc = 0.498530`，
而 `results/pareto/*.json`（11 产出）在同一个 θ=0.15/T_dense=8 基线点上是 **0.670474**。
这两条数字同时进了论文主图 fig1：左栏 (a) 画 0.4985，右栏 (b) 的基线点画 0.6705。
**同一模型同一数据集，两张子图差 0.17 —— 审稿人一眼能看出来。**

两个候选解释，预言相反，必须实测劈开：

  H1 权重漂移：07 的 comparison.json 生成时 `best.pt` 还是 ep0（=近似随机初始化，
     val macro-AUROC 0.5124），而 11 用的是 ep11 的权重。11 的注释里自己记了这笔账
     （「同一脚本两次运行因 best.pt 从 ep0 刷到 ep11，AUROC 得 0.4985 / 0.6705」）。
  H2 LIF 跨 batch 泄漏：上游 `model.py::_run_lif_along_time` 每个 batch 从
     `lif.init_leaky()` 起步，而 snntorch 的 `init_leaky()` 返回的是 **`self.mem`**，
     即上一 batch 最后一个时间步的膜电位（形状 (B,C)，与 batch_size 绑定，
     故只要 batch 划分不变就永远不触发 snntorch 内部的 shape-不匹配清零）。
     ⇒ 每个 batch 都继承上一个 batch 的末尾膜电位。

## 实验设计（用同一份冻结检查点，只改 LIF 状态管理）

  A1  bs=64 原生，第 1 次     ← 07/11 的实际路径
  A2  bs=64 原生，第 2 次     ← **同一 model 实例紧接着再跑一遍**
  B   bs=64 每 batch 前重置 LIF
  C   bs=96 原生
  D   bs=96 每 batch 前重置

判据：
  * A1 != A2  ⇒ **泄漏铁证**。同一权重、同一数据、同一配置，唯一变量是"上一轮
    推理遗留在 self.mem 里的末态"；结果不同就只能由泄漏解释。
  * A1 ≈ B    ⇒ 泄漏对精度的影响可忽略（0.4985 另有原因，即 H1 权重漂移）。
  * B ≈ D     ⇒ 干净路径下 batch 划分不影响 AUROC，可作为统一评测口径。

## 用法

    CUDA_VISIBLE_DEVICES=3 python3 experiments/14_leak_audit.py \\
        --out results/leak_audit.json | tee logs/leak_audit.log
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

NC = Path("/mnt/ECG-SNN-LowPower/third_party/neurocardio")
sys.path.insert(0, str(NC))

from neurocardio.dataset import PTBXLCache  # noqa: E402
from neurocardio.model import NeuroCardio  # noqa: E402
from neurocardio.train import multilabel_metrics  # noqa: E402


# ── LIF 状态管理 ──────────────────────────────────────────────────────
def count_lif(model) -> int:
    """数出模型里带持久膜电位的神经元个数（用于自检：应当 > 0）。"""
    return sum(1 for m in model.modules() if torch.is_tensor(getattr(m, "mem", None)))


def reset_lif(model) -> int:
    """把所有 LIF 的持久膜电位清零，让下一 batch 真正从头积分。

    为什么是置零而不是删掉属性：`init_leaky()` 返回 `self.mem`，而 snntorch 的
    `Leaky.forward` 在拿到 shape 不匹配的 mem 时会按 input 形状重新初始化。
    置成 `zeros_like` 与"首次调用"完全等效（膜电位从 0 起步），且不需要猜
    snntorch 内部初值的 shape（它在不同版本里是 `zeros(1)` 或 `zeros_like(input)`）。
    """
    n = 0
    for m in model.modules():
        mem = getattr(m, "mem", None)
        if torch.is_tensor(mem):
            m.mem = torch.zeros_like(mem)
            n += 1
    return n


# ── 推理 ──────────────────────────────────────────────────────────────
def infer(model, loader, dev, do_reset: bool):
    """fold-10 全量推理；返回 (y_true, y_score, 秒数)。

    `do_reset=True` 时在每个 batch 前清 LIF 膜电位（干净路径）；
    否则走上游原生路径（每 batch 继承上一 batch 的末态）。
    """
    ys, ps = [], []
    t0 = time.time()
    with torch.no_grad():
        for x, y, _ in loader:
            if do_reset:
                reset_lif(model)
            x = x.to(dev)
            ys.append(y.numpy())
            ps.append(torch.sigmoid(model(x)).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps), time.time() - t0


def run_one(tag, model, ds, dev, bs, do_reset):
    loader = DataLoader(ds, batch_size=bs, num_workers=4, pin_memory=True)
    y_true, y_score, secs = infer(model, loader, dev, do_reset)
    m = multilabel_metrics(y_true, y_score, LABELS)
    row = {
        "tag": tag,
        "batch_size": bs,
        "reset_lif": bool(do_reset),
        "macro_auroc": float(m["macro_auroc"]),
        "macro_auprc": float(m["macro_auprc"]),
        "seconds": round(secs, 2),
        "y_score_checksum": float(np.round(y_score.sum(), 4)),
    }
    print(f"  {tag:<4} bs={bs:<3} reset={str(do_reset):<5} "
          f"AUROC={row['macro_auroc']:.6f}  AUPRC={row['macro_auprc']:.6f}  "
          f"Σp={row['y_score_checksum']:.4f}  ({secs:.1f}s)", flush=True)
    return row


LABELS = ["NORM", "MI", "STTC", "CD", "HYP"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/mnt/ECG-SNN-LowPower/results/ptbxl_cache")
    ap.add_argument("--ckpt", default="/mnt/ECG-SNN-LowPower/results/repro_snn/best.pt")
    ap.add_argument("--snapshot",
                    default="/mnt/ECG-SNN-LowPower/results/_snapshot_leak_audit.pt",
                    help="冻结快照；训练会持续刷新 best.pt，不冻结则 6 条路径跨权重不可比")
    ap.add_argument("--out", default="/mnt/ECG-SNN-LowPower/results/leak_audit.json")
    ap.add_argument("--t-dense", type=int, default=8, help="08=模型默认，也是 07/11 的基线")
    ap.add_argument("--theta", type=float, default=0.15, help="训练所用 θ，07/11 的基线")
    ap.add_argument("--refresh-snapshot", action="store_true", help="重建快照（丢弃旧的）")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    snap = Path(args.snapshot)
    if args.refresh_snapshot and snap.exists():
        snap.unlink()
    if not snap.exists():
        if not ckpt_path.exists():
            raise SystemExit(f"找不到 {ckpt_path}")
        snap.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ckpt_path, snap)
        print(f"已冻结检查点快照 → {snap}", flush=True)

    md5 = hashlib.md5(snap.read_bytes()).hexdigest()
    ckpt = torch.load(snap, map_location="cpu", weights_only=False)
    state = ckpt.get("ema") or ckpt.get("state_dict")
    epoch = ckpt.get("epoch", -1)
    best = ckpt.get("best_macro_auroc", None)
    print(f"快照 {snap}\n  epoch={epoch}  md5={md5}  "
          f"ckpt.best_macro_auroc={best}", flush=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev}  θ={args.theta}  T_dense={args.t_dense}", flush=True)

    ds = PTBXLCache(Path(args.cache_dir), fold_subset=[10])
    print(f"fold-10 测试集 {len(ds)} 条", flush=True)

    model = NeuroCardio(num_classes=len(LABELS), T_dense=args.t_dense)
    model.load_state_dict(state)
    model.to(dev).eval()
    n_lif = count_lif(model)
    print(f"模型内 LIF 神经元 {n_lif} 个（>0 才说明重置有作用）", flush=True)
    if n_lif == 0:
        raise SystemExit("没有找到任何 mem 属性 —— reset_lif 会静默失效，中止")

    print("\n=== 逐路径评测 ===", flush=True)
    rows = []
    # A1/A2 必须连续、共用同一个 model 实例，且中间不得重置 —— 这是泄漏的判据
    rows.append(run_one("A1", model, ds, dev, 64, False))
    rows.append(run_one("A2", model, ds, dev, 64, False))
    rows.append(run_one("B", model, ds, dev, 64, True))
    rows.append(run_one("C", model, ds, dev, 96, False))
    rows.append(run_one("D", model, ds, dev, 96, True))

    # ── 判据 ──────────────────────────────────────────────────────────
    by = {r["tag"]: r for r in rows}
    d_a = abs(by["A1"]["macro_auroc"] - by["A2"]["macro_auroc"])
    d_ab = abs(by["A1"]["macro_auroc"] - by["B"]["macro_auroc"])
    d_bd = abs(by["B"]["macro_auroc"] - by["D"]["macro_auroc"])

    print("\n=== 判据 ===", flush=True)
    print(f"  |A1 − A2| = {d_a:.6f}   （同一 model 连跑两次；>0 即泄漏铁证）", flush=True)
    print(f"  |A1 − B | = {d_ab:.6f}   （原生 vs 重置；= 泄漏对精度的实际影响）", flush=True)
    print(f"  |B  − D | = {d_bd:.6f}   （干净路径下 bs 敏感性）", flush=True)

    if d_a > 1e-9:
        verdict = ("LEAK_CONFIRMED：同一权重、同一数据、同一配置连跑两次结果不同，"
                   f"唯一变量是 self.mem 里的残留末态（ΔAUROC={d_a:.6f}）⇒ LIF 跨 batch "
                   "泄漏真实存在，上游评测路径不干净。")
    elif d_ab > 1e-6:
        verdict = (f"LEAK_WEAK：连跑两次一致（A1==A2），但原生与重置差 {d_ab:.6f} "
                   "⇒ 泄漏存在但对精度影响有限。")
    else:
        verdict = ("NO_LEAK：A1==A2 且 A===B ⇒ 泄漏在本数据/本权重下不显著；"
                   "07 的 0.4985 应归因于权重漂移（H1）。")
    print(f"  ⇒ {verdict}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "checkpoint": str(snap),
        "checkpoint_md5": md5,
        "checkpoint_epoch": epoch,
        "theta": args.theta,
        "t_dense": args.t_dense,
        "n_lif_neurons": n_lif,
        "test_size": len(ds),
        "device": dev.type,
        "rows": rows,
        "delta_a1_a2": d_a,
        "delta_a1_b": d_ab,
        "delta_b_d": d_bd,
        "verdict": verdict,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n产物 → {out}", flush=True)


if __name__ == "__main__":
    main()
