#!/usr/bin/env python3
"""MIT-BIH 侧的 θ 扫描：在**第二个架构**上检验「总 SOP 对编码阈值非单调」是否复现。

## 为什么要做这个

PTB-XL 侧的帕累托扫描（`11_pareto_sweep.py`）发现一个反直觉现象：

    θ↑ → **输入层**发放率单调下降（0.4314 → 0.2130），
         但**隐藏层**增发把省下来的量吃回去
         （θ=0.3 处 stem 省 1.18M SOP，却被 block3.conv2 的 0.1138→0.1290 抵消），
    ⇒ `total_sops` 对 θ **非单调**，净最小值出现在**中间** θ，而不是最大 θ。

文献调研（2026-09-24 三路 agent）的结论：这个**具体机制**未见任何已发表工作
报告——已发表的「能耗非单调」都是在讲**神经元发放阈值 V_th**（静默区/饱和区）
或**剪枝稀疏度**，不是**输入编码器阈值 θ**。

但审稿人必然会问一句：「这是不是只是你那套架构 + 那份权重的偶然？」
本脚本就是为挡住这一问：换到**第二个架构**（MIT-BIH 的 CardioSpikeSNN，
272k 参数，与 PTB-XL 的 1.89M 结构不同、数据模态也不同），
用**同一套零样本口径**（权重仍是 θ=0.15 训出来的，只换推理期编码阈值），
看非单调性是否重现。

## 口径一致性

上游 `archive/cardiospike_mitbih/src/evaluate.py::synaptic_op_estimate` 给出

    ops_SNN(layer) ≈ r · C_in · C_out · k · T'        （r = 该层实测发放率）
    ops_ANN(layer) =     C_in · C_out · k · T'

这正是 PTB-XL 侧 `neurocardio.compute.cost_snn` 的对应物。本脚本**直接 import 复用**
它，不自己重写，以免口径漂移。θ 的唯一生效点是 `delta_encode_batch(xb, theta=theta)`。

## 用法

    CUDA_VISIBLE_DEVICES=3 python3 experiments/15_mitbih_theta_sweep.py \\
        --out results/mitbih/pareto | tee logs/mitbih_theta_sweep.log
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
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

NC = Path("/mnt/ECG-SNN-LowPower/third_party/neurocardio")
MITBIH = NC / "archive" / "cardiospike_mitbih"
sys.path.insert(0, str(NC))
sys.path.insert(0, str(MITBIH))
# evaluate.py 位于 MITBIH/src/ 下（顶层模块，不带包前缀），必须单独把 src 加进来
sys.path.insert(0, str(MITBIH / "src"))

from src.encoding import delta_encode_batch  # noqa: E402
from src.model import CardioSpikeSNN  # noqa: E402

import evaluate as up_eval  # noqa: E402  只复用其 synaptic_op_estimate / load_test

CLASSES = ["N", "S", "V", "F", "Q"]
LIF_KEYS = ("input", "s1", "s2", "s3", "s4")


def run_theta(model, X, y, R, dev, theta: float, bs: int = 512):
    """在给定 θ 下跑一遍全量 DS2，返回 (acc, macro_f1, spike_rates, sparsity, 秒数)。"""
    preds, probs = [], []
    sm = {k: 0.0 for k in LIF_KEYS}
    n_batches = 0
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = X[i:i + bs].to(dev)
            rb = R[i:i + bs].to(dev)
            sp = delta_encode_batch(xb, theta=theta)
            logits, tr = model(sp, rr=rb, return_traces=True)
            preds.append(logits.argmax(1).cpu())
            probs.append(F.softmax(logits, dim=1).cpu())
            sm["input"] += float(sp.mean().item())
            for k in ("s1", "s2", "s3", "s4"):
                sm[k] += float(tr[k].float().mean().item())
            n_batches += 1
    for k in sm:
        sm[k] /= n_batches

    preds_np = torch.cat(preds).numpy()
    acc = float(accuracy_score(y, preds_np))
    f1 = float(f1_score(y, preds_np, average="macro", zero_division=0))
    # 复用上游口径算逐层 SOP（T_input 在上游实现里未被使用，层 T' 是硬编码的）
    sparsity = up_eval.synaptic_op_estimate(model, sm, T_input=260)
    return acc, f1, sm, sparsity, time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",
                    default="/mnt/ECG-SNN-LowPower/results/mitbih/models/cardiospike_best.pt",
                    help="我们自己的 MIT-BIH 检查点（不读上游 archive/ 下的，避免污染）")
    ap.add_argument("--snapshot",
                    default="/mnt/ECG-SNN-LowPower/results/mitbih/pareto/_snapshot_best.pt")
    ap.add_argument("--out", default="/mnt/ECG-SNN-LowPower/results/mitbih/pareto")
    ap.add_argument("--theta", default="0.05,0.075,0.1,0.15,0.2,0.3",
                    help="与 11_pareto_sweep.py 保持一致，便于两侧横向对比")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--force", action="store_true", help="重算已存在的点")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    snap = Path(args.snapshot)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 冻结权重：训练/其他脚本可能覆盖，不冻结则各 θ 点跨权重不可比
    if args.force and snap.exists():
        snap.unlink()
    if not snap.exists():
        if not ckpt_path.exists():
            raise SystemExit(f"找不到 {ckpt_path}")
        shutil.copy2(ckpt_path, snap)
        print(f"已冻结检查点快照 → {snap}", flush=True)

    md5 = hashlib.md5(snap.read_bytes()).hexdigest()
    ckpt = torch.load(snap, map_location="cpu", weights_only=False)
    t_dense = ckpt["config"]["T_dense"]
    print(f"快照 {snap}\n  md5={md5}  T_dense={t_dense}（MIT-BIH 侧 T_dense 是训练期参数，"
          f"本扫描只动 θ）", flush=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CardioSpikeSNN(T_dense=t_dense).to(dev)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    Xnp, ynp, recnp, Rnp = up_eval.load_test()
    X = torch.from_numpy(Xnp).float()
    R = torch.from_numpy(Rnp).float()
    print(f"device={dev}  DS2 测试拍数={len(X)}  θ={args.theta}", flush=True)

    thetas = [float(t) for t in args.theta.split(",") if t.strip()]
    rows = []
    print("\n=== 逐 θ 扫描 ===", flush=True)
    for theta in thetas:
        dst = out_dir / f"theta{theta}.json"
        if dst.exists() and not args.force:
            row = json.loads(dst.read_text())
            rows.append(row)
            print(f"  θ={theta:<6} 已存在，跳过", flush=True)
            continue

        acc, f1, sm, sparsity, secs = run_theta(
            model, X, ynp, R, dev, theta, bs=args.batch_size)
        row = {
            "theta": theta,
            "n_test_beats": int(len(X)),
            "accuracy": acc,
            "macro_f1": f1,
            "spike_rates": sm,
            "total_snn_ops": sparsity["total_snn_ops"],
            "total_ann_ops": sparsity["total_ann_ops"],
            "energy_ratio_ann_over_snn": sparsity["energy_ratio_ann_over_snn"],
            "per_layer": sparsity["per_layer"],
            "checkpoint_md5": md5,
            "t_dense": t_dense,
            "seconds": round(secs, 1),
        }
        dst.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append(row)
        print(f"  θ={theta:<6} acc={acc:.6f}  macroF1={f1:.6f}  "
              f"input_rate={sm['input']:.4f}  SOP={sparsity['total_snn_ops']:.1f}  "
              f"ratio={sparsity['energy_ratio_ann_over_snn']:.2f}×  ({secs:.0f}s)", flush=True)

    # ── 非单调性判定 ──────────────────────────────────────────────────
    rows.sort(key=lambda r: r["theta"])
    print("\n=== 非单调性判定（PTB-XL 侧的现象是否在第二架构复现）===", flush=True)
    print(f"  {'θ':<7}{'acc':>9}{'input':>9}{'s1':>8}{'s2':>8}{'s3':>8}{'s4':>8}"
          f"{'total_SOP':>13}{'ratio':>8}", flush=True)
    for r in rows:
        sm = r["spike_rates"]
        print(f"  {r['theta']:<7}{r['accuracy']:>9.4f}{sm['input']:>9.4f}"
              f"{sm['s1']:>8.4f}{sm['s2']:>8.4f}{sm['s3']:>8.4f}{sm['s4']:>8.4f}"
              f"{r['total_snn_ops']:>13.1f}{r['energy_ratio_ann_over_snn']:>8.2f}",
              flush=True)

    sop = [r["total_snn_ops"] for r in rows]
    inp = [r["spike_rates"]["input"] for r in rows]
    hidden_sum = [sum(r["spike_rates"][k] for k in ("s1", "s2", "s3", "s4")) for r in rows]
    argmin_sop = rows[int(np.argmin(sop))]["theta"]

    print(f"\n  输入层发放率单调不增 : "
          f"{'是' if all(inp[i] >= inp[i+1] - 1e-9 for i in range(len(inp)-1)) else '否'}",
          flush=True)
    print(f"  隐藏层合计发放率单调增 : "
          f"{'是' if all(hidden_sum[i] <= hidden_sum[i+1] + 1e-9 for i in range(len(hidden_sum)-1)) else '否'}",
          flush=True)
    print(f"  total_SOP 最小值出现在 θ={argmin_sop}（最大 θ 是 {rows[-1]['theta']}）", flush=True)

    # 非单调 = 最小 SOP 不在端点
    is_nonmono = argmin_sop not in (rows[0]["theta"], rows[-1]["theta"])
    nonmono_hidden = all(hidden_sum[i] <= hidden_sum[i + 1] + 1e-9
                         for i in range(len(hidden_sum) - 1))
    if is_nonmono and nonmono_hidden:
        verdict = (f"REPRODUCED：总 SOP 非单调（最小在 θ={argmin_sop}，非端点），"
                   f"且隐藏层合计发放率随 θ 单调**上升** ⇒ 「输入层省、隐藏层增发」的"
                   f"对冲机制在第二个架构上复现。")
    elif is_nonmono:
        verdict = (f"PARTIAL：总 SOP 非单调（最小在 θ={argmin_sop}），"
                   f"但隐藏层合计发放率不是单调上升，机制细节与 PTB-XL 侧不同。")
    else:
        verdict = (f"NOT_REPRODUCED：总 SOP 单调，最小值在端点 θ={argmin_sop}。"
                   f"⇒ 非单调现象未在此架构重现，PTB-XL 侧发现的偶然性风险上升，"
                   f"论文中必须如实报告这一负面结果。")
    print(f"  ⇒ {verdict}", flush=True)

    summary = {
        "checkpoint_md5": md5,
        "t_dense": t_dense,
        "n_test_beats": int(len(X)),
        "thetas": [r["theta"] for r in rows],
        "rows": rows,
        "input_rate_monotone_decreasing": all(
            inp[i] >= inp[i + 1] - 1e-9 for i in range(len(inp) - 1)),
        "hidden_rate_monotone_increasing": nonmono_hidden,
        "argmin_total_sops_theta": argmin_sop,
        "is_non_monotonic": bool(is_nonmono),
        "verdict": verdict,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n产物 → {out_dir}/summary.json（+ 每个 θ 一个 json）", flush=True)


if __name__ == "__main__":
    main()
