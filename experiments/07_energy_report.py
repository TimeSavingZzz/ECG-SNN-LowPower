#!/usr/bin/env python3
"""能耗 / 延迟 / 精度 对比报告（上游 ``modal_app/app.py::compare_models`` 的本地化版本）。

上游把这段逻辑写在 Modal 函数里（依赖 Modal Volume），这里去掉 Modal，
直接读本机 ``results/<run>/best.pt`` 与 ``results/ptbxl_cache``，产出同构的
``comparison.json``：

  每个 run → fold-10 测试集 macro-AUROC/AUPRC + 解析 MACs/SOPs +
  SNN 实测脉冲发放率 + batch-1 推理延迟 + ROC/PR 曲线。

能量口径（上游常量，见 compute.py）：
  - 稠密 FP32 乘累加 MAC : 3.7 pJ   （45nm CMOS 数字 ASIC）
  - 脉冲突触操作 SOP     : 0.1 pJ   （Loihi 类神经形态芯片，事件驱动）
  → 注意这是**按已发表单算子能耗常量折算的估算**，不是真实硅片焦耳实测。

用法::

    python3 experiments/07_energy_report.py --cache-dir results/ptbxl_cache \\
        --runs-root results --out results/comparison.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

NC = Path("/mnt/ECG-SNN-LowPower/third_party/neurocardio")
sys.path.insert(0, str(NC))

from neurocardio.compute import (  # noqa: E402
    ENERGY_PJ_PER_MAC_FP32, ENERGY_PJ_PER_SOP,
    calibrate_spike_rates, cost_cnn1d, cost_snn, time_inference,
)
from neurocardio.dataset import PTBXLCache  # noqa: E402
from neurocardio.factory import make_model  # noqa: E402
from neurocardio.train import multilabel_metrics  # noqa: E402


def _downsample(arr, n: int = 200):
    if len(arr) <= n:
        return arr.tolist()
    idx = np.round(np.linspace(0, len(arr) - 1, n)).astype(int)
    return arr[idx].round(4).tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/mnt/ECG-SNN-LowPower/results/ptbxl_cache")
    ap.add_argument("--runs-root", default="/mnt/ECG-SNN-LowPower/results")
    ap.add_argument("--out", default="/mnt/ECG-SNN-LowPower/results/comparison.json")
    ap.add_argument("--runs", default="", help="逗号分隔的 run 名；留空则自动发现")
    ap.add_argument("--skip", default="smoke", help="自动发现时跳过的名字（子串匹配）")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    runs_root = Path(args.runs_root)

    if args.runs:
        names = [n.strip() for n in args.runs.split(",") if n.strip()]
    else:
        names = sorted(
            d.name for d in runs_root.iterdir()
            if (d / "best.pt").exists() and args.skip not in d.name
        )
    if not names:
        raise SystemExit("没有找到任何含 best.pt 的 run")
    print(f"对比 run: {names}", flush=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test = PTBXLCache(cache, fold_subset=[10])
    loader = DataLoader(test, batch_size=args.batch_size, num_workers=4, pin_memory=True)
    val = PTBXLCache(cache, fold_subset=[9])
    val_loader = DataLoader(val, batch_size=32, num_workers=2, pin_memory=True)
    print(f"测试集 {len(test)} 条 / 验证集 {len(val)} 条（fold 10 / fold 9）", flush=True)

    from sklearn.metrics import precision_recall_curve, roc_curve

    rows = []
    for run_name in names:
        ckpt_path = runs_root / run_name / "best.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("cfg", {}) or {}
        labels = ckpt.get("label_columns") or ["NORM", "MI", "STTC", "CD", "HYP"]
        model_name = cfg.get("model_name") or "snn"
        print(f"\n=== {run_name} ({model_name}) ===", flush=True)

        model = make_model(model_name, num_classes=len(labels))
        state = ckpt.get("ema") or ckpt.get("state_dict")
        model.load_state_dict(state)
        model.to(dev).eval()

        all_y, all_p = [], []
        t0 = time.time()
        with torch.no_grad():
            for x, y, _ in loader:
                x, y = x.to(dev), y.to(dev)
                all_y.append(y.cpu().numpy())
                all_p.append(torch.sigmoid(model(x)).cpu().numpy())
        infer_total = time.time() - t0
        y_true = np.concatenate(all_y)
        y_score = np.concatenate(all_p)
        metrics = multilabel_metrics(y_true, y_score, labels)
        print(f"  test macro-AUROC={metrics['macro_auroc']:.4f}  "
              f"macro-AUPRC={metrics['macro_auprc']:.4f}", flush=True)

        spike_rates = {}
        if model_name == "snn":
            spike_rates = calibrate_spike_rates(model, val_loader, dev, n_batches=8)
            print("  实测脉冲发放率: " + ", ".join(
                f"{k}={v:.3f}" for k, v in spike_rates.items()), flush=True)
            cost = cost_snn(model, spike_rates, T_in=1000)
        else:
            cost = cost_cnn1d(model, T_in=1000)

        lat = time_inference(model, dev, T_in=1000, n_warmup=5, n_iter=20)
        cost.latency_ms = lat
        cost.latency_device = dev.type
        print(f"  latency: {lat:.2f} ms/inf (B=1, {dev.type})", flush=True)
        print(f"  params={cost.total_params:,}  MACs={cost.total_macs:,}  "
              f"SOPs={cost.total_sops:,}", flush=True)
        print(f"  估算能耗: 稠密={cost.energy_pj_dense / 1e9:.6f} mJ  "
              f"神经形态={cost.energy_pj_neuromorphic / 1e9:.6f} mJ", flush=True)

        roc_curves, pr_curves = {}, {}
        for j, lbl in enumerate(labels):
            col = y_true[:, j].astype(int)
            if col.sum() == 0 or col.sum() == len(col):
                continue
            fpr, tpr, _ = roc_curve(col, y_score[:, j])
            prec, rec, _ = precision_recall_curve(col, y_score[:, j])
            roc_curves[lbl] = {"fpr": _downsample(fpr), "tpr": _downsample(tpr)}
            pr_curves[lbl] = {"precision": _downsample(prec), "recall": _downsample(rec)}

        rows.append({
            "run_name": run_name,
            "model": model_name,
            "params": cost.total_params,
            "macs": cost.total_macs,
            "sops": cost.total_sops,
            "spike_rate_mean": cost.spike_rate_mean,
            "spike_rates": spike_rates,
            "latency_ms": cost.latency_ms,
            "latency_device": cost.latency_device,
            "energy_pj_dense": cost.energy_pj_dense,
            "energy_pj_neuromorphic": cost.energy_pj_neuromorphic,
            "macro_auroc": metrics["macro_auroc"],
            "macro_auprc": metrics["macro_auprc"],
            "per_label": metrics["per_label"],
            "labels": labels,
            "roc_curves": roc_curves,
            "pr_curves": pr_curves,
            "test_inference_seconds": infer_total,
        })

    out = {
        "comparison_generated_at": time.time(),
        "rows": rows,
        "constants": {
            "ENERGY_PJ_PER_MAC_FP32": ENERGY_PJ_PER_MAC_FP32,
            "ENERGY_PJ_PER_SOP": ENERGY_PJ_PER_SOP,
        },
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n→ 已写出 {args.out}", flush=True)

    print("\n=== 汇总 ===", flush=True)
    print(f"{'run':<16}{'model':<8}{'params':>10}{'AUROC':>9}{'mJ(稠密)':>12}{'mJ(神经形态)':>14}{'延迟ms':>9}")
    for r in rows:
        print(f"{r['run_name']:<16}{r['model']:<8}{r['params']:>10,}"
              f"{r['macro_auroc']:>9.4f}{r['energy_pj_dense'] / 1e9:>12.4f}"
              f"{r['energy_pj_neuromorphic'] / 1e9:>14.6f}{r['latency_ms']:>9.2f}")


if __name__ == "__main__":
    main()
