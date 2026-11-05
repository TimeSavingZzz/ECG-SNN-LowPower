#!/usr/bin/env python3
"""二值化前端探针（零训练）：把整流前端换成「阈值二值化」后，现成权重还能剩多少精度？

动机
----
`ab_polarity` / `ab_shallow_pol` 的前端是 Sarhaddi 的连续整流编码
（per-lead z-score → ReLU(z) 与 ReLU(−z)），输出是**连续值**，不是 0/1 spike。
所以这两个配置严格说不是「脉冲输入」，`cost_snn` 对 stem 的计费也需要修正
（见批次 010 的 E9）。

本脚本不训练、不改仓库代码，只在推理时把前端换成

    spike⁺ = 1[z ≥ θ],  spike⁻ = 1[z ≤ −θ]

其中 z 是 per-lead 逐样本 z-score，θ 以 z 为单位。θ 越大越稀疏。
得到的是「零样本二值化的代价」，用于判断重训是否值得，而不是重训后的上限。

用法::

    python3 experiments/53_binary_frontend_probe.py --config ab_shallow_pol --folds 9,10 --gpu 0
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path("/mnt/ECG-SNN-LowPower")
sys.path.insert(0, str(ROOT / "third_party" / "neurocardio"))

from neurocardio.compute import cost_snn  # noqa: E402
from neurocardio.dataset import PTBXLCache  # noqa: E402
from neurocardio.factory import make_model  # noqa: E402
from neurocardio.train import multilabel_metrics  # noqa: E402

CALIB_BATCH_SIZE = 32


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


p0 = _load("p0_spike_profile", ROOT / "experiments" / "36_p0_spike_profile.py")
e5 = _load("e5_sweep", ROOT / "experiments" / "41_e5_theta_sweep_new_config.py")


def dual_polarity_binary_encode(x: torch.Tensor, theta: float = 1.0,
                                eps: float = 1e-5) -> torch.Tensor:
    """per-lead z-score → 双极性二值 spike（θ 以 z 为单位）。"""
    mu = x.mean(dim=-1, keepdim=True)
    sd = x.std(dim=-1, keepdim=True).clamp_min(eps)
    z = (x - mu) / sd
    return torch.cat([(z >= theta).float(), (z <= -theta).float()], dim=1)


def make_binary_encode(theta: float):
    def enc(x, theta=theta, eps=1e-5):
        return dual_polarity_binary_encode(x, theta=theta, eps=eps)
    return enc


def evaluate(model, test_x, test_y, calib_x, n_calib):
    y_true, y_score = [], []
    with torch.no_grad():
        for x, y in zip(test_x, test_y):
            y_true.append(y.cpu().numpy())
            y_score.append(torch.sigmoid(model(x)).cpu().numpy())
    y_true = np.concatenate(y_true)
    y_score = np.concatenate(y_score)
    m = multilabel_metrics(y_true, y_score, ["NORM", "MI", "STTC", "CD", "HYP"])
    rates = e5.calibrate_preloaded(model, calib_x, n_calib)
    cost = cost_snn(model, rates, T_in=1000)
    return m, rates, cost


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="ab_shallow_pol")
    ap.add_argument("--folds", default="9,10")
    ap.add_argument("--theta-bin", default="0.15,0.5,1.0,1.5,2.0")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--calib-batches", type=int, default=8)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    tbins = [float(t) for t in args.theta_bin.split(",") if t.strip()]
    folds = [int(x) for x in args.folds.split(",") if x.strip()]
    dev = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(dev)
    out_path = Path(args.out) if args.out else (
        ROOT / "results" / f"bin_probe_{args.config}.json")

    rows = []
    for fold in folds:
        rd = p0.run_dir(args.config, fold)
        ckpt_path = rd / "best.pt"
        if not ckpt_path.exists():
            print(f"[SKIP] {args.config} fold{fold}: {ckpt_path} 不存在", flush=True)
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("cfg", {}) or {}
        model_name = cfg.get("model_name") or args.config
        state = ckpt.get("ema") or ckpt["state_dict"]
        md5 = hashlib.md5(ckpt_path.read_bytes()).hexdigest()

        val_fold = 10 if fold == 1 else fold - 1
        test_ds = PTBXLCache(ROOT / "results" / "ptbxl_cache", fold_subset=[fold])
        calib_ds = PTBXLCache(ROOT / "results" / "ptbxl_cache", fold_subset=[val_fold])
        test_x, test_y = e5.preload(DataLoader(test_ds, batch_size=args.batch,
                                               num_workers=2), dev)
        calib_x, _ = e5.preload(DataLoader(calib_ds, batch_size=CALIB_BATCH_SIZE,
                                           num_workers=2), dev, args.calib_batches)
        print(f"\n== {args.config} fold{fold} val_fold={val_fold} n_test={len(test_ds)} "
              f"md5={md5[:8]}", flush=True)

        def fresh():
            m = make_model(model_name, num_classes=5)
            m.load_state_dict(state)
            m.T_dense = 8
            return m.to(dev).eval()

        # 0) 连续整流基线（原样）
        model = fresh()
        e5.bind_theta(model, 0.15)   # 连续编码忽略 theta，纯为与既有产物同口径
        t0 = time.time()
        m, rates, cost = evaluate(model, test_x, test_y, calib_x, args.calib_batches)
        del model
        torch.cuda.empty_cache()
        rows.append({
            "config": args.config, "fold": fold, "checkpoint_md5": md5,
            "frontend": "continuous_rectified", "theta_bin": None,
            "macro_auroc": m["macro_auroc"], "macro_auprc": m["macro_auprc"],
            "per_label": m["per_label"],
            "input_rate": rates["input"], "stem_rate": rates["stem"],
            "spike_rates": rates,
            "total_macs": cost.total_macs, "total_sops": cost.total_sops,
            "energy_pj_neuromorphic": cost.energy_pj_neuromorphic,
            "energy_pj_dense": cost.energy_pj_dense,
            "inference_seconds": round(time.time() - t0, 2),
        })
        print(f"  连续   AUROC={m['macro_auroc']:.4f} AUPRC={m['macro_auprc']:.4f} "
              f"input_rate={rates['input']:.4f} SOP={cost.total_sops/1e6:.2f}M", flush=True)

        # 1) 二值化前端（零样本）
        for tb in tbins:
            model = fresh()
            model.delta_encode = make_binary_encode(tb)
            e5.bind_theta(model, tb)
            t0 = time.time()
            m, rates, cost = evaluate(model, test_x, test_y, calib_x, args.calib_batches)
            del model
            torch.cuda.empty_cache()
            rows.append({
                "config": args.config, "fold": fold, "checkpoint_md5": md5,
                "frontend": "binary_rectified", "theta_bin": tb,
                "macro_auroc": m["macro_auroc"], "macro_auprc": m["macro_auprc"],
                "per_label": m["per_label"],
                "input_rate": rates["input"], "stem_rate": rates["stem"],
                "spike_rates": rates,
                "total_macs": cost.total_macs, "total_sops": cost.total_sops,
                "energy_pj_neuromorphic": cost.energy_pj_neuromorphic,
                "energy_pj_dense": cost.energy_pj_dense,
                "inference_seconds": round(time.time() - t0, 2),
            })
            print(f"  二值θ={tb:<4g} AUROC={m['macro_auroc']:.4f} "
                  f"AUPRC={m['macro_auprc']:.4f} input_rate={rates['input']:.4f} "
                  f"SOP={cost.total_sops/1e6:.2f}M", flush=True)

    out_path.write_text(json.dumps({
        "script": "experiments/53_binary_frontend_probe.py",
        "config": args.config, "folds": folds, "theta_bin": tbins,
        "protocol": ("零样本：不改权重、不重训；T_dense=8；batch=64；标定同 46 号脚本"),
        "rows": rows,
    }, indent=2, ensure_ascii=False))
    print(f"\n产物 → {out_path}", flush=True)


if __name__ == "__main__":
    main()
