#!/usr/bin/env python3
"""E12：`ab_polarity`（6 块 + 整流前端）的 θ 扫描 —— 补全 2×2 第四格。

只做推理，不训练。实现上复用：
  - `36_p0_spike_profile.py` 的 `run_dir()`（fold10 回退到原消融目录）；
  - `41_e5_theta_sweep_new_config.py` 的 preload / calibrate_preloaded /
    encoder_probe / bind_theta / clean（与 E5 逐字同口径，保证可比）。

用法::

    python3 experiments/46_ab_polarity_theta.py --folds 1,2,3 --gpu 0 --out results/g0.json
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="ab_polarity")
    ap.add_argument("--folds", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--theta", default="0.05,0.075,0.1,0.15,0.2,0.3")
    ap.add_argument("--t-dense", default="2,4,8,16,32")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--calib-batches", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "results" / "theta_sweep_ab_polarity.json"))
    args = ap.parse_args()

    thetas = [float(t) for t in args.theta.split(",") if t.strip()]
    tds = [int(t) for t in args.t_dense.split(",") if t.strip()]
    folds = [int(x) for x in args.folds.split(",") if x.strip()]
    dev = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(dev)
    out_path = Path(args.out)

    rows = []
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text())
            rows = prev.get("rows", []) if isinstance(prev, dict) else prev
        except Exception:
            rows = []

    def done(fold, theta, td):
        return any(r["fold"] == fold and abs(r["theta"] - theta) < 1e-12
                   and r["t_dense"] == td for r in rows)

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
        epoch = ckpt.get("epoch", -1)

        val_fold = 10 if fold == 1 else fold - 1
        test_ds = PTBXLCache(ROOT / "results" / "ptbxl_cache", fold_subset=[fold])
        calib_ds = PTBXLCache(ROOT / "results" / "ptbxl_cache", fold_subset=[val_fold])
        test_loader = DataLoader(test_ds, batch_size=args.batch, num_workers=2)
        calib_loader = DataLoader(calib_ds, batch_size=CALIB_BATCH_SIZE, num_workers=2)
        test_x, test_y = e5.preload(test_loader, dev)
        calib_x, _ = e5.preload(calib_loader, dev, args.calib_batches)

        probe_model = make_model(model_name, num_classes=5)
        probe_model.load_state_dict(state)
        probe_model.to(dev).eval()
        probe = e5.encoder_probe(probe_model, test_x[0], dev)
        n_blocks = len(probe_model.blocks)
        del probe_model
        torch.cuda.empty_cache()

        print(f"\n== {args.config} fold{fold} val_fold={val_fold} n_test={len(test_ds)} "
              f"n_blocks={n_blocks} md5={md5[:8]} probe_max|dθ|={probe['max_abs_diff']:.3e}",
              flush=True)
        t_fold = time.time()
        for theta in thetas:
            for td in tds:
                if done(fold, theta, td):
                    print(f"  θ={theta:g} T={td:<2} 已存在，跳过", flush=True)
                    continue
                model = make_model(model_name, num_classes=5)
                model.load_state_dict(state)
                model.T_dense = td
                model.to(dev).eval()
                e5.bind_theta(model, theta)

                y_true, y_score = [], []
                t0 = time.time()
                with torch.no_grad():
                    for x, y in zip(test_x, test_y):
                        y_true.append(y.cpu().numpy())
                        y_score.append(torch.sigmoid(model(x)).cpu().numpy())
                y_true = np.concatenate(y_true)
                y_score = np.concatenate(y_score)
                metrics = multilabel_metrics(y_true, y_score, ["NORM", "MI", "STTC", "CD", "HYP"])
                infer_s = time.time() - t0

                rates = e5.calibrate_preloaded(model, calib_x, args.calib_batches)
                cost = cost_snn(model, rates, T_in=1000)
                row = {
                    "config": args.config, "fold": fold, "run_dir": str(rd),
                    "checkpoint_md5": md5, "checkpoint_epoch": epoch, "val_fold": val_fold,
                    "theta": theta, "t_dense": td,
                    "macro_auroc": metrics["macro_auroc"], "macro_auprc": metrics["macro_auprc"],
                    "per_label": metrics["per_label"],
                    "params": cost.total_params, "total_macs": cost.total_macs,
                    "total_sops": cost.total_sops, "spike_rate_mean": cost.spike_rate_mean,
                    "spike_rates": rates,
                    "energy_pj_dense": cost.energy_pj_dense,
                    "energy_pj_neuromorphic": cost.energy_pj_neuromorphic,
                    "n_test": len(test_ds), "zero_shot": True,
                    "encoder_probe": probe, "inference_seconds": round(infer_s, 2),
                }
                rows = [r for r in rows if not (r["fold"] == fold
                        and abs(r["theta"] - theta) < 1e-12 and r["t_dense"] == td)]
                rows.append(row)
                out_path.write_text(json.dumps(e5.clean(
                    {"script": "experiments/46_ab_polarity_theta.py",
                     "config": args.config, "status": "running", "rows": rows}),
                    indent=2, ensure_ascii=False))
                print(f"  θ={theta:g} T={td:<2} AUROC={row['macro_auroc']:.4f} "
                      f"AUPRC={row['macro_auprc']:.4f} SOP={row['total_sops']/1e6:7.2f}M "
                      f"({infer_s:.1f}s)", flush=True)
                del model
                torch.cuda.empty_cache()
        print(f"  fold{fold} 用时 {time.time()-t_fold:.0f}s", flush=True)

    out_path.write_text(json.dumps(e5.clean(
        {"script": "experiments/46_ab_polarity_theta.py", "config": args.config,
         "folds": folds, "thetas": thetas, "t_dense": tds, "rows": rows}),
        indent=2, ensure_ascii=False))
    print(f"\n产物 → {out_path}", flush=True)


if __name__ == "__main__":
    main()
