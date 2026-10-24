#!/usr/bin/env python3
"""P0-4: MAC / SOP accounting for the compact SNN variants.

Why this exists
---------------
The paper's Pareto claim ("3.6x fewer parameters, +1.72pp accuracy") only
says something about *efficiency* if the compact model's spike-weighted
operation count does not regress.  That number had never been computed for
`ab_shallow_pol` (528K); every energy statement in the draft still used the
1.89M `ab_full` model.

Protocol is deliberately identical to `07_energy_report.py` so the numbers are
comparable with `results/comparison.json`:
  * same cache, same val fold (9) for spike-rate calibration, n_batches = 8
  * same constants: 3.7 pJ/MAC (dense FP32, Horowitz ISSCC 2014),
                    0.1 pJ/SOP (neuromorphic convention)
  * same latency measurement (`compute.time_inference`, B=1)

Writes results/p0_4_cost.json (a NEW file; nothing is overwritten).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

NC = Path("/mnt/ECG-SNN-LowPower/third_party/neurocardio")
sys.path.insert(0, str(NC))

from neurocardio.compute import (  # noqa: E402
    ENERGY_PJ_PER_MAC_FP32,
    ENERGY_PJ_PER_SOP,
    calibrate_spike_rates,
    cost_cnn1d,
    cost_snn,
    time_inference,
)
from neurocardio.dataset import PTBXLCache  # noqa: E402
from neurocardio.factory import make_model  # noqa: E402

DEFAULT_RUNS = ",".join([
    "ablation_full",                  # 6-block baseline, 1.89M
    "ablation_shallow_3blocks",       # 3 blocks, 1.0M-ish
    "ablation_encoder_polarity",      # polarity encoder, 6 blocks
    "repro_ab_shallow_pol_fold10",    # 3 blocks + polarity, 528K (the claim)
    "repro_cnn",
    "repro_resnet",
])


def is_snn(model) -> bool:
    """The S-TCN family (base + all ablations) exposes blocks/stem_conv."""
    return hasattr(model, "blocks") and hasattr(model, "stem_conv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/mnt/ECG-SNN-LowPower/results/ptbxl_cache")
    ap.add_argument("--runs-root", default="/mnt/ECG-SNN-LowPower/results")
    ap.add_argument("--runs", default=DEFAULT_RUNS)
    ap.add_argument("--out", default="/mnt/ECG-SNN-LowPower/results/p0_4_cost.json")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n-batches", type=int, default=8)
    args = ap.parse_args()

    dev = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(args.gpu)

    val = PTBXLCache(Path(args.cache_dir), fold_subset=[9])
    val_loader = DataLoader(val, batch_size=32, num_workers=2, pin_memory=True)
    print(f"校准集 fold 9: {len(val)} 条", flush=True)

    names = [n.strip() for n in args.runs.split(",") if n.strip()]
    rows, errors = [], {}

    for run_name in names:
        ckpt_path = Path(args.runs_root) / run_name / "best.pt"
        if not ckpt_path.exists():
            errors[run_name] = "best.pt not found"
            print(f"[SKIP] {run_name}: best.pt 不存在", flush=True)
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("cfg", {}) or {}
        labels = ckpt.get("label_columns") or ["NORM", "MI", "STTC", "CD", "HYP"]
        model_name = cfg.get("model_name") or run_name
        print(f"\n=== {run_name} ({model_name}) ===", flush=True)

        try:
            model = make_model(model_name, num_classes=len(labels))
            state = ckpt.get("ema") or ckpt.get("state_dict")
            model.load_state_dict(state)
            model.to(dev).eval()

            spike_rates = {}
            if is_snn(model):
                spike_rates = calibrate_spike_rates(model, val_loader, dev,
                                                    n_batches=args.n_batches)
                print("  发放率: " + ", ".join(f"{k}={v:.4f}" for k, v in spike_rates.items()),
                      flush=True)
                cost = cost_snn(model, spike_rates, T_in=1000)
            else:
                cost = cost_cnn1d(model, T_in=1000)

            lat = time_inference(model, dev, T_in=1000, n_warmup=5, n_iter=20)
            cost.latency_ms = lat
            cost.latency_device = dev.type

            rows.append({
                "run_name": run_name,
                "model": model_name,
                "class": type(model).__name__,
                "is_snn": is_snn(model),
                "params": int(cost.total_params),
                "macs": int(cost.total_macs),
                "sops": int(cost.total_sops),
                "sops_over_macs": (cost.total_sops / cost.total_macs
                                   if cost.total_macs else None),
                "spike_rate_mean": cost.spike_rate_mean,
                "spike_rates": spike_rates,
                "energy_pj_dense": float(cost.energy_pj_dense),
                "energy_pj_neuromorphic": float(cost.energy_pj_neuromorphic),
                "latency_ms_b1": lat,
                "layers": cost.layers,
            })
            print(f"  params={cost.total_params:,}  MACs={cost.total_macs:,}  "
                  f"SOPs={cost.total_sops:,}  SOP/MAC={cost.total_sops / cost.total_macs:.4f}",
                  flush=True)
            print(f"  稠密={cost.energy_pj_dense / 1e9:.4f} mJ  "
                  f"神经形态={cost.energy_pj_neuromorphic / 1e9:.6f} mJ  "
                  f"延迟={lat:.2f} ms", flush=True)
        except Exception as exc:  # noqa: BLE001
            errors[run_name] = f"{type(exc).__name__}: {exc}"
            print(f"[FAIL] {run_name}: {errors[run_name]}", flush=True)

    # ----- relative-to-baseline view -----
    base = next((r for r in rows if r["run_name"] == "ablation_full"), None)
    if base:
        for r in rows:
            r["vs_ab_full"] = {
                "param_ratio": r["params"] / base["params"],
                "mac_ratio": r["macs"] / base["macs"],
                "sop_ratio": (r["sops"] / base["sops"]) if base["sops"] else None,
                "neuro_energy_ratio": (r["energy_pj_neuromorphic"]
                                       / base["energy_pj_neuromorphic"]),
            }

    out = {
        "generated_at": time.time(),
        "note": "analytic accounting, NOT a hardware measurement; "
                "3.7 pJ/MAC and 0.1 pJ/SOP are literature constants",
        "constants": {
            "ENERGY_PJ_PER_MAC_FP32": ENERGY_PJ_PER_MAC_FP32,
            "ENERGY_PJ_PER_SOP": ENERGY_PJ_PER_SOP,
        },
        "calibration_fold": 9,
        "rows": rows,
        "errors": errors,
    }
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n-> 已写出 {args.out}", flush=True)

    print("\n=== 汇总 ===", flush=True)
    hdr = f"{'run':<30}{'params':>10}{'MACs':>13}{'SOPs':>13}{'SOP/MAC':>9}{'稠密mJ':>11}{'神经mJ':>11}{'延迟ms':>9}"
    print(hdr, flush=True)
    for r in rows:
        print(f"{r['run_name']:<30}{r['params']:>10,}{r['macs']:>13,}{r['sops']:>13,}"
              f"{r['sops_over_macs']:>9.4f}{r['energy_pj_dense'] / 1e9:>11.4f}"
              f"{r['energy_pj_neuromorphic'] / 1e9:>11.6f}{r['latency_ms_b1']:>9.2f}",
              flush=True)
    if errors:
        print("\n失败项:", json.dumps(errors, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
