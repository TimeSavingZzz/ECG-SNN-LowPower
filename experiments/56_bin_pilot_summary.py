#!/usr/bin/env python3
"""Summarize the 012 binary-frontend retrain pilot (E17).

Reads results/repro_ab_shallow_pol_bin*_fold10/ and writes
results/bin_pilot_summary.json.

Pilot recipe is identical to the ten-fold protocol except for theta_bin:
fold 10, 100 epochs, bs 96, lr 1e-3, EMA 0.999, neuron theta 0.15, seed 0.

Usage:
    python3 experiments/56_bin_pilot_summary.py
"""
import glob
import hashlib
import json
import os

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(PROJECT, "results")

rows = []
for d in sorted(glob.glob(os.path.join(ROOT, "repro_ab_shallow_pol_bin*_fold10"))):
    name = os.path.basename(d)
    with open(os.path.join(d, "test_metrics.json")) as f:
        tm = json.load(f)
    with open(os.path.join(d, "history.json")) as f:
        hist = json.load(f)["history"]
    best = max(hist, key=lambda e: e.get("val_macro_auroc") or -1)
    tl = [e["train_loss"] for e in hist if e.get("train_loss") is not None]
    with open(os.path.join(d, "best.pt"), "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    rows.append({
        "model": name.split("_fold10")[0],
        "theta_bin": int(name.split("bin")[1].split("_")[0]) / 100.0,
        "fold": 10,
        "epochs": len(hist),
        "best_ckpt_md5": md5,
        "test_macro_auroc": tm["macro_auroc"],
        "test_macro_auprc": tm["macro_auprc"],
        "best_val_auroc": best["val_macro_auroc"],
        "best_val_epoch": best["epoch"],
        "train_loss_first": tl[0],
        "train_loss_last": tl[-1],
        "per_label": {k: v["auroc"] for k, v in tm["per_label"].items()},
    })
rows.sort(key=lambda r: r["theta_bin"])

out = {
    "note": "012 binary-frontend retrain pilot (E17): fold 10, 100 epochs, 4 GPUs. "
            "Zero-shot counterpart in E16. Continuous reference is the existing "
            "fold-10 ab_shallow_pol run read from its test_metrics.json, i.e. the "
            "same eval protocol as the pilot runs (0.8796). E16 lists 0.8791 for the "
            "same checkpoint under the probe protocol (T_dense=8, batch 64); the "
            "0.04 pp difference is protocol noise but the two must not be mixed.",
    "continuous_reference": {
        "model": "ab_shallow_pol",
        "fold": 10,
        "test_macro_auroc": 0.879566,
        "source": "results/repro_ab_shallow_pol_fold10/test_metrics.json",
        "probe_protocol_value": 0.879148,
    },
    "runs": rows,
}
path = os.path.join(ROOT, "bin_pilot_summary.json")
with open(path, "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

for r in rows:
    print("%-34s theta=%.2f  AUROC=%.4f  AUPRC=%.4f  val=%.4f@%d  md5=%s"
          % (r["model"], r["theta_bin"], r["test_macro_auroc"], r["test_macro_auprc"],
             r["best_val_auroc"], r["best_val_epoch"], r["best_ckpt_md5"][:8]))
print("n_runs=%d  wrote %s" % (len(rows), path))
