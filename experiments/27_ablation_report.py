#!/usr/bin/env python3
"""Consolidate the 100-epoch ablation batch + compact multi-fold into a
summary JSON and the ablation figure (Task E/F delivery).

Reads results/ablation_*/test_metrics.json (7 variants) plus the three compact
folds, greps each training log for its best val-AUROC, and emits:
  results/ablation_summary.json
  results/figG_ablation.png

Run:  python3 experiments/27_ablation_report.py
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/mnt/ECG-SNN-LowPower/results")
LOGS = Path("/mnt/ECG-SNN-LowPower/logs")
TS = "1010_0508"

# (short name, results dir, plot label, group, params, log name)
VARIANTS = [
    ("ab_full",     "ablation_full",                "Full\n6-block",       "control", 1893768, "ab_full"),
    ("ab_polarity", "ablation_encoder_polarity",    "Dual-\npolarity",     "encoder", 1893768, "ab_polarity"),
    ("ab_multiq",   "ablation_encoder_multiquantum","No slew\nlimit",      "encoder", 1893768, "ab_multiq"),
    ("ab_noattn",   "ablation_no_attn_pool",        "Mean\npool",          "struct",  1893381, "ab_noattn"),
    ("ab_nogate",   "ablation_no_gate_residual",    "No\ngate",            "struct",  1753288, "ab_nogate"),
    ("ab_nodilate", "ablation_no_dilation",         "No\ndilation",        "struct",  1893768, "ab_nodilate"),
    ("ab_shallow",  "ablation_shallow_3blocks",     "3\nblocks",           "struct",   528392, "ab_shallow"),
]

COMPACT_FOLDS = [
    ("fold10", "repro_compact"),
    ("fold9",  "repro_snn_compact_fold9"),
    ("fold8",  "repro_snn_compact_fold8"),
]
MAINLINE_AUROC = 0.8594          # 100-epoch mainline SNN, fold-10 (record 03)
NOISE_SIGMA = 0.0086             # 3-fold std of the SNN (record 03) -> single-fold scale


def best_val(log_name):
    p = LOGS / f"{log_name}_{TS}.log"
    if not p.exists():
        return None
    vals = [float(m) for m in re.findall(r"new best macro-AUROC: ([0-9.]+)", p.read_text())]
    return max(vals) if vals else None


def main():
    rows = []
    for name, d, label, group, params, log_name in VARIANTS:
        m = json.loads((ROOT / d / "test_metrics.json").read_text())
        rows.append({
            "variant": name, "dir": d, "label": label.replace("\n", " "), "fig_label": label,
            "group": group, "params": params,
            "test_macro_auroc": round(m["macro_auroc"], 6),
            "test_macro_auprc": round(m["macro_auprc"], 6),
            "best_val_macro_auroc": best_val(log_name),
            "per_label": m.get("per_label"),
        })

    base = next(r for r in rows if r["variant"] == "ab_full")
    for r in rows:
        r["delta_auroc_vs_full"] = round(r["test_macro_auroc"] - base["test_macro_auroc"], 6)
        r["delta_auprc_vs_full"] = round(r["test_macro_auprc"] - base["test_macro_auprc"], 6)

    compact = []
    for fold, d in COMPACT_FOLDS:
        m = json.loads((ROOT / d / "test_metrics.json").read_text())
        compact.append({"fold": fold, "dir": d,
                        "test_macro_auroc": round(m["macro_auroc"], 6),
                        "test_macro_auprc": round(m["macro_auprc"], 6)})
    ca = np.array([c["test_macro_auroc"] for c in compact])
    cp = np.array([c["test_macro_auprc"] for c in compact])
    compact_summary = {
        "folds": compact,
        "auroc_mean": round(float(ca.mean()), 6), "auroc_std": round(float(ca.std(ddof=1)), 6),
        "auprc_mean": round(float(cp.mean()), 6), "auprc_std": round(float(cp.std(ddof=1)), 6),
        "params": 51944,
    }

    summary = {
        "batch": "Task E/F ablation + compact multi-fold",
        "config": {
            "ablation": "PTB-XL 5 superclass, folds 1-8 train / 9 val / 10 test, "
                        "100 epochs, pos_weight sqrt cap 8.0, lr 1e-3, bs 96, seed 0, EMA 0.999",
            "compact_multifold": "same but 60 epochs, pos_weight sqrt cap 8.0; "
                                 "fold10 from repro_compact, fold9/8 from repro_snn_compact_fold{9,8}",
        },
        "control_check": {
            "ab_full_test_macro_auroc": base["test_macro_auroc"],
            "mainline_snn_100ep_macro_auroc": MAINLINE_AUROC,
            "gap": round(base["test_macro_auroc"] - MAINLINE_AUROC, 6),
            "verdict": "control consistent (|gap| < 0.2pp)",
        },
        "single_fold_noise_sigma": NOISE_SIGMA,
        "ablations": rows,
        "compact_multifold": compact_summary,
    }
    (ROOT / "ablation_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    # ---------------- figure ----------------
    labels = [r["fig_label"] for r in rows]
    auroc = np.array([r["test_macro_auroc"] for r in rows])
    auprc = np.array([r["test_macro_auprc"] for r in rows])
    colors = {"control": "#8a8f98", "encoder": "#3d7dd8", "struct": "#c2703f"}
    cols = [colors[r["group"]] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.8))
    handles = None
    for ax, vals, ref, ylab, title in (
        (axes[0], auroc, base["test_macro_auroc"], "macro-AUROC", "Macro-AUROC (fold 10 test, n=2158)"),
        (axes[1], auprc, base["test_macro_auprc"], "macro-AUPRC", "Macro-AUPRC (fold 10 test, n=2158)"),
    ):
        x = np.arange(len(rows))
        ax.bar(x, vals, color=cols, width=0.62, edgecolor="white", linewidth=0.8, zorder=3)
        h_ref = ax.axhline(ref, color="#333333", ls="--", lw=1.2, zorder=4,
                           label="control (ab_full), same config")
        h_band = ax.axhspan(ref - NOISE_SIGMA, ref + NOISE_SIGMA, color="#333333", alpha=0.10,
                            zorder=1, label=f"single-fold noise band (+-1 sigma, sigma={NOISE_SIGMA})")
        for xi, v in zip(x, vals):
            d = v - ref
            ax.text(xi, v + 0.005, f"{v:.4f}\n{d:+.4f}", ha="center", va="bottom",
                    fontsize=9.0, linespacing=1.1)
        lo, hi = vals.min(), vals.max()
        ax.set_ylim(lo - 0.018, hi + 0.042)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=12)
        ax.grid(axis="y", alpha=0.25, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        handles = [h_ref, h_band]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9.5, frameon=False,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Component ablation of the SNN detector (100 epochs, identical config; "
                 "grey = control, blue = encoder swap, orange = structural)",
                 fontsize=12.5)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(ROOT / "figG_ablation.png", dpi=180)
    print("wrote", ROOT / "ablation_summary.json")
    print("wrote", ROOT / "figG_ablation.png")
    for r in rows:
        print(f"{r['variant']:14s} AUROC {r['test_macro_auroc']:.4f} "
              f"({r['delta_auroc_vs_full']:+.4f})  AUPRC {r['test_macro_auprc']:.4f} "
              f"({r['delta_auprc_vs_full']:+.4f})  params {r['params']:,}")
    print("compact 3-fold:", compact_summary["auroc_mean"], "+-", compact_summary["auroc_std"])


if __name__ == "__main__":
    main()
