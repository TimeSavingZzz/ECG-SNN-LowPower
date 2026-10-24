#!/usr/bin/env python3
"""P0/P1 consolidation: 3-fold same-fold control + combined improved models.

P0  ab_full / ab_shallow / ab_polarity  x  folds 10 / 9 / 8   (100 epochs)
    -> 3-fold mean+-std, AND paired within-fold delta (variant - ab_full in the
       SAME fold), which is the only legitimate comparison.
P1  ab_shallow_pol (3 blocks + dual-polarity) and ab_improved (+mean pool),
    fold 10 only.

Emits:
  results/p0_summary.json
  results/figH_multifold_ablation.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path("/mnt/ECG-SNN-LowPower/results")

FOLDS = [10, 9, 8]
FOLD10_DIR = {
    "ab_full": "ablation_full",
    "ab_shallow": "ablation_shallow_3blocks",
    "ab_polarity": "ablation_encoder_polarity",
}
NEW_DIR = {
    ("ab_full", 9): "repro_ab_full_fold9",
    ("ab_full", 8): "repro_ab_full_fold8",
    ("ab_shallow", 9): "repro_ab_shallow_fold9",
    ("ab_shallow", 8): "repro_ab_shallow_fold8",
    ("ab_polarity", 9): "repro_ab_polarity_fold9",
    ("ab_polarity", 8): "repro_ab_polarity_fold8",
}
P1_DIR = {
    "ab_shallow_pol": "repro_ab_shallow_pol_fold10",
    "ab_improved": "repro_ab_improved_fold10",
}
MAINLINE = 0.8594          # mainline 6-block SNN, fold 10, 100 epochs (record 03)
SARHADDI = 0.884           # Sarhaddi BSN 2026 plain SNN (paper value)


def load(d):
    m = json.loads((ROOT / d / "test_metrics.json").read_text())
    return {"auroc": m["macro_auroc"], "auprc": m["macro_auprc"],
            "per_label": {k: v["auroc"] for k, v in m.get("per_label", {}).items()}}


def ms(a):
    a = np.asarray(a, dtype=float)
    return float(a.mean()), float(a.std(ddof=1))


def main():
    runs = {}
    for variant in FOLD10_DIR:
        runs[(variant, 10)] = load(FOLD10_DIR[variant])
        for f in (9, 8):
            runs[(variant, f)] = load(NEW_DIR[(variant, f)])
    p1 = {k: load(d) for k, d in P1_DIR.items()}

    # ---- 3-fold summary per variant ----
    fold_table = {}
    for variant in FOLD10_DIR:
        a = [runs[(variant, f)]["auroc"] for f in FOLDS]
        p = [runs[(variant, f)]["auprc"] for f in FOLDS]
        am, asd = ms(a)
        pm, psd = ms(p)
        fold_table[variant] = {
            "per_fold_auroc": dict(zip([f"fold{f}" for f in FOLDS], [round(x, 6) for x in a])),
            "per_fold_auprc": dict(zip([f"fold{f}" for f in FOLDS], [round(x, 6) for x in p])),
            "auroc_mean": round(am, 6), "auroc_std": round(asd, 6),
            "auprc_mean": round(pm, 6), "auprc_std": round(psd, 6),
            "auroc_range": round(max(a) - min(a), 6),
        }

    # ---- paired within-fold delta vs ab_full ----
    pair = {}
    for variant in ("ab_shallow", "ab_polarity"):
        for metric in ("auroc", "auprc"):
            d = np.array([runs[(variant, f)][metric] - runs[("ab_full", f)][metric]
                          for f in FOLDS], dtype=float)
            m, sd = ms(d)
            t, pval = stats.ttest_rel([runs[(variant, f)][metric] for f in FOLDS],
                                      [runs[("ab_full", f)][metric] for f in FOLDS])
            pair.setdefault(variant, {})[metric] = {
                "per_fold_delta": dict(zip([f"fold{f}" for f in FOLDS],
                                           [round(float(x), 6) for x in d])),
                "delta_mean": round(m, 6), "delta_std": round(sd, 6),
                "delta_sem": round(sd / np.sqrt(len(d)), 6),
                "t": round(float(t), 4), "p": round(float(pval), 4),
                "n_folds_won": int((d > 0).sum()),
                "significant_at_0.05": bool(pval < 0.05),
            }

    # ---- P1 single-fold ----
    p1_summary = {}
    for k, v in p1.items():
        p1_summary[k] = {
            "params": {"ab_shallow_pol": 528392, "ab_improved": 528133}[k],
            "fold10_auroc": round(v["auroc"], 6), "fold10_auprc": round(v["auprc"], 6),
            "per_label_auroc": {kk: round(vv, 6) for kk, vv in v["per_label"].items()},
            "delta_vs_ab_shallow_fold10": round(v["auroc"] - runs[("ab_shallow", 10)]["auroc"], 6),
            "delta_vs_ab_full_fold10": round(v["auroc"] - runs[("ab_full", 10)]["auroc"], 6),
            "delta_vs_mainline_0.8594": round(v["auroc"] - MAINLINE, 6),
        }

    # ---- verdict (R7: no polishing) ----
    verdict = []
    for variant, s in pair.items():
        a = s["auroc"]
        verdict.append({
            "variant": variant,
            "delta_auroc": a["delta_mean"], "delta_sem": a["delta_sem"],
            "p": a["p"], "n_won": a["n_folds_won"],
            "verdict": ("gain SURVIVES (p<0.05)" if a["significant_at_0.05"]
                        else "gain DOES NOT SURVIVE as a solid claim (p>=0.05)"),
        })

    summary = {
        "batch": "P0 same-fold-controlled ablation multi-fold + P1 combined improved models",
        "config": {
            "P0": "PTB-XL 5 superclass, 100 epochs, batch 96, lr 1e-3, wd 1e-4, warmup 3, "
                  "cosine, EMA 0.999, grad clip 1.0, seed 0, pos_weight sqrt cap 8.0",
            "splits": "for test fold f: train = the other 8 folds, val = fold (f-1), test = fold f; "
                      "so folds 10/9/8 have DIFFERENT val/train sets -> fold is the paired unit",
            "P1": "same config, fold 10 only",
        },
        "fold_table": fold_table,
        "paired_delta_vs_full": pair,
        "p1_combined": p1_summary,
        "reference_values": {"mainline_snn_fold10": MAINLINE, "sarhaddi_plain_snn": SARHADDI},
        "verdict": verdict,
        "caveats": [
            "n = 3 folds only: paired t-test has df=2, very low power; a non-significant "
            "result means 'not established', not 'no effect'.",
            "P1 models were run on fold 10 ONLY, so their deltas inherit the single-fold "
            "noise (~1.3pp observed for the paired delta std).",
            "fold 10 values for ab_full/ab_shallow/ab_polarity are the SAME runs as archive 005.",
        ],
    }
    (ROOT / "p0_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    # ================= figure =================
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))

    # --- left: 3-fold paired lines + mean+-std ---
    ax = axes[0]
    colors = {"ab_full": "#8a8f98", "ab_shallow": "#c2703f", "ab_polarity": "#3d7dd8"}
    xs = np.arange(len(FOLDS))
    for variant in FOLD10_DIR:
        a = [runs[(variant, f)]["auroc"] for f in FOLDS]
        ax.plot(xs, a, "-o", color=colors[variant], lw=2, ms=7,
                label=f"{variant}  {fold_table[variant]['auroc_mean']:.4f}"
                      f" +- {fold_table[variant]['auroc_std']:.4f}")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"fold {f}" for f in FOLDS], fontsize=10.5)
    ax.set_ylabel("test macro-AUROC")
    ax.set_ylim(0.845, 0.895)
    ax.set_title("P0: same-fold control, 3 folds (100 epochs)\n"
                 "the three lines are the paired comparison", fontsize=11.5)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="lower left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # --- right: fold-10 single-fold comparison ---
    ax = axes[1]
    labels = ["ab_full\n(control)", "ab_shallow\n3 blocks", "ab_shallow_pol\n+dual-polarity",
              "ab_improved\n+mean pool", "mainline SNN\n(1.89M, rec.03)", "Sarhaddi\n(paper)"]
    vals = [runs[("ab_full", 10)]["auroc"], runs[("ab_shallow", 10)]["auroc"],
            p1["ab_shallow_pol"]["auroc"], p1["ab_improved"]["auroc"],
            MAINLINE, SARHADDI]
    cols = ["#8a8f98", "#c2703f", "#9b5d6e", "#7a6a9b", "#b8b8b8", "#d9d9d9"]
    bars = ax.bar(np.arange(len(vals)), vals, color=cols, width=0.62,
                  edgecolor="white", linewidth=0.8)
    for i, (b, v) in enumerate(zip(bars, vals)):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.4f}",
                ha="center", fontsize=9)
    ax.axhline(runs[("ab_full", 10)]["auroc"], color="#333", ls="--", lw=1.1)
    ax.set_ylim(0.82, 0.90)
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(labels, fontsize=8.8)
    ax.set_ylabel("test macro-AUROC (fold 10)")
    ax.set_title("P1: combined improved models vs control (fold 10 only)", fontsize=11.5)
    ax.grid(axis="y", alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.suptitle("Same-fold-controlled ablation (3 folds) and combined improved models "
                 "(PTB-XL, 5 superclass, macro-AUROC)", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(ROOT / "figH_multifold_ablation.png", dpi=180)

    # ---- console ----
    print("== 3-fold ==")
    for k, v in fold_table.items():
        print(f"{k:14s} {v['auroc_mean']:.4f} +- {v['auroc_std']:.4f}  "
              f"range {v['auroc_range']:.4f}  {v['per_fold_auroc']}")
    print("== paired within-fold delta vs ab_full ==")
    for k, s in pair.items():
        a = s["auroc"]
        print(f"{k:14s} auroc {a['delta_mean']:+.4f} +- {a['delta_std']:.4f} "
              f"(sem {a['delta_sem']:.4f}, t={a['t']}, p={a['p']}, won {a['n_folds_won']}/3)")
    print("== P1 ==")
    for k, v in p1_summary.items():
        print(f"{k:16s} {v['fold10_auroc']:.4f}  vs shallow {v['delta_vs_ab_shallow_fold10']:+.4f}"
              f"  vs full {v['delta_vs_ab_full_fold10']:+.4f}"
              f"  vs mainline {v['delta_vs_mainline_0.8594']:+.4f}")


if __name__ == "__main__":
    main()
