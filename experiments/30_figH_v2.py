#!/usr/bin/env python3
"""figH v2 — redesigned to make the negative result legible at a glance.

Reads results/p1_supplement_summary.json (produced by 29_p0_supplement.py) and
draws a two-panel figure:

  LEFT   same-fold controlled lines over folds 10/9/8, plus a mean+-std summary
         column that shows the between-fold variance explicitly, plus the paired
         p-values printed in the panel. The point is that the lines cross.
  RIGHT  fold-10 bar comparison (control / ablations / 528K combined / mainline /
         Sarhaddi). Any bar that is NOT backed by three folds is hatched and
         marked with a red cross: "single-fold only". This applies to Sarhaddi
         and to the mainline as well -- marking only our own model would be
         selective, and a reviewer would catch it.

English labels only (the container has no CJK font).

Usage:
  python3 experiments/30_figH_v2.py
Output:
  results/figH_multifold_ablation.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/mnt/ECG-SNN-LowPower/results")
SUMMARY = ROOT / "p1_supplement_summary.json"
OUT = ROOT / "figH_multifold_ablation.png"

FOLDS = [10, 9, 8]
MAINLINE = 0.8594
SARHADDI = 0.884

COLORS = {
    "ab_full": "#8a8f98",
    "ab_shallow": "#c2703f",
    "ab_polarity": "#3d7dd8",
    "ab_shallow_pol": "#7b4fa0",
    "ab_improved": "#4f8f6b",
}
LABELS = {
    "ab_full": "ab_full (6 blocks, control)",
    "ab_shallow": "ab_shallow (3 blocks)",
    "ab_polarity": "ab_polarity (dual-polarity encoder)",
    "ab_shallow_pol": "ab_shallow_pol (3 blocks + dual-polarity)",
    "ab_improved": "ab_improved (+ mean pool)",
}
LEFT_LINES = ["ab_full", "ab_shallow", "ab_polarity", "ab_shallow_pol"]
STYLES = {"ab_full": "-", "ab_shallow": "-", "ab_polarity": "--", "ab_shallow_pol": "-"}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=str(SUMMARY))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    summary_path, out_path = Path(args.summary), Path(args.out)

    if not summary_path.exists():
        print(f"missing {summary_path}; run 29_p0_supplement.py first", file=sys.stderr)
        return 2
    s = json.loads(summary_path.read_text())
    ft = s["fold_table"]
    pc = s["paired_comparisons"]

    def fold_vals(model):
        pf = ft[model]["per_fold_auroc"]
        return [pf[f"fold{f}"] for f in FOLDS]

    n_folds_of = {m: sum(1 for f in FOLDS if f"fold{f}" in ft[m]["per_fold_auroc"])
                  for m in ft}

    fig, axes = plt.subplots(1, 2, figsize=(15.2, 6.2),
                             gridspec_kw={"width_ratios": [1.45, 1.0]})

    # ---------------- LEFT: same-fold lines + variance column ----------------
    ax = axes[0]
    xs = np.arange(len(FOLDS))
    for model in LEFT_LINES:
        if model not in ft:
            continue
        ys = fold_vals(model)
        ax.plot(xs, ys, STYLES.get(model, "-"), marker="o", ms=8, lw=2.1,
                color=COLORS[model],
                label=f"{LABELS[model]}   {ft[model]['auroc_mean']:.4f}"
                      f" +- {ft[model]['auroc_std']:.4f}")

    # mean +- std column on the right of the fold axis
    xcol = len(FOLDS) - 1 + 0.62
    for i, model in enumerate(LEFT_LINES):
        if model not in ft:
            continue
        ax.errorbar(xcol + 0.17 * i, ft[model]["auroc_mean"],
                    yerr=ft[model]["auroc_std"], fmt="o", ms=7, capsize=5,
                    lw=1.6, color=COLORS[model])
    ax.axvline(len(FOLDS) - 1 + 0.42, color="#cccccc", lw=1.0, ls=":")
    ax.text(xcol + 0.18, ax.get_ylim()[0], "mean\n$\\pm$1sd", ha="center",
            va="bottom", fontsize=8.5, color="#666666")

    # paired p annotations (same fold, variant - ab_full)
    lines = ["paired $\\Delta$ vs ab_full, same fold:"]
    for model in ("ab_shallow", "ab_polarity", "ab_shallow_pol"):
        key = f"{model} - ab_full"
        if key not in pc:
            continue
        r = pc[key]["auroc"]
        tag = "n.s." if r["p"] >= 0.05 else "p<0.05"
        lines.append(f"  {model}: {r['delta_mean']*100:+.2f}pp, "
                     f"p={r['p']:.3f} ({tag}), won {r['n_folds_won']}/{r['n_folds']}")
    ax.text(0.02, 0.03, "\n".join(lines), transform=ax.transAxes, fontsize=9.3,
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.45", fc="#fbfbfb", ec="#bbbbbb"))

    ax.set_xticks(list(xs) + [xcol + 0.17])
    ax.set_xticklabels([f"fold {f}" for f in FOLDS] + ["3-fold\nsummary"], fontsize=10)
    ax.set_ylabel("test macro-AUROC")
    ax.set_xlim(-0.35, xcol + 0.48)
    ax.margins(y=0.10)
    ax.set_title("Same-fold controlled ablation (100 epochs)\n"
                 "the lines cross: no component wins on every fold", fontsize=11.5)
    ax.grid(alpha=0.25)
    # legend below the axes: at n=4 lines it covers the fold-8 point if placed inside
    ax.legend(fontsize=8.4, loc="upper center", bbox_to_anchor=(0.5, -0.15),
              ncol=2, frameon=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # ---------------- RIGHT: fold-10 bars, single-fold bars marked ----------------
    ax = axes[1]
    bars_spec = [
        ("ab_full\n(control)", ft["ab_full"]["per_fold_auroc"]["fold10"],
         COLORS["ab_full"], n_folds_of["ab_full"], None),
        ("ab_shallow\n(3 blocks)", ft["ab_shallow"]["per_fold_auroc"]["fold10"],
         COLORS["ab_shallow"], n_folds_of["ab_shallow"], "ab_shallow - ab_full"),
        ("ab_shallow_pol\n(528K combined)", ft["ab_shallow_pol"]["per_fold_auroc"]["fold10"],
         COLORS["ab_shallow_pol"], n_folds_of["ab_shallow_pol"], "ab_shallow_pol - ab_full"),
        ("mainline SNN\n(1.89M, 6 blocks)", MAINLINE, "#b8b8b8", 1, None),
        ("Sarhaddi\n(paper value)", SARHADDI, "#dcdcdc", 1, None),
    ]
    xs2 = np.arange(len(bars_spec))
    for i, (lab, val, col, nf, pkey) in enumerate(bars_spec):
        single = nf < 3
        ax.bar(i, val, width=0.64, color=col, edgecolor="white", linewidth=0.8,
               hatch="///" if single else None, alpha=0.95 if not single else 0.75)
        ax.text(i, val + 0.0016, f"{val:.4f}", ha="center", fontsize=9.2)
        if single:
            ax.plot(i, val + 0.0068, marker="x", ms=13, mew=2.6, color="#c0392b")
            ax.text(i, val + 0.0098, "single-fold only", ha="center", fontsize=7.8,
                    color="#c0392b", rotation=0)
        elif pkey and pkey in pc:
            r = pc[pkey]["auroc"]
            ax.text(i, val + 0.0052, f"3-fold: p={r['p']:.3f} (n.s.)", ha="center",
                    fontsize=7.8, color="#8a2b2b")

    ax.axhline(ft["ab_full"]["per_fold_auroc"]["fold10"], color="#333333",
               ls="--", lw=1.1)
    lo = min(b[1] for b in bars_spec) - 0.012
    hi = max(b[1] for b in bars_spec) + 0.024
    ax.set_ylim(lo, hi)
    ax.set_xticks(xs2)
    ax.set_xticklabels([b[0] for b in bars_spec], fontsize=8.6)
    ax.set_ylabel("test macro-AUROC (fold 10)")
    ax.set_title("Fold-10 comparison; hatched = not backed by 3 folds", fontsize=11.5)
    ax.grid(axis="y", alpha=0.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    fig.suptitle("Ablation under same-fold paired testing (PTB-XL, 5 superclasses)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=180)
    print(f"written: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
