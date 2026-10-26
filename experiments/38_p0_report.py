#!/usr/bin/env python3
"""Aggregate P0-2 / P0-3 outputs into tables + figures.

Reads  results/p0_spike_profile_*.json   (P0-3, one file per config)
       results/p0_encoding_fidelity.json (P0-2)
Writes results/p0_aggregate.json, results/fig_p0_depth.png,
       results/fig_p0_encoding.png, and prints markdown tables to stdout.

Figure labels are English because the container has no CJK font installed;
the Chinese prose lives in reports/*.md.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

RESULTS = Path("/mnt/ECG-SNN-LowPower/results")

# validated categorical slots (dataviz reference palette, light mode)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e3e2df"
SURFACE = "#fcfcfb"

ORDER = ["ab_full", "ab_polarity", "ab_shallow", "ab_shallow_pol"]
NICE = {
    "ab_full": "6-block, delta",
    "ab_polarity": "6-block, bipolar",
    "ab_shallow": "3-block, delta",
    "ab_shallow_pol": "3-block, bipolar",
}


def load_profile():
    rows = []
    for p in sorted(glob.glob(str(RESULTS / "p0_spike_profile_*.json"))):
        rows.extend(json.loads(Path(p).read_text()))
    return rows


def agg_profile(rows):
    out = {}
    for cfg in ORDER:
        sub = [r for r in rows if r["config"] == cfg]
        if not sub:
            continue
        n_blocks = sub[0]["n_blocks"]
        rec = {"n_folds": len(sub), "folds": sorted(r["fold"] for r in sub),
               "params": sub[0]["params"], "n_blocks": n_blocks,
               "test_auroc_mean": float(np.mean([r["test_auroc"] for r in sub
                                                 if r.get("test_auroc")])),
               "test_auroc_sd": float(np.std([r["test_auroc"] for r in sub
                                              if r.get("test_auroc")], ddof=1)),
               "val_auroc_mean": float(np.mean([r["val_auroc_best"] for r in sub
                                                if r.get("val_auroc_best")])),
               "val_auroc_sd": float(np.std([r["val_auroc_best"] for r in sub
                                             if r.get("val_auroc_best")], ddof=1)),
               "splits": {}}
        for split in ("test", "val", "train"):
            layers = [[] for _ in range(n_blocks)]
            for r in sub:
                for L in r["splits"][split]["layers"]:
                    layers[L["layer"]].append(L)
            rec["splits"][split] = {
                "rho": [float(np.mean([l["rho"] for l in ls])) for ls in layers],
                "rho_sd": [float(np.std([l["rho"] for l in ls], ddof=1)) for ls in layers],
                "reff": [float(np.mean([l["reff"] for l in ls])) for ls in layers],
                "reff_sd": [float(np.std([l["reff"] for l in ls], ddof=1)) for ls in layers],
                "channels": [ls[0]["channels"] for ls in layers],
            }
        # encoder rate: for delta it IS a spike rate; for bipolar it is the mean
        # of the analog rectified z-score and is NOT a rate.
        rec["encoder_rho"] = float(np.mean(
            [r["splits"]["test"]["encoder_rho"] for r in sub]))
        tl = [(r["train_loss"]["first"], r["train_loss"]["last"], r["train_loss"]["min"])
              for r in sub if "train_loss" in r]
        if tl:
            rec["train_loss_first"] = float(np.mean([t[0] for t in tl]))
            rec["train_loss_last"] = float(np.mean([t[1] for t in tl]))
            rec["train_loss_min"] = float(np.mean([t[2] for t in tl]))
        out[cfg] = rec
    return out


def agg_encoding():
    rows = json.loads((RESULTS / "p0_encoding_fidelity.json").read_text())
    arms = {}
    for r in rows:
        for name, rec in r["arms"].items():
            arms.setdefault(name, []).append(rec)
    out = {}
    for name, recs in arms.items():
        out[name] = {
            "n_folds": len(recs),
            "spikes": float(np.mean([x["spikes_per_sample_test"] for x in recs])),
            "spikes_sd": float(np.std([x["spikes_per_sample_test"] for x in recs], ddof=1)),
            "mean24": float(np.mean([x["features"]["mean24"]["macro_auroc"] for x in recs])),
            "mean24_sd": float(np.std([x["features"]["mean24"]["macro_auroc"] for x in recs], ddof=1)),
            "bin20": float(np.mean([x["features"]["bin20"]["macro_auroc"] for x in recs])),
            "bin20_sd": float(np.std([x["features"]["bin20"]["macro_auroc"] for x in recs], ddof=1)),
        }
    analog = rows[0]["bipolar_analog_stats"]
    return out, analog


def print_tables(prof, enc, analog):
    print("\n## P0-3 per-layer profile (mean over folds, test split)\n")
    print("| config | blocks | params | " +
          " | ".join(f"L{i+1} rho / r_eff" for i in range(6)) + " |")
    print("|---|---|---|" + "---|" * 6)
    for cfg in ORDER:
        if cfg not in prof:
            continue
        r = prof[cfg]
        cells = []
        for i in range(6):
            if i < r["n_blocks"]:
                cells.append(f"{r['splits']['test']['rho'][i]:.4f} / "
                             f"{r['splits']['test']['reff'][i]:.4f}")
            else:
                cells.append("—")
        print(f"| {NICE[cfg]} | {r['n_blocks']} | {r['params']:,} | " +
              " | ".join(cells) + " |")

    print("\n## P0-3 AUROC\n")
    print("| config | val AUROC (best epoch) | test AUROC | folds |")
    print("|---|---|---|---|")
    for cfg in ORDER:
        if cfg not in prof:
            continue
        r = prof[cfg]
        print(f"| {NICE[cfg]} | {r['val_auroc_mean']:.4f} ± {r['val_auroc_sd']:.4f} | "
              f"{r['test_auroc_mean']:.4f} ± {r['test_auroc_sd']:.4f} | {r['n_folds']} |")

    print("\n## P0-2 encoding arms (mean ± sd over folds)\n")
    print("| arm | spikes/sample | AUROC mean-pool(24d) | AUROC 20-bin(480d) |")
    print("|---|---|---|---|")
    for name in sorted(enc):
        e = enc[name]
        print(f"| {name} | {e['spikes']:.0f} ± {e['spikes_sd']:.0f} | "
              f"{e['mean24']:.4f} ± {e['mean24_sd']:.4f} | "
              f"{e['bin20']:.4f} ± {e['bin20_sd']:.4f} |")
    print(f"\nbipolar analog tensor stats: {json.dumps(analog, ensure_ascii=False)}")


def plot_depth(prof):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2,
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.titlecolor": INK, "axes.spines.top": False,
        "axes.spines.right": False, "lines.linewidth": 2.0,
    })
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    panels = [("rho", "Mean pre-pool spike rate  ρ", "a"),
              ("reff", "Effective rank  r_eff", "b")]
    for ax, (key, title, tag) in zip(axes, panels):
        top = 0.0
        for i, cfg in enumerate(ORDER):
            if cfg not in prof:
                continue
            r = prof[cfg]
            y = np.array(r["splits"]["test"][key])
            sd = np.array(r["splits"]["test"][key + "_sd"])
            x = np.arange(1, r["n_blocks"] + 1)
            ax.plot(x, y, color=SERIES[i], marker="o", markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.5,
                    label=NICE[cfg])
            ax.fill_between(x, y - sd, y + sd, color=SERIES[i], alpha=0.12,
                            linewidth=0)
            top = max(top, float((y + sd).max()))
        ax.set_xticks(range(1, 7))
        ax.set_xlim(0.6, 6.4)
        ax.set_ylim(0, top * 1.12)
        ax.set_xlabel("Block index (depth)")
        ax.set_title(f"({tag}) {title}")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("P0-3  spike rate and effective rank across depth "
                 "(test folds, mean ± 1 sd over 10 folds)",
                 fontsize=10, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    out = RESULTS / "fig_p0_depth.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"\n-> {out}")


def plot_encoding(enc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2,
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.titlecolor": INK, "axes.spines.top": False,
        "axes.spines.right": False,
    })
    families = [
        ("delta (binary)", "#2a78d6", "o", lambda n: n.startswith("delta_")),
        ("bipolar, thresholded", "#eb6834", "s", lambda n: n.startswith("bipolar_bin")),
        ("bipolar, analog (not spikes)", "#1baf7a", "^", lambda n: n == "bipolar_analog"),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    # label placement: hand-set offsets, because eight points crowd one corner
    off = {
        "delta_th0.10": (-10, -14), "delta_th0.15": (8, -4),
        "delta_th0.20": (-40, -4), "delta_th0.30": (-46, 2),
        "bipolar_bin_tau0.5": (-14, 9), "bipolar_bin_tau1.0": (4, 9),
        "bipolar_bin_tau2.0": (-6, 10),
        "bipolar_bin_countmatched": (10, 4), "bipolar_analog": (8, 2),
    }
    pretty = {
        "delta_th0.10": "δ θ=0.10", "delta_th0.15": "δ θ=0.15",
        "delta_th0.20": "δ θ=0.20", "delta_th0.30": "δ θ=0.30",
        "bipolar_bin_tau0.5": "τ=0.5", "bipolar_bin_tau1.0": "τ=1.0",
        "bipolar_bin_tau2.0": "τ=2.0",
        "bipolar_bin_countmatched": "budget-matched", "bipolar_analog": "analog",
    }
    for label, color, mk, pred in families:
        names = [n for n in sorted(enc) if pred(n)]
        xs = [enc[n]["spikes"] for n in names]
        ys = [enc[n]["bin20"] for n in names]
        ax.scatter(xs, ys, s=74, color=color, marker=mk, edgecolor=SURFACE,
                   linewidth=1.5, label=label, zorder=3)
        for n, x, y in zip(names, xs, ys):
            dx, dy = off.get(n, (8, 6))
            ax.annotate(pretty.get(n, n), (x, y), textcoords="offset points",
                        xytext=(dx, dy), fontsize=8, color=INK2)
    # highlight the matched-budget pair
    d = enc.get("delta_th0.15")
    b = enc.get("bipolar_bin_countmatched")
    if d and b:
        ax.annotate("", xy=(b["spikes"], b["bin20"]), xytext=(d["spikes"], d["bin20"]),
                    arrowprops=dict(arrowstyle="-", color=INK, linewidth=1.0,
                                    linestyle=(0, (4, 3))))
        ax.annotate(f"same spike budget,\n+{(b['bin20'] - d['bin20']) * 100:.1f} pp AUROC",
                    ((d["spikes"] + b["spikes"]) / 2, (d["bin20"] + b["bin20"]) / 2),
                    textcoords="offset points", xytext=(-96, 4), fontsize=8.5,
                    color=INK, ha="right", va="center")
    ax.set_xscale("log")
    ax.set_xlabel("Spikes per record (log scale)")
    ax.set_ylabel("Linear decodability  (macro-AUROC, 20-bin features)")
    ax.set_title("P0-2  What survives at a fixed spike budget", pad=10)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.margins(x=0.12, y=0.12)
    ax.text(0.0, -0.20,
            "Note: the analog arm emits no spikes — its x is the number of "
            "non-zero entries (12 000 = 24 ch × 1000 steps fully dense).",
            transform=ax.transAxes, fontsize=7.5, color=INK2, va="top")
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=1)
    fig.tight_layout()
    out = RESULTS / "fig_p0_encoding.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"-> {out}")


def main():
    rows = load_profile()
    prof = agg_profile(rows)
    enc, analog = agg_encoding()
    (RESULTS / "p0_aggregate.json").write_text(json.dumps(
        {"profile": prof, "encoding": enc, "bipolar_analog_stats": analog},
        indent=2, ensure_ascii=False))
    print_tables(prof, enc, analog)
    plot_depth(prof)
    plot_encoding(enc)


if __name__ == "__main__":
    main()
