#!/usr/bin/env python3
"""P0 supplement: complete the 3-fold grid for the two combined improved models.

What it reads (15 runs = 5 models x folds 10/9/8)
  ab_full / ab_shallow / ab_polarity   : fold 10 from archive 005, folds 9/8 from archive 006
  ab_shallow_pol / ab_improved         : fold 10 from archive 006, folds 9/8 from THIS batch

For every comparison it reports, per metric (macro-AUROC, macro-AUPRC):
  mean +- std over folds, the PER-FOLD paired delta (variant - baseline in the
  SAME fold; never across folds), the 95% CI of the mean delta, Cohen's d_z,
  the paired t-test statistic/p, and the Holm-adjusted p over the test family.

Emits
  results/p1_supplement_summary.json
  a formatted console report

Stats notes (kept in the JSON as caveats, per the project's R7 rule):
  * n = 3 folds -> df = 2 -> t_{.975,2} = 4.303, so the 95% CIs are wide.
  * Cohen's d_z = mean(delta)/sd(delta) is extremely unstable at n = 3; it is
    reported because the protocol asks for it, not because it is precise.
  * A non-significant p means "difference not established", NOT "no effect".
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path("/mnt/ECG-SNN-LowPower/results")
OUT_JSON = ROOT / "p1_supplement_summary.json"

FOLDS = [10, 9, 8]

RUN_DIR = {
    ("ab_full", 10): "ablation_full",
    ("ab_full", 9): "repro_ab_full_fold9",
    ("ab_full", 8): "repro_ab_full_fold8",
    ("ab_shallow", 10): "ablation_shallow_3blocks",
    ("ab_shallow", 9): "repro_ab_shallow_fold9",
    ("ab_shallow", 8): "repro_ab_shallow_fold8",
    ("ab_polarity", 10): "ablation_encoder_polarity",
    ("ab_polarity", 9): "repro_ab_polarity_fold9",
    ("ab_polarity", 8): "repro_ab_polarity_fold8",
    ("ab_shallow_pol", 10): "repro_ab_shallow_pol_fold10",
    ("ab_shallow_pol", 9): "repro_ab_shallow_pol_fold9",
    ("ab_shallow_pol", 8): "repro_ab_shallow_pol_fold8",
    ("ab_improved", 10): "repro_ab_improved_fold10",
    ("ab_improved", 9): "repro_ab_improved_fold9",
    ("ab_improved", 8): "repro_ab_improved_fold8",
}

# (variant, baseline) -- ordered, one family per metric for Holm correction
COMPARISONS = [
    ("ab_shallow", "ab_full"),
    ("ab_polarity", "ab_full"),
    ("ab_shallow_pol", "ab_full"),
    ("ab_improved", "ab_full"),
    ("ab_shallow_pol", "ab_shallow"),
    ("ab_improved", "ab_shallow"),
    ("ab_improved", "ab_shallow_pol"),
]

PARAMS = {
    "ab_full": 1893768,
    "ab_shallow": 528392,
    "ab_polarity": 1893768,
    "ab_shallow_pol": 528392,
    "ab_improved": 528133,
}
MAINLINE = 0.8594   # mainline 6-block SNN, fold 10, 100 epochs (record 03)
SARHADDI = 0.884    # Sarhaddi BSN 2026 plain SNN (paper value, fold 10)


def load(model, fold):
    d = ROOT / RUN_DIR[(model, fold)] / "test_metrics.json"
    if not d.exists():
        raise FileNotFoundError(str(d))
    m = json.loads(d.read_text())
    return {
        "auroc": float(m["macro_auroc"]),
        "auprc": float(m["macro_auprc"]),
        "per_label": {k: float(v["auroc"]) for k, v in m.get("per_label", {}).items()},
    }


def ms(a):
    a = np.asarray(a, dtype=float)
    return float(a.mean()), float(a.std(ddof=1))


def paired_stats(a, b):
    """a, b: per-fold arrays of the SAME folds. Returns paired-delta statistics."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    n = len(d)
    m, sd = ms(d)
    sem = sd / np.sqrt(n) if n > 1 else float("nan")
    df = n - 1
    tcrit = float(stats.t.ppf(0.975, df)) if df > 0 else float("nan")
    ci_lo, ci_hi = m - tcrit * sem, m + tcrit * sem
    if n > 1:
        t_val, p_val = stats.ttest_rel(a, b)
    else:
        t_val, p_val = float("nan"), float("nan")
    d_z = m / sd if sd > 0 else float("inf")
    return {
        "per_fold_delta": [round(float(x), 6) for x in d],
        "delta_mean": round(m, 6),
        "delta_std": round(sd, 6),
        "delta_sem": round(sem, 6),
        "ci95_low": round(ci_lo, 6),
        "ci95_high": round(ci_hi, 6),
        "t": round(float(t_val), 4),
        "p": round(float(p_val), 4),
        "cohens_dz": round(float(d_z), 3) if np.isfinite(d_z) else None,
        "n_folds_won": int((d > 0).sum()),
        "n_folds": n,
        "significant_at_0.05": bool(p_val < 0.05),
    }


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (monotone, capped at 1)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pvals[i])
        running = max(running, val)
        adj[i] = running
    return adj


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def main():
    # ---------- load ----------
    missing = []
    runs = {}
    for (model, fold), d in RUN_DIR.items():
        try:
            runs[(model, fold)] = load(model, fold)
        except FileNotFoundError:
            missing.append(str(ROOT / d / "test_metrics.json"))
    if missing:
        print("MISSING result files (run the P0 supplement batch first):", file=sys.stderr)
        for p in missing:
            print("  " + p, file=sys.stderr)
        return 2

    # ---------- 3-fold table ----------
    table = {}
    for model in PARAMS:
        a = [runs[(model, f)]["auroc"] for f in FOLDS]
        p = [runs[(model, f)]["auprc"] for f in FOLDS]
        am, asd = ms(a)
        pm, psd = ms(p)
        table[model] = {
            "params": PARAMS[model],
            "per_fold_auroc": dict(zip([f"fold{f}" for f in FOLDS], [round(x, 6) for x in a])),
            "per_fold_auprc": dict(zip([f"fold{f}" for f in FOLDS], [round(x, 6) for x in p])),
            "auroc_mean": round(am, 6), "auroc_std": round(asd, 6),
            "auroc_range": round(max(a) - min(a), 6),
            "auprc_mean": round(pm, 6), "auprc_std": round(psd, 6),
        }

    # ---------- paired comparisons ----------
    comparisons = {}
    for metric in ("auroc", "auprc"):
        rows = []
        for variant, baseline in COMPARISONS:
            av = [runs[(variant, f)][metric] for f in FOLDS]
            bv = [runs[(baseline, f)][metric] for f in FOLDS]
            rows.append(paired_stats(av, bv))
        adj = holm([r["p"] for r in rows])
        for (variant, baseline), r, pa in zip(COMPARISONS, rows, adj):
            r["p_holm"] = round(float(pa), 4)
            r["significant_holm_0.05"] = bool(pa < 0.05)
            comparisons.setdefault(f"{variant} - {baseline}", {})[metric] = r

    # ---------- verdicts (R7: no polishing) ----------
    def verdict_of(r):
        if r["significant_at_0.05"] and r["significant_holm_0.05"]:
            return "difference established (raw p<0.05 and Holm p<0.05)"
        if r["significant_at_0.05"]:
            return "raw p<0.05 but NOT Holm-significant -> treat as exploratory only"
        return ("no statistically significant difference established "
                "(p>0.05); this is a power limitation, not evidence of no effect")

    verdicts = []
    for name, per_metric in comparisons.items():
        r = per_metric["auroc"]
        verdicts.append({
            "comparison": name,
            "delta_auroc": r["delta_mean"],
            "ci95": [r["ci95_low"], r["ci95_high"]],
            "p": r["p"], "p_holm": r["p_holm"],
            "cohens_dz": r["cohens_dz"],
            "n_folds_won": f'{r["n_folds_won"]}/{r["n_folds"]}',
            "verdict": verdict_of(r),
        })

    summary = {
        "batch": "P0 supplement: ab_shallow_pol / ab_improved x folds 9,8 (3-fold completion)",
        "config": {
            "data": "PTB-XL, 5 diagnostic superclasses, multi-label",
            "training": "100 epochs, batch 96, lr 1e-3, wd 1e-4, warmup 3, cosine, "
                        "EMA 0.999, grad clip 1.0, seed 0, pos_weight sqrt cap 8.0",
            "splits": "for test fold f: train = the other 8 folds, val = fold (f-1), "
                      "test = fold f -> folds 10/9/8 have DIFFERENT val sets; the fold "
                      "is the paired unit",
        },
        "fold_table": table,
        "paired_comparisons": comparisons,
        "verdict": verdicts,
        "reference_values": {"mainline_snn_fold10": MAINLINE, "sarhaddi_plain_snn": SARHADDI},
        "caveats": [
            "n = 3 folds -> df = 2. The 95% CI uses t_{.975,2} = 4.303 and is therefore wide.",
            "Cohen's d_z is computed on only 3 paired deltas; it is reported for protocol "
            "compliance and must not be interpreted as a stable effect size.",
            "Holm correction is applied within each metric family of 7 comparisons; "
            "unadjusted p is also reported.",
            "A non-significant result means the difference was NOT ESTABLISHED; it does "
            "not prove the absence of an effect.",
            "ab_full / ab_shallow / ab_polarity fold-10 values are the same runs as "
            "archive 005 (identical config; verified by the -0.17pp mainline offset).",
            "Sarhaddi 0.884 is also a single-fold number and is kept as a reference only.",
        ],
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    # ---------- console report ----------
    line = "=" * 94
    print(line)
    print("P0 SUPPLEMENT — 3-fold same-fold-controlled comparison (PTB-XL, macro-AUROC)")
    print(line)
    print(f"{'model':16s} {'params':>9s} {'fold10':>7s} {'fold9':>7s} {'fold8':>7s} "
          f"{'mean':>7s} {'std':>6s} {'range':>6s}")
    print("-" * 94)
    for model, t in table.items():
        pf = t["per_fold_auroc"]
        print(f"{model:16s} {t['params']:>9,d} {pf['fold10']:>7.4f} {pf['fold9']:>7.4f} "
              f"{pf['fold8']:>7.4f} {t['auroc_mean']:>7.4f} {t['auroc_std']:>6.4f} "
              f"{t['auroc_range']:>6.4f}")

    print()
    print(line)
    print("PAIRED WITHIN-FOLD DELTA (variant - baseline, same fold)")
    print(line)
    print(f"{'comparison':34s} {'metric':6s} {'delta':>9s} {'95% CI':>19s} "
          f"{'d_z':>7s} {'p':>7s} {'p_holm':>7s} {'won':>4s}")
    print("-" * 94)
    for name, per_metric in comparisons.items():
        for metric in ("auroc", "auprc"):
            r = per_metric[metric]
            ci = f"[{r['ci95_low']:+.4f},{r['ci95_high']:+.4f}]"
            dz = "inf" if r["cohens_dz"] is None else f"{r['cohens_dz']:+.2f}"
            print(f"{name:34s} {metric:6s} {r['delta_mean']:>+9.4f} {ci:>19s} "
                  f"{dz:>7s} {fmt_p(r['p']):>7s} {fmt_p(r['p_holm']):>7s} "
                  f"{r['n_folds_won']}/{r['n_folds']:>2d}")

    print()
    print(line)
    print("VERDICTS (AUROC) — R7: unpolished")
    print(line)
    for v in verdicts:
        print(f"  {v['comparison']:34s} {v['delta_auroc']:+.4f}pp  "
              f"CI[{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]  "
              f"p={fmt_p(v['p'])}  p_holm={fmt_p(v['p_holm'])}  "
              f"won {v['n_folds_won']}")
        print(f"  {'':34s} -> {v['verdict']}")
    print(line)
    print(f"written: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
