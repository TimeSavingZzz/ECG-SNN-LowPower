#!/usr/bin/env python3
"""Batch 021 close-out: depth family (§3.3) and bin100 tenfold (§3.4 table 7).

Two independent read-outs, both same-fold paired over the ten PTB-XL folds:

  A. width-matched depth family  ab_pref3/4/5 vs ab_pref6 == ab_full
     (only the number of blocks changes; width/dilation/pool of every retained
     block are identical to the 6-block model)
  B. binary frontend tenfold   ab_shallow_pol_bin100 vs ab_shallow_pol
     (same recipe, same folds, only the frontend differs)

Statistics are the project standard: paired unit = test fold, n = 10, df = 9.
Both the unadjusted paired t and the Nadeau-Bengio corrected t
(SE x sqrt(1 + 10*r), r = n_test/n_train = 0.125 -> factor 1.500) are reported,
plus MDD(80%) under each.  Nothing is Holm-corrected here: the depth family is a
*curve* (3 ordered levels against one reference), and the binary comparison is a
single pre-specified pair.

Writes results/batch021_closeout.json
"""
import json
import math
import os
import sys

from scipy import stats as ss

ROOT = "/mnt/ECG-SNN-LowPower"
NB_FACTOR = 1.5     # sqrt(1 + 10 * 0.125), n_train = the 8 training folds


def dirs_for(model, folds):
    """fold -> run directory.  ab_full's fold 10 lives in ablation_full."""
    out = {}
    for f in folds:
        if model == "ab_full" and f == 10:
            out[f] = "ablation_full"
        else:
            out[f] = "repro_%s_fold%d" % (model, f)
    return out


def auroc(model, fold):
    d = dirs_for(model, [fold])[fold]
    p = os.path.join(ROOT, "results", d, "test_metrics.json")
    if not os.path.exists(p):
        return None
    return float(json.load(open(p))["macro_auroc"])


def auprc_of(model, fold):
    d = dirs_for(model, [fold])[fold]
    p = os.path.join(ROOT, "results", d, "test_metrics.json")
    if not os.path.exists(p):
        return None
    return float(json.load(open(p))["macro_auprc"])


def paired_auprc(ref_model, var_model, folds):
    """Same paired statistics as `paired`, on macro-AUPRC (both directions)."""
    ref, var, used = [], [], []
    for f in folds:
        a, b = auprc_of(ref_model, f), auprc_of(var_model, f)
        if a is None or b is None:
            continue
        ref.append(a); var.append(b); used.append(f)
    n = len(used)
    if n < 2:
        return None
    d = [v - r for r, v in zip(ref, var)]
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
    se = sd / math.sqrt(n)
    df = n - 1
    t = m / se
    p = 2 * ss.t.sf(abs(t), df)
    half = ss.t.ppf(0.975, df) * se
    m80 = mdd80(m, se, df)
    return {
        "folds_used": used, "n_folds": n,
        "reference_mean": round(sum(ref) / n, 6),
        "variant_mean": round(sum(var) / n, 6),
        "per_fold_delta": [round(x, 6) for x in d],
        "delta_mean_pp": round(m * 100, 4),
        "delta_sd_pp": round(sd * 100, 4),
        "ci95_pp": [round((m - half) * 100, 4), round((m + half) * 100, 4)],
        "t": round(t, 4), "p": round(p, 4),
        "n_folds_won": sum(1 for x in d if x > 0),
        "mdd80_pp": round(m80 * 100, 4),
        "nb_factor": NB_FACTOR,
        "t_nb": round(t / NB_FACTOR, 4),
        "p_nb": round(2 * ss.t.sf(abs(t / NB_FACTOR), df), 4),
        "ci95_nb_pp": [round((m - half * NB_FACTOR) * 100, 4),
                       round((m + half * NB_FACTOR) * 100, 4)],
        "mdd80_nb_pp": round(m80 * NB_FACTOR * 100, 4),
    }


def mdd80(delta, se, df, power=0.80, alpha=0.05):
    """Smallest |delta| reaching `power` (noncentral t), given the observed se."""
    ncp_of = lambda d: d / se
    tc = ss.t.ppf(1 - alpha / 2, df)
    lo, hi = 0.0, 40.0 * se
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        pw = ss.nct.sf(tc, df, ncp_of(mid)) + ss.nct.cdf(-tc, df, ncp_of(mid))
        if pw < power:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def paired(ref_model, var_model, folds):
    ref, var, used, missing = [], [], [], []
    for f in folds:
        a, b = auroc(ref_model, f), auroc(var_model, f)
        if a is None or b is None:
            missing.append(f)
            continue
        ref.append(a); var.append(b); used.append(f)
    n = len(used)
    if n < 2:
        return None
    d = [v - r for r, v in zip(ref, var)]
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
    se = sd / math.sqrt(n)
    df = n - 1
    t = m / se
    p = 2 * ss.t.sf(abs(t), df)
    half = ss.t.ppf(0.975, df) * se
    se_nb = se * NB_FACTOR
    t_nb = m / se_nb
    p_nb = 2 * ss.t.sf(abs(t_nb), df)
    m80 = mdd80(m, se, df)
    return {
        "reference": ref_model, "variant": var_model,
        "folds_used": used, "folds_missing": missing, "n_folds": n,
        "auroc_reference": [round(x, 6) for x in ref],
        "auroc_variant": [round(x, 6) for x in var],
        "per_fold_delta": [round(x, 6) for x in d],
        "mean_reference": round(sum(ref) / n, 6),
        "mean_variant": round(sum(var) / n, 6),
        "sd_reference": round(math.sqrt(sum((x - sum(ref) / n) ** 2 for x in ref) / (n - 1)), 6),
        "sd_variant": round(math.sqrt(sum((x - sum(var) / n) ** 2 for x in var) / (n - 1)), 6),
        "delta_mean_pp": round(m * 100, 4),
        "delta_sd_pp": round(sd * 100, 4),
        "delta_sem_pp": round(se * 100, 4),
        "ci95_pp": [round((m - half) * 100, 4), round((m + half) * 100, 4)],
        "t": round(t, 4), "df": df, "p": round(p, 4),
        "cohens_dz": round(m / sd, 4) if sd else None,
        "n_folds_won": sum(1 for x in d if x > 0),
        "mdd80_pp": round(m80 * 100, 4),
        "nb_factor": NB_FACTOR,
        "t_nb": round(t_nb, 4), "p_nb": round(p_nb, 4),
        "ci95_nb_pp": [round((m - half * NB_FACTOR) * 100, 4),
                       round((m + half * NB_FACTOR) * 100, 4)],
        "mdd80_nb_pp": round(m80 * NB_FACTOR * 100, 4),
        "significant_plain": bool(p < 0.05),
        "significant_nb": bool(p_nb < 0.05),
        "effect_below_mdd80_nb": bool(abs(m) < m80 * NB_FACTOR),
    }


def main():
    folds = list(range(1, 11))

    # ---- params per depth ----------------------------------------------------
    sys.path.insert(0, os.path.join(ROOT, "third_party/neurocardio"))
    from neurocardio.factory import make_model
    params = {}
    for name in ("ab_pref3", "ab_pref4", "ab_pref5", "ab_pref6"):
        m = make_model(name, num_classes=5)
        params[name] = sum(p.numel() for p in m.parameters())
    params["ab_flfull_note"] = "ab_pref6 is byte-identical to ab_full"

    # ---- A. depth family ----------------------------------------------------
    depth = []
    for L in (3, 4, 5):
        r = paired("ab_full", "ab_pref%d" % L, folds)
        if r:
            r["depth"] = L
            r["params"] = params["ab_pref%d" % L]
            r["auprc"] = paired_auprc("ab_full", "ab_pref%d" % L, folds)
            depth.append(r)

    # ab_full itself (reference row of the curve)
    full_aurocs = [auroc("ab_full", f) for f in folds]
    full_row = {
        "depth": 6, "model": "ab_full (= ab_pref6)", "params": params["ab_pref6"],
        "mean": round(sum(full_aurocs) / len(full_aurocs), 6),
        "sd": round(math.sqrt(sum((x - sum(full_aurocs) / len(full_aurocs)) ** 2
                                 for x in full_aurocs) / (len(full_aurocs) - 1)), 6),
        "per_fold": [round(x, 6) for x in full_aurocs],
    }

    # ---- B. binary frontend tenfold -----------------------------------------
    binary = paired("ab_shallow_pol", "ab_shallow_pol_bin100", folds)
    if binary:
        def auprc(model):
            vals = []
            for f in folds:
                p = os.path.join(ROOT, "results", dirs_for(model, [f])[f],
                                 "test_metrics.json")
                vals.append(float(json.load(open(p))["macro_auprc"]))
            return vals
        ref_a, bin_a = auprc("ab_shallow_pol"), auprc("ab_shallow_pol_bin100")
        d = [b - r for r, b in zip(ref_a, bin_a)]
        md = sum(d) / len(d)
        sd = math.sqrt(sum((x - md) ** 2 for x in d) / (len(d) - 1))
        binary["auprc"] = {
            "reference_mean": round(sum(ref_a) / len(ref_a), 6),
            "variant_mean": round(sum(bin_a) / len(bin_a), 6),
            "delta_mean_pp": round(md * 100, 4),
            "delta_sd_pp": round(sd * 100, 4),
            "per_fold_delta": [round(x, 6) for x in d],
            "n_folds_won": sum(1 for x in d if x > 0),
        }

    out = {
        "script": "experiments/61_batch_closeout.py",
        "protocol": "paired unit = test fold, n=10, df=9; NB SE inflation 1.500 "
                    "(r = n_test/n_train = 0.125, n_train = 8 training folds); "
                    "no Holm (depth is an ordered curve, binary is a single pair)",
        "depth_family": {
            "design": "prefix truncation: ab_prefL keeps the first L blocks of the "
                      "6-block stack verbatim (channel/dilation/pool plan sliced), "
                      "so only the number of blocks changes",
            "reference_row": full_row,
            "comparisons": depth,
        },
        "binary_tenfold": binary,
    }

    outp = os.path.join(ROOT, "results/batch021_closeout.json")
    json.dump(out, open(outp, "w"), indent=1, ensure_ascii=False)
    print("wrote %s" % outp)

    print("\n== depth family (ab_full 6 blocks: %.4f +- %.4f) ==" % (
        full_row["mean"], full_row["sd"]))
    for r in depth:
        print("  %d blocks (%s, %d params): mean=%.4f  delta=%+.3f pp  "
              "p=%.4f  p_nb=%.4f  MDD_NB=%.3f  won=%d/%d" % (
                  r["depth"], r["variant"], r["params"], r["mean_variant"],
                  r["delta_mean_pp"], r["p"], r["p_nb"], r["mdd80_nb_pp"],
                  r["n_folds_won"], r["n_folds"]))
        a = r.get("auprc")
        if a:
            print("      AUPRC: %.4f -> %.4f  delta=%+.3f pp  p_nb=%.4f  won=%d/%d" % (
                a["reference_mean"], a["variant_mean"], a["delta_mean_pp"],
                a["p_nb"], a["n_folds_won"], a["n_folds"]))
    if binary:
        print("\n== binary frontend tenfold ==")
        print("  continuous %.4f  binary %.4f  delta=%+.3f pp  p=%.4f  p_nb=%.4f  "
              "MDD_NB=%.3f  won=%d/10" % (
                  binary["mean_reference"], binary["mean_variant"],
                  binary["delta_mean_pp"], binary["p"], binary["p_nb"],
                  binary["mdd80_nb_pp"], binary["n_folds_won"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
