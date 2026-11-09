#!/usr/bin/env python3
"""Power / MDD table for the tenfold paired comparisons (E20).

Reads results/p10_summary.json (plain paired t) and results/nb_sensitivity.json
(Nadeau-Bengio inflated SE) and reports, for every primary comparison:

    effect, MDD(80%) plain, MDD(80%) NB,
    achieved power at the OBSERVED effect under both 口径,
    significance verdicts (raw and Holm, both 口径).

No training, no GPU: pure statistics on files that already exist.

Why this exists: the paper must distinguish "not significant" from "no effect".
MDD is the smallest true difference that would still be detected with 80%
power at this design's sigma and n.  An observed effect below MDD means the
design was underpowered FOR THAT EFFECT SIZE -- which is a statement about the
experiment, independent of whether this particular run happened to cross p<.05.

Usage:
    python3 experiments/58_power_mdd_table.py [--out results/power_mdd_table.json]
"""
import argparse
import json
import os
import sys

from scipy.stats import t as tdist
from scipy.stats import nct

ROOT = "/mnt/ECG-SNN-LowPower"
ALPHA = 0.05
POWER_TARGET = 0.80


def power_two_sided(delta, se, df, alpha=ALPHA):
    """Exact power of a two-sided one-sample t test under the noncentral t."""
    if se <= 0:
        return float("nan")
    ncp = delta / se
    tc = tdist.ppf(1.0 - alpha / 2.0, df)
    return float(nct.sf(tc, df, ncp) + nct.cdf(-tc, df, ncp))


def mdd(se, df, power=POWER_TARGET, alpha=ALPHA):
    """Smallest |delta| reaching `power`; power is monotone in delta, so bisect."""
    lo, hi = 0.0, 40.0 * se
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if power_two_sided(mid, se, df, alpha) < power:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/power_mdd_table.json")
    a = ap.parse_args()

    p10 = json.load(open(os.path.join(ROOT, "results/p10_summary.json")))
    nb = json.load(open(os.path.join(ROOT, "results/nb_sensitivity.json")))

    nb_by_name = {}
    for fam in nb["families"].values():
        for c in fam["comparisons"]:
            nb_by_name[c["comparison"]] = c

    rows = []
    for fam_name, fam in p10["families"].items():
        for name, c in fam["comparisons"].items():
            if name not in nb_by_name and fam_name != "primary":
                continue
            n = c["n_folds"]
            df = c["df"]
            d = c["delta_mean"]
            se = c["delta_std"] / (n ** 0.5)
            # sanity: reproduce the stored sem
            assert abs(se - c["delta_sem"]) < 1e-5, (name, se, c["delta_sem"])

            if name in nb_by_name:
                nc = nb_by_name[name]
                infl = nc["se_inflation_x"]
                se_nb = se * infl
                p_nb = nc["p_nb"]
                p_holm_nb = nc["p_holm_nb"]
                sig_nb = nc["significant_nb"]
                sig_holm_nb = nc["significant_holm_nb"]
                n_sig_nb_reported = None
            else:
                infl = None
                se_nb = None
                p_nb = p_holm_nb = sig_nb = sig_holm_nb = None

            # MDD cross-check against the stored value
            mdd_plain = mdd(se, df)
            rows.append({
                "family": fam_name,
                "comparison": name,
                "n_folds": n,
                "df": df,
                "effect_pp": round(d * 100, 4),
                "se_pp": round(se * 100, 4),
                "se_inflation_nb": infl,
                "mdd80_pp_stored": round(c["mdd_80pc"] * 100, 4),
                "mdd80_pp_recomputed": round(mdd_plain * 100, 4),
                "mdd80_nb_pp": (None if se_nb is None
                                else round(mdd(se_nb, df) * 100, 4)),
                "power_at_observed_plain": round(power_two_sided(d, se, df), 4),
                "power_at_observed_nb": (None if se_nb is None
                                         else round(power_two_sided(d, se_nb, df), 4)),
                "p_plain": c["p"],
                "p_nb": p_nb,
                "p_holm_plain": c.get("p_holm"),
                "p_holm_nb": p_holm_nb,
                "sig_plain": c["significant_at_0.05"],
                "sig_nb": sig_nb,
                "sig_holm_plain": c.get("significant_holm"),
                "sig_holm_nb": sig_holm_nb,
                "effect_below_mdd80_plain": bool(abs(d) < mdd_plain),
                "effect_below_mdd80_nb": (None if se_nb is None
                                          else bool(abs(d) < mdd(se_nb, df))),
            })

    sec = nb["families"].get("secondary", {})
    pri = nb["families"]["primary"]

    out = {
        "script": "experiments/58_power_mdd_table.py",
        "sources": ["results/p10_summary.json", "results/nb_sensitivity.json"],
        "alpha": ALPHA,
        "power_target": POWER_TARGET,
        "note": "MDD = smallest true paired difference reaching POWER_TARGET at "
                "ALPHA, exact noncentral t, df = n_folds-1. MDD_NB uses the "
                "Nadeau-Bengio inflated SE (x1.500 for the 10-fold primary "
                "family, x1.173 for the 3-fold secondary family). "
                "'power_at_observed' is the achieved power had the true effect "
                "equalled the observed one; it is <0.80 exactly when the "
                "observed effect sits below MDD(80%).",
        "primary": {
            "family_size": pri["family_size"],
            "n_sig_holm_plain": pri["n_sig_holm_plain"],
            "n_sig_holm_nb": pri["n_sig_holm_nb"],
        },
        "secondary": {
            "family_size": sec.get("family_size"),
            "n_sig_holm_plain": sec.get("n_sig_holm_plain"),
            "n_sig_holm_nb": sec.get("n_sig_holm_nb"),
        },
        "comparisons": rows,
    }

    outp = os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    with open(outp, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("wrote %s" % outp)

    print("\n%-38s %8s %8s %8s %7s %7s" % (
        "comparison", "effect", "MDD80", "MDD_NB", "pow", "pow_NB"))
    for r in rows:
        print("%-38s %8.3f %8.3f %8s %7.3f %7s" % (
            r["comparison"], r["effect_pp"], r["mdd80_pp_stored"],
            ("%.3f" % r["mdd80_nb_pp"]) if r["mdd80_nb_pp"] is not None else "-",
            r["power_at_observed_plain"],
            ("%.3f" % r["power_at_observed_nb"]) if r["power_at_observed_nb"] is not None else "-"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
