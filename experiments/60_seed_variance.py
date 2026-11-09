#!/usr/bin/env python3
"""Seed-variance analysis: how much can seed averaging lower the MDD? (batch 021)

Reads the seed-0 tenfold results (already on disk) plus the extra seeds run by
``logs/p5_seed_variance_20261109.sh`` and answers, in order:

  1. what is sigma_seed (run-to-run AUROC sd of the SAME config on the SAME fold)?
  2. what fraction of the paired-difference variance sigma_delta is seed noise?
  3. if each config were averaged over k seeds, what would MDD(80%) become?
  4. how many extra runs would k seeds cost across all 10 folds?

Model used throughout
---------------------
For a paired difference between two configs, the observed per-fold spread is

    sigma_delta^2 = sigma_fold^2 + 2 * sigma_seed^2          (single seed)

because the two arms carry independent seed noise.  Averaging k seeds inside
each arm replaces the second term with 2*sigma_seed^2/k, so

    sigma_delta(k)^2 = sigma_delta(1)^2 - 2*sigma_seed^2*(1 - 1/k)

and MDD scales with sigma_delta.  The Nadeau-Bengio factor (x1.500 for the
10-fold primary family) multiplies sigma, hence MDD, and is kept in step.

Caveats stated in the output:
  * sigma_seed is measured on 3 folds x 4 configs only (folds 8/9/10, seeds 0/1/2);
  * the model ASSUMES sigma_seed is comparable across configs (pooled estimate);
  * if sigma_fold^2 comes out negative the two variance components are not
    identifiable from this design -- reported as null, not clipped silently.

Usage:
    python3 experiments/60_seed_variance.py --folds 8,9,10 --seeds 0,1,2
"""
import argparse
import glob
import json
import math
import os
import sys

ROOT = "/mnt/ECG-SNN-LowPower"
MODELS = ["ab_full", "ab_shallow", "ab_polarity", "ab_shallow_pol"]
HRS_PER_RUN = 11.3
N_GPUS = 4


def metrics_path(model, fold, seed):
    suffix = "" if seed == 0 else "_s%d" % seed
    return os.path.join(ROOT, "results", "repro_%s_fold%d%s" % (model, fold, suffix),
                        "test_metrics.json")


def load_auroc(model, fold, seed):
    p = metrics_path(model, fold, seed)
    if not os.path.exists(p):
        return None
    try:
        return float(json.load(open(p))["macro_auroc"])
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="8,9,10")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default="results/seed_variance.json")
    ap.add_argument("--target-mdd-pp", type=float, default=0.7,
                    help="MDD(80%%) NB we would like to reach, in percentage points")
    a = ap.parse_args()

    folds = [int(x) for x in a.folds.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]

    # ---- 1. per (model, fold) seed spread ------------------------------------
    per_model = {}
    missing = []
    for m in MODELS:
        groups = []
        for f in folds:
            vals = []
            for s in seeds:
                v = load_auroc(m, f, s)
                if v is None:
                    missing.append("%s fold%d seed%d" % (m, f, s))
                else:
                    vals.append(v)
            if len(vals) >= 2:
                mu = sum(vals) / len(vals)
                var = sum((x - mu) ** 2 for x in vals) / (len(vals) - 1)
                groups.append({"fold": f, "seeds": len(vals),
                               "mean": mu, "sd": math.sqrt(var),
                               "values": vals})
        if groups:
            num = sum((g["seeds"] - 1) * g["sd"] ** 2 for g in groups)
            den = sum(g["seeds"] - 1 for g in groups)
            per_model[m] = {
                "folds": groups,
                "sigma_seed": math.sqrt(num / den) if den else None,
                "df": den,
            }

    if not per_model:
        print("[FAIL] no seed groups found; nothing to analyse")
        return 1

    # ---- 2. pooled sigma_seed across configs --------------------------------
    num = sum(v["df"] * v["sigma_seed"] ** 2 for v in per_model.values() if v["sigma_seed"])
    den = sum(v["df"] for v in per_model.values() if v["sigma_seed"])
    sigma_seed = math.sqrt(num / den) if den else None
    var_seed = sigma_seed ** 2 if sigma_seed else None

    # ---- 3. project MDD under k-seed averaging ------------------------------
    p10 = json.load(open(os.path.join(ROOT, "results/p10_summary.json")))
    nb = json.load(open(os.path.join(ROOT, "results/nb_sensitivity.json")))
    infl = {c["comparison"]: c["se_inflation_x"]
            for fam in nb["families"].values() for c in fam["comparisons"]}

    proj = []
    for name, c in p10["families"]["primary"]["comparisons"].items():
        sd = c["delta_std"]                 # sigma_delta at 1 seed / 1 config-seed
        mdd_plain = c["mdd_80pc"]
        factor = infl.get(name, 1.5)
        mdd_nb = mdd_plain * factor
        var_fold = sd ** 2 - 2.0 * var_seed if var_seed is not None else None
        row = {
            "comparison": name,
            "effect_pp": round(c["delta_mean"] * 100, 4),
            "sigma_delta_pp": round(sd * 100, 4),
            "mdd80_nb_pp_at_k1": round(mdd_nb * 100, 4),
            "var_fold_pp2": (None if var_fold is None else round(var_fold * 1e4, 6)),
            "identifiable": (None if var_fold is None else bool(var_fold > 0)),
        }
        if var_fold is not None and var_fold > 0:
            for k in (2, 3, 5):
                sd_k = math.sqrt(var_fold + 2.0 * var_seed / k)
                row["mdd80_nb_pp_k%d" % k] = round(mdd_nb * (sd_k / sd) * 100, 4)
            # k needed to reach the target MDD
            need = (a.target_mdd_pp / 100.0) / (mdd_nb)
            if need < 1.0:
                want = (need * sd) ** 2            # required sigma_delta^2
                k_need = 2.0 * var_seed / (want - var_fold) if want > var_fold else None
                row["k_needed_for_target"] = (None if k_need is None
                                              else round(max(k_need, 1.0), 2))
                if k_need is None or k_need <= 1.0:
                    row["extra_runs_for_target"] = 0
                else:
                    extra = 2 * 10 * (math.ceil(k_need) - 1)   # 2 arms x 10 folds
                    row["extra_runs_for_target"] = extra
                    row["extra_gpu_hours"] = round(extra * HRS_PER_RUN, 1)
                    row["extra_wall_hours_4gpu"] = round(extra * HRS_PER_RUN / N_GPUS, 1)
            else:
                row["k_needed_for_target"] = None
                row["extra_runs_for_target"] = None
        proj.append(row)

    out = {
        "script": "experiments/60_seed_variance.py",
        "folds": folds, "seeds": seeds, "models": MODELS,
        "missing_runs": missing,
        "sigma_seed": {"pooled": sigma_seed, "df": den,
                       "per_model": {m: {"sigma_seed": v["sigma_seed"], "df": v["df"]}
                                     for m, v in per_model.items()}},
        "model_note": "sigma_delta(k)^2 = sigma_delta(1)^2 - 2*sigma_seed^2*(1-1/k); "
                      "MDD scales as sigma; NB factor kept in step.",
        "caveats": [
            "sigma_seed measured on 3 folds x 4 configs (folds 8/9/10, seeds 0/1/2)",
            "pooled over configs: assumes comparable seed noise across configurations",
            "if var_fold <= 0 the variance components are not identifiable "
            "and the projection is reported as null rather than clipped",
        ],
        "target_mdd_pp": a.target_mdd_pp,
        "projections": proj,
        "per_model_folds": {
            m: [{"fold": g["fold"], "seeds": g["seeds"],
                 "auroc": [round(x, 6) for x in g["values"]],
                 "sd_pp": round(g["sd"] * 100, 4)} for g in v["folds"]]
            for m, v in per_model.items()
        },
    }

    outp = os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    json.dump(out, open(outp, "w"), indent=1, ensure_ascii=False)
    print("wrote %s" % outp)

    print("\nsigma_seed (pooled) = %.4f pp   df=%d" % (sigma_seed * 100, den))
    for m, v in per_model.items():
        print("  %-16s %.4f pp  (df=%d)" % (m, v["sigma_seed"] * 100, v["df"]))
    print("\n%-28s %7s %9s %9s %9s %9s" % (
        "comparison", "effect", "MDD_NB@1", "k=2", "k=3", "k=5"))
    for r in proj:
        print("%-28s %7.3f %9.3f %9s %9s %9s" % (
            r["comparison"], r["effect_pp"], r["mdd80_nb_pp_at_k1"],
            ("%.3f" % r["mdd80_nb_pp_k2"]) if "mdd80_nb_pp_k2" in r else "-",
            ("%.3f" % r["mdd80_nb_pp_k3"]) if "mdd80_nb_pp_k3" in r else "-",
            ("%.3f" % r["mdd80_nb_pp_k5"]) if "mdd80_nb_pp_k5" in r else "-"))
    for r in proj:
        if r.get("extra_runs_for_target") is not None:
            print("  -> %s: k=%s seeds, %d extra runs, %.0f GPU-h (%.0f h on %d GPUs)"
                  % (r["comparison"], r.get("k_needed_for_target"),
                     r["extra_runs_for_target"], r.get("extra_gpu_hours", 0),
                     r.get("extra_wall_hours_4gpu", 0), N_GPUS))
    if missing:
        print("\n[WARN] missing runs: %d (listed in JSON)" % len(missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
