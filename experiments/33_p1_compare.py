#!/usr/bin/env python3
"""P1-1: ten-fold paired comparison of CNN / ResNet against the SNN (ab_full).

Pre-registered in records/01_实验思路.md entry 014:
  * paired unit = test fold, n = 10
  * PRIMARY inference = Nadeau-Bengio corrected resampled t
    (variance scaled by 1/k + n_test/n_train), Holm within the family of 2
  * unadjusted paired t reported alongside for comparability with entry 013
  * delta = 1.0pp practical-significance margin, TOST for equivalence
  * difference test and equivalence test are reported separately

Writes results/p11_compare.json
"""
import json
import math
import os
from statistics import mean, stdev

ROOT = "/mnt/ECG-SNN-LowPower/results"

# fold -> run directory for each model family
SNN_DIR = {10: "ablation_full"}
for f in range(1, 10):
    SNN_DIR[f] = "repro_ab_full_fold%d" % f

# Per-fold split sizes, counted from ptbxl_database.csv under the project's
# filtering rule (drop the 449 records with no diagnostic superclass label):
# fold: (n_test, n_train).  Official PTB-XL folds are NOT equal-sized.
FOLD_SIZES = {
    1: (2139, 17091), 2: (2138, 17111), 3: (2150, 17100), 4: (2130, 17108),
    5: (2135, 17123), 6: (2129, 17124), 7: (2134, 17125), 8: (2129, 17125),
    9: (2146, 17113), 10: (2158, 17084),
}
# mean(n_test / n_train) over the ten folds = 0.12500
R_BAR = sum(t / n for t, n in FOLD_SIZES.values()) / len(FOLD_SIZES)
DELTA = 1.0  # pp, pre-registered practical-significance margin

try:
    from scipy import stats as ss
    HAVE = True
except Exception:
    HAVE = False


def auroc(run_dir):
    p = os.path.join(ROOT, run_dir, "test_metrics.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))["macro_auroc"]


def paired(a_dirs, b_dirs):
    folds, a, b = [], [], []
    for f in sorted(a_dirs):
        va, vb = auroc(a_dirs[f]), auroc(b_dirs[f])
        if va is None or vb is None:
            continue
        folds.append(f)
        a.append(va)
        b.append(vb)
    if len(folds) < 2:
        return None
    d = [(x - y) * 100.0 for x, y in zip(a, b)]  # pp
    k = len(d)
    m = mean(d)
    s = stdev(d)
    se_naive = s / math.sqrt(k)
    # Nadeau-Bengio: Var(mean delta) = (1/k + n_test/n_train) * s^2.
    # r varies only between 0.12432 and 0.12632 across folds; use the mean.
    r = R_BAR
    se_corr = s * math.sqrt(1.0 / k + r)
    out = {
        "folds": folds,
        "per_fold_delta_pp": {str(f): round(x, 3) for f, x in zip(folds, d)},
        "per_fold_a": {str(f): round(x, 4) for f, x in zip(folds, a)},
        "per_fold_b": {str(f): round(x, 4) for f, x in zip(folds, b)},
        "mean_a": round(mean(a), 4), "sd_a": round(stdev(a), 4),
        "mean_b": round(mean(b), 4), "sd_b": round(stdev(b), 4),
        "n": k, "mean_delta_pp": round(m, 3), "sd_delta_pp": round(s, 3),
        "wins_a": sum(1 for x in d if x > 0),
        "se_naive": round(se_naive, 4), "se_corrected": round(se_corr, 4),
    }
    if HAVE:
        t_n = m / se_naive
        t_c = m / se_corr
        tc = ss.t.ppf(0.975, k - 1)
        p_n = 2 * (1 - ss.t.cdf(abs(t_n), k - 1))
        p_c = 2 * (1 - ss.t.cdf(abs(t_c), k - 1))
        # TOST at margin DELTA, using corrected SE (primary)
        t_low = (m + DELTA) / se_corr
        t_high = (DELTA - m) / se_corr
        p_tost = max(1 - ss.t.cdf(t_low, k - 1), 1 - ss.t.cdf(t_high, k - 1))
        out.update({
            "t_naive": round(t_n, 3), "p_naive": float("%.3e" % p_n),
            "t_corrected": round(t_c, 3), "p_corrected": float("%.3e" % p_c),
            "ci95_naive_pp": [round(m - tc * se_naive, 3), round(m + tc * se_naive, 3)],
            "ci95_corrected_pp": [round(m - tc * se_corr, 3), round(m + tc * se_corr, 3)],
            "dz": round(m / s, 3),
            "tost_p_corrected": float("%.3e" % p_tost),
            "tost_equivalent_at_1pp": bool(p_tost < 0.05),
            "mde_80pct_pp": None,
        })
        # minimum detectable effect at 80% power, corrected SE
        df = k - 1
        # solve for the effect size that gives 80% power
        lo, hi = 0.0, 20.0
        for _ in range(200):
            mid = (lo + hi) / 2
            tcrit = ss.t.ppf(0.975, df)
            power = 1 - ss.nct.cdf(tcrit, df, mid) + ss.nct.cdf(-tcrit, df, mid)
            if power < 0.80:
                lo = mid
            else:
                hi = mid
        out["mde_80pct_pp"] = round(hi * se_corr, 3)
    return out


def holm(pvals):
    """Holm-Bonferroni; returns adjusted p in the input order."""
    items = sorted(enumerate(pvals), key=lambda t: t[1])
    m = len(pvals)
    adj = [None] * m
    running = 0.0
    for rank, (idx, p) in enumerate(items):
        val = min(1.0, (m - rank) * p)
        running = max(running, val)
        adj[idx] = round(running, 5)
    return adj


def main():
    if not HAVE:
        raise SystemExit("scipy required")
    snn = SNN_DIR
    cnn = {f: "repro_cnn_fold%d" % f for f in range(1, 11)}
    rsn = {f: "repro_resnet_fold%d" % f for f in range(1, 11)}

    # No fallback to the historical single-fold runs: the pre-registration
    # (records/01_实验思路.md entry 014) says fold 10 is re-run, not reused.
    missing = [n for n, d in (("snn", snn), ("cnn", cnn), ("resnet", rsn))
               for f, x in d.items() if auroc(x) is None]
    res = {"note": "delta = A - B in percentage points; positive means A higher",
           "margin_delta_pp": DELTA, "r_bar": round(R_BAR, 6),
           "nb_factor": round(math.sqrt(1.0 / 10 + R_BAR), 4)}

    a_cnn = paired(cnn, snn)
    a_rsn = paired(rsn, snn)
    res["cnn_minus_snn"] = a_cnn
    res["resnet_minus_snn"] = a_rsn

    ps = [x["p_corrected"] for x in (a_cnn, a_rsn)]
    adj = holm(ps)
    for key, h in zip(("cnn_minus_snn", "resnet_minus_snn"), adj):
        res[key]["p_corrected_holm"] = h
    ps_n = [x["p_naive"] for x in (a_cnn, a_rsn)]
    adj_n = holm(ps_n)
    for key, h in zip(("cnn_minus_snn", "resnet_minus_snn"), adj_n):
        res[key]["p_naive_holm"] = h

    res["missing"] = missing
    out = "/mnt/ECG-SNN-LowPower/results/p11_compare.json"
    json.dump(res, open(out, "w"), indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
