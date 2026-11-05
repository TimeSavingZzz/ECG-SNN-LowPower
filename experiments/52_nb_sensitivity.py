#!/usr/bin/env python3
"""Nadeau–Bengio 修正 t 检验的敏感性分析（零推理，只读 `results/p10_summary.json`）。

背景：论文大纲写的是「n=10 同折配对，Nadeau–Bengio 校正 + Holm」，
但 `experiments/31_p10_report.py` 的 `paired_stats` 实际用的是**普通配对 t**
（`scipy.stats.ttest_rel`，分母 s/√J）。两者在「十折复用同一批数据」的设定下
给出的 p 值不同，本脚本把两种口径并列算出来，供定稿时选一种并如实标注。

Nadeau–Bengio（2003）修正重采样 t：
    t = mean(d) / sqrt( (1/J + n_test/n_train) · s² ),  df = J − 1
其中 s² 为 J 个折间差值的样本方差（ddof=1），r = n_test/n_train 为每折测试/训练样本比。

用法::

    python3 experiments/52_nb_sensitivity.py
"""
from __future__ import annotations

import json
from pathlib import Path

from scipy import stats

ROOT = Path("/mnt/ECG-SNN-LowPower")
R = ROOT / "results"
SRC = R / "p10_summary.json"
OUT = R / "nb_sensitivity.json"

# 每折测试集大小（实测：PTBXLCache(fold_subset=[f]) 的 len）；n_train = 总样本 − n_test − n_val
N_TEST = {1: 2139, 2: 2138, 3: 2150, 4: 2130, 5: 2135,
          6: 2129, 7: 2134, 8: 2129, 9: 2146, 10: 2158}
N_TOTAL = 21388  # 十折测试集之和（带 5 超类标注的样本总量）


def val_fold(f: int) -> int:
    """与 25_multifold.py 的 split() 一致：val = test−1，fold1 时 val=10。"""
    return 10 if f == 1 else f - 1


def n_train_of(f: int) -> int:
    """实际参与训练的样本数：除测试折与验证折之外的 8 折。"""
    return N_TOTAL - N_TEST[f] - N_TEST[val_fold(f)]


def holm(pvals: list[tuple[str, float]]) -> dict:
    order = sorted(pvals, key=lambda kv: kv[1])
    m = len(order)
    out, running = {}, 0.0
    for i, (name, p) in enumerate(order):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        out[name] = round(running, 6)
    return out


def main() -> None:
    d = json.loads(SRC.read_text())
    res = {
        "script": "experiments/52_nb_sensitivity.py",
        "source": "results/p10_summary.json",
        "note": ("普通配对 t 与 Nadeau–Bengio 修正 t 并列。NB 只改分母（方差膨胀）与自由度的用法，"
                 "不改点估计；Holm 在各自口径内部重做。"),
        "n_test_by_fold": N_TEST,
        "n_total": N_TOTAL,
        "families": {},
    }

    for fam_name, fam in d["families"].items():
        comps = fam["comparisons"]
        plain_p, nb_p, rows = [], [], []
        for name, c in comps.items():
            deltas = c["per_fold_delta"]
            folds = c["folds"]
            J = len(deltas)
            mean = sum(deltas) / J
            s = (sum((x - mean) ** 2 for x in deltas) / (J - 1)) ** 0.5
            df = J - 1
            t_plain = float(c.get("t") or mean / (s / J ** 0.5))
            p_plain = float(c.get("p") or 2 * stats.t.sf(abs(t_plain), df))
            # NB：r = n_test/n_train，n_train 是**实际参与训练**的 8 折（验证折不参与训练）
            r = float(sum(N_TEST[f] / n_train_of(f) for f in folds) / J)
            var_nb = s ** 2 * (1.0 / J + r)
            t_nb = float(mean / var_nb ** 0.5)
            p_nb = float(2 * stats.t.sf(abs(t_nb), df))
            plain_p.append((name, p_plain))
            nb_p.append((name, p_nb))
            tcrit = float(stats.t.ppf(0.975, df))
            se_nb = var_nb ** 0.5
            ci_nb_low = float(mean - tcrit * se_nb)
            ci_nb_high = float(mean + tcrit * se_nb)
            rows.append({
                "comparison": name, "folds": folds, "per_fold_delta": deltas,
                "delta_mean": round(mean, 6), "delta_std": round(s, 6), "df": df,
                "r_test_over_train": round(r, 6),
                "se_inflation_x": round((1.0 + J * r) ** 0.5, 4),
                "t_plain": round(t_plain, 4), "p_plain": round(p_plain, 6),
                "t_nb": round(t_nb, 4), "p_nb": round(p_nb, 6),
                "p_inflation_x": round(p_nb / p_plain, 3) if p_plain > 0 else None,
                "ci95_low_plain": c.get("ci95_low"),
                "ci95_high_plain": c.get("ci95_high"),
                "ci95_low_nb": round(ci_nb_low, 6),
                "ci95_high_nb": round(ci_nb_high, 6),
                "significant_plain": p_plain < 0.05,
                "significant_nb": p_nb < 0.05,
            })
        ph, nh = holm(plain_p), holm(nb_p)
        for row in rows:
            row["p_holm_plain"] = ph[row["comparison"]]
            row["p_holm_nb"] = nh[row["comparison"]]
            row["significant_holm_plain"] = row["p_holm_plain"] < 0.05
            row["significant_holm_nb"] = row["p_holm_nb"] < 0.05
        res["families"][fam_name] = {
            "family_size": fam.get("family_size", len(rows)),
            "n_sig_holm_plain": sum(r["significant_holm_plain"] for r in rows),
            "n_sig_holm_nb": sum(r["significant_holm_nb"] for r in rows),
            "comparisons": rows,
        }

    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"→ {OUT}\n")
    for fam_name, fam in res["families"].items():
        print(f"[{fam_name}] m={fam['family_size']}  "
              f"Holm 显著：普通 {fam['n_sig_holm_plain']}/{len(fam['comparisons'])}，"
              f"NB {fam['n_sig_holm_nb']}/{len(fam['comparisons'])}")
        for r in fam["comparisons"]:
            print(f"  {r['comparison']:<34} Δ={r['delta_mean']*100:+.3f} pp  "
                  f"t={r['t_plain']:5.2f}→{r['t_nb']:5.2f}  "
                  f"p={r['p_plain']:.5f}→{r['p_nb']:.5f}  "
                  f"Holm {r['p_holm_plain']:.5f}→{r['p_holm_nb']:.5f}  "
                  f"CI_nb=[{r['ci95_low_nb']*100:+.2f},{r['ci95_high_nb']*100:+.2f}]  "
                  f"{'✔' if r['significant_holm_nb'] else '✘'}")
        print()


if __name__ == "__main__":
    main()
