#!/usr/bin/env python3
"""ab_polarity 的 T_dense 切片汇总（零计算：只重排 `theta_sweep_ab_polarity.json`）。

输入：`results/theta_sweep_ab_polarity.json`（300 行 = 6 θ × 10 折 × 5 T）
输出：`results/ab_polarity_T_summary.json`

每个 (θ, T) 单元给十折 mean±sd；另给两个「无分辨力」检查：
  - 同一 (fold, T) 下跨 θ 的最大差（整流前端应为 0）；
  - 同一 θ 下跨 T 的 AUROC / SOP 极差。

用法::

    python3 experiments/50_ab_polarity_T_summary.py
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

ROOT = Path("/mnt/ECG-SNN-LowPower")
R = ROOT / "results"
SRC = R / "theta_sweep_ab_polarity.json"
OUT = R / "ab_polarity_T_summary.json"


def ms(xs):
    return {"mean": st.mean(xs), "sd": st.stdev(xs), "n": len(xs),
            "min": min(xs), "max": max(xs)}


def main() -> None:
    d = json.loads(SRC.read_text())
    rows = d["rows"]
    thetas = sorted({round(r["theta"], 6) for r in rows})
    tds = sorted({r["t_dense"] for r in rows})
    folds = sorted({r["fold"] for r in rows})

    # ── 1) 每 (θ, T) 的十折统计 ──────────────────────────────────────
    by_t = {}
    for td in tds:
        cells = []
        for th in thetas:
            sub = [r for r in rows if r["t_dense"] == td and abs(r["theta"] - th) < 1e-12]
            cells.append({
                "theta": th,
                "n_folds": len(sub),
                "auroc": ms([r["macro_auroc"] for r in sub]),
                "auprc": ms([r["macro_auprc"] for r in sub]),
                "sops": ms([r["total_sops"] for r in sub]),
            })
        by_t[str(td)] = cells

    # ── 2) θ 无效性：同一 (fold, T) 下跨 θ 的最大差 ────────────────────
    theta_inert = {}
    for td in tds:
        da, ds = 0.0, 0
        for f in folds:
            sub = [r for r in rows if r["t_dense"] == td and r["fold"] == f]
            if not sub:
                continue
            da = max(da, max(r["macro_auroc"] for r in sub) - min(r["macro_auroc"] for r in sub))
            ds = max(ds, max(r["total_sops"] for r in sub) - min(r["total_sops"] for r in sub))
        theta_inert[str(td)] = {"max_auroc_diff_over_theta": da,
                                "max_sops_diff_over_theta": ds}

    # ── 3) T 的影响：同一 θ 下跨 T 的极差（用十折均值）────────────────
    t_effect = {}
    for th in thetas:
        a = [c["auroc"]["mean"] for td in tds for c in by_t[str(td)]
             if abs(c["theta"] - th) < 1e-12]
        s = [c["sops"]["mean"] for td in tds for c in by_t[str(td)]
             if abs(c["theta"] - th) < 1e-12]
        best_i = max(range(len(tds)), key=lambda i: a[i])
        t_effect[str(th)] = {
            "auroc_by_t": {str(td): v for td, v in zip(tds, a)},
            "auroc_range_pp": 100 * (max(a) - min(a)),
            "auroc_argmax_t": tds[best_i],
            "auroc_argmax": a[best_i],
            "sops_range_pct": 100 * (max(s) / min(s) - 1),
        }

    out = {
        "script": "experiments/50_ab_polarity_T_summary.py",
        "source": "results/theta_sweep_ab_polarity.json",
        "config": "ab_polarity",
        "n_rows": len(rows),
        "folds": folds,
        "thetas": thetas,
        "t_dense": tds,
        "protocol": d.get("protocol"),
        "by_t": by_t,
        "theta_inert_by_t": theta_inert,
        "t_effect_by_theta": t_effect,
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(f"{'T':>3} {'AUROC(θ=0.15)':>16} {'SOP(M, θ=0.15)':>16} {'max|Δθ|AUROC':>13}")
    for td in tds:
        c = next(x for x in by_t[str(td)] if abs(x["theta"] - 0.15) < 1e-12)
        print(f"{td:>3} {c['auroc']['mean']:.4f}±{c['auroc']['sd']:.4f}"
              f"   {c['sops']['mean']/1e6:8.2f}±{c['sops']['sd']/1e6:5.2f}"
              f"   {theta_inert[str(td)]['max_auroc_diff_over_theta']:.3e}")
    print("\nθ 跨 T 的 AUROC 极差（pp）:")
    for th in thetas:
        e = t_effect[str(th)]
        print(f"  θ={th:<6g} {e['auroc_range_pp']:5.2f} pp   argmax_T={e['auroc_argmax_t']:<2d}"
              f" AUROC={e['auroc_argmax']:.4f}   SOP 跨 T={e['sops_range_pct']:.3f}%")
    print(f"\n产物 → {OUT}")


if __name__ == "__main__":
    main()
