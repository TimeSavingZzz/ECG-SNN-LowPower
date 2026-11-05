#!/usr/bin/env python3
"""合并 `ab_full`（6 块 Δ）θ 扫描的 4 个分片，并做逐折形状统计。

零推理：只做去重、排序、加元数据与描述性统计。

产出：
  results/theta_sweep_ab_full_10fold.json   完整 10 折 × 6 θ × T=8 网格
  results/ab_full_theta_summary.json        逐折 argmin/argmax、内部极小占比、跨折带

用法::

    python3 experiments/51_merge_ab_full_theta.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/mnt/ECG-SNN-LowPower")
R = ROOT / "results"
SHARDS = [R / f"theta_full_shard_g{i}.json" for i in range(4)]
OUT = R / "theta_sweep_ab_full_10fold.json"
SUM = R / "ab_full_theta_summary.json"

# E10 里的单权重参照：fold10 测试集、权重 d01e7aea（该权重此前被记作 ab_full 的唯一 θ 曲线来源）
REF_WEIGHT_MD5_PREFIX = "d01e7aea"


def main() -> None:
    rows, used = [], []
    for p in SHARDS:
        if not p.exists():
            print(f"[SKIP] 缺 {p.name}")
            continue
        d = json.loads(p.read_text())
        rs = d.get("rows", [])
        print(f"  {p.name}: {len(rs)} 行，folds={sorted({r['fold'] for r in rs})}")
        rows.extend(rs)
        used.append({"shard": p.name, "rows": len(rs)})

    dedup = {}
    for r in rows:
        dedup[(r["fold"], round(r["theta"], 6), r["t_dense"])] = r
    rows = sorted(dedup.values(), key=lambda r: (r["fold"], r["theta"], r["t_dense"]))

    folds = sorted({r["fold"] for r in rows})
    thetas = sorted({round(r["theta"], 6) for r in rows})
    tds = sorted({r["t_dense"] for r in rows})
    ck = {r["fold"]: r["checkpoint_md5"] for r in rows}
    probes = {round(r["encoder_probe"]["max_abs_diff"], 12) for r in rows}

    grid = {(r["fold"], round(r["theta"], 6)): r for r in rows}
    missing = [(f, t) for f in folds for t in thetas if (f, t) not in grid]

    out = {
        "script": "experiments/46_ab_polarity_theta.py (merged by 51_merge_ab_full_theta.py)",
        "config": "ab_full",
        "folds": folds,
        "thetas": thetas,
        "t_dense": tds,
        "protocol": ("fold f 的测试集；发放率标定在 val fold（fold1→10, 其余 f−1），"
                     "batch_size=32，前 8 个 batch；θ=0.15 训练，其余为零样本"),
        "checkpoint_md5_by_fold": ck,
        "encoder_probe_max_abs_diff": sorted(probes),
        "merged_from": used,
        "n_rows": len(rows),
        "n_cells": len(grid),
        "n_cells_expected": len(folds) * len(thetas),
        "missing_cells": [{"fold": f, "theta": t} for f, t in missing],
        "rows": rows,
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n合并 {len(rows)} 行 → {OUT}")
    print(f"  folds={folds} thetas={thetas} t_dense={tds}")
    print(f"  网格 {len(grid)} / {len(folds)*len(thetas)}" +
          ("" if not missing else f"  ⚠️ 缺 {len(missing)} 格"))
    print(f"  probe max|Δθ| 取值={sorted(probes)}")

    # ---- 逐折形状统计（T=8；本批只有 T=8） ----
    per_fold = []
    for f in folds:
        if any((f, t) not in grid for t in thetas):
            print(f"  [SKIP 形状] fold{f} 网格不全")
            continue
        rs = [grid[(f, t)] for t in thetas]
        auroc = [r["macro_auroc"] for r in rs]
        auprc = [r["macro_auprc"] for r in rs]
        sop = [r["total_sops"] / 1e6 for r in rs]
        i_sop = min(range(len(sop)), key=lambda i: sop[i])
        i_auroc = max(range(len(auroc)), key=lambda i: auroc[i])
        interior = 0 < i_sop < len(sop) - 1
        # 相对首点的非单调幅度：若内部极低，给出比首尾更低多少
        per_fold.append({
            "fold": f,
            "checkpoint_md5": ck[f],
            "auroc_by_theta": auroc,
            "auprc_by_theta": auprc,
            "sop_m_by_theta": sop,
            "auroc_argmax_theta": thetas[i_auroc],
            "auroc_range_pp": round((max(auroc) - min(auroc)) * 100, 3),
            "sop_argmin_theta": thetas[i_sop],
            "sop_min_m": sop[i_sop],
            "sop_max_m": max(sop),
            "sop_range_pct": round((max(sop) - min(sop)) / min(sop) * 100, 3),
            "sop_argmin_interior": interior,
            "sop_min_below_both_ends": bool(sop[i_sop] < sop[0] and sop[i_sop] < sop[-1]),
        })

    mean_auroc = [round(sum(p["auroc_by_theta"][i] for p in per_fold) / len(per_fold), 6)
                  for i in range(len(thetas))] if per_fold else []
    mean_sop = [round(sum(p["sop_m_by_theta"][i] for p in per_fold) / len(per_fold), 4)
                for i in range(len(thetas))] if per_fold else []
    n_inter = sum(1 for p in per_fold if p["sop_argmin_interior"])
    n_both = sum(1 for p in per_fold if p["sop_min_below_both_ends"])

    summary = {
        "script": "experiments/51_merge_ab_full_theta.py",
        "config": "ab_full",
        "n_folds": len(per_fold),
        "thetas": thetas,
        "t_dense": tds,
        "mean_auroc_by_theta": mean_auroc,
        "mean_sop_m_by_theta": mean_sop,
        "mean_auroc_argmax_theta": thetas[max(range(len(mean_auroc)), key=lambda i: mean_auroc[i])]
        if mean_auroc else None,
        "mean_sop_argmin_theta": thetas[min(range(len(mean_sop)), key=lambda i: mean_sop[i])]
        if mean_sop else None,
        "n_folds_interior_sop_min": n_inter,
        "n_folds_sop_min_below_both_ends": n_both,
        "frac_folds_interior": round(n_inter / len(per_fold), 3) if per_fold else None,
        "reference_single_weight_md5_prefix": REF_WEIGHT_MD5_PREFIX,
        "ref_weight_folds": [f for f, m in ck.items() if m.startswith(REF_WEIGHT_MD5_PREFIX)],
        "per_fold": per_fold,
        "note": ("E10 的 `ab_full` θ 曲线来自单权重 fold10；本表为十折逐权重复现。"
                 "SOP 为未修正值（Δ 前端不触发 stem 计费口径问题）。"),
    }
    SUM.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n逐折统计 → {SUM}")
    for p in per_fold:
        print(f"  fold{p['fold']:>2}: SOP argmin θ={p['sop_argmin_theta']:<5g} "
              f"内部={'是' if p['sop_argmin_interior'] else '否'} "
              f"极差={p['sop_range_pct']:.2f}% | AUROC argmax θ={p['auroc_argmax_theta']:<5g} "
              f"极差={p['auroc_range_pp']:.2f}pp")
    print(f"  内部极小：{n_inter}/{len(per_fold)} 折；同时低于两端：{n_both}/{len(per_fold)} 折")
    print(f"  十折均值 SOP argmin θ={summary['mean_sop_argmin_theta']}，"
          f"AUROC argmax θ={summary['mean_auroc_argmax_theta']}")


if __name__ == "__main__":
    main()
