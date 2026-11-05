#!/usr/bin/env python3
"""合并 `ab_polarity` θ 扫描的 4 个分片 → `results/theta_sweep_ab_polarity.json`。

零推理：只做去重、排序、加元数据。分片由 46 号脚本在 4 张卡上并行产出。

用法::

    python3 experiments/47_merge_polarity.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/mnt/ECG-SNN-LowPower")
R = ROOT / "results"
SHARDS = [R / f"theta_pol_shard_g{i}.json" for i in range(4)]
OUT = R / "theta_sweep_ab_polarity.json"


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
    all_rows = sorted(dedup.values(), key=lambda r: (r["fold"], r["theta"], r["t_dense"]))

    # 2026-11-05：F1 之后补跑其余 T 切片，本产物保留全部 T（不再过滤 T=8）。
    # 45 / 48 号脚本自行按 T_dense 过滤，故多 T 不影响既有派生表。
    rows = all_rows
    tds_expected = sorted({r["t_dense"] for r in rows})
    coverage = {}
    for r in rows:
        coverage.setdefault((r["fold"], round(r["theta"], 6)), set()).add(r["t_dense"])
    n_full = sum(1 for v in coverage.values() if set(tds_expected) <= v)
    incomplete = [{"fold": f, "theta": t, "have": sorted(v)}
                  for (f, t), v in sorted(coverage.items())
                  if not set(tds_expected) <= v]

    folds = sorted({r["fold"] for r in rows})
    thetas = sorted({round(r["theta"], 6) for r in rows})
    tds = tds_expected
    ck = {r["fold"]: r["checkpoint_md5"] for r in rows}
    probes = {round(r["encoder_probe"]["max_abs_diff"], 12) for r in rows}

    out = {
        "script": "experiments/46_ab_polarity_theta.py (merged by 47_merge_polarity.py)",
        "config": "ab_polarity",
        "folds": folds,
        "thetas": thetas,
        "t_dense": tds,
        "protocol": ("fold f 的测试集；发放率标定在 val fold（fold1→10, 其余 f−1），"
                     "batch_size=32，前 8 个 batch；θ=0.15 训练，其余为零样本"),
        "checkpoint_md5_by_fold": ck,
        "encoder_probe_max_abs_diff": sorted(probes),
        "merged_from": used,
        "n_rows": len(rows),
        "n_cells": len(coverage),
        "n_cells_full_t": n_full,
        "incomplete_cells": incomplete,
        "rows": rows,
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n合并 {len(rows)} 行 → {OUT}")
    print(f"  folds={folds} thetas={thetas} t_dense={tds}")
    print(f"  (fold,θ) 单元 {len(coverage)} 个，T 完整 {n_full} / {len(coverage)}")
    if incomplete:
        print(f"  ⚠️ 不完整单元 {len(incomplete)} 个，前 5：{incomplete[:5]}")
    print(f"  probe max|Δθ| 取值={sorted(probes)}")


if __name__ == "__main__":
    main()
