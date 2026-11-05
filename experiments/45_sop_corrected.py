#!/usr/bin/env python3
"""P1-a：SOP 修正（`stem_conv` 按稠密计费）十折重算。

**为什么修**
  `cost_snn` 用 `spike_rates["input"]` 给 `stem_conv` 打折，这对 Δ 编码器成立
  （其输出是二值脉冲）；但整流前端（`dual_polarity_encode`）输出的是**连续张量**，
  stem 的输入不是脉冲，等价发放率应为 1.0。

**怎么算（零推理、零数据）**
  cost_snn 的总 SOP = 各层 SOP 之和，且只有 `stem_conv` 的口径错误，因此

      SOP_corrected(fold) = SOP_uncorrected(fold) − round(macs_stem × r_in(fold)) + macs_stem
                          = SOP_uncorrected(fold) + macs_stem − round(macs_stem × r_in(fold))

  - `macs_stem` 与权重无关，由模型结构解析计算（`cost_snn` 传全 1 发放率即得）；
  - `SOP_uncorrected(fold)` 与 `r_in(fold)` 直接取既有扫描 JSON 的 `rows`
    （θ=0.15、T_dense=8 的单元），不重新前向。

用法::

    python3 experiments/45_sop_corrected.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/mnt/ECG-SNN-LowPower")
sys.path.insert(0, str(ROOT / "third_party" / "neurocardio"))

from neurocardio.compute import cost_snn  # noqa: E402
from neurocardio.factory import make_model  # noqa: E402

RECTIFIED = {"ab_polarity", "ab_shallow_pol"}   # 前端为连续值的配置
SOURCES = {
    "ab_shallow_pol": "results/theta_sweep_new_config.json",
    "ab_shallow": "results/theta_sweep_ab_shallow.json",
    "ab_polarity": "results/theta_sweep_ab_polarity.json",   # 由 46 号合并后生成
}
THETA = 0.15
T_DENSE = 8
PJ_PER_SOP = 0.1          # cost_snn 的常数：0.1 pJ / SOP


def stem_macs(config: str) -> tuple[int, int, int, int]:
    """返回 (params, total_macs, stem_macs, total_macs_T_dense8)。结构解析，不用数据。"""
    model = make_model(config, num_classes=5)
    model.T_dense = T_DENSE
    rates = {"input": 1.0, "stem": 1.0, "fc": 1.0}
    for i in range(len(model.blocks)):
        rates[f"block{i}"] = 1.0
    cost = cost_snn(model, rates, T_in=1000)
    layers = [l if isinstance(l, dict) else l.__dict__ for l in cost.layers]
    stem = [l for l in layers if l.get("name") == "stem_conv"][0]
    return cost.total_params, int(cost.total_macs), int(stem["macs"]), int(stem["sops"])


def mean_sd(xs):
    import statistics as st
    xs = [float(x) for x in xs]
    return {"mean": st.mean(xs), "sd": st.stdev(xs) if len(xs) > 1 else 0.0,
            "min": min(xs), "max": max(xs), "n": len(xs)}


def main() -> None:
    out_cfgs = {}
    for cfg, rel in SOURCES.items():
        p = ROOT / rel
        if not p.exists():
            print(f"[SKIP] {cfg}: 缺 {rel}")
            continue
        d = json.loads(p.read_text())
        rows = [r for r in d["rows"] if r["t_dense"] == T_DENSE
                and abs(r["theta"] - THETA) < 1e-12]
        rows = sorted(rows, key=lambda r: r["fold"])
        if not rows:
            print(f"[SKIP] {cfg}: 无 θ={THETA} T={T_DENSE} 的行")
            continue
        params, macs, macs_stem, _ = stem_macs(cfg)
        rec = []
        for r in rows:
            unc = int(r["total_sops"])
            r_in = float(r["spike_rates"]["input"])
            corr = unc if cfg not in RECTIFIED else unc + macs_stem - int(round(macs_stem * r_in))
            rec.append({
                "fold": r["fold"],
                "checkpoint_md5": r["checkpoint_md5"],
                "n_test": r["n_test"],
                "val_fold": r.get("val_fold"),
                "input_rate": r_in,
                "sops_uncorrected": unc,
                "sops_corrected": corr,
                "delta": corr - unc,
                "ratio_pct": (corr / unc - 1.0) * 100.0,
                "macro_auroc": r["macro_auroc"],
            })
        dense_pj = float(rows[0]["energy_pj_dense"])
        out_cfgs[cfg] = {
            "config": cfg,
            "rectified_frontend": cfg in RECTIFIED,
            "params": params,
            "total_macs": macs,
            "stem_macs": macs_stem,
            "dense_energy_pj": dense_pj,
            "protocol": f"θ={THETA}, T_dense={T_DENSE}；发放率标定：val fold，K=8 batch × 32（41 号同口径）",
            "rows": rec,
            "summary": {
                "sops_uncorrected": mean_sd([x["sops_uncorrected"] for x in rec]),
                "sops_corrected": mean_sd([x["sops_corrected"] for x in rec]),
                "delta": mean_sd([x["delta"] for x in rec]),
                "ratio_pct": mean_sd([x["ratio_pct"] for x in rec]),
                # 脉冲能耗 = SOP × 0.1 pJ；dense_pj 单位是 pJ，两者同量纲后才能相除
                "ratio_dense_over_corrected": mean_sd(
                    [dense_pj / (x["sops_corrected"] * PJ_PER_SOP) for x in rec]),
                "ratio_dense_over_uncorrected": mean_sd(
                    [dense_pj / (x["sops_uncorrected"] * PJ_PER_SOP) for x in rec]),
                "macro_auroc": mean_sd([x["macro_auroc"] for x in rec]),
            },
        }
        s = out_cfgs[cfg]["summary"]
        print(f"{cfg:<16} stem_macs={macs_stem/1e6:6.2f}M  "
              f"unc {s['sops_uncorrected']['mean']/1e6:7.2f}M → "
              f"corr {s['sops_corrected']['mean']/1e6:7.2f}M "
              f"({s['ratio_pct']['mean']:+6.2f}%)  "
              f"倍率 {s['ratio_dense_over_uncorrected']['mean']:.1f}× → "
              f"{s['ratio_dense_over_corrected']['mean']:.1f}×  n={s['sops_corrected']['n']}")

    out = {
        "script": "experiments/45_sop_corrected.py",
        "note": "零推理：MACs 由结构解析，SOP 取自既有多折扫描 JSON；只修正 stem_conv 的输入口径",
        "rectified_configs": sorted(RECTIFIED),
        "configs": out_cfgs,
    }
    p = ROOT / "results" / "sop_corrected_10fold.json"
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n产物 → {p}")


if __name__ == "__main__":
    main()
