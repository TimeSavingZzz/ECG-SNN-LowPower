#!/usr/bin/env python3
"""P1-b/c/d/e：θ 联合表、噪声底叠表、230× 折间区间、MAC/SOP 同口径表、累加消融表。

**零推理、零训练。** 只读既有 JSON：
  - `results/pareto/theta{θ}_T{T}.json`（30 点，部署权重 d01e7aea，fold10 测试 AUROC；
    发放率标定在 fold9 验证集，batch32 × 8 —— `11_pareto_sweep.py:200-202`）
  - `results/calib_noise_floor_run1.json`（b38：同 fold9/batch32/8-batch，
    2000 次 bootstrap + K=32 低噪点）
  - `results/theta_sweep_ab_shallow.json`（3 块 delta 十折 θ 曲线）
  - `results/block5_stats_10fold.json`（十折各自权重的总 SOP）
  - `results/p0_4_cost.json`（params / macs / 未修正 SOP）
  - `results/sop_corrected_10fold.json`（45 号产物）
  - `results/p10_summary.json`（四配置十折配对检验，plain paired t + Holm）

用法::

    python3 experiments/48_p1_tables.py
"""
from __future__ import annotations

import importlib.util
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path("/mnt/ECG-SNN-LowPower")
R = ROOT / "results"


def load(p):
    return json.loads((R / p).read_text())


def ms(xs):
    xs = [float(x) for x in xs]
    return {"mean": st.mean(xs), "sd": st.stdev(xs) if len(xs) > 1 else 0.0,
            "min": min(xs), "max": max(xs), "n": len(xs)}


def _load_mod(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── 1) θ 联合表（30 点，部署权重）────────────────────────────────────
joint = []
for p in sorted((R / "pareto").glob("theta*_T*.json")):
    d = json.loads(p.read_text())
    joint.append({
        "theta": d["theta"], "t_dense": d["t_dense"],
        "macro_auroc": d["macro_auroc"], "macro_auprc": d["macro_auprc"],
        "sops": int(d["total_sops"]),
        "energy_nm_mj": d["energy_mj_neuromorphic"],
        "energy_dense_mj": d["energy_mj_dense"],
        "energy_dense_pj": d["energy_pj_dense"],
        "ratio_dense_over_nm": d["energy_pj_dense"] / d["energy_pj_neuromorphic"],
        "latency_ms": d.get("latency_ms"),
        "latency_n_iter": d.get("latency_n_iter"),
        "spike_rate_mean": d.get("spike_rate_mean"),
        "checkpoint_md5": d.get("checkpoint_md5"),
    })
joint.sort(key=lambda r: (r["t_dense"], r["theta"]))
assert len(joint) == 30, f"θ 联合表应为 30 点，实际 {len(joint)}"

by_t = {}
for r in joint:
    by_t.setdefault(str(r["t_dense"]), []).append(r)

# 每个 T 切片内，θ 从 0.05→0.30 带来的 AUROC / SOP / 延迟位移
joint_by_t_summary = {}
for t, rows in by_t.items():
    au = [r["macro_auroc"] for r in rows]
    so = [r["sops"] for r in rows]
    la = [r["latency_ms"] for r in rows]
    joint_by_t_summary[t] = {
        "auroc_min": min(au), "auroc_max": max(au),
        "auroc_range_pp": 100 * (max(au) - min(au)),
        "auroc_argmax_theta": rows[au.index(max(au))]["theta"],
        "sops_min": min(so), "sops_max": max(so),
        "sops_range_pct": 100 * (max(so) / min(so) - 1),
        "latency_ms_min": min(la), "latency_ms_max": max(la),
    }

# ── 2) 噪声底（6 块，b38）：曲线 + band + K=32 ────────────────────────
b38 = load("calib_noise_floor_run1.json")
assert b38["calib_fold"] == 9 and b38["calib_batch_size"] == 32
noise_rows = []
for r in b38["rows"]:
    noise_rows.append({
        "theta": r["theta"],
        "sop_curve": int(r["sop_k"]),            # 与 pareto total_sops 同一数
        "sop8_boot_mean": r["sop8_boot_mean"],
        "sop8_boot_std": r["sop8_boot_std"],
        "sop8_p2_5": r["sop8_boot_p2_5"],
        "sop8_p97_5": r["sop8_boot_p97_5"],
        "rel_std_pct": 100 * r["sop8_boot_std"] / r["sop8_boot_mean"],
        "input_k": r["input_k"], "hidden_sum_k": r["hidden_sum_k"],
    })
noise_summary = {
    "calib_fold": b38["calib_fold"], "calib_batch_size": b38["calib_batch_size"],
    "k_batches": b38["k_batches"], "n_boot": b38["n_boot"],
    "argmin_theta": b38["argmin_sop_k_theta"],
    "argmin_is_interior": b38["argmin_sop_k_is_interior"],
    "argmin_hist_8batch": b38["argmin_hist_8batch"],
    "interior_argmin_freq": b38["interior_argmin_freq_8batch"],
    "z_min_vs_left": b38["z_min_vs_left"], "z_min_vs_right": b38["z_min_vs_right"],
    "min_resolvable_at_1.96sigma": bool(max(b38["z_min_vs_left"],
                                            b38["z_min_vs_right"]) >= 1.96),
    "curve_monotone_decreasing": b38["sop_k_monotone_decreasing"],
    "curve_monotone_increasing": b38["sop_k_monotone_increasing"],
    "verdict": b38["verdict"],
    "reading": ("极小点在 θ=0.075 为内部点，且 bootstrap 重抽下 95.2% 仍落在 0.075、"
                "100% 落在内部 ⇒ 非单调排序稳定；但极小点与相邻两点的差仅 "
                "z=1.25 / 0.77，未达 1.96σ ⇒ 单点幅值不可分辨，可写"
                "「排序稳健、幅值在噪声内」。"),
}

# 口径核对：pareto 的 SOP 用「前 8 个 batch」；b38 的 sop_k 用 K=32 个 batch，
# sop8_boot_* 是对 32 个 batch 重抽 8 个的 bootstrap，三者的中心略有不同。
pareto_t8 = {round(r["theta"], 3): r["sops"] for r in joint if r["t_dense"] == 8}
noise_summary["pareto_sop_8batch"] = pareto_t8
noise_summary["sop_k_minus_pareto_pct"] = {
    round(r["theta"], 3): 100 * (r["sop_k"] / pareto_t8[round(r["theta"], 3)] - 1)
    for r in b38["rows"]}
noise_summary["sop8_boot_mean_minus_pareto_pct"] = {
    round(r["theta"], 3): 100 * (r["sop8_boot_mean"] / pareto_t8[round(r["theta"], 3)] - 1)
    for r in b38["rows"]}
noise_summary["offset_note"] = (
    "pareto 曲线 = 前 8 个标定 batch；b38 的 sop8_boot_* = 从 32 个 batch 重抽 8 个的 bootstrap "
    "分布（中心与 pareto 差 0.3%~2.8%）；sop_k = 32 个 batch 的全量均值。"
    "叠图时曲线用 pareto 值，带的宽度用 bootstrap 分布，两者中心差需在图注写明。")

# ── 3) 3 块 delta 的 θ 曲线（十折）────────────────────────────────────
sh = load("theta_sweep_ab_shallow.json")
sh_rows = [r for r in sh["rows"] if r["t_dense"] == 8]
cells = {}
for r in sh_rows:
    cells.setdefault(round(r["theta"], 3), []).append(r)
shallow_curve = []
for th in sorted(cells):
    sub = cells[th]
    shallow_curve.append({
        "theta": th,
        "auroc": ms([r["macro_auroc"] for r in sub]),
        "sops": ms([r["total_sops"] for r in sub]),
        "input_rate": ms([r["spike_rates"]["input"] for r in sub]),
        "n_folds": len({r["fold"] for r in sub}),
    })
sops = [c["sops"]["mean"] for c in shallow_curve]
shallow_summary = {
    "monotone_decreasing": all(b < a for a, b in zip(sops, sops[1:])),
    "drop_pct_0.05_to_0.30": 100 * (1 - sops[-1] / sops[0]),
    "auroc_argmax_theta": max(shallow_curve, key=lambda c: c["auroc"]["mean"])["theta"],
    "auroc_spread_pp": 100 * (max(c["auroc"]["mean"] for c in shallow_curve)
                              - min(c["auroc"]["mean"] for c in shallow_curve)),
}

# ── 4) 230× 的折间区间（十折各一权重 vs 部署点）──────────────────────
e4 = load("block5_stats_10fold.json")
deployed = next(r for r in joint
                if r["t_dense"] == 8 and abs(r["theta"] - 0.15) < 1e-12)
dense_pj = deployed["energy_dense_pj"]          # 1.2050e9 pJ（= 1.2050 mJ）
pJ_per_sop = 0.1
ratio_fold = [dense_pj / (r["total_sops"] * pJ_per_sop) for r in e4["rows"]]
# 同一权重、同一数据（fold9）、只改标定样本量：8-batch 52.31M vs K=32 50.90M
b38_at_015 = next(r for r in b38["rows"] if abs(r["theta"] - 0.15) < 1e-12)
ratio_230 = {
    "deployed_point_8batch": deployed["ratio_dense_over_nm"],
    "deployed_sops_8batch": deployed["sops"],
    "deployed_same_weight_k32": {
        "sops": int(b38_at_015["sop_k"]),
        "ratio": dense_pj / (b38_at_015["sop_k"] * pJ_per_sop),
    },
    "dense_energy_pj": dense_pj,
    "tenfold_ratio_k32": ms(ratio_fold),
    "tenfold_sops_k32": ms([r["total_sops"] for r in e4["rows"]]),
    "note": ("三条口径：① 正文 230.35× = 部署权重 + 前 8 batch 标定（52.31M）；"
             "② 同一权重、同一 fold9、改 K=32 全量标定 ⇒ 50.90M ⇒ 236.8×，"
             "2.8% 的差纯由标定样本量造成；③ 十折各权重统一 K=32 ⇒ 286.6×"
             "（区间 239.7–312.4×）。十折模型普遍比部署权重更稀疏，故倍率更高。"),
}

# ── 5) MAC / SOP 同口径表 ─────────────────────────────────────────────
# 同一口径：都是 p0_4_cost 的 fold-9 / 前 8 batch 标定，逐模型修正 stem_conv。
# 修正恒等式与 45 号脚本一致：SOP_corr = SOP_unc + macs_stem − round(macs_stem·r_in)
cost = load("p0_4_cost.json")
corr = load("sop_corrected_10fold.json")["configs"]
RECTIFIED = {"ab_polarity", "ab_shallow_pol"}
mac_rows = []
for row in cost["rows"]:
    name = row["model"]
    stems = [l for l in row["layers"] if l["name"] == "stem_conv"]
    if stems:
        ms_, r_in = int(stems[0]["macs"]), float(stems[0]["in_spike_rate"])
    else:
        ms_, r_in = None, None
    unc = int(row["sops"])
    is_snn = bool(row["is_snn"])
    corr_single = (unc + ms_ - int(round(ms_ * r_in))
                   if name in RECTIFIED and ms_ is not None else unc)
    c = corr.get(name)
    mac_rows.append({
        "config": name, "is_snn": is_snn,
        "params": row["params"], "macs": row["macs"],
        "macs_stem": ms_, "stem_input_rate": r_in,
        "sops_uncorrected": unc,
        "sops_corrected": corr_single,
        "delta_pct": 100 * (corr_single / unc - 1),
        "rectified_frontend": name in RECTIFIED,
        "sops_over_macs": row["sops_over_macs"],
        "ratio_dense_over_corrected": (row["energy_pj_dense"] / (corr_single * 0.1)
                                       if is_snn else None),
        "tenfold_corrected": (c["summary"]["sops_corrected"]["mean"] if c else None),
    })

# ── 6) 累加消融表（6块delta → +整流 → 3块整流），带 p 值 ─────────────
p10 = load("p10_summary.json")
ft = p10["fold_table"]
cmp_ = p10["families"]["primary"]["comparisons"]
abl = []
for k, c in cmp_.items():
    abl.append({
        "comparison": k, "family": "primary(Holm-4)",
        "delta_mean": c["delta_mean"], "ci95": [c["ci95_low"], c["ci95_high"]],
        "p": c["p"], "p_holm": c["p_holm"], "folds_won": c["n_folds_won"],
        "n": c["n_folds"], "significant_holm": c["significant_holm"],
    })

# 链条中间步：ab_shallow_pol − ab_polarity（既有多重检验集里没有），
# 用 31 号脚本逐字相同的 paired_stats 现算；作为预注册链条的单步对比，
# 同时给出与四条主比较合并后的 Holm-5 校正值。
m31 = _load_mod("p10_report", "experiments/31_p10_report.py")
folds = sorted(ft["ab_polarity"]["per_fold_auroc"], key=lambda s: int(s.replace("fold", "")))
a = [ft["ab_shallow_pol"]["per_fold_auroc"][f] for f in folds]
b = [ft["ab_polarity"]["per_fold_auroc"][f] for f in folds]
step_b = m31.paired_stats(a, b, [int(f.replace("fold", "")) for f in folds])
step_b["variant"], step_b["baseline"] = "ab_shallow_pol", "ab_polarity"
allp = [c["p"] for c in cmp_.values()] + [step_b["p"]]
adj = m31.holm(allp)
step_b["p_holm_over_5"] = round(float(adj[-1]), 4)
step_b["significant_holm_over_5"] = bool(adj[-1] < 0.05)

chain = []
for name, base, var in [("① 换编码器（6块）", "ab_full", "ab_polarity"),
                        ("② 缩深度（整流下）", "ab_polarity", "ab_shallow_pol"),
                        ("③ 换编码器（3块）", "ab_full", "ab_shallow"),
                        ("④ 换编码器（3块整流 vs 3块delta）", "ab_shallow", "ab_shallow_pol")]:
    key = f"{var} - {base}"
    rec = cmp_.get(key)
    if key == "ab_shallow_pol - ab_polarity":
        rec = step_b
    chain.append({
        "step": name, "comparison": key,
        "delta_pp": 100 * rec["delta_mean"],
        "ci95_pp": [100 * rec["ci95_low"], 100 * rec["ci95_high"]],
        "p": rec["p"],
        "p_holm": rec.get("p_holm", rec.get("p_holm_over_5")),
        "folds_won": f"{rec['n_folds_won']}/{rec['n_folds']}",
        "significant": rec.get("significant_holm", rec.get("significant_holm_over_5")),
    })

means = {k: ms(list(v["per_fold_auroc"].values())) for k, v in ft.items()}

out = {
    "script": "experiments/48_p1_tables.py",
    "note": "零推理：只由既有 JSON 重排（31 号脚本的 paired_stats 用于补一个链条中间步）",
    "protocol_note": ("θ 曲线与 b38 噪声底均为：部署权重 d01e7aea，AUROC 在 fold10 测试集，"
                      "发放率标定在 fold9 验证集，batch_size=32，前 8 个 batch；"
                      "SOP = 3.7 pJ/MAC + 0.1 pJ/SOP，T_in=1000，T_dense=8。"),
    "theta_joint": {"rows": joint, "by_t": by_t, "by_t_summary": joint_by_t_summary},
    "noise_floor_6block": {"rows": noise_rows, "summary": noise_summary},
    "theta_curve_3block_delta": {"rows": shallow_curve, "summary": shallow_summary},
    "ratio_230x": ratio_230,
    "mac_table": {"rows": mac_rows,
                  "note": "修正只影响整流前端（ab_polarity / ab_shallow_pol）：stem_conv 输入为连续值，按 r_in=1 计费。"},
    "ablation_10fold": {"chain": chain, "primary_comparisons": abl,
                        "step_b_derived": step_b,
                        "per_config_auroc": means},
}
p = R / "p1_supplement.json"
p.write_text(json.dumps(out, indent=2, ensure_ascii=False))

print(f"θ 联合表：{len(joint)} 点；SOP(θ=0.15,T8)={deployed['sops']/1e6:.2f}M")
for t in sorted(joint_by_t_summary, key=int):
    s = joint_by_t_summary[t]
    print(f"  T={t:<2} AUROC 极差 {s['auroc_range_pp']:5.2f}pp "
          f"(峰 θ={s['auroc_argmax_theta']})  SOP 极差 {s['sops_range_pct']:5.1f}%  "
          f"延迟 {s['latency_ms_min']:.0f}–{s['latency_ms_max']:.0f} ms")
print("b38 中心偏移（sop_k − pareto）%：",
      {k: round(v, 2) for k, v in noise_summary["sop_k_minus_pareto_pct"].items()})
print("b38 中心偏移（boot 均值 − pareto）%：",
      {k: round(v, 2) for k, v in noise_summary["sop8_boot_mean_minus_pareto_pct"].items()})
print(f"6 块：min θ={noise_summary['argmin_theta']}，"
      f"内部频率={noise_summary['interior_argmin_freq']:.3f}，"
      f"z=({noise_summary['z_min_vs_left']:.2f},{noise_summary['z_min_vs_right']:.2f}) "
      f"⇒ 幅值可分辨={noise_summary['min_resolvable_at_1.96sigma']}")
print(f"3 块 delta：单调下降={shallow_summary['monotone_decreasing']}，"
      f"降幅={shallow_summary['drop_pct_0.05_to_0.30']:.1f}%，"
      f"AUROC 峰值 θ={shallow_summary['auroc_argmax_theta']}，"
      f"AUROC 极差={shallow_summary['auroc_spread_pp']:.2f}pp")
print(f"230×：部署(8batch) {ratio_230['deployed_point_8batch']:.2f}×；"
      f"同权重(K32) {ratio_230['deployed_same_weight_k32']['ratio']:.1f}×；十折(K32) "
      f"{ratio_230['tenfold_ratio_k32']['mean']:.1f}× "
      f"[{ratio_230['tenfold_ratio_k32']['min']:.1f}, "
      f"{ratio_230['tenfold_ratio_k32']['max']:.1f}]")
print("\nMAC/SOP 同口径（单模型，fold9 8-batch）：")
for r in mac_rows:
    print(f"  {r['config']:<16} params={r['params']:>9,} macs={r['macs']/1e6:7.2f}M "
          f"SOP {r['sops_uncorrected']/1e6:6.2f}→{r['sops_corrected']/1e6:6.2f}M "
          f"({r['delta_pct']:+5.1f}%) 倍率 {r['ratio_dense_over_corrected']}")

print("\n累加链条：")
for c in chain:
    print(f"  {c['step']:<28} {c['delta_pp']:+6.2f}pp "
          f"[{c['ci95_pp'][0]:+.2f},{c['ci95_pp'][1]:+.2f}] "
          f"p={c['p']:.4f} Holm={c['p_holm']} 胜{c['folds_won']} 显著={c['significant']}")
print(f"\n产物 → {p}")
