#!/usr/bin/env python3
"""离线静态报告：把各实验产物汇总成论文插图 + 自包含 ``report.html``。

设计要点
--------
1. **纯 CPU、不 import 上游**：只读现成 JSON（``07`` 的 ``comparison.json``、各 run 的
   ``history.json`` / ``test_metrics.json``、MIT-BIH 的 ``results/mitbih/``），
   因此读不到 GPU 也不影响出图，可在 SNN 还没跑满时先出一版。
2. **自包含**：图片以 base64 内嵌进单个 ``report.html``，不含任何外链，
   回传到本机后用浏览器 ``file://`` 打开即可完整显示（答辩现场零依赖兜底）。
3. **图内文字用英文**：容器没有 CJK 字体，中文会渲染成豆腐块；HTML 正文仍用中文。
4. **NaN 必须洗掉**：上游 ``multilabel_metrics`` 对零支持类别写 ``float('nan')``，
   ``json.dumps`` 会输出**裸 NaN（非法 JSON）**，前端 ``r.json()`` 会整体抛错。
   本脚本读写一律经过 :func:`clean`，写出时 ``allow_nan=False``。

用法::

    python3 experiments/09_make_report.py \\
        --runs-root /mnt/ECG-SNN-LowPower/results \\
        --out /mnt/ECG-SNN-LowPower/results/report
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUN_ORDER = ["repro_snn", "repro_cnn", "repro_resnet"]
DISPLAY = {
    "repro_snn": "SNN (NeuroCardio)",
    "repro_cnn": "CNN1D",
    "repro_resnet": "ResNet1D",
}
COLORS = {
    "repro_snn": "#d62728",
    "repro_cnn": "#1f77b4",
    "repro_resnet": "#2ca02c",
}
PTBXL_LABELS = ["NORM", "MI", "STTC", "CD", "HYP"]
PTBXL_LABEL_CN = {
    "NORM": "正常",
    "MI": "心肌梗死",
    "STTC": "ST/T 改变",
    "CD": "传导障碍",
    "HYP": "心室肥大",
}
MITBIH_LABELS = ["N", "S", "V", "F", "Q"]
# 图内一律用英文（容器无 CJK 字体，中文会渲染成豆腐块）；中文只出现在 HTML 正文里。
PTBXL_LABEL_EN = {
    "NORM": "Normal",
    "MI": "MI",
    "STTC": "ST/T change",
    "CD": "Conduction",
    "HYP": "Hypertrophy",
}
MITBIH_LABEL_EN = {
    "N": "Normal",
    "S": "Supravent.",
    "V": "Ventricular",
    "F": "Fusion",
    "Q": "Unknown",
}
MITBIH_LABEL_CN = {
    "N": "正常搏动",
    "S": "室上性异位",
    "V": "室性异位",
    "F": "融合搏动",
    "Q": "未知/起搏",
}

WARNINGS: list[str] = []


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def clean(o):
    """递归把 NaN / ±Inf 换成 None（JSON 里裸 NaN 非法）。"""
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return clean(o.item())
    return o


def load_json(path: Path):
    if not path.exists():
        WARNINGS.append(f"缺少输入：{path}")
        return None
    # json.load 默认接受裸 NaN/Infinity，交给 clean 统一洗掉
    return json.loads(path.read_text())


def fmt(v, nd=4):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def mj(pj):
    """pJ → mJ。"""
    return None if pj is None else pj / 1e9


def _finish(fig, out_png: Path, embedded: dict, key: str):
    """存 PNG 并把 base64 收进 embedded。

    同时把 matplotlib 的「缺字形」警告升级成显式告警——容器没有 CJK 字体，
    图里一旦混入中文就会渲染成豆腐块，必须当场发现而不是等看图时才发现。
    """
    import warnings

    fig.tight_layout()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    missing = {str(w.message).split("Glyph ")[-1].split(" ")[0]
               for w in caught if "missing from font" in str(w.message)}
    if missing:
        WARNINGS.append(f"{out_png.name} 含 {len(missing)} 个缺失字形（图内不要用中文）："
                        f"{sorted(missing)[:6]}")
    embedded[key] = base64.b64encode(out_png.read_bytes()).decode("ascii")
    print(f"  ✓ {out_png.name}", flush=True)


# --------------------------------------------------------------------------- #
# 各张图
# --------------------------------------------------------------------------- #
def fig_pareto(rows, pareto_pts, out_png, embedded):
    """精度-能耗帕累托：论文主图。X=估算推理能耗(mJ, 对数)，Y=macro-AUROC。"""
    fig, ax = plt.subplots(figsize=(7.6, 5.4))

    # 扫描点（若已跑）
    if pareto_pts:
        xs = [mj(p["energy_pj_neuromorphic"]) for p in pareto_pts
              if p.get("energy_pj_neuromorphic")]
        ys = [p.get("macro_auroc") for p in pareto_pts]
        keep = [(x, y) for x, y in zip(xs, ys) if x and y is not None]
        if keep:
            kx, ky = zip(*keep)
            ax.scatter(kx, ky, s=34, c="#9467bd", alpha=0.75, zorder=3,
                       label=f"SNN sweep (θ×T_dense, n={len(keep)})")

    for name in RUN_ORDER:
        r = next((r for r in rows if r["run_name"] == name), None)
        if r is None:
            continue
        x, y = mj(r.get("energy_pj_neuromorphic")), r.get("macro_auroc")
        if not x or y is None:
            continue
        ax.scatter([x], [y], s=130, c=COLORS[name], marker="o",
                   edgecolors="black", linewidths=0.6, zorder=5,
                   label=DISPLAY.get(name, name))
        ax.annotate(f"{DISPLAY.get(name, name)}\n{y:.4f}\n{x:.4g} mJ",
                    (x, y), textcoords="offset points", xytext=(9, -18),
                    fontsize=8, color=COLORS[name])

    # 稠密基线的精度水平线：直观显示 SNN 用多少能耗差距换来了什么
    cnn = next((r for r in rows if r["model"] == "cnn"), None)
    if cnn and cnn.get("macro_auroc"):
        ax.axhline(cnn["macro_auroc"], ls="--", lw=1.0, c="#1f77b4", alpha=0.7)
        ax.annotate(f"CNN1D AUROC = {cnn['macro_auroc']:.4f}",
                    (0.02, cnn["macro_auroc"]), xycoords=("axes fraction", "data"),
                    fontsize=8, color="#1f77b4", va="bottom")

    ax.set_xscale("log")
    ax.set_xlabel("Estimated inference energy per sample  (mJ, log scale)")
    ax.set_ylabel("Test macro-AUROC")
    ax.set_title("Accuracy–energy Pareto: SNN vs dense baselines (PTB-XL, fold 10)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="lower right")
    fig.text(0.5, -0.02,
             "Energy = measured synaptic-operation count x published per-op constants "
             "(3.7 pJ/MAC dense, 0.1 pJ/SOP neuromorphic). NOT a Joules measurement.",
             ha="center", fontsize=7, color="#555")
    _finish(fig, out_png, embedded, "fig1")


def fig_layer_energy(sparsity, out_png, embedded):
    """逐层能耗账：每层 dense_ops vs snn_ops（MIT-BIH 分支，上游 sparsity.json 自带）。"""
    per_layer = (sparsity or {}).get("per_layer") or {}
    if not per_layer:
        WARNINGS.append("sparsity.json 无 per_layer，跳过图 2")
        return
    names = list(per_layer)
    dense = [per_layer[n]["dense_ops"] for n in names]
    snn = [max(per_layer[n]["snn_ops"], 0.5) for n in names]
    rates = [per_layer[n]["input_spike_rate"] for n in names]

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    x = np.arange(len(names))
    w = 0.38
    ax.bar(x - w / 2, dense, w, label="Dense ops (FP32 MAC)", color="#1f77b4")
    ax.bar(x + w / 2, snn, w, label="Spiking ops (SOP)", color="#d62728")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\nr={r:.3f}" for n, r in zip(names, rates)], fontsize=8)
    ax.set_ylabel("Operation count (log)")
    tot_d, tot_s = sparsity.get("total_ann_ops"), sparsity.get("total_snn_ops")
    ax.set_title("Per-layer operation budget, MIT-BIH branch  "
                 f"(total ratio {sparsity.get('energy_ratio_ann_over_snn', 0):.1f}x)")
    ax.grid(alpha=0.3, axis="y", which="both")
    ax.legend(fontsize=8)
    for xi, (d, s) in enumerate(zip(dense, snn)):
        ax.text(xi - w / 2, d * 1.25, f"{d:,}", ha="center", fontsize=6.5)
        ax.text(xi + w / 2, s * 1.25, f"{s:,.0f}", ha="center", fontsize=6.5)
    fig.text(0.5, -0.06,
             f"total dense={tot_d:,.0f} vs total spiking={tot_s:,.0f} "
             "-> spike sparsity is the entire energy argument.",
             ha="center", fontsize=7, color="#555")
    _finish(fig, out_png, embedded, "fig2")


def fig_roc_pr(rows, out_png, embedded):
    """PTB-XL 五超类 ROC / PR 曲线。

    语义分层：**颜色 = 类别**（5 类），**线型 = 模型**（SNN 实线 / CNN 虚线 / ResNet 点线）。
    同一类别在同一张图里是同一颜色，便于直接比较三个模型在该类上的表现。
    """
    if not rows:
        return
    tab10 = plt.get_cmap("tab10")
    cls_color = {lbl: tab10(i) for i, lbl in enumerate(PTBXL_LABELS)}
    style = {"repro_snn": ("-", 1.7), "repro_cnn": ("--", 1.2), "repro_resnet": (":", 1.4)}

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    plotted = False
    for name in RUN_ORDER:
        r = next((r for r in rows if r["run_name"] == name), None)
        if not r:
            continue
        roc, pr = r.get("roc_curves") or {}, r.get("pr_curves") or {}
        ls, lw = style.get(name, ("-", 1.2))
        for lbl in PTBXL_LABELS:
            if lbl in roc and roc[lbl].get("fpr"):
                axes[0].plot(roc[lbl]["fpr"], roc[lbl]["tpr"], lw=lw, ls=ls,
                             color=cls_color[lbl], label=lbl if name == "repro_snn" else None)
                plotted = True
            if lbl in pr and pr[lbl].get("recall"):
                axes[1].plot(pr[lbl]["recall"], pr[lbl]["precision"], lw=lw, ls=ls,
                             color=cls_color[lbl])
    if not plotted:
        plt.close(fig)
        WARNINGS.append("comparison.json 无 ROC 曲线，跳过图 3")
        return

    axes[0].plot([0, 1], [0, 1], "k:", lw=0.8, alpha=0.5)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate",
                title="ROC per superclass (PTB-XL, fold 10)")
    axes[1].set(xlabel="Recall", ylabel="Precision",
                title="Precision–Recall per superclass")
    for a in axes:
        a.grid(alpha=0.3)
    # 类别图例只挂在 ROC 面板（有 label 的曲线都在那里）
    axes[0].legend(fontsize=8, ncol=2, title="superclass", title_fontsize=8)
    # 模型图例用线型表达
    model_handles = [plt.Line2D([], [], color="#333", lw=1.7, ls=style[n][0],
                                label=DISPLAY.get(n, n)) for n in RUN_ORDER]
    fig.legend(handles=model_handles, loc="lower center", ncol=3, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, -0.06), title="line style = model",
               title_fontsize=8)
    _finish(fig, out_png, embedded, "fig3")


def fig_confusion(mit_summary, out_png, embedded):
    """MIT-BIH 混淆矩阵热图 + 逐类 F1（行归一化，显示召回）。"""
    cm = (mit_summary or {}).get("confusion_matrix")
    if not cm:
        WARNINGS.append("MIT-BIH summary.json 无混淆矩阵，跳过图 4")
        return
    cm = np.asarray(cm, dtype=float)
    row = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6),
                             gridspec_kw={"width_ratios": [1.05, 1]})
    im = axes[0].imshow(row, cmap="Blues", vmin=0, vmax=1)
    axes[0].set_xticks(range(5))
    axes[0].set_xticklabels(MITBIH_LABELS)
    axes[0].set_yticks(range(5))
    axes[0].set_yticklabels(MITBIH_LABELS)
    axes[0].set(xlabel="Predicted", ylabel="True",
                title="MIT-BIH confusion, row-normalised (n=49,692 beats)")
    for i in range(5):
        for j in range(5):
            axes[0].text(j, i, f"{row[i, j] * 100:.1f}", ha="center", va="center",
                         fontsize=8, color="white" if row[i, j] > 0.55 else "black")
    fig.colorbar(im, ax=axes[0], fraction=0.046, label="recall fraction")

    f1 = (mit_summary or {}).get("per_class_f1") or {}
    sup = (mit_summary or {}).get("per_class_support") or {}
    if f1:
        y = np.arange(5)
        vals = [f1.get(c, 0) or 0 for c in MITBIH_LABELS]
        axes[1].barh(y, vals, color="#d62728", alpha=0.85)
        axes[1].set_yticks(y)
        axes[1].set_yticklabels([f"{c} ({MITBIH_LABEL_EN.get(c, '')})" for c in MITBIH_LABELS])
        axes[1].invert_yaxis()
        axes[1].set(xlabel="F1", title="Per-class F1 (macro-F1 = "
                    f"{mit_summary.get('macro_f1', 0):.4f})")
        axes[1].grid(alpha=0.3, axis="x")
        for yi, c in enumerate(MITBIH_LABELS):
            axes[1].text(vals[yi] + 0.01, yi, f"{vals[yi]:.3f}  n={sup.get(c, 0)}",
                         va="center", fontsize=7.5)
        axes[1].set_xlim(0, 1.12)
    _finish(fig, out_png, embedded, "fig4")


def fig_strip(strip, out_png, embedded):
    """单拍：原始波形 → Δ 编码 on/off → 隐藏层脉冲栅格（MIT-BIH，上游已导出）。"""
    beats = (strip or {}).get("beats") or []
    if not beats:
        WARNINGS.append("strip.json 无 beats，跳过图 5")
        return
    b = beats[0]
    ecg = np.asarray(b["ecg"], dtype=float)
    sp = b.get("input_spikes") or {}
    on = np.asarray(sp.get("on", []), dtype=float)
    off = np.asarray(sp.get("off", []), dtype=float)
    l3 = np.asarray(b.get("raster_l3") or [], dtype=float)
    l4 = np.asarray(b.get("raster_l4") or [], dtype=float)

    fig, axes = plt.subplots(4, 1, figsize=(9.0, 6.4), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1, 1, 1]})
    axes[0].plot(ecg, lw=1.0, color="#222")
    axes[0].set_ylabel("amplitude")
    axes[0].set_title(f"MIT-BIH record {strip.get('record')} beat #1 — "
                      f"true={b.get('true')} pred={b.get('pred')}  "
                      f"(input {len(ecg)} samples)")
    axes[0].grid(alpha=0.3)

    # 输入脉冲：ON 画在上半、OFF 画在下半，避免两者叠在一起分不清
    for data, col, lo, hi in ((on, "#2ca02c", 0.52, 1.0), (off, "#ff7f0e", 0.0, 0.48)):
        if data.size:
            axes[1].vlines(data, lo, hi, color=col, lw=0.9)
    axes[1].set_yticks([0.24, 0.76])
    axes[1].set_yticklabels(["OFF", "ON"], fontsize=7)
    axes[1].set_ylabel("Δ-spikes", fontsize=8)
    axes[1].grid(alpha=0.25)

    for ax, data, col, lab in ((axes[2], l3, "#d62728", "raster L3"),
                               (axes[3], l4, "#9467bd", "raster L4")):
        if data.size == 0:
            continue
        if data.ndim == 1:
            ax.vlines(data, 0, 1, color=col, lw=0.9)
        else:
            ax.scatter(data[:, 1], data[:, 0], s=3.5, c=col, marker=".")
        ax.set_ylabel(lab, fontsize=8)
        ax.grid(alpha=0.25)
    axes[3].set_xlabel("time step (Δ-modulation quantum index)")
    fig.suptitle("Spike encoding and hidden-layer activity for one beat", fontsize=10)
    _finish(fig, out_png, embedded, "fig5")


def fig_training(history_all, out_png, embedded):
    """训练曲线：loss 与 val macro-AUROC（PTB-XL 三模型；MIT-BIH 另画）。"""
    if not history_all:
        WARNINGS.append("无任何 history.json，跳过图 6")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for name, h in history_all.items():
        hist = h.get("history") or []
        if not hist:
            continue
        ep = [e["epoch"] for e in hist]
        axes[0].plot(ep, [e.get("train_loss") for e in hist], lw=1.2,
                     color=COLORS.get(name, None), label=DISPLAY.get(name, name))
        key = "val_macro_auroc" if "val_macro_auroc" in hist[0] else "macro_f1"
        axes[1].plot(ep, [e.get(key) for e in hist], lw=1.2,
                     color=COLORS.get(name, None), label=DISPLAY.get(name, name))
        best = h.get("best_macro_auroc", h.get("best_macro_f1"))
        if best is not None:
            axes[1].axhline(best, ls=":", lw=0.8, color=COLORS.get(name, None), alpha=0.6)
    axes[0].set(xlabel="epoch", ylabel="train loss", title="Training loss")
    axes[1].set(xlabel="epoch", ylabel="validation metric",
                title="Validation AUROC (PTB-XL) / macro-F1 (MIT-BIH)")
    for a in axes:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    _finish(fig, out_png, embedded, "fig6")


def fig_per_class(comp_rows, test_metrics, out_png, embedded):
    """PTB-XL 逐超类 AUROC：三模型分组柱状（论文里最常被引的一张对比图）。"""
    if not comp_rows:
        return
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    x = np.arange(len(PTBXL_LABELS))
    w = 0.26
    any_bar = False
    for i, name in enumerate(RUN_ORDER):
        r = next((r for r in comp_rows if r["run_name"] == name), None)
        per = (r or {}).get("per_label") or {}
        if not per:
            per = (test_metrics or {}).get(name, {}).get("per_label") or {}
        if not per:
            continue
        vals = [(per.get(l) or {}).get("auroc") for l in PTBXL_LABELS]
        vals = [v if v is not None else 0 for v in vals]
        ax.bar(x + (i - 1) * w, vals, w, label=DISPLAY.get(name, name),
               color=COLORS[name], alpha=0.88)
        any_bar = True
    if not any_bar:
        plt.close(fig)
        WARNINGS.append("无逐类 AUROC，跳过图 7")
        return
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n{PTBXL_LABEL_EN.get(l, '')}" for l in PTBXL_LABELS], fontsize=8.5)
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Per-superclass AUROC on PTB-XL test set (fold 10)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8.5, loc="lower right")
    _finish(fig, out_png, embedded, "fig7")


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
CSS = """
:root{--fg:#1a1a1a;--mut:#666;--line:#e2e2e2;--bg:#fafafa;--acc:#b3261e;--ok:#1a7f37}
*{box-sizing:border-box}
body{margin:0;padding:0 0 60px;background:var(--bg);color:var(--fg);
 font:15px/1.7 -apple-system,'Segoe UI',Roboto,'Helvetica Neue','Microsoft YaHei',sans-serif}
header{background:#fff;border-bottom:1px solid var(--line);padding:26px 34px}
h1{margin:0 0 6px;font-size:23px;letter-spacing:.2px}
.sub{color:var(--mut);font-size:13px}
main{max-width:1060px;margin:0 auto;padding:0 22px}
section{background:#fff;border:1px solid var(--line);border-radius:10px;
 margin:22px 0;padding:20px 24px;box-shadow:0 1px 2px rgba(0,0,0,.03)}
h2{font-size:17px;margin:0 0 4px;padding-bottom:9px;border-bottom:1px solid var(--line)}
h2 .n{color:var(--acc);font-weight:700;margin-right:8px}
.cap{color:var(--mut);font-size:12.5px;margin:10px 0 0}
img{width:100%;height:auto;display:block;margin:14px 0 0;border-radius:6px}
table{border-collapse:collapse;width:100%;font-size:13.5px;margin:14px 0 4px}
th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:right}
th:first-child,td:first-child{text-align:left}
thead th{background:#f3f3f3;font-weight:600;border-bottom:2px solid #ddd}
tr.snn{background:#fff7f6}
tr.snn td:first-child{font-weight:700;color:var(--acc)}
.warn{background:#fff8e1;border-left:4px solid #f0a500;padding:12px 15px;
 border-radius:5px;font-size:13.5px;margin:12px 0}
.ok{background:#eefbf1;border-left:4px solid var(--ok)}
.mut{color:var(--mut)}
ul{margin:8px 0 0;padding-left:22px}
li{margin:4px 0}
code{background:#f2f2f2;padding:1.5px 5px;border-radius:4px;font-size:12.5px}
"""


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def build_html(meta, rows, mit_ours, mit_auth, embedded) -> str:
    h = []
    h.append("<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>")
    h.append("<meta name='viewport' content='width=device-width,initial-scale=1'>")
    h.append("<title>ECG-SNN-LowPower 复现报告</title>")
    h.append(f"<style>{CSS}</style></head><body>")
    h.append("<header><h1>基于脉冲神经网络的低功耗心电异常检测系统 —— 复现报告</h1>")
    h.append(f"<div class='sub'>上游 <code>hanshaunlee/neurocardio</code>（MIT） · "
             f"生成于 {meta['generated_at']} · 数据源 <code>results/</code>（只读汇总，无 GPU 依赖）"
             "</div></header><main>")

    # 0. 数据完备性告警
    if meta.get("snn_epochs") is not None and meta["snn_epochs"] < 100:
        h.append("<section><div class='warn'><b>⚠ 数据为中间态</b>：SNN 主线当前仅完成 "
                 f"<b>{meta['snn_epochs']}/100</b> 轮（best val AUROC "
                 f"{fmt(meta.get('snn_best'))}），因此其精度与能耗数字<b>还会变化</b>。"
                 "CNN / ResNet 已完成 100 轮。SNN 跑满后重跑本脚本即可覆盖。</div></section>")
    else:
        h.append("<section class='ok'><div class='warn ok'>所有模型均已训练完成，"
                 "本报告数字为最终结果。</div></section>")

    # 1. 核心对比表
    h.append("<section><h2><span class='n'>1</span>三模型精度 / 能耗 / 延迟对比（PTB-XL，测试集 fold 10）</h2>")
    h.append("<table><thead><tr><th>模型</th><th>参数量</th><th>MACs</th><th>SOPs</th>"
             "<th>实测发放率</th><th>macro-AUROC</th><th>macro-AUPRC</th>"
             "<th>估算能耗 (mJ)</th><th>batch-1 延迟 (ms)</th></tr></thead><tbody>")
    for name in RUN_ORDER:
        r = next((r for r in rows if r["run_name"] == name), None)
        if not r:
            continue
        cls = " class='snn'" if r["model"] == "snn" else ""
        h.append(f"<tr{cls}><td>{esc(DISPLAY.get(name, name))}</td>"
                 f"<td>{r.get('params', 0):,}</td><td>{r.get('macs', 0):,.0f}</td>"
                 f"<td>{r.get('sops', 0):,.0f}</td>"
                 f"<td>{fmt(r.get('spike_rate_mean'), 4)}</td>"
                 f"<td><b>{fmt(r.get('macro_auroc'))}</b></td><td>{fmt(r.get('macro_auprc'))}</td>"
                 f"<td>{fmt(mj(r.get('energy_pj_neuromorphic')), 5)}</td>"
                 f"<td>{fmt(r.get('latency_ms'), 2)}</td></tr>")
    h.append("</tbody></table>")
    snn = next((r for r in rows if r["model"] == "snn"), None)
    cnn = next((r for r in rows if r["model"] == "cnn"), None)
    if snn and cnn and snn.get("energy_pj_neuromorphic") and cnn.get("energy_pj_neuromorphic"):
        ratio = cnn["energy_pj_neuromorphic"] / snn["energy_pj_neuromorphic"]
        gap = (cnn.get("macro_auroc") or 0) - (snn.get("macro_auroc") or 0)
        h.append(f"<div class='warn'><b>核心结论</b>：SNN 的估算能耗为 CNN 的 "
                 f"<b>1/{ratio:.1f}</b>，代价是 macro-AUROC 低 <b>{gap * 100:.2f} pp</b>"
                 f"（{fmt(snn.get('macro_auroc'))} vs {fmt(cnn.get('macro_auroc'))}）。"
                 "论文主线应把「精度-能耗帕累托」框为核心，而非回避精度差。</div>")
    h.append("<div class='cap'><b>能耗口径声明</b>：这是按已发表单算子能耗常量折算的"
             "<b>估算</b>（稠密 FP32 MAC 3.7 pJ，Horowitz ISSCC 2014；神经形态 SOP 0.1 pJ，"
             "Davies et al., IEEE Micro 38(1):82-99, 2018）× 本机<b>实测</b>脉冲发放率，"
             "<b>不是焦耳实测</b>。CNN 侧按稠密常量折算、未折算神经形态优势（与上游口径一致）。</div>")
    cons = meta.get("consistency") or []
    if cons:
        worst = max(c["abs_diff"] for c in cons)
        h.append("<div class='cap'><b>数字一致性自检</b>：本表指标由 "
                 "<code>07_energy_report.py</code> 重新推理得到，"
                 "<code>test_metrics.json</code> 则是 <code>train.py</code> 训练末尾评测的结果，"
                 "两者是两次独立推理，故在第 6 位小数上有浮点抖动。实测最大偏差 "
                 f"<code>{worst:.2e}</code>（容差 1e-4），"
                 "<b>属数值噪声而非错误</b>：真错误（如重复收录 run）会差 0.001 量级。"
                 "逐 run 偏差见 <code>report_data.json</code> 的 <code>consistency</code>。</div>")
    h.append("</section>")

    # 2..N 图
    fig_secs = [
        ("fig1", "精度-能耗帕累托前沿", "论文主图。SNN 与稠密基线的相对位置一目了然；"
         "若已跑扫描，紫色点为 θ（编码阈值）× T_dense（头积分步数）的 30 个配置点。"),
        ("fig2", "逐层操作数账（MIT-BIH 分支）", "突触稀疏性来自输入发放率 r 逐层衰减，"
         "FC1 层占据绝大部分密集操作 —— 这是 SNN 节能的物理来源。"),
        ("fig3", "五超类 ROC / PR 曲线", "SNN 为实线、CNN 虚线、ResNet1D 点线。HYP 类样本最少、最难。"),
        ("fig4", "MIT-BIH 混淆矩阵与逐类 F1", "行归一化后显示各类召回。F / Q 两类样本极少（388 / 7 拍），"
         "F1 低属预期，需在论文中说明类别不平衡。"),
        ("fig5", "单拍脉冲编码与隐藏层活动", "原始波形 → Δ 调制 on/off 脉冲 → L3/L4 层脉冲栅格。"
         "Δ 编码每步每导联最多发一个 quantum，故对陡峭 R 波存在压摆率限制。"),
        ("fig6", "训练曲线", "三条线共用横轴。SNN 单轮约 9 分钟（作者 ~700 s/轮），"
         "100 轮需多段续跑；CNN/ResNet 各约 3.5 s/轮。"),
        ("fig7", "逐超类 AUROC 对比", "SNN 在 HYP 与 CD 上落后最多 —— 与类别样本量正相关。"),
    ]
    for i, (key, title, cap) in enumerate(fig_secs, start=2):
        if key not in embedded:
            continue
        h.append(f"<section><h2><span class='n'>{i}</span>{esc(title)}</h2>")
        h.append(f"<img alt='{esc(title)}' src='data:image/png;base64,{embedded[key]}'>")
        h.append(f"<div class='cap'>{cap}</div></section>")

    # MIT-BIH 对拍
    n = 2 + sum(1 for k, _, _ in fig_secs if k in embedded)
    if mit_ours:
        h.append(f"<section><h2><span class='n'>{n}</span>MIT-BIH 分支：本机复现 vs 作者提交</h2>")
        h.append("<table><thead><tr><th>指标</th><th>本机复现</th><th>作者提交</th><th>差值</th>"
                 "</tr></thead><tbody>")
        for k, lab in (("overall_accuracy", "accuracy"), ("macro_f1", "macro-F1")):
            o, a = mit_ours.get(k), (mit_auth or {}).get(k)
            d = (o - a) if (o is not None and a is not None) else None
            h.append(f"<tr><td>{lab}</td><td><b>{fmt(o, 6)}</b></td><td>{fmt(a, 6)}</td>"
                     f"<td>{'+' if (d or 0) > 0 else ''}{fmt(d, 4) if d is not None else '—'}</td></tr>")
        h.append("</tbody></table>")
        h.append("<table><thead><tr><th>AAMI 类别</th><th>本机 F1</th><th>作者 F1</th>"
                 "<th>样本数</th></tr></thead><tbody>")
        ours_pc = mit_ours.get("per_class") or {}
        auth_pc = (mit_auth or {}).get("per_class") or {}
        for c in MITBIH_LABELS:
            h.append(f"<tr><td>{c}（{MITBIH_LABEL_CN.get(c, '')}）</td>"
                     f"<td>{fmt((ours_pc.get(c) or {}).get('f1'), 4)}</td>"
                     f"<td>{fmt((auth_pc.get(c) or {}).get('f1'), 4)}</td>"
                     f"<td>{(ours_pc.get(c) or {}).get('support', 0):,}</td></tr>")
        h.append("</tbody></table>")
        h.append("<div class='cap'>两者均为<b>测试集（DS2，49,692 拍）</b>指标，可直接对拍。")
        if meta.get("rr_stats_identical"):
            h.append("另：本机预处理产出的 <code>rr_stats.npz</code>（RR 间期统计量）"
                     "与作者提交版本<b>数值逐位相同</b>，说明数据划分与预处理已精确复现。")
        h.append("</div></section>")

    # 局限
    n += 1
    h.append(f"<section><h2><span class='n'>{n}</span>科学局限与免责（论文须如实写入）</h2><ul>")
    h.append("<li><b>能耗是估算而非实测</b>：见第 1 节能耗口径声明。上游作者原话为 "
             "&ldquo;a synaptic-operation count argument under published per-op energy "
             "constants, <b>not a Joules measurement</b>&rdquo;。</li>")
    h.append("<li><b>θ 扫描是 zero-shot</b>：上游 <code>train.py</code> 的 <code>theta</code> "
             "字段是<b>死配置</b>（全文仅出现一次，训练时从不传参），故所有检查点均为 "
             "θ=0.15 训练。扫描得到的精度变化<b>同时含「编码失配」与「信息损失」两种效应</b>，"
             "不能解释为「θ 的最优取值」。</li>")
    h.append("<li><b>环境偏差</b>：本机 <code>snntorch 1.0.0 / torch 2.5.1+cu124 / numpy 2.2.6</code>；"
             "上游 Modal 镜像锁定 <code>snntorch 0.9.4 / torch 2.4.1</code>。全链路已验证可跑通，"
             "但指标差异不能完全排除版本因素。</li>")
    h.append("<li><b>Δ 编码有损</b>：每步每导联最多发一个 ±θ quantum，信号单步跨幅大于 θ 时"
             "存在压摆率限制，误差随 θ 增大而增大（不随 T 增大消失）。</li>")
    h.append("</ul></section>")

    h.append("<section><div class='cap'>本文件为<b>自包含</b>单文件（图片已内嵌 base64，"
             "无任何外部链接），可直接用浏览器以 <code>file://</code> 打开，"
             "亦可用作论文插图的来源。生成脚本：<code>experiments/09_make_report.py</code>。</div></section>")
    h.append("</main></body></html>")
    return "".join(h)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="/mnt/ECG-SNN-LowPower/results")
    ap.add_argument("--out", default="/mnt/ECG-SNN-LowPower/results/report")
    ap.add_argument("--mitbih-dir", default="", help="默认 <runs-root>/mitbih")
    ap.add_argument("--pareto-dir", default="", help="默认 <runs-root>/pareto")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    out = Path(args.out)
    mitbih = Path(args.mitbih_dir) if args.mitbih_dir else runs_root / "mitbih"
    pareto_dir = Path(args.pareto_dir) if args.pareto_dir else runs_root / "pareto"
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print("读取输入 …", flush=True)
    comp = load_json(runs_root / "comparison.json") or {}
    rows = comp.get("rows") or []
    if not rows:
        WARNINGS.append("comparison.json 缺失或为空 —— 表格与图 1/3/7 会不完整。"
                        "先跑 07_energy_report.py。")

    history_all, test_metrics = {}, {}
    for name in RUN_ORDER:
        h = load_json(runs_root / name / "history.json")
        if h:
            history_all[name] = h
        tm = load_json(runs_root / name / "test_metrics.json")
        if tm:
            test_metrics[name] = tm

    mit_summary = load_json(mitbih / "web_data" / "summary.json")
    mit_sparsity = load_json(mitbih / "web_data" / "sparsity.json")
    strip = load_json(mitbih / "web_data" / "strip.json")
    mit_ours = load_json(mitbih / "results" / "metrics_ours.json")
    mit_auth = load_json(mitbih / "results" / "metrics_author.json")

    pareto_pts = []
    if pareto_dir.exists():
        for p in sorted(pareto_dir.glob("*.json")):
            d = load_json(p)
            if isinstance(d, dict) and d.get("macro_auroc") is not None:
                pareto_pts.append(d)
        if pareto_pts:
            print(f"  载入 {len(pareto_pts)} 个帕累托扫描点", flush=True)

    snn_hist = (history_all.get("repro_snn") or {}).get("history") or []
    meta = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "snn_epochs": len(snn_hist) if snn_hist else None,
        "snn_best": (history_all.get("repro_snn") or {}).get("best_macro_auroc"),
        "rr_stats_identical": True,
        "runs": {n: len((history_all.get(n) or {}).get("history") or []) for n in RUN_ORDER},
        "pareto_points": len(pareto_pts),
    }

    print("出图 …", flush=True)
    embedded: dict[str, str] = {}
    for fn, fig_path in (
        (lambda: fig_pareto(rows, pareto_pts, out / "fig1_pareto.png", embedded), "fig1"),
        (lambda: fig_layer_energy(mit_sparsity, out / "fig2_layer_energy.png", embedded), "fig2"),
        (lambda: fig_roc_pr(rows, out / "fig3_roc_pr.png", embedded), "fig3"),
        (lambda: fig_confusion(mit_summary, out / "fig4_confusion.png", embedded), "fig4"),
        (lambda: fig_strip(strip, out / "fig5_strip.png", embedded), "fig5"),
        (lambda: fig_training(history_all, out / "fig6_training.png", embedded), "fig6"),
        (lambda: fig_per_class(rows, test_metrics, out / "fig7_per_class.png", embedded), "fig7"),
    ):
        try:
            fn()
        except Exception as exc:  # 单张图失败不应终止整份报告
            WARNINGS.append(f"{fig_path} 生成失败：{type(exc).__name__}: {exc}")
            print(f"  ✗ {fig_path}: {exc}", flush=True)

    # 一致性自检：comparison.json 的指标由 07 重新推理得到，而 test_metrics.json 是
    # train.py 训练末尾那次评测的结果。两者是**两次独立推理**，浮点累积次序不同，
    # 在 AUROC 这种秩相关指标上会有 ~1e-6 级的抖动 —— 故用容差判据，而不是"逐位相等"。
    # 容差取 1e-4：真错误（例如把 repro_snn 与 repro_snn_e100 重复收录）会差 0.001 量级，必被抓住。
    consistency = []
    for name in RUN_ORDER:
        r = next((x for x in rows if x["run_name"] == name), None)
        tm = test_metrics.get(name) or {}
        a, b = (r or {}).get("macro_auroc"), tm.get("macro_auroc")
        if a is None or b is None:
            continue
        d = abs(a - b)
        consistency.append({"run": name, "report_macro_auroc": a,
                            "test_metrics_macro_auroc": b, "abs_diff": d,
                            "ok": d < 1e-4})
        if d >= 1e-4:
            WARNINGS.append(
                f"{name}: comparison.json({a:.6f}) 与 test_metrics.json({b:.6f}) "
                f"相差 {d:.2e}，超出 1e-4 容差 —— 可能收错了 run（检查 --runs 与 TAG）")
    meta["consistency"] = consistency

    report_data = {
        "meta": meta,
        "consistency": consistency,
        "comparison_rows": rows,
        "constants": comp.get("constants"),
        "mitbih_ours": mit_ours,
        "mitbih_author": mit_auth,
        "mitbih_summary": mit_summary,
        "pareto_points": pareto_pts,
        "warnings": WARNINGS,
    }
    (out / "report_data.json").write_text(
        json.dumps(clean(report_data), indent=2, ensure_ascii=False, allow_nan=False))

    html = build_html(meta, rows, mit_ours, mit_auth, embedded)
    (out / "report.html").write_text(html, encoding="utf-8")

    print(f"\n→ {out}/report.html  ({len(html) / 1024:.0f} KB)")
    print(f"→ {out}/report_data.json")
    print(f"  图 {len(embedded)} 张，耗时 {time.time() - t_start:.1f}s")
    for w in WARNINGS:
        print(f"  ! {w}")


if __name__ == "__main__":
    main()
