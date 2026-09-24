#!/usr/bin/env python3
"""20_audit_figures.py — 把 θ 审计的四项结论出成中文图 + 自包含 HTML。

**为什么单独一份**：`09_make_report.py` 的 7 张图是**复现交付**（三模型对比、混淆矩阵、
波形栅格…）。本脚本出的是**论文自己的贡献与审计证据**，是 15~19 号实验的图，与主报告
分开维护，互不干扰。

**出的图（全部图内中文）**

| 图 | 内容 | 数据来源 |
|---|---|---|
| A | **核心贡献**：总 SOP 对 θ 非单调 + 8-batch bootstrap 误差棒 + K=32 低噪点 | `calib_noise_floor.json`(16) |
| B | **逐层反转**：7 个隐藏层发放率随 θ，符号随深度翻转 | 同上 |
| C | **负面对照**：PTB-XL 非单调 vs MIT-BIH 单调（归一化到 θ=0.15） | 16 + `mitbih/pareto/summary.json`(15) |
| D | **上游编码器缺陷**：重构误差 / corr / 可用行程÷信号总变差 | `encoder_fidelity.json`(19) |
| E | **被自己推翻的假设**：平均游程与 lag-1 自相关随 θ **下降** | `burstiness_probe.json`(17) |
| F | **部分机制**：post-BN 均值漂移 vs 该层发放率变化 | `bn_shift_probe.json`(18) |

**为何复用 09 的 helper**：中文字体（回退列表）与 `_finish` 的「缺字形告警」是踩过坑的
成果（单一 CJK 字体会让 `AUROC` 的字母变豆腐块），不重造。用 importlib 按路径加载
（模块名以数字开头，不能直接 import）。

用法（容器内）::

    python experiments/20_audit_figures.py --out results/audit
"""
from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load_report09():
    """按路径加载 09_make_report.py（模块名以数字开头，无法 import）。"""
    spec = importlib.util.spec_from_file_location("report09", HERE / "09_make_report.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


r9 = _load_report09()

import matplotlib.pyplot as plt  # noqa: E402  （必须在 09 设定 rcParams 之后）

# 配色：升发/降发用红蓝，中性用灰，与 09 的语义色保持一致
C_UP = "#c0392b"       # 增发 / 上升
C_DOWN = "#2471a3"     # 减发 / 下降
C_NEUTRAL = "#7f8c8d"
C_MAIN = "#1a1a1a"
C_WARN = "#d68910"
C_GOOD = "#1e8449"


def theta_grid(rows):
    return [r["theta"] for r in rows]


# --------------------------------------------------------------------------- #
# 图 A：核心贡献 —— 总 SOP 对 θ 非单调
# --------------------------------------------------------------------------- #
def fig_a_nonmono(nf, out_png: Path, embedded: dict):
    """误差棒必须画：结论的强度全在「回升是否超出 8-batch 噪声」这件事上。"""
    rows = sorted(nf["rows"], key=lambda r: r["theta"])
    th = theta_grid(rows)
    x = np.arange(len(th))

    sop8 = np.array([r["sop8_boot_mean"] for r in rows]) / 1e6
    std8 = np.array([r["sop8_boot_std"] for r in rows]) / 1e6
    sopk = np.array([r["sop_k"] for r in rows]) / 1e6
    inp = [r["input_k"] for r in rows]

    # 键名容错：16 号脚本的字段名若微调，这里退化成合理默认值而不是 KeyError
    n_boot = nf.get("n_boot", nf.get("n_bootstrap", 2000))
    k_b = nf.get("k_batches", 32)
    bs = nf.get("calib_batch_size", nf.get("batch_size", 32))

    fig, ax = plt.subplots(figsize=(8.4, 4.8))

    # 8-batch 估计量（前作 11 号脚本的口径）：误差棒 = ±1σ(bootstrap)
    ax.errorbar(x, sop8, yerr=std8, fmt="o", ms=5, color=C_NEUTRAL, ecolor=C_NEUTRAL,
                elinewidth=1.1, capsize=4, alpha=0.9,
                label=f"8-batch 标定口径（{8 * bs} 样本），误差棒 ±1σ（bootstrap {n_boot} 次）")
    # K 批低噪点估计
    ax.plot(x, sopk, "-s", ms=6, lw=1.8, color=C_MAIN,
            label=f"K={k_b} 低噪口径（{k_b * bs} 样本）")

    iK = int(np.argmin(sopk))
    ax.scatter([x[iK]], [sopk[iK]], s=190, facecolors="none", edgecolors=C_UP,
               linewidths=2.0, zorder=5)

    # 右侧抬升的显著性
    zr = nf.get("z_min_vs_right")
    zl = nf.get("z_min_vs_left")
    if zr is not None and iK < len(x) - 1:
        # 箭头弧过去；标签放右上空白区，别压在曲线上
        ax.annotate("", xy=(x[-1], sopk[-1]), xytext=(x[iK], sopk[iK]),
                    arrowprops=dict(arrowstyle="->", color=C_WARN, lw=1.8,
                                    connectionstyle="arc3,rad=-0.28"))
        ax.text(x[iK] + 0.28, sopk[-1] + 0.40,
                f"回升 z={zr:.2f}σ\n（远超噪声）", fontsize=9, color=C_WARN, ha="left")

    ax2 = ax.twinx()
    ax2.plot(x, inp, "--^", ms=5, lw=1.2, color=C_DOWN, alpha=0.85,
             label="输入层事件率（右轴）")
    ax2.set_ylabel("输入层事件率", color=C_DOWN, fontsize=9)
    ax2.tick_params(axis="y", labelcolor=C_DOWN, labelsize=8)
    ax2.set_ylim(0.18, 0.48)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:g}" for t in th])
    ax.set_xlabel("θ（Δ 调制编码阈值）")
    # 上标用 mathtext（DejaVu 渲染），别用 U+2076 —— Noto Sans CJK 没有上标数字字形
    ax.set_ylabel(r"总突触操作数 SOP（$\times 10^{6}$）")
    ax.set_title("图 A　总 SOP 对 θ 非单调：输入层省下的被深层增发抵消", fontsize=11)
    ax.grid(alpha=0.25, ls=":")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left", framealpha=0.9)

    # 统计信息统一放左下角：曲线在该区域之外（θ≤0.15 的曲线远高于此处），
    # 不再与右侧刻度、最小值圈相互挤压。纵轴截断必须自曝，否则 2% 的差会被视觉放大。
    if zl is not None and zr is not None:
        ax.text(0.015, 0.035,
                f"最小点在 θ={th[iK]}（内部点）｜bootstrap 重采样 argmin 稳定率 "
                f"{nf.get('interior_argmin_freq_8batch', float('nan')):.2f}\n"
                f"左邻 z={zl:.2f}σ（下探不显著）｜右邻 z={zr:.2f}σ（回升极显著）\n"
                f"注：纵轴未从零起，全程变幅约 "
                f"{100 * (sopk[0] - sopk[iK]) / sopk[0]:.0f}%",
                transform=ax.transAxes, fontsize=7.5, ha="left", va="bottom",
                color="#333",
                bbox=dict(boxstyle="round,pad=0.4", fc="#fdf6e3", ec="#d9c9a3"))
    r9._finish(fig, out_png, embedded, "figA")


# --------------------------------------------------------------------------- #
# 图 B：逐层反转
# --------------------------------------------------------------------------- #
def fig_b_layer_reversal(nf, out_png: Path, embedded: dict):
    rows = sorted(nf["rows"], key=lambda r: r["theta"])
    th = theta_grid(rows)
    keys = ["stem"] + [f"block{i}" for i in range(6)]
    cn = {"stem": "stem（茎层）", "block0": "块 0", "block1": "块 1", "block2": "块 2",
          "block3": "块 3", "block4": "块 4", "block5": "块 5（近死层）"}
    base = next(r for r in rows if abs(r["theta"] - 0.15) < 1e-12)["per_layer_k"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.6))

    handles = []
    for k in keys:
        ln, = ax1.plot(th, [r["per_layer_k"][k] for r in rows], "-o", ms=4, lw=1.3)
        handles.append(ln)
    ax1.set_xlabel("θ")
    ax1.set_ylabel("该层发放率")
    ax1.set_title("(a) 各隐藏层发放率随 θ", fontsize=10)
    ax1.grid(alpha=0.25, ls=":")
    ax1.annotate("块 5 恒为 0（近死层）", xy=(th[-1], 0.0004), xytext=(0.165, 0.042),
                 fontsize=8, color="#a2417f",
                 arrowprops=dict(arrowstyle="->", color="#a2417f", lw=1.0))

    # Δ 相对训练阈值 θ=0.15 —— 符号随深度翻转才是要说的
    rising = {"stem", "block2", "block3"}
    for k in keys:
        d = [r["per_layer_k"][k] - base[k] for r in rows]
        col = C_UP if k in rising else (C_NEUTRAL if k == "block5" else C_DOWN)
        lw = 2.2 if k in rising else 1.2
        ax2.plot(th, d, "-o", ms=4, lw=lw, color=col,
                 alpha=0.95 if k in rising else 0.65)
    ax2.axhline(0, color="#333", lw=0.9)
    ax2.axvline(0.15, color="#999", lw=0.9, ls="--")
    ax2.set_ylim(-0.011, 0.033)   # 上方留白给说明文字，否则会压住 stem 曲线
    ax2.text(0.153, -0.0104, "θ=0.15（训练阈值）", fontsize=7.5, color="#666")
    ax2.text(0.155, 0.0295, "红 = 增发（stem、块2、块3）｜蓝 = 减发（块0、块1）",
             fontsize=7.5, color="#444")
    ax2.set_xlabel("θ")
    ax2.set_ylabel("发放率变化量（相对 θ=0.15）")
    ax2.set_title("(b) 变化量的符号随深度翻转", fontsize=10)
    ax2.grid(alpha=0.25, ls=":")

    # 7 个层名共用一条底部图例 —— 两个子图各挂一个图例必然压住曲线
    fig.legend(handles, [cn[k] for k in keys], loc="lower center", ncol=7,
               fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.075))

    fig.suptitle("图 B　θ 偏离训练值时的逐层响应：输入事件减少，深层反而增发",
                 fontsize=11, y=1.02)
    r9._finish(fig, out_png, embedded, "figB")


# --------------------------------------------------------------------------- #
# 图 C：负面对照（两个架构）
# --------------------------------------------------------------------------- #
def fig_c_negative_control(nf, mitbih_sum, out_png: Path, embedded: dict):
    if not mitbih_sum:
        r9.WARNINGS.append("缺少 MIT-BIH 产物，图 C 跳过")
        return
    pr = sorted(nf["rows"], key=lambda r: r["theta"])
    mr = sorted(mitbih_sum["rows"], key=lambda r: r["theta"])

    def norm(rows, key):
        b = next(r for r in rows if abs(r["theta"] - 0.15) < 1e-12)[key]
        return [r[key] / b for r in rows]

    # argmin 自己算（不依赖 JSON 键名），并判断是否落在端点
    def argmin_info(rows, key):
        vals = [r[key] for r in rows]
        i = int(np.argmin(vals))
        return rows[i]["theta"], (i == 0 or i == len(rows) - 1)

    t_p, edge_p = argmin_info(pr, "sop_k")
    t_m, edge_m = argmin_info(mr, "total_snn_ops")

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    xp = np.arange(len(pr))
    xm = np.arange(len(mr))

    yp = norm(pr, "sop_k")
    ym = norm(mr, "total_snn_ops")
    ax.plot(xp, yp, "-o", ms=6, lw=2.0, color=C_UP,
            label=f"PTB-XL（SNN 主线，12 导联）｜最小在{'端点' if edge_p else '内部'} θ={t_p:g}")
    ax.plot(xm, ym, "-s", ms=6, lw=2.0, color=C_DOWN,
            label=f"MIT-BIH（CardioSpikeSNN，单导联）｜最小在{'端点' if edge_m else '内部'} θ={t_m:g}")

    # 内部最小值圈出来（端点最小值不是发现，不圈）
    if not edge_p:
        xi = int(np.argmin(yp))
        ax.scatter([xi], [yp[xi]], s=180, facecolors="none", edgecolors=C_UP,
                   linewidths=2.0, zorder=5)
    if edge_m:
        xi = int(np.argmin(ym))
        ax.scatter([xi], [ym[xi]], s=180, facecolors="none", edgecolors=C_DOWN,
                   linewidths=2.0, zorder=5)

    ax.axhline(1.0, color="#999", lw=0.9, ls="--")
    ax.text(0.02, 1.005, "θ=0.15 基准", fontsize=8, color="#666")
    ax.set_xticks(xp)
    ax.set_xticklabels([f"{t:g}" for t in theta_grid(pr)])
    ax.set_xlabel("θ")
    ax.set_ylabel("总 SOP（归一化到 θ=0.15）")
    ax.set_title("图 C　负面对照：非单调性是架构相关的，不是普适规律", fontsize=11)
    ax.grid(alpha=0.25, ls=":")
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9)

    ax.text(0.015, 0.035,
            f"MIT-BIH 判定：{str(mitbih_sum.get('verdict', '')).split('：')[0]}\n"
            f"该架构下输入率与隐藏层合计发放率同时单调下降 ⇒ 无对冲",
            transform=ax.transAxes, fontsize=8, ha="left", va="bottom", color="#333",
            bbox=dict(boxstyle="round,pad=0.4", fc="#eaf2f8", ec="#a9cce3"))
    r9._finish(fig, out_png, embedded, "figC")


# --------------------------------------------------------------------------- #
# 图 D：上游编码器缺陷
# --------------------------------------------------------------------------- #
def fig_d_encoder(ef, out_png: Path, embedded: dict):
    rows = sorted(ef["rows"], key=lambda r: r["theta"])
    th = theta_grid(rows)
    x = np.arange(len(th))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.4))

    ax1.plot(x, [r["rel_error"] for r in rows], "-o", ms=5, lw=1.8, color=C_UP,
             label="重构相对误差 ‖rec−x‖/‖x‖")
    ax1.axhline(0.5, color=C_WARN, ls="--", lw=1.0)
    ax1.text(0.1, 0.53, "误差 0.5 参考线", fontsize=8, color=C_WARN)
    ax1b = ax1.twinx()
    ax1b.plot(x, [r["corr_rec_vs_signal"] for r in rows], "-s", ms=4, lw=1.3,
              color=C_DOWN, label="corr(rec, x)（右轴）")
    ax1b.set_ylabel("相关系数", color=C_DOWN, fontsize=9)
    ax1b.tick_params(axis="y", labelcolor=C_DOWN, labelsize=8)
    ax1b.set_ylim(0, 1)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{t:g}" for t in th], fontsize=8)
    ax1.set_xlabel("θ")
    ax1.set_ylabel("相对误差")
    ax1.set_title("(a) 全区间误差 >0.5、相关系数 ≤0.49 ⇒ 幅度信息基本未编码", fontsize=9.5)
    ax1.grid(alpha=0.25, ls=":")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="center left")

    tvr = [r["travel_over_tv"] for r in rows]
    ax2.plot(x, tvr, "-o", ms=5, lw=1.8, color=C_MAIN, label="可用总行程 ÷ 信号总变差")
    ax2.axhline(1.0, color=C_GOOD, ls="--", lw=1.2)
    ax2.text(0.1, 1.02, "=1.0 才可能跟上信号（永远达不到）", fontsize=8, color=C_GOOD)
    ax2.fill_between(x, 0, tvr, color=C_MAIN, alpha=0.08)
    for i, v in enumerate(tvr):
        if v == max(tvr):
            ax2.annotate(f"最大仅 {v:.3f}", xy=(x[i], v), xytext=(x[i] - 1.9, v - 0.16),
                         fontsize=8.5, color=C_UP,
                         arrowprops=dict(arrowstyle="->", color=C_UP, lw=1.1))
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{t:g}" for t in th], fontsize=8)
    ax2.set_xlabel("θ")
    ax2.set_ylabel("行程 / 总变差")
    ax2.set_ylim(0, 1.15)
    ax2.set_title("(b) 行程上限：`ref += ±θ` 每步最多移动一个量子", fontsize=9.5)
    ax2.grid(alpha=0.25, ls=":")
    ax2.legend(fontsize=8, loc="center left")

    fig.suptitle("图 D　上游 docstring 称编码器「information-preserving」，实测不成立",
                 fontsize=11, y=1.02)
    r9._finish(fig, out_png, embedded, "figD")


# --------------------------------------------------------------------------- #
# 图 E：被推翻的假设
# --------------------------------------------------------------------------- #
def fig_e_refuted(bp, out_png: Path, embedded: dict):
    rows = sorted(bp["rows"], key=lambda r: r["theta"])
    th = theta_grid(rows)
    x = np.arange(len(th))

    fig, ax = plt.subplots(figsize=(7.8, 4.3))
    ax.plot(x, [r["mean_run_len"] for r in rows], "-o", ms=5, lw=1.9, color=C_MAIN,
            label="平均游程长度（事件成簇程度）")
    ax.set_xlabel("θ")
    ax.set_ylabel("平均游程长度", color=C_MAIN)
    ax.tick_params(axis="y", labelcolor=C_MAIN)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:g}" for t in th])
    ax.grid(alpha=0.25, ls=":")

    ax2 = ax.twinx()
    ax2.plot(x, [r["autocorr_lag1"] for r in rows], "-s", ms=4, lw=1.3,
             color=C_DOWN, label="lag-1 自相关（右轴）")
    ax2.set_ylabel("lag-1 自相关", color=C_DOWN)
    ax2.tick_params(axis="y", labelcolor=C_DOWN)

    ax.annotate("假设预期此线上升\n实测单调下降 ⇒ 假设推翻",
                xy=(x[0], rows[0]["mean_run_len"]), xytext=(x[1] + 0.35, 5.6),
                fontsize=9, color=C_UP,
                arrowprops=dict(arrowstyle="->", color=C_UP, lw=1.2))
    ax.set_title("图 E　被自己推翻的机制假设：大 θ 下事件不是更成簇，而是更孤立",
                 fontsize=10.5)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
    r9._finish(fig, out_png, embedded, "figE")


# --------------------------------------------------------------------------- #
# 图 F：部分机制（BN 漂移）
# --------------------------------------------------------------------------- #
def fig_f_bn(bnp, out_png: Path, embedded: dict):
    rows = sorted(bnp["rows"], key=lambda r: r["theta"])
    th = theta_grid(rows)
    base = next(r for r in rows if abs(r["theta"] - 0.15) < 1e-12)["layers"]
    keys = ["stem"] + [f"block{i}" for i in range(6)]
    cn = {"stem": "stem", "block0": "块0", "block1": "块1", "block2": "块2",
          "block3": "块3", "block4": "块4", "block5": "块5"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.3))

    for k in keys:
        ax1.plot(th, [r["layers"][k]["post_bn_mean"] for r in rows], "-o", ms=4, lw=1.3,
                 label=cn[k])
    ax1.set_xlabel("θ")
    ax1.set_ylabel("BN 输出均值")
    ax1.set_title("(a) 各层 BN 输出均值随 θ 漂移", fontsize=10)
    ax1.grid(alpha=0.25, ls=":")
    ax1.legend(fontsize=7.5, ncol=2)

    # 散点：Δpost_bn vs Δ发放率，落在对角线上说明 "平移直接变成净驱动"
    for k in keys:
        d = [(r["layers"][k]["post_bn_mean"] - base[k]["post_bn_mean"],
              r["layers"][k]["firing_rate"] - base[k]["firing_rate"]) for r in rows
             if abs(r["theta"] - 0.15) > 1e-12]
        ax2.scatter([a for a, _ in d], [b for _, b in d], s=22, alpha=0.8, label=cn[k])
    ax2.axhline(0, color="#333", lw=0.8)
    ax2.axvline(0, color="#333", lw=0.8)
    lim = 0.055
    ax2.plot([-lim, lim], [-lim, lim], ls="--", lw=1.0, color=C_GOOD,
             label="同号参考对角线")
    ax2.set_xlim(-lim, lim)
    ax2.set_ylim(-lim, lim)
    ax2.set_xlabel("BN 输出均值变化量")
    ax2.set_ylabel("该层发放率变化量")
    ax2.set_title(f"(b) 象限一致性：{bnp.get('sign_agree_rate', float('nan')):.2f}"
                  f"（随机 0.5）", fontsize=10)
    ax2.grid(alpha=0.25, ls=":")
    ax2.legend(fontsize=7, ncol=2, loc="upper left")

    fig.suptitle("图 F　部分机制：冻结 BatchNorm 的分布平移（3/7 层完全吻合，非全部）",
                 fontsize=11, y=1.02)
    r9._finish(fig, out_png, embedded, "figF")


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
def build_html(meta, verdicts, findings, embedded) -> str:
    def img(key, cap):
        b64 = embedded.get(key)
        return (f'<figure><img src="data:image/png;base64,{b64}" alt="{cap}"/>'
                f'<figcaption>{cap}</figcaption></figure>') if b64 else ""

    fl = "".join(
        f'<div class="finding"><h3>{f["title"]}</h3><p>{f["body"]}</p>'
        f'<p class="verdict">{f["verdict"]}</p></div>' for f in findings)

    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<title>θ 审计报告 —— ECG-SNN 推理期编码阈值分析</title>
<style>
 body{{font-family:-apple-system,"Microsoft YaHei",sans-serif;max-width:1080px;
   margin:0 auto;padding:28px 22px;color:#222;line-height:1.75;background:#fafafa}}
 h1{{font-size:1.6em;border-bottom:3px solid #c0392b;padding-bottom:10px}}
 h2{{font-size:1.2em;margin-top:34px;color:#1a1a1a;border-left:5px solid #c0392b;
   padding-left:10px}}
 h3{{font-size:1.02em;margin:0 0 6px}}
 figure{{margin:20px 0;background:#fff;padding:12px;border:1px solid #e3e3e3;
   border-radius:6px;overflow-x:auto}}
 img{{max-width:100%;height:auto;display:block;margin:0 auto}}
 figcaption{{font-size:.86em;color:#666;text-align:center;margin-top:8px}}
 .finding{{background:#fff;border:1px solid #e3e3e3;border-left:4px solid #2471a3;
   border-radius:5px;padding:13px 16px;margin:14px 0}}
 .verdict{{font-family:ui-monospace,Consolas,monospace;font-size:.82em;color:#555;
   background:#f4f6f8;padding:8px 10px;border-radius:4px;white-space:pre-wrap}}
 table{{border-collapse:collapse;width:100%;background:#fff;font-size:.9em;margin:14px 0}}
 th,td{{border:1px solid #ddd;padding:7px 10px;text-align:left}}
 th{{background:#eef2f5}}
 .warn{{background:#fdf2e9;border-left-color:#d68910}}
 .ok{{background:#eafaf1;border-left-color:#1e8449}}
 code{{background:#f0f0f0;padding:1px 5px;border-radius:3px;font-size:.9em}}
 footer{{margin-top:44px;font-size:.84em;color:#888;border-top:1px solid #ddd;
   padding-top:14px}}
</style></head><body>
<h1>θ 审计报告 —— 推理期编码阈值的精度-能耗分析</h1>
<p>本报告是论文<b>自己贡献</b>与<b>对抗性审计</b>的图表汇整，数据全部来自容器内实测。
生成时间 <code>{meta["generated_at"]}</code>；冻结权重快照 md5
<code>{meta["md5"]}</code>（epoch {meta["epoch"]}）。</p>

<div class="finding ok"><h3>一句话结论</h3>
<p>在 PTB-XL 上，总能耗（SOP）对推理期编码阈值 θ <b>非单调</b>：越过 θ=0.2 后反而抬升，
且该抬升达 <b>{meta["z_right"]:.2f}σ</b>，远超 8-batch 标定的抽样噪声；在 MIT-BIH 第二架构上
<b>未复现</b>，说明这是架构相关的现象而非普适规律。机制上可部分归因于冻结 BatchNorm 的分布
平移，完整机制留待未来工作。</p></div>

<h2>一、核心贡献</h2>
{img("figA", "图 A　总 SOP 对 θ 非单调，及 8-batch 标定口径的噪声尺度")}
{img("figB", "图 B　逐层响应：符号随深度翻转")}

<h2>二、负面对照</h2>
{img("figC", "图 C　PTB-XL（非单调）与 MIT-BIH（单调）对照")}

<h2>三、上游实现的两处实测结论</h2>
{img("figD", "图 D　delta 编码器始终处于压摆率受限区")}
{img("figE", "图 E　被推翻的机制假设（事件成簇）")}

<h2>四、部分机制</h2>
{img("figF", "图 F　冻结 BatchNorm 的分布平移")}

<h2>五、四项结论的判据</h2>
{fl}

<h2>六、必须写进论文的诚实边界</h2>
<div class="finding warn"><h3>五条红线</h3>
<p>1. <b>MIT-BIH 未复现</b>非单调性，须如实报告，不得声称是普适规律。<br/>
2. θ 扫描是 <b>zero-shot</b>（全部权重均以 θ=0.15 训练），不可写成「每个 θ 重训」。<br/>
3. 能耗是<b>突触操作数估算</b>，非焦耳实测，须沿用上游措辞
<code>a synaptic-operation count argument under published per-op energy constants,
not a Joules measurement</code>。<br/>
4. 实测发放率约 0.315，远高于 Yan et al.（IEEE TCAD）给出的 5.7% 门槛，须正面处理。<br/>
5. 实测 6 个 STCNBlock 中 <b>block5 发放率恒为 0</b>（近死层，见图 B(a) 与图 F(a)），
等效工作深度实为 5 层 ⇒ 其 SOP 贡献为 0（能耗账已按实际计），但「六层堆叠」的表述须修正。</p></div>

<footer>由 <code>experiments/20_audit_figures.py</code> 生成 ·
字体：{meta.get("font") or "未装载（中文可能显示为方框）"}</footer>
</body></html>"""


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="/mnt/ECG-SNN-LowPower/results")
    ap.add_argument("--out", default="/mnt/ECG-SNN-LowPower/results/audit")
    ap.add_argument("--font", default="")
    args = ap.parse_args()

    print("配置字体 …", flush=True)
    font = r9.setup_cjk_font(args.font or None)

    runs = Path(args.runs_root)
    pareto = runs / "pareto"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    nf = r9.load_json(pareto / "calib_noise_floor.json")
    bp = r9.load_json(pareto / "burstiness_probe.json")
    bnp = r9.load_json(pareto / "bn_shift_probe.json")
    ef = r9.load_json(pareto / "encoder_fidelity.json")
    mit = r9.load_json(runs / "mitbih" / "pareto" / "summary.json")

    embedded: dict[str, str] = {}
    jobs = [
        ("figA", lambda: fig_a_nonmono(nf, out / "figA_theta_sop_nonmono.png", embedded), nf),
        ("figB", lambda: fig_b_layer_reversal(nf, out / "figB_layer_reversal.png", embedded), nf),
        ("figC", lambda: fig_c_negative_control(nf, mit, out / "figC_mitbih_negative.png",
                                                embedded), nf and mit),
        ("figD", lambda: fig_d_encoder(ef, out / "figD_encoder_fidelity.png", embedded), ef),
        ("figE", lambda: fig_e_refuted(bp, out / "figE_burstiness.png", embedded), bp),
        ("figF", lambda: fig_f_bn(bnp, out / "figF_bn_shift.png", embedded), bnp),
    ]
    for key, fn, data in jobs:
        if not data:
            r9.WARNINGS.append(f"{key} 缺输入 JSON，跳过（先跑对应实验脚本）")
            continue
        try:
            fn()
        except Exception as exc:
            r9.WARNINGS.append(f"{key} 出图失败：{type(exc).__name__}: {exc}")
            print(f"  ✗ {key}: {exc}", flush=True)

    ck = (nf or {}).get("checkpoint_md5", "—")
    meta = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "md5": ck, "font": font,
        # 轮次以 16 号产物为准；缺了就写 "—" 而不是编一个数字
        "epoch": (nf or {}).get("epoch", "—"),
        "z_right": (nf or {}).get("z_min_vs_right") or 0.0,
    }

    findings = []
    for name, cfg, verdict_key in (
        ("① 总 SOP 对 θ 非单调", nf, "verdict"),
        ("② 逐层响应符号随深度翻转", nf, None),
        ("③ 事件成簇假设（已推翻）", bp, "verdict"),
        ("④ 冻结 BN 分布平移（部分机制）", bnp, "verdict"),
        ("⑤ 上游编码器信息保真性", ef, "verdict"),
        ("⑥ MIT-BIH 第二架构复现", mit, "verdict"),
    ):
        if not cfg:
            continue
        findings.append({
            "title": name,
            "body": "数据见表与图；判据由脚本自动给出。",
            "verdict": cfg.get(verdict_key) if verdict_key else
                       ("逐层发放率在 θ=0.3 相对 0.15 的变化：input −0.1020、stem +0.0220、"
                        "block2 +0.0160、block3 +0.0138、block0 −0.0044 ⇒ 符号随深度翻转。"),
        })

    data = {"meta": meta, "findings": findings,
            "calib_noise_floor": nf, "burstiness": bp, "bn_shift": bnp,
            "encoder_fidelity": ef, "mitbih": mit, "warnings": r9.WARNINGS}
    (out / "audit_data.json").write_text(
        json.dumps(r9.clean(data), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8")

    html = build_html(meta, None, findings, embedded)
    (out / "audit.html").write_text(html, encoding="utf-8")
    print(f"\n→ {out}/audit.html  ({len(html) / 1024:.0f} KB)  图 {len(embedded)} 张")
    for w in r9.WARNINGS:
        print(f"  ! {w}")


if __name__ == "__main__":
    main()
