#!/usr/bin/env python3
"""精度-能耗帕累托扫描：θ（Δ编码阈值）× T_dense（分类头积分步数）。

**为什么可行（两轴都是纯推理期参数，零重训、零梯度）**
  - θ：`NeuroCardio.forward(x, theta=0.15)` → `encoding.delta_encode_batch_12lead`。
    θ 不在 `state_dict`，且编码后通道数恒为 2×12=24（`stem_conv.in_channels=24` 硬编码）
    ⇒ 权重形状不变，任何 θ 下都可直接 load。
  - T_dense：`NeuroCardio.__init__(..., T_dense=8)` 只是个 int 属性，不进 state_dict，
    只决定分类头 `for _ in range(self.T_dense)` 的循环次数，输出形状恒为 (B,5)。
    `compute.cost_snn` 从 `model.T_dense` 读它（`compute.py:203`）来算头部 MAC。
  ⇒ 扫描 = 用同一份权重构造不同 `T_dense` 的模型 + 不同 θ 重跑编码。

**⚠️ 学术诚信红线（必须随结果一起声明）**
  `train.py:57` 的 `theta` 字段全文只出现一次，训练时是 `model(x)`（`train.py:298`）
  **不传 theta** ⇒ 所有 checkpoint 都是在 **θ=0.15** 下训练的。因此本扫描是
  「**零训练迁移的推理期编码器灵敏度分析**」，**不是**"每个 θ 下的最优模型"。
  精度下降同时含「编码失配」与「信息损失」两种效应，不可解释为"θ 的最优取值"。

**能耗机制**：`cost_snn` 的 MAC 是解析常量，唯一受影响的乘子是**实测**发放率
  `r = spike_rates["input"]`，`sops = round(macs * r)` ⇒ θ↑ → 发放率↓ → SOP↓ 而精度↓。
  T_dense 则线性乘在 dense head 的 MAC 上。故**每个点都必须重跑 `calibrate_spike_rates`**。

**⚠️ 更正：batch 划分不影响结论（2026-09-24 实测推翻本文件先前的说法）**
  本脚本早先版本在此断言「上游 `snn.Leaky` 的膜电位跨 batch 泄漏，导致评估结果依赖
  batch 划分」——**那个判断是错的**，机制描述也是错的。真实情况：snntorch 1.0.0 的
  `Leaky.init_leaky()` 只是 `reset_mem()` 的**别名**，实现为
  `self.mem = torch.zeros_like(self.mem); return self.mem` ⇒ **每次调用都先清零**，
  故 `_run_lif_along_time` 每个 batch 都从零起步，**不存在跨 batch 继承**。
  `14_leak_audit.py` 的裁断实测（同一份冻结 ep30 权重、fold-10 全量、θ=0.15/T=8）：
  A1（原生路径）＝A2（同一 model 实例紧接着连跑第二次）＝B（每 batch 显式重置）
  ＝ **0.775746**，三者 AUROC/AUPRC/Σp **逐位完全相同**；bs=96 下同理（0.775529，
  与 bs=64 仅差 2.2e-4，是最后一批不满的数值噪声）。
  ⇒ 保留 `--batch-size`=64 与标定 batch=32 只是为了与 `07_energy_report.py` 的产出
  逐位可比，**不是因为有 batch 耦合**。**论文中不要再声称存在该缺陷。**
  另：先前把 07 的 `0.498530` 当成泄漏证据，实为**权重漂移**——那份 `comparison.json`
  生成时 `best.pt` 还是 ep0（val≈0.5124≈随机）。

用法::

    CUDA_VISIBLE_DEVICES=3 python3 experiments/11_pareto_sweep.py \\
        --run repro_snn --out results/pareto \\
        --baseline results/comparison.json

    # 只跑基线单点，用于与 07 对拍（最强的正确性检查）
    CUDA_VISIBLE_DEVICES=3 python3 experiments/11_pareto_sweep.py --theta 0.15 --t-dense 8

注意：`compute.time_inference` 与 `compute.calibrate_spike_rates` **不接受 theta 参数**
（内部写死 `model(x)`），故本脚本用 `bind_theta()` 把 θ 绑到模型实例的 `forward` 上，
不改上游代码（上游是只读底本）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

NC = Path("/mnt/ECG-SNN-LowPower/third_party/neurocardio")
sys.path.insert(0, str(NC))

from neurocardio.compute import (  # noqa: E402
    ENERGY_PJ_PER_MAC_FP32, ENERGY_PJ_PER_SOP,
    calibrate_spike_rates, cost_snn, time_inference,
)
from neurocardio.dataset import PTBXLCache  # noqa: E402
from neurocardio.model import NeuroCardio  # noqa: E402
from neurocardio.train import multilabel_metrics  # noqa: E402

# 与 07_energy_report.py 保持一致：标定 loader 的 batch_size 必须是 32，
# 否则 `calibrate_spike_rates` 取到的前 8 个 batch 不是同一批样本，
# θ=0.15 的基线点就无法与 07 对拍。
CALIB_BATCH_SIZE = 32
TRAIN_THETA = 0.15   # 所有 checkpoint 的训练阈值（见模块 docstring）


# ── 工具 ──────────────────────────────────────────────────────────────
def clean(o):
    """递归把 NaN/±Inf 换成 None，并把 numpy 标量降级成 Python 原生类型。

    必要性：`multilabel_metrics` 对零支持类别写 NaN，而 `json.dump` 默认会输出
    裸 `NaN`（非法 JSON）——前端 `r.json()` 会整体抛错、整个面板静默空白。
    """
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else f
    return o


def bind_theta(model, theta: float):
    """把 θ 固定到 model 实例上。

    `compute.time_inference` / `compute.calibrate_spike_rates` 内部只调 `model(x)`，
    不透传 theta（写死 0.15）。覆盖实例属性 `forward` 是**最小侵入**的做法：
    只改调用约定，`model.T_dense` / `model.stem_conv` / `state_dict()` 等全部照旧，
    也没有 `nn.Module.__getattr__` 转发那类坑。
    """
    orig = model.forward

    def bound(x, *a, **kw):
        kw.setdefault("theta", theta)
        return orig(x, *a, **kw)

    model.forward = bound
    return model


def load_ckpt(ckpt_path: Path):
    """读检查点并校验它是 SNN；返回 (ckpt, labels, state)。"""
    if not ckpt_path.exists():
        raise SystemExit(f"找不到 {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg", {}) or {}
    model_name = cfg.get("model_name") or "snn"
    if model_name != "snn":
        raise SystemExit(f"{ckpt_path} 的 model_name={model_name}，本脚本只支持 snn")
    labels = ckpt.get("label_columns") or ["NORM", "MI", "STTC", "CD", "HYP"]
    state = ckpt.get("ema") or ckpt.get("state_dict")
    return ckpt, labels, state


def build_model(labels, state, t_dense: int):
    """用同一份权重构造指定 T_dense 的模型（T_dense 不进 state_dict，故可直接 load）。"""
    model = NeuroCardio(num_classes=len(labels), T_dense=t_dense)
    model.load_state_dict(state)
    return model


def infer_scores(model, loader, dev):
    """fold-10 全量推理，返回 (y_true, y_score, seconds)。θ 已绑在 model 上。"""
    ys, ps = [], []
    t0 = time.time()
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(dev)
            ys.append(y.numpy())
            ps.append(torch.sigmoid(model(x)).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps), time.time() - t0


# ── 主流程 ────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/mnt/ECG-SNN-LowPower/results/ptbxl_cache")
    ap.add_argument("--runs-root", default="/mnt/ECG-SNN-LowPower/results")
    ap.add_argument("--run", default="repro_snn")
    ap.add_argument("--ckpt", default="",
                    help="显式检查点路径；留空则用 <runs-root>/<run>/best.pt。"
                         "训练会持续刷新 best.pt——实测它从 ep0 刷到 ep11，"
                         "导致同一脚本两次运行得 0.4985 / 0.6705，故扫描前务必先冻结快照。")
    ap.add_argument("--out", default="/mnt/ECG-SNN-LowPower/results/pareto")
    ap.add_argument("--theta", default="0.05,0.075,0.1,0.15,0.2,0.3",
                    help="逗号分隔的编码阈值（0.15 为训练所用值=基线）")
    ap.add_argument("--t-dense", default="2,4,8,16,32",
                    help="逗号分隔的分类头积分步数（8 为模型默认=基线）")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="默认 64，与 07_energy_report.py 对齐以便逐位对拍。"
                         "实测 batch 划分不影响结论（bs=64 与 96 仅差 2.2e-4，"
                         "见 experiments/14_leak_audit.py）；本文件先前的"
                         "「膜电位跨 batch 泄漏」说法已更正为误判。")
    ap.add_argument("--n-calib-batches", type=int, default=8)
    ap.add_argument("--n-latency-iter", type=int, default=20)
    ap.add_argument("--baseline", default="/mnt/ECG-SNN-LowPower/results/comparison.json",
                    help="07 产出的 comparison.json，用于 θ=0.15/T=8 基线自洽检查")
    ap.add_argument("--baseline-tol", type=float, default=1e-4)
    ap.add_argument("--force", action="store_true", help="重算已存在的点")
    args = ap.parse_args()

    thetas = [float(t) for t in args.theta.split(",") if t.strip()]
    t_dense_list = [int(t) for t in args.t_dense.split(",") if t.strip()]
    runs_root = Path(args.runs_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev}  run={args.run}  θ={thetas}  T_dense={t_dense_list}", flush=True)
    print(f"共 {len(thetas) * len(t_dense_list)} 个配置点", flush=True)

    cache = Path(args.cache_dir)
    test = PTBXLCache(cache, fold_subset=[10])
    test_loader = DataLoader(test, batch_size=args.batch_size,
                             num_workers=4, pin_memory=True)
    # 标定用 fold-9 验证集，batch_size 固定 32（对齐 07，见文件头注释）
    val = PTBXLCache(cache, fold_subset=[9])
    calib_loader = DataLoader(val, batch_size=CALIB_BATCH_SIZE,
                              num_workers=2, pin_memory=True)
    print(f"测试集 {len(test)} 条(fold10) / 标定集 {len(val)} 条(fold9)", flush=True)

    # 权重只读一次并**冻结**（md5 记进产物）。训练会持续刷新 best.pt，
    # 不冻结则 30 个点会跨越权重刷新而彼此不可比（实测踩过：同一脚本两次
    # 运行因 best.pt 从 ep0 刷到 ep11，AUROC 得 0.4985 / 0.6705）。
    ckpt_path = Path(args.ckpt) if args.ckpt else (runs_root / args.run / "best.pt")
    ckpt, labels, state = load_ckpt(ckpt_path)
    ckpt_epoch = ckpt.get("epoch", -1)
    ckpt_md5 = hashlib.md5(ckpt_path.read_bytes()).hexdigest()
    print(f"检查点 {ckpt_path}\n  epoch={ckpt_epoch}  md5={ckpt_md5}", flush=True)

    # 07 的 SNN 行，用于基线自洽检查
    baseline = None
    bp = Path(args.baseline)
    if bp.exists():
        try:
            for r in json.loads(bp.read_text()).get("rows", []):
                if r.get("model") == "snn":
                    baseline = r
                    break
        except Exception as e:            # comparison.json 可能含裸 NaN（07 未清洗）
            print(f"⚠ 读 {bp} 失败（{e}）；跳过基线自洽检查", flush=True)
    if baseline:
        print(f"基线来源 {bp.name}：macro_auroc={baseline.get('macro_auroc'):.6f} "
              f"sops={baseline.get('sops')}", flush=True)
    else:
        print("⚠ 没有可用的基线 comparison.json，跳过自洽检查", flush=True)

    points, skipped = [], []
    total = len(thetas) * len(t_dense_list)
    k = 0
    for theta in thetas:
        for td in t_dense_list:
            k += 1
            tag = f"theta{theta}_T{td}"
            dst = out_dir / f"{tag}.json"
            if dst.exists() and not args.force:
                try:
                    points.append(json.loads(dst.read_text()))
                    skipped.append(tag)
                    print(f"[{k}/{total}] {tag} 已存在，跳过", flush=True)
                    continue
                except Exception:
                    pass  # 文件残缺 → 重算

            print(f"\n[{k}/{total}] θ={theta}  T_dense={td}", flush=True)
            model = build_model(labels, state, td)   # 同一份冻结权重
            model.to(dev).eval()
            bind_theta(model, theta)                 # θ 生效于 model(x)

            y_true, y_score, infer_s = infer_scores(model, test_loader, dev)
            metrics = multilabel_metrics(y_true, y_score, labels)

            # 发放率必须逐点重标（θ 直接决定编码脉冲数）
            spike_rates = calibrate_spike_rates(
                model, calib_loader, dev, n_batches=args.n_calib_batches)
            cost = cost_snn(model, spike_rates, T_in=1000)

            lat = time_inference(model, dev, T_in=1000,
                                 n_warmup=5, n_iter=args.n_latency_iter)
            cost.latency_ms = lat
            cost.latency_device = dev.type

            is_baseline = (abs(theta - TRAIN_THETA) < 1e-12) and td == 8
            pt = {
                "theta": theta,
                "t_dense": td,
                "is_baseline": is_baseline,
                "run_name": args.run,
                "checkpoint_epoch": ckpt_epoch,
                "checkpoint_md5": ckpt_md5,
                "checkpoint_path": str(ckpt_path),
                "macro_auroc": metrics["macro_auroc"],
                "macro_auprc": metrics["macro_auprc"],
                "per_label": metrics["per_label"],
                "params": cost.total_params,
                "total_macs": cost.total_macs,
                "total_sops": cost.total_sops,
                "spike_rate_mean": cost.spike_rate_mean,
                "spike_rates": spike_rates,
                "input_spike_rate": spike_rates.get("input"),
                "latency_ms": lat,
                "latency_device": dev.type,
                "latency_n_iter": args.n_latency_iter,
                "energy_pj_dense": cost.energy_pj_dense,
                "energy_pj_neuromorphic": cost.energy_pj_neuromorphic,
                "energy_mj_dense": cost.energy_pj_dense / 1e9,
                "energy_mj_neuromorphic": cost.energy_pj_neuromorphic / 1e9,
                # compute.py 的 ModelCost.layers 已经是 list[dict]（内部调过 asdict）；
                # 但为兼容上游可能的改动，两种形态都接受。
                "layers": [l if isinstance(l, dict) else asdict(l) for l in cost.layers],
                "test_inference_seconds": infer_s,
                "labels": labels,
                "n_test": len(test),
                "zero_shot": True,
            }
            # 先打印再落盘：万一下游序列化再出问题，数值也不会白算
            print(f"    → macro_auroc={metrics['macro_auroc']:.4f}  "
                  f"input_rate={spike_rates.get('input'):.4f}  "
                  f"sops={cost.total_sops:,}  "
                  f"E_nm={cost.energy_pj_neuromorphic / 1e9:.6f} mJ  "
                  f"lat={lat:.1f}ms  ({infer_s:.1f}s 推理)", flush=True)
            dst.write_text(json.dumps(clean(pt), indent=2, allow_nan=False))
            points.append(pt)

            del model
            if dev.type == "cuda":
                torch.cuda.empty_cache()

    # ── 基线自洽检查（最强的正确性验证）────────────────────────────────
    consistency = None
    base_pt = next((p for p in points if p.get("is_baseline")), None)
    if baseline and base_pt:
        checks = {}
        for key, mine, theirs in (
            ("macro_auroc", base_pt["macro_auroc"], baseline.get("macro_auroc")),
            ("total_sops", base_pt["total_sops"], baseline.get("sops")),
            ("spike_rate_mean", base_pt["spike_rate_mean"], baseline.get("spike_rate_mean")),
        ):
            if theirs is None:
                continue
            d = abs(float(mine) - float(theirs))
            # 这三项都含**隐藏层**实测发放率 ⇒ 只在同一份权重下可比。
            checks[key] = {"ours": float(mine), "reference_07": float(theirs),
                           "abs_diff": d, "ok": d <= args.baseline_tol,
                           "weight_invariant": False}
        # 07 的 row 里存了逐层发放率；**input 层只取决于输入信号与 θ，与权重无关**，
        # 所以这是本检查里唯一能跨权重对拍的项，也是唯一有判定力的项——
        # best.pt 被训练刷新之后，只有它还应该 OK。
        ref_in = ((baseline or {}).get("spike_rates") or {}).get("input")
        if ref_in is not None and base_pt.get("input_spike_rate") is not None:
            d = abs(float(base_pt["input_spike_rate"]) - float(ref_in))
            checks["input_spike_rate*"] = {
                "ours": float(base_pt["input_spike_rate"]),
                "reference_07": float(ref_in),
                "abs_diff": d, "ok": d <= args.baseline_tol,
                "weight_invariant": True}
        drift = [k for k, c in checks.items()
                 if not c["ok"] and not c["weight_invariant"]]
        consistency = {
            "checks": checks,
            "weight_drift_suspected": bool(drift),
            "drifted_metrics": drift,
            "authoritative_check": "input_spike_rate*（与权重无关，唯一可跨权重对拍）",
            "note": (
                "带 * 的 input_spike_rate 与权重无关，是本检查唯一有判定力的项。"
                "其余三项都随隐藏层发放率变化，只在 comparison.json 生成时刻的权重下可比；"
                "若 best.pt 在其后被训练刷新，它们必然不同——这属于权重漂移，不是实现错误，"
                "且 07 不记录 checkpoint_md5，脚本无法自动识别，只能提示。"
                "实现等价性已单独验证：同一进程内用同一份权重，11 的推理路径"
                "（bind_theta + nw=4 + pin_memory）与 07 的复刻路径（原生 model(x) + nw=2）"
                "在 batch_size=64 下 AUROC 与 yp_sum 逐位相同（0.670474 / 3949.4321）。"
                "实测 input_spike_rate 在 ep0 与 ep11 两版权重下都是 0.3150，可作旁证。"
            ),
        }
        print("\n=== 与 07 的基线自洽检查 (θ=0.15, T_dense=8) ===", flush=True)
        for key, c in checks.items():
            flag = "OK " if c["ok"] else "FAIL"
            print(f"  [{flag}] {key:<20} ours={c['ours']:.6f}  "
                  f"07={c['reference_07']:.6f}  diff={c['abs_diff']:.2e}", flush=True)
        if drift:
            print(f"  ⇒ 预期结果，非实现错误：带 * 的项与权重无关，已 OK；"
                  f"FAIL 的 {'/'.join(drift)} 都含隐藏层发放率，而 07 的 comparison.json "
                  f"生成于 best.pt 被训练刷新之前（07 不记录 checkpoint_md5，无法自动识别）。"
                  f"SNN 跑满 100 轮后重跑 07 + 11，这三项才会同时 OK。", flush=True)

    # ── 单调性检查（θ↑ ⇒ 输入发放率↓ ⇒ SOP↓）─────────────────────────
    mono = None
    if len(t_dense_list) >= 1:
        td_fixed = 8 if 8 in t_dense_list else t_dense_list[0]
        series = sorted((p for p in points if p["t_dense"] == td_fixed),
                        key=lambda p: p["theta"])
        if len(series) >= 2:
            rates = [p["input_spike_rate"] for p in series]
            sops = [p["total_sops"] for p in series]
            mono = {
                "t_dense_fixed": td_fixed,
                "theta": [p["theta"] for p in series],
                "input_spike_rate": rates,
                "total_sops": sops,
                "rate_monotone_decreasing": all(
                    rates[i] >= rates[i + 1] - 1e-12 for i in range(len(rates) - 1)),
                "sops_monotone_decreasing": all(
                    sops[i] >= sops[i + 1] for i in range(len(sops) - 1)),
            }
            print(f"\n单调性 (T_dense={td_fixed}): 发放率单调降="
                  f"{mono['rate_monotone_decreasing']}  SOP 单调降="
                  f"{mono['sops_monotone_decreasing']}", flush=True)

    summary = {
        "generated_at": time.time(),
        "run_name": args.run,
        "checkpoint_path": str(ckpt_path),
        "checkpoint_epoch": ckpt_epoch,
        "checkpoint_md5": ckpt_md5,
        "device": dev.type,
        "n_test": len(test),
        "theta_values": thetas,
        "t_dense_values": t_dense_list,
        "train_theta": TRAIN_THETA,
        "calib_batch_size": CALIB_BATCH_SIZE,
        "n_calib_batches": args.n_calib_batches,
        "energy_constants": {
            "ENERGY_PJ_PER_MAC_FP32": ENERGY_PJ_PER_MAC_FP32,
            "ENERGY_PJ_PER_SOP": ENERGY_PJ_PER_SOP,
        },
        "zero_shot_caveat": (
            "所有 checkpoint 均以 θ=0.15 训练（train.py 静默忽略 theta）。"
            "本扫描是零训练迁移的推理期编码器/积分步数灵敏度分析，"
            "不是「每个 θ 下的最优模型」；精度下降含编码失配与信息损失两种效应。"
        ),
        "baseline_consistency": consistency,
        "monotonicity": mono,
        "skipped_existing": skipped,
        "points": points,
    }
    summ_path = out_dir / "pareto_summary.json"
    summ_path.write_text(json.dumps(clean(summary), indent=2, allow_nan=False))
    print(f"\n→ 已写出 {len(points)} 个点 + {summ_path}", flush=True)


if __name__ == "__main__":
    main()
