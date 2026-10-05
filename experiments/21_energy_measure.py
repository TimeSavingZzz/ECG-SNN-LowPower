#!/usr/bin/env python3
"""真实 GPU 能耗实测：pynvml 采样功耗 x 实测时长 -> 焦耳/样本。

三种口径并置：
  1) 理论稠密能耗   MAC x 3.7 pJ（45nm CMOS 数字 ASIC 常量）
  2) 理论神经形态   SOP x 0.1 pJ（Loihi 类事件驱动常量，仅 SNN）
  3) 实测 GPU 能耗  RTX3090 上 nvidia-smi(pynvml) 采样，动态功率 x 时间 / 样本数

结论预期：230x 只能在神经形态硅上兑现；在 GPU 上 SNN 因 Python 逐时间步模拟
反而最耗能，把"SNN 慢是假象"从延迟维度补到能耗维度（诚实声明，R7）。

用法::
    python3 experiments/21_energy_measure.py --gpu 0 --duration 15 --out results/energy_measure.json
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import sys

NC = Path("/mnt/ECG-SNN-LowPower/third_party/neurocardio")
sys.path.insert(0, str(NC))

from neurocardio.compute import (  # noqa: E402
    ENERGY_PJ_PER_MAC_FP32,
    ENERGY_PJ_PER_SOP,
    calibrate_spike_rates,
    cost_cnn1d,
    cost_snn,
    time_inference,
)
from neurocardio.dataset import PTBXLCache  # noqa: E402
from neurocardio.factory import make_model  # noqa: E402

try:
    import pynvml
    pynvml.nvmlInit()
    HAS_NVML = True
except Exception as exc:  # noqa: BLE001
    pynvml = None
    HAS_NVML = False
    print("[!] pynvml 不可用: " + repr(exc), flush=True)


class PowerSampler:
    """后台线程按固定间隔采样 GPU 功耗，返回 (t, watts) 序列。"""

    def __init__(self, handle, interval: float = 0.005):
        self.handle = handle
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._t0 = 0.0

    def _run(self) -> None:
        self._t0 = time.time()
        while not self._stop.is_set():
            try:
                watts = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
                self.samples.append((time.time() - self._t0, watts))
            except Exception:  # noqa: BLE001
                pass
            time.sleep(self.interval)

    def start(self) -> None:
        self.samples = []
        self._t0 = time.time()
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self._stop.set()

    def stats(self):
        if not self.samples:
            return (0.0, 0.0, 0.0, 0.0)
        watts = np.array([s[1] for s in self.samples], dtype=np.float64)
        return (float(watts.mean()), float(watts.std()),
                float(watts.min()), float(watts.max()))


def measure_idle(handle, seconds: float = 3.0):
    ps = PowerSampler(handle)
    ps.start()
    time.sleep(seconds)
    ps.stop()
    return ps.stats()


def measure_sustained(model, loader, dev, handle, duration: float, n_warmup: int = 3):
    """持续推理 duration 秒，采样功耗并统计样本数。"""
    model.eval()
    with torch.no_grad():
        for i, (x, _, _) in enumerate(loader):
            if i >= n_warmup:
                break
            model(x.to(dev))
    torch.cuda.synchronize()

    ps = PowerSampler(handle)
    n_samples = 0
    ps.start()
    t0 = time.time()
    with torch.no_grad():
        while time.time() - t0 < duration:
            for x, _, _ in loader:
                model(x.to(dev))
                n_samples += int(x.size(0))
                if time.time() - t0 >= duration:
                    break
    torch.cuda.synchronize()
    wall = time.time() - t0
    ps.stop()
    mean_w, std_w, min_w, max_w = ps.stats()
    return {
        "wall_s": wall,
        "n_samples": n_samples,
        "power_mean_w": mean_w,
        "power_std_w": std_w,
        "power_min_w": min_w,
        "power_max_w": max_w,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/mnt/ECG-SNN-LowPower/results/ptbxl_cache")
    ap.add_argument("--runs-root", default="/mnt/ECG-SNN-LowPower/results")
    ap.add_argument("--runs", default="repro_snn,repro_cnn,repro_resnet")
    ap.add_argument("--out", default="/mnt/ECG-SNN-LowPower/results/energy_measure.json")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--duration", type=float, default=15.0)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    if not HAS_NVML:
        raise SystemExit("pynvml 不可用，请先 pip install nvidia-ml-py")

    torch.cuda.set_device(args.gpu)
    dev = torch.device(f"cuda:{args.gpu}")
    handle = pynvml.nvmlDeviceGetHandleByIndex(args.gpu)
    gpu_name = pynvml.nvmlDeviceGetName(handle)
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode(errors="replace")

    names = [n.strip() for n in args.runs.split(",") if n.strip()]

    test = PTBXLCache(Path(args.cache_dir), fold_subset=[10])
    loader = DataLoader(test, batch_size=args.batch_size, num_workers=4, pin_memory=True)
    loader_b1 = DataLoader(test, batch_size=1, num_workers=4, pin_memory=True)
    val = PTBXLCache(Path(args.cache_dir), fold_subset=[9])
    val_loader = DataLoader(val, batch_size=32, num_workers=2, pin_memory=True)
    print(f"GPU {args.gpu}: {gpu_name} | 测试集 {len(test)} 条 | 时长 {args.duration}s", flush=True)

    print("\n=== 测 idle 基线 ===", flush=True)
    idle = measure_idle(handle, seconds=3.0)
    p_idle = idle[0]
    print(f"  idle power: mean={p_idle:.2f}W std={idle[1]:.2f}W", flush=True)

    rows = []
    for run_name in names:
        ckpt_path = Path(args.runs_root) / run_name / "best.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("cfg", {}) or {}
        labels = ckpt.get("label_columns") or ["NORM", "MI", "STTC", "CD", "HYP"]
        model_name = cfg.get("model_name") or "snn"
        print(f"\n=== {run_name} ({model_name}) ===", flush=True)

        model = make_model(model_name, num_classes=len(labels))
        state = ckpt.get("ema") or ckpt.get("state_dict")
        model.load_state_dict(state)
        model.to(dev).eval()

        spike_rates = {}
        if model_name == "snn":
            spike_rates = calibrate_spike_rates(model, val_loader, dev, n_batches=8)
            cost = cost_snn(model, spike_rates, T_in=1000)
        else:
            cost = cost_cnn1d(model, T_in=1000)

        e_dense_mj = cost.energy_pj_dense / 1e9
        e_neuro_mj = cost.energy_pj_neuromorphic / 1e9
        lat_ms = time_inference(model, dev, T_in=1000, n_warmup=5, n_iter=20)

        print(f"  持续推理 (B={args.batch_size})...", flush=True)
        m = measure_sustained(model, loader, dev, handle, args.duration)
        dyn_p = m["power_mean_w"] - p_idle
        energy_per_sample_j = dyn_p * m["wall_s"] / m["n_samples"]
        throughput = m["n_samples"] / m["wall_s"]

        print(f"  持续推理 (B=1)...", flush=True)
        m1 = measure_sustained(model, loader_b1, dev, handle, args.duration)
        dyn_p1 = m1["power_mean_w"] - p_idle
        steady_lat_s = m1["wall_s"] / m1["n_samples"]
        energy_b1_j = dyn_p1 * steady_lat_s

        row = {
            "run_name": run_name,
            "model": model_name,
            "params": cost.total_params,
            "macs": cost.total_macs,
            "sops": cost.total_sops,
            "theory_dense_mj": e_dense_mj,
            "theory_neuromorphic_mj": e_neuro_mj,
            "latency_ms_b1": lat_ms,
            "measured": {
                "batched": {
                    "batch_size": args.batch_size,
                    "power_mean_w": m["power_mean_w"],
                    "power_std_w": m["power_std_w"],
                    "power_min_w": m["power_min_w"],
                    "power_max_w": m["power_max_w"],
                    "dynamic_power_w": dyn_p,
                    "wall_s": m["wall_s"],
                    "n_samples": m["n_samples"],
                    "throughput_sps": throughput,
                    "energy_per_sample_j": energy_per_sample_j,
                },
                "b1": {
                    "power_mean_w": m1["power_mean_w"],
                    "dynamic_power_w": dyn_p1,
                    "steady_state_latency_s": steady_lat_s,
                    "latency_ms": lat_ms,
                    "energy_per_sample_j": energy_b1_j,
                },
            },
        }
        rows.append(row)

        print(f"  理论: 稠密={e_dense_mj:.4f} mJ  神经形态={e_neuro_mj:.6f} mJ  B1延迟={lat_ms:.2f}ms", flush=True)
        print(f"  实测B={args.batch_size}: 动态功率={dyn_p:.2f}W 吞吐={throughput:.1f}样本/s "
              f"每样本={energy_per_sample_j * 1e3:.4f} mJ", flush=True)
        print(f"  实测B=1: 动态功率={dyn_p1:.2f}W 稳态延迟={steady_lat_s * 1e3:.2f}ms "
              f"每样本={energy_b1_j * 1e3:.4f} mJ", flush=True)

    out = {
        "generated_at": time.time(),
        "gpu": {"index": args.gpu, "name": gpu_name},
        "idle_power_w": p_idle,
        "duration_s": args.duration,
        "constants": {
            "ENERGY_PJ_PER_MAC_FP32": ENERGY_PJ_PER_MAC_FP32,
            "ENERGY_PJ_PER_SOP": ENERGY_PJ_PER_SOP,
        },
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n-> 已写出 {args.out}", flush=True)

    print("\n=== 汇总（实测 vs 理论） ===", flush=True)
    print(f"{'run':<14}{'理论稠密mJ':>12}{'理论神经mJ':>12}{'实测B64mJ':>14}{'实测B1mJ':>14}", flush=True)
    for r in rows:
        b64 = r["measured"]["batched"]["energy_per_sample_j"] * 1e3
        b1 = r["measured"]["b1"]["energy_per_sample_j"] * 1e3
        print(f"{r['run_name']:<14}{r['theory_dense_mj']:>12.4f}{r['theory_neuromorphic_mj']:>12.6f}"
              f"{b64:>14.4f}{b1:>14.4f}", flush=True)

    pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
