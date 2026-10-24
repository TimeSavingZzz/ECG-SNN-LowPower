#!/usr/bin/env python3
"""P0-3: execution-path control for the GPU energy measurement (CUDA Graph).

Question this answers
---------------------
The 89x GPU-energy penalty of the SNN is dominated by *dispatch* overhead, not
by arithmetic: at B=1 the steady-state latency was ~1.8 s per ECG window, i.e.
~1.8 ms per timestep, for kernels that do microseconds of work.  This script
re-runs the same forward pass under CUDA Graph capture (one launch per
inference instead of thousands) and measures the same quantity again.

What it does NOT answer
-----------------------
Nothing here changes the algorithmic billing (230x).  A GPU is not an
event-driven chip; the result is an IMPLEMENTATION-LAYER observation and must
be reported as such.

Protocol
--------
Both arms (eager / graph) replay the SAME static input buffer for the whole
measurement window, so data loading and H2D transfer are excluded from both
and the only difference is the launch mechanism.  Power is sampled with
pynvml every 5 ms; dynamic power = mean - idle; energy/sample = dyn*wall/n.

Writes results/energy_cudagraph.json (a NEW file).
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

NC = Path("/mnt/ECG-SNN-LowPower/third_party/neurocardio")
sys.path.insert(0, str(NC))

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
    def __init__(self, handle, interval: float = 0.005):
        self.handle = handle
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._t0 = 0.0
        self._thread = None

    def _run(self) -> None:
        self._t0 = time.time()
        while not self._stop.is_set():
            try:
                w = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
                self.samples.append((time.time() - self._t0, w))
            except Exception:  # noqa: BLE001
                pass
            time.sleep(self.interval)

    def start(self) -> None:
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def stats(self):
        if not self.samples:
            return (0.0, 0.0, 0.0, 0.0)
        w = np.array([s[1] for s in self.samples], dtype=np.float64)
        return float(w.mean()), float(w.std()), float(w.min()), float(w.max())


def measure_idle(handle, seconds: float = 3.0) -> float:
    ps = PowerSampler(handle)
    ps.start()
    time.sleep(seconds)
    ps.stop()
    return ps.stats()[0]


def capture_graph(model, dev, B: int, T: int = 1000, C: int = 12, warmup: int = 3):
    """Capture one forward pass into a CUDA Graph.  Returns (graph, static_in)."""
    static_in = torch.zeros(B, C, T, device=dev)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        with torch.no_grad():
            for _ in range(warmup):
                model(static_in)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        _ = model(static_in)
    torch.cuda.synchronize()
    return g, static_in


def run_arm(fn, B: int, duration: float, handle, p_idle: float, warmup: int = 3) -> dict:
    """Time + energy for `fn()` called repeatedly for `duration` seconds."""
    with torch.no_grad():
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()

    ps = PowerSampler(handle)
    n_iter = 0
    ps.start()
    t0 = time.time()
    with torch.no_grad():
        while time.time() - t0 < duration:
            fn()
            n_iter += 1
    torch.cuda.synchronize()
    wall = time.time() - t0
    ps.stop()

    mean_w, std_w, min_w, max_w = ps.stats()
    dyn = mean_w - p_idle
    n_samples = n_iter * B
    return {
        "wall_s": wall,
        "n_iter": n_iter,
        "n_samples": n_samples,
        "power_mean_w": mean_w,
        "power_std_w": std_w,
        "power_min_w": min_w,
        "power_max_w": max_w,
        "dynamic_power_w": dyn,
        "latency_ms_per_iter": 1000.0 * wall / max(n_iter, 1),
        "throughput_sps": n_samples / wall,
        "energy_per_sample_j": dyn * wall / max(n_samples, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="/mnt/ECG-SNN-LowPower/results")
    ap.add_argument("--runs", default="repro_snn,repro_cnn,repro_resnet")
    ap.add_argument("--out", default="/mnt/ECG-SNN-LowPower/results/energy_cudagraph.json")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--duration", type=float, default=15.0, help="B=64 臂的时长")
    ap.add_argument("--duration-b1", type=float, default=30.0, help="B=1 臂的时长")
    args = ap.parse_args()

    if not HAS_NVML:
        raise SystemExit("pynvml 不可用")

    torch.cuda.set_device(args.gpu)
    dev = torch.device(f"cuda:{args.gpu}")
    handle = pynvml.nvmlDeviceGetHandleByIndex(args.gpu)
    name = pynvml.nvmlDeviceGetName(handle)
    if isinstance(name, bytes):
        name = name.decode(errors="replace")

    p_idle = measure_idle(handle, 3.0)
    print(f"GPU {args.gpu}: {name} | idle={p_idle:.2f}W", flush=True)

    rows = []
    for run_name in [n.strip() for n in args.runs.split(",") if n.strip()]:
        ckpt_path = Path(args.runs_root) / run_name / "best.pt"
        if not ckpt_path.exists():
            print(f"[SKIP] {run_name}: 无 best.pt", flush=True)
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("cfg", {}) or {}
        labels = ckpt.get("label_columns") or ["NORM", "MI", "STTC", "CD", "HYP"]
        model_name = cfg.get("model_name") or run_name
        model = make_model(model_name, num_classes=len(labels))
        model.load_state_dict(ckpt.get("ema") or ckpt.get("state_dict"))
        model.to(dev).eval()
        print(f"\n=== {run_name} ({model_name}) ===", flush=True)

        for B, dur in ((64, args.duration), (1, args.duration_b1)):
            static_in = torch.randn(B, 12, 1000, device=dev)
            entry = {"run_name": run_name, "model": model_name, "batch_size": B,
                     "arms": {}, "errors": {}}

            # ---- eager ----
            e = run_arm(lambda: model(static_in), B, dur, handle, p_idle)
            entry["arms"]["eager"] = e
            print(f"  B={B:<3} eager: {e['latency_ms_per_iter']:9.2f} ms/iter  "
                  f"{e['energy_per_sample_j'] * 1e3:9.4f} mJ/样本  "
                  f"(n={e['n_samples']})", flush=True)

            # ---- cuda graph ----
            torch.cuda.empty_cache()
            try:
                t_cap = time.time()
                g, g_in = capture_graph(model, dev, B)
                cap_s = time.time() - t_cap
                r = run_arm(g.replay, B, dur, handle, p_idle)
                r["capture_s"] = cap_s
                entry["arms"]["cudagraph"] = r
                print(f"  B={B:<3} graph: {r['latency_ms_per_iter']:9.2f} ms/iter  "
                      f"{r['energy_per_sample_j'] * 1e3:9.4f} mJ/样本  "
                      f"(n={r['n_samples']}, capture={cap_s:.1f}s)", flush=True)
                entry["speedup"] = (e["latency_ms_per_iter"]
                                    / max(r["latency_ms_per_iter"], 1e-9))
                entry["energy_ratio_eager_over_graph"] = (
                    e["energy_per_sample_j"] / max(r["energy_per_sample_j"], 1e-30))
                print(f"         -> 加速 {entry['speedup']:.1f}x, "
                      f"能耗降低 {entry['energy_ratio_eager_over_graph']:.1f}x", flush=True)
            except Exception as exc:  # noqa: BLE001
                msg = f"{type(exc).__name__}: {exc}"
                entry["errors"]["cudagraph"] = msg[:400]
                print(f"  B={B:<3} graph: 捕获失败 -> {msg[:200]}", flush=True)

            del static_in
            torch.cuda.empty_cache()
            rows.append(entry)

    out = {
        "generated_at": time.time(),
        "gpu": {"index": args.gpu, "name": name},
        "idle_power_w": p_idle,
        "protocol": "static input, data loading excluded from BOTH arms; "
                    "pynvml 5ms sampling; dynamic = mean - idle",
        "caveat": "GPU implementation-layer observation only; does NOT change "
                  "the algorithmic 230x billing and must not be extrapolated to "
                  "event-driven hardware",
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n-> 已写出 {args.out}", flush=True)

    print("\n=== 汇总（每样本 mJ） ===", flush=True)
    print(f"{'run':<14}{'B':>4}{'eager':>12}{'graph':>12}{'加速':>10}", flush=True)
    for r in rows:
        e = r["arms"]["eager"]["energy_per_sample_j"] * 1e3
        gg = r["arms"].get("cudagraph")
        gs = (gg["energy_per_sample_j"] * 1e3) if gg else float("nan")
        sp = r.get("speedup", float("nan"))
        print(f"{r['run_name']:<14}{r['batch_size']:>4}{e:>12.4f}{gs:>12.4f}{sp:>9.1f}x", flush=True)

    pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
