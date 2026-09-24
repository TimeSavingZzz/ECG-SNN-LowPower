#!/usr/bin/env python3
"""现场可交互 demo 的后端替身 —— 上游 ``modal_app/app.py::web()`` 的 stdlib 本地化。

**为什么需要它**：上游根前端 ``web/index.html`` 在 :10,:570 用**绝对路径**
``/static/style.css``、``/static/app.js``，靠 Modal 的
``api.mount("/static", StaticFiles(directory="/web"))``（``app.py:557``）把 ``/static/X``
**平铺**映射到 ``web/X``；而且前端要调 7 个接口。所以上游自带的
``scripts/serve.sh``（``cd web && python3 -m http.server``）**实际是失效脚本**（资源全 404），
纯静态托管根前端不可行。

本脚本用 stdlib ``http.server``（``BaseHTTPRequestHandler`` 原生支持 POST，不必装 FastAPI）
复刻 ``web()`` 的全部路由。**逐路由对照**（上游行号来自 ``modal_app/app.py``）：

===========  ==========================================  ==========================
路由          上游实现                                     本脚本
===========  ==========================================  ==========================
``GET /``    :547 ``index()`` 读 ``/web/index.html`` 加版本串  ``_index()``
``/static/`` :557 ``api.mount`` 平铺到 ``/web``              ``_static()`` 复刻平铺
``GET /warm`` :559 预载模型                                  ``_warm()``
``/history``  :565 读 ``runs/<run>/history.json``            ``_history()``
``/comparison`` :576 读 ``comparison.json``                   ``_comparison()``
``GET /status`` :587 ckpt epoch + cache_ready                ``_status()``
``/examples`` :610 逐标签取样填下拉框                        ``_examples()``
``/infer_example`` :638 单条 PTB-XL 推理 + 三模型对比        ``_infer_example()``
``POST /infer_upload`` :666 上传 .npy(12,T)                  ``_infer_upload()``
===========  ==========================================  ==========================

**与上游的三处有意差异**（都是本地化必需，非改动语义）：

1. **目录映射**：上游在 Modal Volume 上（``VOL_MOUNT/runs/<run>/best.pt``、
   ``VOL_MOUNT/comparison.json``、``VOL_MOUNT/ptbxl_cache``）；本地对应
   ``results/<run>/best.pt``、``results/comparison.json``、``results/ptbxl_cache``。
2. **SNN 检查点解析**：上游 ``_resolve_ckpt()`` :509 取 ``runs/*/best.pt`` 中 **mtime 最新**的
   一个。我们 ``results/`` 下同时有 ``repro_snn``/``repro_cnn``/``repro_resnet``，
   照抄会**取到 CNN/ResNet 的权重当 SNN 用**。故本脚本按 ``--snn-run`` 显式指定
   （默认 ``repro_snn``），只在该 run 内做 mtime 热重载。
3. **``/static`` 走查 + 推理串行化**：见下。

**两个必须自己处理的坑**（都有实测依据）：

* **推理必须串行化**。上游 ``neurocardio/model.py`` 的 ``snn.Leaky.forward`` 把传入的
  ``mem`` 存进 ``self.mem``，且**只在 shape 不匹配时**才清零；``init_leaky()`` 返回
  ``self.mem.clone()`` ⇒ 每个 batch 会继承上一个 batch 的终态膜电位。实测同一
  ``best.pt``、同一 θ/T_dense 下 ``batch_size=64`` → macro-AUROC **0.498530** 而
  ``batch_size=96`` → **0.487912**（确定性，重跑逐位相同）。多线程并发请求会互相污染这份
  状态，故所有推理走同一把 ``threading.Lock``（上游 FastAPI 单 worker 天然串行）。
* **NaN 必须洗掉**。``train.py:143`` 对零支持类别写 ``float("nan")``，而
  ``json.dump`` 默认输出**裸 ``NaN``（非法 JSON）**；前端 ``app.js:31-35`` 的
  ``jget()`` 一次 ``r.json()`` 抛错就**整个面板静默空白**。故所有响应统一
  ``allow_nan=False`` + 递归 ``clean()``。

**静态文件**：``/static/X`` → ``<web>/X``（平铺），并对 ``index.html`` 里的
``/static/style.css``、``/static/app.js`` 注入 ``?v=<内容哈希>``，复刻上游 :534
``_asset_version()`` 对抗浏览器启发式缓存的意图。

用法::

    CUDA_VISIBLE_DEVICES=3 nohup python3 experiments/10_serve_demo.py \\
        --port 8000 > logs/web.log 2>&1 &

容器内自测（容器**没有 curl**，用 urllib）::

    python3 -c "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:8000/health').read().decode())"
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/mnt/ECG-SNN-LowPower")
NC = ROOT / "third_party" / "neurocardio"
sys.path.insert(0, str(NC))

from neurocardio.infer import CardioInferencer  # noqa: E402
from neurocardio.factory import make_model  # noqa: E402

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
WIN_LEN = 1000
# 上游 :460 BASELINE_RUNS 用的是 Modal Volume 上的名字，这里映射到本工程的 run 名。
BASELINE_RUNS = {"cnn": "repro_cnn", "resnet": "repro_resnet"}


# ---------------------------------------------------------------------------
# JSON 卫生：裸 NaN 会让前端整页静默空白
# ---------------------------------------------------------------------------
def clean(o):
    """递归把 NaN/±Inf 换成 None，并把 numpy 标量降级成 Python 原生类型。

    ``json.dump`` 的 ``allow_nan=False`` 只负责**拒绝**，不会自己转换；
    而 ``allow_nan=True``（默认）会写出裸 ``NaN`` —— 非法 JSON。
    两边都不可接受，故先洗再序列化。与 ``11_pareto_sweep.py::clean`` 同一约定。
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


def dumps(o) -> bytes:
    """洗过 NaN 再序列化，并保证 ``allow_nan=False`` 兜底。"""
    return json.dumps(clean(o), allow_nan=False).encode("utf-8")


# ---------------------------------------------------------------------------
# 后端状态
# ---------------------------------------------------------------------------
class Demo:
    """常驻状态：缓存（mmap）、SNN 句柄、基线句柄、推理锁。"""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.results = Path(args.results)
        self.cache_dir = Path(args.cache_dir)
        self.web = Path(args.web)
        self.snn_run = args.snn_run

        # 推理串行化：上游 LIF 有跨 batch 的膜电位残留（见模块 docstring）。
        self.lock = threading.Lock()

        self._infer = None            # CardioInferencer
        self._infer_mtime = 0.0
        self._infer_path: str | None = None
        self._baselines: dict = {}
        self._ckpt_info = None        # /status 用，按 mtime 缓存

        # 缓存常驻：mmap 单条信号是廉价的，但每次都 np.load 不必要。
        self.signals = None
        self.meta = None
        self.label_cols = SUPERCLASSES
        if (self.cache_dir / "labels.csv").exists():
            import pandas as pd
            self.meta = pd.read_csv(self.cache_dir / "labels.csv")
            self.signals = np.load(self.cache_dir / f"signals_100.npy", mmap_mode="r")
            with open(self.cache_dir / "label_columns.txt") as f:
                self.label_cols = [ln.strip() for ln in f if ln.strip()]
            print(f"[cache] {len(self.meta)} 条 · 标签 {self.label_cols}", flush=True)
        else:
            print(f"[cache] 未找到 {self.cache_dir/'labels.csv'} —— "
                  f"/examples 与 /infer_example 不可用", flush=True)

    # -------- 检查点解析（上游 _resolve_ckpt/_get_infer :509-533） --------
    def _snn_ckpt(self) -> Path | None:
        """只在 ``--snn-run`` 内找 best.pt，并保留 mtime 热重载。

        不能照抄上游的 ``runs/*/best.pt`` 全局最新 —— 我们会拿到 CNN/ResNet。
        热重载是有意义的：SNN 训练仍在续跑，``best.pt`` 会被刷新，
        演示时应展示最新权重（日志里会打印每次重载）。
        """
        p = self.results / self.snn_run / "best.pt"
        return p if p.exists() else None

    def inferencer(self) -> CardioInferencer | None:
        p = self._snn_ckpt()
        if p is None:
            return None
        m = p.stat().st_mtime
        if self._infer is None or self._infer_mtime < m:
            t0 = time.time()
            self._infer = CardioInferencer(p)
            self._infer_mtime = m
            self._infer_path = str(p)
            print(f"[snn] 载入 {p} (mtime={m:.0f}, {time.time()-t0:.1f}s) "
                  f"θ={self._infer.theta}", flush=True)
        return self._infer

    def baseline(self, key: str):
        """上游 ``_get_baseline`` :461 的本地版：按 key 惰性加载 + mtime 热重载。"""
        run_name = BASELINE_RUNS.get(key)
        if not run_name:
            return None
        ckpt_path = self.results / run_name / "best.pt"
        if not ckpt_path.exists():
            return None
        m = ckpt_path.stat().st_mtime
        cached = self._baselines.get(key)
        if cached is None or cached["mtime"] < m:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            labels = ckpt.get("label_columns") or SUPERCLASSES
            cfg = ckpt.get("cfg", {}) or {}
            name = cfg.get("model_name") or key
            model = make_model(name, num_classes=len(labels))
            state = ckpt.get("ema") or ckpt.get("state_dict")
            model.load_state_dict(state)
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(dev).eval()
            self._baselines[key] = {"model": model, "labels": labels,
                                    "mtime": m, "dev": dev, "name": name}
            print(f"[baseline] 载入 {key} ({name}) ← {ckpt_path}", flush=True)
        return self._baselines[key]

    def baseline_probs(self, sig: np.ndarray) -> dict:
        """上游 ``_baseline_probs`` :489：**同一个 sig** 跑所有基线，保证公平对比。

        单个基线失败绝不拖垮 SNN 主路径（上游注释原话）。
        """
        out: dict = {}
        for key in BASELINE_RUNS:
            try:
                b = self.baseline(key)
                if b is None:
                    continue
                x = torch.from_numpy(np.ascontiguousarray(sig)).float()
                x = x.unsqueeze(0).to(b["dev"])
                with torch.no_grad():
                    logits = b["model"](x)
                probs = torch.sigmoid(logits)[0].cpu().numpy()
                out[key] = {c: float(probs[i]) for i, c in enumerate(b["labels"])}
            except Exception as e:
                print(f"[baseline] {key} 失败：{e}", flush=True)
        return out

    def snn_full(self, sig: np.ndarray) -> dict:
        """SNN 推理 + traces（上游直接调 ``infer.infer(sig)``）。

        ``CardioInferencer.infer`` 返回的键与前端契约**逐字对应**：
        ``probs/label_names/spike_rates{input,block0..5}/rasters{block_last,
        block_last_shape}/input_spikes/pool_gates/pool_gates_T/vout``。
        它内部用 ``NeuroCardio(num_classes=5)``（``T_dense`` 默认 8，与训练一致）
        并优先取 ``ckpt["ema"]``。
        """
        inf = self.inferencer()
        if inf is None:
            raise RuntimeError(f"没有可用检查点（{self.results/self.snn_run/'best.pt'}）")
        return inf.infer(sig)

    # -------- 路由实现 --------
    def _asset_version(self) -> str:
        """上游 :534 ``_asset_version``：CSS+JS 的内容哈希，只为让 URL 变化。"""
        h = hashlib.md5()
        try:
            for f in ("style.css", "app.js"):
                h.update((self.web / f).read_bytes())
            return h.hexdigest()[:8]
        except Exception:
            return "dev"

    def _index(self) -> tuple[int, bytes, str]:
        html = (self.web / "index.html").read_text(encoding="utf-8")
        ver = self._asset_version()
        html = html.replace("/static/style.css", f"/static/style.css?v={ver}")
        html = html.replace("/static/app.js", f"/static/app.js?v={ver}")
        return 200, html.encode("utf-8"), "text/html; charset=utf-8"

    def _static(self, rel: str) -> tuple[int, bytes, str]:
        """复刻 ``api.mount("/static", StaticFiles(directory="/web"))`` 的**平铺**语义。

        前端请求 ``/static/app.js``，而文件在 ``web/app.js``（不是 ``web/static/app.js``）。
        必须剥掉 ``/static/`` 前缀后在 web 根下直查，否则 404 → 页面空白。
        """
        rel = rel.lstrip("/")
        # 只允许单层文件名，杜绝 ../ 穿越
        if not rel or "/" in rel or rel.startswith("."):
            return 404, b"not found", "text/plain"
        p = self.web / rel
        if not p.is_file():
            return 404, b"not found", "text/plain"
        ctype = {
            ".js": "application/javascript; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".html": "text/html; charset=utf-8",
            ".json": "application/json; charset=utf-8",
            ".png": "image/png",
            ".svg": "image/svg+xml",
        }.get(p.suffix, "application/octet-stream")
        return 200, p.read_bytes(), ctype

    def _warm(self) -> dict:
        return {"loaded": self.inferencer() is not None}

    def _history(self, run_name: str) -> tuple[int, bytes] | tuple[int, dict]:
        """上游 :565 —— 固定 ``run_name`` 参数；前端硬编码 ``run_name=main``。

        我们的 run 叫 ``repro_snn``，故把 ``main`` 映射过去（找不到则回退 snn_run）。
        """
        name = run_name or self.snn_run
        if name in ("main", "default"):
            name = self.snn_run
        p = self.results / name / "history.json"
        if not p.exists():
            return 404, {"detail": f"no history.json for run {name}"}
        return 200, json.loads(p.read_text(encoding="utf-8"))

    def _comparison(self) -> tuple[int, bytes] | tuple[int, dict]:
        """上游 :576 原样透传 ``comparison.json``（07 产出，schema 完全吻合）。

        前端 ``renderComparison`` 要求 ``rows`` 里**同时有 snn 与 cnn 两行**，
        否则顶部的节能倍数与 Holter 卡片不填充。
        """
        p = self.results / "comparison.json"
        if not p.exists():
            return 404, {"detail": "no comparison.json yet —— 先跑 07_energy_report.py"}
        return 200, json.loads(p.read_text(encoding="utf-8"))

    def _ckpt_meta(self):
        """``/status`` 需要 epoch/best_macro_auroc；按 mtime 缓存避免每 30s 重读 30MB。"""
        p = self._snn_ckpt()
        if p is None:
            return None, None
        m = p.stat().st_mtime
        if self._ckpt_info is None or self._ckpt_info[0] < m:
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            self._ckpt_info = (m, {
                "epoch": ckpt.get("epoch"),
                "best_macro_auroc": ckpt.get("best_macro_auroc"),
                "has_ema": ckpt.get("ema") is not None,
            })
        return p, self._ckpt_info[1]

    def _status(self) -> dict:
        """上游 :587。``checkpoint: null`` 是**合法**状态，前端 :670 会显示 "no checkpoint"。"""
        p, meta = self._ckpt_meta()
        info = {
            "checkpoint": str(p) if p else None,
            "checkpoint_mtime": p.stat().st_mtime if p else None,
        }
        if meta:
            info.update(meta)
        info["cache_ready"] = (self.cache_dir / "labels.csv").exists()
        return info

    def _examples(self, n: int, fold: int) -> tuple[int, dict]:
        """上游 :610 —— 逐标签取样保证多样，去重后截断到 n。

        ⚠️ 前端 ``LABELS`` 的**唯一**来源就是这里的 ``labels`` 字段，
        且 ``examples`` 不能为空（``app.js:686`` 之后主流程会自动跑第一个样本）。
        """
        if self.meta is None:
            return 404, {"detail": "cache not built"}
        labels = self.label_cols
        test = self.meta[self.meta.strat_fold == fold]
        out = []
        for label in labels:
            sub = test[test[label] == 1].head(max(1, n // len(labels)))
            for _, row in sub.iterrows():
                out.append({
                    "ecg_id": int(row["ecg_id"]),
                    "labels": [c for c in labels if int(row[c]) == 1],
                })
        seen, deduped = set(), []
        for o in out:
            if o["ecg_id"] in seen:
                continue
            seen.add(o["ecg_id"])
            deduped.append(o)
        return 200, {"labels": labels, "examples": deduped[:n]}

    def _infer_example(self, ecg_id: int) -> tuple[int, dict]:
        """上游 :638。

        **缓存里的信号已经是 per-lead 鲁棒 z-score 归一化过的**
        （``data.build_cache`` 里 ``sig[i] = normalize(raw)``），
        而 ``CardioInferencer.infer`` 的 docstring 明说输入要 "already normalized"
        ⇒ 这条路径**不需要**再做归一化（上传路径才需要，见 ``_infer_upload``）。
        """
        if self.signals is None or self.meta is None:
            return 404, {"detail": "cache not built"}
        idx_arr = self.meta.index[self.meta["ecg_id"] == ecg_id].tolist()
        if not idx_arr:
            return 404, {"detail": f"ecg_id {ecg_id} not in cache"}
        idx = idx_arr[0]
        sig = np.asarray(self.signals[idx]).astype(np.float32)   # (12, T)
        true_labels = [c for c in self.label_cols if int(self.meta.iloc[idx][c]) == 1]

        with self.lock:                       # ← 见模块 docstring：LIF 有跨 batch 残留
            result = self.snn_full(sig)
            result["compare_probs"] = {"snn": result["probs"],
                                       **self.baseline_probs(sig)}
        result["ecg_id"] = int(ecg_id)
        result["true_labels"] = true_labels
        result["ecg"] = sig.tolist()          # (12, T)：drawECG 按 ecg[lead] 取
        result["fs"] = 100
        return 200, result

    def _infer_upload(self, body: bytes, boundary: bytes) -> tuple[int, dict]:
        """上游 :666 —— 上传 ``.npy``（形状 (12,T)）。

        ⚠️ 上游在**这里**做归一化（``infer.py`` 内部不做）：先截断/补齐到 1000，
        再 per-lead 鲁棒 z-score ``(x-median)/(1.4826*MAD+1e-6)``。
        顺序不能颠倒 —— 先归一化再 pad 会把 0 当成真实样本值参与统计。
        """
        fields = _parse_multipart(body, boundary)
        raw = fields.get("file")
        if raw is None:
            return 400, {"detail": "缺少 multipart 字段 'file'"}
        try:
            sig = np.load(io.BytesIO(raw))
        except Exception as e:
            return 400, {"detail": f"invalid .npy file: {e}"}
        if sig.ndim != 2 or sig.shape[0] != 12:
            return 400, {"detail": f"expected shape (12, T), got {sig.shape}"}
        sig = sig[:, :WIN_LEN].astype(np.float32)
        if sig.shape[1] < WIN_LEN:
            sig = np.pad(sig, ((0, 0), (0, WIN_LEN - sig.shape[1])))
        med = np.median(sig, axis=1, keepdims=True)
        mad = np.median(np.abs(sig - med), axis=1, keepdims=True) + 1e-6
        sig = (sig - med) / (1.4826 * mad)

        with self.lock:
            result = self.snn_full(sig)
            result["compare_probs"] = {"snn": result["probs"],
                                       **self.baseline_probs(sig)}
        result["ecg"] = sig.tolist()
        result["fs"] = 100
        return 200, result

    def _health(self) -> dict:
        """非契约路由：容器内自检用（容器没有 curl）。"""
        p = self._snn_ckpt()
        return {
            "ok": True,
            "snn_run": self.snn_run,
            "snn_ckpt": str(p) if p else None,
            "snn_ckpt_exists": p is not None,
            "web_dir": str(self.web),
            "web_index_exists": (self.web / "index.html").is_file(),
            "cache_dir": str(self.cache_dir),
            "cache_ready": self.meta is not None,
            "n_cached": 0 if self.meta is None else int(len(self.meta)),
            "labels": self.label_cols,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "torch": torch.__version__,
        }


def _parse_multipart(body: bytes, boundary: bytes) -> dict:
    """极简 multipart/form-data 解析，只取字段名 → 原始字节。

    不用 ``cgi.FieldStorage``：它在 3.11 起 deprecated、3.13 已移除，
    而我们只需要 ``file`` 一个二进制字段，手写更稳、更少惊喜。
    """
    out: dict = {}
    for part in body.split(b"--" + boundary)[1:]:
        if part[:2] == b"--":            # 结尾分隔符
            break
        head, sep, data = part.partition(b"\r\n\r\n")
        if not sep:
            continue
        if data.endswith(b"\r\n"):
            data = data[:-2]
        m = re.search(rb'name="([^"]*)"', head)
        if m:
            out[m.group(1).decode("utf-8", "replace")] = data
    return out


# ---------------------------------------------------------------------------
# HTTP 层
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "ECGSNNDemo/1.0"
    protocol_version = "HTTP/1.1"        # 前端有 3 个轮询（30/60/120 s），长连接省事

    def _send(self, code: int, payload: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass                          # 浏览器切页/轮询中断是常态

    def _json(self, code: int, obj):
        self._send(code, dumps(obj), "application/json; charset=utf-8")

    def do_GET(self):                     # noqa: N802
        demo: Demo = self.server.demo      # type: ignore[attr-defined]
        path, _, qs = self.path.partition("?")
        q = {}
        for kv in qs.split("&"):
            if "=" in kv:
                k, _, v = kv.partition("=")
                q[k] = v
        try:
            if path == "/":
                self._send(*demo._index())
            elif path.startswith("/static/"):
                self._send(*demo._static(path[len("/static/"):]))
            elif path == "/health":
                self._json(200, demo._health())
            elif path == "/warm":
                self._json(200, demo._warm())
            elif path == "/status":
                self._json(200, demo._status())
            elif path == "/examples":
                n = int(q.get("n", 20))
                fold = int(q.get("fold", 10))
                self._json(*demo._examples(n, fold))
            elif path == "/infer_example":
                self._json(*demo._infer_example(int(q["ecg_id"])))
            elif path == "/history":
                self._json(*demo._history(q.get("run_name", "")))
            elif path == "/comparison":
                self._json(*demo._comparison())
            else:
                self._json(404, {"detail": f"no route {path}"})
        except KeyError as e:
            self._json(400, {"detail": f"missing query param {e}"})
        except Exception as e:
            traceback.print_exc()
            self._json(500, {"detail": f"{type(e).__name__}: {e}"})

    def do_POST(self):                    # noqa: N802
        demo: Demo = self.server.demo      # type: ignore[attr-defined]
        path, _, _ = self.path.partition("?")
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(n) if n else b""
            if path == "/infer_upload":
                ctype = self.headers.get("Content-Type") or ""
                m = re.search(r'boundary=([^;]+)', ctype)
                if not m:
                    self._json(400, {"detail": "expected multipart/form-data"})
                    return
                boundary = m.group(1).strip().strip('"').encode("utf-8")
                self._json(*demo._infer_upload(body, boundary))
            else:
                self._json(404, {"detail": f"no route {path}"})
        except Exception as e:
            traceback.print_exc()
            self._json(500, {"detail": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt, *a):
        sys.stderr.write("%s [%s] %s\n" % (
            time.strftime("%H:%M:%S"), self.address_string(), fmt % a))


def main() -> None:
    ap = argparse.ArgumentParser(description="现场 demo 的后端替身（stdlib）")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--cache-dir", default=str(ROOT / "results" / "ptbxl_cache"))
    ap.add_argument("--web", default=str(NC / "web"))
    ap.add_argument("--snn-run", default="repro_snn",
                    help="SNN 检查点所在 run（勿用上游的全局最新逻辑，会拿到 CNN）")
    args = ap.parse_args()

    demo = Demo(args)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True
    httpd.demo = demo                      # type: ignore[attr-defined]
    print(f"[serve] http://{args.host}:{args.port}  web={args.web}", flush=True)
    print(f"[serve] SNN <- {args.results}/{args.snn_run}/best.pt", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("[serve] 退出", flush=True)


if __name__ == "__main__":
    main()
