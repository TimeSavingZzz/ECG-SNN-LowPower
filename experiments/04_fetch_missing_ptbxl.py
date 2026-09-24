#!/usr/bin/env python3
"""补齐 PTB-XL 缺失记录（断点续传，跳过已存在文件）。

背景：`/mnt/ptb-xl` 现有副本来自夸克网盘搬运，只搬了一半——
21799 条记录里缺 10799 条（``records100/00000/00513_lr`` 之后全无）。
本脚本以 ``ptbxl_database.csv`` 为准，逐条核对 ``.hea``/``.dat``，
从 PhysioNet 官方 1.0.3 版补齐，不重下已有文件。

性能：``urllib.request.urlretrieve`` 每下 1 个文件就重开一次 TLS 连接，
在 PhysioNet（单次往返 ~1.4s）上只有约 55 条/分钟。这里改为**每条线程持有
一条 keep-alive 连接**，把延迟摊到同一条连接的多个请求上。

用法::

    python3 experiments/04_fetch_missing_ptbxl.py              # 补齐 100Hz + 500Hz
    python3 experiments/04_fetch_missing_ptbxl.py --rate 100   # 只补 100Hz（训练用）
"""
from __future__ import annotations

import argparse
import http.client
import os
import queue
import threading
import time

import pandas as pd

HOST = "physionet.org"
BASE = "/files/ptb-xl/1.0.3"


class Fetcher(threading.Thread):
    """一条线程 = 一条 keep-alive 连接，循环消费队列里的待下文件。"""

    def __init__(self, root, tasks, counters, lock, stop):
        super().__init__(daemon=True)
        self.root, self.tasks = root, tasks
        self.counters, self.lock, self.stop = counters, lock, stop
        self.conn = None

    def _connect(self):
        if self.conn is not None:
            self.conn.close()
        self.conn = http.client.HTTPSConnection(HOST, timeout=45)

    def _get(self, path):
        """返回 bytes；失败抛异常。"""
        for attempt in range(3):
            try:
                self.conn.request("GET", path)
                resp = self.conn.getresponse()
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}")
                return resp.read()
            except Exception:
                if attempt == 2:
                    raise
                self._connect()
                time.sleep(1.5)
        raise RuntimeError("unreachable")

    def run(self):
        self._connect()
        while not self.stop.is_set():
            try:
                rel, ext = self.tasks.get_nowait()
            except queue.Empty:
                return
            dst = os.path.join(self.root, rel + ext)
            try:
                data = self._get(f"{BASE}/{rel}{ext}")
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with open(dst + ".part", "wb") as f:
                    f.write(data)
                os.replace(dst + ".part", dst)
                with self.lock:
                    self.counters["ok"] += 1
            except Exception as e:  # noqa: BLE001
                with self.lock:
                    self.counters["err"] += 1
                    self.counters["errors"].append((rel + ext, f"{type(e).__name__} {e}"))
                if self.counters["err"] < 40:
                    print(f"  [失败] {rel}{ext}: {type(e).__name__} {e}", flush=True)
            with self.lock:
                self.counters["done"] += 1


def complete(root: str, col: str, workers: int) -> list[tuple[str, str]]:
    df = pd.read_csv(os.path.join(root, "ptbxl_database.csv"))
    missing = []
    for rel in df[col]:
        for ext in (".hea", ".dat"):
            p = os.path.join(root, rel + ext)
            if not (os.path.exists(p) and os.path.getsize(p) > 0):
                missing.append((rel, ext))
    print(f"[{col}] 待补 {len(missing)} 个文件 / {len(df)} 条记录", flush=True)
    if not missing:
        return []

    tasks: queue.Queue = queue.Queue()
    for item in missing:
        tasks.put(item)

    counters = {"done": 0, "ok": 0, "err": 0, "errors": []}
    lock = threading.Lock()
    stop = threading.Event()
    threads = [Fetcher(root, tasks, counters, lock, stop) for _ in range(workers)]
    for t in threads:
        t.start()

    t0 = time.time()
    total = len(missing)
    while any(t.is_alive() for t in threads):
        time.sleep(5)
        with lock:
            done, ok, err = counters["done"], counters["ok"], counters["err"]
        el = max(time.time() - t0, 0.1)
        print(f"  [{done}/{total} {done * 100 // total}%] ok={ok} err={err} "
              f"{done / el:.1f} 文件/s  已用 {el / 60:.1f} 分钟", flush=True)
        if done >= total:
            break

    print(f"[{col}] 完成：{counters['ok']} 成功 / {counters['err']} 失败，"
          f"用时 {(time.time() - t0) / 60:.1f} 分钟", flush=True)
    return counters["errors"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/ptb-xl")
    ap.add_argument("--rate", default="both", choices=["100", "500", "both"])
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    cols = {"100": ["filename_lr"], "500": ["filename_hr"],
            "both": ["filename_lr", "filename_hr"]}[args.rate]

    all_errors: list[tuple[str, str]] = []
    for col in cols:
        all_errors += complete(args.root, col, args.workers)

    if all_errors:
        print("\n失败清单（前 20 条）:")
        for rel, err in all_errors[:20]:
            print("  ", rel, err)
    else:
        print("\n全部补齐，无失败。")


if __name__ == "__main__":
    main()
