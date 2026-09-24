#!/usr/bin/env python3
"""补齐 PTB-XL 缺失记录（断点续传，跳过已存在文件）。

背景：`/mnt/ptb-xl` 现有副本来自夸克网盘搬运，只搬了一半——
21799 条记录里缺 10799 条（``records100/00000/00513_lr`` 之后全无）。
本脚本以 ``ptbxl_database.csv`` 为准，逐条核对 ``.hea``/``.dat``，
从 PhysioNet 官方 1.0.3 版补齐，不重下已有文件。

用法::

    python3 experiments/04_fetch_missing_ptbxl.py              # 补齐 100Hz + 500Hz
    python3 experiments/04_fetch_missing_ptbxl.py --rate 100   # 只补 100Hz（训练用）
"""
from __future__ import annotations

import argparse
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

BASE = "https://physionet.org/files/ptb-xl/1.0.3"


def fetch_one(root: str, rel: str) -> tuple[str, str | None]:
    """下载单条记录的 .hea + .dat，已存在则跳过。返回 (rel, 错误信息)。"""
    for ext in (".hea", ".dat"):
        url = f"{BASE}/{rel}{ext}"
        dst = os.path.join(root, rel + ext)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        for attempt in range(4):
            try:
                urllib.request.urlretrieve(url, dst + ".part")
                os.replace(dst + ".part", dst)
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 3:
                    return rel, f"{ext}: {type(e).__name__} {e}"
                time.sleep(2)
    return rel, None


def complete(root: str, col: str, workers: int) -> list[tuple[str, str]]:
    df = pd.read_csv(os.path.join(root, "ptbxl_database.csv"))
    todo = [
        f for f in df[col]
        if not (os.path.exists(os.path.join(root, f + ".hea"))
                and os.path.exists(os.path.join(root, f + ".dat")))
    ]
    print(f"[{col}] 待补 {len(todo)} / {len(df)} 条", flush=True)
    if not todo:
        return []

    errors: list[tuple[str, str]] = []
    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch_one, root, rel) for rel in todo]
        for fut in as_completed(futures):
            rel, err = fut.result()
            done += 1
            if err:
                errors.append((rel, err))
            if done % 500 == 0 or done == len(todo):
                el = max(time.time() - t0, 0.1)
                print(f"  [{done}/{len(todo)} {done * 100 // len(todo)}%] "
                      f"{done / el:.0f} 条/s  {el:.0f}s", flush=True)
    print(f"[{col}] 完成，{len(errors)} 条失败", flush=True)
    return errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/ptb-xl")
    ap.add_argument("--rate", default="100",
                    help="100 / 500 / both")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    cols = {"100": ["filename_lr"], "500": ["filename_hr"],
            "both": ["filename_lr", "filename_hr"]}[args.rate]

    all_errors: list[tuple[str, str]] = []
    for col in cols:
        all_errors += complete(args.root, col, args.workers)

    if all_errors:
        print("\n失败清单:")
        for rel, err in all_errors[:20]:
            print("  ", rel, err)
    else:
        print("\n全部补齐，无失败。")


if __name__ == "__main__":
    main()
