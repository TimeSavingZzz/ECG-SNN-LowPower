#!/usr/bin/env python3
"""构建 PTB-XL 训练缓存。

读取固定位置的原始 PTB-XL（``/mnt/ptb-xl``，100 Hz 版本），逐条 wfdb 读取 →
按导联做鲁棒 z-score（median / 1.4826·MAD）→ 存成内存映射 npy + labels.csv，
即上游 ``neurocardio.train`` 消费的格式。

划分沿用文献惯例（Strodthoff et al. 2020）：fold 1-8 训练 / fold 9 验证 / fold 10 测试。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

NC = Path("/mnt/ECG-SNN-LowPower/third_party/neurocardio")
sys.path.insert(0, str(NC))

from neurocardio.data import PTBXLConfig, build_cache  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/ptb-xl", help="PTB-XL 原始数据目录（固定位置）")
    ap.add_argument("--out", default="/mnt/ECG-SNN-LowPower/results/ptbxl_cache",
                    help="缓存输出目录")
    ap.add_argument("--sampling-rate", type=int, default=100, choices=[100, 500])
    ap.add_argument("--label-set", default="diagnostic_superclass",
                    choices=["diagnostic_superclass", "diagnostic_subclass"])
    ap.add_argument("--limit", type=int, default=0, help=">0 时只处理前 N 条（冒烟测试用）")
    args = ap.parse_args()

    cfg = PTBXLConfig(root=Path(args.root), label_set=args.label_set,
                      sampling_rate=args.sampling_rate)
    indices = range(1, args.limit + 1) if args.limit > 0 else None

    t0 = time.time()
    build_cache(cfg, Path(args.out), indices=indices)
    print(f"\n缓存完成，用时 {time.time() - t0:.0f}s → {args.out}")


if __name__ == "__main__":
    main()
