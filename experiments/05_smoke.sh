#!/usr/bin/env bash
# 冒烟测试：只取前 N 条 PTB-XL 记录建小缓存，跑 1 epoch / 30 步，
# 验证 数据读取 → Δ调制脉冲编码 → SNN → BPTT 训练 → 指标评测 全链路可通过。
#
#   N=400 bash experiments/05_smoke.sh
#
# 目的：在等全量数据补齐/建缓存的同时，提前暴露 snntorch 1.0 与上游代码
# （针对 snntorch>=0.9 编写）之间的 API 兼容问题。
set -euo pipefail

ROOT=${ROOT:-/mnt/ECG-SNN-LowPower}
NC=$ROOT/third_party/neurocardio
PY=${PY:-/opt/miniconda3/envs/shadocformer/bin/python}
N=${N:-400}
GPU=${GPU:-0}
CACHE=$ROOT/results/smoke_cache
OUT=$ROOT/results/smoke_run

mkdir -p "$ROOT/logs"
cd "$NC"

echo "== 1/2 构建 $N 条记录的冒烟缓存 =="
PYTHONPATH=. "$PY" -c "
from pathlib import Path
from neurocardio.data import PTBXLConfig, build_cache
build_cache(PTBXLConfig(root=Path('/mnt/ptb-xl'), sampling_rate=100),
            Path('$CACHE'), indices=range(1, $N + 1))
"

echo "== 2/2 训练 1 epoch / 最多 30 步 =="
PYTHONPATH=. CUDA_VISIBLE_DEVICES=$GPU "$PY" -m neurocardio.train \
  --cache-dir "$CACHE" \
  --output-dir "$OUT" \
  --epochs 1 --batch-size 16 --num-workers 0 \
  --max-steps-per-epoch 30 \
  --device cuda

echo "== 冒烟结果 =="
cat "$OUT/test_metrics.json" 2>/dev/null || echo "（未产出指标文件，见上方日志）"
