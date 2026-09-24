#!/usr/bin/env bash
# PTB-XL 主线复现（SNN / CNN / ResNet1D），后台 nohup 运行。
#
#   MODEL=snn EPOCHS=60 GPU=0 bash experiments/02_run_ptbxl.sh
#
# 与作者提交配置对齐：batch 96 / lr 1e-3 / weight-decay 1e-4 / 余弦退火 /
# sqrt 正类加权 / EMA 0.999 —— 仅 epoch 数与墙钟上限按本机 3090 调整。
set -euo pipefail

ROOT=${ROOT:-/mnt/ECG-SNN-LowPower}
NC=$ROOT/third_party/neurocardio
PY=${PY:-/opt/miniconda3/envs/shadocformer/bin/python}

MODEL=${MODEL:-snn}
EPOCHS=${EPOCHS:-60}
GPU=${GPU:-0}
BATCH=${BATCH:-96}
LR=${LR:-1e-3}
WALLCLOCK=${WALLCLOCK:-6}
CACHE=${CACHE:-$ROOT/results/ptbxl_cache}
TAG=${TAG:-repro_${MODEL}_e${EPOCHS}}

mkdir -p "$ROOT/logs" "$ROOT/results"
LOG=$ROOT/logs/train_$TAG.log

cd "$NC"
PYTHONPATH=. CUDA_VISIBLE_DEVICES=$GPU nohup "$PY" -u -m neurocardio.train \
  --cache-dir "$CACHE" \
  --output-dir "$ROOT/results/$TAG" \
  --label-set diagnostic_superclass \
  --model "$MODEL" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH" \
  --lr "$LR" \
  --num-workers 4 \
  --wallclock-hours "$WALLCLOCK" \
  --device cuda \
  > "$LOG" 2>&1 &

echo "PID: $!  (MODEL=$MODEL EPOCHS=$EPOCHS GPU=$GPU)"
echo "日志: $LOG"
echo "查看: tail -f $LOG"
