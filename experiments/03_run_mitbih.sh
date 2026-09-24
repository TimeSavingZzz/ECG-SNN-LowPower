#!/usr/bin/env bash
# MIT-BIH 分支复现：下载 → 预处理(DS1/DS2 病人不交叉) → 训练 → 评测。
#
#   GPU=0 EPOCHS=25 bash experiments/03_run_mitbih.sh
#
# 数据落到固定位置 /mnt/mit-bih（工程内以软链接接入），训练后台 nohup 运行。
set -euo pipefail

ROOT=${ROOT:-/mnt/ECG-SNN-LowPower}
CA=$ROOT/third_party/neurocardio/archive/cardiospike_mitbih
PY=${PY:-/opt/miniconda3/envs/shadocformer/bin/python}
EPOCHS=${EPOCHS:-25}
GPU=${GPU:-0}
STAGE=${STAGE:-all}

mkdir -p "$ROOT/logs" /mnt/mit-bih
cd "$CA"

if [ "$STAGE" = "all" ] || [ "$STAGE" = "data" ]; then
  echo "== 1/3 下载 MIT-BIH（44 条记录，约 84MB，已存在则跳过）=="
  PYTHONPATH=. "$PY" src/download_data.py
  echo "   已就绪文件数: $(ls /mnt/mit-bih | wc -l)"

  echo "== 2/3 预处理：DS1/DS2 划分 + AAMI 五类标注 =="
  PYTHONPATH=. "$PY" src/preprocess.py
  ls -la data/processed/
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "train" ]; then
  echo "== 3/3 训练 Conv-LIF SNN（后台）=="
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=$GPU nohup "$PY" src/train.py \
    --epochs "$EPOCHS" \
    > "$ROOT/logs/mitbih_train.log" 2>&1 &
  echo "PID: $!  日志: $ROOT/logs/mitbih_train.log"
  echo "训练结束后评测: cd $CA && PYTHONPATH=. $PY src/evaluate.py"
fi
