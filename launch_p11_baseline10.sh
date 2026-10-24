#!/bin/bash
# P1-1: put the CNN and ResNet1D baselines on the SAME ten-fold grid as the SNN.
#
# Why: the headline contrast (SNN 0.859 vs CNN 0.904) currently rests on a single
# test fold (fold 10) and a single seed, while every SNN claim rests on 10 folds.
# Cost is tiny (each CNN/ResNet run is ~6 min at 100 epochs) so there is no reason
# to leave the baseline on n=1.
#
# Config is byte-identical to repro_cnn / repro_resnet (100 ep, bs 96, lr 1e-3,
# wd 1e-4, warmup 3, cosine, EMA 0.999, seed 0, sqrt pos-weight cap 8.0).
# Output: results/repro_{cnn,resnet}_fold{1..10}  (new dirs, nothing overwritten).
#
#   nohup bash /mnt/ECG-SNN-LowPower/launch_p11_baseline10.sh > logs/launch_p11.out 2>&1 &

set -u
ROOT=/mnt/ECG-SNN-LowPower
cd "$ROOT" || exit 1
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate shadocformer
mkdir -p logs

TS=$(date +%m%d_%H%M)
SUM="$ROOT/logs/p11_baseline10_${TS}.log"

run_series() {
  local gpu=$1 model=$2
  for f in 1 2 3 4 5 6 7 8 9 10; do
    local out="$ROOT/results/repro_${model}_fold${f}"
    if [ -f "$out/test_metrics.json" ]; then
      echo "[SKIP] GPU$gpu $model fold$f (exists)" >> "$SUM"
      continue
    fi
    local log="$ROOT/logs/p11_${model}_fold${f}_${TS}.log"
    echo "[START] GPU$gpu $model fold$f $(date)" >> "$SUM"
    CUDA_VISIBLE_DEVICES=$gpu python3 experiments/25_multifold.py \
        --model "$model" --test-folds "$f" --epochs 100 >> "$log" 2>&1
    local rc=$?
    if [ "$rc" -eq 0 ] && [ -f "$out/test_metrics.json" ]; then
      echo "[DONE] GPU$gpu $model fold$f $(date)" >> "$SUM"
    else
      echo "[FAIL] GPU$gpu $model fold$f rc=$rc" >> "$SUM"
    fi
  done
}

echo "P11 baseline-10fold start $(date) TS=$TS" | tee -a "$SUM"
( run_series 1 cnn ) &
( run_series 2 resnet ) &
wait
echo "P11 ALL DONE $(date)" | tee -a "$SUM"
