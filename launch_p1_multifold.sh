#!/bin/bash
# P0 supplement (总纲 Task 1): complete the 3-fold grid for the two combined models.
#
# The P0 batch only ran ab_shallow_pol / ab_improved on fold 10. A 3-fold claim
# needs folds 9 and 8 as well -- and with the SAME protocol as everything else in
# this project (25_multifold.py: val = test_fold-1, train = the other 8 folds).
#
# 4 runs, one per GPU, all in parallel. Each run is ~11.5-12 h at 100 epochs,
# so the whole batch lands in roughly 12 h rather than 48 h.
#
#   GPU0: ab_shallow_pol fold9
#   GPU1: ab_shallow_pol fold8
#   GPU2: ab_improved    fold9
#   GPU3: ab_improved    fold8
#
# Config is byte-identical to 26_ablation.py / launch_p0.sh:
#   100 epochs, bs 96, lr 1e-3, wd 1e-4, warmup 3, cosine, EMA 0.999,
#   grad_clip 1.0, seed 0, pos_weight = sqrt(cap 8.0)
#
# Usage (on the container):
#   nohup bash /mnt/ECG-SNN-LowPower/launch_p1_multifold.sh \
#         > /mnt/ECG-SNN-LowPower/logs/launch_p1_multifold.out 2>&1 &
cd /mnt/ECG-SNN-LowPower || exit 1
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate shadocformer
mkdir -p logs
TS=$(date +%m%d_%H%M)
echo "LAUNCH P1-MULTIFOLD TS=$TS $(date)"

run() {  # $1=gpu  $2=model  $3=test_fold
  echo "START $2 fold$3 on GPU$1 $(date)"
  CUDA_VISIBLE_DEVICES=$1 python3 experiments/25_multifold.py \
      --model "$2" --test-folds "$3" --epochs 100 \
      > "logs/p1m_${2}_fold${3}_${TS}.log" 2>&1
  rc=$?
  echo "DONE $2 fold$3 rc=$rc $(date)"
}

( run 0 ab_shallow_pol 9 ) &
( run 1 ab_shallow_pol 8 ) &
( run 2 ab_improved    9 ) &
( run 3 ab_improved    8 ) &
wait
echo "P1 MULTIFOLD ALL DONE $(date)"
