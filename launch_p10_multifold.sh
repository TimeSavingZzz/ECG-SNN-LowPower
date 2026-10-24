#!/bin/bash
# P10: extend the same-fold paired ablation from n=3 to n=10 (PTB-XL official 10 folds).
#
# Adds test folds 7..1 for the four models that enter the primary comparison.
# ab_improved is deliberately NOT extended: its 3-fold mean-pooling null already
# has a narrow CI, and it stays in the exploratory family.
#
# 28 runs, dynamic queue over 4 GPUs (flock claim). Each run ~11-12 h at 100
# epochs. A faster model therefore picks up the next job instead of idling.
# Re-running this script is safe: any fold that already has test_metrics.json
# is skipped, and a partial run resumes from latest.pt (TrainConfig resume=True).
#
# Pre-registration: records/01_实验思路.md, entry 013.
#
# Usage:
#   nohup bash /mnt/ECG-SNN-LowPower/launch_p10_multifold.sh #         > /mnt/ECG-SNN-LowPower/logs/launch_p10_multifold.out 2>&1 &

set -u
ROOT=/mnt/ECG-SNN-LowPower
cd "$ROOT" || exit 1
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate shadocformer
mkdir -p logs

TS=$(date +%m%d_%H%M)
QUEUE="$ROOT/logs/p10_queue_${TS}.txt"
LOCK="$ROOT/logs/p10_queue_${TS}.lock"
FAILED="$ROOT/logs/p10_failed_${TS}.txt"
SUM="$ROOT/logs/p10_orchestrator_${TS}.log"

MODELS="ab_full ab_shallow ab_polarity ab_shallow_pol"
FOLDS="7 6 5 4 3 2 1"

# Interleave models within each fold so the four GPUs start on different variants.
: > "$QUEUE"
for f in $FOLDS; do
  for m in $MODELS; do
    echo "$m $f" >> "$QUEUE"
  done
done
echo "QUEUE $QUEUE ($(wc -l < "$QUEUE") jobs) TS=$TS $(date)" | tee -a "$SUM"

claim() {
  local job=""
  {
    flock -x 9
    if [ -s "$QUEUE" ]; then
      job=$(head -n1 "$QUEUE")
      tail -n +2 "$QUEUE" > "$QUEUE.tmp" && mv "$QUEUE.tmp" "$QUEUE"
    fi
  } 9>>"$LOCK"
  printf '%s' "$job"
}

worker() {
  local gpu=$1 job m f out log rc
  while :; do
    job=$(claim)
    if [ -z "$job" ]; then
      echo "[GPU$gpu] queue empty, exit $(date)" | tee -a "$SUM"
      break
    fi
    read -r m f <<< "$job"
    out="$ROOT/results/repro_${m}_fold${f}"
    if [ -f "$out/test_metrics.json" ]; then
      echo "[SKIP] GPU$gpu $m fold$f (already done) $(date)" | tee -a "$SUM"
      continue
    fi
    log="$ROOT/logs/p10_${m}_fold${f}_${TS}.log"
    echo "[START] GPU$gpu $m fold$f $(date) log=$log" | tee -a "$SUM"
    CUDA_VISIBLE_DEVICES=$gpu python3 experiments/25_multifold.py         --model "$m" --test-folds "$f" --epochs 100 >> "$log" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ] && [ -f "$out/test_metrics.json" ]; then
      echo "[DONE] GPU$gpu $m fold$f rc=0 $(date)" | tee -a "$SUM"
    else
      echo "[FAIL] GPU$gpu $m fold$f rc=$rc $(date)" | tee -a "$SUM"
      echo "$job" >> "$FAILED"
    fi
  done
}

for g in 0 1 2 3; do
  worker "$g" &
done
wait
echo "P10 ALL DONE $(date)" | tee -a "$SUM"
