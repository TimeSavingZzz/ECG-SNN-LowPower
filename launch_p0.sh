#!/bin/bash
# P0: 严格"同折"对照消融多折 —— ab_shallow / ab_polarity / ab_full 各补 fold 9、8
#     （fold 10 已有，合并后每个变体都是 3 折 mean±std，且 Δ 在同折内计算，不受折间漂移污染）
# P1: 组合改进模型 —— ab_shallow_pol（3块+双极性编码）、ab_improved（再+均值池化），单折 fold10
#
# 配置与 26_ablation.py 完全一致：100 epoch、folds1-8 train / fold-1 val、bs96、
# lr1e-3、wd1e-4、warmup3、cos、EMA0.999、grad_clip1.0、seed0、pos_weight=sqrt cap8.0
# 25_multifold.py 的 split(): val = test_fold-1（折1时为折10），train = 其余 8 折
#
# 4 卡 × 2 轮，每轮 ~15h，预计总计 ~30h
#   GPU0: ab_shallow fold9   -> ab_full fold9
#   GPU1: ab_shallow fold8   -> ab_full fold8
#   GPU2: ab_polarity fold9  -> ab_shallow_pol fold10   (P1a)
#   GPU3: ab_polarity fold8  -> ab_improved     fold10   (P1b)
cd /mnt/ECG-SNN-LowPower || exit 1
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate shadocformer
mkdir -p logs
TS=$(date +%m%d_%H%M)
echo "LAUNCH TS=$TS $(date)"

run() {  # $1=gpu  $2=model  $3=fold
  echo "START $2 fold$3 on GPU$1 $(date)"
  CUDA_VISIBLE_DEVICES=$1 python3 experiments/25_multifold.py \
      --model "$2" --test-folds "$3" --epochs 100 \
      > "logs/p0_${2}_fold${3}_${TS}.log" 2>&1
  rc=$?
  echo "DONE $2 fold$3 rc=$rc $(date)"
}

( run 0 ab_shallow  9 ; run 0 ab_full        9 ) &
( run 1 ab_shallow  8 ; run 1 ab_full        8 ) &
( run 2 ab_polarity 9 ; run 2 ab_shallow_pol 10 ) &
( run 3 ab_polarity 8 ; run 3 ab_improved    10 ) &
wait
echo "P0/P1 ALL DONE $(date)"
