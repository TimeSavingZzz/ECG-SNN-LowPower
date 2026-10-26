#!/bin/bash
# P0-3 全量：4 个配置并行占 4 张卡，各跑 10 折。
# 每个配置写各自的 json（p0_spike_profile_<cfg>.json），不并发写同一文件。
# 注意：CUDA_VISIBLE_DEVICES 已经把进程限制成单卡，所以对进程而言可见序号
# 永远是 0，--gpu 必须传 0（传 $i 会报 invalid device ordinal）。
cd /mnt/ECG-SNN-LowPower || exit 1
mkdir -p logs
TS=$(date +%m%d_%H%M)
PY=/opt/miniconda3/envs/shadocformer/bin/python
i=0
for cfg in ab_full ab_polarity ab_shallow ab_shallow_pol; do
  CUDA_VISIBLE_DEVICES=$i setsid nohup $PY experiments/36_p0_spike_profile.py \
      --config "$cfg" --folds 1,2,3,4,5,6,7,8,9,10 --gpu 0 \
      > "logs/p0_profile_${cfg}_${TS}.log" 2>&1 &
  echo "$cfg -> GPU$i PID $!"
  i=$((i+1))
done
echo "logs -> logs/p0_profile_*_${TS}.log"
