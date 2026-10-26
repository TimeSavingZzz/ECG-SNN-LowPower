#!/usr/bin/env bash
# 2026-10-26 重跑失效产物（第 1 步）：全部为纯推理，无训练。
# 与 logs/rerun_artifacts.sh 的唯一差别：11 号显式 --ckpt 绑定冻结快照，
# 不再隐式读 results/repro_snn/best.pt（两者 md5 相同，结果应逐位一致）。
set -euo pipefail
cd /mnt/ECG-SNN-LowPower
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate shadocformer
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

echo "===== [0/8] freeze weight snapshot ====="
date -Is
cp -v results/repro_snn/best.pt results/pareto/_snapshot_best.pt
md5sum results/repro_snn/best.pt results/pareto/_snapshot_best.pt

echo "===== [1/8] 11 pareto sweep (30 pts, bound to snapshot) ====="
date -Is
python experiments/11_pareto_sweep.py --run repro_snn   --ckpt results/pareto/_snapshot_best.pt   --out results/pareto --baseline results/comparison.json --force

echo "===== [2/8] 16 calib noise floor ====="
date -Is
python experiments/16_calib_noise_floor.py

echo "===== [3/8] 18 bn shift probe ====="
date -Is
python experiments/18_bn_shift_probe.py

echo "===== [4/8] 20 audit figures ====="
date -Is
python experiments/20_audit_figures.py --out results/audit

echo "===== [5/8] 09 report ====="
date -Is
python experiments/09_make_report.py --runs-root results --out results/report

echo "===== [6/8] 13 offline demo ====="
date -Is
python experiments/13_build_offline_demo.py --all

echo "===== [7/8] 14 leak audit ====="
date -Is
python experiments/14_leak_audit.py --out results/leak_audit.json --refresh-snapshot

echo "===== [8/8] all done ====="
date -Is
ls -la results/pareto/pareto_summary.json results/audit results/report results/offline results/leak_audit.json
echo DONE
