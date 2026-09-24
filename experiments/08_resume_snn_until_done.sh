#!/usr/bin/env bash
# SNN 续跑直到跑满 TARGET 轮，然后重跑能耗报告。
#
#   nohup bash experiments/08_resume_snn_until_done.sh > logs/resume_snn.log 2>&1 &
#
# 为什么需要它：SNN 主线单轮约 10~12 分钟（作者提交的 history.json 亦为 ~700s/轮），
# 100 轮 ≈ 16~20h GPU 时间，远超单次 10h 墙钟。作者自己也是**分多段续跑**跑完 200 轮
# 的——其 results/main/history.json 里 wallclock_seconds 非单调
# （708→1411→2109→…→ep49=12795→ep99=10599→ep149=41608→ep199=1218），
# 正是每次进程重启后 t_run_start 归零的痕迹；配置里 wallclock_hours=8.0 只是单段上限。
#
# 续跑的数值正确性由 train.py 保证：latest.pt 里存了 model/ema/opt/**sched**，
# 恢复时 sched.load_state_dict 使余弦退火沿原 T_max=--epochs 连续推进，
# epoch_start = ckpt.epoch + 1，且墙钟是在**轮首**检查的，到点即干净退出。
# 因此 --epochs 必须与首次启动时一致（本工程统一用 100）。
set -uo pipefail

ROOT=${ROOT:-/mnt/ECG-SNN-LowPower}
NC=$ROOT/third_party/neurocardio
PY=${PY:-/opt/miniconda3/envs/shadocformer/bin/python}
LOG=$ROOT/logs
CACHE=${CACHE:-$ROOT/results/ptbxl_cache}
OUT=${OUT:-$ROOT/results/repro_snn}

TARGET=${TARGET:-100}                     # 必须与 06 的 EPOCHS 一致（决定 LR 的 T_max）
GPU=${GPU:-0}
HOURS_PER_ATTEMPT=${HOURS_PER_ATTEMPT:-12}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-10}
BATCH=${BATCH:-96}

mkdir -p "$LOG" "$OUT"
log() { echo "[$(date '+%F %T')] $*"; }

done_epochs() {
  "$PY" -c "
import json, sys
d = json.load(open(sys.argv[1]))
h = d['history'] if isinstance(d, dict) else d
print(len(h))
" "$OUT/history.json" 2>/dev/null || echo 0
}

log "=== SNN 续跑守护启动 (TARGET=$TARGET 轮, 每段 ${HOURS_PER_ATTEMPT}h, GPU$GPU) ==="

# ---- 1. 等 06 编排退出（避免与它启动的训练进程争抢同一 output-dir）----
if ps -ef | grep -q '[0]6_auto_pipeline'; then
  log "06_auto_pipeline 仍在运行，等它退出 …"
  while ps -ef | grep -q '[0]6_auto_pipeline'; do sleep 120; done
  log "06_auto_pipeline 已退出"
fi

# ---- 2. 反复续跑直到跑满 ----
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  n=$(done_epochs)
  log "当前进度 $n/$TARGET 轮"
  [ "$n" -ge "$TARGET" ] && break

  log "第 $attempt 次续跑：wallclock=${HOURS_PER_ATTEMPT}h，日志追加至 $LOG/train_repro_snn.log"
  cd "$NC"
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=$GPU "$PY" -u -m neurocardio.train \
    --cache-dir "$CACHE" \
    --output-dir "$OUT" \
    --label-set diagnostic_superclass \
    --model snn \
    --epochs "$TARGET" \
    --batch-size "$BATCH" \
    --lr 1e-3 \
    --num-workers 4 \
    --wallclock-hours "$HOURS_PER_ATTEMPT" \
    --device cuda \
    >> "$LOG/train_repro_snn.log" 2>&1
  log "第 $attempt 段退出码 $?，进度 $(done_epochs)/$TARGET 轮"
done

# ---- 3. 跑满则重跑能耗报告（覆盖 06 结束时那份被墙钟截断的版本）----
n=$(done_epochs)
if [ "$n" -ge "$TARGET" ]; then
  log "SNN 已跑满 $n 轮 → 重跑能耗对比（覆盖 results/comparison.json）"
  "$PY" -u "$ROOT/experiments/07_energy_report.py" \
    --cache-dir "$CACHE" --runs-root "$ROOT/results" \
    --out "$ROOT/results/comparison.json" >> "$LOG/energy_report_final.log" 2>&1
  log "能耗报告退出码 $?"
  for t in repro_snn repro_cnn repro_resnet; do
    f="$ROOT/results/$t/test_metrics.json"
    [ -f "$f" ] && log "$t: $("$PY" -c "
import json, sys
d = json.load(open(sys.argv[1]))
print('macro-AUROC=%.4f macro-AUPRC=%.4f' % (d['macro_auroc'], d['macro_auprc']))
" "$f")"
  done
  log "=== 全部完成；对拍基准：SNN 0.8663 / CNN ~0.90 / ResNet1D 0.9017 ==="
else
  log "!! 仍只有 $n/$TARGET 轮（已达最大尝试次数 $MAX_ATTEMPTS），未跑满；"
  log "   可再手动跑： TARGET=$TARGET nohup bash experiments/08_resume_snn_until_done.sh"
fi
