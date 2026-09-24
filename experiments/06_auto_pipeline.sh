#!/usr/bin/env bash
# 无人值守编排：等数据齐备 → 建全量缓存 → 三卡并发训练 → 能耗对比 → 汇总
#
#   nohup bash experiments/06_auto_pipeline.sh > logs/orchestrator.log 2>&1 &
#
# 全程后台运行，Claude / SSH 断开都不影响。观察进度： tail -f logs/orchestrator.log
# 环境变量：EPOCHS（默认 100）、WALLCLOCK（默认 10 小时，到点优雅停止并保存最优）
set -uo pipefail

ROOT=${ROOT:-/mnt/ECG-SNN-LowPower}
PY=${PY:-/opt/miniconda3/envs/shadocformer/bin/python}
LOG=$ROOT/logs
CACHE=${CACHE:-$ROOT/results/ptbxl_cache}
EPOCHS=${EPOCHS:-100}
WALLCLOCK=${WALLCLOCK:-10}

mkdir -p "$LOG" "$ROOT/results"

log() { echo "[$(date '+%F %T')] $*"; }

log "=== 编排启动 (EPOCHS=$EPOCHS, WALLCLOCK=${WALLCLOCK}h, CACHE=$CACHE) ==="

# ---- 1. 等 PTB-XL records100 齐备 ----
while true; do
  H=$(find /mnt/ptb-xl/records100 -name '*_lr.hea' 2>/dev/null | wc -l)
  if [ "$H" = "21799" ]; then
    log "PTB-XL records100 齐备 ($H/21799)"
    break
  fi
  log "  等待数据 … $H/21799"
  sleep 60
done

# ---- 2. 建全量缓存 ----
if [ ! -f "$CACHE/labels.csv" ]; then
  log "构建全量缓存 → $CACHE"
  PYTHONPATH="$ROOT/third_party/neurocardio" "$PY" -u \
    "$ROOT/experiments/01_build_ptbxl_cache.py" --out "$CACHE" \
    >> "$LOG/build_cache_full.log" 2>&1
  log "缓存构建退出码 $?"
fi
if [ ! -f "$CACHE/labels.csv" ]; then
  log "!! 缓存不可用，编排终止（见 $LOG/build_cache_full.log）"
  exit 1
fi
log "缓存就绪：$(du -sh "$CACHE" | cut -f1)"

# ---- 3. 三卡并发训练（GPU3 留给 MIT-BIH 分支）----
pids=""
for spec in "snn 0" "cnn 1" "resnet 2"; do
  m=${spec% *}
  g=${spec#* }
  pid=$(MODEL="$m" EPOCHS="$EPOCHS" GPU="$g" WALLCLOCK="$WALLCLOCK" \
        TAG="repro_$m" CACHE="$CACHE" \
        bash "$ROOT/experiments/02_run_ptbxl.sh" | awk '/^PID:/{print $2}')
  log "启动 $m (GPU$g) PID=$pid → $LOG/train_repro_$m.log"
  pids="$pids $pid"
done

# ---- 4. 等三张卡都跑完 ----
for p in $pids; do
  while kill -0 "$p" 2>/dev/null; do sleep 120; done
  log "PID $p 已结束"
done

# ---- 5. 能耗 / 延迟 / 精度对比 ----
log "生成 comparison.json"
"$PY" -u "$ROOT/experiments/07_energy_report.py" \
  --cache-dir "$CACHE" --runs-root "$ROOT/results" \
  --out "$ROOT/results/comparison.json" >> "$LOG/energy_report.log" 2>&1
log "能耗报告退出码 $?"

# ---- 6. 汇总核心数字 ----
for t in repro_snn repro_cnn repro_resnet; do
  f="$ROOT/results/$t/test_metrics.json"
  if [ -f "$f" ]; then
    log "$t: $("$PY" -c "
import json
d = json.load(open('$f'))
print('macro-AUROC=%.4f macro-AUPRC=%.4f' % (d['macro_auroc'], d['macro_auprc']))
")"
  else
    log "$t: 无 test_metrics.json"
  fi
done

log "=== 编排结束 ==="
