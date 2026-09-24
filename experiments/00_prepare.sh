#!/usr/bin/env bash
# 容器内一次性准备：把固定位置的数据集接到上游工程期望的路径上。
# 数据集只保留 /mnt/ 下的单一副本，不在工程里重复下载。
set -euo pipefail

ROOT=${ROOT:-/mnt/ECG-SNN-LowPower}
NC=$ROOT/third_party/neurocardio
CA=$NC/archive/cardiospike_mitbih
PTBXL_SRC=${PTBXL_SRC:-/mnt/ptb-xl}
MITBIH_SRC=${MITBIH_SRC:-/mnt/mit-bih}

mkdir -p "$ROOT"/{logs,results,experiments} "$MITBIH_SRC"

echo "== 检查 PTB-XL 数据集 =="
for f in ptbxl_database.csv scp_statements.csv records100 records500; do
  [ -e "$PTBXL_SRC/$f" ] && echo "  ok   $f" || { echo "  缺失 $f"; exit 1; }
done
echo "  records100 子目录: $(ls "$PTBXL_SRC/records100" | wc -l), records500 子目录: $(ls "$PTBXL_SRC/records500" | wc -l)"

echo "== 把 MIT-BIH 固定目录接到工程期望路径 =="
if [ ! -e "$CA/data/mitdb" ]; then
  mkdir -p "$CA/data"
  ln -s "$MITBIH_SRC" "$CA/data/mitdb"
fi
ls -ld "$CA/data/mitdb"

echo "== 环境自检 =="
PY=${PY:-/opt/miniconda3/envs/shadocformer/bin/python}
$PY - <<'PYEOF'
import torch, snntorch, wfdb, numpy, pandas, sklearn, scipy
print("  torch      ", torch.__version__, "| cuda", torch.cuda.is_available(), "| gpus", torch.cuda.device_count())
print("  snntorch   ", snntorch.__version__)
print("  wfdb       ", wfdb.__version__)
print("  numpy      ", numpy.__version__)
print("  sklearn    ", sklearn.__version__)
PYEOF

echo "== 就绪 =="
