# 复现说明

本文档给出从零复现本项目核心结果的完整步骤与**可检验的验收标准**。
核心结果是十折同折配对消融（4 个模型 × 10 折 = 40 次训练），见第 5 节。

---

## 1. 环境

| 项目 | 值 |
|---|---|
| Python | 3.10.20（conda env `shadocformer`） |
| PyTorch | 2.5.1+cu124 |
| snntorch | 1.0.0 |
| GPU | NVIDIA RTX 3090 × 4 |
| 宿主驱动 / CUDA | 555.42.02 / 12.5 |

```bash
conda create -n shadocformer python=3.10 -y
conda activate shadocformer
pip install -r requirements.txt
```

**已知版本偏差**：上游 neurocardio 锁定 snntorch 0.9.4 / torch 2.4.1，本项目使用 1.0.0 / 2.5.1+cu124。
该偏差不是配置错误，已记录在论文的「版本偏差」一节；它意味着与上游论文的逐位复现**不成立**，
但本项目内部的全部比较（消融、十折配对）都在这同一版本下完成，不受影响。

---

## 2. 上游代码

```bash
git clone https://github.com/hanshaunlee/neurocardio third_party/neurocardio
git -C third_party/neurocardio checkout 5b5fd23645dfd8753d48aa6ddf4bac291c16a757
cp experiments/neurocardio_ext/{ablation,compact,sarhaddi,factory}.py \
   third_party/neurocardio/neurocardio/
```

`third_party/` 在本仓库的 `.gitignore` 中（上游副本不入库），所以我们新增/修改的
全部模型代码由 `experiments/neurocardio_ext/` 存档，见该目录的 `README.md`。

上游为 MIT 许可；`factory.py` 是其派生文件，`factory.patch` 给出相对上游的完整 diff。

---

## 3. 数据准备

**数据集**：PTB-XL v1.0.3（12 导联、10 s、100 Hz 与 500 Hz 两套，本项目只使用 100 Hz）。
下载地址：https://physionet.org/content/ptb-xl/1.0.3/
本项目约定数据集放在 `/mnt/ptb-xl`，**不进入仓库**（`.gitignore` 已排除）。

```bash
# 一次性把数据集接到工程期望路径
bash experiments/00_prepare.sh

# 构建缓存（结果写到 results/ptbxl_cache）
python3 experiments/01_build_ptbxl_cache.py
```

**数据口径**（必须与之一致，否则指标不可比）：

- 原始 21 837 条记录；剔除 449 条不含任何诊断超类标签的记录后保留 **21 388 条**。
- 标签：`diagnostic_superclass` 五类（NORM / MI / STTC / CD / HYP），多标签、one-vs-rest。
- 官方 `strat_fold` 十折，每折约 2 146 条。
- 采样率 100 Hz，输入长度 T = 1000。

准备完成后自检：

```bash
python3 experiments/05_smoke.sh   # 或直接跑一次 1-epoch 冒烟
```

---

## 4. 训练命令

### 4.1 单折（fold 10，主线三个基线）

```bash
MODEL=snn   EPOCHS=100 GPU=0 bash experiments/02_run_ptbxl.sh
MODEL=cnn   EPOCHS=100 GPU=1 bash experiments/02_run_ptbxl.sh
MODEL=resnet EPOCHS=100 GPU=2 bash experiments/02_run_ptbxl.sh
```

### 4.2 单个消融变体，单个测试折

```bash
python3 experiments/26_ablation.py --model ab_shallow_pol --epochs 100
# 可选 model: ab_full ab_shallow ab_polarity ab_multiq ab_noattn ab_nogate ab_nodilate
```

### 4.3 多折（任意测试折，val = fold-1，fold 1 时 val = 10）

```bash
python3 experiments/25_multifold.py --model ab_shallow --test-folds 10,9,8 --epochs 100
```

### 4.4 十折全套（28 次新训练，4 卡动态队列）

```bash
nohup bash launch_p10_multifold.sh > logs/launch_p10.out 2>&1 &
```

每个 run 独占一张 3090 时约 4.7 h（100 epoch），4 卡并发时因 NFS 争用约 12 h。
`ab_full` 的 fold 10 不在这个脚本里（它由 4.2 的 `ab_full` 单独产出）。

### 4.5 汇总与统计

```bash
python3 experiments/31_p10_report.py     # -> results/p10_summary.json
```

---

## 5. 验收标准

复现是否成功，按下面三条判定。三条**全部**满足才算复现成功。

### 验收 1：结构量必须精确相等（决定性）

```bash
python3 -c "
import sys; sys.path.insert(0,'third_party/neurocardio')
from neurocardio.factory import make_model
print(sum(p.numel() for p in make_model('ab_full',5).parameters()))        # 期望 1893768
print(sum(p.numel() for p in make_model('ab_shallow_pol',5).parameters())) # 期望 528392
"
```

| 模型 | 参数量（必须精确相等） | 稠密 MAC（批 1，必须精确相等） |
|---|---|---|
| `ab_full` | 1 893 768 | 325 684 480 |
| `ab_shallow_pol` | 528 392 | 由 `cost_snn` 现场计算 |

参数量与 MAC 数与训练无关，**任何偏差都说明模型代码没有对上**，不需要继续往下测。

### 验收 2：单折结果（抽样验收，允许小量数值差异）

任选 `ab_shallow_pol` + 测试折 10 复现一次：

| 指标 | 记录值 | 允许偏差 |
|---|---|---|
| test macro-AUROC | 0.8796 | ≤ 0.2pp |
| test macro-AUPRC | 0.7245 | ≤ 0.5pp |

偏差来源是 CUDA 非确定性 kernel 与 DataLoader 线程调度，**不是**超参差异。
若偏差 > 0.5pp，先核对 `config.json` 是否逐字段一致，再怀疑数据缓存。

### 验收 3：十折均值（结论级验收）

| 模型 | 十折 AUROC 均值 ± 标准差 | 允许偏差 |
|---|---|---|
| `ab_full` | 0.8769 ± 0.0108 | 均值 ≤ 0.1pp |
| `ab_shallow` | 0.8872 ± 0.0085 | ≤ 0.1pp |
| `ab_polarity` | 0.8859 ± 0.0093 | ≤ 0.1pp |
| `ab_shallow_pol` | 0.8941 ± 0.0081 | ≤ 0.1pp |

配对差值（组合 − 6 块）= **+1.72pp**，是论文的核心数字。若复现后该值方向相反或绝对值 < 1.0pp，
则论文第 6 章的核心结论不成立，应视为复现失败而非数值漂移。

---

## 6. 已知偏差与不可复现项

| 项目 | 说明 |
|---|---|
| snntorch / torch 版本 | 上游锁 0.9.4 / 2.4.1，本项目 1.0.0 / 2.5.1+cu124 |
| Sarhaddi SNN-SRNN | 论文未公开循环层动力学细节，我方重建版未复现原文指标（0.7747 vs 0.889），属可复现性边界 |
| 能耗实测 | 依赖具体 GPU 与并发负载，绝对值不可跨机器比较；只应在同机同条件下比较 |
| 训练随机性 | 全部实验固定 `seed=0`；跨机器的 bitwise 复现不保证，按第 5 节容差判定 |
