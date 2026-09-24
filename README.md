# ECG-SNN-LowPower

**基于脉冲神经网络的低功耗心电异常检测系统** —— 本科毕业论文复现工程。

## 复现对象

上游项目：[`hanshaunlee/neurocardio`](https://github.com/hanshaunlee/neurocardio)（MIT）

选择理由：它是目前唯一同时满足以下条件、且真正能跑通的公开工程——

1. **覆盖两个目标数据集**：主线是 PTB-XL 12 导联五超类诊断（文献 CNN/Transformer 区间 0.92–0.94 AUROC 的**竞争性**基准），`archive/cardiospike_mitbih/` 是 MIT-BIH 心律失常（DS1/DS2 **病人不交叉**划分 + AAMI 五类）。
2. **低功耗叙事是量出来的，不是吹的**：`neurocardio/compute.py` 用实测脉冲发放率算突触操作数（SOP），对比参数匹配的稠密 CNN 基线，给出每层能量账。
3. **完整可复现**：仓库自带数据下载/预处理、训练、评测、基线对比脚本，以及作者提交的权重与指标 JSON（可直接对拍）。
4. 附带 web 演示（`web/`、`modal_app/`），对应"系统怎么呈现"这一环节。

## 参考指标（作者提交，待本机复现对拍）

| 模型 | PTB-XL macro-AUROC | 说明 |
|---|---|---|
| **SNN（主线）** | **0.8663** | 参数更少，能耗约 1/287 |
| ResNet1D 基线 | 0.9017 | 稠密基线 |
| CNN 基线 | ~0.90 | 稠密基线 |

MIT-BIH 分支（`archive/cardiospike_mitbih/`）：Conv-LIF SNN ≈272k 参数，25 epoch，作者报告 accuracy **0.8683**、macro-F1 **0.4153**。

## 目录约定（CLAUDE.md 规定）

| 位置 | 路径 |
|---|---|
| 容器工作根 | `/mnt/` |
| 容器项目目录 | `/mnt/ECG-SNN-LowPower/` |
| 本地代码目录 | `C:\迅雷下载\实验\ECG-SNN-LowPower\` |
| 数据集（固定） | `/mnt/ptb-xl/`（PTB-XL 1.0.3）、`/mnt/mit-bih/`（MIT-BIH mitdb） |

同步方向：**本地 → GitHub → 容器 `git pull`**，不在容器内直接改代码。

```
/mnt/ECG-SNN-LowPower/
├── experiments/            本仓库内容：复现驱动脚本
├── third_party/neurocardio 上游代码（clone，只读底本）
├── logs/                   训练日志（nohup 输出）
├── results/                复现产出的指标与权重
└── legacy_july/            2026-07 旧 SNN 实验产物归档
```

## 复现步骤

```bash
# 0. 容器内首次准备（数据集按固定位置软链，避免重复下载）
bash experiments/00_prepare.sh

# 1. 构建 PTB-XL 缓存（21837 条记录 → data/ptbxl_cache）
python3 experiments/01_build_ptbxl_cache.py

# 2. PTB-XL 主线训练（MODEL=snn|cnn|resnet，GPU=0..3）
MODEL=snn EPOCHS=60 GPU=0 bash experiments/02_run_ptbxl.sh

# 3. MIT-BIH 分支（下载 → 预处理 → 训练 → 评测）
bash experiments/03_run_mitbih.sh
```

## 进度

- [x] 立项、选定上游工程、环境就绪（`snntorch 1.0.0` / `torch 2.5.1+cu124` / 4×RTX 3090）
- [ ] PTB-XL 缓存构建
- [ ] PTB-XL SNN 复现，与 0.8663 对拍
- [ ] PTB-XL CNN / ResNet 基线复现，与 0.9017 对拍
- [ ] MIT-BIH 分支复现，与 0.8683 / 0.4153 对拍
- [ ] 能耗对比表复现（`compute.py`）
