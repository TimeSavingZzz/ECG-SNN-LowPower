# ECG-SNN-LowPower

**基于脉冲神经网络的低功耗心电异常检测系统** —— 本科毕业论文复现工程。

## 复现对象

上游项目：[`hanshaunlee/neurocardio`](https://github.com/hanshaunlee/neurocardio)（MIT）

选择理由：它是目前唯一同时满足以下条件、且真正能跑通的公开工程——

1. **覆盖两个目标数据集**：主线是 PTB-XL 12 导联五超类诊断（文献 CNN/Transformer 区间 0.92–0.94 AUROC 的**竞争性**基准），`archive/cardiospike_mitbih/` 是 MIT-BIH 心律失常（DS1/DS2 **病人不交叉**划分 + AAMI 五类）。
2. **低功耗叙事是量出来的，不是吹的**：`neurocardio/compute.py` 用**实测**脉冲发放率算突触操作数（SOP），对比参数匹配的稠密基线，给出每层能量账。
3. **完整可复现**：仓库自带数据下载/预处理、训练、评测、基线对比，以及作者提交的权重与指标 JSON（可直接对拍）。
4. 附带 web 演示（`web/`、`modal_app/`），对应"系统怎么呈现"这一环节。

## 参考指标（作者提交，待本机复现对拍）

| 模型 | PTB-XL macro-AUROC | 参数 | 说明 |
|---|---|---|---|
| **SNN（主线）** | **0.8663** | 1.89 M | 估算能耗约为稠密基线的 1/287 |
| ResNet1D 基线 | 0.9017 | — | 稠密基线 |
| CNN 基线 | ~0.90 | — | 稠密基线 |

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
├── logs/                   全部日志（nohup 输出）
├── results/                缓存、训练产出、comparison.json
└── legacy_july/            2026-07 旧 SNN 实验产物归档
```

## 环境与数据

**环境偏差（重要）**：上游在 Modal 镜像里锁定了 `snntorch==0.9.4` / `torch==2.4.1` / `numpy==1.26.4` / `wfdb==4.1.2`；本机容器是
`snntorch 1.0.0` / `torch 2.5.1+cu124` / `numpy 2.2.6` / `wfdb 4.3.1`。
冒烟测试（`05_smoke.sh`）已确认全链路在该组合下可跑通。论文中应如实说明此偏差。

**数据获取**：

- 容器**可直连 PhysioNet**（实测 `https://physionet.org/files/ptb-xl/1.0.3/` → 200 / 1.4s），因此 PTB-XL 走直连补全（约 16 文件/s）。
- 2026-07 经**夸克网盘**搬来的那份 PTB-XL 是残档：21799 条记录只搬来 11000 条（`records100/00000/00513_lr` 之后大面积缺失），
  且搬运过程刷出过 816 GB 的登录失败日志。**该渠道不再用于数据集**。
- 上游提到 PTB-XL 在 **AWS Open Data** 有镜像（`s3://physionet-open/ptb-xl/1.0.3/`），云端 GPU 上吞吐远高于 physionet.org；
  本机若再需大批量拉取可考虑，但当前直连已够用。

## 复现步骤

```bash
# 0. 容器内首次准备（数据集按固定位置软链，环境自检）
bash experiments/00_prepare.sh

# 1. 补齐 PTB-XL 残档（幂等，跳过已存在文件）
python3 experiments/04_fetch_missing_ptbxl.py --rate both --workers 12

# 2. 冒烟测试：前 400 条记录走通全链路（约 1 分钟）
bash experiments/05_smoke.sh

# 3. 构建全量缓存（21388 条带诊断标签的记录）
python3 experiments/01_build_ptbxl_cache.py

# 4. PTB-XL 主线训练（单跑，GPU=0..3，后台 nohup）
MODEL=snn EPOCHS=100 GPU=0 WALLCLOCK=10 bash experiments/02_run_ptbxl.sh

# 5. MIT-BIH 分支（下载 → 预处理 → 训练 → 评测）
bash experiments/03_run_mitbih.sh

# 6. 能耗 / 延迟 / 精度对比表
python3 experiments/07_energy_report.py
```

**无人值守（推荐）**：第 1~6 步可交给编排脚本一次跑完，全程后台，Claude / SSH 断开都不影响：

```bash
nohup bash experiments/06_auto_pipeline.sh > logs/orchestrator.log 2>&1 &
tail -f logs/orchestrator.log        # 看进度
```

编排逻辑：等 `records100` 齐备 → 建缓存 → GPU0/1/2 并发训练 SNN/CNN/ResNet → 等三张卡收工 → 生成 `results/comparison.json` → 打印核心数字。

## 进度

- [x] 立项、选定上游工程
- [x] 环境就绪（`snntorch 1.0.0` / `torch 2.5.1+cu124` / 4×RTX 3090）
- [x] 冒烟测试通过：数据 → Δ调制脉冲编码 → SNN → BPTT → 评测 全链路，SNN 参数 1,893,768
- [x] 发现并修复 PTB-XL 残档（夸克搬运缺一半）
- [ ] PTB-XL 数据补齐 + 全量缓存
- [ ] PTB-XL SNN 复现，与 0.8663 对拍
- [ ] PTB-XL CNN / ResNet 基线复现，与 0.9017 对拍
- [ ] MIT-BIH 分支复现，与 0.8683 / 0.4153 对拍
- [ ] 能耗对比表复现（`07_energy_report.py`）
