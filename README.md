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

### 关于训练时长（对拍时的重要发现）

作者配置是 `epochs=200` + `wallclock_hours=8.0`，乍看"200 轮压在 8 小时内"，实则不然。查其提交的 `results/main/history.json`，每轮 `wallclock_seconds` **非单调**：

```
ep0=708s  ep1=1411s  ep2=2109s  ep49=12795s  ep99=10599s  ep149=41608s  ep199=1218s
```

`ep0→ep2` 每个差值约 700s（=单轮真实耗时），但 ep49/ep99/ep149/ep199 的值忽大忽小——这正是**每次进程重启后计时归零**的痕迹。结论：**作者是分多段续跑跑满 200 轮的**（单段到 8h 墙钟即优雅停止、下次从 `latest.pt` 恢复），200 轮总计约 39 小时 GPU 时间。

本机实测：SNN 单轮无争用 **566s**、与下载/其他训练争用时约 33 分钟，与作者同量级（甚至更快）。所以 100 轮需 16~20h，同样必须续跑，由 `experiments/08_resume_snn_until_done.sh` 负责。

**续跑的数值等价性**：`latest.pt` 里除 model/ema/opt 外还存了 **`sched` 状态**，恢复时 `sched.load_state_dict` 使余弦退火沿原 `T_max=--epochs` 连续推进，`epoch_start = ckpt.epoch + 1`，且墙钟是在**轮首**检查的（到点即干净退出）。因此只要 `--epochs` 与首次启动一致（本工程统一 100），多段续跑 ≙ 一次跑完。

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

# 7. SNN 续跑到 100 轮（作者亦为多段续跑），跑满后自动重跑 07 出最终对比表
nohup bash experiments/08_resume_snn_until_done.sh > logs/resume_snn.log 2>&1 &
```

> 第 4 步的 SNN 会被 `WALLCLOCK` 到点截断，此时第 6 步产出的 `comparison.json` 是**截断版**；
> 第 7 步跑满 100 轮后会重新生成 `results/comparison.json`，以那份为准。

**无人值守（推荐）**：第 1~6 步可交给编排脚本一次跑完，全程后台，Claude / SSH 断开都不影响：

```bash
nohup bash experiments/06_auto_pipeline.sh > logs/orchestrator.log 2>&1 &
tail -f logs/orchestrator.log        # 看进度
```

编排逻辑：等 `records100` 齐备 → 建缓存 → GPU0/1/2 并发训练 SNN/CNN/ResNet → 等三张卡收工 → 生成 `results/comparison.json` → 打印核心数字。

SNN 训练时间长于单次墙钟，故把**续跑守护**也一起挂上（它会先等 06 退出再接手，互不冲突）：

```bash
nohup bash experiments/08_resume_snn_until_done.sh > logs/resume_snn.log 2>&1 &
```

## 进度

- [x] 立项、选定上游工程
- [x] 环境就绪（`snntorch 1.0.0` / `torch 2.5.1+cu124` / 4×RTX 3090）
- [x] 冒烟测试通过：数据 → Δ调制脉冲编码 → SNN → BPTT → 评测 全链路，SNN 参数 1,893,768
- [x] 发现并修复 PTB-XL 残档（夸克搬运缺一半）
- [x] PTB-XL 数据补齐（records100 齐备 21799）+ 全量缓存（21388 条带诊断标签，745 MB）
- [x] PTB-XL CNN / ResNet 基线复现（100 轮）：CNN **0.9039** vs 作者 0.9023；ResNet **0.8967** vs 作者 0.9017
- [ ] PTB-XL SNN 复现，与 0.8663 对拍（训练中，需续跑至 100 轮）
- [ ] MIT-BIH 分支复现，与 0.8683 / 0.4153 对拍
- [ ] 能耗对比表复现（`07_energy_report.py`，待 SNN 跑满后由 08 重跑）
