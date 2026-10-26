# 重跑失效产物 · 执行清单

- 日期：2026-10-26
- 任务依据：`项目说明_ECG-SNN-LowPower.md` 第 5 节（重跑顺序）
- 启动脚本：`experiments/39_rerun_artifacts_1026.sh`
- 运行日志：`logs/rerun_1026_1026_1344.log`（2026-10-26T13:44:32Z → 14:45:23Z）
- 全部为**纯推理**，无任何训练；全程锁定同一份冻结权重

---

## 0. 一句话结论

5 个失效产物已用最终权重（`best.pt` md5 `d01e7aea…`，epoch 96）全部重建；`results/comparison.json` 中 SNN 的 `macro_auroc` = **0.859357**（落在交接文档要求的 0.8594~0.8595 区间），**不是** 0.4985 / 0.6705 的陈旧值。与 10/05 上一次重跑逐点对比，**30 个帕累托点的所有科学量差值为 0**，13 张 PNG **逐字节相同**，7 个 JSON 的差异**仅在时间戳与计时字段**。

但重跑过程中查出**三处比"产物失效"更严重的问题**：交接文档里引用的两组审计数字（16 号的 z 值、4.4 节的 SOP 走势）在最终权重下**不可复现**，以及 20 号脚本把一段逐层数字**硬编码**成了字面量。这三处直接决定论文怎么措辞，见第 5 节。

## 1. 环境与快照

| 项 | 值 |
|---|---|
| 容器 | `66711ebf8f4b`，IP `172.17.0.2`（`hostname -i` 现取） |
| GPU | 4× RTX 3090 24GB，本次仅用 `CUDA_VISIBLE_DEVICES=0` |
| conda 环境 | `shadocformer`（`/opt/miniconda3/envs/shadocformer/bin/python`） |
| 冻结快照 | `results/pareto/_snapshot_best.pt` |
| 权重 md5 | `d01e7aea2afd27977180f2b589542ffc`（`best.pt` 与快照**同一 md5**） |
| checkpoint epoch | 96 |
| 训练阈值 | `train_theta = 0.15`（所有点共用，θ 扫描是 zero-shot 灵敏度分析） |
| 测试集 | 官方第 10 折，n_test = 2158 |
| 标定集 | fold 9 验证集，2146 条，取 32 个 batch = 1024 样本 |

`best.pt` 的 mtime 为 2026-09-25 07:38，自那以后未被改动，因此本次重跑与 10/05 那次用的是**同一份权重**。

## 2. 执行记录（8 步时间线，UTC）

| 步 | 脚本 | 开始 | 关键输出 |
|---|---|---|---|
| 0 | 冻结快照 `cp best.pt → _snapshot_best.pt` | 13:44:32 | 两侧 md5 均为 `d01e7aea…` |
| 1 | `11_pareto_sweep.py`（30 点） | 13:44:33 | 30 点全部完成，14:31 前结束 |
| 2 | `16_calib_noise_floor.py` | 14:31:57 | K=32 低噪曲线 + 2000 次 bootstrap |
| 3 | `18_bn_shift_probe.py` | 14:36:42 | 逐层 BN 输出平移 |
| 4 | `20_audit_figures.py` | 14:41:12 | 6 图 + `audit.html` |
| 5 | `09_make_report.py` | 14:41:17 | 7 图 + `report.html` |
| 6 | `13_build_offline_demo.py --all` | 14:41:21 | 单文件离线包（从容器内 8000 端口 shim 现取） |
| 7 | `14_leak_audit.py` | 14:41:49 | A1/A2/B/C/D 五组 |
| 8 | 收尾 | 14:45:23 | `DONE` |

与交接文档第 5 节给定的 5 步相比，本脚本**多跑了 16、18 两步**（它们在 `logs/rerun_artifacts.sh` 里本来就有）。理由：20 号的图 B、图 F 直接读这两个产物，不重跑就会把新旧权重混在一套图里。步序未变。

唯一的口径改动：11 号显式传 `--ckpt results/pareto/_snapshot_best.pt`，不再隐式读 `results/repro_snn/best.pt`。两者 md5 相同，结果应逐位一致——实测确实一致（第 4 节）。

**补记**：第 4 步在 14:51 又单独重跑了一次，原因是第 5.3 节的脚本修正（修完必须重出图与 html）。`results/audit` 的最终生成时刻是 **14:51:39**，不是 14:41。

## 3. 产物清单

| 产物 | 路径 | 最终生成 | 权重 md5 记录 |
|---|---|---|---|
| θ×T_dense 帕累托 30 点 | `results/pareto/theta*_T*.json` | 10-26 13:44–14:31 | `d01e7aea…`（30/30 点一致） |
| 帕累托汇总 | `results/pareto/pareto_summary.json` | 10-26 14:31 | `d01e7aea…` |
| 噪声地板 | `results/pareto/calib_noise_floor.json` | 10-26 14:35 | `d01e7aea…` |
| BN 平移 | `results/pareto/bn_shift_probe.json` | 10-26 14:41 | `d01e7aea…` |
| 中文审计 6 图 + html | `results/audit/` | 10-26 14:51 | `d01e7aea…` |
| 报告 7 图 + html | `results/report/` | 10-26 14:41 | 经 pareto 间接绑定 |
| 答辩离线包 | `results/offline/` | 10-26 14:41 | 经 shim 间接绑定 |
| 泄漏自洽检查 | `results/leak_audit.json` | 10-26 14:45 | `d01e7aea…` |
| 能耗对比（**未重跑**，仅核对） | `results/comparison.json` | 09-25 08:07 | 不记录 md5（07 的设计缺陷） |

`results/comparison.json` 不在本次重跑链里：它是 `07_energy_report.py` 的产物，交接文档已确认 9/25 用最终权重重跑过。本次只做核对（第 4.4 节）。

## 4. 复现性核对

### 4.1 主判据：comparison.json 的 SNN macro-AUROC

| run | macro_auroc | macro_auprc | params |
|---|---|---|---|
| repro_cnn | 0.903927 | 0.770372 | 1,741,381 |
| repro_resnet | 0.896683 | 0.755296 | 1,879,685 |
| **repro_snn** | **0.859357** | 0.695244 | 1,893,768 |

落在 0.8594~0.8595 区间内 ✅（0.859357 四舍五入为 0.8594）。未出现 0.4985 或 0.6705。

能耗：稠密 1.205033 mJ / 神经形态 0.005231 mJ = **230.35×**；精度代价 0.903927 − 0.859357 = **4.457 pp**。与交接文档的「约 230×、约 4.5 pp」一致。

> 口径提醒：`test_metrics.json` 记的是 0.8594542，比这里高 9.7e-5。原因是 `14_leak_audit.py` 实测的 batch 敏感性：bs=64 得 0.859357，bs=96 得 0.859454（差 9.7e-5）。**论文正文只能出现一个数**，建议用 `comparison.json` 的 0.8594（主流程口径），并在方法节说明评测 batch=64。

### 4.2 30 个帕累托点：与 10/05 逐点对比

10/05 那次重跑的产物已备份到 `results/_backup_pre_rerun_1026/`，逐点配对比较（n=30）：

| 量 | 最大绝对差 | 有差异的点数 |
|---|---|---|
| macro_auroc | 0.000e+00 | 0 |
| macro_auprc | 0.000e+00 | 0 |
| total_sops | 0.000e+00 | 0 |
| spike_rate_mean | 0.000e+00 | 0 |
| input_spike_rate | 0.000e+00 | 0 |
| energy_pj_neuromorphic | 0.000e+00 | 0 |
| total_macs | 0.000e+00 | 0 |
| params | 0.000e+00 | 0 |

30/30 点的 `checkpoint_md5` 均为 `d01e7aea…`、`checkpoint_epoch` 均为 96。

### 4.3 图与 JSON

13 张 PNG **逐字节相同**（报告 7 张 + 审计 6 张）⇒ 绘图链路完全确定性。

JSON/HTML 的差异分类（排除时间戳、墙钟、路径后）：

| 文件 | 总差异叶 | 其中科学量差异 |
|---|---|---|
| `report/report_data.json` | 91 | **0** |
| `audit/audit_data.json` | 7（修脚本前） | **0** |
| `offline/ptbxl_bundle.json` | 1（仅 `generated_at`） | **0** |
| `leak_audit.json` | 5 | **0** |
| `pareto/calib_noise_floor.json` | 6 | **0** |
| `pareto/pareto_summary.json` | 92 | **0** |
| `pareto/bn_shift_probe.json` | 0（逐字节相同） | **0** |

差异项只有 `checkpoint_path`（由 `best.pt` 改为快照路径）、`latency_ms`、`test_inference_seconds`、`generated_at`。离线包的推理结果**逐位相同**。

### 4.4 泄漏自洽检查（14 号）

| tag | batch | 重置 LIF | macro_auroc |
|---|---|---|---|
| A1 | 64 | 否 | 0.8593575 |
| A2 | 64 | 否（连跑第二次） | 0.8593575 |
| B | 64 | 每 batch 前重置 | 0.8593575 |
| C | 96 | 否 | 0.8594542 |
| D | 96 | 是 | 0.8594542 |

|A1−A2| = 0、|A1−B| = 0 ⇒ **跨 batch 泄漏不存在**，交接文档的证伪结论成立。|B−D| = 9.7e-5 是纯 batch 划分效应。

## 5. ⚠️ 三处与交接文档冲突的地方（写论文前必须先拍板）

这三处不是"产物没重跑"，是**重跑之后暴露出来的、文档自己引用的数字对不上**。

### 5.1 交接文档 §7 的 16 号结论（θ=0.2 触底、z=8.84 / 1.51）不可复现

文档 §7 与任务书都写：「θ=0.2 触底的频率 0.99，相对右邻 z=8.84（铁证），相对左邻仅 z=1.51」。

最终权重下的实测：

```
θ       input      隐藏合计        SOP(K)    SOP(8-batch)        ±std
0.05      0.4349    0.9982      48732662        48741155      904106
0.075     0.4022    0.9655      47408176        47417333     1056465
0.1       0.3728    0.9755      48216433        48230625     1283745
0.15      0.3220    1.0200      50898803        50930165     1608480
0.2       0.2808    1.0517      52471112        52500648     1631648
0.3       0.2200    1.0376      50790822        50804950     1566750

K 曲线最小点 θ=0.075（内部）
8-batch 内部最小点频率 = 1.000；argmin 落点 0.075:0.952、0.1:0.048
最小点相对左右邻 z：左 1.25、右 0.77
```

**证据链**：文档那组数字逐字出现在 `logs/calib_noise_floor.log`（2026-09-24 22:18），该日志第 2 行记录 `md5=738d5ed18ce58af86185db557736d4a6` —— **不是**最终权重 `d01e7aea…`。也就是说，**§7 引用的是一份中间权重下的审计结果**，正是交接文档第 5 节要求重跑的那一类失效产物，只是当时漏掉了 16 号。

**影响**：任务书 §1 第 3 条要求写「θ=0.2 比 0.15 更低，不显著（z=1.51）」。**这句话不能写**——最终权重下 θ=0.2 的 SOP 比 0.15 **高** 1.57 M（52.47 vs 50.90），方向都反了。

### 5.2 交接文档 §4.4 的 SOP 走势与实测不符

文档写：「θ 从 0.05 升到 0.2 时 SOP 下降，但 θ=0.3 反弹」。

实测（T_dense=8，全测试集，单位 10⁶ SOP）：

```
θ     0.05   0.075   0.1    0.15   0.2    0.3
SOP   49.01  47.84   48.93  52.31  53.71  51.01
```

走势是**降 → 升 → 降**：最低在 θ=0.075（内部），最高在 θ=0.2。K=32 标定曲线（48.73 / 47.41 / 48.22 / 50.90 / 52.47 / 50.79）给出完全相同的形状。

**影响**：非单调性**结论仍然成立且更强**（两个最低点都在内部，8-batch 估计量 100% 都能找到内部最小点），但文档描述的**形状错了**。

### 5.3 20 号脚本把逐层数字硬编码了（已修）

`20_audit_figures.py` 原先在 `findings` 循环里写死：

> 逐层发放率在 θ=0.3 相对 0.15 的变化：input −0.1020、stem **+0.0220**、block2 +0.0160、block3 +0.0138、block0 −0.0044

实测（fold-9 标定集）：

> input −0.1020、stem **−0.0213**、block0 −0.0180、block1 −0.0043、block2 +0.0175、block3 +0.0107、block4 +0.0029、block5 −0.0006、fc +0.0306

**stem 的符号是反的**。这组字面量同样来自 9/24 的中间权重。

已改为从 `calib_noise_floor.json` 的 `per_layer_k` **现算**（新增 `layer_delta_text()`），并重跑 20 号。6 张 PNG 与修正前逐字节相同 ⇒ 图本身没问题，错的只是那段结论文字。原实现备份在 git 上一版。

## 6. 那么「θ 非单调」这条贡献现在该怎么写

三处修正后，能站住的表述是下面这几句（每句都有本次数据背书）：

1. **总 SOP 对 θ 非单调，最小值在内部点**。T_dense=8 时最低点 θ=0.075（49.01→47.84→48.93→52.31→53.71→51.01 ×10⁶）。
2. **「内部最小」是稳健的，具体落点不精确**。K=32 低噪曲线 + 2000 次 bootstrap：内部最小点出现频率 **1.000**（0.075 占 0.952、0.1 占 0.048）；但最小点相对左右邻的 z 只有 **1.25 和 0.77** ⇒ 只能说"最低点在 θ≈0.075–0.1 一带"，**不能说它精确等于某个 θ**。
3. **机制是逐层响应符号随深度翻转，但只有两端站得住**。输入层 −0.1020（两套数据完全一致，且与权重无关），stem −0.021、block0 −0.018；block2 +0.0175、block3 +0.0107、分类头 fc +0.031。block4/block5 的变化幅度都 <0.005，且**测试集与标定集符号不一致**（block4 测试集 −0.0047、标定集 +0.0029）⇒ **这两层不予主张**。
4. **这条是架构相关的**。MIT-BIH 上 `NOT_REPRODUCED`：总 SOP 单调，最小值在端点 θ=0.3。

论文里**不能**出现的表述：
- ❌「θ=0.2 比 0.15 更低」（方向反了）
- ❌「z=8.84，铁证」（中间权重的数）
- ❌「stem 在 θ=0.3 时增发 +0.0220」（符号反了）
- ❌「θ 从 0.05 升到 0.2 时 SOP 下降」

## 7. 本次未做 / 遗留

| 项 | 状态 | 说明 |
|---|---|---|
| `12_localize_web.py` 的 `40.6M effective SOPs` / `~4.1 µJ` | **未改** | 现值应为 52.3 M / 5.23 µJ（基线点）。不在第 5 节重跑链里，改它会牵动 `results/web_cn/` 与离线包，建议单独作为一次改动处理 |
| `audit_data.json` 的 `meta.epoch` = `"—"` | 未修 | 16 号产物里没写 epoch 字段，20 号只能写 `—`。属于显示缺陷，不影响数字 |
| `results/p1_compare.json`、`energy_cudagraph.json` | 未归档 | 仍散在 `results/`，未进编号目录 |
| 备份 | 保留 | `results/_backup_pre_rerun_1026/`（不含 `.pt`），可随时回滚或做对照 |

## 8. 复现命令

```bash
# 一键（约 61 分钟，纯推理）
cd /mnt/ECG-SNN-LowPower
nohup setsid bash experiments/39_rerun_artifacts_1026.sh > logs/rerun_$(date +%m%d_%H%M).log 2>&1 < /dev/null &

# 分步
cp results/repro_snn/best.pt results/pareto/_snapshot_best.pt
python experiments/11_pareto_sweep.py --run repro_snn --ckpt results/pareto/_snapshot_best.pt \
       --out results/pareto --baseline results/comparison.json --force
python experiments/16_calib_noise_floor.py
python experiments/18_bn_shift_probe.py
python experiments/20_audit_figures.py --out results/audit
python experiments/09_make_report.py --runs-root results --out results/report
python experiments/13_build_offline_demo.py --all
python experiments/14_leak_audit.py --out results/leak_audit.json --refresh-snapshot
```

第 6 步要求容器内 `127.0.0.1:8000` 上有一个 `10_serve_demo.py` shim（本次用的 PID 14076，自 9/24 常驻）。没有的话先起它，否则 `13 --all` 会失败。
