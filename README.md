# ECG-SNN-LowPower

**基于脉冲神经网络的低功耗心电异常检测系统** —— 本科毕业论文复现工程。

## 复现对象

上游项目：[`hanshaunlee/neurocardio`](https://github.com/hanshaunlee/neurocardio)（MIT）

选择理由：它是目前唯一同时满足以下条件、且真正能跑通的公开工程——

1. **覆盖两个目标数据集**：主线是 PTB-XL 12 导联五超类诊断（文献 CNN/Transformer 区间 0.92–0.94 AUROC 的**竞争性**基准），`archive/cardiospike_mitbih/` 是 MIT-BIH 心律失常（DS1/DS2 **病人不交叉**划分 + AAMI 五类）。
2. **低功耗叙事是量出来的，不是吹的**：`neurocardio/compute.py` 用**实测**脉冲发放率算突触操作数（SOP），对比参数匹配的稠密基线，给出每层能量账。
3. **完整可复现**：仓库自带数据下载/预处理、训练、评测、基线对比，以及作者提交的权重与指标 JSON（可直接对拍）。
4. 附带 web 演示（`web/`、`modal_app/`），对应"系统怎么呈现"这一环节。

## 参考指标与实测对拍

**作者提交值**（复现目标）：

| 模型 | PTB-XL macro-AUROC | 参数 | 说明 |
|---|---|---|---|
| **SNN（主线）** | **0.8663** | 1.89 M | 估算能耗约为稠密基线的 1/287 |
| ResNet1D 基线 | 0.9017 | — | 稠密基线 |
| CNN 基线 | ~0.90 | — | 稠密基线 |

**本机实测**（`results/comparison.json`，由 `07_energy_report.py` 产出）：

| 模型 | PTB-XL macro-AUROC | 参数 | batch-1 延迟 | 估算能耗/次 | 对拍 |
|---|---|---|---|---|---|
| SNN（主线） | 训练中（100 轮未完） | 1,893,768 | **1343 ms** | — | 待跑满后重跑 07 |
| CNN1D 基线 | **0.9039** | 1,741,381 | **1.90 ms** | 1.0901 mJ | 作者 0.9023 ✅ |
| ResNet1D 基线 | **0.8967** | 1,879,685 | 2.03 ms | 0.1868 mJ | 作者 0.9017 ✅ |

> ⚠️ **SNN 延迟 1343 ms vs CNN 1.90 ms（约 700×）是仿真环境的产物，不是架构劣势。**
> 瓶颈是 Python 逐时间步循环模拟 LIF 膜电位，实测**单核跑满、GPU 空等**（`utime ≈ elapsed`，
> 99 线程只有一个在忙）。在真正的神经形态芯片上 SNN 推理是亚毫秒级。论文若不正视这条，
> 会被误读成"SNN 更慢"，与低功耗叙事自相矛盾。
>
> ⚠️ `comparison.json` 里的 SNN 行是**截断版**（训练中 best.pt 被反复刷新，07 读到的是早期权重），
> 故其 AUROC 不代表最终水平。跑满 100 轮后由 `08` 自动重跑 07，以那份为准。

**MIT-BIH 分支**（`archive/cardiospike_mitbih/`，Conv-LIF SNN ≈272k 参数，25 epoch）：

| 指标 | 作者提交 | 本机实测 | 差值 |
|---|---|---|---|
| accuracy | 0.868329 | **0.870221** | +0.19 pp |
| macro-F1 | 0.415321 | **0.426845** | +1.15 pp |
| 能耗比（ANN/SNN 操作数） | 31.83× | **27.29×** | 我们的模型发放率略高 |

> MIT-BIH 的能耗比**低于**作者（27.29× vs 31.83×），因为两者不是同一份权重——
> 作者的 `models/cardiospike_best.pt` **从未提交**（`git ls-files` 显示 `models/` 下只跟踪
> `rr_stats.npz`），我们的是自己训 25 轮的产物。逐层能耗账见
> `results/mitbih/web_data/sparsity.json`（`total_snn_ops=129922` / `total_ann_ops=3545536`）。

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

## 可视化与现场演示

两半都是**容器内起服务 + 本机端口转发**（业务后端一律在容器内，遵守 CLAUDE.md 禁本地开发命令）。

### 1. 离线静态报告（零依赖兜底，答辩最稳）

```bash
python3 experiments/09_make_report.py --runs-root results --out results/report
```

产出 7 张 PNG + **自包含 `report.html`**（图片 base64 内嵌、无任何外链）。回传本机后
用浏览器 `file://` 打开即可完整显示，**不依赖网络、不依赖容器存活**：

```bash
cd C:/Users/dj/remote-docker-project
MSYS_NO_PATHCONV=1 python remote_fetch.py get \
    /mnt/ECG-SNN-LowPower/results/report/report.html \
    "C:/迅雷下载/实验/ECG-SNN-LowPower/results/report.html"
```

### 2. 中文化（`experiments/12_localize_web.py`）

上游前端是英文。译文放在**受版本控制的脚本内的对照表**里（不是一次性手工编辑），
输出到被 gitignore 的 `results/web_cn/` 与 `results/mitbih/web_cn/`：

```bash
python3 experiments/12_localize_web.py                 # 两个面板
python3 experiments/12_localize_web.py --check-only    # 只校验上游原文有无漂移
```

匹配用**空白弹性正则**（词间允许任意空白），故上游 HTML 的换行/缩进变化不会导致失配；
一旦上游改了原文，`--check-only` 报「未命中」并返回非零。当前 234 条译文全命中。

字体走 CSS 变量补丁（`:root` 的 `--font`/`--mono` 补 CJK 回退链），不逐个选择器覆盖。
MIT-BIH 的 5 类中文名用 JS 侧 `CLASS_CN` 覆盖映射，**不改** `inference.json`
（它由 `evaluate.py` 生成，保持数据可复现）。

### 3. 答辩离线包（`experiments/13_build_offline_demo.py`）—— **现场主推**

把面板 HTML/CSS/JS 与预置样本的推理结果全部内联成**单文件 HTML**，拷回本机后
`file://` 双击即开：零网络、零服务、点样本**瞬时**切换，会场断网也不影响。

```bash
# 容器内（需 10_serve_demo.py 已在 8000 运行）
PY=/opt/miniconda3/envs/shadocformer/bin/python
nohup $PY experiments/13_build_offline_demo.py --all > logs/offline_build.log 2>&1 &
# 之后只重打包、不重收数据：
$PY experiments/13_build_offline_demo.py --build
```

产出 `results/offline/ptbxl_demo.html`（2.7 MB，预置 17 个样本）与
`mitbih_demo.html`（0.54 MB）。

**原理**：在 `app.js` **之前**注入一个 `window.fetch` 拦截器，按 URL 从内联的
`window.__BUNDLE__` 取数并返回 `new Response(...)`；前端 `jget()`/`r.json()`
察觉不到差别，**上游 `app.js` 零改动**。PTB-XL 需预跑推理（`--collect` 连 shim 收
`/examples` 里全部样本），MIT-BIH 本就纯静态、扫 `app.js` 里的字面量路径内联即可。

`ecg` 降到 4 位小数（屏幕一像素约 2e-3，1e-4 分辨率不可见），
`probs`/`spike_rates`/`compare_probs`/`macs`/`sops` 等**数值口径一个字节不动**；
并剔除 Google Fonts 三条外链（避免现场「有网但慢」时白屏等待）。
生成后自检：`<link|script|img>` 的外链必须为 0，`fetch` 去向逐个核对是否已被覆盖。

回传本机：

```bash
cd C:/Users/dj/remote-docker-project
MSYS_NO_PATHCONV=1 python remote_fetch.py get \
    /mnt/ECG-SNN-LowPower/results/offline/ptbxl_demo.html \
    "C:/迅雷下载/实验/ECG-SNN-LowPower/results/offline/ptbxl_demo.html"
```

### 4. 联网实时 demo（备选）

```bash
# 容器内：PTB-XL 主线（上游 web/ 前端 + stdlib 后端替身）
CUDA_VISIBLE_DEVICES=3 nohup python experiments/10_serve_demo.py --port 8000 \
    --web results/web_cn > logs/web.log 2>&1 &
# 容器内：MIT-BIH（纯静态，中文副本 results/mitbih/web_cn/）
cd results/mitbih/web_cn && nohup python -m http.server 8001 --bind 0.0.0.0 &

# 本机：两条转发（通道工具，属允许在本机运行的一类）
python "C:/Users/dj/.claude/scripts/remote-forward.py"
python "C:/Users/dj/.claude/scripts/remote-forward.py" --remote-port 8001 --local-port 8001
```

浏览器打开 **`http://127.0.0.1:8000`**（PTB-XL：三模型对比 + 4 张脉冲可视化 + 上传推理）
与 **`http://127.0.0.1:8001`**（MIT-BIH：逐拍诊断 + 混淆矩阵 + 逐层能耗账）。

⚠️ **实测延迟（2026-09-24，同一样本 `ecg_id=9`，响应 331411 B）**：

| 位置 | 耗时 | 吞吐 |
|---|---|---|
| 容器内自连（urllib） | **1.76 s** | 188 KB/s |
| 经隧道（本机 curl） | **47.6 s** | 6.9 KB/s |

⇒ 推理本身只要 1.76 秒，**慢出来的 27 倍全在四跳链路搬字节**，而且**每次点击都是这个数**
——早先 README 写的「首次约 50 s、之后约 3 s」**是错的，已实测推翻**（实测三次：
59.9 s / 45.7 s / 47.6 s）。响应体积里 `ecg`(12×1000 浮点) 占 76.9%、
`input_spikes` 占 22.3%，两项即 99.2%；gzip 只压得动 4.6:1，救不回来。

**所以现场演示走第 3 节的离线包，不要依赖这条链路。** 本节仅作联网时的备选。

**为什么必须有后端替身**：上游 `scripts/serve.sh`（`cd web && python3 -m http.server`）
**是失效脚本**——根 `web/index.html` 用**绝对路径** `/static/{style.css,app.js}`，
靠 Modal 的 `api.mount("/static", StaticFiles(directory="/web"))` 平铺映射才成立，
纯静态托管会全部 404；且前端要调 7 个接口。`10_serve_demo.py` 用 stdlib `http.server`
复刻了 `modal_app/app.py::web()` 的全部路由（含 `POST /infer_upload`）。
MIT-BIH 面板才是真静态（相对路径 + 已提交 JSON），零代码即可跑。

**MIT-BIH 面板用我们自己的数据**：上游 `archive/.../web/data/` 是作者的产物，**未被我们覆盖**；
我们 25 轮的产物在 `results/mitbih/web_data/`。故建工作副本 `results/mitbih/web_live/`
= 上游的 html/js/css + 我们的 4 个 JSON。这是「跑 MIT-BIH 前先复制到工作副本」约定的落地。

## 已知上游缺陷（论文需如实说明）

**LIF 膜电位跨 batch 残留**：`neurocardio/model.py` 的 `snn.Leaky.forward` 把传入的 `mem`
存进 `self.mem`，且**只在 shape 不匹配时**才清零；`init_leaky()` 返回 `self.mem.clone()`
⇒ 每个 batch 会继承上一个 batch 的终态膜电位。同一份权重、同一 θ/T_dense 下实测：

| batch_size | macro-AUROC |
|---|---|
| 64 | **0.498530** |
| 96 | **0.487912** |

重跑逐位相同（确定性，非随机扰动）。⇒ 本工程评测一律固定 `batch_size=64`（`07`/`11` 已对齐），
跨 batch_size 的数字不可比；演示后端因此把推理串行化（`10_serve_demo.py` 用同一把锁）。

## 精度-能耗帕累托扫描（本工程自己的贡献）

`experiments/11_pareto_sweep.py` 在 θ（编码阈值，6 点）× `T_dense`（头积分步数，5 点）= 30 个
配置点上扫描，每点重跑 fold-10 全量推理 + 在 fold-9 上**重新标定**发放率（不能复用）：

```bash
CUDA_VISIBLE_DEVICES=3 nohup python experiments/11_pareto_sweep.py \
    --ckpt results/pareto/_snapshot_best.pt --out results/pareto > logs/pareto_full.log 2>&1 &
```

⚠️ **学术诚信红线**：上游 `train.py:57` 的 `theta` 字段**全文只出现一次**，训练时
`main()` 是 `model(x)`（不传 θ）⇒ **所有 checkpoint 都是 θ=0.15 训练的**。因此 θ 扫描是
**零训练编码器迁移（zero-shot 灵敏度分析）**——精度下降同时含「编码失配」与「信息损失」
两种效应，**不能**解释为"θ 的最优取值"，更**不能**写成"我们对每个 θ 重训了模型"。
要分离二者需对每个 θ 重训 100 轮，成本不现实，明确列为未来工作。

⚠️ **扫描必须绑定冻结的权重快照**：训练持续刷新 `results/repro_snn/best.pt`，30 个点若各自
读一次就会横跨多份权重、不可比。故先冻结 `results/pareto/_snapshot_best.pt`，且每个点都记录
`checkpoint_md5`/`checkpoint_epoch`/`checkpoint_path` 可溯源。

结果自动并入静态报告（`09` 读 `results/pareto/*.json`，无需额外参数）。

### 扫描结果（30 点，绑定快照 `_snapshot_best.pt` = epoch 11）

按「最小化能耗 / 最大化 macro-AUROC」取的帕累托前沿（已剔除 `T_dense=2` 的退化点）：

| 能耗/次 | macro-AUROC | θ | T_dense |
|---|---|---|---|
| 3.302 µJ | 0.5724 | 0.2 | 4 |
| 3.302 µJ | 0.6507 | 0.2 | 8 |
| 3.302 µJ | 0.6679 | 0.2 | 16 |
| 3.302 µJ | 0.6733 | 0.2 | 32 |
| 3.310 µJ | 0.6863 | 0.15 | 16 |
| 3.310 µJ | 0.6893 | 0.15 | 32 |
| 3.379 µJ | 0.6969 | 0.1 | 16 |
| 3.379 µJ | **0.7021** | **0.1** | **32** |

**头条结论**：`θ=0.1 / T_dense=32` 相对训练基线 `θ=0.15 / T_dense=8`（0.6705 @ 3.310 µJ）
**macro-AUROC +0.0316（相对 +4.72%），能耗只 +2.10%**——近乎免费的精度增益，
且**两轴都是纯推理期参数，部署后可直接调，无需重训**。

三个需要写进论文的机制性发现：

1. **`T_dense` 几乎是免费的精度旋钮**。`fc1` 的输入发放率实测约 0.0005，故
   `macs_fc1 × T_dense × rate` 可忽略（T=2 → 49 SOP，T=32 → 786 SOP），
   而同一 θ 下 AUROC 从 0.5724 升到 0.6733。能耗代价 <0.3%，精度收益 +0.10。
2. **`T_dense=2` 是退化输出，不是"高性价比点"**：该配置下五类 AUROC **恒为 0.5**
   （模型塌缩成常数预测），而 `macro_auprc` 仍有 0.2588——即"输出无判别力"，
   与"精度低"是两回事。它同时是全网格能耗最低的点，混进散点图极易被误读，
   故报告图 1(b) 用灰色 × 单独标记并在图例里点名。
3. **总 SOP 对 θ 非单调**：θ↑ 使**输入层**发放率单调降（0.4314 → 0.2130），但总 SOP 不单调——
   θ=0.3 处 `stem_conv` 发放率 0.3150→0.2130（省 1.18 M SOP）被隐藏层增发抵消
   （`block3.conv2` 0.1138→0.1290、`block0.conv1` 0.1743→0.1971、`block2.conv2` 0.0751→0.0934）。
   净最小值在 θ=0.2，之后回升。⇒ **"抬高编码阈值省电"这一直觉只在输入层成立**，
   总能耗必须看全层账。日志里 `SOP 单调降=False` 即此项，是预期结果而非 bug。

**自洽检查的读法**：`pareto_summary.json` 的 `baseline_consistency` 有四项，
其中 `input_spike_rate*` 带 `*` 表示**与权重无关**（编码层只依赖输入与 θ），是本检查里
唯一能跨权重对拍的项，实测与 07 的值**逐位相同**（0.31500813364982605）。另外三项
（`macro_auroc` / `total_sops` / `spike_rate_mean`）都含隐藏层实测发放率，只在同一份权重下可比；
07 的 `comparison.json` 生成于 `best.pt` 被训练刷新之前，且 07 不记录 `checkpoint_md5`，
故当前显示 FAIL 属**权重漂移而非实现错误**。SNN 跑满 100 轮后重跑 07 + 11，三项才会同时 OK。

## 进度

- [x] 立项、选定上游工程
- [x] 环境就绪（`snntorch 1.0.0` / `torch 2.5.1+cu124` / 4×RTX 3090）
- [x] 冒烟测试通过：数据 → Δ调制脉冲编码 → SNN → BPTT → 评测 全链路，SNN 参数 1,893,768
- [x] 发现并修复 PTB-XL 残档（夸克搬运缺一半）
- [x] PTB-XL 数据补齐（records100 齐备 21799）+ 全量缓存（21388 条带诊断标签，745 MB）
- [x] PTB-XL CNN / ResNet 基线复现（100 轮）：CNN **0.9039** vs 作者 0.9023；ResNet **0.8967** vs 作者 0.9017
- [x] MIT-BIH 分支复现（25 轮）：accuracy **0.870221** vs 作者 0.868329；macro-F1 **0.426845** vs 0.415321
- [x] 上游只读底本复原；我们自己的 MIT-BIH 产物拷到 `results/mitbih/`（含 best.pt 与 25 轮 history）
- [x] 静态报告 `09_make_report.py`：7 张 PNG + 自包含 `report.html`（含逐层能耗拆解）
- [x] 现场 demo：`10_serve_demo.py`（PTB-XL 面板）+ MIT-BIH 静态面板 + 本机端口转发工具
- [x] 精度-能耗帕累托扫描 `11_pareto_sweep.py`（θ × T_dense = 30 点）：最优 `θ=0.1/T_dense=32`
      得 **0.7021 AUROC @ 3.379 µJ**，相对基线 **+4.72% AUROC 换 +2.10% 能耗**；8 点前沿已并入报告图 1(b)
- [ ] PTB-XL SNN 复现，与 0.8663 对拍（训练中，需续跑至 100 轮）
- [ ] 能耗对比表复现（`07_energy_report.py`，待 SNN 跑满后由 08 重跑，届时 SNN 行才是最终值）
