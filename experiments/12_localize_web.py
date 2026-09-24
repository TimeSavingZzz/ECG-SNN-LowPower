#!/usr/bin/env python3
"""把两套演示前端中文化 —— 译文表驱动，产物落 ``results/``（gitignore）。

**为什么是脚本而不是直接改文件**：产物目录 ``results/`` 被 gitignore，直接改动
无法进版本库、也无法复现。译文表放在本脚本里，重跑即可从上游原文重新生成
中文版；上游若改文案，脚本会**报出未命中的条目**而不是静默漏译。

**匹配用「空白归一化」而不是精确串**：上游 HTML 的段落里英文是折行缩进的，
精确串匹配会被换行击穿。故把英文按 token 切分后用 ``\\s+`` 连接成正则，
既能跨折行匹配，也不受缩进影响（见 :func:`_pat`）。

**两套前端的形态差异**（决定产物落点与启动方式）：
- PTB-XL ``web/index.html`` 用**绝对路径** ``/static/{style.css,app.js}``，
  必须由 ``10_serve_demo.py`` 的 ``_static()`` 平铺映射 ── 故用 ``--web`` 指到产物目录。
- MIT-BIH ``web/index.html`` 用**相对路径**（``style.css`` / ``data/*.json``），
  纯静态可跑 ── 故数据目录用软链带过去，仍由 ``http.server`` 托管。

用法::

    python experiments/12_localize_web.py              # 生成两套中文前端
    python experiments/12_localize_web.py --check-only # 只校验，不落盘
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

ROOT = Path(os.environ.get("ROOT", "/mnt/ECG-SNN-LowPower"))
NC = ROOT / "third_party" / "neurocardio"

# ══════════════════════════════════════════════════════════════════════
# 译文表
#
# 每条是 (英文, 中文)。英文按**单空格**书写即可，匹配时会自动容忍折行缩进。
# 刻意保留英文不译的（写进 ALLOW 白名单，见文件末尾）：模型名、数据集名、
# 指标缩写（AUROC/MAC/SOP 等）、AAMI 类别代号（NORM/MI/STTC/CD/HYP、N/S/V/F/Q）、
# 硬件与文献名（Loihi/SpiNNaker/Horowitz/ISSCC）、颜色值、CSS 类名。
# ══════════════════════════════════════════════════════════════════════

PTBXL_HTML = [
    ('<html lang="en">', '<html lang="zh-CN">'),
    ('<title>NeuroCardio — Can a spiking neural network diagnose heart disease?</title>',
     '<title>NeuroCardio — 脉冲神经网络能诊断心脏病吗？</title>'),

    # ── 顶栏 / 主视觉 ──
    ('<div class="badge muted">loading…</div>', '<div class="badge muted">加载中…</div>'),
    ('A neuromorphic computing experiment · PTB-XL · 21,837 ECGs',
     '类脑计算实验 · PTB-XL · 21,837 份心电图'),
    ('Can a computer diagnose<br>heart disease using <span class="grad">less power<br>than a blinking LED?</span>',
     '计算机能否用<span class="grad">比闪烁的 LED<br>更低的功耗</span><br>诊断心脏病？'),
    ("""Modern AI can read an electrocardiogram as well as a cardiologist —
      but it requires a data center to do it. This project asks whether a
      fundamentally different kind of neural network, one that works more
      like your actual neurons, can match that accuracy
      while running on a chip the size of a postage stamp.""",
     '现代 AI 已能像心内科医生一样读懂心电图——但它需要一座数据中心。'
     '本项目要问的是：一种根本不同的神经网络，一种更像你真实神经元的网络，'
     '能否在一块邮票大小的芯片上达到同样的精度。'),
    ('less energy per diagnosis<br>for a 3.6 pp accuracy cost',
     '单次诊断能耗更低<br>代价是 3.6 个百分点的精度'),
    ('<div class="scroll-cue">scroll<span></span></div>',
     '<div class="scroll-cue">向下滚动<span></span></div>'),

    # ── 01 问题 ──
    ('</span> · The problem</div>', '</span> · 问题</div>'),
    ('<h2>The problem with always-on AI</h2>', '<h2>常开 AI 的困境</h2>'),
    ("""Cardiovascular disease kills roughly <strong>17 million people every year</strong> — more than
      cancer, more than respiratory disease, more than anything else. A significant fraction
      of those deaths are from conditions that are both detectable and treatable, but only
      if caught in time.""",
     '心血管疾病每年夺去约 <strong>1700 万人的生命</strong>——超过癌症、超过呼吸系统疾病、超过其他任何死因。'
     '其中相当一部分死亡源于既可检出、又可治疗的病症，前提是及时发现。'),
    ("""The catch is that the most dangerous arrhythmias are intermittent. A patient can walk into
      a clinic feeling fine, get a clean ECG, and then suffer a fatal event three days later.
      The gold standard for catching these is <strong>continuous ambulatory monitoring</strong>:
      a small device the patient wears for 24 to 72 hours — or longer — that records and analyzes
      the heart non-stop.""",
     '问题在于，最危险的心律失常往往是间歇性的。患者可以毫无不适地走进诊室、做出一份干净的心电图，'
     '三天后却发生致命事件。捕获这类事件的金标准是<strong>连续动态监测</strong>：'
     '一台患者佩戴 24 至 72 小时（甚至更久）、不间断记录并分析心脏活动的小型设备。'),
    ("""Today's AI-powered ECG analysis works beautifully in the hospital. It doesn't work on
      a wearable. The problem is energy. A deep convolutional network running on a battery-powered
      device would drain it in hours. Wearable cardiac monitors instead use simple threshold
      detectors — they're efficient, but they miss the subtle patterns that modern deep learning
      catches.""",
     '今天由 AI 驱动的心电分析在医院里表现优异，但放到可穿戴设备上就不行，问题出在能耗。'
     '一个深度卷积网络跑在电池供电的设备上，几小时就会耗尽电量。'
     '可穿戴心脏监测器因此只能用简单的阈值检测器——它们省电，却会漏掉现代深度学习能捕获的细微模式。'),
    ("""There's a third option. It's been promising for years but rarely proven on real clinical data.""",
     '还有第三种选择。它被寄予厚望多年，却很少在真实临床数据上得到验证。'),

    # ── 02 思路 ──
    ('</span> · The insight</div>', '</span> · 思路</div>'),
    ("<h2>Neurons don't run all the time</h2>", '<h2>神经元并非时刻运转</h2>'),
    ("""Your brain contains roughly 86 billion neurons, each one a tiny electrical device. But at any
      given moment, most of them are <em>silent</em>. A neuron only fires — only consumes energy —
      when the input to its cell body accumulates past a threshold. Then it fires a spike, resets,
      and goes quiet again. The whole system is exquisitely event-driven.""",
     '你的大脑约有 860 亿个神经元，每一个都是一个微型电子器件。但在任意时刻，其中大多数是<em>静默的</em>。'
     '只有当胞体收到的输入累积超过阈值时，神经元才发放——才消耗能量。'
     '随后它发出一个脉冲、复位，再次归于安静。整个系统是精巧的事件驱动。'),
    ("""This is exactly the opposite of how a traditional neural network works. A CNN processes every
      pixel, every sample, every time step, regardless of whether anything interesting is happening.
      That's fine when you have a power outlet. It's fatal when you're running on a CR2032 coin cell.""",
     '这与传统神经网络的工作方式恰好相反。CNN 处理每一个像素、每一个采样点、每一个时间步，'
     '无论当下是否发生了值得关注的事。有电源插座时这没问题；靠一颗 CR2032 纽扣电池供电时，这是致命的。'),
    ("""<strong>Spiking neural networks (SNNs)</strong> are artificial networks built around the same
      principle. Hidden units are leaky integrate-and-fire (LIF) neurons: they accumulate input, and
      only fire — only do work — when the accumulated charge crosses a threshold. In between firings,
      they're idle. On purpose-built neuromorphic chips like Intel's Loihi, each spike triggers
      a single cheap accumulate operation (~0.1 picojoules). A standard GPU multiply-accumulate
      costs ~3.7 pJ — <strong>37× more expensive per operation</strong>, before even counting
      how many fewer operations an SNN performs.""",
     '<strong>脉冲神经网络（SNN）</strong>正是围绕同一原理构建的人工网络。隐藏单元是泄漏积分点火（LIF）神经元：'
     '它们累积输入，只有当累积电荷越过阈值时才发放——才做功。两次发放之间，它们处于空闲状态。'
     '在 Intel Loihi 这类专用神经形态芯片上，每个脉冲只触发一次廉价累加运算（约 0.1 皮焦）。'
     '而标准 GPU 的一次乘加约需 3.7 pJ——<strong>单次运算贵 37 倍</strong>，'
     '这还没算上 SNN 的运算次数本就少得多。'),

    # ── 数字断点 ──
    ('of SNN neurons are silent<br>at any given time-step',
     '的 SNN 神经元<br>在任一时刻处于静默'),
    ('cheaper per-operation cost<br>on neuromorphic silicon',
     '神经形态芯片上<br>单次运算成本更低'),
    ('total energy reduction<br>per ECG inference',
     '单次心电图推理的<br>总能耗降低'),

    # ── 03 数据 ──
    ('</span> · The data</div>', '</span> · 数据</div>'),
    ('<h2>The experiment</h2>', '<h2>实验</h2>'),
    ("""The claim sounds good. But does it hold up on a real clinical dataset, against real baselines,
      with a proper held-out test set? That's what this project tests.""",
     '这个说法听起来不错。但在真实临床数据集上、面对真实的基线模型、用规范的留出测试集检验时，它还站得住吗？'
     '这正是本项目要验证的。'),
    ("""The dataset is <strong>PTB-XL</strong>, the largest open 12-lead ECG database in the world:
      21,837 recordings, each 10 seconds long, sampled at 100 Hz, annotated by two cardiologists.
      Each recording is labeled with one or more of five diagnostic superclasses defined by
      the Association for the Advancement of Medical Instrumentation (AAMI):""",
     '数据集是 <strong>PTB-XL</strong>——全球最大的公开 12 导联心电数据库：'
     '21,837 条记录，每条 10 秒、100 Hz 采样，由两位心内科医生标注。'
     '每条记录被标注为美国医疗器械促进协会（AAMI）定义的五个诊断超类中的一个或多个：'),

    # ── 五类说明 ──
    ('<div class="de-name">NORM — Normal</div>', '<div class="de-name">NORM — 正常</div>'),
    ('<div class="de-desc">No pathology found. Normal sinus rhythm.</div>',
     '<div class="de-desc">未见病理改变，正常窦性心律。</div>'),
    ('<div class="de-n">7,596 positive in training set</div>',
     '<div class="de-n">训练集中 7,596 例阳性</div>'),

    ('<div class="de-name">MI — Myocardial Infarction</div>',
     '<div class="de-name">MI — 心肌梗死</div>'),
    ('<div class="de-desc">Heart attack. Death of cardiac muscle from a blocked artery. Often shows as ST-elevation or Q-waves.</div>',
     '<div class="de-desc">俗称心脏病发作，冠状动脉阻塞导致心肌坏死，常表现为 ST 段抬高或 Q 波。</div>'),
    ('<div class="de-n">4,379 positive in training set</div>',
     '<div class="de-n">训练集中 4,379 例阳性</div>'),

    ('<div class="de-name">STTC — ST/T-wave Change</div>',
     '<div class="de-name">STTC — ST/T 波改变</div>'),
    ('<div class="de-desc">Altered repolarization pattern. Can indicate ischemia, drug effects, or electrolyte imbalance.</div>',
     '<div class="de-desc">复极模式异常，可能提示缺血、药物影响或电解质失衡。</div>'),
    ('<div class="de-n">4,186 positive in training set</div>',
     '<div class="de-n">训练集中 4,186 例阳性</div>'),

    ('<div class="de-name">CD — Conduction Disturbance</div>',
     '<div class="de-name">CD — 传导障碍</div>'),
    ('<div class="de-desc">Abnormal electrical pathway. Bundle-branch blocks, AV blocks, or pacing abnormalities.</div>',
     '<div class="de-desc">电传导通路异常，包括束支阻滞、房室阻滞或起搏异常。</div>'),
    ('<div class="de-n">3,907 positive in training set</div>',
     '<div class="de-n">训练集中 3,907 例阳性</div>'),

    ('<div class="de-name">HYP — Hypertrophy</div>',
     '<div class="de-name">HYP — 心室肥厚</div>'),
    ('<div class="de-desc">Thickened chamber wall, often from chronic hypertension. The hardest class — fewest examples, subtlest signal.</div>',
     '<div class="de-desc">心室壁增厚，常由慢性高血压引起。最难的一类——样本最少、信号最细微。</div>'),
    ('<div class="de-n">2,119 positive — rarest class, biggest challenge</div>',
     '<div class="de-n">2,119 例阳性——最稀有的类别，最大的挑战</div>'),

    ("""A recording can carry multiple labels simultaneously — a patient can have both a conduction
      disturbance and signs of past infarction. This is a <strong>multi-label classification problem</strong>,
      which is harder than single-label and much closer to clinical reality.""",
     '一条记录可以同时带有多个标签——患者可能既有传导障碍、又有陈旧性梗死的征象。'
     '这是一个<strong>多标签分类问题</strong>，比单标签更难，也更贴近临床实际。'),
    ("""We use the <strong>PTB-XL recommended stratified split</strong>: folds 1–8 for training
      (17,084 recordings), fold 9 for validation (2,146), fold 10 for the final test (2,158).
      Patients never appear in more than one fold — there is no data leakage.
      All three models trained on exactly the same data, evaluated on exactly the same test set.""",
     '我们采用 <strong>PTB-XL 官方推荐的分层划分</strong>：fold 1–8 用于训练（17,084 条），'
     'fold 9 用于验证（2,146 条），fold 10 用于最终测试（2,158 条）。'
     '同一患者不会出现在多个 fold 中——不存在数据泄漏。'
     '三个模型训练数据完全相同，测试集也完全相同。'),

    # ── 04 网络 ──
    ('</span> · The network</div>', '</span> · 网络结构</div>'),
    ('<h2>The spiking network: S-TCN</h2>', '<h2>脉冲网络：S-TCN</h2>'),
    ("""The model is a <strong>Spiking Temporal Convolutional Network (S-TCN)</strong> — a deep network
      built entirely from LIF neurons. It processes the ECG in five stages:""",
     '该模型是一个<strong>脉冲时序卷积网络（S-TCN）</strong>——完全由 LIF 神经元构成的深度网络。'
     '它分五个阶段处理心电图：'),

    ('<div class="pipe-title">Delta-modulation encoding</div>',
     '<div class="pipe-title">Δ 调制编码</div>'),
    ("""The raw voltage waveform is never fed directly to the network. Instead, a threshold encoder
        fires an ON spike when voltage rises by θ=0.15 mV, and an OFF spike when it falls by the same.
        Between crossings: <strong>silence, zero events, zero energy</strong>. Flat baseline = idle.
        The QRS complex (the big heartbeat spike) produces a burst of events.
        12 leads × 2 directions = 24 event channels.""",
     '原始电压波形从不直接送入网络。取而代之的是阈值编码器：电压上升 θ=0.15 mV 时发出一个 ON 脉冲，'
     '下降同样幅度时发出一个 OFF 脉冲。两次跨越之间是<strong>静默、零事件、零能耗</strong>。'
     '基线平坦即空闲。QRS 波群（心跳的大尖峰）会产生一阵事件爆发。'
     '12 导联 × 2 个方向 = 24 个事件通道。'),

    ('<div class="pipe-title">Six dilated spiking blocks</div>',
     '<div class="pipe-title">六个空洞脉冲块</div>'),
    ("""Six temporal convolutional blocks with exponentially increasing dilation (1, 2, 4, 8, 16, 32).
        Each block's receptive field doubles, so by block 6 every neuron can "see" the entire
        10-second recording. <strong>Novel: membrane-gated residuals.</strong> Standard residual
        connections don't work in SNNs — adding binary spikes to binary spikes produces multi-bit
        activations. Instead, skip paths inject current directly into the output LIF's membrane
        potential. Spikes stay binary. A learnable gate per channel controls the balance.""",
     '六个时序卷积块，空洞率指数增长（1, 2, 4, 8, 16, 32）。每个块的感受野翻倍，'
     '到第 6 块时每个神经元都能"看到"完整的 10 秒记录。'
     '<strong>创新点：膜电位门控残差。</strong>标准残差连接在 SNN 中不成立——'
     '把二值脉冲加到二值脉冲上会产生多比特激活值。'
     '这里的跳连改为把电流直接注入输出 LIF 的膜电位，脉冲始终保持二值；'
     '每个通道一个可学习门控来调节两路权重。'),

    ('<div class="pipe-title">Spiking attention pooling</div>',
     '<div class="pipe-title">脉冲注意力池化</div>'),
    ("""A small LIF "gate" neuron accumulates per-step evidence and fires at the time-steps where
        the ECG is most informative (typically the QRS complex and the T-wave).
        <strong>Novel: this is the first event-driven attention mechanism for ECG SNNs.</strong>
        No softmax, no real-valued mixing — just a spike pattern that selects which moments matter.
        The classifier pools only from those time-steps.""",
     '一个小的 LIF「门控」神经元逐时间步累积证据，在心电信息量最大的时刻发放'
     '（通常是 QRS 波群和 T 波）。'
     '<strong>创新点：这是首个面向心电 SNN 的事件驱动注意力机制。</strong>'
     '没有 softmax、没有实值加权——只有一个脉冲模式，用来选出哪些时刻重要。'
     '分类器只从这些时间步做池化。'),

    ('<div class="pipe-title">Non-leaky integrator → diagnosis</div>',
     '<div class="pipe-title">非泄漏积分器 → 诊断</div>'),
    ("""A final dense LIF layer feeds a non-leaky integrator that accumulates evidence over
        T<sub>dense</sub> extra steps. The membrane potential at the end of accumulation — one
        scalar per disease class — is passed through a sigmoid. Trained with binary cross-entropy.""",
     '最后一层稠密 LIF 驱动一个非泄漏积分器，在额外的 T<sub>dense</sub> 步上累积证据。'
     '累积结束时的膜电位——每个疾病类别一个标量——经 sigmoid 输出，用二元交叉熵训练。'),

    # ── 05 现场演示 ──
    ('</span> · Live demo</div>', '</span> · 现场演示</div>'),
    ('<h2>Try it: one ECG, three models, side by side</h2>',
     '<h2>试一试：一份心电图，三个模型，并排对比</h2>'),
    ("""Every recording in the dropdown below is from fold 10 — the test set none of the models ever saw.
      Pick one and watch the full SNN inference pipeline run in real time: its spike patterns, what it
      attends to, and its final diagnosis. Then see <strong>the exact same recording run through the
      CNN and ResNet baselines</strong>, so you can judge for yourself where the spiking network
      agrees, where it differs, and what that 3.6-point accuracy gap actually looks like on a single heart.""",
     '下面下拉框中的每一条记录都来自 fold 10——三个模型都未曾见过的测试集。'
     '选一条，观察完整的 SNN 推理流水线实时运行：它的脉冲模式、它关注的位置，以及最终诊断。'
     '然后看<strong>同一条记录在 CNN 与 ResNet 基线下的结果</strong>，'
     '由你自己判断：脉冲网络在哪里一致、在哪里分歧，那 3.6 个百分点的精度差距在单颗心脏上究竟是什么样子。'),
    ('<option value="">— loading test recordings —</option>',
     '<option value="">— 正在加载测试记录 —</option>'),
    ('<button id="runBtn" class="btn primary">Run all three models</button>',
     '<button id="runBtn" class="btn primary">运行全部三个模型</button>'),
    ('<span class="picker-sep">or</span>', '<span class="picker-sep">或</span>'),
    ('<label class="btn">Upload .npy<input type="file" id="uploadInput" accept=".npy" hidden /></label>',
     '<label class="btn">上传 .npy<input type="file" id="uploadInput" accept=".npy" hidden /></label>'),

    ('<span class="vl-step">Diagnosis — all three models on this recording</span>',
     '<span class="vl-step">诊断 —— 三个模型对本条记录的判断</span>'),
    ('<span class="vl-sub">Sigmoid probability per AAMI class for each model. Truth from cardiologist annotation. Bars ≥ 50% = a positive call.</span>',
     '<span class="vl-sub">各模型对每个 AAMI 类别的 Sigmoid 概率。真值来自心内科医生标注。条形 ≥ 50% 即为阳性判定。</span>'),
    ('<span class="vl-step">Raw ECG</span>', '<span class="vl-step">原始心电图</span>'),
    ("""<span class="vl-sub" id="ecgMeta">10 seconds · 100 Hz · 12 leads</span>""",
     '<span class="vl-sub" id="ecgMeta">10 秒 · 100 Hz · 12 导联</span>'),
    ('<span class="vl-step">SNN · Step 1 — Delta-modulation spikes</span>',
     '<span class="vl-step">SNN · 第 1 步 —— Δ 调制脉冲</span>'),
    ('<span class="vl-sub">Only threshold crossings fire. Silence = no energy consumed.</span>',
     '<span class="vl-sub">只有越过阈值时才发放。静默 = 不消耗能量。</span>'),
    ('<span class="vl-step">SNN · Step 2 — Final spiking block</span>',
     '<span class="vl-step">SNN · 第 2 步 —— 末端脉冲块</span>'),
    ('<span class="vl-sub">384 LIF channels after 6× dilated processing. Time axis shrunk ~16×.</span>',
     '<span class="vl-sub">经 6 层空洞处理后得到 384 个 LIF 通道。时间轴压缩约 16 倍。</span>'),
    ('<span class="vl-step">SNN · Step 3 — Attention gate</span>',
     '<span class="vl-step">SNN · 第 3 步 —— 注意力门控</span>'),
    ('<span class="vl-sub">Green = time-steps the gate LIF selected. Only these feed the classifier.</span>',
     '<span class="vl-sub">绿色 = 门控 LIF 选中的时间步，只有这些进入分类器。</span>'),
    ('<span class="vl-step">SNN · Step 4 — Output integrator</span>',
     '<span class="vl-step">SNN · 第 4 步 —— 输出积分器</span>'),
    ('<span class="vl-sub">Evidence accumulation per class. Steep slope = confident diagnosis.</span>',
     '<span class="vl-sub">各类别的证据累积。斜率越陡 = 诊断越有把握。</span>'),
    ('<span class="vl-step">SNN · Layer-by-layer spike rate</span>',
     '<span class="vl-step">SNN · 逐层脉冲发放率</span>'),
    ('<span class="vl-sub">The fraction of neurons firing per step. Lower = less energy on neuromorphic hardware.</span>',
     '<span class="vl-sub">每个时间步发放脉冲的神经元比例。越低 = 神经形态硬件上越省电。</span>'),

    # ── 06 结果 ──
    ('</span> · The results</div>', '</span> · 结果</div>'),
    ('<h2>The findings</h2>', '<h2>主要发现</h2>'),
    ("""We trained three models on the same data with the same training budget:
      the SNN (S-TCN), a parameter-matched 1-D CNN, and a parameter-matched 1-D ResNet.
      Then we evaluated all three on the held-out fold 10 — 2,158 recordings the models
      had never seen.""",
     '我们在相同数据、相同训练预算下训练了三个模型：SNN（S-TCN）、参数量匹配的一维 CNN、'
     '参数量匹配的一维 ResNet。随后在留出的 fold 10 上评测三者——2,158 条模型从未见过的记录。'),
    ("""The metric is <strong>macro-AUROC</strong> (area under the ROC curve, averaged across all five
      disease classes). A score of 1.0 means perfect classification. A score of 0.5 means
      the model is no better than flipping a coin. Published state-of-the-art models on PTB-XL
      typically score between 0.87 and 0.93.""",
     '指标是 <strong>宏平均 AUROC</strong>（ROC 曲线下面积，在五个疾病类别上取平均）。'
     '1.0 表示完美分类，0.5 表示模型与抛硬币无异。'
     '已发表的 PTB-XL 最先进模型通常落在 0.87 到 0.93 之间。'),

    ('<div class="rr-sub">1,893,768 params · 40.6M effective SOPs<br>~4.1 µJ per inference (neuromorphic)</div>',
     '<div class="rr-sub">1,893,768 参数 · 40.6M 等效 SOP<br>单次推理约 4.1 µJ（神经形态）</div>'),
    ('<div class="rr-name">CNN1D baseline</div>', '<div class="rr-name">CNN1D 基线</div>'),
    ('<div class="rr-sub">1,741,381 params · 294.6M MACs<br>~1,090 µJ per inference (dense GPU)</div>',
     '<div class="rr-sub">1,741,381 参数 · 294.6M MAC<br>单次推理约 1,090 µJ（稠密 GPU）</div>'),
    ('<div class="rr-name">ResNet1D baseline</div>', '<div class="rr-name">ResNet1D 基线</div>'),
    ('<div class="rr-sub">1,879,685 params · 50.5M MACs<br>~187 µJ per inference (dense GPU)</div>',
     '<div class="rr-sub">1,879,685 参数 · 50.5M MAC<br>单次推理约 187 µJ（稠密 GPU）</div>'),
    ('<div class="rr-note">macro-AUROC · fold-10 test set · 2,158 held-out recordings</div>',
     '<div class="rr-note">宏平均 AUROC · fold-10 测试集 · 2,158 条留出记录</div>'),

    ("""The SNN scores <strong>0.8663</strong> versus the CNN's <strong>0.9023</strong> — a gap of
      3.6 percentage points. That sounds like a lot, but it comes with a tradeoff that changes the
      arithmetic completely.""",
     'SNN 得分为 <strong>0.8663</strong>，CNN 为 <strong>0.9023</strong>——相差 3.6 个百分点。'
     '这听起来不少，但它伴随着一项权衡，而这项权衡彻底改变了这笔账。'),

    ('<h3>Accuracy vs. baselines</h3>', '<h3>精度对比基线</h3>'),
    ('<div class="card-sub">Macro-AUROC and AUPRC on the held-out test fold. Higher = better. Parameter counts matched within 8%.</div>',
     '<div class="card-sub">留出测试集上的宏平均 AUROC 与 AUPRC。越高越好。参数量匹配误差在 8% 以内。</div>'),
    ('<h3>Per-class AUROC breakdown</h3>', '<h3>逐类 AUROC 拆解</h3>'),
    ('<div class="card-sub">The SNN tracks the CNN closely on NORM, MI, and STTC. The gap is concentrated in HYP — the rarest and hardest class. CD improved substantially in later training epochs.</div>',
     '<div class="card-sub">在 NORM、MI、STTC 上 SNN 紧追 CNN。差距集中在 HYP——最稀有、最难的一类。'
     'CD 在训练后期有显著提升。</div>'),
    ('<h3>ROC curves · per class · all three models</h3>', '<h3>ROC 曲线 · 逐类 · 三个模型</h3>'),
    ('<div class="card-sub"><span style="color:#ff5470;font-weight:700">solid red</span> = SNN · <span style="color:#48cae4">dashed azure</span> = CNN · <span style="color:#ffc44d">dotted amber</span> = ResNet. AUC labeled per model. Diagonal = random chance.</div>',
     '<div class="card-sub"><span style="color:#ff5470;font-weight:700">红色实线</span> = SNN · '
     '<span style="color:#48cae4">蓝色虚线</span> = CNN · '
     '<span style="color:#ffc44d">琥珀色点线</span> = ResNet。每条曲线标注 AUC。对角线 = 随机猜测。</div>'),
    ('<h3>Precision–recall curves · per class</h3>', '<h3>精确率-召回率曲线 · 逐类</h3>'),
    ('<div class="card-sub">Where ROC can flatter a model on imbalanced classes, PR curves don\'t. Same line styles. The dashed baseline in each panel is the class prevalence (a no-skill classifier). Telling for rare classes like HYP.</div>',
     '<div class="card-sub">在类别不平衡时 ROC 会美化模型表现，PR 曲线不会。线型含义同上。'
     '每个子图中的虚线是该类别的基准患病率（无技能分类器）。对 HYP 这类稀有类别尤其说明问题。</div>'),

    # ── 07 能耗 ──
    ('</span> · The energy</div>', '</span> · 能耗</div>'),
    ('<h2>Now look at the energy</h2>', '<h2>再看能耗</h2>'),
    ("""The energy calculation uses the Horowitz (ISSCC 2014) constants, the standard reference
      in the neuromorphic hardware literature. A dense 32-bit floating-point multiply-accumulate
      on a 45 nm ASIC costs approximately <strong>3.7 picojoules</strong>. A spike-triggered
      accumulate on a neuromorphic chip like Intel's Loihi costs approximately <strong>0.1 pJ</strong>.""",
     '能耗计算采用 Horowitz（ISSCC 2014）常量——神经形态硬件文献中的标准参考。'
     '45 nm ASIC 上一次稠密 32 位浮点乘加约需 <strong>3.7 皮焦</strong>；'
     'Intel Loihi 这类神经形态芯片上一次脉冲触发的累加约需 <strong>0.1 pJ</strong>。'),
    ("""The SNN performs 40.6 million effective synaptic ops per inference —
      "effective" meaning we measured the actual per-layer spike rates on the validation set
      and multiplied. At 0.1 pJ/SOP that's <strong>4.06 µJ</strong> per ECG window.
      The CNN performs 294.6 million MACs at 3.7 pJ each — <strong>1,090 µJ</strong>.
      That's a factor of <strong><span class="js-eratio">287</span>×</strong>, driven by two compounding effects: fewer total ops,
      and cheaper ops per unit.""",
     'SNN 单次推理执行 4,060 万次有效突触运算——「有效」是指我们在验证集上实测了逐层发放率并相乘。'
     '按 0.1 pJ/SOP 计，每个心电窗口 <strong>4.06 µJ</strong>。'
     'CNN 执行 2.946 亿次 MAC，每次 3.7 pJ——<strong>1,090 µJ</strong>。'
     '两者相差 <strong><span class="js-eratio">287</span> 倍</strong>，由两个叠加效应驱动：'
     '总运算次数更少，且单次运算更便宜。'),
    ('<h3>Energy per inference</h3>', '<h3>单次推理能耗</h3>'),
    ('<div class="card-sub">SNN energy assumes neuromorphic hardware (Loihi). CNN and ResNet assume dense ASIC. The <span class="js-eratio">287</span>× gap comes from both fewer ops (7.8×) and cheaper ops per unit (37×).</div>',
     '<div class="card-sub">SNN 能耗按神经形态硬件（Loihi）估算，CNN 与 ResNet 按稠密 ASIC 估算。'
     '<span class="js-eratio">287</span> 倍的差距同时来自运算更少（7.8 倍）与单次运算更便宜（37 倍）。</div>'),
    ('<h2>What does <span class="js-eratio">287</span>× actually mean?</h2>',
     '<h2><span class="js-eratio">287</span>× 究竟意味着什么？</h2>'),
    ("""Here's a concrete scenario: a wearable Holter monitor analyzing a 10-second ECG window
      every 10 seconds — 8,640 inferences over a 24-hour monitoring session.""",
     '举一个具体场景：一台可穿戴 Holter 监测仪每 10 秒分析一个 10 秒的心电窗口——'
     '24 小时监测中合计 8,640 次推理。'),
    ('<div class="hc-loading">loading energy calculations…</div>',
     '<div class="hc-loading">正在加载能耗计算…</div>'),
    ("""The 3.6 pp AUROC gap is real. Whether it matters depends entirely on the deployment.
      For a <strong>wearable screening device</strong> — one that flags suspected pathology and
      routes the patient to a clinical follow-up ECG — the lower sensitivity is acceptable. You're
      trading a small increase in false negatives for a device that works for days on a single charge.""",
     '3.6 个百分点的 AUROC 差距是真实存在的。它是否要紧，完全取决于部署场景。'
     '对于<strong>可穿戴筛查设备</strong>——提示疑似病理、把患者引导去做临床复查心电图——'
     '较低的灵敏度是可以接受的。你用假阴性略微增加，换来一台单次充电可工作数天的设备。'),
    ("""For <strong>definitive diagnosis</strong> in a clinical setting where you have a GPU and
      a power outlet, the CNN is the right choice.""",
     '而在有 GPU、有电源插座的临床场景下做<strong>确诊</strong>，CNN 才是正确选择。'),
    ("""One more thing to note about latency: the measured inference times here (1,500 ms for
      the SNN vs. 2 ms for the CNN) are an artifact of <em>simulating</em> spiking dynamics
      on a dense GPU in PyTorch. On actual Loihi 2 silicon, SNN inference is typically
      sub-millisecond. The latency comparison is not representative of deployed neuromorphic hardware.""",
     '关于延迟还需说明一点：这里的实测推理时间（SNN 1,500 ms 对 CNN 2 ms）是在稠密 GPU 上用 PyTorch '
     '<em>模拟</em>脉冲动力学造成的假象。在真实的 Loihi 2 芯片上，SNN 推理通常在亚毫秒级。'
     '这一延迟对比并不能代表部署后的神经形态硬件。'),

    ('<h3>Complete comparison</h3>', '<h3>完整对比</h3>'),
    ('<div class="card-sub">All metrics on fold-10 test set. MACs/SOPs counted analytically. Spike rates measured on validation fold. Latency on Modal T4 (B=1). <strong>Bold</strong> = best in column.</div>',
     '<div class="card-sub">全部指标基于 fold-10 测试集。MAC/SOP 为解析计数，发放率在验证 fold 上实测。'
     '延迟测于 Modal T4（B=1）。<strong>加粗</strong> = 该列最优。</div>'),
    ('<div id="compareTable" class="compare-wrap"><div class="empty muted">loading…</div></div>',
     '<div id="compareTable" class="compare-wrap"><div class="empty muted">加载中…</div></div>'),

    # ── 08 训练 ──
    ('</span> · The training</div>', '</span> · 训练</div>'),
    ('<h2>Did the SNN actually converge?</h2>', '<h2>SNN 真的收敛了吗？</h2>'),
    ("""A common concern with spiking networks is training instability. Surrogate-gradient
      backprop (the technique that makes SNNs trainable at all) is known to be sensitive to
      hyperparameters, and some architectures fail to learn meaningful representations.""",
     '关于脉冲网络的一个常见担忧是训练不稳定。替代梯度反向传播'
     '（正是它让 SNN 可训练）对超参数敏感，而有些架构根本学不到有意义的表征。'),
    ("""The training curve below shows <strong>200 full epochs</strong> of S-TCN on PTB-XL.
      It does learn. It converges cleanly to a validation AUROC of <strong>0.8748</strong> by
      epoch 155, and the best checkpoint holds through epoch 200 — no collapse, no divergence.
      The per-class lines show each disease superclass tracking upward together.""",
     '下图是 S-TCN 在 PTB-XL 上<strong>完整 200 轮</strong>的训练曲线。它确实在学。'
     '到第 155 轮时干净地收敛到验证 AUROC <strong>0.8748</strong>，最优检查点一直保持到第 200 轮——'
     '没有崩溃、没有发散。逐类曲线显示五个疾病超类同步上升。'),
    ('<h3>SNN training curve · 200 epochs</h3>', '<h3>SNN 训练曲线 · 200 轮</h3>'),
    ("""<span style="color:#48cae4">azure</span> = val macro-AUROC ·
      <span style="color:#3ddc97">dashed green</span> = running best ·
      <span style="color:#ffc44d">amber</span> = train loss.
      Thin lines = per-class AUROC. Best val-AUROC: 0.8748 at epoch 155.""",
     '<span style="color:#48cae4">蓝色</span> = 验证集宏平均 AUROC · '
     '<span style="color:#3ddc97">绿色虚线</span> = 历史最优 · '
     '<span style="color:#ffc44d">琥珀色</span> = 训练损失。细线 = 逐类 AUROC。'
     '最优验证 AUROC：0.8748，出现在第 155 轮。'),

    # ── 页脚 ──
    ('<div><strong>Model</strong>S-TCN · 1,893,768 params · 6 dilated LIF blocks · snntorch 0.9.4 + PyTorch 2.4.1</div>',
     '<div><strong>模型</strong>S-TCN · 1,893,768 参数 · 6 个空洞 LIF 块 · snntorch 0.9.4 + PyTorch 2.4.1</div>'),
    ('<div><strong>Dataset</strong><a href="https://physionet.org/content/ptb-xl/1.0.3/" target="_blank">PTB-XL 1.0.3 (PhysioNet)</a> · 21,837 recordings · 5 AAMI superclasses</div>',
     '<div><strong>数据集</strong><a href="https://physionet.org/content/ptb-xl/1.0.3/" target="_blank">PTB-XL 1.0.3 (PhysioNet)</a> · 21,837 条记录 · 5 个 AAMI 超类</div>'),
    ('<div><strong>Training</strong>Modal A10G · cosine LR 1e-3 → 0 · AdamW · EMA=0.999 · 200 epochs · fold 1–8</div>',
     '<div><strong>训练</strong>Modal A10G · 余弦学习率 1e-3 → 0 · AdamW · EMA=0.999 · 200 轮 · fold 1–8</div>'),
    ('<div><strong>Energy reference</strong>Horowitz, ISSCC 2014 · 3.7 pJ/MAC (FP32) · 0.1 pJ/SOP (neuromorphic accumulate)</div>',
     '<div><strong>能耗参考</strong>Horowitz, ISSCC 2014 · 3.7 pJ/MAC (FP32) · 0.1 pJ/SOP（神经形态累加）</div>'),
]


PTBXL_JS = [
    # ── 类别说明（画在图上，也会被 canvas 直接绘制）──
    ('NORM: "Normal sinus rhythm",', 'NORM: "正常窦性心律",'),
    ('MI:   "Myocardial Infarction",', 'MI:   "心肌梗死",'),
    ('STTC: "ST/T-wave Change",', 'STTC: "ST/T 波改变",'),
    ('CD:   "Conduction Disturbance",', 'CD:   "传导障碍",'),
    ('HYP:  "Hypertrophy",', 'HYP:  "心室肥厚",'),

    # ── 模型标签（注意 L232 的 replace 正则必须同步改成「基线」）──
    ('const MODEL_LABEL  = { snn:"SNN (NeuroCardio)", cnn:"CNN1D baseline", resnet:"ResNet1D baseline" };',
     'const MODEL_LABEL  = { snn:"SNN (NeuroCardio)", cnn:"CNN1D 基线", resnet:"ResNet1D 基线" };'),
    ("const name = MODEL_LABEL[v.m].replace(/ \\(.*\\)| baseline/,\"\");",
     "const name = MODEL_LABEL[v.m].replace(/ \\(.*\\)| 基线/,\"\");"),

    # ── 画布文字 ──
    ('ctx.fillText("ON  (rising)", 8, 14);', 'ctx.fillText("ON（上升）", 8, 14);'),
    ('ctx.fillText("OFF (falling)", 8, h/2+14);', 'ctx.fillText("OFF（下降）", 8, h/2+14);'),
    ('ctx.fillText(`${C} channels`, 8,12);', 'ctx.fillText(`${C} 通道`, 8,12);'),
    ('ctx.fillText(`${T} time-steps`, 8,26);', 'ctx.fillText(`${T} 个时间步`, 8,26);'),
    ('ctx.fillText(`spike rate: ${(r*100).toFixed(1)}%`, 8, 40);',
     'ctx.fillText(`发放率：${(r*100).toFixed(1)}%`, 8, 40);'),
    ('ctx.fillText(`${events.length} gate spikes (pools at ${((events.length/Math.max(T,1))*100).toFixed(1)}% of time-steps)`, 8, 14);',
     'ctx.fillText(`${events.length} 个门控脉冲（在 ${((events.length/Math.max(T,1))*100).toFixed(1)}% 的时间步上参与池化）`, 8, 14);'),
    ('ctx.fillStyle="rgba(174,183,205,0.5)"; ctx.font="10px \'Space Mono\',ui-monospace,monospace";\n  ctx.fillText("macro-AUROC / AUPRC →", PAD.l, PAD.t-10);',
     'ctx.fillStyle="rgba(174,183,205,0.5)"; ctx.font="10px \'Space Mono\',ui-monospace,monospace";\n  ctx.fillText("宏平均 AUROC / AUPRC →", PAD.l, PAD.t-10);'),
    ('ctx.fillText("estimated energy / inference →", PAD.l, PAD.t-6);',
     'ctx.fillText("单次推理估算能耗 →", PAD.l, PAD.t-6);'),
    ('ctx.fillText("recall", x0+pw*0.36, y0+ph+14);', 'ctx.fillText("召回率", x0+pw*0.36, y0+ph+14);'),
    ('ctx.fillText("precision", -14, 0);', 'ctx.fillText("精确率", -14, 0);'),
    ('ctx.fillText("epoch →", w-PAD.r+4, h-PAD.b+10);', 'ctx.fillText("轮次 →", w-PAD.r+4, h-PAD.b+10);'),

    # ── 三模型判定表 ──
    ('<div class="cr-h-class">Class</div>', '<div class="cr-h-class">类别</div>'),
    ('      <div>truth</div>', '      <div>真值</div>'),
    ("""el.innerHTML = `<span>Uploaded recording — no cardiologist ground truth to compare against. The bars above still show how all three trained models read this ECG.</span>`;""",
     """el.innerHTML = `<span>上传的记录——没有医生标注真值可供对比。上方的条形仍展示三个已训练模型如何解读这份心电图。</span>`;"""),
    ('? `<span class="agree">✓ exact match</span>`', '? `<span class="agree">✓ 完全一致</span>`'),
    ('const truthTxt = truth.length\n    ? truth.map(t => `<strong style="color:${CLASS_COLORS[t]}">${t}</strong>`).join(", ")\n    : "<strong>NORM-only / none</strong>";',
     'const truthTxt = truth.length\n    ? truth.map(t => `<strong style="color:${CLASS_COLORS[t]}">${t}</strong>`).join(", ")\n    : "<strong>仅 NORM / 无标注</strong>";'),
    ('    ? `All ${have.length} models agree with the cardiologist here.`',
     '    ? `${have.length} 个模型都与医生标注一致。`'),
    ('      ? `No model nailed every label — a genuinely hard recording.`',
     '      ? `没有模型完全命中所有标签——这是一条很难的记录。`'),
    ('      : `${nAgree} of ${have.length} models matched the cardiologist exactly.`;',
     '      : `${have.length} 个模型中有 ${nAgree} 个与医生标注完全一致。`;'),
    ('el.innerHTML = `Ground truth: ${truthTxt}. &nbsp; ${consensus}<br>${vlines}`;',
     'el.innerHTML = `真值：${truthTxt}。 &nbsp; ${consensus}<br>${vlines}`;'),

    # ── 逐层发放率 ──
    ("""$("sparsityNote").textContent =
    `Mean spike rate: ${(mean*100).toFixed(1)}% · `+
    `On a neuromorphic chip, effective ops ≈ MACs × spike_rate per layer.`;""",
     """$("sparsityNote").textContent =
    `平均发放率：${(mean*100).toFixed(1)}% · `+
    `在神经形态芯片上，逐层有效运算量 ≈ MAC 数 × 发放率。`;"""),

    # ── 能耗说明 ──
    ("""      `The SNN uses <strong>${energyRatio.toFixed(0)}× less</strong> estimated energy per inference
       (${fmtJ(snnE_pj)} neuromorphic vs. ${fmtJ(cnnE_pj)} dense) — driven by both
       <strong>${opsRatio.toFixed(1)}× fewer operations</strong> (SOPs vs. MACs) and
       <strong>37× cheaper per-operation cost</strong> (0.1 pJ/SOP vs. 3.7 pJ/MAC on 45 nm CMOS).
       The macro-AUROC cost is <strong>${Math.abs(aurocGap*100).toFixed(2)} pp</strong>.
       Wall-clock latency on a dense GPU still favors the CNN because we <em>simulate</em> LIF
       dynamics in PyTorch — on actual Loihi 2 silicon, SNN inference is sub-millisecond.`;""",
     """      `SNN 单次推理的估算能耗<strong>低 ${energyRatio.toFixed(0)} 倍</strong>
       （神经形态 ${fmtJ(snnE_pj)} 对稠密 ${fmtJ(cnnE_pj)}）——同时来自
       <strong>运算次数少 ${opsRatio.toFixed(1)} 倍</strong>（SOP 对 MAC）与
       <strong>单次运算便宜 37 倍</strong>（45 nm CMOS 上 0.1 pJ/SOP 对 3.7 pJ/MAC）。
       代价是宏平均 AUROC 下降 <strong>${Math.abs(aurocGap*100).toFixed(2)} pp</strong>。
       在稠密 GPU 上墙钟延迟仍偏向 CNN，因为我们是<em>模拟</em> PyTorch 中的 LIF 动力学——
       在真实的 Loihi 2 芯片上，SNN 推理是亚毫秒级的。`;"""),

    ("""      `The SNN achieves <strong>${(snn.macro_auroc*100).toFixed(1)}% macro-AUROC</strong>
       (${(aurocGap*100).toFixed(2)} pp vs CNN) while using
       <strong>${opsRatio.toFixed(1)}× fewer effective operations</strong> and
       <strong>${energyRatio.toFixed(0)}× less energy</strong>.
       The mean spike rate of ${snn.spike_rate_mean!=null?(snn.spike_rate_mean*100).toFixed(1)+"%":"~12%"}
       means ~${(100-(snn.spike_rate_mean||0.126)*100).toFixed(0)}% of synapses are idle at any
       given time-step — the core reason neuromorphic hardware wins on energy.`;""",
     """      `SNN 取得 <strong>宏平均 AUROC ${(snn.macro_auroc*100).toFixed(1)}%</strong>
       （相对 CNN ${(aurocGap*100).toFixed(2)} pp），同时
       <strong>有效运算次数少 ${opsRatio.toFixed(1)} 倍</strong>、
       <strong>能耗低 ${energyRatio.toFixed(0)} 倍</strong>。
       平均发放率 ${snn.spike_rate_mean!=null?(snn.spike_rate_mean*100).toFixed(1)+"%":"~12%"}
       意味着任一时刻约有 ${(100-(snn.spike_rate_mean||0.126)*100).toFixed(0)}% 的突触处于空闲——
       这正是神经形态硬件在能耗上取胜的根本原因。`;"""),

    # ── Holter 卡片 ──
    ('<div class="hc-label">SNN on Loihi (neuromorphic)</div>',
     '<div class="hc-label">SNN 跑在 Loihi 上（神经形态）</div>'),
    ('<div class="hc-label">CNN on dense ASIC</div>',
     '<div class="hc-label">CNN 跑在稠密 ASIC 上</div>'),
    ('<div class="hc-label">The tradeoff</div>', '<div class="hc-label">这笔权衡</div>'),
    ("""          CR2032 coin cell lasts <strong>${snnDays.toFixed(0)} days</strong> of continuous
          inference (${fmtJ(snnE_pj)} × 8,640/day = ${(snnJ_day*1000).toFixed(0)} mJ/day)""",
     """          一颗 CR2032 纽扣电池可支撑 <strong>${snnDays.toFixed(0)} 天</strong>连续推理
          （${fmtJ(snnE_pj)} × 8,640 次/天 = ${(snnJ_day*1000).toFixed(0)} mJ/天）"""),
    ("""          Same CR2032 lasts only <strong>${cnnHours.toFixed(0)} hours</strong>
          (${fmtJ(cnnE_pj)} × 8,640/day = ${cnnJ_day.toFixed(1)} J/day)""",
     """          同一颗 CR2032 只能撑 <strong>${cnnHours.toFixed(0)} 小时</strong>
          （${fmtJ(cnnE_pj)} × 8,640 次/天 = ${cnnJ_day.toFixed(1)} J/天）"""),
    ("""          AUROC gap: ${(snn.macro_auroc*100).toFixed(1)}% SNN vs ${(cnn.macro_auroc*100).toFixed(1)}% CNN.
          For a screening wearable that routes positives to clinical follow-up,
          this tradeoff is favorable.""",
     """          AUROC 差距：SNN ${(snn.macro_auroc*100).toFixed(1)}% 对 CNN ${(cnn.macro_auroc*100).toFixed(1)}%。
          对于把阳性者引导去做临床复查的筛查型可穿戴设备而言，
          这笔权衡是划算的。"""),

    # ── 对比表表头 ──
    ('const hdrs=["Model","AUROC","AUPRC","Params","MACs","Spike rate","Eff. SOPs","Latency","Energy / inf"];',
     'const hdrs=["模型","AUROC","AUPRC","参数量","MAC 数","发放率","等效 SOP","延迟","单次能耗"];'),

    # ── 状态徽标 ──
    ('el.innerHTML=`<div class="badge warn">no checkpoint</div>`;',
     'el.innerHTML=`<div class="badge warn">无检查点</div>`;'),
    ('el.innerHTML=`<div class="badge ok"><span class="dot"></span>ep ${epoch} · val-AUROC ${auroc}</div>`;',
     'el.innerHTML=`<div class="badge ok"><span class="dot"></span>第 ${epoch} 轮 · 验证 AUROC ${auroc}</div>`;'),

    # ── 下拉框 / 提示 ──
    ('sel.innerHTML=`<option value="">— ${EXAMPLES.length} test recordings —</option>`+',
     'sel.innerHTML=`<option value="">— ${EXAMPLES.length} 条测试记录 —</option>`+'),
    ('$("runHint").textContent="examples unavailable";',
     '$("runHint").textContent="样本列表不可用";'),
    ('$("runHint").textContent="running inference…"; $("runBtn").disabled=true;',
     '$("runHint").textContent="推理中…"; $("runBtn").disabled=true;'),
    ('$("runHint").textContent="uploading…"; $("runBtn").disabled=true;',
     '$("runHint").textContent="上传中…"; $("runBtn").disabled=true;'),
    ('$("runHint").textContent=`error: ${e.message}`;',
     '$("runHint").textContent=`错误：${e.message}`;'),
    ('    $("runHint").textContent=`error: ${e.message}`;',
     '    $("runHint").textContent=`错误：${e.message}`;'),

    # ── 训练曲线图例与元信息 ──
    ('{col:"#48cae4", label:"macro AUROC", dash:false},', '{col:"#48cae4", label:"宏平均 AUROC", dash:false},'),
    ('{col:"#3ddc97", label:"best", dash:true},', '{col:"#3ddc97", label:"最优", dash:true},'),
    ('{col:"#ffc44d", label:"train loss", dash:false},', '{col:"#ffc44d", label:"训练损失", dash:false},'),
    ('`<span class="mono">epoch ${maxEp} · val macro-AUROC ${cur.toFixed(4)} · best ${best.toFixed(4)} · ${hist.length} epochs</span>`;',
     '`<span class="mono">第 ${maxEp} 轮 · 验证宏平均 AUROC ${cur.toFixed(4)} · 最优 ${best.toFixed(4)} · 共 ${hist.length} 轮</span>`;'),

    # ── 侧边导航 ──
    ('{ id:"top",      label:"Top"      },', '{ id:"top",      label:"顶部"    },'),
    ('{ id:"problem",  label:"Problem"  },', '{ id:"problem",  label:"问题"    },'),
    ('{ id:"insight",  label:"Insight"  },', '{ id:"insight",  label:"思路"    },'),
    ('{ id:"data",     label:"Data"     },', '{ id:"data",     label:"数据"    },'),
    ('{ id:"network",  label:"Network"  },', '{ id:"network",  label:"网络"    },'),
    ('{ id:"demo",     label:"Demo"     },', '{ id:"demo",     label:"演示"    },'),
    ('{ id:"results",  label:"Results"  },', '{ id:"results",  label:"结果"    },'),
    ('{ id:"energy",   label:"Energy"   },', '{ id:"energy",   label:"能耗"    },'),
    ('{ id:"training", label:"Training" },', '{ id:"training", label:"训练"    },'),

    # ── ECG 时间轴刻度 ──
    ('ctx.fillText(s+"s", x+2, h-4);', 'ctx.fillText(s+" 秒", x+2, h-4);'),
]


# CSS：只改两个字体变量，把中文字体插到泛型兜底之前。
# 改 :root 变量而不是逐个选择器 —— 页面里 body 用 var(--font)、大量 UI 用 var(--mono)，
# 改这两处即全站生效；Latin 字形仍取列表首位的 Manrope / Space Mono，
# 中文字形按「逐字形回退」落到雅黑/苹方，观感与页面设计一致。
PTBXL_CSS = [
    ('--font: "Manrope", ui-sans-serif, -apple-system, "Helvetica Neue", sans-serif;',
     '--font: "Manrope", "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", "Source Han Sans SC", ui-sans-serif, -apple-system, "Helvetica Neue", sans-serif;'),
    ('--mono: "Space Mono", ui-monospace, "SF Mono", monospace;',
     '--mono: "Space Mono", "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", ui-monospace, "SF Mono", monospace;'),
]


# ══════════════════════════════════════════════════════════════════════
# MIT-BIH 面板
# ══════════════════════════════════════════════════════════════════════

MITBIH_HTML = [
    ('<html lang="en">', '<html lang="zh-CN">'),
    ('<title>CardioSpike — Spiking Neural Networks for ECG Arrhythmia Detection</title>',
     '<title>CardioSpike — 脉冲神经网络心电心律失常检测</title>'),
    ('<span class="brand-sub">Spiking Neural Networks for ECG Arrhythmia Detection</span>',
     '<span class="brand-sub">面向心电心律失常检测的脉冲神经网络</span>'),

    ('<div class="stat-lab">overall acc</div>', '<div class="stat-lab">总体准确率</div>'),
    ('<div class="stat-lab">macro-F1</div>', '<div class="stat-lab">宏平均 F1</div>'),
    ('<div class="stat-lab">syn-op savings</div>', '<div class="stat-lab">突触操作节省</div>'),
    ('<div class="stat-lab">parameters</div>', '<div class="stat-lab">参数量</div>'),
    ('<div class="stat-lab">test beats</div>', '<div class="stat-lab">测试心拍数</div>'),

    ('<div class="kicker">Trained on MIT-BIH Arrhythmia Database · inter-patient DS1→DS2 split (de Chazal et&nbsp;al., 2004)</div>',
     '<div class="kicker">训练于 MIT-BIH 心律失常数据库 · 病人不交叉 DS1→DS2 划分（de Chazal et&nbsp;al., 2004）</div>'),
    ('<h1>A brain-inspired model that reads a heart, spike by spike.</h1>',
     '<h1>一个类脑模型，逐脉冲地读懂心脏。</h1>'),
    ("""Continuous ECG voltage is converted into a stream of asynchronous events
    using delta modulation — the same trick a silicon retina uses for light.
    Those events propagate through a stack of leaky integrate-and-fire neurons
    trained with surrogate-gradient BPTT. The model's hidden layers fire on
    only a small fraction of timesteps, so on neuromorphic hardware it would
    burn a fraction of the energy a dense CNN does to make the same call.""",
     '连续的心电电压通过 Δ 调制被转换成一路异步事件流——与硅视网膜处理光的方式同源。'
     '这些事件在一叠泄漏积分点火神经元中传播，网络用替代梯度 BPTT 训练。'
     '模型的隐藏层只在很小一部分时间步上发放，因此在神经形态硬件上，'
     '做出同样的判断只需消耗稠密 CNN 的一小部分能量。'),

    ('<h2>Live ECG stream → spikes → prediction</h2>', '<h2>实时心电流 → 脉冲 → 预测</h2>'),
    ("""<div class="card-sub">Replay of MIT-BIH record <span class="mono">200</span> (DS2 test set). Every visual element is driven by the trained SNN — no animation tricks.</div>""",
     '<div class="card-sub">回放 MIT-BIH 记录 <span class="mono">200</span>（DS2 测试集）。'
     '每一个视觉元素都由训练好的 SNN 驱动——没有任何动画特效。</div>'),
    ('<button id="playBtn" class="btn primary">⏸ Pause</button>',
     '<button id="playBtn" class="btn primary">⏸ 暂停</button>'),
    ('<label class="speed">speed <input id="speedSlider" type="range" min="0.3" max="3" step="0.1" value="1.2" /></label>',
     '<label class="speed">速度 <input id="speedSlider" type="range" min="0.3" max="3" step="0.1" value="1.2" /></label>'),

    ('<div class="lane-label">ECG (mV, normalized)</div>',
     '<div class="lane-label">心电图（mV，已归一化）</div>'),
    ('<div class="lane-label">Δ-encoded input spikes &nbsp;<span class="legend on">●</span>ON <span class="legend off">●</span>OFF</div>',
     '<div class="lane-label">Δ 编码输入脉冲 &nbsp;<span class="legend on">●</span>ON <span class="legend off">●</span>OFF</div>'),
    ('<div class="lane-label">Hidden layer 3 — 64 conv-LIF channels</div>',
     '<div class="lane-label">隐藏层 3 —— 64 个卷积 LIF 通道</div>'),
    ('<div class="lane-label">Hidden layer 4 — 128 dense-LIF neurons (evidence accumulation)</div>',
     '<div class="lane-label">隐藏层 4 —— 128 个稠密 LIF 神经元（证据累积）</div>'),
    ('<div class="lane-label">Beat classification (live)</div>',
     '<div class="lane-label">心拍分类（实时）</div>'),

    ('<h2>Per-class anatomy</h2>', '<h2>逐类解剖</h2>'),
    ("""<div class="card-sub">Pick an AAMI class. We show a real DS2 test beat, its delta-encoded input, the spike raster of conv layer 3, and the output-neuron membrane potentials integrating to a decision.</div>""",
     '<div class="card-sub">选择一个 AAMI 类别。我们会展示一个真实的 DS2 测试心拍、它的 Δ 编码输入、'
     '卷积层 3 的脉冲栅格，以及输出神经元积分得出判断的膜电位轨迹。</div>'),
    ('<div class="anat-label">Raw ECG (260 samples ≈ 722 ms)</div>',
     '<div class="anat-label">原始心电图（260 采样点 ≈ 722 ms）</div>'),
    ('<div class="anat-label">Δ-encoded spikes (ON = red, OFF = blue)</div>',
     '<div class="anat-label">Δ 编码脉冲（ON = 红，OFF = 蓝）</div>'),
    ('<div class="anat-label">Conv-LIF layer 3 spike raster · 64 channels × 32 t-steps</div>',
     '<div class="anat-label">卷积 LIF 层 3 脉冲栅格 · 64 通道 × 32 时间步</div>'),
    ('<div class="anat-label">Output neuron membrane potential (logit accumulation, T<sub>dense</sub> steps)</div>',
     '<div class="anat-label">输出神经元膜电位（logit 累积，T<sub>dense</sub> 步）</div>'),

    ('<h2>Confusion matrix · DS2 hold-out</h2>', '<h2>混淆矩阵 · DS2 留出集</h2>'),
    ('<div class="card-sub">Rows = true class, columns = predicted. Diagonal weight is recall; off-diagonals show where mistakes go.</div>',
     '<div class="card-sub">行 = 真实类别，列 = 预测类别。对角线深浅代表召回率，非对角线显示错误去向。</div>'),
    ('<h2>Synaptic-operation budget</h2>', '<h2>突触操作预算</h2>'),
    ('<div class="card-sub">Per beat, MAC-equivalent operations. The SNN cost is gated by measured input spike rates per layer — the speedup an event-driven chip would see.</div>',
     '<div class="card-sub">每心拍的 MAC 等效运算量。SNN 的开销由逐层实测的输入发放率决定——'
     '这正是事件驱动芯片所能获得的加速。</div>'),

    ('<h2>Why this is a spiking model, not just a CNN with extra steps</h2>',
     '<h2>为什么这是脉冲模型，而不只是多加了几步的 CNN</h2>'),
    ("""<li><strong>Event-driven input.</strong> A delta-modulation encoder turns
      analog mV into binary {ON, OFF} events. Quiet ECG segments emit zero
      events; QRS complexes emit dense bursts. This is the same input
      modality a real silicon-retina / cochlea pushes onto neuromorphic chips.</li>""",
     '<li><strong>事件驱动输入。</strong>Δ 调制编码器把模拟毫伏量转成二值 {ON, OFF} 事件。'
     '安静的心电段不产生事件；QRS 波群产生密集爆发。'
     '这与真实硅视网膜 / 耳蜗送入神经形态芯片的输入模态完全一致。</li>'),
    ("""<li><strong>Leaky integrate-and-fire neurons everywhere.</strong> Every
      hidden unit has a membrane potential <em>V(t) = β·V(t−1) + I(t)</em>;
      it fires when V crosses 1 and resets. There is no ReLU anywhere in
      the network.</li>""",
     '<li><strong>处处都是泄漏积分点火神经元。</strong>每个隐藏单元都有膜电位 '
     '<em>V(t) = β·V(t−1) + I(t)</em>；当 V 越过 1 时发放并复位。'
     '网络中不存在任何 ReLU。</li>'),
    ("""<li><strong>Surrogate-gradient BPTT.</strong> The non-differentiable
      spike function is bypassed during backprop using a fast-sigmoid
      surrogate, exactly as in <em>Neftci et al. 2019</em>. Time is folded
      back through the unrolled network on the backward pass.</li>""",
     '<li><strong>替代梯度 BPTT。</strong>反向传播时用 fast-sigmoid 替代函数绕过不可导的脉冲函数，'
     '与 <em>Neftci et al. 2019</em> 完全一致。反向传播时时间维度沿展开的网络折叠回去。</li>'),
    ("""<li><strong>Sparse hidden activity.</strong> Measured spike rates per
      layer are reported above — they are well below 50%, which is the
      regime where neuromorphic silicon (Loihi, SpiNNaker) wins on energy.</li>""",
     '<li><strong>稀疏的隐藏层活动。</strong>上文报告了逐层实测发放率——远低于 50%，'
     '这正是神经形态芯片（Loihi、SpiNNaker）在能耗上取胜的区间。</li>'),
    ("""<li><strong>Honest evaluation.</strong> Patients in the training set
      never appear in the test set. This is the AAMI-recommended
      inter-patient protocol; intra-patient scores in the literature are
      a known overestimate.</li>""",
     '<li><strong>诚实的评测。</strong>训练集中的患者绝不会出现在测试集里。'
     '这是 AAMI 推荐的病人不交叉协议；文献中的病人内（intra-patient）分数是已知的高估。</li>'),

    ("""<footer>
  Built with <span class="mono">snntorch</span> + <span class="mono">PyTorch</span> · trained on Apple Silicon MPS · data: MIT-BIH Arrhythmia Database (PhysioNet).
</footer>""",
     """<footer>
  构建于 <span class="mono">snntorch</span> + <span class="mono">PyTorch</span> · 训练于 Apple Silicon MPS · 数据：MIT-BIH Arrhythmia Database (PhysioNet)。
</footer>"""),
]


MITBIH_JS = [
    # 类别全名来自 data/inference.json（evaluate.py 产出），不在前端硬编码。
    # 直接改 JSON 会破坏「数据由脚本生成」的可复现性，故在前端加一层映射覆盖。
    ('let SUMMARY = null, STRIP = null, INF = null;',
     'const CLASS_CN = { N:"正常心拍", S:"室上性异位", V:"室性异位", F:"融合心拍", Q:"未知 / 起搏" };\n\nlet SUMMARY = null, STRIP = null, INF = null;'),
    ('<span>${c} · ${INF.class_names[c]}</span>',
     '<span>${c} · ${CLASS_CN[c] || INF.class_names[c]}</span>'),

    ("${correct ? '✓ correct' : '✗ wrong'}", "${correct ? '✓ 正确' : '✗ 错误'}"),
    ('<span class="badge dim">record ${ex.record_id}</span>',
     '<span class="badge dim">记录 ${ex.record_id}</span>'),
    ("""    <span>True <strong style="color:${CLASS_COLORS[ex.true]}">${ex.true}</strong> ·
         Predicted <strong style="color:${CLASS_COLORS[ex.pred]}">${ex.pred}</strong>
         (p=${(ex.probs[ex.pred]*100).toFixed(1)}%) ·
         runner-up <strong style="color:${CLASS_COLORS[second[0]]}">${second[0]}</strong>
         (p=${(second[1]*100).toFixed(1)}%)</span>""",
     """    <span>真实 <strong style="color:${CLASS_COLORS[ex.true]}">${ex.true}</strong> ·
         预测 <strong style="color:${CLASS_COLORS[ex.pred]}">${ex.pred}</strong>
         （概率 ${(ex.probs[ex.pred]*100).toFixed(1)}%） ·
         次优 <strong style="color:${CLASS_COLORS[second[0]]}">${second[0]}</strong>
         （概率 ${(second[1]*100).toFixed(1)}%）</span>"""),

    ('<span class="pred-tag" style="color:${CLASS_COLORS[truth] || \'#888\'}">truth: ${truth}</span>',
     '<span class="pred-tag" style="color:${CLASS_COLORS[truth] || \'#888\'}">真实：${truth}</span>'),
    ('<span class="pred-tag" style="color:${CLASS_COLORS[winner] || \'#888\'}">model: ${winner}</span>',
     '<span class="pred-tag" style="color:${CLASS_COLORS[winner] || \'#888\'}">模型：${winner}</span>'),

    ('`pred ${c}`', '`预测 ${c}`'),
    ('<div class="cm-cell cm-hdr">recall</div>', '<div class="cm-cell cm-hdr">召回率</div>'),
    ('`true ${classes[i]}`', '`真实 ${classes[i]}`'),

    ("""      `SNN cost is <strong>${SUMMARY.energy_ratio_ann_over_snn.toFixed(2)}×</strong> cheaper per beat than the dense baseline (mean hidden spike rate ${(SUMMARY.mean_hidden_spike_rate*100).toFixed(1)}%).`;""",
     """      `SNN 每心拍开销比稠密基线<strong>低 ${SUMMARY.energy_ratio_ann_over_snn.toFixed(2)} 倍</strong>（隐藏层平均发放率 ${(SUMMARY.mean_hidden_spike_rate*100).toFixed(1)}%）。`;"""),
    ('<div class="energy-name" style="text-align:center">dense baseline (ANN)</div>',
     '<div class="energy-name" style="text-align:center">稠密基线（ANN）</div>'),
    ('<div class="energy-name" style="text-align:center">spiking, measured</div>',
     '<div class="energy-name" style="text-align:center">脉冲模型（实测）</div>'),
    ('<div class="energy-name" style="text-align:right">SNN / ANN</div>',
     '<div class="energy-name" style="text-align:right">SNN / ANN</div>'),
    ("""    `Per-beat SNN cost is <strong>${ratio.toFixed(2)}×</strong> below the dense baseline. ` +
    `This is the speedup an event-driven neuromorphic chip (Loihi, SpiNNaker) would see — ` +
    `silicon counts MAC operations on actual spikes, not on zeros.`;""",
     """    `SNN 每心拍开销比稠密基线<strong>低 ${ratio.toFixed(2)} 倍</strong>。` +
    `这正是事件驱动神经形态芯片（Loihi、SpiNNaker）所能获得的加速——` +
    `芯片只对真实脉冲计 MAC 运算，不计零。`;"""),

    ('e.target.textContent = live.playing ? "⏸ Pause" : "▶ Play";',
     'e.target.textContent = live.playing ? "⏸ 暂停" : "▶ 播放";'),
]


MITBIH_CSS = [
    ('--font: ui-sans-serif, -apple-system, "Inter", "Helvetica Neue", sans-serif;',
     '--font: ui-sans-serif, -apple-system, "Inter", "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", "Helvetica Neue", sans-serif;'),
    ('--mono: "JetBrains Mono", ui-monospace, "SF Mono", monospace;',
     '--mono: "JetBrains Mono", "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", ui-monospace, "SF Mono", monospace;'),
]


TABLES = {
    "ptbxl": {"index.html": PTBXL_HTML, "app.js": PTBXL_JS, "style.css": PTBXL_CSS},
    "mitbih": {"index.html": MITBIH_HTML, "app.js": MITBIH_JS, "style.css": MITBIH_CSS},
}

# 扫残留英文时的白名单：模型 / 数据集 / 硬件 / 文献名、指标缩写、AAMI 代号、
# 单位、颜色、CSS 类名与选择器。
ALLOW = {
    "AUROC", "AUPRC", "ROC", "PR", "MAC", "MACs", "SOP", "SOPs", "LIF", "SNN", "SNNs",
    "CNN", "CNN1D", "ResNet", "ResNet1D", "TCN", "PTB", "XL", "AAMI", "DS1", "DS2",
    "MIT-BIH", "Loihi", "SpiNNaker", "Horner", "Horowitz", "ISSCC", "PyTorch", "snntorch",
    "AdamW", "EMA", "Modal", "MPS", "ASIC", "CMOS", "GPU", "LED", "CR2032", "Holter",
    "FPR", "TPR", "NORM", "MI", "STTC", "CD", "HYP", "ON", "OFF", "Inter", "Manrope",
    "Space", "Mono", "JetBrains", "YaHei", "PingFang", "Source", "Han", "Neftci",
    "html", "lang", "css", "js", "json", "npy", "http", "https", "physionet", "org",
    "style", "app", "data", "index", "px", "mV", "Hz", "ms", "µJ", "pJ", "pp", "Sigmoid",
    "sigmoid", "softmax", "ReLU", "BPTT", "NaN", "B", "T", "N", "S", "V", "F", "Q",
    "NeuroCardio", "CardioSpike", "S-TCN", "S", "TCN", "Chazal", "al",
}


def _pat(en: str) -> re.Pattern:
    """把英文串编译成「空白弹性」正则：词间允许任意空白（含换行缩进）。"""
    toks = en.split()
    return re.compile(r"\s+".join(re.escape(t) for t in toks))


def apply_table(text: str, pairs) -> tuple[str, list[str], int]:
    """按译文表替换。返回 (新文本, 未命中的英文, 命中总数)。

    **按 token 数降序应用**：长条目先落，避免短条目抢先吃掉长条目的一部分
    （例：``truth`` 与 ``<div>truth</div>`` 并存时，先匹配带标签的那个）。

    用函数式 ``subn`` 而非字符串替换式：函数返回的字符串**不做反斜杠转义处理**，
    中文译文里若出现 ``\\`` 不会被二次解释（字符串式替换会把 ``\\n`` 当成转义）。
    """
    ordered = sorted(pairs, key=lambda kv: -len(kv[0].split()))
    missing, hits = [], 0
    for en, cn in ordered:
        pat = _pat(en)
        text, n = pat.subn(lambda _m, c=cn: c, text)
        if n == 0:
            missing.append(en)
        hits += n
    return text, missing, hits


def scan_leftover(text: str, kind: str) -> list[str]:
    """扫出疑似残留英文（供人工确认，不判失败）。"""
    if kind == "html":
        body = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
        body = re.sub(r"<(script|style)\b.*?</\1>", " ", body, flags=re.S | re.I)
        body = re.sub(r"<[^>]+>", " ", body)
        cands = re.findall(r"[A-Za-z][A-Za-z'’\-]{3,}(?:\s+[A-Za-z][A-Za-z'’\-]{2,})+", body)
    elif kind == "js":
        body = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
        body = re.sub(r"^\s*//.*$", " ", body, flags=re.M)
        cands = []
        for lit in re.findall(r"\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'|`(?:[^`\\]|\\.)*`", body):
            inner = lit[1:-1]
            if any("一" <= ch <= "鿿" for ch in inner):
                continue
            if re.search(r"^[#\d]", inner) or len(inner) < 6:
                continue
            cands.append(inner)
    else:
        return []

    out, seen = [], set()
    for c in cands:
        toks = [t for t in re.split(r"[^A-Za-z'’\-]+", c) if len(t) >= 3 and t not in ALLOW]
        if not toks:
            continue
        c = re.sub(r"\s+", " ", c).strip()
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def build(panel: str, src_dir: Path, out_dir: Path, check_only: bool) -> dict:
    files = TABLES[panel]
    result = {"panel": panel, "src": str(src_dir), "out": str(out_dir),
              "files": [], "missing": [], "leftover": []}

    if not src_dir.is_dir():
        print(f"[错误] 源目录不存在：{src_dir}", file=sys.stderr)
        sys.exit(2)

    if not check_only:
        out_dir.mkdir(parents=True, exist_ok=True)

    for name, pairs in files.items():
        src = src_dir / name
        if not src.is_file():
            print(f"[错误] 缺源文件：{src}", file=sys.stderr)
            sys.exit(2)
        text = src.read_text(encoding="utf-8")
        new, missing, hits = apply_table(text, pairs)
        kind = "html" if name.endswith(".html") else ("js" if name.endswith(".js") else "css")
        leftover = scan_leftover(new, kind) if kind != "css" else []

        result["files"].append({"name": name, "pairs": len(pairs), "hits": hits,
                                "missing": missing, "leftover": leftover})
        for m in missing:
            result["missing"].append(f"{name}: {m[:80]}")
        result["leftover"] += [f"{name}: {x}" for x in leftover]

        if not check_only:
            (out_dir / name).write_text(new, encoding="utf-8")

    # MIT-BIH 的数据目录（summary/strip/inference/sparsity.json）用软链带过来，
    # 不复制 —— 数据由 evaluate.py 生成，前端只是消费者，复制会造成两份真相。
    if panel == "mitbih" and not check_only:
        link = out_dir / "data"
        target = src_dir / "data"
        if link.is_symlink() or link.exists():
            link.unlink() if link.is_symlink() else None
        if not link.exists():
            link.symlink_to(target, target_is_directory=True)
        result["data_link"] = f"{link} -> {target}"

    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="中文化两套演示前端")
    ap.add_argument("--src-ptbxl", default=str(NC / "web"))
    ap.add_argument("--src-mitbih", default=str(ROOT / "results/mitbih/web_live"))
    ap.add_argument("--out-ptbxl", default=str(ROOT / "results/web_cn"))
    ap.add_argument("--out-mitbih", default=str(ROOT / "results/mitbih/web_cn"))
    ap.add_argument("--check-only", action="store_true",
                    help="只校验译文表能否全部命中，不写文件")
    args = ap.parse_args()

    jobs = [("ptbxl", Path(args.src_ptbxl), Path(args.out_ptbxl)),
            ("mitbih", Path(args.src_mitbih), Path(args.out_mitbih))]

    reports = [build(p, s, o, args.check_only) for p, s, o in jobs]

    print("=" * 68)
    for r in reports:
        print(f"\n【{r['panel']}】{r['src']}\n    → {r['out']}")
        for f in r["files"]:
            flag = "OK " if not f["missing"] else "!! "
            print(f"  {flag}{f['name']:<12} 条目 {f['pairs']:>3}  命中 {f['hits']:>3}"
                  f"  未命中 {len(f['missing'])}")
        if r.get("data_link"):
            print(f"  data 软链：{r['data_link']}")

    bad = [m for r in reports for m in r["missing"]]
    if bad:
        print(f"\n[未命中的条目] 共 {len(bad)} 条 —— 上游文案可能已变，需更新译文表：")
        for m in bad:
            print(f"  · {m}")
    else:
        print("\n[译文表] 全部条目均已命中上游原文。")

    left = [x for r in reports for x in r["leftover"]]
    if left:
        print(f"\n[残留英文候选] 共 {len(left)} 条（白名单外的英文串，需人工确认"
              f"是否属于应译未译）：")
        for x in left:
            print(f"  ? {x}")

    if args.check_only:
        print("\n（--check-only：未落盘）")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
