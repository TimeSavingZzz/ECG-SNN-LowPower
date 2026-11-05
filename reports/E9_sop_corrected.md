# E9 SOP 修正：`stem_conv` 按稠密计费（十折）

> 产物：`results/sop_corrected_10fold.json`（md5 `30521a23`）
> 脚本：`experiments/45_sop_corrected.py`（md5 `7376fb53`）。
> **零推理**：MACs 由结构解析，SOP 取自既有多折扫描 JSON，不重新前向。
> 本文件只陈述事实，不决定论文怎么写。

---

## 0 问题

`cost_snn` 给每一层按"该层输入的脉冲发放率"打折：

```python
sops_layer = macs_layer × spike_rate_in
```

对 Δ 编码器成立（前端输出是二值脉冲）。但**整流前端** `dual_polarity_encode`
输出的是**连续张量**，`stem_conv` 的输入不是脉冲；`spike_rates["input"]`
仍是一个 0.31 左右的数（被误当成"输入发放率"），于是 stem 被打了 31% 的折。
等价发放率应为 **1.0**。

## 1 修正恒等式

只有 `stem_conv` 一层口径错，其余层都由真实脉冲驱动，因此逐折：

```
SOP_corrected = SOP_uncorrected + macs_stem − round(macs_stem × r_in)
```

- `macs_stem` 与权重无关（结构解析，两种 3 块配置都是 **11 520 000**）；
- `r_in` 取该折扫描行里的 `spike_rates.input`；
- 修正为**加性常数**：十折里 Δ 落在 7.91–7.98 M，因为 `r_in` 只在 0.305–0.315 间波动。

## 2 十折结果（θ=0.15，T_dense=8）

| 配置 | 前端 | SOP 未修正 (M) | SOP 修正 (M) | Δ (M) | 相对增幅 | 稠密/脉冲（未→修正） |
|---|---|---|---|---|---|---|
| `ab_shallow_pol` | 整流 | 19.96 ± 1.75 | **27.90 ± 1.74** | +7.93 | **+40.0%**（34.0–43.6%） | 274.7× → **196.0×** |
| `ab_shallow` | Δ | 18.17 ± 1.05 | 18.17 ± 1.05 | 0 | 0 | 300.8× → 300.8× |
| `ab_polarity` | 整流 | 47.80 ± 3.93 | **55.74 ± 3.93** | +7.94 | **+16.7%**（15.2–20.2%） | 253.8× → **217.3×** |

- `ab_shallow` 的逐折 Δ 恒为 **0**，验证了修正只作用于整流前端。
- `ab_shallow_pol` 的 AUROC 不受影响（修正只改计费，不改前向）：0.8939 ± 0.0082。
- 3 块整流相对 3 块 Δ 的**能效劣势**从 −8.7%（274.7 vs 300.8）扩大到 **−34.8%**
  （196.0 vs 300.8）。修正前"整流只贵一点"的印象，大部分是 stem 打折造成的；
  修正后整流前端明显更费能。

## 3 哪些数字会被这条修正改变

| 位置 | 修正前 | 修正后 |
|---|---|---|
| `ab_shallow_pol` 十折 SOP | 19.96 M | 27.90 M |
| 对应稠密/脉冲倍率 | 274.7× | 196.0× |
| 单模型（fold-9 标定）SOP | 18.54 M | 26.48 M（`p1_supplement.json`） |
| 单模型倍率 | 293.9× | 205.7× |

**不改变**：`ab_full` 的 230.35×（Δ 前端，无修正）、`ab_shallow` 的一切数字、
所有 AUROC / AUPRC / 参数量 / MACs。

## 4 一个必须记住的坑

`45` 号脚本初版的"倍率"列写成了 `dense_pj / sops`，**漏乘 SOP 的 0.1 pJ**，
于是把 274.7× 印成 27.5×。已在 `7376fb53` 版修正；`results/sop_corrected_10fold.json`
是修正后重跑的产物，**旧版输出作废**。

## 5 复现

```bash
cd /mnt/ECG-SNN-LowPower
python experiments/45_sop_corrected.py
```

输入：`theta_sweep_new_config.json`、`theta_sweep_ab_shallow.json`、
`theta_sweep_ab_polarity.json`（第 7 步产出后自动纳入）。全部既有，零推理。
