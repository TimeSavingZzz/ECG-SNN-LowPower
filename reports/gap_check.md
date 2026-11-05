# 缺口盘点：5 项核对 + 调整方案

> 本文件**不产生任何新计算**，只核对已有产物覆盖了哪些格子。
> 盘点日期：2026-11-05。核对手段：读取容器内既有 JSON / 脚本 / 目录清单。

---

## 1. 五项核对结果

| # | 核对项 | 状态 | 已有产物 | 缺口 |
|---|---|---|---|---|
| 1 | **2×2 四格**（6块delta / 3块delta / 3块整流 / **6块整流**） | **三格全、第四格缺一半** | 6块整流 = `ab_polarity`：十折 AUROC（`results/p10_summary.json → fold_table.ab_polarity.per_fold_auroc`）、逐层剖面（`results/p0_aggregate.json → profile.ab_polarity`）、MAC/SOP（`results/p0_4_cost.json`）、十折权重（`results/repro_ab_polarity_fold1..9` + fold10 原消融目录） | **缺 θ 扫描**。E5 只扫了 `ab_shallow_pol` / `ab_shallow` 两个 3 块配置 |
| 2 | **逐层有效秩 L1–L6** | **已有（十折 mean±std）** | `results/p0_aggregate.json` 的 `profile[cfg].splits.test.reff` 与 `reff_sd`（4 配置 × 6 层）；`reports/E7_p02_p03_tables.md` 表 B 已表化 | 无 |
| 3 | **230× 的折间范围 / CI** | **缺（可零计算得到）** | 点估计在 `results/comparison.json`（230.3495）；折间数据在 `results/block5_stats_10fold.json → rows[*].total_sops`（十折 39.09–50.27 M） | 逐折比值未算。**注意口径冲突**：十折平均 42.33 M ⇒ 比值 ≈ **284.6×**；部署权重 52.31 M ⇒ **230.35×**。两个数不能混用 |
| 4 | **不同 θ 的推理延迟** | **已有（零计算可出表）** | `results/pareto/theta{θ}_T{T}.json` 每个都含 `latency_ms`、`latency_n_iter`、`latency_device`、`test_inference_seconds` | 无（只需并表） |
| 5 | **MAC/参数表与修正后 SOP 同口径** | **部分** | `results/p0_4_cost.json`：`params` / `macs` 正确可用；`sops` 是**未修正**值 | `stem_conv` 对整流前端按 `input_rate` 打折是错的，需重算 SOP 列。**整流配置有两个**（`ab_polarity`、`ab_shallow_pol`），不是一个 |

---

## 2. 缺口定位（要动哪里）

| 缺口 | 需要什么 | 是否要训练 |
|---|---|---|
| G1 6块整流 θ 扫描 | 10 折权重已在；跑 6 档 θ × 10 折 + T 切片 | **否**，纯推理 |
| G2 230× 折间区间 | 用 `block5_stats_10fold.json` 的 `total_sops` 除以常数稠密能耗 1.2050 mJ | **否**，零计算 |
| G3 SOP 修正十折 | 两个整流配置各 10 折重算 `cost_snn` 并把 `stem_conv.sops := macs` | **否**，纯推理 |
| G4 θ 联合表（含延迟） | 并 30 个 `theta*_T*.json` 的 `macro_auroc` / `total_sops` / `energy_pj_neuromorphic` / `latency_ms` | **否**，零计算 |
| G5 噪声底叠图 | b38 与 E3 都在 **calib fold 9**；θ 曲线（`pareto/*.json`）的发放率也标定在 **fold 9**（`11_pareto_sweep.py:200-202`，batch 32，K=8），与 b38 逐字同口径 ⇒ **可直接叠，零推理** | **否**，0 计算 |
| G6 消融累积表 | `p10_summary.json` 四配置逐折 + `p11_compare.json` 的 p 值 | **否**，零计算 |

---

## 3. 四项必须调整（原清单的问题）

**① 第四格不止「θ 扫描」，还要进 SOP 修正。**
原清单第 3 项只提"stem_conv 按稠密计费"，但整流前端有 `ab_polarity` 与 `ab_shallow_pol` 两个，两个都得修，否则 2×2 里的两格口径不一致。

**②（已更正）噪声底叠图确实是零计算。**
先前记成"θ 曲线测在 test fold 10"是错的。核对 `11_pareto_sweep.py:197-202`：AUROC 在 fold10 测试集上前向，但**发放率/SOP 的标定在 fold-9 验证集，batch_size=32，前 8 个 batch**——与 b38（`calib_fold=9`、`calib_batch_size=32`、8-batch bootstrap）、E3、E5/46 完全同一协议。因此噪声带与 θ 曲线可以直接叠，**不需要任何新推理**。

仍有一处需要写明：b38 的 6 块 SOP-K 与 E1 的 θ 曲线 SOP 来自**同一个部署权重 `d01e7aea`、同一个 fold9 标定集**，但前者是"重抽 8 个标定 batch"的标定噪声，后者是 θ 变化带来的真实位移；图上应把两者区分成"噪声带"与"曲线"，不能把噪声带当成置信区间。

**③ 230× 的折间区间会与头号数字冲突，需先定口径。**
- 部署权重（`d01e7aea`，fold10 测试集）：SOP 52.31 M ⇒ **230.35×**（已进摘要）。
- 十折各一个权重：SOP 39.09–50.27 M，平均 42.33 M ⇒ **≈284.6×**（区间约 240–308×）。
两者不是同一件事（单权重 vs 十个权重）。**建议**：正文主句保留 230×，折间区间作为"十折重现的敏感性"另起一句，并写明"十折平均模型比部署模型更稀疏，故倍率更高"。不得让两个数字并列出现而不解释。

**④ 消融累积表必须带 p 值，且 n=3 的结论已被 n=10 推翻。**
链条是两步**不同维度**：6块delta → 6块整流（换编码器）→ 3块整流（缩深度）。

- **n=3（011/012）**：三处增益均未越过噪声（`ab_shallow` p=0.229、`ab_polarity` p=0.448、`ab_shallow_pol` p=0.203）。这是旧结论。
- **n=10（`p10_summary.json`，Nadeau–Bengio + Holm）**：四个主比较**全部显著**——
  `ab_polarity−ab_full` +0.0090 [+0.0016,+0.0164] p=0.023（Holm 0.023，8/10 折胜）；
  `ab_shallow−ab_full` +0.0103 [+0.0047,+0.0160] p=0.0025（Holm 0.0075，9/10）；
  `ab_shallow_pol−ab_full` +0.0172 [+0.0102,+0.0242] p=0.0004（Holm 0.0016，10/10）；
  `ab_shallow_pol−ab_shallow` +0.0068 [+0.0026,+0.0111] p=0.0054（Holm 0.0108，10/10）。

⇒ 表里必须同时写 n 与 p，并注明"n=3 不显著、n=10 显著"的原因（自由度从 2 变 9，CI 从 t₀.₉₇₅,₂=4.303 缩到约 2.26）。**不得**再沿用"增益未越过噪声"的说法。

---

## 4. 执行清单（按依赖排序）

| 步 | 内容 | 性质 | 预计 | 产物 |
|---|---|---|---|---|
| 1 | 本文件 | 0 计算 | — | `reports/gap_check.md` |
| 2 | SOP 修正十折（`ab_polarity` + `ab_shallow_pol`，含 `ab_full`/`ab_shallow` 对照） | 纯推理 | ~20 min | `results/sop_corrected_10fold.json`、`reports/E9_sop_corrected.md` |
| 3 | θ 联合表（θ / T / AUROC / SOP / E_nm / 延迟） | 0 计算 | ~10 min | `results/theta_joint_table.json`、`reports/E10_theta_joint.md` |
| 4 | 噪声底叠图（b38 8-batch bootstrap + E3 SOP-K 叠到 θ 曲线） | 0 计算（同一 fold9/K=8 协议） | ~10 min | 并入 `results/p1_supplement.json`、`reports/E10_theta_joint.md` |
| 5 | 230× 折间区间 + MAC/参数同口径表 | 0 计算 | ~15 min | `results/p1_supplement.json`、`reports/E11_p1_supplement.md` |
| 6 | 累加消融表（带 p 值） | 0 计算 | ~15 min | 并入 E11 |
| 7 | **（可选）`ab_polarity` θ 扫描**，补全 2×2 | 纯推理，**不训练** | 1–2 h | `results/theta_sweep_ab_polarity.json`、`reports/E12_ab_polarity_theta.md` |

**结论：本清单不需要任何新训练。** 唯一涉及训练的是"双极性改真二值脉冲"（两模型 × 十折 ≈ 30 h），不在本清单内。

---

## 5 执行状态（2026-11-05）

| 步 | 状态 | 产物 |
|---|---|---|
| 1 缺口盘点 | ✅ | `reports/gap_check.md`（本文） |
| 2 SOP 修正十折 | 🟡 两/三 | `results/sop_corrected_10fold.json`、`reports/E9_sop_corrected.md`：`ab_shallow_pol` 19.96→27.90 M（+40.0%），`ab_shallow` 不变；`ab_polarity` 等第 7 步 |
| 3 θ 联合表 | ✅ | `results/p1_supplement.json → theta_joint`、`reports/E10_theta_joint.md` |
| 4 噪声底叠图 | ✅（更正为 0 计算） | `results/figs/p1_noise_overlay.png`、`reports/E10_theta_joint.md` §2 |
| 5 230× 区间 + MAC 同口径表 | ✅ | `results/p1_supplement.json → ratio_230x / mac_table`、`reports/E11_p1_supplement.md` |
| 6 累加消融表（带 p） | ✅ | `reports/E11_p1_supplement.md` §3 |
| 7 `ab_polarity` θ 扫描 | 🟡 运行中 | 4 分片，只跑 T_dense=8（6 θ × 10 折），不训练 |

**口径修正（执行中发现，已回改本文件）**

- §2 的 ②：θ 曲线与 b38 同为 fold9 / batch32 / 8-batch，叠图零计算。
- §3 的 ④：n=3 的"未越过噪声"已被 n=10 推翻，四条主比较 Holm 后全显著。
- 新增：`45` 号脚本原先的"倍率"列漏乘 SOP 的 0.1 pJ，已修（`ab_shallow_pol` 的
  "27.5× → 19.6×"是错的，正确为 **274.7× → 196.0×**）。
