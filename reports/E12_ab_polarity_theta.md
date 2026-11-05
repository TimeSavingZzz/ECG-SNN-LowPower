# E12 `ab_polarity`（6 块 + 整流前端）θ 扫描 —— 2×2 第四格

> 产物：`results/theta_sweep_ab_polarity.json`（md5 `b44fa083`，60 行 = 6 θ × 10 折）。
> 脚本：`experiments/46_ab_polarity_theta.py`（4 卡分片）+ `experiments/47_merge_polarity.py`。
> **纯推理，不训练。** 所有 checkpoint 都是 θ=0.15 训练；θ 扫描为零样本灵敏度分析。

---

## 0 口径

| 项 | 值 |
|---|---|
| 配置 | `ab_polarity`：6 个 S-TCN block，前端 `dual_polarity_encode`（连续整流值，**无 θ 参数**） |
| 折 | 1–10，每折用该折测试集 |
| θ | 0.05 / 0.075 / 0.1 / 0.15 / 0.2 / 0.3 |
| T_dense | **只扫 8**（运行时间考虑；T 切片对 SOP 无影响、对 AUROC <1 pp，见 E10） |
| 标定 | val fold（fold1→10，其余 f−1），`batch_size=32`，前 8 batch |
| 权重 | 十折各自 `best.pt`（`repro_ab_polarity_fold*` + fold10 原消融目录） |

## 1 结果：θ 完全无效

| θ | AUROC | SOP (M) | input 发放率 | 修正后 SOP (M) |
|---|---|---|---|---|
| 0.05 | 0.885869 ± 0.009280 | 47.80 ± 3.93 | 0.3112 | 55.74 |
| 0.075 | 0.885869 ± 0.009280 | 47.80 ± 3.93 | 0.3112 | 55.74 |
| 0.1 | 0.885869 ± 0.009280 | 47.80 ± 3.93 | 0.3112 | 55.74 |
| 0.15 | 0.885869 ± 0.009280 | 47.80 ± 3.93 | 0.3112 | 55.74 |
| 0.2 | 0.885869 ± 0.009280 | 47.80 ± 3.93 | 0.3112 | 55.74 |
| 0.3 | 0.885869 ± 0.009280 | 47.80 ± 3.93 | 0.3112 | 55.74 |

**逐折检查：每一折的 6 个 θ 单元 AUROC 与 SOP 都逐位相同（12 位小数）**，不是"差异很小"，
是**完全不动**。编码器探针给出同一结论：

- `theta_effective = false`（10/10 折）；
- `max_abs_diff = 0.0`、`mean_abs_diff = 0.0`（θ=0.15 vs 0.30 的编码器输出完全相同）；
- 两档 θ 的发放率逐折相同（0.2995–0.3149）。

原因在代码层：`ablation.py:160` 用 `lambda x, theta=0.15: dual_polarity_encode(x)`
覆盖了 `delta_encode`，θ 根本不进前向。

## 2 2×2 四格齐全后的对照

| | Δ 编码前端 | 整流前端 |
|---|---|---|
| **6 块** | `ab_full`：SOP 随 θ **非单调**，内部最低点 θ=0.075（排序稳健 95.2%，幅值 z=1.25/0.77 在噪声内） | `ab_polarity`：θ **完全无效**（本节） |
| **3 块** | `ab_shallow`：SOP 随 θ **严格单调下降 −21.4%** | `ab_shallow_pol`：θ **完全无效**（E5） |

⇒ θ 旋钮是否存在，**只由前端编码器决定，与深度无关**：两个 Δ 配置都有（形状不同），
两个整流配置都完全没有。

## 3 SOP 修正（整流前端，`stem_conv` 稠密计费）

见 E9。`ab_polarity` 十折：

| 项 | 值 |
|---|---|
| stem MACs | 11 520 000 |
| SOP 未修正 | 47.80 ± 3.93 M |
| SOP 修正 | **55.74 ± 3.93 M**（**+16.7%**） |
| 稠密/脉冲 | 253.8× → **217.3×** |
| AUROC | 不受影响（0.8859） |

单模型口径（fold-9 标定，`p1_supplement.json`）：49.10 → **57.05 M（+16.2%）**，倍率 245.4× → 211.2×。
两个口径差 1.3 M，来源是十折测试集 vs 单一 fold-9 标定，**不得混引**。

## 4 事实层结论

1. 2×2 第四格补齐：`ab_polarity` 与 `ab_shallow_pol` 一样，θ 对输出**零影响**。
2. 6 块的非单调 SOP 现象**只在 Δ 前端出现**；换成整流前端后 θ 连入口都没有。
3. 因此"θ 是零成本旋钮"这一说法必须限定为 **Δ 编码器实现**；整流前端需要别的机制
   （或重新训练成真二值脉冲，不在本阶段范围）。
4. 分辨率限制：本配置只扫了 T_dense=8。若论文要画 T 切片，需补跑（纯推理，约 2 h/配置）。

## 5 复现

```bash
cd /mnt/ECG-SNN-LowPower
# 4 卡分片（已跑完，产物在 results/theta_pol_shard_g*.json）
python experiments/46_ab_polarity_theta.py --folds 1,2,3 --gpu 0 --t-dense 8 --out results/theta_pol_shard_g0.json
python experiments/47_merge_polarity.py
python experiments/45_sop_corrected.py
```
