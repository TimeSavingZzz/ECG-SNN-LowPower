# E13 `ab_polarity` 的 T_dense 切片（补跑，取消 E12 的 T=8 限定）

> 产物：`results/theta_sweep_ab_polarity.json`（md5 `d907417c`，**300 行** = 6 θ × 10 折 × 5 T）、
> `results/ab_polarity_T_summary.json`（md5 `88ffb2d6`）。
> 脚本：`experiments/46_ab_polarity_theta.py`（补跑）+ `47_merge_polarity.py`（改为保留全部 T）
> + `50_ab_polarity_T_summary.py`（汇总，零计算）。
> **纯推理，不训练**；协议与 E12 完全一致（fold f 测试集，val fold 标定，batch_size=32，前 8 batch）。

---

## 0 补跑范围

上一轮全 T 扫描被中止时，**fold 1 / 4 / 7 / 9 已经跑完全部 5 个 T 切片**（各 30 行），
fold 2/3/5/6/8/10 只有 T=8。本次补齐后 **(fold, θ) 单元 60 个，T 完整 60 / 60**。

| 分片 | folds | 行数 | 说明 |
|---|---|---|---|
| `theta_pol_shard_g0.json` | 1,2,3 | 90 | 本次补 fold 2/3 的 T≠8 |
| `theta_pol_shard_g1.json` | 4,5,6 | 90 | 本次补 fold 5/6 |
| `theta_pol_shard_g2.json` | 7,8,10 | 90 | 本次补 fold 8/10 |
| `theta_pol_shard_g3.json` | 9 | 30 | 已完成，未重跑 |

墙钟约 65 min（4 卡并行，单格 45–51 s）。

## 1 θ 在每个 T 上都完全无效

逐折逐 T 检查跨 θ 的极差：

| T_dense | max\|ΔAUROC\| over θ | max\|ΔSOP\| over θ |
|---|---|---|
| 2 | **0.0** | **0** |
| 4 | **0.0** | **0** |
| 8 | **0.0** | **0** |
| 16 | **0.0** | **0** |
| 32 | **0.0** | **0** |

⇒ E12 的结论（θ 对 `ab_polarity` 零影响）**不是 T=8 的局部现象**，在全部 5 个 T 切片、
全部 10 折上成立（60 个单元 × 6 θ，逐位相同）。

## 2 T_dense 的影响（θ=0.15，十折均值）

| T_dense | AUROC | AUPRC | SOP (M) | 相对 T=8 |
|---|---|---|---|---|
| 2 | 0.8744 ± 0.0081 | 0.7191 | 47.76 ± 3.92 | −1.15 pp |
| 4 | 0.8845 ± 0.0089 | 0.7392 | 47.77 ± 3.93 | −0.14 pp |
| **8** | **0.8859 ± 0.0093** | **0.7411** | 47.80 ± 3.93 | — |
| 16 | 0.8858 ± 0.0095 | 0.7400 | 47.86 ± 3.95 | −0.01 pp |
| 32 | 0.8788 ± 0.0092 | **0.6787** | 47.99 ± 3.97 | −0.71 pp |

- **每个 θ 的跨 T AUROC 极差都是 1.14 pp，argmax 都在 T=8** ⇒ T 的形状与 θ 无关。
- **SOP 几乎不随 T 变**：跨 T 极差 **0.476%**（47.76–47.99 M）。T_dense 只改分类头。
- **T=32 的 AUPRC 掉 6.24 pp**（0.7411 → 0.6787），远大于 AUROC 的 −0.71 pp；
  即长时窗下排序不变、但正类高分端的排序变差。若论文报 AUPRC，不要用 T=32。

## 3 与 6 块 delta（`ab_full`）的 T 形状对照

| 配置 | 统计对象 | T=2 相对 T=8 | 跨 T 极差 | θ 依赖 |
|---|---|---|---|---|
| `ab_polarity`（6 块整流） | 十折均值 | −1.15 pp | 1.14 pp | **无** |
| `ab_full`（6 块 Δ） | 单权重 fold10（E10） | −1.31 pp | 1.31 pp | 有（非单调） |

- 两者的 T 形状**同向同量级**（T=2 掉约 1.2 pp、T=8 附近饱和），说明"时间步太短会伤精度"是
  主干的性质，与前端编码器无关。
- **但两个统计对象不同（十折均值 vs 单权重），不能直接相减或画在同一误差棒上。**

## 4 对既有结论的影响

1. **E12 §4 第 4 条的分辨率限制解除**：不再需要写"本配置只扫了 T=8"。
2. **2×2 的结论加强**：θ 旋钮的有无由前端决定——这一条现在在**完整 θ×T 网格**上成立，
   不只是单个 T 切片。
3. **派生数字不变**：`sop_corrected_10fold.json`（`30521a23`）与 `p1_supplement.json`
   （`85255baf`）在补跑后重算的 md5 **逐字节相同**——因为它们只取 θ=0.15 / T=8 的行，
   而该切片的 60 个单元与补跑前完全一致。

## 5 复现

```bash
cd /mnt/ECG-SNN-LowPower
# 4 卡分片补跑（已完成的单元会自动跳过）
python experiments/46_ab_polarity_theta.py --folds 1,2,3  --gpu 0 --t-dense 2,4,8,16,32 --out results/theta_pol_shard_g0.json
python experiments/46_ab_polarity_theta.py --folds 4,5,6  --gpu 1 --t-dense 2,4,8,16,32 --out results/theta_pol_shard_g1.json
python experiments/46_ab_polarity_theta.py --folds 7,8,10 --gpu 2 --t-dense 2,4,8,16,32 --out results/theta_pol_shard_g2.json
python experiments/46_ab_polarity_theta.py --folds 9      --gpu 3 --t-dense 2,4,8,16,32 --out results/theta_pol_shard_g3.json
python experiments/47_merge_polarity.py
python experiments/50_ab_polarity_T_summary.py
```
