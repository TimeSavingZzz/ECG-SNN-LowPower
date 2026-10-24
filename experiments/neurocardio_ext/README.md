# neurocardio 扩展模块（消融 / 紧凑 / Sarhaddi 复现）

本目录是 `third_party/neurocardio`（上游仓库）里**我们新增或修改过的全部代码**的存档。
`third_party/` 在 `.gitignore` 中，上游副本不入库；若只靠容器里的 working tree，
十折实验的模型定义将没有任何版本记录。本目录解决这个问题。

## 上游基线

- 仓库：hanshaunlee/neurocardio
- 提交：`5b5fd23645dfd8753d48aa6ddf4bac291c16a757`（见 `UPSTREAM_COMMIT.txt`）

## 内容

| 文件 | 类型 | 说明 |
|---|---|---|
| `ablation.py` | 新增 | 6 个消融变体（polarity / multiquantum / noattn / nogate / nodilation / shallow），以及组合体 shallow+polarity 与再加均值池化 |
| `compact.py` | 新增 | 上游论文的 52K 紧凑 SNN（复现用） |
| `sarhaddi.py` | 新增 | BSN 2026（Sarhaddi 等）plain SNN 与 SNN-SRNN 的参数级重建 |
| `factory.py` | 修改 | `make_model()` 注册上述所有模型名 |
| `factory.patch` | 补丁 | `factory.py` 相对上游的 diff（供核对） |

## 复原步骤

1. `git clone` 上游仓库到 `third_party/neurocardio`
2. `git -C third_party/neurocardio checkout 5b5fd23645dfd8753d48aa6ddf4bac291c16a757`
3. 把本目录的 `ablation.py`、`compact.py`、`sarhaddi.py`、`factory.py` 复制到
   `third_party/neurocardio/neurocardio/`

复原后 `make_model('ab_shallow_pol', 5)` 必须能实例化，且参数量为 **528 392**。

## 与论文的对应

- `ab_full`：6 块基线，1 893 768 参数，十折 AUROC 0.8769 ± 0.0108
- `ab_shallow_pol`：3 块 + 双极性编码，528 392 参数，十折 AUROC 0.8941 ± 0.0081
- 训练入口：`experiments/26_ablation.py`（单折）、`experiments/25_multifold.py`（多折）
- 十折批跑：`launch_p10_multifold.sh`

## 许可

上游 neurocardio 为 MIT 许可。本目录中的 factory.py 是其派生文件（改动见 factory.patch），
其余三个文件为新增模块，仅 import 上游代码、未复制其源码。上游 clone 中未附 LICENSE 文件，
复现时请以上游仓库页面标注的许可为准。
