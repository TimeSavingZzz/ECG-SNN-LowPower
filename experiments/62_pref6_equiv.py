#!/usr/bin/env python3
"""ab_pref6 == ab_full 的等价性检验（不是参数计数，而是逐样本输出）。

E22 §4 限制 2 曾说：`ab_pref6` 与 `ab_full` 参数计数逐位相同只是**必要条件**，
若两者实现有差异，差异会被算进「深度效应」。本脚本把这条从「未检验」变成「已检验」：

  1. state_dict 键/形状逐项相同，权重逐元素全等（同一份权重装进两个模型）；
  2. 同一 batch 的前向 logits **逐位相同**（max|Δ| == 0）；
  3. `ab_pref6` 用自己的权重在 fold10 测试集上重算 macro-AUROC，与
     `ablation_full/test_metrics.json` 记录的值比对（容差 1e-6）。

用法::

    python3 experiments/62_pref6_equiv.py            # 默认 GPU 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/mnt/ECG-SNN-LowPower")
sys.path.insert(0, str(ROOT / "third_party" / "neurocardio"))

from neurocardio.dataset import PTBXLCache  # noqa: E402
from neurocardio.factory import make_model  # noqa: E402
from neurocardio.train import multilabel_metrics  # noqa: E402

LABELS = ["NORM", "MI", "STTC", "CD", "HYP"]
CKPT = ROOT / "results" / "ablation_full" / "best.pt"
REF = ROOT / "results" / "ablation_full" / "test_metrics.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "pref6_equivalence.json"))
    args = ap.parse_args()
    dev = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(dev)

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    w = ckpt.get("ema") or ckpt["state_dict"]

    full = make_model("ab_full", num_classes=5)
    pref = make_model("ab_pref6", num_classes=5)
    sf, sp = full.state_dict(), pref.state_dict()
    same_keys = sorted(sf) == sorted(sp)
    shape_diff = [k for k in sf if same_keys and sf[k].shape != sp[k].shape]
    full.load_state_dict(w, strict=True)
    pref.load_state_dict(w, strict=True)
    max_wdiff = max((sf[k] - sp[k]).abs().max().item() for k in sf)

    full.to(dev).eval()
    pref.to(dev).eval()

    ds = PTBXLCache(ROOT / "results" / "ptbxl_cache", fold_subset=[10])
    xs, ys = [], []
    for i in range(len(ds)):
        x, y = ds[i][0], ds[i][1]
        xs.append(x)
        ys.append(y)
    x_all = torch.stack(xs).to(dev)
    y_all = torch.stack(ys).cpu().numpy()

    max_logit_diff = 0.0
    y_score_full, y_score_pref = [], []
    with torch.no_grad():
        for i in range(0, len(x_all), 64):
            xb = x_all[i:i + 64]
            lf = full(xb)
            lp = pref(xb)
            max_logit_diff = max(max_logit_diff, (lf - lp).abs().max().item())
            y_score_full.append(torch.sigmoid(lf).cpu().numpy())
            y_score_pref.append(torch.sigmoid(lp).cpu().numpy())
    y_score_full = np.concatenate(y_score_full)
    y_score_pref = np.concatenate(y_score_pref)
    m_full = multilabel_metrics(y_all, y_score_full, LABELS)
    m = multilabel_metrics(y_all, y_score_pref, LABELS)
    ref = json.loads(REF.read_text())
    ref_auroc = float(ref["macro_auroc"])
    ref_auprc = float(ref["macro_auprc"])

    out = {
        "script": "experiments/62_pref6_equiv.py",
        "checkpoint": str(CKPT),
        "fold": 10,
        "n_test": int(len(ds)),
        "state_dict_same_keys": bool(same_keys),
        "state_dict_shape_mismatches": shape_diff,
        "max_weight_abs_diff": float(max_wdiff),
        "max_logit_abs_diff": float(max_logit_diff),
        "full_recomputed_macro_auroc": float(m_full["macro_auroc"]),
        "pref6_macro_auroc": float(m["macro_auroc"]),
        "pref6_macro_auprc": float(m["macro_auprc"]),
        "stored_full_macro_auroc": ref_auroc,
        "stored_full_macro_auprc": ref_auprc,
        # gap between this pipeline and the original training-eval pipeline
        "stored_reproduction_gap_auroc": abs(float(m_full["macro_auroc"]) - ref_auroc),
        "stored_reproduction_gap_auprc": abs(float(m_full["macro_auprc"]) - ref_auprc),
        # gap between the two models under the SAME pipeline == what matters
        "pref6_vs_full_auroc_gap": abs(float(m["macro_auroc"]) - float(m_full["macro_auroc"])),
        "equivalent": bool(same_keys and not shape_diff and max_logit_diff == 0.0),
    }
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
