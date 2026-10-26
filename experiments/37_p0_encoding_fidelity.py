#!/usr/bin/env python3
"""P0-2: encoding fidelity under a matched spike budget.

Why this is not the naive "delta vs bipolar" comparison
-------------------------------------------------------
`dual_polarity_encode` returns cat([relu(z), relu(-z)]) where z is the
per-lead z-score.  That is a CONTINUOUS non-negative tensor, not a spike
train, and the model feeds it straight into `stem_conv` with no threshold.
So comparing "delta (binary)" against "bipolar (analog)" on fidelity or on
linear decodability is a category error -- the analog arm wins by
construction because it never quantised anything.

This script therefore reports three kinds of arm:

  delta_th*            true binary encoders (the deployed baseline)
  bipolar_analog       the deployed ab_polarity / ab_shallow_pol encoder,
                       reported ONLY as an upper bound.  Not a spike scheme.
  bipolar_bin_*        the analog tensor thresholded to binary, either at a
                       fixed tau or at a per-sample tau chosen so that the
                       sample emits EXACTLY as many spikes as delta does.
                       This is the only fair "same budget, different scheme"
                       comparison.

Decodability protocol
---------------------
  features   temporal mean of the 24-channel tensor          -> 24 dims
             and 20-bin temporal mean                        -> 480 dims
  classifier one-vs-rest logistic regression per label
  fit on     2000-sample fixed random subset of the 8 train folds (seed 0)
  eval on    the full test fold
  metric     macro-AUROC over the 5 superclasses (+ per-label)

Everything is fit on train and evaluated on test; the test fold is never
used to pick a threshold, a tau, or a C.

Output: results/p0_encoding_fidelity.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset

NC = Path("/mnt/ECG-SNN-LowPower/third_party/neurocardio")
sys.path.insert(0, str(NC))

from neurocardio.dataset import PTBXLCache  # noqa: E402
from neurocardio.encoding import delta_encode_batch_12lead  # noqa: E402
from neurocardio.sarhaddi import dual_polarity_encode  # noqa: E402

CACHE = "/mnt/ECG-SNN-LowPower/results/ptbxl_cache"
LABELS = ["NORM", "MI", "STTC", "CD", "HYP"]
BINS = 20


def bipolar_analog(x: torch.Tensor) -> torch.Tensor:
    return dual_polarity_encode(x)


def counts_of(sp: torch.Tensor) -> np.ndarray:
    """(B, 24, T) -> per-sample number of non-zero entries."""
    return (sp != 0).sum(dim=(1, 2)).cpu().numpy()


def binarise_analog(a: torch.Tensor, tau) -> torch.Tensor:
    """a: (B,24,T) non-negative analog. tau may be scalar or (B,1,1)."""
    return (a > tau).to(a.dtype)


def tau_for_count(a: torch.Tensor, target: np.ndarray):
    """Per-sample threshold so that #{a > tau} == target (as close as ties allow).

    Uses the k-th largest order statistic of each sample; exact when there are
    no ties, which holds because the inputs are continuous floats.
    """
    B = a.shape[0]
    flat = a.reshape(B, -1)
    taus = torch.zeros(B, 1, 1, device=a.device, dtype=a.dtype)
    for i in range(B):
        k = int(target[i])
        if k <= 0:
            taus[i] = flat[i].max() + 1.0
            continue
        if k >= flat.shape[1]:
            taus[i] = -1.0
            continue
        # kth largest value; spike iff strictly greater
        kth = torch.topk(flat[i], k, largest=True).values[-1]
        taus[i] = kth
    return taus


def features(sp: torch.Tensor) -> dict[str, np.ndarray]:
    """(B,24,T) -> {'mean24': (B,24), 'bin20': (B,480)}"""
    B, C, T = sp.shape
    mean24 = sp.mean(dim=-1)
    edges = torch.linspace(0, T, BINS + 1).long()
    bins = torch.stack([sp[:, :, edges[i]:edges[i + 1]].mean(dim=-1)
                        for i in range(BINS)], dim=-1)      # (B,C,BINS)
    return {"mean24": mean24.cpu().numpy(),
            "bin20": bins.reshape(B, C * BINS).cpu().numpy()}


def macro_auroc(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, dict]:
    per = {}
    aucs = []
    for j, lab in enumerate(LABELS):
        if y_true[:, j].sum() == 0 or y_true[:, j].sum() == len(y_true):
            per[lab] = None
            continue
        a = float(roc_auc_score(y_true[:, j], scores[:, j]))
        per[lab] = a
        aucs.append(a)
    return float(np.mean(aucs)) if aucs else float("nan"), per


def eval_decoder(xtr, ytr, xte, yte, C=1.0) -> tuple[float, dict]:
    sc = StandardScaler().fit(xtr)
    xtr_s, xte_s = sc.transform(xtr), sc.transform(xte)
    scores = np.zeros((xte.shape[0], len(LABELS)), dtype=np.float64)
    for j in range(len(LABELS)):
        clf = LogisticRegression(C=C, max_iter=2000, solver="liblinear")
        clf.fit(xtr_s, ytr[:, j])
        scores[:, j] = clf.predict_proba(xte_s)[:, 1]
    return macro_auroc(yte, scores)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--train-subset", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--C", type=float, default=1.0,
                    help="logistic regression inverse regularisation (fixed a priori)")
    ap.add_argument("--out", default="/mnt/ECG-SNN-LowPower/results/p0_encoding_fidelity.json")
    args = ap.parse_args()

    folds = [int(x) for x in args.folds.split(",") if x.strip()]
    out_path = Path(args.out)
    results = []

    for fold in folds:
        t0 = time.time()
        val_fold = 10 if fold == 1 else fold - 1
        train_folds = [f for f in range(1, 11) if f not in (fold, val_fold)]
        tr_ds = PTBXLCache(CACHE, fold_subset=train_folds)
        rng = np.random.default_rng(0)
        sel = np.sort(rng.choice(len(tr_ds), size=min(args.train_subset, len(tr_ds)),
                                 replace=False))
        tr_ds = Subset(tr_ds, sel.tolist())
        te_ds = PTBXLCache(CACHE, fold_subset=[fold])

        def collect(ds):
            dl = DataLoader(ds, batch_size=args.batch, num_workers=2)
            xs, ys = [], []
            for x, y, _f in dl:
                xs.append(x)
                ys.append(y.numpy())
            return torch.cat(xs), np.concatenate(ys)

        x_tr, y_tr = collect(tr_ds)
        x_te, y_te = collect(te_ds)

        # ---- reference: delta at theta = 0.15 (the deployed baseline) ----
        sp_ref_tr = delta_encode_batch_12lead(x_tr, theta=0.15)
        sp_ref_te = delta_encode_batch_12lead(x_te, theta=0.15)
        n_ref_tr = counts_of(sp_ref_tr)
        n_ref_te = counts_of(sp_ref_te)

        # ---- bipolar analog ----
        a_tr = bipolar_analog(x_tr)
        a_te = bipolar_analog(x_te)

        arms: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        arms["delta_th0.15"] = (sp_ref_tr, sp_ref_te)
        for th in (0.10, 0.20, 0.30):
            arms[f"delta_th{th:.2f}"] = (delta_encode_batch_12lead(x_tr, theta=th),
                                         delta_encode_batch_12lead(x_te, theta=th))
        arms["bipolar_analog"] = (a_tr, a_te)
        for tau in (0.5, 1.0, 2.0):
            arms[f"bipolar_bin_tau{tau:.1f}"] = (binarise_analog(a_tr, tau),
                                                 binarise_analog(a_te, tau))
        # same spike budget as delta_th0.15, per sample
        arms["bipolar_bin_countmatched"] = (
            binarise_analog(a_tr, tau_for_count(a_tr, n_ref_tr)),
            binarise_analog(a_te, tau_for_count(a_te, n_ref_te)),
        )

        entry = {
            "fold": fold, "val_fold": val_fold,
            "n_train": int(x_tr.shape[0]), "n_test": int(x_te.shape[0]),
            "bipolar_analog_stats": {
                "min": float(a_tr.min()), "max": float(a_tr.max()),
                "mean": float(a_tr.mean()), "std": float(a_tr.std()),
                "frac_binary": float(((a_tr == 0) | (a_tr == 1)).float().mean()),
            },
            "arms": {},
        }
        for name, (str_, ste) in arms.items():
            ntr, nte = counts_of(str_), counts_of(ste)
            feats_tr, feats_te = features(str_), features(ste)
            rec = {
                "spikes_per_sample_train": float(ntr.mean()),
                "spikes_per_sample_test": float(nte.mean()),
                "features": {},
            }
            for fname in ("mean24", "bin20"):
                t1 = time.time()
                auc, per = eval_decoder(feats_tr[fname], y_tr,
                                        feats_te[fname], y_te, C=args.C)
                rec["features"][fname] = {"macro_auroc": auc, "per_label": per,
                                          "fit_seconds": round(time.time() - t1, 1)}
            entry["arms"][name] = rec
            print(f"  f{fold} {name:<26} spikes={ntr.mean():8.1f} "
                  f"mean24={rec['features']['mean24']['macro_auroc']:.4f} "
                  f"bin20={rec['features']['bin20']['macro_auroc']:.4f}", flush=True)

        entry["seconds"] = round(time.time() - t0, 1)
        results = [r for r in results if r.get("fold") != fold]
        results.append(entry)
        results.sort(key=lambda r: r["fold"])
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"[OK] fold{fold} ({entry['seconds']}s) -> {out_path}", flush=True)

    print("\n完成 P0-2", flush=True)


if __name__ == "__main__":
    main()
