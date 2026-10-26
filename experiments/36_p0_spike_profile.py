#!/usr/bin/env python3
"""P0-3: depth-wise spike-rate / effective-rank profile (mechanism B).

Pure inference on existing seed-0 checkpoints. No training, no new weights.

For each (config, fold, split) it records, per block:
  rho        mean firing rate of the block OUTPUT spikes *before* temporal
             pooling (this is the quantity that decides whether a deep block
             still carries events forward)
  rho_post   mean rate after pooling (what the next block is fed)
  r_eff      effective rank of the pre-pool spike map, i.e. the participation
             ratio of its channel covariance, normalised by channel count
  C, T       channel count and (pre-pool) sequence length of that block

Definition of effective rank  (quoted verbatim in the report)
--------------------------------------------------------------
For a pre-pool spike map of a batch, S in R^{B x C x T}, treat the N = B*T
channel activity vectors as observations and the C channels as features:
    X  in R^{C x N},     mu = mean over N
    Cov = (X - mu)(X - mu)^T / N      in R^{C x C}
    PR  = (sum_i lambda_i)^2 / sum_i lambda_i^2        (participation ratio)
    r_eff = PR / C                                     in (0, 1]
The covariance is accumulated over the WHOLE split, not per batch.  A layer
that only ever uses k of its C channels has r_eff ~ k/C regardless of how
large its mean firing rate is.

Splits
------
test   full test fold                (fold f)
val    full val fold                 (fold f-1, fold 1 -> 10)
train  fixed random 2000-sample subset of the remaining 8 folds (seed 0)

Output: results/p0_spike_profile.json (new file; one entry per config/fold).
Rerunning a (config, fold) replaces that entry, so the 4 configs can be run
in parallel on 4 GPUs against the same output file via --shard-out.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

NC = Path("/mnt/ECG-SNN-LowPower/third_party/neurocardio")
sys.path.insert(0, str(NC))

from neurocardio.dataset import PTBXLCache  # noqa: E402
from neurocardio.factory import make_model  # noqa: E402

CACHE = "/mnt/ECG-SNN-LowPower/results/ptbxl_cache"
RESULTS = Path("/mnt/ECG-SNN-LowPower/results")

# config -> (factory model name, original ablation dir for fold 10, repro template)
CONFIGS = {
    "ab_full":        ("ab_full",        "ablation_full",
                       "repro_ab_full_fold{f}"),
    "ab_polarity":    ("ab_polarity",    "ablation_encoder_polarity",
                       "repro_ab_polarity_fold{f}"),
    "ab_shallow":     ("ab_shallow",     "ablation_shallow_3blocks",
                       "repro_ab_shallow_fold{f}"),
    "ab_shallow_pol": ("ab_shallow_pol", "repro_ab_shallow_pol_fold10",
                       "repro_ab_shallow_pol_fold{f}"),
}


def run_dir(cfg: str, fold: int) -> Path:
    """fold-10 checkpoints live in the original ablation dirs, folds 1-9 in repro_*."""
    _, base, tmpl = CONFIGS[cfg]
    cand = RESULTS / tmpl.format(f=fold)
    if (cand / "best.pt").exists():
        return cand
    return RESULTS / base


class LayerAccumulator:
    """Accumulates rate + channel covariance for one layer over a whole split."""

    def __init__(self):
        self.c = None
        self.T = None
        self.S = None          # (C, C) float64, sum of X X^T
        self.m = None          # (C,)   float64, sum of X over observations
        self.n_obs = 0
        self.spike_sum = 0.0
        self.count = 0
        self.post_sum = 0.0
        self.post_count = 0

    def add(self, pre: torch.Tensor, post: torch.Tensor) -> None:
        B, C, T = pre.shape
        if self.c is None:
            self.c = C
            self.S = torch.zeros(C, C, dtype=torch.float64)
            self.m = torch.zeros(C, dtype=torch.float64)
        X = pre.permute(1, 0, 2).reshape(C, B * T).float()
        self.S += (X @ X.t()).double().cpu()
        self.m += X.sum(dim=1).double().cpu()
        self.n_obs += B * T
        self.spike_sum += float(pre.sum())
        self.count += pre.numel()
        self.post_sum += float(post.sum())
        self.post_count += post.numel()
        self.T = T

    def summary(self, idx: int) -> dict:
        if self.c is None:
            return {"layer": idx, "channels": 0, "rho": None, "reff": None}
        n = max(self.n_obs, 1)
        mu = self.m / n
        Cov = self.S / n - torch.outer(mu, mu)
        ev = torch.linalg.eigvalsh(Cov).clamp(min=0.0)
        s1 = float(ev.sum())
        s2 = float((ev ** 2).sum())
        pr = (s1 * s1 / s2) if s2 > 0 else 0.0
        return {
            "layer": idx,
            "channels": int(self.c),
            "pre_pool_T": int(self.T or 0),
            "rho": self.spike_sum / max(self.count, 1),
            "rho_post": self.post_sum / max(self.post_count, 1),
            "reff_raw": pr,
            "reff": pr / float(self.c),
        }


class Profiler:
    """Captures pre-pool spikes (tpool pre-hook) and the stem output
    (block-0 pre-hook, i.e. what the first block is fed)."""

    def __init__(self, model):
        self.model = model
        self.accs = None
        self.stem = None
        self._h = []
        for blk in model.blocks:
            self._h.append(
                blk.tpool.register_forward_pre_hook(self._make_hook(len(self._h))))

        def stem_hook(_m, inputs):
            if self.stem is not None:
                s = inputs[0].detach()
                self.stem["sum"] += float(s.sum())
                self.stem["count"] += s.numel()

        self._h.append(model.blocks[0].register_forward_pre_hook(stem_hook))

    def _make_hook(self, idx):
        def hook(mod, inputs):
            if self.accs is None:
                return
            pre = inputs[0].detach()
            # NB: must not call mod(pre) here — the hook lives on `tpool`
            # itself and calling it would recurse.
            pool = self.model.blocks[idx].pool
            post = F.max_pool1d(pre, pool) if pool > 1 else pre
            self.accs[idx].add(pre, post)
        return hook

    def reset(self, with_stem: bool = True):
        self.accs = [LayerAccumulator() for _ in self.model.blocks]
        self.stem = {"sum": 0.0, "count": 0} if with_stem else None

    def freeze(self):
        self.accs = None
        self.stem = None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=list(CONFIGS))
    ap.add_argument("--folds", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--train-subset", type=int, default=2000)
    ap.add_argument("--out", default=None,
                    help="默认写到 results/p0_spike_profile_<config>.json，"
                         "以免 4 个进程并发写同一文件")
    args = ap.parse_args()
    if args.out is None:
        args.out = str(RESULTS / f"p0_spike_profile_{args.config}.json")

    dev = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(dev)
    folds = [int(x) for x in args.folds.split(",") if x.strip()]
    out_path = Path(args.out)

    for fold in folds:
        rd = run_dir(args.config, fold)
        ckpt_path = rd / "best.pt"
        if not ckpt_path.exists():
            print(f"[SKIP] {args.config} fold{fold}: {ckpt_path} 不存在", flush=True)
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("cfg", {}) or {}
        model_name = cfg.get("model_name") or CONFIGS[args.config][0]
        model = make_model(model_name, num_classes=5)
        model.load_state_dict(ckpt.get("ema") or ckpt["state_dict"])
        model.to(dev).eval()
        prof = Profiler(model)

        val_fold = 10 if fold == 1 else fold - 1
        train_folds = [f for f in range(1, 11) if f not in (fold, val_fold)]
        tr = PTBXLCache(CACHE, fold_subset=train_folds)
        rng = np.random.default_rng(0)
        sel = np.sort(rng.choice(len(tr), size=min(args.train_subset, len(tr)),
                                 replace=False))
        splits = {
            "test": PTBXLCache(CACHE, fold_subset=[fold]),
            "val": PTBXLCache(CACHE, fold_subset=[val_fold]),
            "train": Subset(tr, sel.tolist()),
        }

        entry = {
            "config": args.config, "model": model_name, "fold": fold,
            "run_dir": str(rd), "val_fold": val_fold,
            "train_folds": train_folds,
            "n_blocks": len(model.blocks),
            "params": int(sum(p.numel() for p in model.parameters())),
            "splits": {}, "val_auroc": None, "test_auroc": None,
        }
        tm = rd / "test_metrics.json"
        if tm.exists():
            entry["test_auroc"] = json.loads(tm.read_text()).get("macro_auroc")
        hist = rd / "history.json"
        if hist.exists():
            h = json.loads(hist.read_text()).get("history", [])
            if h:
                tl = [e["train_loss"] for e in h if e.get("train_loss") is not None]
                va = [e["val_macro_auroc"] for e in h
                      if e.get("val_macro_auroc") is not None]
                entry["train_loss"] = {"first": tl[0], "last": tl[-1],
                                       "min": min(tl), "epochs": len(h)}
                if va:
                    entry["val_auroc_best"] = max(va)
                    entry["val_auroc_best_epoch"] = int(np.argmax(va))
                    entry["val_auroc_last"] = va[-1]

        for split, ds in splits.items():
            dl = DataLoader(ds, batch_size=args.batch, num_workers=2)
            prof.reset()
            t0 = time.time()
            enc = {"sum": 0.0, "count": 0}
            with torch.no_grad():
                for i, (x, _y, _f) in enumerate(dl):
                    x = x.to(dev)
                    if i == 0:                      # encoder rate from one batch
                        sp = model.delta_encode(x)
                        enc["sum"] += float(sp.sum())
                        enc["count"] += sp.numel()
                    _ = model(x)
            entry["splits"][split] = {
                "n": len(ds),
                "encoder_rho": enc["sum"] / max(enc["count"], 1),
                "stem_rho": (prof.stem["sum"] / max(prof.stem["count"], 1)
                             if prof.stem else None),
                "layers": [a.summary(i) for i, a in enumerate(prof.accs)],
                "seconds": round(time.time() - t0, 1),
            }
            prof.freeze()
            rates = " ".join(f"{l['rho']:.4f}" for l in entry["splits"][split]["layers"])
            ranks = " ".join(f"{l['reff']:.3f}" for l in entry["splits"][split]["layers"])
            print(f"  {args.config} f{fold:<2} {split:<5} n={len(ds):<6} "
                  f"rho=[{rates}] reff=[{ranks}] ({time.time() - t0:.0f}s)", flush=True)

        rows = []
        if out_path.exists():
            try:
                rows = json.loads(out_path.read_text())
            except Exception:  # noqa: BLE001
                rows = []
        rows = [r for r in rows
                if not (r.get("config") == args.config and r.get("fold") == fold)]
        rows.append(entry)
        rows.sort(key=lambda r: (r.get("config", ""), r.get("fold", 0)))
        out_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
        print(f"[OK] {args.config} fold{fold} -> {out_path}", flush=True)

        del model, prof
        torch.cuda.empty_cache()

    print(f"\n完成：{args.config} {folds}", flush=True)


if __name__ == "__main__":
    main()
