#!/usr/bin/env python3
"""Multi-fold cross-validation for a given model (Task D): mean±std macro-AUROC.

Trains the model once per test fold (val = test_fold-1 wrapping, train = the
other 8 folds), then the test macro-AUROC is collected across folds to report
mean±std instead of a single fold-10 number.

Usage:
  python3 experiments/25_multifold.py --model snn_srnn --test-folds 9,8
  (fold 10 already produced by the main run; merged into the summary later)
"""
import sys
from pathlib import Path

NC = Path("/mnt/ECG-SNN-LowPower/third_party/neurocardio")
sys.path.insert(0, str(NC))

from neurocardio.train import TrainConfig, main  # noqa: E402

CACHE = "/mnt/ECG-SNN-LowPower/results/ptbxl_cache"
ROOT = "/mnt/ECG-SNN-LowPower/results"


def split(test_fold):
    val_fold = test_fold - 1 if test_fold > 1 else 10
    train = tuple(f for f in range(1, 11) if f not in (test_fold, val_fold))
    return train, (val_fold,), (test_fold,)


def run(model_name, test_fold, epochs, batch_size, lr):
    train_folds, val_folds, test_folds = split(test_fold)
    pos_mode = "inv" if model_name in ("sarhaddi_snn", "snn_srnn") else "sqrt"
    pos_cap = 1e9 if pos_mode == "inv" else 8.0
    out = f"{ROOT}/repro_{model_name}_fold{test_fold}"
    cfg = TrainConfig(
        cache_dir=CACHE, output_dir=out, label_set="diagnostic_superclass",
        model_name=model_name, sampling_rate=100,
        train_folds=train_folds, val_folds=val_folds, test_folds=test_folds,
        batch_size=batch_size, num_workers=4, lr=lr, weight_decay=1e-4,
        epochs=epochs, warmup_epochs=3,
        pos_weight_mode=pos_mode, pos_weight_cap=pos_cap,
        theta=0.15, ema_decay=0.999, grad_clip=1.0, seed=0,
        save_every=1, eval_use_ema=True, device="cuda", log_every=50,
        wallclock_hours=0.0, max_steps_per_epoch=0, resume=True,
    )
    main(cfg)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="snn_srnn")
    ap.add_argument("--test-folds", default="9,8")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    folds = [int(x) for x in args.test_folds.split(",")]
    for f in folds:
        print(f"\n########## {args.model} test_fold={f} ##########", flush=True)
        run(args.model, f, args.epochs, args.batch_size, args.lr)
        print(f"########## DONE fold {f} ##########", flush=True)
