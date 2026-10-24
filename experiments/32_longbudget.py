#!/usr/bin/env python3
"""Training-budget probe: does the SNN keep improving past 100 epochs?

Motivation (audit finding): history.json shows the CNN/ResNet baselines peak at
epoch 46 / 33 and then overfit badly, while the SNN validation AUROC is still
rising at epoch 96. The headline 2.4pp gap to CNN may therefore be partly a
training-budget artifact rather than an architecture difference.

This script trains ONE model on ONE test fold with a longer budget and writes to
its own output directory so no existing result is touched.

  python3 experiments/32_longbudget.py --model ab_shallow_pol --test-fold 10 --epochs 200
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


def run(model_name, test_fold, epochs, batch_size, lr, tag):
    train_folds, val_folds, test_folds = split(test_fold)
    out = f"{ROOT}/{tag}"
    cfg = TrainConfig(
        cache_dir=CACHE, output_dir=out, label_set="diagnostic_superclass",
        model_name=model_name, sampling_rate=100,
        train_folds=train_folds, val_folds=val_folds, test_folds=test_folds,
        batch_size=batch_size, num_workers=4, lr=lr, weight_decay=1e-4,
        epochs=epochs, warmup_epochs=3,
        pos_weight_mode="sqrt", pos_weight_cap=8.0,
        theta=0.15, ema_decay=0.999, grad_clip=1.0, seed=0,
        save_every=10, eval_use_ema=True, device="cuda", log_every=50,
        wallclock_hours=0.0, max_steps_per_epoch=0, resume=True,
    )
    main(cfg)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ab_shallow_pol")
    ap.add_argument("--test-fold", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or f"budget_{args.model}_fold{args.test_fold}_e{args.epochs}"
    print("### long-budget probe: %s fold%d %dep -> %s" % (args.model, args.test_fold, args.epochs, tag), flush=True)
    run(args.model, args.test_fold, args.epochs, args.batch_size, args.lr, tag)
    print("### DONE %s" % tag, flush=True)
