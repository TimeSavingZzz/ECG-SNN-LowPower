#!/usr/bin/env python3
"""Train the reimplemented Sarhaddi SNN / SNN-SRNN on PTB-XL (paper Table I).

Reuses neurocardio.train's loop (EMA, cosine LR, resume, checkpoints) with the
paper's loss weighting: pos_weight_mode="inv" (positive weight = neg/pos ratio).
Trains plain SNN then SNN-SRNN sequentially; writes test_metrics.json each.

Smoke check:  python3 experiments/23_train_sarhaddi.py --smoke
Full run:     python3 experiments/23_train_sarhaddi.py --epochs 60
"""
import sys
from pathlib import Path

NC = Path("/mnt/ECG-SNN-LowPower/third_party/neurocardio")
sys.path.insert(0, str(NC))

from neurocardio.train import TrainConfig, main  # noqa: E402

CACHE = "/mnt/ECG-SNN-LowPower/results/ptbxl_cache"
ROOT = "/mnt/ECG-SNN-LowPower/results"

# (model_name, output_dir)
MODELS = [
    ("sarhaddi_snn", "repro_sarhaddi_snn"),
    ("snn_srnn",     "repro_sarhaddi_srnn"),
]


def run(model_name, output_dir, epochs, batch_size, lr, max_steps, resume):
    cfg = TrainConfig(
        cache_dir=CACHE,
        output_dir=f"{ROOT}/{output_dir}",
        label_set="diagnostic_superclass",
        model_name=model_name,
        sampling_rate=100,
        train_folds=(1, 2, 3, 4, 5, 6, 7, 8),
        val_folds=(9,),
        test_folds=(10,),
        batch_size=batch_size,
        num_workers=4,
        lr=lr,
        weight_decay=1e-4,
        epochs=epochs,
        warmup_epochs=3,
        pos_weight_mode="inv",       # paper: positive-class weight = neg/pos ratio
        pos_weight_cap=1e9,          # no cap (faithful to paper)
        theta=0.15,                  # unused by sarhaddi models (no delta encoder)
        ema_decay=0.999,
        grad_clip=1.0,
        seed=0,
        save_every=1,
        eval_use_ema=True,
        device="cuda",
        log_every=50,
        wallclock_hours=0.0,
        max_steps_per_epoch=max_steps,
        resume=resume,
    )
    main(cfg)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    epochs = 1 if args.smoke else args.epochs
    max_steps = 2 if args.smoke else 0
    resume = (not args.smoke)
    for model_name, out in MODELS:
        print(f"\n########## {model_name} -> {out} ##########", flush=True)
        run(model_name, out, epochs, args.batch_size, args.lr, max_steps, resume)
        print(f"########## DONE {model_name} ##########", flush=True)
