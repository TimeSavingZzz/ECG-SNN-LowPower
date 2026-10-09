#!/usr/bin/env python3
"""Train CompactNeuroCardio (~54K params) on PTB-XL fold-10.

Fair-comparison counterpart to Sarhaddi's compact models (16K-55K): same
param scale, our architecture (delta encoder + S-TCN + attention pool).
Uses our mainline training config (pos_weight sqrt/cap8, theta 0.15) so the
only thing that differs from our 0.8594 SNN is model scale.

Smoke:  python3 experiments/24_train_compact.py --smoke
Full:   python3 experiments/24_train_compact.py --epochs 60
"""
import sys
from pathlib import Path

NC = Path("/mnt/ECG-SNN-LowPower/third_party/neurocardio")
sys.path.insert(0, str(NC))

from neurocardio.train import TrainConfig, main  # noqa: E402

CACHE = "/mnt/ECG-SNN-LowPower/results/ptbxl_cache"
ROOT = "/mnt/ECG-SNN-LowPower/results"


def run(epochs, batch_size, lr, max_steps, resume):
    cfg = TrainConfig(
        cache_dir=CACHE,
        output_dir=f"{ROOT}/repro_compact",
        label_set="diagnostic_superclass",
        model_name="snn_compact",
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
        pos_weight_mode="sqrt",       # our mainline setting (not Sarhaddi's "inv")
        pos_weight_cap=8.0,
        theta=0.15,
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

    run(1 if args.smoke else args.epochs, args.batch_size, args.lr,
        2 if args.smoke else 0, not args.smoke)
