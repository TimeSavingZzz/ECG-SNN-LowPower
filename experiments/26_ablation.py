#!/usr/bin/env python3
"""Train one ablation variant on PTB-XL fold-10 (Task E/F).

All variants share the SAME 100-epoch config as the compact/Sarhaddi runs, so
they are comparable to ab_full (the same-config control) — and roughly to the
100-epoch mainline (0.8594).

  python3 experiments/26_ablation.py --model ab_polarity
  python3 experiments/26_ablation.py --list
"""
import sys
from pathlib import Path

NC = Path("/mnt/ECG-SNN-LowPower/third_party/neurocardio")
sys.path.insert(0, str(NC))

from neurocardio.train import TrainConfig, main  # noqa: E402

CACHE = "/mnt/ECG-SNN-LowPower/results/ptbxl_cache"
ROOT = "/mnt/ECG-SNN-LowPower/results"

MODELS = {
    "ab_full":     "ablation_full",
    "ab_polarity": "ablation_encoder_polarity",
    "ab_multiq":   "ablation_encoder_multiquantum",
    "ab_noattn":   "ablation_no_attn_pool",
    "ab_nogate":   "ablation_no_gate_residual",
    "ab_nodilate": "ablation_no_dilation",
    "ab_shallow":  "ablation_shallow_3blocks",
    # P1 combined models
    "ab_shallow_pol": "improved_shallow_polarity",
    "ab_improved":    "improved_shallow_polarity_meanpool",
}


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
        pos_weight_mode="sqrt",     # mainline setting, so ab_full ~ control
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
    ap.add_argument("--model", default="ab_full")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for m, d in MODELS.items():
            print(f"{m:14s} -> {d}")
        raise SystemExit(0)

    if args.model not in MODELS:
        raise SystemExit(f"unknown model {args.model}; choose from {list(MODELS)}")

    print(f"\n########## {args.model} -> {MODELS[args.model]} ##########", flush=True)
    run(args.model, MODELS[args.model],
        1 if args.smoke else args.epochs, args.batch_size, args.lr,
        2 if args.smoke else 0, not args.smoke)
    print(f"########## DONE {args.model} ##########", flush=True)
