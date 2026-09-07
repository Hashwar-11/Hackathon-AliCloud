"""
Recover Stage 2 (raw) and Stage 3 (Channel-Boosted) residential metrics on CPU.

WHY THIS EXISTS
    The Kaggle/Colab benchmark returned the trained WEIGHTS (models/checkpoints/)
    but the consolidated metrics table was never merged back into the repo, so
    experiments_results/benchmark_results.json only contained Stage 1. This script
    re-runs *inference only* (no training) with the returned checkpoints on the
    frozen val/test splits and writes real stage2_raw + stage3_boosted entries back
    into the JSON, so the numbers are reproducible from what is actually on disk.

WEIGHTS USED (verified by timestamp + the code that writes each file)
    Stage 2 raw      : models/checkpoints/residential_stage2.pt     (num_channels=1)
    Stage 3 boosted  : models/checkpoints/residential_model_best.pt (num_channels=4)
                       -- stage3_boosted.py saves its FINAL model to this production
                          name (line 252), so this IS the Stage 3 boosted detector.
    NOT used         : models/checkpoints/residential_stage3.pt -- that name was
                       overwritten by finetune_pakistan.py (42 all-normal Pakistan
                       rows), so it is NOT the Kaggle Stage 3 model.

HONESTY
    - Industrial detector is reported SKIPPED (Stage 2 industrial was skipped: the
      consumer-type proxy leaves too few industrial-typed rows to be reliable).
    - Stage 2 val ROC-AUC is cross-checked against stage2_val_probs.npz (the probs
      the Kaggle run itself saved). If they agree, the recovery is validated.
    - Stage 5 (Pakistan transfer) is written as PENDING/blocked-by-data: the target
      file has a single class, so transfer cannot be trained or measured.

Run:
    python -m src.experiments.recover_stage_metrics --config config/config.yaml
    # add --no-write to compute + print without touching the JSON
"""
import argparse
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from src.agents.residential.model import ResidentialModel
from src.experiments.common import (
    load_config, load_split, classification_metrics, best_f1_threshold,
    save_stage_results, print_metrics,
)
# Reuse the exact, already-tested inference helpers so nothing is re-implemented:
from src.experiments.stage2_raw import predict_proba_residential as predict_raw
from src.experiments.stage3_boosted import predict_proba_residential as predict_stacked, stack_split
from src.experiments.stage5_transfer import load_channels

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def res_model(num_channels: int, seq_len: int, cfg) -> ResidentialModel:
    m = cfg["residential_model"]
    return ResidentialModel(
        num_channels=num_channels, seq_len=seq_len,
        cnn_filters=m["cnn_filters"], cnn_kernel_size=m["cnn_kernel_size"],
        lstm_hidden_units=m["lstm_hidden_units"], dropout=m["dropout"],
    ).to(DEVICE)


def load_residential(path: str, num_channels: int, seq_len: int, cfg) -> ResidentialModel:
    model = res_model(num_channels, seq_len, cfg)
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--no-write", action="store_true",
                    help="compute and print only; do not modify benchmark_results.json")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=1024,
                    help="rows per channel-stacking chunk (lower = less peak RAM)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    processed = cfg["paths"]["processed_dir"]
    ckpt = cfg["paths"]["checkpoints_dir"]
    print(f"[recover] device={DEVICE}  checkpoints={ckpt}")

    X_va, y_va, ct_va = load_split(processed, "val")
    X_te, y_te, ct_te = load_split(processed, "test")
    va_res, te_res = (ct_va == "residential"), (ct_te == "residential")
    seq_len = int(X_va.shape[1])
    print(f"[recover] seq_len={seq_len}  val_res={int(va_res.sum())} (pos {int(y_va[va_res].sum())})"
          f"  test_res={int(te_res.sum())} (pos {int(y_te[te_res].sum())})")

    # ---------------- Stage 2: raw sequence, single channel ----------------
    s2_path = os.path.join(ckpt, "residential_stage2.pt")
    if not os.path.exists(s2_path):
        raise SystemExit(f"[recover] missing Stage 2 weights: {s2_path}")
    m2 = load_residential(s2_path, num_channels=1, seq_len=seq_len, cfg=cfg)
    p2_va = predict_raw(m2, torch.tensor(X_va[va_res], dtype=torch.float32), batch_size=args.batch_size)
    p2_te = predict_raw(m2, torch.tensor(X_te[te_res], dtype=torch.float32), batch_size=args.batch_size)
    thr2 = best_f1_threshold(y_va[va_res], p2_va)
    s2_val = classification_metrics(y_va[va_res], p2_va, thr2)
    s2_val05 = classification_metrics(y_va[va_res], p2_va, 0.5)
    s2_test = classification_metrics(y_te[te_res], p2_te, thr2)
    print_metrics("Stage 2 residential (raw) — VAL", s2_val)
    print_metrics("Stage 2 residential (raw) — TEST", s2_test)

    # cross-check against the Kaggle-saved validation probabilities
    npz = os.path.join(ckpt, "stage2_val_probs.npz")
    npz_auc = None
    if os.path.exists(npz):
        z = np.load(npz)
        if z["res_prob"].size and len(np.unique(z["res_y"])) > 1:
            npz_auc = round(float(roc_auc_score(z["res_y"], z["res_prob"])), 4)
            print(f"[recover] CROSS-CHECK Stage 2 val ROC-AUC — npz(Kaggle)={npz_auc} "
                  f"vs re-inference={s2_val['roc_auc']} "
                  f"({'MATCH' if abs(npz_auc - s2_val['roc_auc']) <= 0.01 else 'DIFFER'})")

    # ---------------- Stage 3: Channel-Boosted, 4 channels ----------------
    ae, pretext, freq = load_channels(seq_len, ckpt)
    print("[recover] stacking channels for val/test residential splits ...")
    S_va = stack_split(X_va[va_res], ae, pretext, freq, chunk=args.chunk)
    S_te = stack_split(X_te[te_res], ae, pretext, freq, chunk=args.chunk)
    num_channels = int(S_va.shape[1])

    s3_path = os.path.join(ckpt, "residential_model_best.pt")
    if not os.path.exists(s3_path):
        raise SystemExit(f"[recover] missing Stage 3 weights: {s3_path}")
    m3 = load_residential(s3_path, num_channels=num_channels, seq_len=seq_len, cfg=cfg)
    p3_va = predict_stacked(m3, S_va, batch_size=args.batch_size)
    p3_te = predict_stacked(m3, S_te, batch_size=args.batch_size)
    thr3 = best_f1_threshold(y_va[va_res], p3_va)
    s3_val = classification_metrics(y_va[va_res], p3_va, thr3)
    s3_val05 = classification_metrics(y_va[va_res], p3_va, 0.5)
    s3_test = classification_metrics(y_te[te_res], p3_te, thr3)
    print_metrics("Stage 3 residential (boosted) — VAL", s3_val)
    print_metrics("Stage 3 residential (boosted) — TEST", s3_test)

    delta_auc = round(s3_val["roc_auc"] - s2_val["roc_auc"], 4)
    delta_f1 = round(s3_val["f1"] - s2_val["f1"], 4)
    print(f"[recover] Stage 3 - Stage 2 delta (val): ROC-AUC {delta_auc:+.4f}  F1 {delta_f1:+.4f}")

    if args.no_write:
        print("[recover] --no-write set: benchmark_results.json NOT modified.")
        return

    note = ("Metrics recovered by CPU re-inference (no training) from the Kaggle-returned "
            "checkpoints on the frozen val/test splits. Stage 3 uses residential_model_best.pt "
            "(stage3_boosted.py writes its final model there); residential_stage3.pt is NOT used "
            "(overwritten by finetune_pakistan.py). Industrial detector not recovered "
            "(too few industrial-typed rows to be statistically meaningful).")

    save_stage_results("stage2_raw", {
        "device": DEVICE, "recovered": True, "recovery_note": note,
        "residential": {
            "val_metrics": s2_val, "val_metrics_thr05": s2_val05, "test_metrics": s2_test,
            "n_val": int(va_res.sum()), "n_test": int(te_res.sum()),
            "npz_crosscheck_val_auc": npz_auc,
        },
        "industrial": {"status": "SKIPPED",
                       "reason": "too few industrial-typed rows (consumer_type_proxy @500 kWh/day)"},
    })
    save_stage_results("stage3_boosted", {
        "device": DEVICE, "recovered": True, "recovery_note": note,
        "residential": {
            "val_metrics": s3_val, "val_metrics_thr05": s3_val05, "test_metrics": s3_test,
            "n_val": int(va_res.sum()), "n_test": int(te_res.sum()),
        },
        "industrial": {"status": "SKIPPED",
                       "reason": "not recovered — industrial sample too small to be reliable"},
        "comparison_vs_stage2": {
            "note": "Stage 3 (Channel Boosting) vs Stage 2 (raw), same val split",
            "stage2_residential_val": s2_val, "stage3_residential_val": s3_val,
            "delta_roc_auc": delta_auc, "delta_f1": delta_f1,
        },
    })
    save_stage_results("stage5_transfer", {
        "status": "PENDING",
        "blocker": "data",
        "reason": ("Pakistan target has 42 rows of a SINGLE class (0 confirmed theft in "
                   "pakistan_target.csv; pakistan_theft_cases.xlsx is an empty template). "
                   "Transfer needs both classes (>=3 each; realistically >=50 confirmed cases)."),
        "designed": True,
        "honesty_note": ("stage5_transfer.py is implemented and ready; it correctly writes PENDING "
                         "until a both-class Pakistan dataset exists. Not validated."),
    })

    from src.experiments.run_all_benchmark import print_consolidated_table
    print_consolidated_table()


if __name__ == "__main__":
    main()
