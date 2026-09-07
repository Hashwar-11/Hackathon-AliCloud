"""
Stage 5 (Pakistan) — HONEST ZERO-SHOT TRANSFER EVALUATION.

WHY ZERO-SHOT, NOT FINE-TUNE
    The confirmed Pakistani target set (data/raw/pakistan/pakistan_target.csv) holds
    42 rows that are ALL theft (FLAG = 1) and ZERO normals. A discriminative fine-tune
    is mathematically impossible on a single class: ROC-AUC / precision are undefined and
    the weighted-BCE pos_weight (n_neg / n_pos) is 0, so every positive is ignored.
    Fine-tuning here would emit a degenerate model and overwrite good SGCC weights (this
    already corrupted residential_stage3.pt once — finetune_pakistan.py now ABORTS on
    single-class input instead).

    What IS defensible with 42 confirmed theft cases and no Pakistani training data:
    run the FROZEN SGCC-trained Stage 3 detector over them and measure DETECTION RECALL
    — the fraction of real, confirmed Pakistani theft cases that the China-trained model
    flags with zero Pakistani fine-tuning. That is a genuine cross-domain (China ->
    Pakistan) transfer result, reported honestly with its limitations and never as an AUC.

METHOD (inference only; reuses the exact, already-tested Stage 3 path)
    1. preprocess() the target CSV with the SAME per-row transform used for SGCC.
    2. front-pad 365 -> 1034 days with zeros (matching the production
       transform_daily_series alignment, so recent history sits at the end).
    3. stack_split() through the FROZEN autoencoder / masked-pretext / FFT channels.
    4. Load the FROZEN residential_model_best.pt (the Stage 3 Channel-Boosted detector).
    5. predict -> theft probability per consumer; report detection recall at the production
       threshold (0.65), the recovered Stage-3 best-F1 threshold, and 0.5, plus the spread.

Run:
    python -m src.experiments.stage5_zeroshot_pakistan --config config/config.yaml
    # add --no-write to compute + print without touching benchmark_results.json
"""
import argparse
import json
import os

import numpy as np
import torch

from src.preprocessing.preprocess import load_sgcc, preprocess
from src.experiments.common import load_config, save_stage_results, RESULTS_PATH
from src.experiments.stage3_boosted import stack_split, predict_proba_residential
from src.experiments.stage5_transfer import load_channels
from src.experiments.recover_stage_metrics import load_residential

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROD_THRESHOLD = 0.65  # is_theft_suspect cut used across the app / coordinator


def _stage3_best_threshold() -> float:
    """Recovered Stage-3 best-F1 threshold from benchmark_results.json (else 0.5)."""
    if os.path.exists(RESULTS_PATH):
        try:
            with open(RESULTS_PATH) as f:
                return float(json.load(f)["stage3_boosted"]["residential"]
                             ["val_metrics"]["threshold"])
        except Exception:
            pass
    return 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--target_csv", default="data/raw/pakistan/pakistan_target.csv")
    ap.add_argument("--no-write", action="store_true",
                    help="compute and print only; do not modify benchmark_results.json")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ckpt = cfg["paths"]["checkpoints_dir"]
    processed = cfg["paths"]["processed_dir"]
    print(f"[stage5-zs] device={DEVICE}  ckpt={ckpt}")

    if not os.path.exists(args.target_csv):
        raise SystemExit(
            f"[stage5-zs] target CSV not found: {args.target_csv!r}. Run first:\n"
            "  python -m src.experiments.prepare_pakistan_target "
            "--input pakistan_sgcc_format.xlsx")

    # ---- Load + preprocess the Pakistani target with the SAME SGCC transform ----
    df = load_sgcc(args.target_csv)
    X_pk, y_pk, ctype = preprocess(df, cfg)
    n = int(len(y_pk))
    n_theft = int((y_pk == 1).sum())
    n_normal = int((y_pk == 0).sum())
    n_res = int((ctype == "residential").sum())
    n_ind = int((ctype == "industrial").sum())
    print(f"[stage5-zs] target rows={n}  theft={n_theft}  normal={n_normal}  "
          f"| consumer-type proxy: residential={n_res} industrial={n_ind}")
    if n_theft == 0:
        raise SystemExit("[stage5-zs] no theft rows in target — nothing to evaluate.")
    if n_normal > 0:
        print(f"[stage5-zs] NOTE: {n_normal} normal row(s) present; recall is still computed "
              "on the theft subset (precision/AUC need many more normals to be meaningful).")

    # ---- Align 365 -> 1034 to match production transform_daily_series ----
    # IMPORTANT: FRONT-pad zeros (not tail). The deployed API left-pads so recent
    # history occupies the final timesteps; this eval must use the SAME alignment
    # or the reported recall would not reflect what the API actually produces.
    seq_len = int(np.load(os.path.join(processed, "X_val.npy"), mmap_mode="r").shape[1])
    raw_len = int(X_pk.shape[1])
    if raw_len != seq_len:
        if raw_len < seq_len:
            pad = seq_len - raw_len
            X_pk = np.pad(X_pk, ((0, 0), (pad, 0)), mode="constant", constant_values=0.0)
        else:
            X_pk = X_pk[:, -seq_len:]
    print(f"[stage5-zs] length aligned {raw_len} -> {seq_len} "
          "(FRONT-padded zeros = production alignment)")

    # ---- Frozen channel stack + frozen Stage 3 residential detector ----
    ae, pretext, freq = load_channels(seq_len, ckpt)
    S = stack_split(X_pk, ae, pretext, freq, chunk=max(1, min(n, 64)))
    num_channels = int(S.shape[1])
    s3_path = os.path.join(ckpt, "residential_model_best.pt")
    if not os.path.exists(s3_path):
        raise SystemExit(f"[stage5-zs] missing Stage 3 weights: {s3_path}")
    model = load_residential(s3_path, num_channels=num_channels, seq_len=seq_len, cfg=cfg)
    probs = predict_proba_residential(model, S, batch_size=args.batch_size)

    # ---- Detection recall on the confirmed theft subset ----
    theft_probs = probs[y_pk == 1]
    s3_thr = _stage3_best_threshold()

    def recall_at(thr: float) -> float:
        return float((theft_probs >= thr).mean())

    def flagged_at(thr: float) -> int:
        return int((theft_probs >= thr).sum())

    r_prod, r_s3, r_50 = recall_at(PROD_THRESHOLD), recall_at(s3_thr), recall_at(0.5)
    stats = {
        "mean": round(float(theft_probs.mean()), 4),
        "median": round(float(np.median(theft_probs)), 4),
        "min": round(float(theft_probs.min()), 4),
        "max": round(float(theft_probs.max()), 4),
        "std": round(float(theft_probs.std()), 4),
    }

    print("\n[stage5-zs] ===== ZERO-SHOT TRANSFER (China SGCC -> Pakistan), NO fine-tuning =====")
    print(f"[stage5-zs] confirmed theft cases evaluated: {n_theft}")
    print(f"[stage5-zs] detection recall @ production thr {PROD_THRESHOLD:.2f}: "
          f"{r_prod * 100:.1f}%  ({flagged_at(PROD_THRESHOLD)}/{n_theft} flagged)")
    print(f"[stage5-zs] detection recall @ Stage-3 best-F1 thr {s3_thr:.2f}: "
          f"{r_s3 * 100:.1f}%  ({flagged_at(s3_thr)}/{n_theft} flagged)")
    print(f"[stage5-zs] detection recall @ 0.50: "
          f"{r_50 * 100:.1f}%  ({flagged_at(0.5)}/{n_theft} flagged)")
    print(f"[stage5-zs] theft-score distribution: {stats}")

    if args.no_write:
        print("[stage5-zs] --no-write set: benchmark_results.json NOT modified.")
        return

    save_stage_results("stage5_transfer", {
        "status": "ZERO-SHOT EVALUATED",
        "method": "zero_shot_transfer",
        "trained_on_pakistan": False,
        "device": DEVICE,
        "target_csv": args.target_csv,
        "n_target": n,
        "target_classes": {"theft": n_theft, "normal": n_normal},
        "consumer_type_proxy": {"residential": n_res, "industrial": n_ind},
        "frozen_model": "residential_model_best.pt (SGCC Stage 3, 4-channel boosted)",
        "sequence_length_adaptation": f"{raw_len} -> {seq_len} (FRONT-padded zeros = production alignment)",
        "detection_recall": {
            "at_production_threshold_0.65": round(r_prod, 4),
            "at_stage3_best_f1_threshold": round(r_s3, 4),
            "stage3_best_f1_threshold_value": s3_thr,
            "at_0.50": round(r_50, 4),
        },
        "flagged_counts": {
            "at_0.65": flagged_at(PROD_THRESHOLD),
            "at_stage3_thr": flagged_at(s3_thr),
            "at_0.50": flagged_at(0.5),
            "of_total": n_theft,
        },
        "theft_score_distribution": stats,
        "why_not_finetuned": (
            "Single class: 42 confirmed theft, 0 Pakistani normals. A discriminative "
            "fine-tune is undefined here (ROC-AUC/precision need both classes; BCE "
            "pos_weight = n_neg/n_pos = 0). finetune_pakistan.py now ABORTS on single-class "
            "input instead of writing a degenerate model over the good SGCC weights."),
        "limitations": [
            "No Pakistani normals -> precision, specificity and ROC-AUC are undefined; only "
            "recall (detection rate) on confirmed theft is measurable.",
            "Target series are 365 days FRONT-padded to the SGCC 1034-day window, matching "
            "the production transform_daily_series alignment, so this recall reflects the "
            "deployed preprocessing; 365d of real history is still shorter than the 1034d "
            "the model was trained on.",
            "All rows are scored by the residential Stage 3 detector (the only validated "
            "discriminative model); the industrial autoencoder is anomaly-based and was not "
            "recovered, and the >=500 kWh/day proxy marks most of these accounts 'industrial'.",
            "Zero-shot: the model saw NO Pakistani data. A fine-tune stays PENDING until a "
            "both-class Pakistani set exists (>=50 confirmed theft + matching normals).",
        ],
        "honesty_note": (
            "Genuine China->Pakistan ZERO-SHOT transfer measurement on 42 user-confirmed theft "
            "cases (FLAG = -1 == theft, per the data owner). NOT a fine-tuned result; reported "
            "as detection recall, never as AUC."),
    })
    print("[stage5-zs] Wrote stage5_transfer (zero-shot) -> benchmark_results.json")


if __name__ == "__main__":
    main()
