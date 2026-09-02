"""
Bonus refinement run (post-benchmark) — NOT part of the honest 5-stage table.

Reuses the FROZEN channel nets already trained in Stage 3
(models/checkpoints/*.pt) and:
  1. retrains the boosted residential CNN+BiLSTM with a stronger recipe
     (AdamW + weight decay + cosine LR decay + longer patience),
  2. retrains the Stage 1 XGBoost (fast) to obtain its probabilities,
  3. sweeps an ensemble weight w on the validation split
     (score = w*CNN + (1-w)*XGBoost) and evaluates that fixed w + threshold
     on the held-out test split.

Honesty rules kept:
  - original benchmark results are untouched; this writes under a separate
    'stage3_tuned_and_ensemble' key;
  - the ensemble weight is chosen on VALIDATION and reported as such;
  - the production checkpoint residential_model_best.pt is overwritten ONLY
    if the tuned model beats the original Stage 3 validation ROC-AUC.

Run (GPU strongly recommended):
    python -m src.experiments.tune_and_ensemble --config config/config.yaml
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

from src.agents.residential.model import ResidentialModel
from src.experiments.common import (
    load_config, load_split, classification_metrics, best_f1_threshold,
    save_stage_results, print_metrics, STAGE_CKPT_DIR, RESULTS_PATH,
)
from src.experiments.stage3_boosted import stack_split, predict_proba_residential
from src.experiments.stage5_transfer import load_channels
from src.experiments.stage1_xgboost import extract_features

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# improved recipe vs the original Stage 3 run (same architecture)
EPOCHS = 40
PATIENCE = 10
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64


def train_tuned(S_tr, y_tr, S_va, y_va, cfg):
    """Same ResidentialModel as Stage 3, better optimisation recipe."""
    mcfg = cfg["residential_model"]
    model = ResidentialModel(num_channels=S_tr.shape[1], seq_len=S_tr.shape[2],
                             cnn_filters=mcfg["cnn_filters"],
                             cnn_kernel_size=mcfg["cnn_kernel_size"],
                             lstm_hidden_units=mcfg["lstm_hidden_units"],
                             dropout=mcfg["dropout"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=mcfg["learning_rate"],
                            weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.BCELoss()
    y_t = torch.tensor(y_tr, dtype=torch.float32)

    ckpt = os.path.join(STAGE_CKPT_DIR, "residential_stage3_tuned.pt")
    best_auc, patience_counter = 0.0, 0
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(S_tr.shape[0])
        total_loss, n = 0.0, 0
        for i in range(0, S_tr.shape[0], BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb = S_tr[idx].to(DEVICE)
            yb = y_t[idx].to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
            n += xb.size(0)
        sched.step()

        prob = predict_proba_residential(model, S_va)
        val_auc = float(roc_auc_score(y_va, prob))
        print(f"[tuned residential] epoch {epoch+1}/{EPOCHS}  "
              f"loss={total_loss/n:.4f}  val_auc={val_auc:.4f}  "
              f"lr={sched.get_last_lr()[0]:.5f}")
        if val_auc > best_auc:
            best_auc, patience_counter = val_auc, 0
            torch.save(model.state_dict(), ckpt)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"[tuned residential] early stop at epoch {epoch+1}, "
                      f"best val_auc={best_auc:.4f}")
                break

    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    return model, best_auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    processed = cfg["paths"]["processed_dir"]
    prod_ckpt = cfg["paths"]["checkpoints_dir"]
    os.makedirs(STAGE_CKPT_DIR, exist_ok=True)
    print(f"[tune] device: {DEVICE}")

    X_tr, y_tr, ct_tr = load_split(processed, "train")
    X_va, y_va, ct_va = load_split(processed, "val")
    X_te, y_te, ct_te = load_split(processed, "test")

    # ---- frozen channels from the finished Stage 3 run (no re-training) ----
    seq_len = X_tr.shape[1]
    ae, pretext, freq = load_channels(seq_len, prod_ckpt)
    tr_res = ct_tr == "residential"
    va_res = ct_va == "residential"
    te_res = ct_te == "residential"

    print("[tune] building channel stacks ...")
    S_tr = stack_split(X_tr[tr_res], ae, pretext, freq)
    S_va = stack_split(X_va[va_res], ae, pretext, freq)
    S_te = stack_split(X_te[te_res], ae, pretext, freq)

    # ---- 1. tuned CNN+BiLSTM on the boosted stack ----
    model, best_auc = train_tuned(S_tr, y_tr[tr_res], S_va, y_va[va_res], cfg)
    prob_va = predict_proba_residential(model, S_va)
    prob_te = predict_proba_residential(model, S_te)

    tuned_val = classification_metrics(y_va[va_res], prob_va,
                                       best_f1_threshold(y_va[va_res], prob_va))
    tuned_val["pr_auc"] = round(float(average_precision_score(y_va[va_res], prob_va)), 4)
    print_metrics("Tuned boosted CNN — VALIDATION", tuned_val)

    # ---- 2. XGBoost probabilities (identical Stage 1 recipe) ----
    print("[tune] extracting features + retraining XGBoost ...")
    from xgboost import XGBClassifier
    F_tr, F_va, F_te = extract_features(X_tr), extract_features(X_va), extract_features(X_te)
    xgb = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=float((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1),
        eval_metric="auc", n_jobs=-1,
    )
    xgb.fit(F_tr, y_tr)
    xgb_va = xgb.predict_proba(F_va)[:, 1][va_res]
    xgb_te = xgb.predict_proba(F_te)[:, 1][te_res]

    # ---- 3. ensemble weight sweep on validation ----
    best_w, best_w_auc = 1.0, -1.0
    for w in np.arange(0.0, 1.01, 0.05):
        blend = w * prob_va + (1.0 - w) * xgb_va
        auc = float(roc_auc_score(y_va[va_res], blend))
        if auc > best_w_auc:
            best_w, best_w_auc = float(w), auc
    print(f"[tune] best ensemble weight w={best_w:.2f} (CNN) -> val_auc={best_w_auc:.4f}")

    blend_va = best_w * prob_va + (1.0 - best_w) * xgb_va
    blend_te = best_w * prob_te + (1.0 - best_w) * xgb_te
    thr = best_f1_threshold(y_va[va_res], blend_va)
    ens_val = classification_metrics(y_va[va_res], blend_va, thr)
    ens_val["pr_auc"] = round(float(average_precision_score(y_va[va_res], blend_va)), 4)
    ens_test = classification_metrics(y_te[te_res], blend_te, thr)
    ens_test["pr_auc"] = round(float(average_precision_score(y_te[te_res], blend_te)), 4)
    print_metrics("ENSEMBLE (tuned CNN + XGBoost) — VALIDATION", ens_val)
    print_metrics("ENSEMBLE (tuned CNN + XGBoost) — TEST", ens_test)

    # ---- update production checkpoint only if the tuned model is better ----
    prior_s3_auc = None
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            prior_s3_auc = (json.load(f).get("stage3_boosted", {})
                            .get("residential", {}).get("val_metrics", {})
                            .get("roc_auc"))
    replaced = False
    if prior_s3_auc is None or tuned_val["roc_auc"] > prior_s3_auc:
        torch.save(model.state_dict(), os.path.join(prod_ckpt, "residential_model_best.pt"))
        replaced = True
        print(f"[tune] production residential_model_best.pt REPLACED "
              f"(tuned {tuned_val['roc_auc']} > stage3 {prior_s3_auc})")
    else:
        print(f"[tune] production checkpoint kept — tuned model did not beat "
              f"Stage 3 ({tuned_val['roc_auc']} <= {prior_s3_auc})")

    save_stage_results("stage3_tuned_and_ensemble", {
        "note": "BONUS refinement run, separate from the honest 5-stage table. "
                "Ensemble weight w chosen on validation; test uses the fixed w.",
        "recipe": {"epochs": EPOCHS, "patience": PATIENCE,
                   "weight_decay": WEIGHT_DECAY, "scheduler": "cosine",
                   "optimizer": "AdamW"},
        "tuned_cnn_val": tuned_val,
        "ensemble_weight_cnn": best_w,
        "ensemble_val": ens_val,
        "ensemble_test": ens_test,
        "production_checkpoint_replaced": replaced,
    })


if __name__ == "__main__":
    main()
