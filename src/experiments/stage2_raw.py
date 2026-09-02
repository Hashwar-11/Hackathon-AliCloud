"""
Stage 2 — single deep models on the RAW daily sequence (no Channel Boosting).

Two detectors:
  1. Residential: 1D-CNN + BiLSTM binary classifier, fed (batch, 1, seq_len).
  2. Industrial : autoencoder trained ONLY on normal industrial-typed rows;
                  reconstruction error is the anomaly score, tau = 95th
                  percentile of error on a held-out normal subset.

Evaluated on the untouched (imbalanced) validation split. Checkpoints go to
models/stage_checkpoints/ (the production pipeline keeps its own in
models/checkpoints/). Results + comparison vs Stage 1 are appended to
experiments_results/benchmark_results.json.

Run:
    python -m src.experiments.stage2_raw --config config/config.yaml
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from src.agents.residential.model import ResidentialModel
from src.agents.industrial.model import IndustrialAutoencoder, derive_tau
from src.experiments.common import (
    load_config, load_split, classification_metrics, best_f1_threshold,
    save_stage_results, print_metrics, STAGE_CKPT_DIR, RESULTS_PATH,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def predict_proba_residential(model, X: torch.Tensor, batch_size: int = 512) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            xb = X[i:i + batch_size].unsqueeze(1).to(DEVICE)  # (B, 1, T) raw channel
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out)


def train_residential(X_train, y_train, X_val, y_val, cfg, tag="stage2"):
    mcfg = cfg["residential_model"]
    seq_len = X_train.shape[1]
    model = ResidentialModel(num_channels=1, seq_len=seq_len,
                             cnn_filters=mcfg["cnn_filters"],
                             cnn_kernel_size=mcfg["cnn_kernel_size"],
                             lstm_hidden_units=mcfg["lstm_hidden_units"],
                             dropout=mcfg["dropout"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=mcfg["learning_rate"])
    loss_fn = nn.BCELoss()
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    X_v = torch.tensor(X_val, dtype=torch.float32)

    best_auc, patience_counter = 0.0, 0
    ckpt = os.path.join(STAGE_CKPT_DIR, f"residential_{tag}.pt")
    history = []
    for epoch in range(mcfg["epochs"]):
        model.train()
        perm = torch.randperm(X_t.shape[0])
        total_loss, n = 0.0, 0
        for i in range(0, X_t.shape[0], mcfg["batch_size"]):
            idx = perm[i:i + mcfg["batch_size"]]
            xb = X_t[idx].unsqueeze(1).to(DEVICE)
            yb = y_t[idx].to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
            n += xb.size(0)

        prob = predict_proba_residential(model, X_v)
        val_auc = float(roc_auc_score(y_val, prob))
        history.append({"epoch": epoch + 1, "train_loss": total_loss / n, "val_auc": val_auc})
        print(f"[{tag} residential] epoch {epoch+1}/{mcfg['epochs']}  "
              f"loss={total_loss/n:.4f}  val_auc={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc, patience_counter = val_auc, 0
            torch.save(model.state_dict(), ckpt)
        else:
            patience_counter += 1
            if patience_counter >= mcfg["early_stopping_patience"]:
                print(f"[{tag} residential] early stop at epoch {epoch+1}, best val_auc={best_auc:.4f}")
                break

    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    return model, best_auc, history


def train_industrial(X_tr_norm, X_va_norm, X_va_all, y_va_all, cfg, tag="stage2"):
    """AE on normal industrial rows only. Returns metrics dict + error distribution."""
    icfg = cfg["industrial_model"]
    n_tr = X_tr_norm.shape[0]
    seq_len = X_tr_norm.shape[1]

    if n_tr < 4:
        return {
            "status": "SKIPPED",
            "reason": f"only {n_tr} normal industrial-typed training rows — "
                      "not enough to train an autoencoder or derive tau",
            "n_train_normals": int(n_tr),
        }

    # Hold out 20% of the (tiny) normal pool purely for tau derivation.
    rng = np.random.default_rng(cfg["preprocessing"]["random_seed"])
    perm = rng.permutation(n_tr)
    n_tau = max(1, int(0.2 * n_tr))
    X_tau, X_fit = X_tr_norm[perm[:n_tau]], X_tr_norm[perm[n_tau:]]

    model = IndustrialAutoencoder(num_channels=1, seq_len=seq_len,
                                  latent_dim=icfg["autoencoder_latent_dim"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=icfg["learning_rate"])
    loss_fn = nn.MSELoss()
    X_fit_t = torch.tensor(X_fit, dtype=torch.float32)

    for epoch in range(icfg["epochs"]):
        model.train()
        p = torch.randperm(X_fit_t.shape[0])
        total_loss = 0.0
        for i in range(0, X_fit_t.shape[0], icfg["batch_size"]):
            batch = X_fit_t[p[i:i + icfg["batch_size"]]].unsqueeze(1).to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.size(0)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[{tag} industrial] epoch {epoch+1}/{icfg['epochs']}  "
                  f"loss={total_loss/X_fit_t.shape[0]:.6f}")

    tau = derive_tau(model, torch.tensor(X_tau, dtype=torch.float32).unsqueeze(1).to(DEVICE),
                     percentile=icfg["tau_percentile"])

    # Reconstruction-error distribution on the val split (industrial-typed, both labels)
    err = model.reconstruction_error(
        torch.tensor(X_va_all, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    ).cpu().numpy()
    err_norm = err[y_va_all == 0]
    err_theft = err[y_va_all == 1]

    def dist(e):
        if len(e) == 0:
            return {"n": 0}
        return {
            "n": int(len(e)), "mean": float(np.mean(e)), "median": float(np.median(e)),
            "std": float(np.std(e)), "p95": float(np.percentile(e, 95)),
        }

    flagged = int((err > tau).sum())
    metrics = None
    if len(np.unique(y_va_all)) > 1:
        metrics = classification_metrics(y_va_all, err, threshold=None)  # AUC uses err as score
        metrics["flagged_above_tau"] = flagged
        flag_pred = (err > tau).astype(int)
        from sklearn.metrics import precision_score, recall_score, f1_score
        metrics["precision"] = round(float(precision_score(y_va_all, flag_pred, zero_division=0)), 4)
        metrics["recall"] = round(float(recall_score(y_va_all, flag_pred, zero_division=0)), 4)
        metrics["f1"] = round(float(f1_score(y_va_all, flag_pred, zero_division=0)), 4)
        metrics.pop("threshold", None)
        metrics["tau"] = float(tau)

    torch.save(model.state_dict(), os.path.join(STAGE_CKPT_DIR, f"industrial_{tag}.pt"))
    with open(os.path.join(STAGE_CKPT_DIR, f"industrial_tau_{tag}.txt"), "w") as f:
        f.write(str(float(tau)))

    return {
        "status": "trained",
        "n_train_normals": int(n_tr),
        "n_train_fit": int(len(X_fit)),
        "n_tau_holdout": int(len(X_tau)),
        "n_val_normals": int((y_va_all == 0).sum()),
        "n_val_theft": int((y_va_all == 1).sum()),
        "tau": float(tau),
        "flagged_above_tau": flagged,
        "recon_error_distribution": {"normal_val": dist(err_norm), "theft_val": dist(err_theft)},
        "val_metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    processed = cfg["paths"]["processed_dir"]
    os.makedirs(STAGE_CKPT_DIR, exist_ok=True)
    print(f"[stage2] device: {DEVICE}")

    X_tr, y_tr, ct_tr = load_split(processed, "train")
    X_va, y_va, ct_va = load_split(processed, "val")
    X_te, y_te, ct_te = load_split(processed, "test")

    # ---- Residential (raw sequence, single channel) ----
    tr_res = (ct_tr == "residential")
    va_res = (ct_va == "residential")
    te_res = (ct_te == "residential")
    print(f"[stage2] residential rows — train: {tr_res.sum()} "
          f"(pos {y_tr[tr_res].sum()}), val: {va_res.sum()} (pos {y_va[va_res].sum()}), "
          f"test: {te_res.sum()} (pos {y_te[te_res].sum()})")

    model, best_auc, _ = train_residential(
        X_tr[tr_res], y_tr[tr_res], X_va[va_res], y_va[va_res], cfg, tag="stage2")

    prob_va = predict_proba_residential(model, torch.tensor(X_va[va_res], dtype=torch.float32))
    prob_te = predict_proba_residential(model, torch.tensor(X_te[te_res], dtype=torch.float32))
    thr = best_f1_threshold(y_va[va_res], prob_va)
    val_m = classification_metrics(y_va[va_res], prob_va, thr)
    val_m_05 = classification_metrics(y_va[va_res], prob_va, 0.5)
    test_m = classification_metrics(y_te[te_res], prob_te, thr)
    print_metrics("Stage 2 residential (raw) — VALIDATION", val_m)
    print_metrics("Stage 2 residential (raw) — validation @0.5", val_m_05)
    print_metrics("Stage 2 residential (raw) — TEST", test_m)

    # ---- Industrial (autoencoder, anomaly score) ----
    tr_ind_norm = (ct_tr == "industrial") & (y_tr == 0)
    va_ind_norm = (ct_va == "industrial") & (y_va == 0)
    va_ind_all = (ct_va == "industrial")
    te_ind_all = (ct_te == "industrial")
    print(f"[stage2] INDUSTRIAL SAMPLE COUNTS — train normals: {tr_ind_norm.sum()}, "
          f"val industrial total: {va_ind_all.sum()} (normals {va_ind_norm.sum()}, "
          f"theft {y_va[va_ind_all].sum()}), test industrial total: {te_ind_all.sum()}")

    ind_res = train_industrial(
        X_tr[tr_ind_norm], X_va[va_ind_norm], X_va[va_ind_all], y_va[va_ind_all],
        cfg, tag="stage2")
    print(f"[stage2] industrial result: {json.dumps(ind_res, indent=2)}")

    # Save val probabilities for the Stage 4 scoring report.
    np.savez(os.path.join(STAGE_CKPT_DIR, "stage2_val_probs.npz"),
             res_prob=prob_va, res_y=y_va[va_res],
             ind_err=np.array([]), ind_y=np.array([]))

    # ---- Compare vs Stage 1 ----
    comparison = None
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            prior = json.load(f)
        s1 = prior.get("stage1_xgboost", {}).get("val_metrics")
        if s1:
            comparison = {
                "note": "Stage 2 residential (raw) vs Stage 1 XGBoost, both on the "
                        "same imbalanced validation split",
                "stage1_val": s1,
                "stage2_residential_val": val_m,
                "delta_roc_auc": round(val_m["roc_auc"] - s1["roc_auc"], 4),
                "delta_f1": round(val_m["f1"] - s1["f1"], 4),
            }
            print(f"[stage2] vs Stage 1 — delta ROC-AUC: {comparison['delta_roc_auc']:+.4f}, "
                  f"delta F1: {comparison['delta_f1']:+.4f}")

    save_stage_results("stage2_raw", {
        "device": DEVICE,
        "residential": {
            "val_metrics": val_m, "val_metrics_thr05": val_m_05, "test_metrics": test_m,
            "best_val_auc_during_training": round(best_auc, 4),
            "n_train": int(tr_res.sum()), "n_val": int(va_res.sum()), "n_test": int(te_res.sum()),
        },
        "industrial": ind_res,
        "comparison_vs_stage1": comparison,
    })


if __name__ == "__main__":
    main()
