"""
Stage 3 — the Stage 2 detectors WITH Channel Boosting.

Pipeline:
  1. Pretrain the 3 auxiliary channel nets on the TRAIN split only
     (autoencoder on normals; pretext + frequency on all rows) and freeze.
  2. Stack raw + 3 channels -> (batch, 4, seq_len) for every split.
  3. Retrain the residential 1D-CNN + BiLSTM and the industrial autoencoder
     on the stacked input.
  4. Report the same metrics as Stages 1-2 and the delta vs Stage 2 — the
     point of this stage is to measure whether Channel Boosting helps, not
     just to add it.

Channel checkpoints are saved to models/checkpoints/ (the production names)
so the FastAPI service can use them; the boosted detector heads go to
models/stage_checkpoints/ AND the production names.

Run:
    python -m src.experiments.stage3_boosted --config config/config.yaml
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from src.channels.autoencoder_channel import ResidualAutoencoder, train_on_normals
from src.channels.pretext_channel import MaskedPretextEncoder, train_pretext
from src.channels.frequency_channel import FrequencyProjection, train_frequency_projection
from src.channels.stack_channels import build_channel_stack
from src.agents.residential.model import ResidentialModel
from src.agents.industrial.model import IndustrialAutoencoder, derive_tau
from src.experiments.common import (
    load_config, load_split, classification_metrics, best_f1_threshold,
    save_stage_results, print_metrics, STAGE_CKPT_DIR, RESULTS_PATH,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def pretrain_channels(X_train_np, y_train, cfg, ckpt_dir):
    """Pretrain + freeze the 3 auxiliary nets. Returns (ae, pretext, freq)."""
    ch = cfg.get("channel_training", cfg["residential_model"])
    seq_len = X_train_np.shape[1]
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    X_normal = X_train[torch.tensor(y_train == 0)]

    print(f"[stage3] pretraining Channel 2 (residual AE) on {X_normal.shape[0]} normal rows ...")
    ae = train_on_normals(ResidualAutoencoder(seq_len=seq_len), X_normal,
                          epochs=ch["epochs"], lr=ch["learning_rate"],
                          batch_size=ch["batch_size"], device=DEVICE)
    torch.save(ae.state_dict(), os.path.join(ckpt_dir, "autoencoder_channel.pt"))

    print(f"[stage3] pretraining Channel 3 (masked pretext) on all {X_train.shape[0]} rows ...")
    pretext = train_pretext(MaskedPretextEncoder(seq_len=seq_len), X_train,
                            epochs=ch["epochs"], lr=ch["learning_rate"],
                            batch_size=ch["batch_size"], device=DEVICE)
    torch.save(pretext.state_dict(), os.path.join(ckpt_dir, "pretext_channel.pt"))

    print("[stage3] pretraining Channel 4 (learned FFT projection) on all rows ...")
    freq = train_frequency_projection(FrequencyProjection(seq_len=seq_len), X_train,
                                      epochs=ch["epochs"], lr=ch["learning_rate"],
                                      batch_size=ch["batch_size"], device=DEVICE)
    torch.save(freq.state_dict(), os.path.join(ckpt_dir, "frequency_channel.pt"))

    for net in (ae, pretext, freq):
        net.eval()
        for p in net.parameters():
            p.requires_grad = False
    return ae, pretext, freq


def stack_split(X_np, ae, pretext, freq, chunk: int = 4096) -> torch.Tensor:
    """Build the frozen (n, 4, seq_len) stack for a whole split, chunked."""
    parts = []
    for i in range(0, len(X_np), chunk):
        xb = torch.tensor(X_np[i:i + chunk], dtype=torch.float32).to(DEVICE)
        parts.append(build_channel_stack(xb, ae, pretext, freq).cpu())
    return torch.cat(parts, dim=0)


def predict_proba_residential(model, stacked: torch.Tensor, batch_size: int = 512) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, stacked.shape[0], batch_size):
            out.append(model(stacked[i:i + batch_size].to(DEVICE)).cpu().numpy())
    return np.concatenate(out)


def train_residential_boosted(S_train, y_train, S_val, y_val, cfg, tag="stage3"):
    mcfg = cfg["residential_model"]
    model = ResidentialModel(num_channels=S_train.shape[1], seq_len=S_train.shape[2],
                             cnn_filters=mcfg["cnn_filters"],
                             cnn_kernel_size=mcfg["cnn_kernel_size"],
                             lstm_hidden_units=mcfg["lstm_hidden_units"],
                             dropout=mcfg["dropout"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=mcfg["learning_rate"])
    loss_fn = nn.BCELoss()
    y_t = torch.tensor(y_train, dtype=torch.float32)

    best_auc, patience_counter = 0.0, 0
    ckpt = os.path.join(STAGE_CKPT_DIR, f"residential_{tag}.pt")
    for epoch in range(mcfg["epochs"]):
        model.train()
        perm = torch.randperm(S_train.shape[0])
        total_loss, n = 0.0, 0
        for i in range(0, S_train.shape[0], mcfg["batch_size"]):
            idx = perm[i:i + mcfg["batch_size"]]
            xb = S_train[idx].to(DEVICE)
            yb = y_t[idx].to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
            n += xb.size(0)

        prob = predict_proba_residential(model, S_val)
        val_auc = float(roc_auc_score(y_val, prob))
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
    return model, best_auc


def train_industrial_boosted(S_tr_norm, S_va_all, y_va_all, cfg, tag="stage3"):
    """Boosted industrial AE. Same tau logic and honesty rules as Stage 2."""
    icfg = cfg["industrial_model"]
    n_tr = S_tr_norm.shape[0]
    if n_tr < 4:
        return {"status": "SKIPPED",
                "reason": f"only {n_tr} normal industrial-typed training rows",
                "n_train_normals": int(n_tr)}

    rng = np.random.default_rng(cfg["preprocessing"]["random_seed"])
    perm = rng.permutation(n_tr)
    n_tau = max(1, int(0.2 * n_tr))
    S_tau, S_fit = S_tr_norm[perm[:n_tau]], S_tr_norm[perm[n_tau:]]

    model = IndustrialAutoencoder(num_channels=S_fit.shape[1], seq_len=S_fit.shape[2],
                                  latent_dim=icfg["autoencoder_latent_dim"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=icfg["learning_rate"])
    loss_fn = nn.MSELoss()

    for epoch in range(icfg["epochs"]):
        model.train()
        p = torch.randperm(S_fit.shape[0])
        total_loss = 0.0
        for i in range(0, S_fit.shape[0], icfg["batch_size"]):
            batch = S_fit[p[i:i + icfg["batch_size"]]].to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.size(0)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[{tag} industrial] epoch {epoch+1}/{icfg['epochs']}  "
                  f"loss={total_loss/S_fit.shape[0]:.6f}")

    tau = derive_tau(model, S_tau.to(DEVICE), percentile=icfg["tau_percentile"])
    err = model.reconstruction_error(S_va_all.to(DEVICE)).cpu().numpy()

    def dist(e):
        if len(e) == 0:
            return {"n": 0}
        return {"n": int(len(e)), "mean": float(np.mean(e)), "median": float(np.median(e)),
                "std": float(np.std(e)), "p95": float(np.percentile(e, 95))}

    metrics = None
    if len(np.unique(y_va_all)) > 1:
        metrics = classification_metrics(y_va_all, err, threshold=None)
        from sklearn.metrics import precision_score, recall_score, f1_score
        flag = (err > tau).astype(int)
        metrics["precision"] = round(float(precision_score(y_va_all, flag, zero_division=0)), 4)
        metrics["recall"] = round(float(recall_score(y_va_all, flag, zero_division=0)), 4)
        metrics["f1"] = round(float(f1_score(y_va_all, flag, zero_division=0)), 4)
        metrics.pop("threshold", None)
        metrics["tau"] = float(tau)
        metrics["flagged_above_tau"] = int(flag.sum())

    torch.save(model.state_dict(), os.path.join(STAGE_CKPT_DIR, f"industrial_{tag}.pt"))
    with open(os.path.join(STAGE_CKPT_DIR, f"industrial_tau_{tag}.txt"), "w") as f:
        f.write(str(float(tau)))

    return {
        "status": "trained",
        "n_train_normals": int(n_tr),
        "n_tau_holdout": int(n_tau),
        "n_val_normals": int((y_va_all == 0).sum()),
        "n_val_theft": int((y_va_all == 1).sum()),
        "tau": float(tau),
        "recon_error_distribution": {
            "normal_val": dist(err[y_va_all == 0]),
            "theft_val": dist(err[y_va_all == 1]),
        },
        "val_metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    processed = cfg["paths"]["processed_dir"]
    prod_ckpt = cfg["paths"]["checkpoints_dir"]
    os.makedirs(STAGE_CKPT_DIR, exist_ok=True)
    os.makedirs(prod_ckpt, exist_ok=True)
    print(f"[stage3] device: {DEVICE}")

    X_tr, y_tr, ct_tr = load_split(processed, "train")
    X_va, y_va, ct_va = load_split(processed, "val")
    X_te, y_te, ct_te = load_split(processed, "test")

    ae, pretext, freq = pretrain_channels(X_tr, y_tr, cfg, prod_ckpt)

    print("[stage3] building channel stacks (this materialises ~1 GB for train) ...")
    tr_res = (ct_tr == "residential")
    va_res = (ct_va == "residential")
    te_res = (ct_te == "residential")
    S_tr_res = stack_split(X_tr[tr_res], ae, pretext, freq)
    S_va_res = stack_split(X_va[va_res], ae, pretext, freq)
    S_te_res = stack_split(X_te[te_res], ae, pretext, freq)

    # ---- Residential, boosted ----
    model, best_auc = train_residential_boosted(
        S_tr_res, y_tr[tr_res], S_va_res, y_va[va_res], cfg, tag="stage3")
    prob_va = predict_proba_residential(model, S_va_res)
    prob_te = predict_proba_residential(model, S_te_res)
    thr = best_f1_threshold(y_va[va_res], prob_va)
    val_m = classification_metrics(y_va[va_res], prob_va, thr)
    val_m_05 = classification_metrics(y_va[va_res], prob_va, 0.5)
    test_m = classification_metrics(y_te[te_res], prob_te, thr)
    print_metrics("Stage 3 residential (boosted) — VALIDATION", val_m)
    print_metrics("Stage 3 residential (boosted) — validation @0.5", val_m_05)
    print_metrics("Stage 3 residential (boosted) — TEST", test_m)
    # production-compatible checkpoint
    torch.save(model.state_dict(), os.path.join(prod_ckpt, "residential_model_best.pt"))

    # ---- Industrial, boosted ----
    tr_ind_norm = (ct_tr == "industrial") & (y_tr == 0)
    va_ind_all = (ct_va == "industrial")
    te_ind_all = (ct_te == "industrial")
    print(f"[stage3] INDUSTRIAL SAMPLE COUNTS — train normals: {tr_ind_norm.sum()}, "
          f"val industrial: {va_ind_all.sum()} (normals {(y_va[va_ind_all] == 0).sum()}, "
          f"theft {y_va[va_ind_all].sum()}), test industrial: {te_ind_all.sum()}")
    S_tr_ind = stack_split(X_tr[tr_ind_norm], ae, pretext, freq)
    S_va_ind = stack_split(X_va[va_ind_all], ae, pretext, freq)
    ind_res = train_industrial_boosted(S_tr_ind, S_va_ind, y_va[va_ind_all], cfg, tag="stage3")
    print(f"[stage3] industrial result: {json.dumps(ind_res, indent=2)}")
    if ind_res["status"] == "trained":
        import shutil
        shutil.copy(os.path.join(STAGE_CKPT_DIR, "industrial_stage3.pt"),
                    os.path.join(prod_ckpt, "industrial_model.pt"))
        shutil.copy(os.path.join(STAGE_CKPT_DIR, "industrial_tau_stage3.txt"),
                    os.path.join(prod_ckpt, "industrial_tau.txt"))

    # ---- Compare vs Stage 2 ----
    comparison = None
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            prior = json.load(f)
        s2 = prior.get("stage2_raw", {})
        s2_res = s2.get("residential", {}).get("val_metrics")
        if s2_res:
            comparison = {
                "note": "Stage 3 (Channel Boosting) vs Stage 2 (raw only), same val split",
                "stage2_residential_val": s2_res,
                "stage3_residential_val": val_m,
                "delta_roc_auc": round(val_m["roc_auc"] - s2_res["roc_auc"], 4),
                "delta_f1": round(val_m["f1"] - s2_res["f1"], 4),
            }
            print(f"[stage3] vs Stage 2 — delta ROC-AUC: {comparison['delta_roc_auc']:+.4f}, "
                  f"delta F1: {comparison['delta_f1']:+.4f}")
        s2_ind = s2.get("industrial", {})
        if ind_res.get("val_metrics") and s2_ind.get("val_metrics"):
            comparison["stage2_industrial_val_auc"] = s2_ind["val_metrics"]["roc_auc"]
            comparison["stage3_industrial_val_auc"] = ind_res["val_metrics"]["roc_auc"]

    save_stage_results("stage3_boosted", {
        "device": DEVICE,
        "residential": {
            "val_metrics": val_m, "val_metrics_thr05": val_m_05, "test_metrics": test_m,
            "best_val_auc_during_training": round(best_auc, 4),
            "n_train": int(tr_res.sum()), "n_val": int(va_res.sum()), "n_test": int(te_res.sum()),
        },
        "industrial": ind_res,
        "industrial_sample_counts": {
            "train_normals": int(tr_ind_norm.sum()),
            "val_industrial_total": int(va_ind_all.sum()),
            "test_industrial_total": int(te_ind_all.sum()),
        },
        "comparison_vs_stage2": comparison,
    })


if __name__ == "__main__":
    main()
