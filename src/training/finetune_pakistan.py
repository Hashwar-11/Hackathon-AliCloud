"""
Fine-tunes the SGCC-pretrained Residential and Industrial Agent models
on the Pakistani dataset (data/raw/pakistan/pakistan_target.csv).

The Pakistani CSV is in SGCC-style format:
    CONS_NO, FLAG, day_1, day_2, ..., day_N
where FLAG = -1 means THEFT, FLAG = 0 means NORMAL.

Fine-tuning strategy:
    - Load SGCC-pretrained model weights from checkpoints_dir
    - Freeze the 3 channel networks (already frozen from pretrain)
    - Run short fine-tuning with low LR + weight_decay regularisation
    - Save updated weights as *_stage3.pt (matches existing naming convention)

Usage (run from repo root):
    python -m src.training.finetune_pakistan --config config/config.yaml

NOTE: ~43 Pakistan rows is a tiny dataset. We fine-tune on the full set and
      evaluate on the same set (no held-out split). Report this limitation.
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score, f1_score, classification_report  # noqa: F401

from src.channels.stack_channels import build_channel_stack
from src.agents.residential.model import ResidentialModel
from src.agents.industrial.model import IndustrialAutoencoder, derive_tau
from src.training.train_residential import load_frozen_channels


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Load and preprocess the Pakistani dataset
# ─────────────────────────────────────────────────────────────────────────────

def load_pakistan_data(csv_path: str, target_seq_len: int, outlier_sigma_cap: float = 2.0):
    """
    Reads pakistan_target.csv and returns (X, y) tensors aligned to
    the SGCC-trained model's expected seq_len.

    FLAG = -1  -> theft  -> label = 1
    FLAG =  0  -> normal -> label = 0

    Rows with more than 50% missing days are dropped.
    Remaining NaNs are forward-filled then zero-filled.
    Sequences are trimmed (last N days) or zero-padded to match target_seq_len.
    """
    df = pd.read_csv(csv_path)

    meta_cols = ["CONS_NO", "FLAG"]
    day_cols  = [c for c in df.columns if c not in meta_cols]

    # Drop rows with >50% missing readings
    consumption = df[day_cols].astype(float)
    missing_frac = consumption.isna().mean(axis=1)
    df = df[missing_frac <= 0.5].reset_index(drop=True)
    consumption = df[day_cols].astype(float)

    print(f"[Pakistan] Rows after dropping >50% NaN: {len(df)}")

    # Forward-fill per row, then zero-fill
    arr = consumption.values.astype(np.float64)
    for i in range(arr.shape[0]):
        row = arr[i]
        # forward fill
        last_val = 0.0
        for j in range(len(row)):
            if np.isnan(row[j]):
                row[j] = last_val
            else:
                last_val = row[j]
        # backward fill remaining NaNs at start
        last_val = 0.0
        for j in range(len(row) - 1, -1, -1):
            if np.isnan(row[j]):
                row[j] = last_val
            else:
                last_val = row[j]
        arr[i] = row

    consumption = arr.astype(np.float32)

    # Outlier cap per sample: clip at mean + sigma_cap * std
    for i in range(consumption.shape[0]):
        row = consumption[i]
        mu, sigma = row.mean(), row.std()
        if sigma > 0:
            consumption[i] = np.clip(row, 0.0, mu + outlier_sigma_cap * sigma)

    # Per-row min-max scale to [0, 1]
    row_min = consumption.min(axis=1, keepdims=True)
    row_max = consumption.max(axis=1, keepdims=True)
    row_range = np.where(row_max - row_min == 0, 1.0, row_max - row_min)
    consumption = (consumption - row_min) / row_range

    # Align sequence length
    T = consumption.shape[1]
    if T >= target_seq_len:
        consumption = consumption[:, -target_seq_len:]   # take last N days
    else:
        pad = np.zeros((consumption.shape[0], target_seq_len - T), dtype=np.float32)
        consumption = np.concatenate([pad, consumption], axis=1)

    # Labels: FLAG = -1 → theft=1, otherwise 0
    flags  = df["FLAG"].astype(int).values
    labels = np.where(flags == -1, 1, 0).astype(np.float32)

    print(f"[Pakistan] Label distribution - Normal: {(labels==0).sum()}, Theft: {(labels==1).sum()}")

    X = torch.tensor(consumption, dtype=torch.float32)
    y = torch.tensor(labels,      dtype=torch.float32)
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Fine-tune Residential Model
# ─────────────────────────────────────────────────────────────────────────────

def finetune_residential(X, y, cfg, ckpt_dir, device):
    """
    Loads residential_model_best.pt and continues training on Pakistani data.
    Saves as residential_stage3.pt.
    """
    seq_len = X.shape[1]
    autoencoder, pretext, freq_proj = load_frozen_channels(seq_len, ckpt_dir, device)

    ccfg = cfg["channels"]
    mcfg = cfg["residential_model"]
    num_channels = 1 + sum([
        ccfg["use_autoencoder_residual"],
        ccfg["use_pretext_embedding"],
        ccfg["use_frequency_projection"]
    ])

    model = ResidentialModel(
        num_channels=num_channels, seq_len=seq_len,
        cnn_filters=mcfg["cnn_filters"], cnn_kernel_size=mcfg["cnn_kernel_size"],
        lstm_hidden_units=mcfg["lstm_hidden_units"], dropout=mcfg["dropout"],
    ).to(device)

    pretrained_path = os.path.join(ckpt_dir, "residential_model_best.pt")
    model.load_state_dict(torch.load(pretrained_path, map_location=device))
    print(f"[Residential] Loaded pretrained weights from {pretrained_path}")

    X_ft = X.to(device)
    y_ft = y.to(device)

    # Fine-tuning hyperparams
    ft_lr     = mcfg["learning_rate"] * 0.1    # 10x lower than SGCC training
    ft_epochs = 20
    ft_wd     = 1e-4
    ft_bs     = min(8, X.shape[0])

    # Weighted BCE for class imbalance
    n_pos = y_ft.sum().item()
    n_neg = (y_ft == 0).sum().item()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    loss_fn_weighted = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.Adam(model.parameters(), lr=ft_lr, weight_decay=ft_wd)

    print("\n[Residential] Starting fine-tune on Pakistan data...")
    model.train()

    for epoch in range(ft_epochs):
        perm = torch.randperm(X_ft.shape[0])
        total_loss = 0.0

        for i in range(0, X_ft.shape[0], ft_bs):
            idx = perm[i:i + ft_bs]
            xb  = X_ft[idx]
            yb  = y_ft[idx]

            with torch.no_grad():
                stacked = build_channel_stack(
                    xb, autoencoder, pretext, freq_proj,
                    ccfg["use_autoencoder_residual"],
                    ccfg["use_pretext_embedding"],
                    ccfg["use_frequency_projection"]
                )

            opt.zero_grad()
            probs  = model(stacked)
            # Convert sigmoid output back to logits for BCEWithLogitsLoss
            logits = torch.log(probs.clamp(1e-7, 1 - 1e-7) / (1 - probs.clamp(1e-7, 1 - 1e-7)))
            loss   = loss_fn_weighted(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * xb.size(0)

        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                stacked_all = build_channel_stack(
                    X_ft, autoencoder, pretext, freq_proj,
                    ccfg["use_autoencoder_residual"],
                    ccfg["use_pretext_embedding"],
                    ccfg["use_frequency_projection"]
                )
                all_probs = model(stacked_all).cpu().numpy()
            y_np = y.numpy().astype(int)
            preds_bin = (all_probs > 0.5).astype(int)
            try:
                auc = roc_auc_score(y_np, all_probs)
                f1  = f1_score(y_np, preds_bin, zero_division=0)
                print(f"  Epoch {epoch+1}/{ft_epochs}  loss={total_loss/X_ft.shape[0]:.4f}  "
                      f"auc={auc:.4f}  f1={f1:.4f}")
            except ValueError:
                print(f"  Epoch {epoch+1}/{ft_epochs}  loss={total_loss/X_ft.shape[0]:.4f}  "
                      f"(AUC undefined — only one class in labels)")
            model.train()

    # Final report
    model.eval()
    with torch.no_grad():
        stacked_all = build_channel_stack(
            X_ft, autoencoder, pretext, freq_proj,
            ccfg["use_autoencoder_residual"],
            ccfg["use_pretext_embedding"],
            ccfg["use_frequency_projection"]
        )
        all_probs = model(stacked_all).cpu().numpy()

    y_np = y.numpy().astype(int)
    print("\n[Residential] Final Pakistan evaluation (trained & evaluated on same set):")
    print(classification_report(y_np, (all_probs > 0.5).astype(int),
                                 target_names=["Normal", "Theft"], zero_division=0))

    save_path = os.path.join(ckpt_dir, "residential_stage3.pt")
    torch.save(model.state_dict(), save_path)
    print(f"[Residential] Saved fine-tuned model -> {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Fine-tune Industrial Model (re-derive tau on Pakistani normals)
# ─────────────────────────────────────────────────────────────────────────────

def finetune_industrial(X, y, cfg, ckpt_dir, device):
    """
    Continues training the IndustrialAutoencoder on Pakistani normal profiles,
    then re-derives tau. Saves as industrial_stage3.pt + industrial_tau_stage3.txt.
    """
    seq_len = X.shape[1]
    autoencoder, pretext, freq_proj = load_frozen_channels(seq_len, ckpt_dir, device)

    ccfg = cfg["channels"]
    icfg = cfg["industrial_model"]
    num_channels = 1 + sum([
        ccfg["use_autoencoder_residual"],
        ccfg["use_pretext_embedding"],
        ccfg["use_frequency_projection"]
    ])

    normal_mask = (y == 0)
    X_normal    = X[normal_mask]

    print(f"\n[Industrial] Pakistan normal profiles available: {X_normal.shape[0]}")
    if X_normal.shape[0] < 3:
        print("[Industrial] Too few normal profiles (<3). Skipping industrial fine-tune.")
        return

    model = IndustrialAutoencoder(
        num_channels=num_channels,
        seq_len=seq_len,
        latent_dim=icfg["autoencoder_latent_dim"]
    ).to(device)

    pretrained_path = os.path.join(ckpt_dir, "industrial_model.pt")
    model.load_state_dict(torch.load(pretrained_path, map_location=device))
    print(f"[Industrial] Loaded pretrained weights from {pretrained_path}")

    X_normal_dev = X_normal.to(device)
    with torch.no_grad():
        stacked_normal = build_channel_stack(
            X_normal_dev, autoencoder, pretext, freq_proj,
            ccfg["use_autoencoder_residual"],
            ccfg["use_pretext_embedding"],
            ccfg["use_frequency_projection"]
        )

    opt     = torch.optim.Adam(model.parameters(), lr=icfg["learning_rate"] * 0.1, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    ft_epochs = 20
    ft_bs     = min(4, X_normal.shape[0])

    print("[Industrial] Fine-tuning autoencoder on Pakistan normal profiles...")
    for epoch in range(ft_epochs):
        model.train()
        perm = torch.randperm(stacked_normal.shape[0])
        total_loss = 0.0
        for i in range(0, stacked_normal.shape[0], ft_bs):
            idx   = perm[i:i + ft_bs]
            batch = stacked_normal[idx]
            opt.zero_grad()
            recon = model(batch)
            loss  = loss_fn(recon, batch)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.size(0)
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{ft_epochs}  loss={total_loss/stacked_normal.shape[0]:.6f}")

    # Re-derive tau from Pakistan normals
    tau = derive_tau(model, stacked_normal, percentile=icfg["tau_percentile"])
    print(f"\n[Industrial] New tau on Pakistan normals ({icfg['tau_percentile']}th pct) = {tau:.6f}")

    # Evaluate on Pakistan theft profiles
    theft_mask = (y == 1)
    X_theft    = X[theft_mask]
    if X_theft.shape[0] > 0:
        X_theft_dev = X_theft.to(device)
        with torch.no_grad():
            stacked_theft = build_channel_stack(
                X_theft_dev, autoencoder, pretext, freq_proj,
                ccfg["use_autoencoder_residual"],
                ccfg["use_pretext_embedding"],
                ccfg["use_frequency_projection"]
            )
        errs    = model.reconstruction_error(stacked_theft).cpu().numpy()
        flagged = (errs > tau).sum()
        print(f"[Industrial] Theft profiles flagged above tau: {flagged}/{X_theft.shape[0]}")

    save_path = os.path.join(ckpt_dir, "industrial_stage3.pt")
    tau_path  = os.path.join(ckpt_dir, "industrial_tau_stage3.txt")
    torch.save(model.state_dict(), save_path)
    with open(tau_path, "w") as f:
        f.write(str(tau))
    print(f"[Industrial] Saved -> {save_path}")
    print(f"[Industrial] Tau   -> {tau_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fine-tune theft detection models on Pakistani data")
    parser.add_argument("--config",           required=True, help="Path to config/config.yaml")
    parser.add_argument("--skip-residential", action="store_true")
    parser.add_argument("--skip-industrial",  action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ckpt_dir = cfg["paths"]["checkpoints_dir"]
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Fine-tune] Device: {device}")

    # Determine target seq_len from SGCC training data
    processed_dir  = cfg["paths"]["processed_dir"]
    X_sgcc         = np.load(os.path.join(processed_dir, "X_train.npy"), mmap_mode="r")
    target_seq_len = X_sgcc.shape[1]
    print(f"[Fine-tune] SGCC seq_len = {target_seq_len}")
    del X_sgcc

    # Load Pakistani dataset
    pakistan_csv = "data/raw/pakistan/pakistan_target.csv"
    X_pk, y_pk   = load_pakistan_data(
        pakistan_csv,
        target_seq_len=target_seq_len,
        outlier_sigma_cap=cfg["preprocessing"]["outlier_sigma_cap"]
    )
    print(f"[Fine-tune] Pakistan tensor shape: X={X_pk.shape}, y={y_pk.shape}")

    if not args.skip_residential:
        finetune_residential(X_pk, y_pk, cfg, ckpt_dir, device)

    if not args.skip_industrial:
        finetune_industrial(X_pk, y_pk, cfg, ckpt_dir, device)

    print("\nFine-tuning complete.")
    print("   Residential -> models/checkpoints/residential_stage3.pt")
    print("   Industrial  -> models/checkpoints/industrial_stage3.pt")
    print("   Tau         -> models/checkpoints/industrial_tau_stage3.txt")
    print("\nREPORT LIMITATION: Fine-tuned on ~43 Pakistani profiles only.")
    print("   Train and eval sets are identical. Treat metrics as indicative only.")


if __name__ == "__main__":
    main()
