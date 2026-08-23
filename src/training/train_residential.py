"""
Trains the Residential Agent classifier on the channel-stacked training
split (residential-typed rows only, per the proxy consumer_type label).
Auxiliary channel networks are loaded FROZEN from checkpoints produced by
train_channels.py — they are not fine-tuned here.
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

from src.channels.autoencoder_channel import ResidualAutoencoder
from src.channels.pretext_channel import MaskedPretextEncoder
from src.channels.frequency_channel import FrequencyProjection
from src.channels.stack_channels import build_channel_stack
from src.agents.residential.model import ResidentialModel


def load_frozen_channels(seq_len, ckpt_dir, device):
    autoencoder = ResidualAutoencoder(seq_len=seq_len)
    autoencoder.load_state_dict(torch.load(os.path.join(ckpt_dir, "autoencoder_channel.pt"), map_location=device))
    autoencoder.to(device).eval()
    for p in autoencoder.parameters():
        p.requires_grad = False

    pretext = MaskedPretextEncoder(seq_len=seq_len)
    pretext.load_state_dict(torch.load(os.path.join(ckpt_dir, "pretext_channel.pt"), map_location=device))
    pretext.to(device).eval()
    for p in pretext.parameters():
        p.requires_grad = False

    freq_proj = FrequencyProjection(seq_len=seq_len)
    freq_proj.load_state_dict(torch.load(os.path.join(ckpt_dir, "frequency_channel.pt"), map_location=device))
    freq_proj.to(device).eval()
    for p in freq_proj.parameters():
        p.requires_grad = False

    return autoencoder, pretext, freq_proj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    processed = cfg["paths"]["processed_dir"]
    ckpt_dir = cfg["paths"]["checkpoints_dir"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_split(name, ctype_filter="residential"):
        X = np.load(os.path.join(processed, f"X_{name}.npy"))
        y = np.load(os.path.join(processed, f"y_{name}.npy"))
        ctype = np.load(os.path.join(processed, f"type_{name}.npy"))
        mask = ctype == ctype_filter
        return (torch.tensor(X[mask], dtype=torch.float32),
                torch.tensor(y[mask], dtype=torch.float32))

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    seq_len = X_train.shape[1]

    autoencoder, pretext, freq_proj = load_frozen_channels(seq_len, ckpt_dir, device)

    mcfg = cfg["residential_model"]
    ccfg = cfg["channels"]
    num_channels = 1 + sum([ccfg["use_autoencoder_residual"], ccfg["use_pretext_embedding"], ccfg["use_frequency_projection"]])

    model = ResidentialModel(
        num_channels=num_channels, seq_len=seq_len,
        cnn_filters=mcfg["cnn_filters"], cnn_kernel_size=mcfg["cnn_kernel_size"],
        lstm_hidden_units=mcfg["lstm_hidden_units"], dropout=mcfg["dropout"],
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=mcfg["learning_rate"])
    loss_fn = nn.BCELoss()

    best_val_auc = 0.0
    patience_counter = 0

    for epoch in range(mcfg["epochs"]):
        model.train()
        perm = torch.randperm(X_train.shape[0])
        total_loss = 0.0

        for i in range(0, X_train.shape[0], mcfg["batch_size"]):
            idx = perm[i:i + mcfg["batch_size"]]
            xb = X_train[idx].to(device)
            yb = y_train[idx].to(device)

            stacked = build_channel_stack(xb, autoencoder, pretext, freq_proj,
                                           ccfg["use_autoencoder_residual"],
                                           ccfg["use_pretext_embedding"],
                                           ccfg["use_frequency_projection"])
            opt.zero_grad()
            preds = model(stacked)
            loss = loss_fn(preds, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)

        # validation — this split was NOT SMOTE-resampled, this is the honest number
        model.eval()
        with torch.no_grad():
            stacked_val = build_channel_stack(X_val.to(device), autoencoder, pretext, freq_proj,
                                               ccfg["use_autoencoder_residual"],
                                               ccfg["use_pretext_embedding"],
                                               ccfg["use_frequency_projection"])
            val_preds = model(stacked_val).cpu().numpy()
            val_auc = roc_auc_score(y_val.numpy(), val_preds)
            val_f1 = f1_score(y_val.numpy(), (val_preds > 0.5).astype(int))

        print(f"Epoch {epoch+1}/{mcfg['epochs']}  train_loss={total_loss/X_train.shape[0]:.4f}  "
              f"val_auc={val_auc:.4f}  val_f1={val_f1:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "residential_model_best.pt"))
        else:
            patience_counter += 1
            if patience_counter >= mcfg["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch+1}. Best val_auc={best_val_auc:.4f}")
                break

    print(f"Training done. Best validation ROC-AUC = {best_val_auc:.4f}")
    print("This is the number to report — not a figure copied from a published paper.")


if __name__ == "__main__":
    main()
