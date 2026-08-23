"""
Trains the Industrial Agent's autoencoder on verified-normal industrial-typed
rows only, then derives tau (the reconstruction-error threshold) from a
held-out normal validation set. Note: SGCC has very few industrial-magnitude
accounts under the proxy label, so treat this model's reliability as
unproven until you check the actual sample count printed below.
"""
import argparse
import os
import numpy as np
import torch
import yaml

from src.channels.autoencoder_channel import ResidualAutoencoder
from src.channels.pretext_channel import MaskedPretextEncoder
from src.channels.frequency_channel import FrequencyProjection
from src.channels.stack_channels import build_channel_stack
from src.agents.industrial.model import IndustrialAutoencoder, derive_tau
from src.training.train_residential import load_frozen_channels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    processed = cfg["paths"]["processed_dir"]
    ckpt_dir = cfg["paths"]["checkpoints_dir"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_split(name):
        X = np.load(os.path.join(processed, f"X_{name}.npy"))
        y = np.load(os.path.join(processed, f"y_{name}.npy"))
        ctype = np.load(os.path.join(processed, f"type_{name}.npy"))
        mask = (ctype == "industrial") & (y == 0)  # normal industrial only
        return torch.tensor(X[mask], dtype=torch.float32)

    X_train_normal = load_split("train")
    X_val_normal = load_split("val")

    print(f"[CHECK] Industrial-typed normal training rows: {X_train_normal.shape[0]}")
    print("If this number is small (proxy label routes most accounts to residential), "
          "this model's reliability is unproven — say so in your report.")

    seq_len = X_train_normal.shape[1] if X_train_normal.shape[0] > 0 else None
    if seq_len is None:
        raise RuntimeError("No industrial-typed normal rows found. Check the proxy threshold in config.yaml.")

    autoencoder, pretext, freq_proj = load_frozen_channels(seq_len, ckpt_dir, device)
    ccfg = cfg["channels"]
    num_channels = 1 + sum([ccfg["use_autoencoder_residual"], ccfg["use_pretext_embedding"], ccfg["use_frequency_projection"]])

    icfg = cfg["industrial_model"]
    model = IndustrialAutoencoder(num_channels=num_channels, seq_len=seq_len,
                                   latent_dim=icfg["autoencoder_latent_dim"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=icfg["learning_rate"])
    loss_fn = torch.nn.MSELoss()

    stacked_train = build_channel_stack(X_train_normal.to(device), autoencoder, pretext, freq_proj,
                                         ccfg["use_autoencoder_residual"], ccfg["use_pretext_embedding"],
                                         ccfg["use_frequency_projection"])

    for epoch in range(icfg["epochs"]):
        model.train()
        perm = torch.randperm(stacked_train.shape[0])
        total_loss = 0.0
        for i in range(0, stacked_train.shape[0], icfg["batch_size"]):
            idx = perm[i:i + icfg["batch_size"]]
            batch = stacked_train[idx]
            opt.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.size(0)
        print(f"Epoch {epoch+1}/{icfg['epochs']}  loss={total_loss/stacked_train.shape[0]:.6f}")

    stacked_val = build_channel_stack(X_val_normal.to(device), autoencoder, pretext, freq_proj,
                                       ccfg["use_autoencoder_residual"], ccfg["use_pretext_embedding"],
                                       ccfg["use_frequency_projection"])
    tau = derive_tau(model, stacked_val, percentile=icfg["tau_percentile"])

    torch.save(model.state_dict(), os.path.join(ckpt_dir, "industrial_model.pt"))
    with open(os.path.join(ckpt_dir, "industrial_tau.txt"), "w") as f:
        f.write(str(tau))

    print(f"Derived tau ({icfg['tau_percentile']}th percentile of val reconstruction error) = {tau:.6f}")
    print(f"Saved model + tau to {ckpt_dir}")


if __name__ == "__main__":
    main()
