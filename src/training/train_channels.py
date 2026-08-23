"""
Pretrains the 3 auxiliary channel networks on the TRAINING split only.
Saves frozen weights to models/checkpoints/. These must be pretrained
before train_residential.py or train_industrial.py can run, since both
depend on stack_channels.py loading these checkpoints.
"""
import argparse
import os
import numpy as np
import torch
import yaml

from src.channels.autoencoder_channel import ResidualAutoencoder, train_on_normals
from src.channels.pretext_channel import MaskedPretextEncoder, train_pretext
from src.channels.frequency_channel import FrequencyProjection, train_frequency_projection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    processed = cfg["paths"]["processed_dir"]
    ckpt_dir = cfg["paths"]["checkpoints_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    X_train = torch.tensor(np.load(os.path.join(processed, "X_train.npy")), dtype=torch.float32)
    y_train = np.load(os.path.join(processed, "y_train.npy"))
    seq_len = X_train.shape[1]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Channel 2 — autoencoder trained ONLY on confirmed-normal rows (y==0)
    X_normal = X_train[torch.tensor(y_train == 0)]
    autoencoder = ResidualAutoencoder(seq_len=seq_len)
    autoencoder = train_on_normals(
        autoencoder, X_normal,
        epochs=cfg["residential_model"]["epochs"],
        lr=cfg["residential_model"]["learning_rate"],
        batch_size=cfg["residential_model"]["batch_size"],
        device=device,
    )
    torch.save(autoencoder.state_dict(), os.path.join(ckpt_dir, "autoencoder_channel.pt"))

    # Channel 3 — masked pretext, trained on ALL rows (labels irrelevant here)
    pretext = MaskedPretextEncoder(seq_len=seq_len)
    pretext = train_pretext(
        pretext, X_train,
        epochs=cfg["residential_model"]["epochs"],
        lr=cfg["residential_model"]["learning_rate"],
        batch_size=cfg["residential_model"]["batch_size"],
        device=device,
    )
    torch.save(pretext.state_dict(), os.path.join(ckpt_dir, "pretext_channel.pt"))

    # Channel 4 — learned frequency projection, trained on ALL rows
    freq_proj = FrequencyProjection(seq_len=seq_len)
    freq_proj = train_frequency_projection(
        freq_proj, X_train,
        epochs=cfg["residential_model"]["epochs"],
        lr=cfg["residential_model"]["learning_rate"],
        batch_size=cfg["residential_model"]["batch_size"],
        device=device,
    )
    torch.save(freq_proj.state_dict(), os.path.join(ckpt_dir, "frequency_channel.pt"))

    print(f"Saved all 3 auxiliary channel checkpoints to {ckpt_dir}")
    print("These are now FROZEN. train_residential.py and train_industrial.py load them read-only.")


if __name__ == "__main__":
    main()
