"""
Industrial Agent — unsupervised anomaly scorer for the industrial theft
signature (selective peak-shaving, partial load stripping). Trained only
on verified-normal industrial profiles; flags customers whose
reconstruction error exceeds tau.

NOTE: this is a SECOND autoencoder, separate from the Channel 2 auxiliary
autoencoder in src/channels/. That one produces an input channel for the
residential model. This one is the industrial scoring model itself.
Don't conflate the two when debugging.
"""
import torch
import torch.nn as nn


class IndustrialAutoencoder(nn.Module):
    def __init__(self, num_channels: int, seq_len: int, latent_dim: int = 32):
        super().__init__()
        flat_dim = num_channels * seq_len
        self.flatten = nn.Flatten()
        self.encoder = nn.Sequential(
            nn.Linear(flat_dim, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 512), nn.ReLU(),
            nn.Linear(512, flat_dim), nn.Sigmoid(),
        )
        self.num_channels = num_channels
        self.seq_len = seq_len

    def forward(self, x):
        flat = self.flatten(x)
        z = self.encoder(flat)
        recon_flat = self.decoder(z)
        recon = recon_flat.view(-1, self.num_channels, self.seq_len)
        return recon

    def reconstruction_error(self, x):
        with torch.no_grad():
            recon = self.forward(x)
            err = torch.mean((x - recon) ** 2, dim=(1, 2))
        return err


def derive_tau(model: IndustrialAutoencoder, X_val_normal: torch.Tensor, percentile: int = 95) -> float:
    """Tau = given percentile of reconstruction error on a VERIFIED-NORMAL validation set.
    Do not derive tau from a set that contains any theft-flagged accounts."""
    errors = model.reconstruction_error(X_val_normal)
    tau = torch.quantile(errors, percentile / 100.0).item()
    return tau
