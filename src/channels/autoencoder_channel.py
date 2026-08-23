"""
Channel 2 — reconstruction-residual channel.

Trained ONLY on confirmed-normal profiles (y == 0). At inference, every
customer's sequence is passed through this frozen network; the per-day
absolute reconstruction error becomes an extra input channel for the
main classifier. This is what makes it "Channel Boosting" rather than
a hand-crafted feature: the channel comes out of a trained network,
not a formula.
"""
import torch
import torch.nn as nn


class ResidualAutoencoder(nn.Module):
    def __init__(self, seq_len: int, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(seq_len, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 256), nn.ReLU(),
            nn.Linear(256, seq_len), nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon

    def residual_channel(self, x):
        """Returns per-timestep |x - recon|, same shape as x. Use with torch.no_grad()."""
        recon = self.forward(x)
        return torch.abs(x - recon)


def train_on_normals(model: ResidualAutoencoder, X_train_normal: torch.Tensor,
                      epochs: int, lr: float, batch_size: int, device: str = "cpu"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    n = X_train_normal.shape[0]

    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            batch = X_train_normal[idx].to(device)
            opt.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.size(0)
        print(f"[autoencoder_channel] epoch {epoch+1}/{epochs}  loss={total_loss/n:.6f}")

    return model
