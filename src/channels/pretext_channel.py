"""
Channel 3 — masked-timestep self-supervised pretext embedding.

Substitutes for the lack of an ImageNet-style pretrained backbone in the
time-series domain: mask random days in the sequence, train a small model
to predict them from context, then reuse its bottleneck output as an
input channel for the main classifier. Train this on ALL data (labeled
and unlabeled, normal and theft) since it never uses the FLAG label.
"""
import numpy as np
import torch
import torch.nn as nn


class MaskedPretextEncoder(nn.Module):
    def __init__(self, seq_len: int, embed_dim: int = 32):
        super().__init__()
        self.seq_len = seq_len
        self.encoder = nn.Sequential(
            nn.Linear(seq_len, 256), nn.ReLU(),
            nn.Linear(256, embed_dim), nn.ReLU(),
        )
        self.decoder = nn.Linear(embed_dim, seq_len)

    def forward(self, x_masked):
        z = self.encoder(x_masked)
        pred = self.decoder(z)
        return pred, z

    def embedding_channel(self, x):
        """Returns the bottleneck embedding broadcast to seq_len for stacking as a channel."""
        with torch.no_grad():
            _, z = self.forward(x)
            # broadcast embedding across timesteps so it can stack alongside the raw sequence
            return z.mean(dim=1, keepdim=True).expand(-1, self.seq_len)


def random_mask(x: torch.Tensor, mask_ratio: float = 0.15) -> torch.Tensor:
    mask = (torch.rand_like(x) > mask_ratio).float()
    return x * mask


def train_pretext(model: MaskedPretextEncoder, X_all: torch.Tensor, epochs: int,
                   lr: float, batch_size: int, mask_ratio: float = 0.15, device: str = "cpu"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    n = X_all.shape[0]

    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            batch = X_all[idx].to(device)
            masked = random_mask(batch, mask_ratio)
            opt.zero_grad()
            pred, _ = model(masked)
            loss = loss_fn(pred, batch)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.size(0)
        print(f"[pretext_channel] epoch {epoch+1}/{epochs}  loss={total_loss/n:.6f}")

    return model
