"""
Channel 4 — learned frequency-domain channel.

Deliberately NOT raw FFT coefficients (that would be a hand-crafted
feature, not Channel Boosting). A small trainable projection sits on
top of the FFT magnitude spectrum so the channel is still the output
of a learned function.
"""
import numpy as np
import torch
import torch.nn as nn


class FrequencyProjection(nn.Module):
    def __init__(self, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.proj = nn.Sequential(
            nn.Linear(seq_len, seq_len), nn.ReLU(),
            nn.Linear(seq_len, seq_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len) real-valued sequence
        spectrum = torch.abs(torch.fft.rfft(x, n=self.seq_len))
        # pad/truncate spectrum back to seq_len so it can stack as a channel
        if spectrum.shape[-1] < self.seq_len:
            pad = self.seq_len - spectrum.shape[-1]
            spectrum = nn.functional.pad(spectrum, (0, pad))
        else:
            spectrum = spectrum[..., :self.seq_len]
        return self.proj(spectrum)


def train_frequency_projection(model: FrequencyProjection, X_all: torch.Tensor,
                                epochs: int, lr: float, batch_size: int, device: str = "cpu"):
    """
    Self-supervised: train the projection to reconstruct the raw sequence
    from its own frequency representation, so it learns a useful mapping
    rather than staying at random init.
    """
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
            opt.zero_grad()
            out = model(batch)
            loss = loss_fn(out, batch)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.size(0)
        print(f"[frequency_channel] epoch {epoch+1}/{epochs}  loss={total_loss/n:.6f}")

    return model
