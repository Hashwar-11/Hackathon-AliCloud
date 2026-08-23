"""
Combines the raw sequence + 3 auxiliary channels into one (batch, 4, seq_len)
tensor for the residential/industrial models to consume. All auxiliary
networks are used in eval() / no_grad mode here — they must already be
frozen from src/training/train_channels.py before this is called.
"""
import torch


def build_channel_stack(x_raw: torch.Tensor, autoencoder, pretext_encoder, freq_proj,
                         use_autoencoder=True, use_pretext=True, use_frequency=True) -> torch.Tensor:
    channels = [x_raw]

    if use_autoencoder:
        autoencoder.eval()
        with torch.no_grad():
            channels.append(autoencoder.residual_channel(x_raw))

    if use_pretext:
        pretext_encoder.eval()
        with torch.no_grad():
            channels.append(pretext_encoder.embedding_channel(x_raw))

    if use_frequency:
        freq_proj.eval()
        with torch.no_grad():
            channels.append(freq_proj(x_raw))

    return torch.stack(channels, dim=1)  # (batch, num_channels, seq_len)
