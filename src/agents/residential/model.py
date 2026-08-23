"""
Residential Agent — supervised classifier for the residential theft signature
(sudden baseline shift toward zero from meter bypass / phase tapping).

Input: (batch, num_channels, seq_len) stacked tensor from stack_channels.py
Output: theft_probability in [0, 1] via sigmoid
"""
import torch
import torch.nn as nn


class ResidentialModel(nn.Module):
    def __init__(self, num_channels: int, seq_len: int,
                 cnn_filters: int = 64, cnn_kernel_size: int = 3,
                 lstm_hidden_units: int = 128, dropout: float = 0.3):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=num_channels, out_channels=cnn_filters,
                               kernel_size=cnn_kernel_size, padding=cnn_kernel_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.bilstm = nn.LSTM(input_size=cnn_filters, hidden_size=lstm_hidden_units,
                               batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(lstm_hidden_units * 2, 1)

    def forward(self, x):
        # x: (batch, num_channels, seq_len)
        x = self.relu(self.conv(x))          # (batch, cnn_filters, seq_len)
        x = self.dropout(x)
        x = x.transpose(1, 2)                # (batch, seq_len, cnn_filters) for LSTM
        out, (h_n, _) = self.bilstm(x)
        last_forward = h_n[-2]
        last_backward = h_n[-1]
        combined = torch.cat([last_forward, last_backward], dim=1)
        logit = self.classifier(combined)
        return torch.sigmoid(logit).squeeze(-1)
