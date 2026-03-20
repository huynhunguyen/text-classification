import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch_size, seq_len, d_model)"""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 15,
        max_len: int = 128,
        dropout: float = 0.1,
        pad_index: int = 0,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_index)
        self.positional_encoding = PositionalEncoding(embed_dim, dropout=dropout, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="relu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.cls_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

        self.max_len = max_len

    def forward(self, token_ids: torch.Tensor, attention_mask: torch.Tensor):
        """token_ids: (batch, seq_len); attention_mask: (batch, seq_len)"""
        x = self.embed(token_ids)  # (batch, seq_len, embed_dim)
        x = self.positional_encoding(x)

        # Transformer expects (seq_len, batch, embed_dim)
        x = x.transpose(0, 1)

        # transformer uses src_key_padding_mask with True for positions to MASK
        src_key_padding_mask = ~attention_mask

        out = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        out = out.transpose(0, 1)  # (batch, seq_len, embed_dim)

        # use mean pooling over unmasked tokens
        mask = attention_mask.unsqueeze(-1).float()
        out = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        logits = self.cls_head(out)
        return logits
