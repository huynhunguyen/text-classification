import torch
import torch.nn as nn


class RNNClassifier(nn.Module):
    """BiLSTM classifier with mean pooling."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 15,
        pad_index: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_index)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, token_ids: torch.Tensor, attention_mask: torch.Tensor):
        x = self.embed(token_ids)
        lengths = attention_mask.sum(dim=1).cpu()

        # pack padded sequences for efficiency (optional)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        # Ensure output is padded to the full sequence length (max_len) so it matches the attention mask.
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=attention_mask.size(1),
        )

        mask = attention_mask.unsqueeze(-1).float()
        out = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        logits = self.classifier(out)
        return logits
