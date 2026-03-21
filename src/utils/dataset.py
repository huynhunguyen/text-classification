import torch
from torch.utils.data import Dataset


class DBpediaDataset(Dataset):
    def __init__(
        self,
        samples,
        vocab,
        tokenizer,
        max_len: int = 128,
        pad_index: int = 0,
    ):
        self.samples = samples
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_index = pad_index

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        label, text = self.samples[idx]
        if 1 <= label <= 14:
            label0 = label - 1
        elif 0 <= label <= 13:
            label0 = label
        else:
            raise ValueError(
                f"Invalid label {label} at index {idx}: expected 0..13 or 1..14"
            )

        tokens = self.tokenizer(text.lower())
        token_ids = self.vocab(tokens)
        token_ids = token_ids[: self.max_len]
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label0, dtype=torch.long)
