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
        tokens = self.tokenizer(text.lower())
        token_ids = self.vocab(tokens)
        token_ids = token_ids[: self.max_len]
        # convert label from 1..14 to 0..13 for PyTorch CrossEntropy
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label - 1, dtype=torch.long)
