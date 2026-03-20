import csv
import random
from typing import List, Optional, Tuple

from torchtext.vocab import build_vocab_from_iterator


class FixedVocab:
    """Minimal vocab-like object that can be saved/loaded easily."""

    def __init__(self, stoi: dict, unk_token: str = "<unk>"):
        self.stoi = dict(stoi)
        self.unk_index = self.stoi.get(unk_token, 0)

    def __getitem__(self, token: str) -> int:
        return self.stoi.get(token, self.unk_index)

    def __call__(self, tokens):
        return [self[token] for token in tokens]

    def __len__(self):
        return len(self.stoi)

    def get_stoi(self):
        return self.stoi


def read_dbpedia_csv(
    path: str,
    max_rows: Optional[int] = None,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> List[Tuple[int, str]]:
    """Read DBpedia csv file and return list of (label, text).

    If `shuffle` is True, the rows are shuffled (random order) before returning.
    Providing `seed` makes the shuffling deterministic.
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if max_rows is not None and idx >= max_rows:
                break
            if len(row) < 3:
                continue
            label = int(row[0])
            text = row[2]
            rows.append((label, text))

    if shuffle:
        if seed is not None:
            random.seed(seed)
        random.shuffle(rows)

    return rows


def build_vocab(samples: List[Tuple[int, str]], tokenizer, max_size: int = 50000):
    """Build a vocabulary from samples using torchtext."""

    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text.lower())

    vocab = build_vocab_from_iterator(
        yield_tokens(samples),
        specials=["<pad>", "<unk>"],
        max_tokens=max_size,
    )
    vocab.set_default_index(vocab["<unk>"])

    # Wrap in FixedVocab so we can serialize the mapping for inference
    return FixedVocab(vocab.get_stoi(), unk_token="<unk>")


def split_dataset(samples, val_ratio: float = 0.1):
    """Split a list of samples into (test, val) using val_ratio.

    samples: list of (label, text)
    """
    total = len(samples)
    val_size = int(total * val_ratio)
    val_samples = samples[:val_size]
    test_samples = samples[val_size:]
    return test_samples, val_samples, val_size, total - val_size
