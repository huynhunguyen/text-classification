import argparse
import os
import sys
from typing import List, Optional

import torch

from torchtext.data.utils import get_tokenizer

from models.rnn import RNNClassifier
from models.transformer import TransformerClassifier
from utils.helpers import load_checkpoint
from utils.preprocessing import FixedVocab


def load_config(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext not in [".yaml", ".yml"]:
        raise ValueError("Config file must be .yaml or .yml")

    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to read YAML config files (pip install pyyaml)") from exc

    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Config must contain a YAML mapping")
    return cfg


def read_text_file(path: str) -> List[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def tokenize_and_pad(
    texts: List[str],
    tokenizer,
    vocab,
    max_len: int,
    pad_index: int = 0,
    device: Optional[str] = None,
):
    token_ids = []
    for t in texts:
        toks = tokenizer(t.lower())
        ids = vocab(toks)[:max_len]
        token_ids.append(ids)

    batch_size = len(token_ids)
    seqs = torch.full((batch_size, max_len), pad_index, dtype=torch.long)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    for i, ids in enumerate(token_ids):
        length = min(len(ids), max_len)
        seqs[i, :length] = torch.tensor(ids, dtype=torch.long)
        mask[i, :length] = 1

    if device is not None:
        seqs = seqs.to(device)
        mask = mask.to(device)

    return seqs, mask


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a trained model")
    parser.add_argument("--config_file", default="config.yml", help="Path to YAML config file")
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path from config")
    parser.add_argument("--input_text", type=str, help="Single text to classify")
    parser.add_argument("--input_file", type=str, help="File with one text per line")
    parser.add_argument("--max_len", type=int, default=None, help="Optional override max len")
    return parser.parse_args()


def main():
    args = parse_args()

    if (args.input_text is None) == (args.input_file is None):
        raise ValueError("Provide either --input_text or --input_file (but not both)")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = load_config(args.config_file)

    model_type = config.get("model")
    if model_type not in ["transformer", "rnn"]:
        raise ValueError("model must be 'transformer' or 'rnn' in config")

    checkpoint_path = args.checkpoint or config.get("checkpoint")
    if not checkpoint_path:
        raise ValueError("checkpoint path must be set in --checkpoint or config")

    max_len = args.max_len if args.max_len is not None else config.get("max_len", 128)

    checkpoint = load_checkpoint(None, None, checkpoint_path, device)
    vocab_stoi = checkpoint.get("vocab_stoi")
    if vocab_stoi is None:
        raise ValueError("Checkpoint must include vocab_stoi for inference")

    vocab = FixedVocab(vocab_stoi)
    tokenizer = get_tokenizer("basic_english")

    # build model
    if args.model == "transformer":
        model = TransformerClassifier(
            vocab_size=len(vocab),
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_classes=14,
            max_len=args.max_len,
            pad_index=vocab["<pad>"],
        )
    else:
        model = RNNClassifier(
            vocab_size=len(vocab),
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_classes=14,
            pad_index=vocab["<pad>"],
        )

    model.to(device)
    load_checkpoint(model, None, args.checkpoint, device)
    model.eval()

    if args.input_text is not None:
        texts = [args.input_text]
    else:
        texts = read_text_file(args.input_file)

    seqs, mask = tokenize_and_pad(texts, tokenizer, vocab, max_len, pad_index=vocab["<pad>"], device=device)

    with torch.no_grad():
        logits = model(seqs, mask)
        preds = torch.argmax(logits, dim=-1).cpu().tolist()

    for text, pred in zip(texts, preds):
        print(pred + 1, "\t", text)  # convert 0-based to 1-based label for DBpedia


if __name__ == "__main__":
    main()
