import argparse
import os
import sys
from typing import List, Optional

import torch

from torchtext.data.utils import get_tokenizer

from models.rnn import RNNClassifier
from models.transformer import TransformerClassifier
from utils.helpers import load_checkpoint
from utils.preprocessing import build_vocab, read_dbpedia_csv


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
    parser.add_argument("--model", choices=["transformer", "rnn"], required=True)
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pth)")

    parser.add_argument("--train_csv", required=False, default="data/train.csv", help="Path to train CSV to build vocab (optional if checkpoint has vocab_stoi)")
    parser.add_argument("--max_vocab", type=int, default=50000, help="Only used if vocab is not saved inside checkpoint")
    parser.add_argument("--max_len", type=int, default=128)

    parser.add_argument("--input_text", type=str, help="Single text to classify")
    parser.add_argument("--input_file", type=str, help="File with one text per line")

    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)

    return parser.parse_args()


def main():
    args = parse_args()

    if (args.input_text is None) == (args.input_file is None):
        raise ValueError("Provide either --input_text or --input_file (but not both)")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model + vocab from checkpoint (vocab is stored in checkpoint)
    checkpoint = load_checkpoint(None, None, args.checkpoint, device)
    vocab_stoi = checkpoint.get("vocab_stoi")

    if vocab_stoi is None:
        if args.train_csv is None:
            raise ValueError("--train_csv is required when checkpoint does not include vocab_stoi")
        # fallback: rebuild vocab from training data (requires matching max_vocab)
        train_samples = read_dbpedia_csv(args.train_csv)
        tokenizer = get_tokenizer("basic_english")
        vocab = build_vocab(train_samples, tokenizer, max_size=args.max_vocab)
    else:
        from utils.preprocessing import FixedVocab

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
            num_classes=15,
            max_len=args.max_len,
            pad_index=vocab["<pad>"],
        )
    else:
        model = RNNClassifier(
            vocab_size=len(vocab),
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_classes=15,
            pad_index=vocab["<pad>"],
        )

    model.to(device)
    load_checkpoint(model, None, args.checkpoint, device)
    model.eval()

    if args.input_text is not None:
        texts = [args.input_text]
    else:
        texts = read_text_file(args.input_file)

    seqs, mask = tokenize_and_pad(texts, tokenizer, vocab, args.max_len, pad_index=vocab["<pad>"], device=device)

    with torch.no_grad():
        logits = model(seqs, mask)
        preds = torch.argmax(logits, dim=-1).cpu().tolist()

    for text, pred in zip(texts, preds):
        print(pred + 1, "\t", text)  # convert 0-based to 1-based label for DBpedia


if __name__ == "__main__":
    main()
