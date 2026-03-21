import argparse
import json
import os
import re
import time

import torch
from torch.utils.data import DataLoader

from torchtext.data.utils import get_tokenizer

try:
    from src.utils.dataset import DBpediaDataset
    from src.utils.collate import collate_batch
    from src.models.rnn import RNNClassifier
    from src.models.transformer import TransformerClassifier
    from src.utils.helpers import accuracy, save_checkpoint, load_checkpoint
    from src.utils.preprocessing import build_vocab, read_dbpedia_csv, split_dataset
except ImportError:
    # Support running the script as `python src/train.py` (sys.path includes src/)
    from utils.dataset import DBpediaDataset
    from utils.collate import collate_batch
    from models.rnn import RNNClassifier
    from models.transformer import TransformerClassifier
    from utils.helpers import accuracy, save_checkpoint, load_checkpoint
    from utils.preprocessing import build_vocab, read_dbpedia_csv, split_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train PyTorch text classifier on DBpedia")
    parser.add_argument(
        "--config_file",
        type=str,
        default="config.json",
        help="Path to JSON config file with parameters (default: config.json)",
    )
    return parser.parse_args()


def load_config(path):
    ext = os.path.splitext(path)[1].lower()
    if ext not in [".yaml", ".yml"]:
        raise ValueError("Config file must be .yaml or .yml")

    with open(path, "r", encoding="utf-8") as f:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to read YAML config files. "
                "Install it with `pip install pyyaml`."
            ) from exc
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Config file must contain an object/dictionary")
    return config


def build_args_from_config(config):
    required_global = [
        "model",
        "train_csv",
        "test_csv",
        "output_dir",
        "max_len",
        "batch_size",
        "epochs",
        "max_rows",
        "lr",
        "weight_decay",
        "max_vocab",
        "seed",
        "save_name",
        "save_all",
    ]

    # required model-specific parameters, to be defined in nested config
    required_model = ["embed_dim", "hidden_dim", "num_layers", "dropout"]

    for key in required_global:
        if key not in config:
            raise ValueError(f"Config must include '{key}' at top-level")

    model = config["model"]
    if model not in ["transformer", "rnn"]:
        raise ValueError("model must be 'transformer' or 'rnn'")

    model_config = config.get(model)
    if not isinstance(model_config, dict):
        raise ValueError(f"Config must include nested '{model}' config with model-specific values")

    for key in required_model:
        if key not in model_config:
            raise ValueError(f"Config must include '{model}.{key}'")

    if model == "transformer" and "num_heads" not in model_config:
        raise ValueError("Config must include 'transformer.num_heads'")

    final_config = {k: config[k] for k in required_global}
    final_config.update(model_config)

    # keep nested model info for readability but not used in the runner
    final_config["transformer"] = config.get("transformer", {})
    final_config["rnn"] = config.get("rnn", {})

    return argparse.Namespace(**final_config)


def main():

    # 1. Cấu hình (arguments) + chuẩn bị môi trường

    args = parse_args()
    if not args.config_file:
        raise ValueError("--config_file is required")
    config = load_config(args.config_file)
    args = build_args_from_config(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    # 2. Chuẩn bị dữ liệu

    train_samples = read_dbpedia_csv(
        args.train_csv, seed=args.seed, max_rows=args.max_rows
    )
    test_samples = read_dbpedia_csv(
        args.test_csv, seed=args.seed, max_rows=args.max_rows
    )

    tokenizer = get_tokenizer("basic_english")
    vocab = build_vocab(train_samples, tokenizer, max_size=args.max_vocab)

    # Split test into val + test (10% val)
    test_samples, val_samples, val_size, test_size = split_dataset(test_samples, val_ratio=0.1)
    print(f"Using {val_size} samples for val and {test_size} samples for test.")

    train_dataset = DBpediaDataset(
        train_samples,
        vocab,
        tokenizer,
        max_len=args.max_len,
        pad_index=vocab["<pad>"],
    )
    val_dataset = DBpediaDataset(
        val_samples,
        vocab,
        tokenizer,
        max_len=args.max_len,
        pad_index=vocab["<pad>"],
    )
    test_dataset = DBpediaDataset(
        test_samples,
        vocab,
        tokenizer,
        max_len=args.max_len,
        pad_index=vocab["<pad>"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(
            batch, pad_index=vocab["<pad>"], max_len=args.max_len
        ),
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(
            batch, pad_index=vocab["<pad>"], max_len=args.max_len
        ),
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(
            batch, pad_index=vocab["<pad>"], max_len=args.max_len
        ),
        num_workers=0,
    )

    # 3. Tạo model + optimizer + loss

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
            dropout=args.dropout,
        )
    else:
        model = RNNClassifier(
            vocab_size=len(vocab),
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_classes=14,
            pad_index=vocab["<pad>"],
            dropout=args.dropout,
        )

    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0

    # Organize saved models by model type (transformer/rnn)
    model_output_dir = os.path.join(args.output_dir, args.model)
    os.makedirs(model_output_dir, exist_ok=True)

    # Determine base checkpoint name (include model prefix + timestamp)
    if args.save_name:
        base_name = args.save_name
    else:
        current_time = time.strftime("%Y%m%d_%H%M%S")
        base_name = f"{args.model}_{current_time}"

    best_model_path = os.path.join(model_output_dir, f"{base_name}.pth")

    # 4. Training loop

    for epoch in range(1, args.epochs + 1):
        print(f"Starting epoch {epoch}/{args.epochs}...")
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        start_time = time.time()

        for batch_idx, (token_ids, attn_mask, labels) in enumerate(train_loader, start=1):
            token_ids = token_ids.to(device)
            attn_mask = attn_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(token_ids, attn_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * token_ids.size(0)
            epoch_acc += accuracy(logits, labels) * token_ids.size(0)

            if batch_idx % 50 == 0:
                print(f"  epoch {epoch}: batch {batch_idx}/{len(train_loader)}")

        epoch_loss /= len(train_dataset)
        epoch_acc /= len(train_dataset)

        model.eval()
        val_acc = 0.0
        with torch.no_grad():
            for token_ids, attn_mask, labels in val_loader:
                token_ids = token_ids.to(device)
                attn_mask = attn_mask.to(device)
                labels = labels.to(device)
                logits = model(token_ids, attn_mask)
                val_acc += accuracy(logits, labels) * token_ids.size(0)
        val_acc /= len(val_dataset)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch}/{args.epochs} | train_loss={epoch_loss:.4f} | train_acc={epoch_acc:.4f} | "
            f"val_acc={val_acc:.4f} | time={elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            extras = {"vocab_stoi": vocab.get_stoi()}

            # Save the best checkpoint
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best_model_path,
                extras=extras,
            )

            # Optionally save every improving checkpoint
            if args.save_all:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                extra_path = os.path.join(
                    model_output_dir,
                    f"{base_name}_epoch{epoch:02d}_val{val_acc:.4f}_{timestamp}.pth",
                )
                save_checkpoint(model, optimizer, epoch, extra_path, extras=extras)

    print(f"Best val accuracy: {best_val_acc:.4f}")

    # Final evaluation on the held-out test set
    if os.path.exists(best_model_path):
        load_checkpoint(model, None, best_model_path, device)
        model.eval()
        test_acc = 0.0
        with torch.no_grad():
            for token_ids, attn_mask, labels in test_loader:
                token_ids = token_ids.to(device)
                attn_mask = attn_mask.to(device)
                labels = labels.to(device)
                logits = model(token_ids, attn_mask)
                test_acc += accuracy(logits, labels) * token_ids.size(0)
        test_acc /= len(test_dataset)
        print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
