import argparse
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
    parser.add_argument("--model", choices=["transformer", "rnn"], default="transformer")
    parser.add_argument(
        "--train_csv",
        default="data/train.csv",
        help="Path to DBpedia train CSV (default: data/train.csv)",
    )
    parser.add_argument(
        "--test_csv",
        default="data/test.csv",
        help="Path to DBpedia test CSV (default: data/test.csv)",
    )
    parser.add_argument("--output_dir", default="models", help="Directory to save models")

    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="If set, read at most this many rows from each CSV (useful for quick sanity runs)",
    )

    # model hyperparameters (defaults assigned based on chosen model)
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=None,
        help="Embedding dimension (default depends on model)",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=None,
        help="Number of attention heads (transformer only)",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=None,
        help="Hidden dimension of the model (transformer/RNN)",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=None,
        help="Number of layers in the model (transformer/RNN)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Dropout probability (transformer/RNN)",
    )

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--max_vocab", type=int, default=50000)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--save_name",
        type=str,
        default=None,
        help="Name for the saved checkpoint (default: next sequential number)",
    )
    parser.add_argument(
        "--save_all",
        action="store_true",
        help="If set, keep every improving checkpoint (instead of only the best)",
    )

    return parser.parse_args()


def main():

    # 1. Cấu hình (arguments) + chuẩn bị môi trường

    args = parse_args()
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

    # set model-specific defaults if user didn’t pass them
    transformer_defaults = {
        "embed_dim": 128,
        "num_heads": 4,
        "hidden_dim": 256,
        "num_layers": 2,
        "dropout": 0.1,
    }
    rnn_defaults = {
        "embed_dim": 128,
        "hidden_dim": 256,
        "num_layers": 2,
        "dropout": 0.1,
    }
    defaults = transformer_defaults if args.model == "transformer" else rnn_defaults

    embed_dim = args.embed_dim if args.embed_dim is not None else defaults["embed_dim"]
    hidden_dim = args.hidden_dim if args.hidden_dim is not None else defaults["hidden_dim"]
    num_layers = args.num_layers if args.num_layers is not None else defaults["num_layers"]
    dropout = args.dropout if args.dropout is not None else defaults["dropout"]

    # transformer-only defaults
    num_heads = args.num_heads if args.num_heads is not None else transformer_defaults["num_heads"]

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
            embed_dim=embed_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=14,
            max_len=args.max_len,
            pad_index=vocab["<pad>"],
            dropout=dropout,
        )
    else:
        model = RNNClassifier(
            vocab_size=len(vocab),
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=14,
            pad_index=vocab["<pad>"],
            dropout=dropout,
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

    # Determine base checkpoint name
    if args.save_name:
        base_name = args.save_name
    else:
        # Find existing numbered checkpoints and pick next index
        existing = [f for f in os.listdir(model_output_dir) if f.endswith(".pth")]
        nums = []
        for f in existing:
            m = re.match(r"^(\d+)", f)
            if m:
                nums.append(int(m.group(1)))
        next_id = max(nums) + 1 if nums else 1
        base_name = f"{next_id:03d}"

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
