import os

import torch


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)


def save_checkpoint(model, optimizer, epoch: int, path: str, extras: dict = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    if extras:
        checkpoint.update(extras)
    torch.save(checkpoint, path)


def load_checkpoint(model, optimizer, path: str, device: str):
    checkpoint = torch.load(path, map_location=device)

    # Allow loading just the checkpoint dict (e.g., to access saved vocab) without
    # requiring a model/optimizer instance.
    if model is not None:
        model.load_state_dict(checkpoint["model_state"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    return checkpoint
