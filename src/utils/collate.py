import torch


def collate_batch(batch, pad_index: int = 0, max_len: int = 128):
    """Collate fn for DataLoader: pad sequences to max_len and build attention mask."""
    token_ids_list, labels = zip(*batch)

    batch_size = len(token_ids_list)
    seqs = torch.full((batch_size, max_len), pad_index, dtype=torch.long)
    attn_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    for i, seq in enumerate(token_ids_list):
        length = min(len(seq), max_len)
        seqs[i, :length] = seq[:length]
        attn_mask[i, :length] = 1

    labels = torch.stack(labels)
    return seqs, attn_mask, labels
