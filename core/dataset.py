"""Batching over a flat uint16 token file.

The corpus is one long token stream on disk. A batch is B random windows of
length T+1 out of it, split into inputs (first T) and targets (last T) -- the
shift Transformer.forward deliberately does not do.

The memmap is reopened per batch. Holding one open across a long run leaks
page-cache references and the resident set grows without bound; reopening is
effectively free because the OS keeps the pages cached anyway.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch


class TokenDataset:
    def __init__(self, path: Path, context_len: int, seed: int = 0):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"no token bin at {self.path}")
        self.context_len = context_len
        self.n_tokens = self.path.stat().st_size // 2  # uint16
        if self.n_tokens < context_len + 1:
            raise ValueError(
                f"{self.path.name} holds {self.n_tokens} tokens, "
                f"need more than context_len ({context_len})"
            )
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.n_tokens

    def get_batch(
        self, batch_size: int, device: str = "cpu", pin: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        data = np.memmap(self.path, dtype=np.uint16, mode="r")
        high = len(data) - self.context_len - 1
        offsets = self.rng.integers(0, high, size=batch_size)

        # int64 for the embedding lookup; uint16 cannot be indexed with directly.
        xs = np.stack([data[o : o + self.context_len] for o in offsets]).astype(np.int64)
        ys = np.stack(
            [data[o + 1 : o + 1 + self.context_len] for o in offsets]
        ).astype(np.int64)

        x = torch.from_numpy(xs)
        y = torch.from_numpy(ys)
        if device.startswith("cuda"):
            if pin:
                x, y = x.pin_memory(), y.pin_memory()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    def deterministic_batches(self, batch_size: int, n_batches: int, device: str = "cpu"):
        """Fixed, non-overlapping windows -- for a validation loss you can compare
        across checkpoints without sampling noise."""
        data = np.memmap(self.path, dtype=np.uint16, mode="r")
        stride = self.context_len
        usable = (len(data) - 1) // stride
        take = min(n_batches * batch_size, usable)
        for start in range(0, take, batch_size):
            idx = [i * stride for i in range(start, min(start + batch_size, take))]
            if not idx:
                continue
            xs = np.stack([data[o : o + stride] for o in idx]).astype(np.int64)
            ys = np.stack([data[o + 1 : o + 1 + stride] for o in idx]).astype(np.int64)
            yield torch.from_numpy(xs).to(device), torch.from_numpy(ys).to(device)


def load_meta(tokenized_dir: Path) -> dict:
    meta_path = Path(tokenized_dir) / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"no meta.json in {tokenized_dir} -- run: python -m data.tokenize_corpus"
        )
    return json.loads(meta_path.read_text(encoding="utf-8"))
