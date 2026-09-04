"""Supervised fine-tuning: turn a text completer into an assistant.

    python -m post.sft --base runs/base/best.pt --data data/sft/train.jsonl --out runs/sft

Input is JSONL, one conversation per line:

    {"messages": [{"role": "user", "content": "..."},
                  {"role": "assistant", "content": "..."}]}

Loss is computed on assistant content only. Training on the prompt as well
teaches the model to generate plausible *questions*, which is not the job, and
it dilutes the gradient that actually matters.

Sequences are right-padded. No attention mask is needed for that: causal
attention means a pad can only influence positions after it, and those are
pads too, whose labels are -100 and contribute nothing to the loss.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, Optional

import torch

from core.config import ModelConfig
from core.model import Transformer
from core.tokenizer import PAD_ID, Message, encode_conversation, load as load_tokenizer

IGNORE = -100


def read_conversations(path: Path) -> list[list[Message]]:
    convs = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            msgs = row.get("messages")
            if not msgs:
                raise ValueError(f"{path}:{lineno} has no 'messages'")
            convs.append([Message(m["role"], m["content"]) for m in msgs])
    return convs


class ConversationDataset:
    def __init__(self, conversations, tokenizer, max_len: int, seed: int = 0):
        self.examples = []
        self.skipped = 0
        for conv in conversations:
            ids, labels = encode_conversation(tokenizer, conv)
            if len(ids) > max_len:
                self.skipped += 1
                continue
            if all(l == IGNORE for l in labels):
                self.skipped += 1      # nothing supervised; would give a zero gradient
                continue
            self.examples.append((ids, labels))
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.examples)

    def batches(self, batch_size: int, device: str, shuffle: bool = True) -> Iterator:
        order = list(range(len(self.examples)))
        if shuffle:
            self.rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            chunk = [self.examples[i] for i in order[start : start + batch_size]]
            yield self._collate(chunk, device)

    @staticmethod
    def _collate(chunk, device: str):
        width = max(len(ids) for ids, _ in chunk)
        x = torch.full((len(chunk), width), PAD_ID, dtype=torch.long)
        y = torch.full((len(chunk), width), IGNORE, dtype=torch.long)
        for row, (ids, labels) in enumerate(chunk):
            x[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            y[row, : len(labels)] = torch.tensor(labels, dtype=torch.long)
        # Shift here, matching the Transformer.forward contract.
        return x[:, :-1].to(device), y[:, 1:].to(device)


def load_base(path: Path, device: str) -> tuple[Transformer, ModelConfig]:
    state = torch.load(path, map_location=device, weights_only=False)
    mc = ModelConfig(**state["model_config"])
    model = Transformer(mc)
    model.load_state_dict(state["model"])
    return model.to(device), mc


def run_sft(
    base: Path,
    data: Path,
    out_dir: Path,
    tokenizer_path: Path,
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 1e-5,
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.0,
    grad_clip: float = 1.0,
    device: Optional[str] = None,
    seed: int = 0,
    val_data: Optional[Path] = None,
    log_every: int = 10,
) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(tokenizer_path)
    model, mc = load_base(base, device)

    ds = ConversationDataset(read_conversations(data), tokenizer, mc.context_len, seed)
    if not len(ds):
        raise SystemExit(f"no usable examples in {data}")
    val_ds = (
        ConversationDataset(read_conversations(val_data), tokenizer, mc.context_len, seed)
        if val_data else None
    )

    steps_per_epoch = math.ceil(len(ds) / batch_size)
    total_steps = steps_per_epoch * epochs
    warmup = max(1, int(total_steps * warmup_ratio))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay
    )
    print(
        f"[sft] {len(ds)} examples ({ds.skipped} skipped), {total_steps} steps, "
        f"{device}, lr {lr:g}",
        file=sys.stderr,
    )

    model.train()
    step = 0
    history = []
    # Tracked separately from `history`, which only samples every log_every
    # steps -- reporting the last *logged* loss as the final loss is wrong
    # whenever logging is sparse.
    first_loss = last_loss = None
    t0 = time.perf_counter()

    for epoch in range(epochs):
        for x, y in ds.batches(batch_size, device):
            scale = min(1.0, (step + 1) / warmup)
            progress = max(0.0, (step - warmup) / max(1, total_steps - warmup))
            cur_lr = lr * scale * (0.5 * (1 + math.cos(math.pi * progress)) if step >= warmup else 1.0)
            for g in optimizer.param_groups:
                g["lr"] = cur_lr

            optimizer.zero_grad(set_to_none=True)
            _, loss, _ = model(x, targets=y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            last_loss = loss.item()
            if first_loss is None:
                first_loss = last_loss

            if step % log_every == 0:
                print(f"  epoch {epoch} step {step:>5}  loss {loss.item():.4f}  "
                      f"lr {cur_lr:.2e}", file=sys.stderr)
                history.append({"step": step, "loss": loss.item()})
            step += 1

        if val_ds:
            vl = _eval(model, val_ds, batch_size, device)
            print(f"  epoch {epoch} val {vl:.4f}", file=sys.stderr)

    state = {
        "model": model.state_dict(),
        "model_config": asdict(mc),
        "stage": "sft",
        "base": str(base),
        "history": history,
    }
    path = out_dir / "sft.pt"
    torch.save(state, path.with_suffix(".pt.tmp"))
    path.with_suffix(".pt.tmp").replace(path)

    summary = {
        "examples": len(ds),
        "skipped": ds.skipped,
        "steps": step,
        "first_loss": first_loss,
        "final_loss": last_loss,
        "seconds": round(time.perf_counter() - t0, 2),
        "checkpoint": str(path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


@torch.no_grad()
def _eval(model, ds, batch_size, device) -> float:
    model.eval()
    losses = []
    for x, y in ds.batches(batch_size, device, shuffle=False):
        _, loss, _ = model(x, targets=y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else float("nan")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--val-data", type=Path, default=None)
    p.add_argument("--out", type=Path, default=Path("runs/sft"))
    p.add_argument("--tokenizer", type=Path, default=Path("tokenizer.json"))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=10)
    args = p.parse_args(argv)

    summary = run_sft(
        args.base, args.data, args.out, args.tokenizer,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        device=args.device, seed=args.seed, val_data=args.val_data,
        log_every=args.log_every,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
