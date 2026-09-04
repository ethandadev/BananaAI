"""Direct Preference Optimization.

    python -m post.dpo --sft runs/sft/sft.pt --data data/dpo/prefs.jsonl --out runs/dpo

Input is JSONL of preference pairs:

    {"prompt": "...", "chosen": "...", "rejected": "..."}

DPO skips the reward model entirely. It raises the policy's log-probability of
the chosen response and lowers it for the rejected one, while a frozen copy of
the SFT model anchors the policy so it cannot drift into gibberish that
happens to satisfy the preference direction.

    loss = -log sigmoid(beta * [(pi_c - ref_c) - (pi_r - ref_r)])

Expect improved tone and formatting, not new capability. Keep the SFT
checkpoint: DPO can and does regress models, and you want the comparison.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, Optional

import torch
import torch.nn.functional as F

from core.config import ModelConfig
from core.model import Transformer
from core.tokenizer import PAD_ID, Message, encode_conversation, load as load_tokenizer

IGNORE = -100


def sequence_logprob(model, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Sum of log p(token) over supervised positions, per sequence.

    Summed rather than averaged: DPO compares whole responses, and averaging
    would make a short response and a long one with the same per-token quality
    look identical, which biases the model toward terseness.
    """
    logits, _, _ = model(x)
    logprobs = F.log_softmax(logits.float(), dim=-1)
    mask = y != IGNORE
    safe_y = y.masked_fill(~mask, 0)
    picked = logprobs.gather(-1, safe_y.unsqueeze(-1)).squeeze(-1)
    return (picked * mask).sum(dim=-1)


def dpo_loss(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    pi_logratio = policy_chosen - policy_rejected
    ref_logratio = ref_chosen - ref_rejected
    logits = pi_logratio - ref_logratio
    loss = -F.logsigmoid(beta * logits).mean()

    with torch.no_grad():
        metrics = {
            "accuracy": (logits > 0).float().mean().item(),
            "margin": logits.mean().item(),
            "chosen_reward": (beta * (policy_chosen - ref_chosen)).mean().item(),
            "rejected_reward": (beta * (policy_rejected - ref_rejected)).mean().item(),
        }
    return loss, metrics


class PreferenceDataset:
    def __init__(self, path: Path, tokenizer, max_len: int, seed: int = 0):
        self.pairs = []
        self.skipped = 0
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                enc = []
                for key in ("chosen", "rejected"):
                    conv = [Message("user", row["prompt"]), Message("assistant", row[key])]
                    ids, labels = encode_conversation(tokenizer, conv)
                    enc.append((ids, labels))
                if any(len(ids) > max_len for ids, _ in enc):
                    self.skipped += 1
                    continue
                self.pairs.append(tuple(enc))
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.pairs)

    def batches(self, batch_size: int, device: str, shuffle: bool = True) -> Iterator:
        order = list(range(len(self.pairs)))
        if shuffle:
            self.rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            chunk = [self.pairs[i] for i in order[start : start + batch_size]]
            chosen = _collate([c for c, _ in chunk], device)
            rejected = _collate([r for _, r in chunk], device)
            yield chosen, rejected


def _collate(items, device: str):
    width = max(len(ids) for ids, _ in items)
    x = torch.full((len(items), width), PAD_ID, dtype=torch.long)
    y = torch.full((len(items), width), IGNORE, dtype=torch.long)
    for row, (ids, labels) in enumerate(items):
        x[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        y[row, : len(labels)] = torch.tensor(labels, dtype=torch.long)
    return x[:, :-1].to(device), y[:, 1:].to(device)


def run_dpo(
    sft: Path,
    data: Path,
    out_dir: Path,
    tokenizer_path: Path,
    beta: float = 0.1,
    epochs: int = 1,
    batch_size: int = 4,
    lr: float = 5e-7,
    grad_clip: float = 1.0,
    device: Optional[str] = None,
    seed: int = 0,
    log_every: int = 10,
) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(tokenizer_path)
    state = torch.load(sft, map_location=device, weights_only=False)
    mc = ModelConfig(**state["model_config"])

    policy = Transformer(mc)
    policy.load_state_dict(state["model"])
    policy = policy.to(device)

    # The reference is a frozen copy of the starting policy.
    reference = copy.deepcopy(policy).to(device)
    reference.eval()
    for p in reference.parameters():
        p.requires_grad_(False)

    ds = PreferenceDataset(data, tokenizer, mc.context_len, seed)
    if not len(ds):
        raise SystemExit(f"no usable preference pairs in {data}")

    total_steps = math.ceil(len(ds) / batch_size) * epochs
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, betas=(0.9, 0.95))
    print(
        f"[dpo] {len(ds)} pairs ({ds.skipped} skipped), {total_steps} steps, "
        f"beta {beta}, {device}",
        file=sys.stderr,
    )

    policy.train()
    step = 0
    history = []
    # Kept outside `history`, which only samples every log_every steps.
    last = {"loss": None, "accuracy": None}
    t0 = time.perf_counter()

    for epoch in range(epochs):
        for (cx, cy), (rx, ry) in ds.batches(batch_size, device):
            with torch.no_grad():
                ref_c = sequence_logprob(reference, cx, cy)
                ref_r = sequence_logprob(reference, rx, ry)

            pol_c = sequence_logprob(policy, cx, cy)
            pol_r = sequence_logprob(policy, rx, ry)

            loss, metrics = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
            optimizer.step()

            last = {"loss": loss.item(), "accuracy": metrics["accuracy"]}

            if step % log_every == 0:
                print(
                    f"  epoch {epoch} step {step:>5}  loss {loss.item():.4f}  "
                    f"acc {metrics['accuracy']:.2f}  margin {metrics['margin']:+.3f}",
                    file=sys.stderr,
                )
                history.append({"step": step, "loss": loss.item(), **metrics})
            step += 1

    out_state = {
        "model": policy.state_dict(),
        "model_config": asdict(mc),
        "stage": "dpo",
        "base": str(sft),
        "beta": beta,
        "history": history,
    }
    path = out_dir / "dpo.pt"
    torch.save(out_state, path.with_suffix(".pt.tmp"))
    path.with_suffix(".pt.tmp").replace(path)

    summary = {
        "pairs": len(ds),
        "skipped": ds.skipped,
        "steps": step,
        "final_loss": last["loss"],
        "final_accuracy": last["accuracy"],
        "seconds": round(time.perf_counter() - t0, 2),
        "checkpoint": str(path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sft", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("runs/dpo"))
    p.add_argument("--tokenizer", type=Path, default=Path("tokenizer.json"))
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-7)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=10)
    args = p.parse_args(argv)

    print(json.dumps(run_dpo(
        args.sft, args.data, args.out, args.tokenizer,
        beta=args.beta, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, device=args.device, seed=args.seed, log_every=args.log_every,
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
