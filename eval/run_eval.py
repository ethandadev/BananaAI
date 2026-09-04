"""Evaluate a checkpoint: perplexity, plus generations you actually read.

    python -m eval.run_eval --ckpt runs/base/best.pt --data data/tokenized
    python -m eval.run_eval --ckpt runs/sft/sft.pt --prompts eval/prompts.jsonl --chat

Perplexity on held-out tokens is the number to track across a training run.
It is also not sufficient: two checkpoints with the same perplexity can differ
a lot in whether they follow an instruction. So this prints generations too,
and the comparison mode puts two checkpoints side by side on the same prompts
with the same seed.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Optional

import torch

from core.config import ModelConfig
from core.dataset import TokenDataset
from core.model import Transformer
from core.sample import SamplingConfig, generate
from core.tokenizer import Message, load as load_tokenizer, render

DEFAULT_PROMPTS = [
    "The main difference between a list and a tuple is",
    "def fibonacci(n):",
    "In machine learning, overfitting means",
    "The capital of France is",
    "To reverse a string in Python you can",
]


def load_checkpoint(path: Path, device: str) -> tuple[Transformer, dict]:
    state = torch.load(path, map_location=device, weights_only=False)
    mc = ModelConfig(**state["model_config"])
    model = Transformer(mc)
    model.load_state_dict(state["model"])
    return model.to(device).eval(), state


@torch.no_grad()
def perplexity(model, data_dir: Path, batch_size: int, batches: int, device: str) -> dict:
    ds = TokenDataset(Path(data_dir) / "val.bin", model.cfg.context_len)
    losses = []
    tokens = 0
    for x, y in ds.deterministic_batches(batch_size, batches, device):
        _, loss, _ = model(x, targets=y)
        losses.append(loss.item())
        tokens += y.numel()
    if not losses:
        return {"loss": None, "perplexity": None, "tokens": 0}
    mean = sum(losses) / len(losses)
    return {
        "loss": round(mean, 4),
        "perplexity": round(math.exp(mean), 2),
        "tokens_scored": tokens,
        "bits_per_token": round(mean / math.log(2), 4),
    }


def load_prompts(path: Optional[Path]) -> list[str]:
    if path is None:
        return DEFAULT_PROMPTS
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                row = json.loads(line)
                rows.append(row["prompt"] if isinstance(row, dict) else str(row))
    return rows


def run_generations(model, tokenizer, prompts, chat: bool, device: str,
                    max_new_tokens: int, seed: int) -> list[dict]:
    out = []
    for prompt in prompts:
        text = render([Message("user", prompt)], add_generation_prompt=True) if chat else prompt
        ids = torch.tensor(
            [tokenizer.encode(text, add_special_tokens=False).ids],
            dtype=torch.long, device=device,
        )
        stops = tuple(
            t for t in (tokenizer.token_to_id("<|eos|>"), tokenizer.token_to_id("<|end|>"))
            if t is not None
        )
        t0 = time.perf_counter()
        result = generate(model, ids, SamplingConfig(
            max_new_tokens=max_new_tokens, temperature=0.8, seed=seed, stop_tokens=stops,
        ))
        completion = tokenizer.decode(result[0, ids.size(1):].tolist())
        out.append({
            "prompt": prompt,
            "completion": completion,
            "new_tokens": result.size(1) - ids.size(1),
            "seconds": round(time.perf_counter() - t0, 3),
        })
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--compare", type=Path, default=None, help="a second checkpoint")
    p.add_argument("--data", type=Path, default=None, help="tokenized dir, for perplexity")
    p.add_argument("--tokenizer", type=Path, default=Path("tokenizer.json"))
    p.add_argument("--prompts", type=Path, default=None)
    p.add_argument("--chat", action="store_true", help="wrap prompts in the chat template")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--batches", type=int, default=20)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", action="store_true", help="machine-readable output only")
    args = p.parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(args.tokenizer)
    prompts = load_prompts(args.prompts)

    report: dict = {}
    targets = [("model", args.ckpt)] + ([("compare", args.compare)] if args.compare else [])

    for label, path in targets:
        model, state = load_checkpoint(path, device)
        entry = {
            "checkpoint": str(path),
            "stage": state.get("stage", "base"),
            "parameters": model.num_params(),
        }
        if args.data:
            entry.update(perplexity(model, args.data, args.batch_size, args.batches, device))
        entry["generations"] = run_generations(
            model, tokenizer, prompts, args.chat, device, args.max_new_tokens, args.seed
        )
        report[label] = entry

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    for label, entry in report.items():
        print(f"\n=== {label}: {Path(entry['checkpoint']).name}  "
              f"({entry['stage']}, {entry['parameters'] / 1e6:.1f}M) ===")
        if entry.get("perplexity") is not None:
            print(f"  val loss {entry['loss']}   perplexity {entry['perplexity']}   "
                  f"{entry['bits_per_token']} bits/token")
        for g in entry["generations"]:
            print(f"\n  > {g['prompt']}")
            print(f"    {g['completion'].strip()[:400]}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
