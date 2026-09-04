"""Token generation with a KV cache.

Written by hand rather than delegated to a library. At this scale the cache is
the only optimisation that matters: without it, generating N tokens re-reads
the whole prefix N times and cost grows quadratically.

Sampling is a pipeline of independent transforms on the logits -- repetition
penalty, temperature, top-k, top-p -- applied in that order, because scaling by
temperature before penalising repeats would make the penalty depend on the
temperature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Optional

import torch
import torch.nn.functional as F

from .model import Transformer


@dataclass
class SamplingConfig:
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.95
    repetition_penalty: float = 1.1
    stop_tokens: tuple = ()
    seed: Optional[int] = None


def apply_repetition_penalty(
    logits: torch.Tensor, generated: torch.Tensor, penalty: float
) -> torch.Tensor:
    """Divide the logit of any already-seen token (CTRL-style).

    Negative logits are multiplied instead of divided -- dividing a negative
    number by 1.1 moves it *up*, rewarding the repetition it should suppress.
    """
    if penalty == 1.0 or generated.numel() == 0:
        return logits
    for b in range(logits.size(0)):
        seen = torch.unique(generated[b])
        vals = logits[b, seen]
        logits[b, seen] = torch.where(vals > 0, vals / penalty, vals * penalty)
    return logits


def filter_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k is None or k <= 0 or k >= logits.size(-1):
        return logits
    threshold = torch.topk(logits, k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def filter_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus filtering: keep the smallest set of tokens with cumulative
    probability >= p."""
    if p is None or p >= 1.0:
        return logits
    ordered, indices = torch.sort(logits, descending=True, dim=-1)
    cumulative = torch.softmax(ordered, dim=-1).cumsum(dim=-1)

    remove = cumulative - torch.softmax(ordered, dim=-1) >= p
    remove[..., 0] = False  # always keep the single most likely token

    ordered = ordered.masked_fill(remove, float("-inf"))
    return torch.empty_like(logits).scatter_(-1, indices, ordered)


@torch.no_grad()
def generate(
    model: Transformer,
    prompt_ids: torch.Tensor,
    cfg: SamplingConfig = SamplingConfig(),
    device: Optional[str] = None,
    on_token: Optional[Callable[[int], None]] = None,
) -> torch.Tensor:
    """Generate a continuation. Returns the full sequence including the prompt.

    A single sequence goes through stream(), which honours stop tokens; a batch
    cannot stop early per row, so it runs to max_new_tokens.
    """
    if prompt_ids.size(0) == 1:
        produced = list(stream(model, prompt_ids, cfg, device, on_token))
        if not produced:
            return prompt_ids
        tail = torch.tensor([produced], dtype=torch.long, device=prompt_ids.device)
        return torch.cat([prompt_ids, tail], dim=1)
    return _generate_batched(model, prompt_ids, cfg, device)


@torch.no_grad()
def stream(
    model: Transformer,
    prompt_ids: torch.Tensor,
    cfg: SamplingConfig = SamplingConfig(),
    device: Optional[str] = None,
    on_token: Optional[Callable[[int], None]] = None,
) -> Iterator[int]:
    """Yield generated token ids one at a time. Batch size must be 1."""
    if prompt_ids.size(0) != 1:
        raise ValueError("stream() handles a single sequence; use generate() for batches")

    model.eval()
    device = device or next(model.parameters()).device
    ids = prompt_ids.to(device)

    gen = torch.Generator(device="cpu")
    if cfg.seed is not None:
        gen.manual_seed(cfg.seed)

    max_ctx = model.cfg.context_len
    if ids.size(1) >= max_ctx:
        ids = ids[:, -(max_ctx - 1):]

    logits, _, past = model(ids, use_cache=True)
    produced = ids

    for _ in range(cfg.max_new_tokens):
        next_id = _pick(logits[:, -1, :].float(), produced, cfg, gen)
        token = int(next_id.item())

        if token in cfg.stop_tokens:
            return
        yield token
        if on_token is not None:
            on_token(token)

        produced = torch.cat([produced, next_id], dim=1)
        if produced.size(1) >= max_ctx:
            return  # context is full; the caller decides whether to re-prompt

        logits, _, past = model(next_id, past_kvs=past, use_cache=True)


def _pick(logits, produced, cfg: SamplingConfig, gen) -> torch.Tensor:
    logits = apply_repetition_penalty(logits.clone(), produced, cfg.repetition_penalty)

    if cfg.temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / cfg.temperature
    logits = filter_top_k(logits, cfg.top_k)
    logits = filter_top_p(logits, cfg.top_p)

    probs = torch.softmax(logits, dim=-1)
    # Sampling on CPU keeps a seeded run reproducible regardless of device.
    choice = torch.multinomial(probs.cpu(), num_samples=1, generator=gen)
    return choice.to(logits.device)


@torch.no_grad()
def _generate_batched(model, prompt_ids, cfg, device):
    """Batched greedy/sampled generation without early stopping per sequence."""
    model.eval()
    device = device or next(model.parameters()).device
    ids = prompt_ids.to(device)
    gen = torch.Generator(device="cpu")
    if cfg.seed is not None:
        gen.manual_seed(cfg.seed)

    logits, _, past = model(ids, use_cache=True)
    for _ in range(cfg.max_new_tokens):
        next_ids = _pick(logits[:, -1, :].float(), ids, cfg, gen)
        ids = torch.cat([ids, next_ids], dim=1)
        if ids.size(1) >= model.cfg.context_len:
            break
        logits, _, past = model(next_ids, past_kvs=past, use_cache=True)
    return ids


def generate_text(model, tokenizer, prompt: str, cfg: SamplingConfig = SamplingConfig()) -> str:
    """Convenience wrapper: string in, string out."""
    ids = torch.tensor(
        [tokenizer.encode(prompt, add_special_tokens=False).ids],
        dtype=torch.long,
        device=next(model.parameters()).device,
    )
    out = list(stream(model, ids, cfg))
    return tokenizer.decode(out)
