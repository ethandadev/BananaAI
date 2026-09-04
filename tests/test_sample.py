"""Sampling: logit transforms, determinism, stop tokens, cache correctness."""

from __future__ import annotations

import torch

from tests.harness import Suite

from core.config import ModelConfig
from core.model import Transformer
from core.sample import (
    SamplingConfig, apply_repetition_penalty, filter_top_k, filter_top_p, generate, stream,
)
from core.train import PRESETS

suite = Suite("sampling")
test = suite.test

TINY = ModelConfig(**PRESETS["tiny"], vocab_size=256)


def model():
    torch.manual_seed(0)
    return Transformer(TINY).eval()


@test
def top_k_keeps_exactly_k():
    logits = torch.tensor([[5.0, 1.0, 4.0, 2.0, 3.0]])
    out = filter_top_k(logits.clone(), 2)
    kept = torch.isfinite(out).sum().item()
    assert kept == 2, f"kept {kept} logits, expected 2"
    assert torch.isfinite(out[0, 0]) and torch.isfinite(out[0, 2]), "wrong two kept"
    return "5 logits -> top 2 survive"


@test
def top_k_is_a_noop_when_k_exceeds_the_vocabulary():
    logits = torch.randn(1, 10)
    assert torch.equal(filter_top_k(logits.clone(), 50), logits)
    assert torch.equal(filter_top_k(logits.clone(), 0), logits)
    return "k >= vocab and k = 0 both pass through"


@test
def top_p_keeps_the_smallest_sufficient_set():
    # Probabilities 0.5, 0.25, 0.125, 0.125 after softmax of these log values.
    logits = torch.log(torch.tensor([[0.5, 0.25, 0.125, 0.125]]))
    out = filter_top_p(logits.clone(), 0.7)
    kept = torch.isfinite(out).sum().item()
    assert kept == 2, f"p=0.7 kept {kept} tokens; 0.5 alone is short of 0.7"
    return "p=0.7 over [.5 .25 .125 .125] keeps 2"


@test
def top_p_always_keeps_at_least_one_token():
    """A dominant token must not be filtered out by its own probability."""
    logits = torch.log(torch.tensor([[0.99, 0.005, 0.005]]))
    out = filter_top_p(logits.clone(), 0.5)
    assert torch.isfinite(out).sum().item() >= 1, "filtered every token"
    assert torch.isfinite(out[0, 0]), "dropped the most likely token"
    return "p below the top token's mass still keeps it"


@test
def repetition_penalty_moves_seen_tokens_down_both_ways():
    """The sign bug: dividing a negative logit by 1.1 raises it."""
    logits = torch.tensor([[2.0, -2.0, 0.5]])
    seen = torch.tensor([[0, 1]])
    out = apply_repetition_penalty(logits.clone(), seen, 1.1)
    assert out[0, 0] < 2.0, "positive logit was not reduced"
    assert out[0, 1] < -2.0, "negative logit was raised instead of lowered"
    assert out[0, 2] == 0.5, "an unseen token was modified"
    return "positive divided, negative multiplied, unseen untouched"


@test
def repetition_penalty_of_one_is_identity():
    logits = torch.randn(1, 20)
    out = apply_repetition_penalty(logits.clone(), torch.tensor([[3, 7]]), 1.0)
    assert torch.equal(out, logits)
    return "penalty 1.0 changes nothing"


@test
def zero_temperature_is_deterministic_argmax():
    m = model()
    prompt = torch.randint(0, TINY.vocab_size, (1, 5))
    cfg = SamplingConfig(max_new_tokens=8, temperature=0.0, repetition_penalty=1.0)
    a = list(stream(m, prompt, cfg))
    b = list(stream(m, prompt, cfg))
    assert a == b, "greedy decoding was not reproducible"
    return f"identical 8-token output across runs"


@test
def seeded_sampling_is_reproducible():
    m = model()
    prompt = torch.randint(0, TINY.vocab_size, (1, 5))
    cfg = SamplingConfig(max_new_tokens=12, temperature=0.9, seed=42)
    assert list(stream(m, prompt, cfg)) == list(stream(m, prompt, cfg)), \
        "same seed produced different output"
    return "same seed, same 12 tokens"


@test
def different_seeds_diverge():
    m = model()
    prompt = torch.randint(0, TINY.vocab_size, (1, 5))
    a = list(stream(m, prompt, SamplingConfig(max_new_tokens=16, temperature=1.0, seed=1)))
    b = list(stream(m, prompt, SamplingConfig(max_new_tokens=16, temperature=1.0, seed=2)))
    assert a != b, "different seeds gave identical output"
    return "seeds 1 and 2 differ"


@test
def stop_token_ends_generation():
    m = model()
    prompt = torch.randint(0, TINY.vocab_size, (1, 4))
    # Stopping on every token the model could emit must yield nothing at all.
    everything = tuple(range(TINY.vocab_size))
    out = list(stream(m, prompt, SamplingConfig(max_new_tokens=20, stop_tokens=everything)))
    assert out == [], f"generated {len(out)} tokens despite stopping on all of them"
    return "stop token halts before emitting"


@test
def generation_respects_max_new_tokens():
    m = model()
    prompt = torch.randint(0, TINY.vocab_size, (1, 4))
    out = list(stream(m, prompt, SamplingConfig(max_new_tokens=7, temperature=0.8, seed=0)))
    assert len(out) == 7, f"asked for 7 tokens, got {len(out)}"
    return "exactly 7 tokens produced"


@test
def generate_returns_prompt_plus_completion():
    m = model()
    prompt = torch.randint(0, TINY.vocab_size, (1, 6))
    out = generate(m, prompt, SamplingConfig(max_new_tokens=5, temperature=0.0))
    assert out.size(1) == 11, f"expected 6 + 5 tokens, got {out.size(1)}"
    assert torch.equal(out[:, :6], prompt), "the prompt was altered"
    return "6-token prompt + 5 generated = 11"


@test
def generation_stops_at_the_context_limit():
    m = model()
    prompt = torch.randint(0, TINY.vocab_size, (1, TINY.context_len - 4))
    out = list(stream(m, prompt, SamplingConfig(max_new_tokens=100, temperature=0.0)))
    assert len(out) <= 4, f"generated {len(out)} tokens past the context window"
    return f"halted after {len(out)} tokens at context_len {TINY.context_len}"


@test
def batched_generation_matches_single_for_greedy():
    m = model()
    prompt = torch.randint(0, TINY.vocab_size, (1, 5))
    cfg = SamplingConfig(max_new_tokens=6, temperature=0.0, repetition_penalty=1.0)
    single = generate(m, prompt, cfg)
    batched = generate(m, prompt.repeat(3, 1), cfg)
    assert torch.equal(batched[0], single[0]), "batched greedy diverged from single"
    assert torch.equal(batched[0], batched[2]), "identical rows gave different output"
    return "3 identical rows agree with the single-sequence path"


if __name__ == "__main__":
    p, n = suite.run()
    raise SystemExit(0 if p == n else 1)
