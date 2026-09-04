"""Correctness tests for the model, runnable on CPU.

No GPU and no pytest required:

    python tests/test_model.py

These check the things that fail silently -- a KV cache that disagrees with a
full forward pass produces a model that trains perfectly and then generates
nonsense, which is an expensive bug to find late.
"""

import torch

from tests.harness import Suite

from core.config import ModelConfig, TrainConfig
from core.model import Transformer, build_rope_cache, apply_rope

suite = Suite("model")
test = suite.test

TINY = ModelConfig(
    n_layers=2, d_model=64, n_heads=4, n_kv_heads=2,
    ffn_hidden=128, vocab_size=256, context_len=64,
)

@test
def rope_preserves_norm():
    """Rotation is orthogonal, so it must not change vector length."""
    cos, sin = build_rope_cache(16, 8, 10000.0)
    x = torch.randn(2, 4, 8, 16)
    y = apply_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5), "RoPE changed vector norms"
    assert not torch.allclose(x, y), "RoPE was a no-op"
    return "norms preserved, values rotated"


@test
def rope_is_relative():
    """Dot products must depend on relative position, not absolute."""
    cos, sin = build_rope_cache(16, 32, 10000.0)
    q = torch.randn(1, 1, 1, 16)
    k = torch.randn(1, 1, 1, 16)

    def dot(i, j):
        qi = apply_rope(q, cos[i:i + 1], sin[i:i + 1])
        kj = apply_rope(k, cos[j:j + 1], sin[j:j + 1])
        return (qi * kj).sum().item()

    assert abs(dot(3, 5) - dot(10, 12)) < 1e-4, "RoPE is not translation-invariant"
    return "offset 2 gives the same score at any absolute position"


@test
def initial_loss_matches_uniform():
    """An untrained model should be exactly as uncertain as a uniform guess."""
    torch.manual_seed(0)
    model = Transformer(TINY)
    tokens = torch.randint(0, TINY.vocab_size, (2, 33))
    _, loss, _ = model(tokens[:, :-1], targets=tokens[:, 1:])
    expected = torch.log(torch.tensor(float(TINY.vocab_size))).item()
    assert abs(loss.item() - expected) < 0.15, f"loss {loss.item():.3f} != ln(V) {expected:.3f}"
    return f"loss {loss.item():.3f}, ln(V) {expected:.3f}"


@test
def gradients_reach_every_parameter():
    """A parameter with no gradient is a parameter wired up wrong."""
    torch.manual_seed(0)
    model = Transformer(TINY)
    tokens = torch.randint(0, TINY.vocab_size, (2, 17))
    _, loss, _ = model(tokens[:, :-1], targets=tokens[:, 1:])
    loss.backward()
    dead = [n for n, p in model.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert not dead, f"no gradient reached: {dead}"
    return f"all {sum(1 for _ in model.parameters())} tensors received gradient"


@test
def causal_mask_blocks_the_future():
    """Changing a later token must not alter an earlier position's logits."""
    torch.manual_seed(0)
    model = Transformer(TINY).eval()
    a = torch.randint(0, TINY.vocab_size, (1, 16))
    b = a.clone()
    b[0, -1] = (b[0, -1] + 1) % TINY.vocab_size
    with torch.no_grad():
        la, _, _ = model(a)
        lb, _, _ = model(b)
    assert torch.allclose(la[:, :-1], lb[:, :-1], atol=1e-5), "information leaked backwards"
    return "earlier positions unaffected by a later edit"


@test
def kv_cache_matches_full_forward():
    """Incremental decoding must reproduce the full-sequence result exactly."""
    torch.manual_seed(0)
    model = Transformer(TINY).eval()
    seq = torch.randint(0, TINY.vocab_size, (1, 20))

    with torch.no_grad():
        full, _, _ = model(seq)

        # Prefill on the first 12 tokens, then decode the rest one at a time.
        cached, _, past = model(seq[:, :12], use_cache=True)
        logits = [cached[:, -1]]
        for t in range(12, 19):
            step, _, past = model(seq[:, t:t + 1], past_kvs=past, use_cache=True)
            logits.append(step[:, -1])

    got = torch.stack(logits, dim=1)
    want = full[:, 11:19]
    err = (got - want).abs().max().item()
    assert err < 1e-4, f"cache diverges from full forward by {err:.2e}"
    return f"8 decoded positions match, max error {err:.2e}"


@test
def gqa_actually_shrinks_the_cache():
    """4 KV heads instead of 16 is the whole point of grouped-query attention."""
    torch.manual_seed(0)
    model = Transformer(TINY).eval()
    idx = torch.randint(0, TINY.vocab_size, (1, 8))
    with torch.no_grad():
        _, _, past = model(idx, use_cache=True)
    k, _ = past[0]
    assert k.shape[1] == TINY.n_kv_heads, f"cache holds {k.shape[1]} heads, expected {TINY.n_kv_heads}"
    ratio = TINY.n_heads // TINY.n_kv_heads
    return f"cache stores {k.shape[1]} of {TINY.n_heads} heads ({ratio}x smaller)"


@test
def config_predicts_real_parameter_count():
    """verify_env.py asserts these match exactly -- check it on the real config."""
    cfg = ModelConfig()
    model = Transformer(cfg)
    actual = model.num_params()
    predicted = cfg.n_params()
    assert actual == predicted, f"built {actual:,}, formula says {predicted:,}"
    assert model.lm_head.weight.data_ptr() == model.embed.weight.data_ptr(), "embeddings not tied"
    return f"{actual:,} params, head tied to embedding"


@test
def batch_schedule_is_consistent():
    tc, mc = TrainConfig(), ModelConfig()
    accum = tc.grad_accum_steps(mc.context_len)
    total = tc.total_steps * tc.tokens_per_step
    assert tc.micro_batch * mc.context_len * accum == tc.tokens_per_step
    assert 6.5e9 < total < 7.5e9, f"{total / 1e9:.2f}B tokens is off target"
    return f"micro {tc.micro_batch} x accum {accum} x ctx {mc.context_len}, {total / 1e9:.2f}B tokens"


if __name__ == "__main__":
    p, n = suite.run()
    raise SystemExit(0 if p == n else 1)
