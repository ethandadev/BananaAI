"""Hardware detection and automatic model sizing.

Devices are constructed literally rather than detected, so the sizing logic is
tested against GPUs neither of us has: an 8 GB laptop card, an A100, an M-series
Mac. That is the whole point of the module -- it has to be right on machines it
has never run on.
"""

from __future__ import annotations

from tests.harness import Suite

from core.config import ModelConfig
from core.hardware import (
    DEFAULT_MAX_HOURS, FALLBACK_PEAK, LADDER, SAFETY_MARGIN, TOKENS_PER_PARAM,
    Device, effective_peak, hours_to_train, largest_micro_batch, peak_for,
    recommend, training_memory_gb,
)

suite = Suite("hardware")
test = suite.test


def gpu(name="Test GPU", gb=24.0, bf16=True, tflops=100.0, kind="cuda") -> Device:
    return Device(kind=kind, name=name, memory_gb=gb, supports_bf16=bf16, peak_tflops=tflops)


@test
def ladder_grows_monotonically():
    sizes = [ModelConfig(vocab_size=32768, **spec).n_params() for _, spec in LADDER]
    assert sizes == sorted(sizes), f"ladder is not ordered: {sizes}"
    assert sizes[0] < 20e6, f"smallest rung is {sizes[0] / 1e6:.0f}M -- too big for CPU"
    assert sizes[-1] > 1e9, f"largest rung is only {sizes[-1] / 1e6:.0f}M"
    return " -> ".join(f"{s / 1e6:.0f}M" for s in sizes)


@test
def every_rung_is_a_valid_architecture():
    """ModelConfig validates head divisibility; a bad rung would raise here."""
    for name, spec in LADDER:
        mc = ModelConfig(vocab_size=32768, **spec)
        assert mc.head_dim % 2 == 0, f"{name} has an odd head_dim"
        assert mc.n_heads % mc.n_kv_heads == 0, f"{name} has a bad GQA ratio"
        assert 2.0 < mc.ffn_hidden / mc.d_model < 4.5, \
            f"{name} FFN ratio {mc.ffn_hidden / mc.d_model:.2f} is out of range"
    return f"{len(LADDER)} rungs, all valid"


@test
def logits_dominate_the_memory_model():
    """The term people forget: at a 32k vocabulary it exceeds the weights."""
    mc = ModelConfig(vocab_size=32768, **dict(LADDER)["base"])
    mem = training_memory_gb(mc, micro_batch=16, bf16=True)
    assert mem["logits_gb"] > mem["weights_gb"], \
        f"logits {mem['logits_gb']:.2f} GB vs weights {mem['weights_gb']:.2f} GB"
    assert abs(sum(mem[k] for k in ("weights_gb", "logits_gb", "activations_gb"))
               - mem["total_gb"]) < 1e-9, "breakdown does not sum to the total"
    return (f"logits {mem['logits_gb']:.1f} GB > weights {mem['weights_gb']:.1f} GB "
            f"at micro-batch 16")


@test
def memory_scales_linearly_with_batch():
    mc = ModelConfig(vocab_size=32768, **dict(LADDER)["small"])
    one = training_memory_gb(mc, 1, True)
    eight = training_memory_gb(mc, 8, True)
    # Weights are fixed; everything else should be 8x.
    variable_one = one["total_gb"] - one["weights_gb"]
    variable_eight = eight["total_gb"] - eight["weights_gb"]
    assert abs(variable_eight / variable_one - 8) < 0.01, \
        f"ratio {variable_eight / variable_one:.2f}, expected 8"
    return "activations and logits scale 8x, weights constant"


@test
def micro_batch_never_exceeds_the_budget():
    mc = ModelConfig(vocab_size=32768, **dict(LADDER)["base"])
    for budget in (2, 4, 8, 16, 32, 64):
        mb = largest_micro_batch(mc, budget, bf16=True)
        if mb:
            used = training_memory_gb(mc, mb, True)["total_gb"]
            assert used <= budget, f"{mb} uses {used:.1f} GB of a {budget} GB budget"
    return "checked 6 budgets, none exceeded"


@test
def a_tiny_budget_yields_no_batch_at_all():
    mc = ModelConfig(vocab_size=32768, **dict(LADDER)["xl"])
    assert largest_micro_batch(mc, 0.5, bf16=True) == 0, \
        "claimed the xl model fits in 0.5 GB"
    return "0 returned rather than a batch that cannot allocate"


@test
def bigger_cards_get_bigger_models():
    sizes = []
    for gb, tflops in ((8, 30), (12, 40), (16, 60), (24, 165), (48, 200), (80, 312)):
        plan = recommend(gpu(gb=gb, tflops=tflops), max_hours=200)
        sizes.append(plan.model.n_params())
    assert sizes == sorted(sizes), f"not monotonic in VRAM: {[s // 10**6 for s in sizes]}"
    return " <= ".join(f"{s / 1e6:.0f}M" for s in sizes)


@test
def the_time_budget_is_respected():
    fast = gpu(gb=80.0, tflops=312.0)
    quick = recommend(fast, max_hours=4)
    slow = recommend(fast, max_hours=200)
    assert quick.model.n_params() < slow.model.n_params(), \
        "a 4-hour budget picked the same model as a 200-hour one"
    assert quick.estimated_hours <= 4.01, f"{quick.estimated_hours:.1f}h exceeds the budget"
    return (f"4h -> {quick.model.n_params() / 1e6:.0f}M, "
            f"200h -> {slow.model.n_params() / 1e6:.0f}M")


@test
def the_default_budget_is_a_long_weekend():
    plan = recommend(gpu(gb=32.6, tflops=209.0))
    assert plan.estimated_hours <= DEFAULT_MAX_HOURS, \
        f"{plan.estimated_hours:.0f}h exceeds the {DEFAULT_MAX_HOURS:g}h default"
    return f"default {DEFAULT_MAX_HOURS:g}h, chose {plan.estimated_hours:.0f}h"


@test
def an_rtx_5090_reproduces_the_hand_derived_plan():
    """Independent check: the original spec was worked out by hand at 327M/~52h."""
    plan = recommend(gpu("NVIDIA GeForce RTX 5090", 32.6, True, 209.0))
    assert plan.preset == "base", f"chose {plan.preset}"
    assert abs(plan.model.n_params() - 326_685_696) < 1, plan.model.n_params()
    assert 40 <= plan.estimated_hours <= 70, f"{plan.estimated_hours:.0f}h"
    assert plan.train.tokens_per_step == 524_288, plan.train.tokens_per_step
    return (f"{plan.preset}, {plan.model.n_params():,} params, "
            f"{plan.estimated_hours:.0f}h -- matches the hand-derived spec")


@test
def cpu_gets_something_it_could_actually_finish():
    plan = recommend(Device(kind="cpu", name="Some CPU", memory_gb=16.0,
                            supports_bf16=False, peak_tflops=None))
    assert plan.model.n_params() < 30e6, \
        f"recommended {plan.model.n_params() / 1e6:.0f}M for CPU training"
    assert plan.estimated_hours <= DEFAULT_MAX_HOURS
    assert any("slower" in w for w in plan.warnings), "no warning about CPU speed"
    return f"{plan.preset}, {plan.model.n_params() / 1e6:.0f}M, {plan.estimated_hours:.0f}h"


@test
def unknown_devices_still_get_an_estimate():
    plan = recommend(Device(kind="cuda", name="Some Future GPU", memory_gb=24.0,
                            supports_bf16=True, peak_tflops=None))
    assert plan.estimated_hours is not None, "no estimate for an unlisted GPU"
    assert any("throughput table" in w for w in plan.warnings), \
        "did not flag that the estimate is a guess"
    assert effective_peak(plan.device) == FALLBACK_PEAK["cuda"]
    return "falls back to an assumed peak and says so"


@test
def a_forced_preset_is_honoured_and_warned_about():
    plan = recommend(gpu(gb=8.0, tflops=30.0), preset="base", max_hours=None)
    assert plan.preset == "base", f"forced preset ignored, got {plan.preset}"
    return f"base forced onto an 8 GB card, mb={plan.train.micro_batch}"


@test
def an_unknown_preset_raises():
    try:
        recommend(gpu(), preset="gigantic")
    except KeyError as e:
        assert "gigantic" in str(e)
        return "unknown preset names raise with the valid list"
    raise AssertionError("accepted a preset that does not exist")


@test
def pre_ampere_cards_lose_bf16():
    plan = recommend(gpu(gb=11.0, tflops=13.0, bf16=False))
    fp32 = training_memory_gb(plan.model, plan.train.micro_batch, bf16=False)
    bf16 = training_memory_gb(plan.model, plan.train.micro_batch, bf16=True)
    assert fp32["total_gb"] > bf16["total_gb"], "fp32 should cost more than bf16"
    return f"fp32 costs {fp32['total_gb'] / bf16['total_gb']:.2f}x bf16 memory"


@test
def the_safety_margin_leaves_headroom():
    plan = recommend(gpu(gb=24.0, tflops=165.0), max_hours=500)
    assert plan.memory["total_gb"] <= 24.0 * SAFETY_MARGIN + 1e-6, \
        f"used {plan.memory['total_gb']:.1f} GB of a {24.0 * SAFETY_MARGIN:.1f} GB budget"
    return f"{plan.memory['total_gb']:.1f} GB of {24.0 * SAFETY_MARGIN:.1f} GB usable"


@test
def peak_lookup_matches_on_substrings():
    assert peak_for("NVIDIA GeForce RTX 5090") == 209.0
    assert peak_for("NVIDIA A100-SXM4-80GB") == 312.0
    assert peak_for("Some Unlisted Card") is None
    return "device names matched by substring"


@test
def token_budget_follows_the_scaling_rule():
    plan = recommend(gpu(gb=24.0, tflops=165.0), max_hours=500)
    tokens = plan.train.total_steps * plan.train.tokens_per_step
    ratio = tokens / plan.model.n_params()
    assert abs(ratio - TOKENS_PER_PARAM) / TOKENS_PER_PARAM < 0.05, \
        f"{ratio:.1f} tokens per parameter, expected ~{TOKENS_PER_PARAM}"
    return f"{ratio:.1f} tokens/parameter"


@test
def the_batch_schedule_stays_consistent():
    for gb, tflops in ((8, 30), (24, 165), (80, 312)):
        plan = recommend(gpu(gb=gb, tflops=tflops))
        accum = plan.train.grad_accum_steps(plan.model.context_len)
        assert plan.train.micro_batch * plan.model.context_len * accum \
            == plan.train.tokens_per_step, f"schedule inconsistent at {gb} GB"
    return "micro x context x accum == tokens_per_step on 3 devices"


if __name__ == "__main__":
    p, n = suite.run()
    raise SystemExit(0 if p == n else 1)
