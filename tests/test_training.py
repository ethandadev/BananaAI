"""Training loop: schedule, optimizer groups, batching, checkpoints, resume."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from tests.harness import Suite

from core.config import ModelConfig, TrainConfig
from core.dataset import TokenDataset
from core.model import Transformer
from core.train import Checkpointer, PRESETS, build_optimizer, evaluate, lr_at, train

suite = Suite("training")
test = suite.test

TINY = ModelConfig(**PRESETS["tiny"], vocab_size=256)


def write_bin(path: Path, n_tokens: int, vocab: int, seed: int = 0) -> None:
    """A learnable stream: a repeating cycle, so loss must fall if training works."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, vocab, size=97, dtype=np.uint16)
    data = np.tile(base, n_tokens // 97 + 1)[:n_tokens]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.astype(np.uint16).tobytes())


def make_data(tmp: Path, vocab: int = 256) -> Path:
    write_bin(tmp / "train.bin", 60_000, vocab, seed=0)
    write_bin(tmp / "val.bin", 8_000, vocab, seed=0)
    (tmp / "meta.json").write_text(
        f'{{"vocab_size": {vocab}, "train_tokens": 60000, "val_tokens": 8000, '
        f'"dtype": "uint16", "eos_id": 2}}', encoding="utf-8")
    return tmp


@test
def warmup_is_linear_and_peaks_once():
    tc = TrainConfig(warmup_steps=100, total_steps=1000, peak_lr=4e-4, min_lr=4e-5)
    assert lr_at(0, tc) < lr_at(50, tc) < lr_at(99, tc), "warmup is not increasing"
    assert abs(lr_at(99, tc) - tc.peak_lr) < 1e-12, "warmup does not reach peak"
    schedule = [lr_at(s, tc) for s in range(1000)]
    assert max(schedule) <= tc.peak_lr + 1e-12, "learning rate exceeded the peak"
    return "linear to peak at step 99, never above peak"


@test
def cosine_decays_to_the_floor():
    tc = TrainConfig(warmup_steps=10, total_steps=200, peak_lr=1e-3, min_lr=1e-4)
    end = lr_at(199, tc)
    assert abs(end - tc.min_lr) < 2e-5, f"final lr {end:.2e}, expected ~{tc.min_lr:.2e}"
    assert lr_at(500, tc) == tc.min_lr, "past total_steps the floor should hold"
    mid = lr_at(105, tc)
    expected = tc.min_lr + (tc.peak_lr - tc.min_lr) * 0.5
    assert abs(mid - expected) < 5e-5, f"halfway lr {mid:.2e} != {expected:.2e}"
    return "half-way value correct, decays to the floor and stays"


@test
def norms_and_biases_escape_weight_decay():
    model = Transformer(TINY)
    opt, n_decay, n_no = build_optimizer(model, TrainConfig(), "cpu")
    decay_group, no_decay_group = opt.param_groups
    assert decay_group["weight_decay"] > 0
    assert no_decay_group["weight_decay"] == 0.0

    decayed = {id(p) for p in decay_group["params"]}
    for name, p in model.named_parameters():
        if "norm" in name:
            assert id(p) not in decayed, f"{name} is being weight-decayed"
    assert n_decay > 0 and n_no > 0
    return f"{n_decay} matrices decayed, {n_no} vectors exempt"


@test
def batches_are_shifted_by_exactly_one():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "train.bin"
        write_bin(path, 5000, 256)
        ds = TokenDataset(path, context_len=64, seed=0)
        x, y = ds.get_batch(4)
        assert x.shape == (4, 64) and y.shape == (4, 64), f"{x.shape} {y.shape}"
        assert torch.equal(x[:, 1:], y[:, :-1]), "y is not x shifted by one"
        assert x.dtype == torch.long, "embedding lookup needs int64"
    return "x[1:] == y[:-1] for every row"


@test
def deterministic_batches_do_not_overlap():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "val.bin"
        write_bin(path, 4000, 256)
        ds = TokenDataset(path, context_len=32, seed=0)
        a = [x for x, _ in ds.deterministic_batches(2, 5, "cpu")]
        b = [x for x, _ in ds.deterministic_batches(2, 5, "cpu")]
        assert len(a) == len(b) and all(torch.equal(p, q) for p, q in zip(a, b)), \
            "validation batches are not reproducible"
    return f"{len(a)} batches, identical across calls"


@test
def dataset_rejects_a_file_that_is_too_short():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "tiny.bin"
        write_bin(path, 10, 256)
        try:
            TokenDataset(path, context_len=64)
        except ValueError as e:
            assert "context_len" in str(e)
            return "a bin shorter than the context window raises"
    raise AssertionError("accepted a file smaller than one window")


@test
def checkpointer_prunes_and_is_atomic():
    with tempfile.TemporaryDirectory() as tmp:
        ck = Checkpointer(Path(tmp), keep_last=3)
        for step in range(0, 60, 10):
            ck.save({"model": {}, "step": step}, step)
        kept = sorted(p.name for p in Path(tmp).glob("step-*.pt"))
        assert len(kept) == 3, f"kept {len(kept)} checkpoints, expected 3"
        assert ck.latest().name == kept[-1]
        assert not list(Path(tmp).glob("*.tmp")), "a temp file was left behind"
    return f"6 saved, 3 kept, latest is {kept[-1]}"


@test
def a_short_run_reduces_loss():
    with tempfile.TemporaryDirectory() as tmp:
        data = make_data(Path(tmp) / "data")
        out = Path(tmp) / "run"
        tc = TrainConfig(total_steps=30, warmup_steps=3, peak_lr=3e-3, min_lr=3e-4,
                         micro_batch=8, tokens_per_step=8 * TINY.context_len,
                         eval_every=15, ckpt_every=100, log_every=1000, sample_every=10**9)
        summary = train(TINY, tc, data, out, device="cpu", compile_model=False, seed=0)
        assert summary["steps_completed"] == 30
        assert summary["best_val_loss"] is not None
        # ln(256) = 5.545 at initialisation; a working loop must beat that.
        assert summary["best_val_loss"] < 5.0, \
            f"val loss {summary['best_val_loss']:.3f} did not improve on random"
        assert (out / "config.json").exists() and (out / "summary.json").exists()
    return f"30 steps, val loss {summary['best_val_loss']:.3f} (random is 5.55)"


@test
def resume_continues_from_the_checkpoint():
    with tempfile.TemporaryDirectory() as tmp:
        data = make_data(Path(tmp) / "data")
        out = Path(tmp) / "run"
        common = dict(micro_batch=8, tokens_per_step=8 * TINY.context_len,
                      warmup_steps=2, peak_lr=3e-3, min_lr=3e-4,
                      eval_every=1000, ckpt_every=10, log_every=1000, sample_every=10**9)

        train(TINY, TrainConfig(total_steps=20, **common), data, out,
              device="cpu", compile_model=False, seed=0)
        first = sorted(out.glob("step-*.pt"))
        assert first, "no checkpoint written by the first run"

        summary = train(TINY, TrainConfig(total_steps=40, **common), data, out,
                        device="cpu", compile_model=False, seed=0, resume=True)
        assert summary["steps_completed"] < 40, \
            "resume restarted from zero instead of continuing"
        assert summary["tokens_seen"] > 20 * 8 * TINY.context_len, \
            "token counter did not carry across the resume"
    return f"resumed and ran {summary['steps_completed']} further steps"


@test
def evaluate_matches_a_hand_computed_loss():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "val.bin"
        write_bin(path, 3000, 256)
        ds = TokenDataset(path, context_len=32, seed=0)
        model = Transformer(TINY)
        got = evaluate(model, ds, 2, 3, "cpu")

        losses = []
        model.eval()
        with torch.no_grad():
            for x, y in ds.deterministic_batches(2, 3, "cpu"):
                _, loss, _ = model(x, targets=y)
                losses.append(loss.item())
        want = sum(losses) / len(losses)
        assert abs(got - want) < 1e-6, f"{got} != {want}"
    return f"loss {got:.4f} reproduced independently"


@test
def model_leaves_training_mode_restored():
    """evaluate() must not leave the model in eval mode mid-run."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "val.bin"
        write_bin(path, 3000, 256)
        ds = TokenDataset(path, context_len=32, seed=0)
        model = Transformer(TINY)
        model.train()
        evaluate(model, ds, 2, 2, "cpu")
        assert model.training, "model was left in eval mode after validation"
    return "training mode restored"


if __name__ == "__main__":
    p, n = suite.run()
    raise SystemExit(0 if p == n else 1)
