"""SFT and DPO: collation, masking, and the preference objective."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import torch

from tests.harness import REPO, Suite, ensure_fixtures

ensure_fixtures()

from core.config import ModelConfig
from core.model import Transformer
from core.tokenizer import PAD_ID, Message, train as train_tokenizer
from core.train import PRESETS
from post.dpo import PreferenceDataset, dpo_loss, sequence_logprob
from post.sft import IGNORE, ConversationDataset, read_conversations, run_sft

suite = Suite("post-training")
test = suite.test

FIXTURES = REPO / "tests" / "fixtures"
TINY = ModelConfig(**PRESETS["test"], vocab_size=512)

_TOK = None


def tokenizer():
    global _TOK
    if _TOK is None:
        texts = [c["messages"][-1]["content"] for c in
                 (json.loads(l) for l in (FIXTURES / "sft.jsonl").read_text().splitlines() if l)]
        texts += ["the model explains a concept in one clear sentence"] * 20
        _TOK = train_tokenizer(texts, vocab_size=512, min_frequency=1)
    return _TOK


def base_checkpoint(tmp: Path) -> Path:
    torch.manual_seed(0)
    model = Transformer(TINY)
    path = tmp / "base.pt"
    torch.save({"model": model.state_dict(),
                "model_config": {k: v for k, v in vars(TINY).items()
                                 if not k.startswith("_")},
                "step": 0, "stage": "base"}, path)
    return path


@test
def conversations_load_from_jsonl():
    convs = read_conversations(FIXTURES / "sft.jsonl")
    assert len(convs) == 24, f"loaded {len(convs)} conversations"
    assert all(isinstance(m, Message) for c in convs for m in c)
    assert any(m.role == "system" for c in convs for m in c), "no system turns in the fixture"
    return f"{len(convs)} conversations, roles parsed"


@test
def collate_pads_and_shifts():
    ds = ConversationDataset(read_conversations(FIXTURES / "sft.jsonl"),
                             tokenizer(), TINY.context_len)
    x, y = next(iter(ds.batches(4, "cpu", shuffle=False)))
    assert x.shape == y.shape, f"{x.shape} != {y.shape}"
    assert x.size(0) == 4
    # The shift: input position i must predict label position i, which is the
    # *next* token of the original sequence.
    assert (y != IGNORE).any(), "every label was masked"
    return f"batch {tuple(x.shape)}, padded and shifted"


@test
def padding_never_contributes_to_the_loss():
    ds = ConversationDataset(read_conversations(FIXTURES / "sft.jsonl"),
                             tokenizer(), TINY.context_len)
    x, y = next(iter(ds.batches(6, "cpu", shuffle=False)))
    pad_positions = (x == PAD_ID)
    # Every padded input position must carry an ignored label.
    overlap = pad_positions & (y != IGNORE)
    assert not overlap.any(), f"{overlap.sum().item()} padded positions are supervised"
    return "all padded positions carry -100"


@test
def examples_with_nothing_supervised_are_skipped():
    convs = [[Message("user", "a question with no answer at all")]]
    ds = ConversationDataset(convs, tokenizer(), TINY.context_len)
    assert len(ds) == 0, "kept an example with a zero gradient"
    assert ds.skipped == 1
    return "user-only conversation dropped"


@test
def over_length_examples_are_skipped_not_truncated():
    long_answer = " ".join(["word"] * 5000)
    convs = [[Message("user", "hi"), Message("assistant", long_answer)]]
    ds = ConversationDataset(convs, tokenizer(), max_len=64)
    assert len(ds) == 0 and ds.skipped == 1, "an over-length example was kept"
    return "truncation would corrupt the answer, so it is dropped"


@test
def sft_reduces_loss_on_its_own_data():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        tok_path = tmp / "tok.json"
        tokenizer().save(str(tok_path))
        summary = run_sft(
            base_checkpoint(tmp), FIXTURES / "sft.jsonl", tmp / "out", tok_path,
            epochs=6, batch_size=4, lr=1e-3, device="cpu", log_every=10**9,
        )
        assert summary["examples"] == 24
        assert summary["final_loss"] < math.log(TINY.vocab_size), \
            f"final loss {summary['final_loss']:.3f} no better than random"
        assert Path(summary["checkpoint"]).exists()
    return f"loss {summary['final_loss']:.3f} < ln(512) = 6.24"


@test
def dpo_loss_is_ln2_when_policy_equals_reference():
    zeros = torch.zeros(4)
    loss, metrics = dpo_loss(zeros, zeros, zeros, zeros, beta=0.1)
    assert abs(loss.item() - math.log(2)) < 1e-6, f"got {loss.item():.6f}"
    assert metrics["margin"] == 0.0
    return f"loss {loss.item():.4f} = ln 2, margin 0"


@test
def dpo_loss_falls_as_the_margin_grows():
    ref = torch.zeros(1)
    small, _ = dpo_loss(torch.tensor([0.5]), torch.tensor([0.0]), ref, ref, beta=1.0)
    large, m = dpo_loss(torch.tensor([5.0]), torch.tensor([0.0]), ref, ref, beta=1.0)
    assert large < small, "a wider margin did not reduce the loss"
    assert m["accuracy"] == 1.0
    return f"margin 0.5 -> {small.item():.3f}, margin 5.0 -> {large.item():.3f}"


@test
def dpo_penalises_the_wrong_preference():
    ref = torch.zeros(1)
    wrong, m = dpo_loss(torch.tensor([0.0]), torch.tensor([3.0]), ref, ref, beta=1.0)
    assert wrong.item() > math.log(2), "preferring the rejected answer was not penalised"
    assert m["accuracy"] == 0.0
    return f"inverted preference gives loss {wrong.item():.3f} > ln 2"


@test
def reference_model_cancels_out_of_the_objective():
    """Adding a constant to both reference terms must not change the loss."""
    pc, pr = torch.tensor([1.0]), torch.tensor([0.0])
    a, _ = dpo_loss(pc, pr, torch.tensor([0.0]), torch.tensor([0.0]), beta=0.1)
    b, _ = dpo_loss(pc, pr, torch.tensor([7.0]), torch.tensor([7.0]), beta=0.1)
    assert abs(a.item() - b.item()) < 1e-6, "a constant reference shift changed the loss"
    return "only the reference log-ratio matters"


@test
def sequence_logprob_ignores_masked_positions():
    torch.manual_seed(0)
    model = Transformer(TINY).eval()
    x = torch.randint(0, TINY.vocab_size, (2, 16))
    y = x.clone()
    y[:, :8] = IGNORE

    with torch.no_grad():
        masked = sequence_logprob(model, x, y)
        full = sequence_logprob(model, x, x)
    assert (masked > full).all(), "masking did not reduce the summed log-probability"
    assert (masked < 0).all(), "log-probabilities must be negative"
    return "masked positions excluded from the sum"


@test
def preference_pairs_load_and_batch():
    ds = PreferenceDataset(FIXTURES / "prefs.jsonl", tokenizer(), TINY.context_len)
    assert len(ds) == 12, f"loaded {len(ds)} pairs"
    (cx, cy), (rx, ry) = next(iter(ds.batches(4, "cpu", shuffle=False)))
    assert cx.size(0) == rx.size(0) == 4, "chosen and rejected batch sizes differ"
    assert (cy != IGNORE).any() and (ry != IGNORE).any(), "no supervised positions"
    return f"{len(ds)} pairs, chosen/rejected collated separately"


if __name__ == "__main__":
    p, n = suite.run()
    raise SystemExit(0 if p == n else 1)
