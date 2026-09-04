"""Tokenizer: training, round-tripping, and the chat template."""

from __future__ import annotations

import random

from tests.harness import Suite

from core.tokenizer import (
    BOS_ID, EOS_ID, PAD_ID, SPECIAL_TOKENS, Message, compression_ratio,
    encode_conversation, render, train,
)

suite = Suite("tokenizer")
test = suite.test

WORDS = ("model token gradient corpus attention vector layer batch weight loss "
         "kernel matrix sample entropy encoder decoder").split()


def corpus(n: int = 400, seed: int = 5) -> list[str]:
    rng = random.Random(seed)
    docs = []
    for _ in range(n):
        sentences = [
            " ".join(rng.choice(WORDS) for _ in range(rng.randint(6, 14))) + "."
            for _ in range(rng.randint(3, 7))
        ]
        docs.append(" ".join(sentences))
    # A little code, so the merges are not purely prose.
    for i in range(40):
        docs.append(f"def step_{i}(x, y):\n    return (x * {i}) + y / 2.0\n")
    return docs


_TOK = None


def tokenizer():
    global _TOK
    if _TOK is None:
        _TOK = train(corpus(), vocab_size=1200, min_frequency=1)
    return _TOK


@test
def special_tokens_get_the_expected_ids():
    tok = tokenizer()
    assert tok.token_to_id("<|pad|>") == PAD_ID
    assert tok.token_to_id("<|bos|>") == BOS_ID
    assert tok.token_to_id("<|eos|>") == EOS_ID
    for name in SPECIAL_TOKENS:
        assert tok.token_to_id(name) is not None, f"{name} missing from the vocabulary"
    return f"{len(SPECIAL_TOKENS)} specials at fixed low ids"


@test
def encoding_round_trips_exactly():
    tok = tokenizer()
    for text in [
        "the model produces a gradient",
        "def f(x):\n    return x * 2\n",
        "unicode: naive cafe résumé 日本語 emoji 🎉",
        "   leading and trailing whitespace   ",
        "tabs\tand\nnewlines\r\nmixed",
    ]:
        out = tok.decode(tok.encode(text, add_special_tokens=False).ids)
        assert out == text, f"round trip changed {text!r} into {out!r}"
    return "5 strings including unicode and whitespace survive intact"


@test
def byte_level_has_no_unknown_token():
    """Byte fallback is the reason there is no <unk> and no data loss."""
    tok = tokenizer()
    exotic = "​́ ᚠᚢᚦᚨ 𝕳𝖊𝖑𝖑𝖔 \x1b[31m"
    ids = tok.encode(exotic, add_special_tokens=False).ids
    assert ids, "produced no tokens at all"
    assert tok.decode(ids) == exotic, "unseen characters were not reconstructed"
    return "never-seen scripts round trip byte-exactly"


@test
def digits_are_split_individually():
    tok = tokenizer()
    ids = tok.encode("31415926", add_special_tokens=False).ids
    assert len(ids) == 8, f"8 digits produced {len(ids)} tokens, expected 8"
    return "31415926 -> 8 tokens, one per digit"


@test
def compression_is_measured_not_assumed():
    tok = tokenizer()
    ratio = compression_ratio(tok, corpus(50, seed=99))
    assert ratio > 1.5, f"chars per token is only {ratio:.2f}"
    return f"{ratio:.2f} chars/token on held-out text"


@test
def chat_template_is_well_formed():
    out = render([Message("user", "hi"), Message("assistant", "hello")])
    assert out.startswith("<|bos|>"), "no BOS"
    assert out == "<|bos|><|user|>hi<|end|><|assistant|>hello<|end|>", out
    return "roles and terminators in the expected order"


@test
def generation_prompt_leaves_the_turn_open():
    out = render([Message("user", "hi")], add_generation_prompt=True)
    assert out.endswith("<|assistant|>"), out
    assert not out.endswith("<|end|>"), "closed the assistant turn before generation"
    return "trailing <|assistant|> with no terminator"


@test
def template_rejects_unknown_roles():
    try:
        render([Message("wizard", "abracadabra")])
    except ValueError:
        return "unknown role raises"
    raise AssertionError("an unknown role was accepted")


@test
def only_assistant_content_is_supervised():
    tok = tokenizer()
    ids, labels = encode_conversation(
        tok, [Message("user", "what is a gradient"), Message("assistant", "a derivative")]
    )
    assert len(ids) == len(labels), "labels are not aligned to inputs"

    supervised = [i for i, l in enumerate(labels) if l != -100]
    assert supervised, "nothing is supervised; the gradient would be zero"
    for i in supervised:
        assert labels[i] == ids[i], "supervised labels must equal the inputs"

    # The user's text must be entirely masked out.
    user_ids = tok.encode("what is a gradient", add_special_tokens=False).ids
    first = supervised[0]
    assert first > len(user_ids), "user content leaked into the supervised region"
    return f"{len(supervised)} of {len(labels)} positions supervised, prompt masked"


@test
def system_turns_are_masked_too():
    tok = tokenizer()
    _, labels = encode_conversation(tok, [
        Message("system", "you are a concise assistant"),
        Message("user", "hi"),
        Message("assistant", "hello"),
    ])
    supervised = sum(1 for l in labels if l != -100)
    total = len(labels)
    assert 0 < supervised < total * 0.5, \
        f"{supervised}/{total} supervised -- the system prompt is not masked"
    return f"{supervised}/{total} positions supervised"


if __name__ == "__main__":
    p, n = suite.run()
    raise SystemExit(0 if p == n else 1)
