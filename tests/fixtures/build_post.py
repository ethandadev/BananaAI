"""Generate small SFT and DPO fixtures.

    python tests/fixtures/build_post.py

The DPO pairs are built so 'chosen' and 'rejected' differ in a way the tests
can reason about: chosen answers the question, rejected is evasive or
repetitive. That gives DPO an actual preference direction to learn rather than
noise.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent

TOPICS = [
    ("what a tokenizer does", "A tokenizer splits text into the discrete units a model reads."),
    ("why we use a validation split", "A held-out split measures generalisation rather than memorisation."),
    ("what gradient clipping prevents", "Clipping bounds the update size so one bad batch cannot blow up training."),
    ("what an embedding is", "An embedding maps a discrete token to a dense vector the network can use."),
    ("why attention is quadratic", "Every position attends to every other, so cost grows with the square of length."),
    ("what a learning rate warmup does", "Warmup raises the rate slowly so early updates do not destabilise the model."),
    ("what bf16 buys you", "It halves memory versus fp32 while keeping the exponent range that avoids overflow."),
    ("why we shuffle training data", "Shuffling stops the model learning the order of the corpus instead of the language."),
    ("what perplexity measures", "Perplexity is the exponential of the loss, read as an effective branching factor."),
    ("what a residual connection does", "It lets gradients reach early layers directly instead of vanishing through depth."),
    ("what weight decay does", "It pulls weights toward zero, which discourages the model from relying on any single one."),
    ("what a KV cache saves", "It stores past keys and values so each new token costs linear work, not quadratic."),
]

EVASIVE = [
    "That is a really interesting question to think about.",
    "It depends on a lot of different factors, honestly.",
    "I am not sure I can explain that very well right now.",
    "Well, that is complicated and hard to summarise.",
]


def build(seed: int = 3) -> dict:
    rng = random.Random(seed)

    sft = []
    for question, answer in TOPICS:
        sft.append({"messages": [
            {"role": "user", "content": f"Explain {question}."},
            {"role": "assistant", "content": answer},
        ]})
        sft.append({"messages": [
            {"role": "system", "content": "You are a concise technical assistant."},
            {"role": "user", "content": f"In one sentence, {question}?"},
            {"role": "assistant", "content": answer},
        ]})

    prefs = []
    for question, answer in TOPICS:
        prefs.append({
            "prompt": f"Explain {question}.",
            "chosen": answer,
            "rejected": rng.choice(EVASIVE),
        })

    val = sft[: 6]

    (HERE / "sft.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in sft) + "\n", encoding="utf-8")
    (HERE / "sft_val.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in val) + "\n", encoding="utf-8")
    (HERE / "prefs.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in prefs) + "\n", encoding="utf-8")

    return {"sft_examples": len(sft), "val_examples": len(val), "preference_pairs": len(prefs)}


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
