"""Generate a deterministic test corpus.

    python tests/fixtures/build.py

Produces a small JSONL corpus that deliberately contains every case the
pipeline is supposed to handle: clean prose, real source code, documents that
each filter should reject, exact duplicates, and near-duplicates that differ by
a few words. Tests assert on the exact counts, so the seed is fixed.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

SUBJECTS = [
    "the compiler", "a transformer model", "the training loop", "this dataset",
    "the tokenizer", "a gradient", "the optimizer", "our benchmark",
    "the researcher", "an engineer", "the kernel", "that experiment",
    "the corpus", "a checkpoint", "the scheduler", "this architecture",
]
VERBS = [
    "produces", "requires", "explains", "avoids", "measures", "improves",
    "describes", "compresses", "predicts", "validates", "reduces", "reveals",
]
OBJECTS = [
    "a substantial improvement in throughput", "the underlying distribution",
    "several unexpected failure modes", "an order of magnitude less memory",
    "the relationship between depth and width", "a clear signal in the loss curve",
    "consistent results across every random seed", "the cost of attention at long context",
    "meaningful gains on held-out data", "the same answer by a different route",
]
CONNECTIVES = [
    "However,", "In practice,", "As a result,", "By contrast,", "More importantly,",
    "For this reason,", "In the general case,", "Surprisingly,",
]


def sentence(rng: random.Random) -> str:
    parts = []
    if rng.random() < 0.4:
        parts.append(rng.choice(CONNECTIVES))
    parts += [rng.choice(SUBJECTS), rng.choice(VERBS), rng.choice(OBJECTS)]
    s = " ".join(parts)
    return s[0].upper() + s[1:] + "."


def paragraph(rng: random.Random, n: int) -> str:
    return " ".join(sentence(rng) for _ in range(n))


def prose_doc(rng: random.Random) -> str:
    return "\n\n".join(paragraph(rng, rng.randint(4, 9)) for _ in range(rng.randint(3, 6)))


def real_code_docs() -> list[str]:
    """Use the project source as the code half -- it is real code, already here.

    Only files that actually survive the code filter are included. Otherwise
    the expected counts drift every time a new module is added to the repo,
    and a test that should be checking the pipeline starts failing for a
    reason that has nothing to do with the pipeline.
    """
    import sys

    sys.path.insert(0, str(REPO))
    from data.filters import code_reject_reason, normalise

    out = []
    for path in sorted(REPO.rglob("*.py")):
        if any(part in str(path) for part in (".venv", "fixtures")):
            continue
        text = path.read_text(encoding="utf-8")
        if code_reject_reason(normalise(text)) is None:
            out.append(text)
    return out


def junk_docs() -> list[tuple[str, str]]:
    """(text, the filter reason it should trigger)."""
    return [
        ("short.", "too_short"),
        ("aa bb cc " * 200, "no_stop_words"),
        ("\n".join("- a bullet point about the thing" for _ in range(60)), "all_bullets"),
        ("\n".join(f"Line {i} of the text continues..." for i in range(60)), "truncated_lines"),
        ("the same line repeated\n" * 80, "duplicate_lines"),
        ("the model the model " * 300, "repeated_bigram"),
        ("### " * 500 + " the quick brown fox jumps over a lazy dog " * 20, "symbol_ratio"),
        ("x " * 500, "mean_word_length"),
    ]


def build(out_dir: Path = HERE / "corpus", seed: int = 7) -> dict:
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    docs: list[dict] = []
    expect = {}

    # 60 clean prose documents
    clean = [prose_doc(rng) for _ in range(60)]
    for text in clean:
        docs.append({"text": text, "source": "synthetic-prose", "kind": "prose"})

    # 6 exact duplicates of existing prose
    for text in clean[:6]:
        docs.append({"text": text, "source": "synthetic-prose", "kind": "prose"})
    expect["exact_duplicates"] = 6

    # 5 near-duplicates: same document with one sentence appended
    for text in clean[10:15]:
        docs.append({
            "text": text + " " + sentence(rng),
            "source": "synthetic-prose",
            "kind": "prose",
        })
    expect["near_duplicates"] = 5

    # real source files
    code = real_code_docs()
    for text in code:
        docs.append({"text": text, "source": "repo-code", "kind": "code"})
    expect["code_docs"] = len(code)

    # documents every filter should reject
    junk = junk_docs()
    for text, _reason in junk:
        docs.append({"text": text, "source": "junk", "kind": "prose"})
    expect["junk_docs"] = len(junk)
    expect["clean_prose"] = len(clean)

    rng.shuffle(docs)

    path = out_dir / "corpus.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(json.dumps(doc, ensure_ascii=False) + "\n")

    expect["total_written"] = len(docs)
    (out_dir / "expected.json").write_text(json.dumps(expect, indent=2), encoding="utf-8")
    return expect


if __name__ == "__main__":
    info = build()
    print(json.dumps(info, indent=2))
