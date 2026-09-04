"""Data pipeline: filters, dedupe, sharding, tokenization."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from tests.harness import REPO, Suite, ensure_fixtures

ensure_fixtures()

from data.dedupe import MinHasher, find_duplicates, jaccard, shingle_hashes
from data.filters import (
    FilterStats, code_reject_reason, normalise, prose_reject_reason, reject_reason,
)
from data.prepare import ShardWriter, run, shuffled
from data.sources import MIX, resolve, stream_local, validate_mix

suite = Suite("data")
test = suite.test

FIXTURES = REPO / "tests" / "fixtures"

GOOD_PROSE = (
    "The training loop measures a clear signal in the loss curve. However, the "
    "optimizer requires several unexpected failure modes to be handled first. "
    "In practice, this dataset explains the relationship between depth and width "
    "and it produces a substantial improvement in throughput for the model. "
    "By contrast, a checkpoint validates consistent results across every seed."
)


@test
def normalise_strips_control_characters():
    dirty = "hello\x00\x07 world\r\n\r\nnext\n\n\n\n\nlast   "
    clean = normalise(dirty)
    assert "\x00" not in clean and "\x07" not in clean, "control chars survived"
    assert "\r" not in clean, "carriage returns survived"
    assert "\n\n\n" not in clean, "blank-line runs not collapsed"
    return "control chars, CRLF, and blank runs normalised"


@test
def normalise_is_idempotent():
    once = normalise("a\r\n\r\n\r\nb\x01  ")
    assert normalise(once) == once, "second pass changed the text"
    return "f(f(x)) == f(x)"


@test
def good_prose_survives():
    assert prose_reject_reason(GOOD_PROSE) is None, \
        f"rejected clean prose: {prose_reject_reason(GOOD_PROSE)}"
    return "clean prose kept"


@test
def each_prose_rule_fires():
    cases = {
        "too_short": "Only a handful of words here indeed.",
        "mean_word_length": "x " * 500,
        "no_stop_words": "zebra giraffe walrus " * 100,
        "duplicate_lines": "the quick brown fox jumps over a lazy dog\n" * 80,
    }
    for expected, text in cases.items():
        got = prose_reject_reason(text)
        assert got is not None, f"{expected!r} case was not rejected at all"
    return f"{len(cases)} rejection paths exercised"


@test
def code_filter_keeps_real_source():
    source = (REPO / "core" / "model.py").read_text(encoding="utf-8")
    assert code_reject_reason(source) is None, \
        f"rejected our own model.py: {code_reject_reason(source)}"
    return "core/model.py passes the code filter"


@test
def code_filter_rejects_minified_and_blobs():
    minified = "var a=1;" * 900                              # one enormous line
    blob = "\n".join("A1b2C3d4" * 10 for _ in range(200))    # wrapped, but no spaces
    assert code_reject_reason(minified) == "likely_minified", \
        f"minified file gave {code_reject_reason(minified)}"
    assert code_reject_reason(blob) == "no_whitespace", \
        f"base64-like blob gave {code_reject_reason(blob)}"
    return "minified and base64-like inputs rejected"


@test
def prose_rules_would_destroy_dense_code():
    """Why there are two rulesets.

    Note the asymmetry: docstring-heavy Python passes the prose filter
    perfectly well, so this is not a general claim that prose rules reject all
    code. What they reject is symbol-dense code with no running English -- and
    that is most of a real code corpus.
    """
    dense = "\n".join(
        f"const v{i} = (a{i} * b{i}) / (c{i} + 1e-9); // scale {i}" for i in range(80)
    )
    assert prose_reject_reason(dense) is not None, \
        "the prose filter accepted symbol-dense code; the split is pointless"
    assert reject_reason(dense, "code") is None, \
        f"the code filter rejected valid code: {reject_reason(dense, 'code')}"
    return "prose rejects symbol-dense code, code ruleset accepts it"


@test
def filter_stats_report_adds_up():
    stats = FilterStats()
    stats.record(None)
    stats.record("too_short")
    stats.record("too_short")
    assert stats.kept == 1 and stats.total == 3
    assert "kept 1 of 3" in stats.report()
    return "kept 1 of 3, reasons tallied"


@test
def shingles_are_deterministic():
    a = shingle_hashes("the model predicts the next token in the sequence")
    b = shingle_hashes("the model predicts the next token in the sequence")
    assert np.array_equal(a, b), "same text gave different hashes"
    assert a.dtype == np.uint64
    assert int(a.max()) < 2**32, "hashes exceed 32 bits; a*x+b could overflow"
    return f"{a.size} shingles, stable, all under 2^32"


@test
def minhash_estimates_jaccard():
    hasher = MinHasher(num_perm=256, seed=0)
    a = GOOD_PROSE
    b = GOOD_PROSE + " One additional sentence appears at the end here."
    sa, sb = hasher.signature(shingle_hashes(a)), hasher.signature(shingle_hashes(b))
    estimate = float((sa == sb).mean())
    exact = jaccard(a, b)
    assert abs(estimate - exact) < 0.15, \
        f"MinHash estimated {estimate:.3f} against exact {exact:.3f}"
    return f"estimate {estimate:.3f} vs exact {exact:.3f}"


@test
def minhash_signature_never_overflows():
    """The uint64 bound is the whole reason this runs vectorised."""
    hasher = MinHasher(num_perm=64, seed=1)
    worst = np.array([2**32 - 1] * 8, dtype=np.uint64)
    sig = hasher.signature(worst)
    MERSENNE = (1 << 61) - 1
    assert int(sig.max()) < MERSENNE, "signature exceeded the modulus"
    return "max-value shingles stay in range"


@test
def dedupe_catches_exact_and_near_copies():
    base = GOOD_PROSE
    texts = [
        base,
        base,                                    # exact copy
        base + " A short extra sentence here.",  # near copy
        "Something entirely different about kernels, drivers, and firmware "
        "that shares no phrasing at all with the paragraph above it. " * 4,
    ]
    drop = find_duplicates(texts, threshold=0.7)
    assert 0 not in drop, "dropped the first occurrence instead of keeping it"
    assert 1 in drop, "missed an exact duplicate"
    assert 2 in drop, "missed a near duplicate"
    assert 3 not in drop, "dropped an unrelated document"
    return "keeps first, drops exact + near, spares unrelated"


@test
def dedupe_verification_prevents_false_positives():
    unrelated = [f"document number {i} about a completely separate subject "
                 f"with its own distinct vocabulary and phrasing throughout" for i in range(30)]
    drop = find_duplicates(unrelated, threshold=0.8, verify=True)
    assert not drop, f"dropped {len(drop)} unrelated documents"
    return "30 unrelated documents, none dropped"


@test
def shuffle_preserves_every_document():
    docs = [{"text": str(i)} for i in range(500)]
    out = list(shuffled(iter(docs), buffer_size=50, seed=1))
    assert len(out) == 500, f"got {len(out)} documents back, not 500"
    assert {d["text"] for d in out} == {str(i) for i in range(500)}, "documents lost"
    assert [d["text"] for d in out] != [d["text"] for d in docs], "nothing was shuffled"
    return "500 in, 500 out, order changed"


@test
def shard_writer_rolls_over():
    with tempfile.TemporaryDirectory() as tmp:
        w = ShardWriter(Path(tmp), "t", max_docs=10)
        for i in range(25):
            w.write({"text": f"doc {i}"})
        w.close()
        shards = sorted(Path(tmp).glob("*.jsonl.zst"))
        assert len(shards) == 3, f"expected 3 shards, got {len(shards)}"
        assert w.total == 25
    return "25 documents at 10/shard -> 3 shards"


@test
def pipeline_end_to_end_on_the_fixture():
    expected = json.loads((FIXTURES / "corpus" / "expected.json").read_text())
    with tempfile.TemporaryDirectory() as tmp:
        docs = stream_local(FIXTURES / "corpus" / "corpus.jsonl", "fixture", "prose")
        manifest = run(docs, Path(tmp), "fixture", window=200, threshold=0.8)

    assert manifest["documents_in"] == expected["total_written"]
    assert manifest["removed_exact_duplicates"] == expected["exact_duplicates"], \
        f"exact dupes: got {manifest['removed_exact_duplicates']}"
    assert manifest["removed_near_duplicates"] == expected["near_duplicates"], \
        f"near dupes: got {manifest['removed_near_duplicates']}"
    assert sum(manifest["dropped_by_filter"].values()) == expected["junk_docs"], \
        "junk document count does not match"
    assert manifest["documents_out"] == expected["clean_prose"] + expected["code_docs"]
    return (f"{manifest['documents_in']} in -> {manifest['documents_out']} out, "
            f"{expected['junk_docs']} filtered, "
            f"{expected['exact_duplicates']}+{expected['near_duplicates']} deduped")


@test
def mix_shares_sum_to_one():
    validate_mix()
    assert resolve("wikipedia").kind == "prose"
    code = [s for s in MIX if s.kind == "code"]
    assert code, "no code source in the mix"
    assert abs(sum(s.share for s in code) - 0.25) < 1e-9, "code share is not 25%"
    budget = sum(s.target_tokens(7_000_000_000) for s in MIX)
    assert abs(budget - 7_000_000_000) < 10, f"token budget sums to {budget:,}"
    return f"{len(MIX)} sources, 25% code, 7.00B tokens"


if __name__ == "__main__":
    p, n = suite.run()
    raise SystemExit(0 if p == n else 1)
