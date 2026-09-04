"""Ingesting a folder of your own documents."""

from __future__ import annotations

import json

from tests.harness import Suite, temp_dir

from data.custom import (
    CHARS_PER_TOKEN, MAX_FILE_BYTES, SKIP_DIRS, assess, classify, documents,
    read_text, walk,
)

suite = Suite("custom-data")
test = suite.test

PROSE = (
    "The training loop measures a clear signal in the loss curve. However, the "
    "optimizer requires several unexpected failure modes to be handled first. "
    "In practice, this dataset explains the relationship between depth and width "
    "and it produces a substantial improvement in throughput for the model. "
    "By contrast, a checkpoint validates consistent results across every seed."
)
CODE = "\n".join(f"def step_{i}(x, y):\n    return (x * {i}) + y / 2.0\n" for i in range(30))


def build_folder(tmp):
    (tmp / "notes").mkdir()
    (tmp / "notes" / "a.md").write_text(PROSE, encoding="utf-8")
    (tmp / "notes" / "b.txt").write_text(PROSE.replace("loss", "gradient"), encoding="utf-8")
    (tmp / "src").mkdir()
    (tmp / "src" / "main.py").write_text(CODE, encoding="utf-8")

    # ignored by extension, before anything is read
    (tmp / "photo.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 500)
    (tmp / "data.bin").write_bytes(b"\x00\x01\x02" * 4000)
    # a text extension that is not text -- only content sniffing catches this
    (tmp / "notes" / "corrupt.txt").write_bytes(b"\x00\x01\x02" * 4000)
    (tmp / ".git").mkdir()
    (tmp / ".git" / "config").write_text("[core]\n", encoding="utf-8")
    (tmp / "node_modules").mkdir()
    (tmp / "node_modules" / "lib.js").write_text(CODE, encoding="utf-8")
    return tmp


@test
def suffixes_are_classified():
    assert classify(__import__("pathlib").Path("a.md")) == "prose"
    assert classify(__import__("pathlib").Path("a.py")) == "code"
    assert classify(__import__("pathlib").Path("a.jsonl")) == "jsonl"
    assert classify(__import__("pathlib").Path("a.pdf")) == "pdf"
    assert classify(__import__("pathlib").Path("a.jpg")) is None
    return "prose, code, jsonl, pdf recognised; images ignored"


@test
def walk_skips_vcs_and_dependency_folders():
    with temp_dir() as tmp:
        build_folder(tmp)
        names = {p.name for p in walk(tmp)}
    assert "a.md" in names and "main.py" in names, names
    assert "config" not in names, ".git was walked into"
    assert "lib.js" not in names, "node_modules was walked into"
    assert ".git" in SKIP_DIRS and "node_modules" in SKIP_DIRS
    return f"{len(names)} files found, .git and node_modules skipped"


@test
def binary_files_are_detected_by_content():
    with temp_dir() as tmp:
        (tmp / "b.txt").write_bytes(b"hello\x00world" + b"\x00" * 100)
        assert read_text(tmp / "b.txt") is None, "a NUL-containing file was read as text"
        (tmp / "c.txt").write_text("plain text", encoding="utf-8")
        assert read_text(tmp / "c.txt") == "plain text"
    return "NUL bytes in the first block mark a file binary"


@test
def invalid_utf8_is_replaced_not_fatal():
    with temp_dir() as tmp:
        (tmp / "x.txt").write_bytes("café".encode("latin-1"))
        got = read_text(tmp / "x.txt")
    assert got is not None and "caf" in got, got
    return "a bad byte becomes U+FFFD instead of raising"


@test
def a_folder_yields_documents_with_the_pipeline_shape():
    with temp_dir() as tmp:
        build_folder(tmp)
        stats: dict = {}
        docs = list(documents(tmp, "mine", stats=stats))
    assert docs, "nothing ingested"
    for d in docs:
        assert set(d) >= {"text", "source", "kind"}, d.keys()
        assert d["source"] == "mine"
        assert d["kind"] in ("prose", "code")
    kinds = {d["kind"] for d in docs}
    assert kinds == {"prose", "code"}, f"expected both kinds, got {kinds}"
    assert stats["kept"] == len(docs)
    assert stats.get("skipped_binary", 0) >= 1, "the binary file was not counted"
    return f"{len(docs)} documents, both kinds, stats recorded"


@test
def jsonl_files_are_expanded_row_by_row():
    with temp_dir() as tmp:
        rows = [{"text": PROSE}, {"text": PROSE.replace("loss", "signal")},
                {"content": CODE, "kind": "code"}]
        (tmp / "d.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        docs = list(documents(tmp, "mine"))
    assert len(docs) == 3, f"expected 3 rows, got {len(docs)}"
    assert any(d["kind"] == "code" for d in docs), "the kind field was ignored"
    return "3 rows became 3 documents, kind honoured"


@test
def malformed_jsonl_lines_are_counted_not_fatal():
    with temp_dir() as tmp:
        (tmp / "d.jsonl").write_text(
            json.dumps({"text": PROSE}) + "\n{ broken\n", encoding="utf-8")
        stats: dict = {}
        docs = list(documents(tmp, "mine", stats=stats))
    assert len(docs) == 1, "a broken line stopped the whole file"
    assert stats.get("skipped_bad_json") == 1
    return "one good row kept, one bad row counted"


@test
def quality_filters_apply_by_default_and_can_be_waived():
    with temp_dir() as tmp:
        (tmp / "junk.txt").write_text("hi there", encoding="utf-8")   # far too short
        with_filters = list(documents(tmp, "mine", apply_filters=True))
        without = list(documents(tmp, "mine", apply_filters=False))
    assert not with_filters, "a two-word file passed the quality rules"
    assert len(without) == 1, "--no-filters did not keep it"
    return "filtered by default, kept when waived"


@test
def oversized_files_are_skipped():
    with temp_dir() as tmp:
        big = tmp / "huge.txt"
        with open(big, "w", encoding="utf-8") as fh:
            fh.write("word " * (MAX_FILE_BYTES // 4))
        stats: dict = {}
        list(documents(tmp, "mine", stats=stats))
    assert stats.get("skipped_too_large") == 1, stats
    return f"files over {MAX_FILE_BYTES // 1024**2} MB are skipped"


@test
def a_missing_folder_raises_clearly():
    try:
        list(documents(__import__("pathlib").Path("no-such-folder-anywhere")))
    except FileNotFoundError as e:
        assert "no such folder" in str(e)
        return "a missing folder raises FileNotFoundError"
    raise AssertionError("silently accepted a missing folder")


@test
def assessment_is_honest_about_a_tiny_folder():
    """The common case: a personal folder is nowhere near enough to pretrain."""
    got = assess(characters=2_000_000, model_params=326_685_696)
    assert got["share_of_budget"] < 0.01, got
    assert "fine-tune" in got["advice"], got["advice"]
    return f"2M chars vs 327M params -> {got['verdict']}"


@test
def assessment_recommends_blending_at_a_useful_share():
    chars = int(326_685_696 * 21 * 0.2 * CHARS_PER_TOKEN)   # 20% of the budget
    got = assess(chars, 326_685_696)
    assert "blend" in got["advice"], got["advice"]
    assert 0.15 < got["share_of_budget"] < 0.25, got["share_of_budget"]
    return f"20% of budget -> {got['advice'][:48]}..."


@test
def assessment_recognises_a_corpus_big_enough_alone():
    chars = int(13_000_000 * 21 * 1.5 * CHARS_PER_TOKEN)    # 1.5x a nano budget
    got = assess(chars, 13_000_000)
    assert got["share_of_budget"] >= 1.0
    assert "train directly" in got["advice"], got["advice"]
    return "a sufficient corpus is recognised as such"


@test
def assessment_dismisses_a_handful_of_files():
    got = assess(characters=5_000, model_params=326_685_696)
    assert "far too small" in got["advice"], got["advice"]
    return "5k characters -> told plainly it is not a training set"


if __name__ == "__main__":
    p, n = suite.run()
    raise SystemExit(0 if p == n else 1)
