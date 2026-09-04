"""Quality heuristics for the pretraining corpus.

Prose and code need genuinely different rules: a filter that rejects documents
for a high symbol-to-word ratio would throw away every source file, and one
that demands stop words would throw away JSON. So there are two filters, and
each source declares which one applies.

Everything here works on plain strings and returns a reason for rejection, so
the pipeline can report exactly why a corpus shrank.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Optional

# A small closed-class set. Their near-total absence from a long document is a
# reliable signal it is not running English prose -- it is a link farm, a table
# dump, or navigation chrome.
STOP_WORDS = frozenset(
    "the be to of and a in that have i it for not on with he as you do at this "
    "but his by from they we say her she or an will my one all would there their".split()
)

CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)


def normalise(text: str) -> str:
    """Canonical unicode, no control characters, no trailing whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = CONTROL_CHARS.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in text.split("\n")]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


@dataclass
class FilterStats:
    """Counts of why documents were dropped, for the pipeline report."""

    kept: int = 0
    dropped: Counter = None

    def __post_init__(self):
        if self.dropped is None:
            self.dropped = Counter()

    def record(self, reason: Optional[str]) -> bool:
        if reason is None:
            self.kept += 1
            return True
        self.dropped[reason] += 1
        return False

    @property
    def total(self) -> int:
        return self.kept + sum(self.dropped.values())

    def report(self) -> str:
        if not self.total:
            return "no documents seen"
        pct = 100 * self.kept / self.total
        lines = [f"kept {self.kept:,} of {self.total:,} ({pct:.1f}%)"]
        for reason, n in self.dropped.most_common():
            lines.append(f"    {reason:<24} {n:>10,}")
        return "\n".join(lines)


def _repetition_ratio(lines: list[str]) -> float:
    """Fraction of lines that are duplicates of an earlier line."""
    if len(lines) < 2:
        return 0.0
    seen, dupes = set(), 0
    for ln in lines:
        key = ln.strip()
        if not key:
            continue
        if key in seen:
            dupes += 1
        seen.add(key)
    return dupes / len(lines)


def _top_ngram_ratio(words: list[str], n: int) -> float:
    """Share of the document taken by its single most common n-gram."""
    if len(words) < n * 2:
        return 0.0
    grams = Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
    top, count = grams.most_common(1)[0]
    return (count * n) / len(words)


def prose_reject_reason(
    text: str, min_words: int = 50, max_words: int = 100_000
) -> Optional[str]:
    """Gopher-style quality rules. Returns None if the document should be kept."""
    words = WORD_RE.findall(text)
    n = len(words)

    if n < min_words:
        return "too_short"
    if n > max_words:
        return "too_long"

    mean_len = sum(len(w) for w in words) / n
    if not 3.0 <= mean_len <= 10.0:
        return "mean_word_length"

    alpha = sum(1 for w in words if any(c.isalpha() for c in w))
    if alpha / n < 0.80:
        return "not_alphabetic"

    lowered = {w.lower() for w in words}
    if len(lowered & STOP_WORDS) < 2:
        return "no_stop_words"

    symbols = text.count("#") + text.count("...") + text.count("…")
    if symbols / n > 0.10:
        return "symbol_ratio"

    lines = text.split("\n")
    bullets = sum(1 for ln in lines if ln.lstrip().startswith(("*", "-", "•")))
    if lines and bullets / len(lines) > 0.90:
        return "all_bullets"

    ellipsis = sum(1 for ln in lines if ln.rstrip().endswith(("...", "…")))
    if lines and ellipsis / len(lines) > 0.30:
        return "truncated_lines"

    if _repetition_ratio(lines) > 0.30:
        return "duplicate_lines"
    if _top_ngram_ratio(words, 2) > 0.20:
        return "repeated_bigram"

    return None


def code_reject_reason(
    text: str, min_chars: int = 200, max_chars: int = 200_000
) -> Optional[str]:
    """Rules tuned for source files rather than prose."""
    n = len(text)
    if n < min_chars:
        return "too_short"
    if n > max_chars:
        return "too_long"

    lines = text.split("\n")
    if not lines:
        return "empty"

    # Minified or generated files: enormous lines, no structure to learn from.
    mean_line = n / len(lines)
    if mean_line > 200:
        return "likely_minified"
    if max(len(ln) for ln in lines) > 2000:
        return "very_long_line"

    alnum = sum(1 for c in text if c.isalnum())
    if alnum / n < 0.25:
        return "low_alphanumeric"

    # Base64 blobs, lockfiles, and vendored bundles are mostly one long token
    # stream with no whitespace.
    if text.count(" ") / n < 0.02:
        return "no_whitespace"

    if _repetition_ratio(lines) > 0.50:
        return "duplicate_lines"

    return None


def reject_reason(text: str, kind: str) -> Optional[str]:
    """Dispatch to the right ruleset. `kind` is 'prose' or 'code'."""
    if kind == "code":
        return code_reject_reason(text)
    if kind == "prose":
        return prose_reject_reason(text)
    raise ValueError(f"unknown document kind: {kind!r}")
