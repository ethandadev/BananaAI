"""The corpus mix.

Ratios are token shares of the 7B-token budget, set for a model that should
handle general chat and code. Code is a quarter of the mix -- enough for the
model to learn syntax and common idioms, not so much that prose fluency
suffers.

Nothing here downloads anything on import. `stream()` is a generator, so the
pipeline pulls documents lazily and never materialises a dataset on disk
beyond the shards it writes itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional


@dataclass(frozen=True)
class Source:
    name: str
    share: float               # fraction of the total token budget
    kind: str                  # 'prose' or 'code' -- selects the filter ruleset
    hf_path: str
    hf_config: Optional[str] = None
    split: str = "train"
    text_key: str = "text"
    languages: tuple = field(default_factory=tuple)   # code sources only

    def target_tokens(self, budget: int) -> int:
        return int(self.share * budget)


TOKEN_BUDGET = 7_000_000_000

MIX: tuple[Source, ...] = (
    Source("fineweb-edu", 0.55, "prose", "HuggingFaceFW/fineweb-edu", "sample-10BT"),
    Source("starcoder",   0.25, "code",  "bigcode/the-stack-smol", text_key="content",
           languages=("python", "javascript", "typescript", "rust", "c", "go", "shell")),
    Source("wikipedia",   0.08, "prose", "wikimedia/wikipedia", "20231101.en"),
    Source("stackexchange", 0.07, "prose", "HuggingFaceTB/stackexchange-clean"),
    Source("openwebmath", 0.05, "prose", "open-web-math/open-web-math"),
)


def validate_mix(mix=MIX) -> None:
    total = sum(s.share for s in mix)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"source shares sum to {total}, not 1.0")


def stream_hf(source: Source, limit: Optional[int] = None) -> Iterator[dict]:
    """Stream documents from HuggingFace.

    Imported lazily so the rest of the pipeline -- and its tests -- run without
    the datasets package or a network connection.
    """
    from datasets import load_dataset

    ds = load_dataset(
        source.hf_path, source.hf_config, split=source.split, streaming=True
    )
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            return
        text = row.get(source.text_key)
        if not text:
            continue
        yield {"text": text, "source": source.name, "kind": source.kind}


def stream_local(path: Path, name: str, kind: str, limit: Optional[int] = None) -> Iterator[dict]:
    """Stream JSONL from disk. Used for smoke tests and for your own data.

    Accepts either a single .jsonl file or a directory of them.
    """
    path = Path(path)
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    n = 0
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                if limit is not None and n >= limit:
                    return
                row = json.loads(line)
                text = row.get("text") or row.get("content")
                if not text:
                    continue
                n += 1
                yield {
                    "text": text,
                    "source": row.get("source", name),
                    "kind": row.get("kind", kind),
                }


def resolve(spec: str) -> Source:
    """Look up a source by name, for the --only flag."""
    for s in MIX:
        if s.name == spec:
            return s
    known = ", ".join(s.name for s in MIX)
    raise KeyError(f"unknown source {spec!r} -- known sources: {known}")
