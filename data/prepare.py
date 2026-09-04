"""Corpus pipeline: stream, normalise, filter, dedupe, shard.

    python -m data.prepare --local tests/fixtures/corpus --out data/processed
    python -m data.prepare --source wikipedia --limit 50000 --out data/processed

Output is zstd-compressed JSONL shards plus a manifest recording exactly what
went in and what was dropped, so a corpus is reproducible from its manifest.

On dedupe scope: exact duplicates are caught globally with a hash set (cheap --
one 16-byte digest per document). Near-duplicates are caught with MinHash
within shards of `--dedupe-window` documents, because a global LSH index over
tens of millions of documents does not fit in 32 GB. Shuffling before sharding
means near-duplicates from the same origin usually land in the same window.
This is the standard compromise; it is not exhaustive, and the manifest
reports what each stage removed so you can see the effect.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator

import zstandard as zstd

from .dedupe import find_duplicates
from .filters import FilterStats, normalise, reject_reason
from .sources import MIX, resolve, stream_hf, stream_local


def normalise_and_filter(docs: Iterable[dict], stats: FilterStats) -> Iterator[dict]:
    for doc in docs:
        text = normalise(doc["text"])
        reason = reject_reason(text, doc["kind"]) if text else "empty"
        if stats.record(reason):
            doc["text"] = text
            yield doc


def drop_exact_duplicates(docs: Iterable[dict], seen: set) -> Iterator[dict]:
    """Global exact-match dedupe on a digest of the normalised text."""
    for doc in docs:
        digest = hashlib.blake2b(doc["text"].encode("utf-8"), digest_size=16).digest()
        if digest in seen:
            continue
        seen.add(digest)
        yield doc


def windowed_near_dedupe(docs: Iterable[dict], window: int, threshold: float) -> Iterator[dict]:
    """MinHash dedupe inside fixed-size windows."""
    buffer: list[dict] = []
    for doc in docs:
        buffer.append(doc)
        if len(buffer) >= window:
            yield from _flush(buffer, threshold)
            buffer = []
    if buffer:
        yield from _flush(buffer, threshold)


def _flush(buffer: list[dict], threshold: float) -> Iterator[dict]:
    drop = find_duplicates((d["text"] for d in buffer), threshold=threshold)
    for i, doc in enumerate(buffer):
        if i not in drop:
            yield doc


class ShardWriter:
    """Writes zstd-compressed JSONL shards, rolling over at a size limit."""

    def __init__(self, out_dir: Path, prefix: str = "shard", max_docs: int = 100_000):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.max_docs = max_docs
        self.index = 0
        self.in_shard = 0
        self.total = 0
        self.bytes_written = 0
        self._fh = None
        self._writer = None

    def _open(self) -> None:
        path = self.out_dir / f"{self.prefix}-{self.index:05d}.jsonl.zst"
        self._fh = open(path, "wb")
        self._writer = zstd.ZstdCompressor(level=6).stream_writer(self._fh)

    def write(self, doc: dict) -> None:
        if self._writer is None:
            self._open()
        line = (json.dumps(doc, ensure_ascii=False) + "\n").encode("utf-8")
        self._writer.write(line)
        self.bytes_written += len(line)
        self.in_shard += 1
        self.total += 1
        if self.in_shard >= self.max_docs:
            self.close()
            self.index += 1
            self.in_shard = 0

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._fh.close()
            self._writer = None
            self._fh = None


def run(
    docs: Iterable[dict],
    out_dir: Path,
    prefix: str,
    window: int = 50_000,
    threshold: float = 0.8,
    max_docs_per_shard: int = 100_000,
    shuffle_buffer: int = 10_000,
    seed: int = 0,
) -> dict:
    """Run the full pipeline and return a manifest."""
    t0 = time.perf_counter()
    stats = FilterStats()
    seen_exact: set = set()

    stream = normalise_and_filter(docs, stats)
    n_before_exact = [0]

    def counted(it):
        for d in it:
            n_before_exact[0] += 1
            yield d

    stream = counted(stream)
    stream = drop_exact_duplicates(stream, seen_exact)
    after_exact = [0]

    def counted2(it):
        for d in it:
            after_exact[0] += 1
            yield d

    stream = counted2(stream)
    stream = shuffled(stream, shuffle_buffer, seed)
    stream = windowed_near_dedupe(stream, window, threshold)

    writer = ShardWriter(out_dir, prefix, max_docs_per_shard)
    chars = 0
    per_source: dict[str, int] = {}
    try:
        for doc in stream:
            writer.write(doc)
            chars += len(doc["text"])
            per_source[doc["source"]] = per_source.get(doc["source"], 0) + 1
    finally:
        writer.close()

    return {
        "documents_in": stats.total,
        "passed_filters": stats.kept,
        "dropped_by_filter": dict(stats.dropped),
        "removed_exact_duplicates": n_before_exact[0] - after_exact[0],
        "removed_near_duplicates": after_exact[0] - writer.total,
        "documents_out": writer.total,
        "characters_out": chars,
        "estimated_tokens": chars // 4,     # ~4 chars/token before the real tokenizer exists
        "shards": writer.index + (1 if writer.in_shard else 0),
        "per_source": per_source,
        "seconds": round(time.perf_counter() - t0, 2),
    }


def shuffled(docs: Iterable[dict], buffer_size: int, seed: int) -> Iterator[dict]:
    """Reservoir-style streaming shuffle. Bounded memory, good enough mixing."""
    rng = random.Random(seed)
    buf: list[dict] = []
    for doc in docs:
        if len(buf) < buffer_size:
            buf.append(doc)
            continue
        i = rng.randrange(buffer_size)
        yield buf[i]
        buf[i] = doc
    rng.shuffle(buf)
    yield from buf


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--local", type=Path, help="a .jsonl file or directory of them")
    g.add_argument("--source", help="a named source from data/sources.py")
    g.add_argument("--all", action="store_true", help="every source in the mix")
    p.add_argument("--out", type=Path, default=Path("data/processed"))
    p.add_argument("--limit", type=int, default=None, help="max documents to read")
    p.add_argument("--kind", default="prose", choices=("prose", "code"), help="for --local")
    p.add_argument("--window", type=int, default=50_000, help="near-dedupe window size")
    p.add_argument("--threshold", type=float, default=0.8, help="Jaccard dedupe threshold")
    p.add_argument("--shard-size", type=int, default=100_000, help="documents per shard")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    if args.local:
        prefix = args.local.stem
        docs = stream_local(args.local, prefix, args.kind, args.limit)
    elif args.source:
        src = resolve(args.source)
        prefix = src.name
        docs = stream_hf(src, args.limit)
    else:
        prefix = "mix"

        def chained():
            for src in MIX:
                yield from stream_hf(src, args.limit)

        docs = chained()

    manifest = run(
        docs,
        out_dir=args.out,
        prefix=prefix,
        window=args.window,
        threshold=args.threshold,
        max_docs_per_shard=args.shard_size,
        seed=args.seed,
    )

    manifest_path = args.out / f"{prefix}-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    print(f"\nwrote {manifest['shards']} shard(s) to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
