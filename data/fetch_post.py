"""Build the SFT and DPO training sets.

    python -m data.fetch_post sft --target 50000 --out data/sft
    python -m data.fetch_post dpo --target 20000 --out data/dpo

Downloads each source, converts it through its adapter, applies quality rules,
dedupes prompts across every source, shuffles, and writes train/val JSONL plus
a manifest saying exactly what was kept and dropped.

Sizing: 40-60k SFT examples and 15-25k preference pairs is the useful range at
327M parameters. More does not help much -- a small model saturates on
instruction format quickly -- and the mix matters far more than the count.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterator, Optional

from .post_sources import (
    DPO_MIX, SFT_MIX, PostSource, conversation_reject_reason, preference_reject_reason,
    prompt_key, resolve, stream_hf_rows, validate,
)


def convert(source: PostSource, rows: Iterator[dict], stats: Counter):
    """Adapt raw rows and apply the quality rules for this kind."""
    is_pref = source in DPO_MIX
    reject = preference_reject_reason if is_pref else conversation_reject_reason

    for row in rows:
        try:
            item = source.adapter(row)
        except (KeyError, TypeError, AttributeError):
            stats["adapter_error"] += 1
            continue
        if item is None:
            stats["unadaptable"] += 1
            continue
        reason = reject(item)
        if reason:
            stats[reason] += 1
            continue
        stats["kept"] += 1
        yield item


def collect(
    mix: tuple[PostSource, ...],
    target: int,
    only: Optional[str],
    read_multiplier: float,
    seed: int,
) -> tuple[list, dict]:
    """Pull from every source in proportion to its share, deduping globally."""
    sources = [resolve(only, mix)] if only else list(mix)
    if only:
        sources[0] = PostSource(**{**sources[0].__dict__, "share": 1.0})

    seen: set[bytes] = set()
    items: list = []
    report: dict = {"per_source": {}, "dropped": {}}

    for source in sources:
        want = int(target * source.share)
        stats: Counter = Counter()
        # Read more rows than needed, because filtering and dedupe remove some.
        budget = int(want * read_multiplier)

        got = 0
        duplicates = 0
        try:
            rows = stream_hf_rows(source, limit=budget)
            for item in convert(source, rows, stats):
                key = prompt_key(item)
                if key in seen:
                    duplicates += 1
                    continue
                seen.add(key)
                items.append(item)
                got += 1
                if got >= want:
                    break
        except Exception as e:                      # noqa: BLE001
            print(f"  [{source.name}] failed: {type(e).__name__}: {e}", file=sys.stderr)
            report["per_source"][source.name] = {"error": f"{type(e).__name__}: {e}"}
            continue

        report["per_source"][source.name] = {
            "requested": want,
            "collected": got,
            "cross_source_duplicates": duplicates,
            "dropped_by_filter": dict(stats),
        }
        short = want - got
        note = f" ({short} short -- raise --read-multiplier)" if short > 0 else ""
        print(f"  [{source.name}] {got}/{want} kept, {duplicates} duplicates{note}",
              file=sys.stderr)

    random.Random(seed).shuffle(items)
    return items, report


def write_split(items: list, out_dir: Path, val_fraction: float, name: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_val = max(1, int(len(items) * val_fraction)) if items else 0
    val, train = items[:n_val], items[n_val:]

    def dump(rows, path):
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                payload = row if isinstance(row, dict) else {"messages": row}
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    dump(train, out_dir / f"{name}.jsonl")
    dump(val, out_dir / f"{name}_val.jsonl")
    return {"train": len(train), "val": len(val)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("kind", choices=("sft", "dpo"))
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--target", type=int, default=None, help="examples to collect")
    p.add_argument("--only", help="a single source, e.g. magicoder")
    p.add_argument("--val-fraction", type=float, default=0.02)
    p.add_argument("--read-multiplier", type=float, default=2.5,
                   help="rows to read per row kept, to cover filtering losses")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    mix = SFT_MIX if args.kind == "sft" else DPO_MIX
    validate(mix)
    target = args.target or (50_000 if args.kind == "sft" else 20_000)
    out_dir = args.out or Path("data") / args.kind
    name = "train" if args.kind == "sft" else "prefs"

    print(f"[{args.kind}] target {target:,} from {len(mix)} sources", file=sys.stderr)
    t0 = time.perf_counter()
    items, report = collect(mix, target, args.only, args.read_multiplier, args.seed)

    if not items:
        print("\nNothing collected. Every source failed -- check the network and "
              "that `datasets` is installed.", file=sys.stderr)
        return 1

    split = write_split(items, out_dir, args.val_fraction, name)
    manifest = {
        "kind": args.kind,
        "total": len(items),
        **split,
        "seconds": round(time.perf_counter() - t0, 1),
        **report,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps({k: v for k, v in manifest.items() if k != "per_source"}, indent=2))
    print(f"\nwrote {split['train']:,} train + {split['val']:,} val to {out_dir}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
