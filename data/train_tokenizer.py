"""Train the BPE tokenizer on a sample of the corpus.

    python -m data.train_tokenizer --shards data/processed --vocab 32768

Sampling matters: train on a mix that reflects the real corpus. A tokenizer
trained only on prose gives terrible code compression, because it never learns
the indentation and punctuation runs that dominate source files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.tokenizer import (
    DEFAULT_VOCAB_SIZE,
    SPECIAL_TOKENS,
    compression_ratio,
    stream_texts_from_shards,
    train,
)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--shards", type=Path, default=Path("data/processed"))
    p.add_argument("--out", type=Path, default=Path("tokenizer.json"))
    p.add_argument("--vocab", type=int, default=DEFAULT_VOCAB_SIZE)
    p.add_argument("--limit", type=int, default=200_000, help="documents to train on")
    p.add_argument("--min-frequency", type=int, default=2)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    texts = list(stream_texts_from_shards(args.shards, args.limit))
    if not texts:
        raise SystemExit(f"no documents found in {args.shards}")

    if not args.quiet:
        chars = sum(len(t) for t in texts)
        print(f"training on {len(texts):,} documents ({chars / 1e6:.1f}M characters)")

    tok = train(
        texts,
        vocab_size=args.vocab,
        out_path=args.out,
        min_frequency=args.min_frequency,
        show_progress=not args.quiet,
    )

    # Held-out slice, so the ratio is not measured on training data.
    sample = texts[: min(500, len(texts))]
    ratio = compression_ratio(tok, sample)
    info = {
        "vocab_size": tok.get_vocab_size(),
        "special_tokens": SPECIAL_TOKENS,
        "documents_trained_on": len(texts),
        "chars_per_token": round(ratio, 3),
    }
    Path(str(args.out) + ".info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
