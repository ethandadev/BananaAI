"""Turn corpus shards into flat uint16 token bins for training.

    python -m data.tokenize_corpus --shards data/processed --out data/tokenized

The training set is one long concatenated stream of tokens, each document
terminated by <|eos|>, with no padding. The dataloader slices random windows
out of it. uint16 because a 32768-token vocabulary fits in 16 bits -- half the
disk and half the page cache of uint32, which matters when the dataloader is
doing nothing but random reads.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np

from core.tokenizer import EOS_ID, load, stream_texts_from_shards


def encode_stream(tok, texts: Iterable[str], eos_id: int = EOS_ID):
    """Yield numpy arrays of token ids, one per document, with EOS appended."""
    for text in texts:
        ids = tok.encode(text, add_special_tokens=False).ids
        if not ids:
            continue
        ids.append(eos_id)
        yield np.asarray(ids, dtype=np.uint16)


def write_bin(path: Path, chunks: Iterable[np.ndarray]) -> int:
    """Append token arrays to a flat binary file. Returns the token count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(path, "wb") as fh:
        for chunk in chunks:
            fh.write(chunk.tobytes())
            total += chunk.size
    return total


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--shards", type=Path, default=Path("data/processed"))
    p.add_argument("--tokenizer", type=Path, default=Path("tokenizer.json"))
    p.add_argument("--out", type=Path, default=Path("data/tokenized"))
    p.add_argument("--val-tokens", type=int, default=10_000_000,
                   help="tokens held out for validation")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)

    tok = load(args.tokenizer)
    vocab = tok.get_vocab_size()
    if vocab > 65535:
        raise SystemExit(f"vocab {vocab} exceeds uint16 -- widen the bin dtype")

    t0 = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=True)

    texts = stream_texts_from_shards(args.shards, args.limit)
    stream = encode_stream(tok, texts)

    # The validation split is taken from the head of the stream, before any
    # training token is written, so the two never overlap.
    val_chunks, val_count = [], 0
    for chunk in stream:
        val_chunks.append(chunk)
        val_count += chunk.size
        if val_count >= args.val_tokens:
            break

    n_val = write_bin(args.out / "val.bin", val_chunks)
    n_train = write_bin(args.out / "train.bin", stream)

    if n_train == 0:
        raise SystemExit(
            "no training tokens left -- --val-tokens is larger than the corpus"
        )

    meta = {
        "vocab_size": vocab,
        "train_tokens": n_train,
        "val_tokens": n_val,
        "dtype": "uint16",
        "eos_id": EOS_ID,
        "seconds": round(time.perf_counter() - t0, 2),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
