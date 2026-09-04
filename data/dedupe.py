"""Near-duplicate detection with MinHash + LSH.

Written out rather than pulled from a library, because dedupe is where most of
the real quality in a pretraining corpus comes from and it is worth being able
to see exactly what it does.

The idea: estimate Jaccard similarity between documents cheaply. Hash every
shingle (word 5-gram) of a document, apply P random permutations, and keep the
minimum under each -- the chance two documents share a given minimum equals
their Jaccard similarity. Comparing every pair is quadratic, so LSH splits the
P-value signature into bands and only compares documents that match exactly on
some band.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Iterable, Iterator

import numpy as np

MERSENNE_61 = (1 << 61) - 1
SHINGLE_SIZE = 5


def shingle_hashes(text: str, k: int = SHINGLE_SIZE) -> np.ndarray:
    """Hash the word k-grams of a document to 32-bit values held in uint64.

    blake2b rather than the builtin hash(): Python randomises string hashing
    per process, which would make dedupe results irreproducible across runs.

    32 bits, not 64, so that the permutation a*x + b below stays under 2**64
    and can run in native numpy integers. At 32 bits a document would need
    around 65k distinct shingles before collisions become likely, and a
    collision only ever nudges one of 128 signature values.
    """
    words = text.lower().split()
    if len(words) < k:
        grams = [" ".join(words)] if words else []
    else:
        grams = [" ".join(words[i : i + k]) for i in range(len(words) - k + 1)]

    if not grams:
        return np.empty(0, dtype=np.uint64)

    seen = {}
    for g in grams:
        if g not in seen:
            digest = hashlib.blake2b(g.encode("utf-8"), digest_size=4).digest()
            seen[g] = int.from_bytes(digest, "big")
    return np.fromiter(seen.values(), dtype=np.uint64, count=len(seen))


class MinHasher:
    """Fixed set of random permutations, shared across all documents."""

    def __init__(self, num_perm: int = 128, seed: int = 0):
        self.num_perm = num_perm
        rng = np.random.default_rng(seed)
        # h(x) = (a*x + b) mod (2^61 - 1). Keeping a, b and x all under 2**32
        # bounds a*x + b by 2**64 - 2**32, so uint64 never wraps -- which is
        # what lets this run vectorised instead of in Python objects.
        self.a = rng.integers(1, 1 << 32, size=num_perm).astype(np.uint64)
        self.b = rng.integers(0, 1 << 32, size=num_perm).astype(np.uint64)

    def signature(self, hashes: np.ndarray) -> np.ndarray:
        """Reduce a document's shingle hashes to a num_perm-length signature."""
        if hashes.size == 0:
            return np.full(self.num_perm, MERSENNE_61, dtype=np.uint64)
        # (n_shingles, num_perm) via broadcasting, then min down the shingles.
        permuted = (self.a[None, :] * hashes[:, None] + self.b[None, :]) % MERSENNE_61
        return permuted.min(axis=0)


def estimate_threshold(bands: int, rows: int) -> float:
    """Similarity at which a pair has a 50% chance of becoming a candidate."""
    return (1.0 / bands) ** (1.0 / rows)


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            # Keep the lower index as the representative, so the survivor of a
            # duplicate cluster is the document that appeared first.
            lo, hi = (rx, ry) if rx < ry else (ry, rx)
            self.parent[hi] = lo


def jaccard(a: str, b: str, k: int = SHINGLE_SIZE) -> float:
    """Exact Jaccard over shingles -- for tests and for verifying candidates."""
    sa, sb = set(shingle_hashes(a, k).tolist()), set(shingle_hashes(b, k).tolist())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def find_duplicates(
    texts: Iterable[str],
    num_perm: int = 128,
    bands: int = 16,
    verify: bool = True,
    threshold: float = 0.8,
) -> set[int]:
    """Return the indices to DROP, keeping the first member of each cluster.

    With verify=True, LSH candidates are confirmed with an exact Jaccard
    computation. LSH is tuned for recall and does produce false positives; at
    corpus scale that means throwing away good documents for no reason.
    """
    texts = list(texts)
    if not texts:
        return set()
    if num_perm % bands:
        raise ValueError("num_perm must be divisible by bands")
    rows = num_perm // bands

    hasher = MinHasher(num_perm)
    signatures = [hasher.signature(shingle_hashes(t)) for t in texts]

    buckets: dict[tuple, list[int]] = defaultdict(list)
    for idx, sig in enumerate(signatures):
        for band in range(bands):
            key = (band, sig[band * rows : (band + 1) * rows].tobytes())
            buckets[key].append(idx)

    uf = UnionFind(len(texts))
    checked: set[tuple[int, int]] = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        anchor = members[0]
        for other in members[1:]:
            pair = (anchor, other) if anchor < other else (other, anchor)
            if pair in checked:
                continue
            checked.add(pair)
            if verify and jaccard(texts[pair[0]], texts[pair[1]]) < threshold:
                continue
            uf.union(anchor, other)

    return {i for i in range(len(texts)) if uf.find(i) != i}


def dedupe_stream(docs: Iterable[dict], text_key: str = "text", **kwargs) -> Iterator[dict]:
    """Convenience wrapper over find_duplicates for a list of document dicts."""
    docs = list(docs)
    drop = find_duplicates((d[text_key] for d in docs), **kwargs)
    for i, doc in enumerate(docs):
        if i not in drop:
            yield doc
