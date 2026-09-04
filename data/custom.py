"""Ingest a folder of your own documents.

    python -m data.custom --dir ~/notes --out data/processed

Walks a directory, reads what it recognises, and emits the same document shape
the public-corpus pipeline produces, so everything downstream is unchanged.

A personal folder is almost always far too small to pretrain on -- a few
hundred megabytes of text is a rounding error against 7B tokens, and training
on it alone produces a model that has memorised your files and learned no
language. So this reports how far the folder goes and recommends how to use
it: blended into a public corpus at a small share, or held back for
fine-tuning, which is the better use of a small, high-value dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator, Optional

from .filters import normalise, reject_reason

# Extension -> how to read it. Anything not listed is skipped silently; a
# documents folder is full of images and binaries and that is not an error.
TEXT_SUFFIXES = {
    ".txt", ".md", ".markdown", ".rst", ".org", ".tex", ".csv", ".tsv", ".log",
}
CODE_SUFFIXES = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".rs", ".go", ".c", ".h", ".cpp", ".hpp",
    ".java", ".kt", ".rb", ".php", ".cs", ".swift", ".sh", ".bash", ".sql",
    ".html", ".css", ".scss", ".yaml", ".yml", ".toml", ".json", ".xml",
}
JSONL_SUFFIXES = {".jsonl", ".ndjson"}
PDF_SUFFIXES = {".pdf"}

SKIP_DIRS = {
    ".git", ".svn", ".hg", "node_modules", "__pycache__", ".venv", "venv",
    ".tox", "dist", "build", ".next", "target", ".idea", ".vscode",
}

MAX_FILE_BYTES = 20 * 1024 * 1024        # a 20 MB text file is a log, not a document


def classify(path: Path) -> Optional[str]:
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return "prose"
    if suffix in CODE_SUFFIXES:
        return "code"
    if suffix in JSONL_SUFFIXES:
        return "jsonl"
    if suffix in PDF_SUFFIXES:
        return "pdf"
    return None


def read_pdf(path: Path) -> Optional[str]:
    """Extract text if a PDF library is installed; otherwise say so once."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return None
    try:
        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:                              # noqa: BLE001 -- corrupt PDFs are common
        return ""


def read_text(path: Path) -> Optional[str]:
    """Read as UTF-8, tolerating the odd bad byte. None means it is binary."""
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    # A NUL in the first block is the usual binary tell.
    if b"\x00" in raw[:8192]:
        return None
    return raw.decode("utf-8", errors="replace")


def walk(root: Path, follow_symlinks: bool = False) -> Iterator[Path]:
    """Yield candidate files, skipping version control and dependency folders."""
    root = Path(root)
    stack = [root]
    seen: set = set()
    while stack:
        directory = stack.pop()
        try:
            entries = list(directory.iterdir())
        except (PermissionError, OSError):
            continue
        for entry in sorted(entries):
            if entry.is_symlink() and not follow_symlinks:
                continue
            if entry.is_dir():
                if entry.name in SKIP_DIRS or entry.name.startswith("."):
                    continue
                # Guard against symlink loops when following is enabled.
                key = entry.resolve()
                if key in seen:
                    continue
                seen.add(key)
                stack.append(entry)
            elif entry.is_file():
                yield entry


def documents(
    root: Path,
    source_name: str = "custom",
    apply_filters: bool = True,
    stats: Optional[dict] = None,
) -> Iterator[dict]:
    """Yield {'text', 'source', 'kind'} for everything readable under `root`."""
    stats = stats if stats is not None else {}

    def bump(key: str, n: int = 1) -> None:
        stats[key] = stats.get(key, 0) + n

    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"no such folder: {root}")

    for path in walk(root):
        kind = classify(path)
        if kind is None:
            bump("skipped_unsupported")
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                bump("skipped_too_large")
                continue
        except OSError:
            continue

        if kind == "jsonl":
            yield from _from_jsonl(path, source_name, apply_filters, bump)
            continue

        if kind == "pdf":
            text = read_pdf(path)
            if text is None:
                bump("skipped_pdf_no_reader")
                continue
            kind = "prose"
        else:
            text = read_text(path)
            if text is None:
                bump("skipped_binary")
                continue

        text = normalise(text)
        if not text:
            bump("skipped_empty")
            continue
        if apply_filters:
            reason = reject_reason(text, kind)
            if reason:
                bump(f"filtered_{reason}")
                continue
        bump("kept")
        bump("characters", len(text))
        yield {"text": text, "source": source_name, "kind": kind,
               "path": str(path.relative_to(root))}


def _from_jsonl(path: Path, source_name: str, apply_filters: bool, bump) -> Iterator[dict]:
    raw = read_text(path)
    if raw is None:
        bump("skipped_binary")
        return
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            bump("skipped_bad_json")
            continue
        text = row.get("text") or row.get("content") or ""
        if not isinstance(text, str):
            bump("skipped_bad_json")
            continue
        kind = row.get("kind", "prose")
        if kind not in ("prose", "code"):
            kind = "prose"
        text = normalise(text)
        if not text:
            bump("skipped_empty")
            continue
        if apply_filters and reject_reason(text, kind):
            bump("filtered_jsonl_row")
            continue
        bump("kept")
        bump("characters", len(text))
        yield {"text": text, "source": source_name, "kind": kind,
               "path": str(path.name)}


# --------------------------------------------------------------------------
# advice
# --------------------------------------------------------------------------

CHARS_PER_TOKEN = 3.8       # measured on mixed prose and code


def assess(characters: int, model_params: int) -> dict:
    """How far this folder goes, and what to do about it.

    The honest answer for almost every personal folder is 'not far'. Saying so
    with numbers is more useful than letting someone discover it after a
    two-day training run.
    """
    tokens = int(characters / CHARS_PER_TOKEN)
    wanted = model_params * 21          # the Chinchilla-ish budget
    share = tokens / wanted if wanted else 0.0

    if share >= 1.0:
        verdict = "enough to pretrain on by itself"
        advice = "train directly on this corpus"
    elif share >= 0.10:
        verdict = f"covers {share * 100:.0f}% of what this model size wants"
        advice = ("blend with a public corpus -- set data.custom_share to about "
                  f"{min(0.5, share):.2f} in bananaai.toml")
    elif tokens >= 100_000:
        verdict = f"covers {share * 100:.1f}% of what this model size wants"
        advice = ("too small to pretrain on: pretrain on public data, then "
                  "fine-tune on this folder with post/sft.py")
    else:
        verdict = f"only {tokens:,} tokens"
        advice = ("far too small for training; useful as evaluation prompts or "
                  "a handful of fine-tuning examples")

    return {
        "characters": characters,
        "estimated_tokens": tokens,
        "tokens_wanted": wanted,
        "share_of_budget": round(share, 4),
        "verdict": verdict,
        "advice": advice,
    }


def main(argv=None) -> int:
    from .prepare import run

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dir", type=Path, required=True, help="folder to ingest")
    p.add_argument("--out", type=Path, default=Path("data/processed"))
    p.add_argument("--name", default="custom", help="source name in the manifest")
    p.add_argument("--no-filters", action="store_true",
                   help="keep everything readable, skipping the quality rules")
    p.add_argument("--params", type=int, default=326_685_696,
                   help="model size the advice is relative to")
    p.add_argument("--dry-run", action="store_true", help="report without writing")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    stats: dict = {}
    docs = documents(args.dir, args.name, not args.no_filters, stats)

    if args.dry_run:
        for _ in docs:
            pass
        manifest = {"documents_out": stats.get("kept", 0)}
    else:
        manifest = run(docs, args.out, args.name, window=10_000)

    advice = assess(stats.get("characters", 0), args.params)
    report = {"folder": str(args.dir), "scan": stats, "corpus": manifest, "assessment": advice}

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"\n{args.dir}")
        print(f"  kept          {stats.get('kept', 0):,} documents "
              f"({stats.get('characters', 0) / 1e6:.1f}M characters)")
        for key in sorted(k for k in stats if k.startswith(("skipped_", "filtered_"))):
            print(f"  {key:<28} {stats[key]:,}")
        print(f"\n  {advice['verdict']}")
        print(f"  {advice['advice']}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
