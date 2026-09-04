"""Byte-level BPE tokenizer: training, loading, and the chat template.

Byte-level rather than SentencePiece for two reasons. It cannot produce an
out-of-vocabulary token for any byte sequence, so no <unk> and no data loss on
odd unicode; and it maps directly onto the GGUF "gpt2" tokenizer model, so
export needs no custom vocabulary conversion.

The chat special tokens are reserved here, during tokenizer training, not
bolted on before SFT. Adding them later means resizing the embedding matrix
and fine-tuning rows the base model has never once seen.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

from tokenizers import Tokenizer, decoders, pre_tokenizers, processors, trainers
from tokenizers.models import BPE

# Order matters: these become ids 0..N-1 and are baked into checkpoints.
SPECIAL_TOKENS = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
]

PAD_ID, BOS_ID, EOS_ID = 0, 1, 2
DEFAULT_VOCAB_SIZE = 32768


def build_tokenizer() -> Tokenizer:
    """An untrained byte-level BPE with the pre-tokenization we want."""
    tok = Tokenizer(BPE(unk_token=None, fuse_unk=False))
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        # Split digits apart so the model sees 1-9-9-8 rather than one "1998"
        # token. Arithmetic is hopeless when numbers tokenize arbitrarily.
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
    ])
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=False)
    return tok


def train(
    texts: Iterable[str],
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    out_path: Optional[Path] = None,
    min_frequency: int = 2,
    show_progress: bool = False,
) -> Tokenizer:
    """Train BPE on an iterator of documents."""
    tok = build_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=show_progress,
    )
    tok.train_from_iterator(texts, trainer=trainer)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tok.save(str(out_path))
    return tok


def load(path: Path = Path("tokenizer.json")) -> Tokenizer:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"no tokenizer at {path} -- train one with: python -m data.train_tokenizer"
        )
    return Tokenizer.from_file(str(path))


def compression_ratio(tok: Tokenizer, texts: Iterable[str]) -> float:
    """Characters per token. Healthy: ~3.8 on prose, ~3.2 on code."""
    chars = tokens = 0
    for t in texts:
        chars += len(t)
        tokens += len(tok.encode(t).ids)
    return chars / tokens if tokens else 0.0


# --------------------------------------------------------------------------
# chat template
# --------------------------------------------------------------------------


@dataclass
class Message:
    role: str      # 'system' | 'user' | 'assistant'
    content: str


ROLE_TOKEN = {"system": "<|system|>", "user": "<|user|>", "assistant": "<|assistant|>"}


def render(messages: list[Message], add_generation_prompt: bool = False) -> str:
    """Render a conversation into the training/inference string format.

        <|bos|><|user|>hello<|end|><|assistant|>hi<|end|><|eos|>

    add_generation_prompt leaves the trailing <|assistant|> open, which is how
    inference asks the model to start replying.
    """
    parts = ["<|bos|>"]
    for m in messages:
        if m.role not in ROLE_TOKEN:
            raise ValueError(f"unknown role: {m.role!r}")
        parts.append(ROLE_TOKEN[m.role] + m.content + "<|end|>")
    if add_generation_prompt:
        parts.append("<|assistant|>")
    return "".join(parts)


def encode_conversation(
    tok: Tokenizer, messages: list[Message], mask_prompt: bool = True
) -> tuple[list[int], list[int]]:
    """Return (input_ids, labels) with prompt positions masked to -100.

    Labels are aligned to inputs, NOT shifted -- the training loop does the
    shift, matching the contract in Transformer.forward.
    """
    ids: list[int] = []
    labels: list[int] = []

    def add(text: str, supervised: bool) -> None:
        chunk = tok.encode(text, add_special_tokens=False).ids
        ids.extend(chunk)
        labels.extend(chunk if supervised or not mask_prompt else [-100] * len(chunk))

    add("<|bos|>", False)
    for m in messages:
        # The role marker is part of the prompt; the content is supervised only
        # for the assistant.
        supervised = m.role == "assistant"
        add(ROLE_TOKEN[m.role], False)
        add(m.content + "<|end|>", supervised)
    add("<|eos|>", False)
    return ids, labels


def stream_texts_from_shards(shard_dir: Path, limit: Optional[int] = None) -> Iterator[str]:
    """Read documents back out of the pipeline's zstd JSONL shards."""
    import zstandard as zstd

    n = 0
    for shard in sorted(Path(shard_dir).glob("*.jsonl.zst")):
        with open(shard, "rb") as fh:
            reader = zstd.ZstdDecompressor().stream_reader(fh)
            for line in reader.read().decode("utf-8").splitlines():
                if not line.strip():
                    continue
                if limit is not None and n >= limit:
                    return
                n += 1
                yield json.loads(line)["text"]
