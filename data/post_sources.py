"""Instruction and preference dataset sources, with per-dataset adapters.

Every public SFT dataset uses a different schema -- OpenHermes nests turns
under "conversations" with from/value keys, Tulu already uses messages with
role/content, Magicoder is a flat problem/solution pair. So each source
declares an adapter that converts one raw row into our format, and everything
downstream sees the same shape.

Adapters are pure functions over a dict. That is deliberate: it means the
whole conversion layer is testable with a literal row, without a network
connection or the datasets package.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

# --------------------------------------------------------------------------
# adapters -- raw row in, our schema out (or None to drop the row)
# --------------------------------------------------------------------------

ROLE_ALIASES = {
    "human": "user", "user": "user", "prompter": "user",
    "gpt": "assistant", "assistant": "assistant", "chatgpt": "assistant", "bot": "assistant",
    "system": "system",
}


def _clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def adapt_sharegpt(row: dict) -> Optional[list[dict]]:
    """OpenHermes-2.5 and friends: {"conversations": [{"from", "value"}, ...]}."""
    turns = row.get("conversations") or row.get("conversation")
    if not turns:
        return None
    out = []
    for turn in turns:
        role = ROLE_ALIASES.get(str(turn.get("from", "")).lower())
        content = _clean(turn.get("value", ""))
        if role is None or not content:
            return None
        out.append({"role": role, "content": content})
    return out or None


def adapt_messages(row: dict) -> Optional[list[dict]]:
    """Tulu and most modern sets: already {"messages": [{"role", "content"}]}."""
    turns = row.get("messages")
    if not turns:
        return None
    out = []
    for turn in turns:
        role = ROLE_ALIASES.get(str(turn.get("role", "")).lower())
        content = _clean(turn.get("content", ""))
        if role is None or not content:
            return None
        out.append({"role": role, "content": content})
    return out or None


def adapt_instruct_pair(row: dict) -> Optional[list[dict]]:
    """Flat single-turn sets: a problem/instruction field and a solution field."""
    prompt = _clean(
        row.get("problem") or row.get("instruction") or row.get("question") or ""
    )
    answer = _clean(
        row.get("solution") or row.get("output") or row.get("response") or row.get("answer") or ""
    )
    if not prompt or not answer:
        return None
    context = _clean(row.get("input") or "")
    if context:
        prompt = f"{prompt}\n\n{context}"
    return [{"role": "user", "content": prompt},
            {"role": "assistant", "content": answer}]


def adapt_preference_messages(row: dict) -> Optional[dict]:
    """UltraFeedback-binarized: chosen/rejected are each a message list."""
    chosen, rejected = row.get("chosen"), row.get("rejected")
    if not isinstance(chosen, list) or not isinstance(rejected, list):
        return None

    def last_assistant(turns):
        for turn in reversed(turns):
            if str(turn.get("role", "")).lower() == "assistant":
                return _clean(turn.get("content", ""))
        return ""

    prompt = _clean(row.get("prompt", ""))
    if not prompt:
        for turn in chosen:
            if str(turn.get("role", "")).lower() == "user":
                prompt = _clean(turn.get("content", ""))
                break

    c, r = last_assistant(chosen), last_assistant(rejected)
    if not (prompt and c and r):
        return None
    return {"prompt": prompt, "chosen": c, "rejected": r}


def adapt_preference_flat(row: dict) -> Optional[dict]:
    """Orca-DPO-pairs style: plain strings, sometimes under different keys."""
    prompt = _clean(row.get("prompt") or row.get("question") or row.get("input") or "")
    chosen = _clean(row.get("chosen") or row.get("response_a") or "")
    rejected = _clean(row.get("rejected") or row.get("response_b") or "")
    if not (prompt and chosen and rejected):
        return None
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PostSource:
    name: str
    hf_path: str
    adapter: Callable
    share: float
    hf_config: Optional[str] = None
    split: str = "train"
    note: str = ""


SFT_MIX: tuple[PostSource, ...] = (
    PostSource("openhermes", "teknium/OpenHermes-2.5", adapt_sharegpt, 0.45,
               note="broad general-purpose instruction following"),
    PostSource("tulu", "allenai/tulu-3-sft-mixture", adapt_messages, 0.25,
               note="multi-turn dialogue and reasoning"),
    PostSource("magicoder", "ise-uiuc/Magicoder-OSS-Instruct-75K", adapt_instruct_pair, 0.30,
               note="code instructions -- matches the 25% code share of pretraining"),
)

DPO_MIX: tuple[PostSource, ...] = (
    PostSource("ultrafeedback", "HuggingFaceH4/ultrafeedback_binarized",
               adapt_preference_messages, 0.75, split="train_prefs",
               note="general helpfulness preferences"),
    PostSource("orca-dpo", "argilla/distilabel-intel-orca-dpo-pairs",
               adapt_preference_flat, 0.25,
               note="reasoning-heavy preference pairs"),
)


def resolve(name: str, mix: tuple[PostSource, ...]) -> PostSource:
    for s in mix:
        if s.name == name:
            return s
    known = ", ".join(s.name for s in mix)
    raise KeyError(f"unknown source {name!r} -- known: {known}")


def validate(mix: tuple[PostSource, ...]) -> None:
    total = sum(s.share for s in mix)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"shares sum to {total}, not 1.0")


# --------------------------------------------------------------------------
# quality rules
# --------------------------------------------------------------------------

# Assistant turns that begin like this are refusals or meta-commentary about
# being a language model. Training on them teaches the model to refuse and to
# talk about itself, neither of which is the point of an SFT set.
BAD_PREFIXES = (
    "i'm sorry, but i can",
    "i cannot fulfill",
    "i can't fulfill",
    "as an ai language model",
    "as an ai assistant, i",
    "i'm just an ai",
)


def conversation_reject_reason(
    messages: list[dict], min_answer_chars: int = 20, max_chars: int = 24_000
) -> Optional[str]:
    if not messages:
        return "empty"
    if messages[-1]["role"] != "assistant":
        return "does_not_end_with_assistant"

    roles = [m["role"] for m in messages]
    if "user" not in roles:
        return "no_user_turn"
    if roles.count("system") > 1:
        return "multiple_system_turns"
    if any(m["role"] == "system" for m in messages[1:]):
        return "system_turn_out_of_place"

    # Roles must alternate after any leading system turn; a doubled assistant
    # turn means the source was flattened wrongly.
    body = [r for r in roles if r != "system"]
    for a, b in zip(body, body[1:]):
        if a == b:
            return "roles_do_not_alternate"

    total = sum(len(m["content"]) for m in messages)
    if total > max_chars:
        return "too_long"

    for m in messages:
        if m["role"] != "assistant":
            continue
        if len(m["content"]) < min_answer_chars:
            return "answer_too_short"
        head = m["content"][:40].lower()
        if any(head.startswith(p) for p in BAD_PREFIXES):
            return "refusal_or_meta"
    return None


def preference_reject_reason(
    pair: dict,
    min_chars: int = 20,
    min_prompt_chars: int = 10,
    max_chars: int = 12_000,
) -> Optional[str]:
    """Responses carry a higher length floor than prompts.

    A one-line prompt is normal and useful ("What is 2+2?"), so holding it to
    the same minimum as a response would throw away good pairs. A one-line
    *response*, by contrast, usually means a truncated or degenerate row.
    """
    if not pair:
        return "empty"
    if pair["chosen"].strip() == pair["rejected"].strip():
        return "identical_responses"      # zero gradient, pure waste
    if len(pair["prompt"]) < min_prompt_chars:
        return "prompt_too_short"
    for key in ("chosen", "rejected"):
        if len(pair[key]) < min_chars:
            return f"{key}_too_short"
    if len(pair["prompt"]) + max(len(pair["chosen"]), len(pair["rejected"])) > max_chars:
        return "too_long"
    head = pair["chosen"][:40].lower()
    if any(head.startswith(p) for p in BAD_PREFIXES):
        return "chosen_is_a_refusal"
    return None


def prompt_key(item) -> bytes:
    """Digest of the first user turn, for cross-source dedupe.

    These datasets overlap heavily -- OpenHermes and Tulu share upstream
    sources -- so the same prompt genuinely does appear in several of them.
    """
    if isinstance(item, dict):
        text = item.get("prompt", "")
    else:
        text = next((m["content"] for m in item if m["role"] == "user"), "")
    normalised = re.sub(r"\s+", " ", text.lower()).strip()
    return hashlib.blake2b(normalised.encode("utf-8"), digest_size=16).digest()


def stream_hf_rows(source: PostSource, limit: Optional[int] = None) -> Iterator[dict]:
    """Stream raw rows. Imported lazily so tests need no network."""
    from datasets import load_dataset

    ds = load_dataset(source.hf_path, source.hf_config, split=source.split, streaming=True)
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            return
        yield row
