"""SFT/DPO dataset adapters and quality rules.

Every adapter is a pure function over a dict, so these run against literal
rows in each upstream schema -- no network, no datasets package. The rows here
are shaped exactly like the real ones, which is the point: a schema change
upstream should fail here rather than halfway through a download.
"""

from __future__ import annotations

from collections import Counter

from tests.harness import Suite, temp_dir

from data.fetch_post import convert, write_split
from data.post_sources import (
    DPO_MIX, SFT_MIX, adapt_instruct_pair, adapt_messages, adapt_preference_flat,
    adapt_preference_messages, adapt_sharegpt, conversation_reject_reason,
    preference_reject_reason, prompt_key, resolve, validate,
)

suite = Suite("post-data")
test = suite.test

ANSWER = "A gradient is the vector of partial derivatives of the loss."
LONGER = "Attention lets every position read from every other position in the sequence."


@test
def sharegpt_schema_converts():
    row = {"conversations": [
        {"from": "human", "value": "What is a gradient?"},
        {"from": "gpt", "value": ANSWER},
    ]}
    got = adapt_sharegpt(row)
    assert got == [
        {"role": "user", "content": "What is a gradient?"},
        {"role": "assistant", "content": ANSWER},
    ], got
    return "from/value -> role/content"


@test
def sharegpt_system_turns_are_recognised():
    row = {"conversations": [
        {"from": "system", "value": "You are concise."},
        {"from": "human", "value": "Explain attention."},
        {"from": "gpt", "value": LONGER},
    ]}
    got = adapt_sharegpt(row)
    assert [m["role"] for m in got] == ["system", "user", "assistant"], got
    return "system/human/gpt all mapped"


@test
def unknown_roles_drop_the_row():
    """A role we cannot map means the conversation would be malformed."""
    row = {"conversations": [
        {"from": "wizard", "value": "abracadabra"},
        {"from": "gpt", "value": ANSWER},
    ]}
    assert adapt_sharegpt(row) is None
    return "unmappable role drops the whole row"


@test
def messages_schema_passes_through():
    row = {"messages": [
        {"role": "user", "content": "Explain attention."},
        {"role": "assistant", "content": LONGER},
    ]}
    got = adapt_messages(row)
    assert got == row["messages"], got
    return "role/content preserved"


@test
def instruct_pair_folds_context_into_the_prompt():
    row = {"instruction": "Summarise this.", "input": "Some source text here.",
           "output": "A summary of the source text provided above."}
    got = adapt_instruct_pair(row)
    assert got[0]["content"] == "Summarise this.\n\nSome source text here.", got[0]
    assert got[1]["role"] == "assistant"
    return "instruction + input joined into one user turn"


@test
def instruct_pair_handles_magicoder_keys():
    row = {"problem": "Write a function that reverses a list.",
           "solution": "def reverse(xs):\n    return xs[::-1]\n"}
    got = adapt_instruct_pair(row)
    assert got and got[0]["role"] == "user" and "reverse" in got[1]["content"]
    return "problem/solution keys recognised"


@test
def adapters_return_none_on_empty_fields():
    for fn, row in (
        (adapt_sharegpt, {"conversations": []}),
        (adapt_messages, {"messages": [{"role": "user", "content": "   "}]}),
        (adapt_instruct_pair, {"instruction": "hi", "output": ""}),
        (adapt_preference_messages, {"chosen": "a string", "rejected": []}),
        (adapt_preference_flat, {"prompt": "", "chosen": "x", "rejected": "y"}),
    ):
        assert fn(row) is None, f"{fn.__name__} accepted an empty row"
    return "5 adapters reject empty input"


@test
def preference_messages_take_the_last_assistant_turn():
    row = {
        "prompt": "Explain attention.",
        "chosen": [{"role": "user", "content": "Explain attention."},
                   {"role": "assistant", "content": LONGER}],
        "rejected": [{"role": "user", "content": "Explain attention."},
                     {"role": "assistant", "content": "It is complicated to explain."}],
    }
    got = adapt_preference_messages(row)
    assert got["chosen"] == LONGER and got["prompt"] == "Explain attention."
    assert got["rejected"].startswith("It is complicated")
    return "prompt plus both assistant responses extracted"


@test
def preference_prompt_falls_back_to_the_user_turn():
    """UltraFeedback rows do not always carry a separate prompt field."""
    row = {
        "chosen": [{"role": "user", "content": "Explain attention."},
                   {"role": "assistant", "content": LONGER}],
        "rejected": [{"role": "user", "content": "Explain attention."},
                     {"role": "assistant", "content": "It is complicated to explain."}],
    }
    got = adapt_preference_messages(row)
    assert got["prompt"] == "Explain attention.", got
    return "prompt recovered from the conversation"


@test
def conversation_must_end_with_the_assistant():
    convs = [{"role": "user", "content": "Explain attention."},
             {"role": "assistant", "content": LONGER},
             {"role": "user", "content": "And what about RoPE?"}]
    assert conversation_reject_reason(convs) == "does_not_end_with_assistant"
    return "a trailing user turn has nothing to supervise"


@test
def roles_must_alternate():
    convs = [{"role": "user", "content": "Explain attention."},
             {"role": "user", "content": "I mean self-attention."},
             {"role": "assistant", "content": LONGER}]
    assert conversation_reject_reason(convs) == "roles_do_not_alternate"
    return "doubled user turn rejected"


@test
def system_turn_must_come_first():
    convs = [{"role": "user", "content": "Explain attention."},
             {"role": "system", "content": "Be concise."},
             {"role": "assistant", "content": LONGER}]
    assert conversation_reject_reason(convs) == "system_turn_out_of_place"
    return "a mid-conversation system turn is rejected"


@test
def refusals_and_meta_commentary_are_dropped():
    for bad in ("As an AI language model, I do not have opinions on this topic.",
                "I'm sorry, but I cannot help with that particular request."):
        convs = [{"role": "user", "content": "Explain attention."},
                 {"role": "assistant", "content": bad}]
        assert conversation_reject_reason(convs) == "refusal_or_meta", bad
    return "refusals and 'as an AI' openers filtered"


@test
def a_good_conversation_survives():
    convs = [{"role": "system", "content": "You are concise."},
             {"role": "user", "content": "Explain attention."},
             {"role": "assistant", "content": LONGER}]
    assert conversation_reject_reason(convs) is None
    return "system + user + assistant kept"


@test
def identical_preference_responses_are_dropped():
    """Chosen == rejected gives an exactly zero gradient."""
    pair = {"prompt": "Explain attention please.", "chosen": LONGER, "rejected": LONGER}
    assert preference_reject_reason(pair) == "identical_responses"
    return "zero-gradient pair rejected"


@test
def a_good_preference_pair_survives():
    pair = {"prompt": "Explain attention please.", "chosen": LONGER,
            "rejected": "It is quite complicated to explain briefly."}
    assert preference_reject_reason(pair) is None
    return "distinct responses kept"


@test
def prompt_key_is_stable_across_formatting():
    a = [{"role": "user", "content": "Explain  Attention."},
         {"role": "assistant", "content": LONGER}]
    b = [{"role": "user", "content": "explain attention."},
         {"role": "assistant", "content": "A different answer entirely here."}]
    assert prompt_key(a) == prompt_key(b), "case and whitespace changed the key"

    c = {"prompt": "Explain attention.", "chosen": LONGER, "rejected": "no"}
    assert prompt_key(c) == prompt_key(a), "dict and list forms disagree"
    return "case, spacing, and both shapes give one key"


@test
def convert_counts_every_rejection():
    source = resolve("openhermes", SFT_MIX)
    rows = [
        {"conversations": [{"from": "human", "value": "Explain attention."},
                           {"from": "gpt", "value": LONGER}]},                  # kept
        {"conversations": []},                                                  # unadaptable
        {"conversations": [{"from": "human", "value": "Explain attention."},
                           {"from": "gpt", "value": "ok"}]},                    # too short
        {"nothing": "useful"},                                                  # unadaptable
    ]
    stats: Counter = Counter()
    out = list(convert(source, iter(rows), stats))
    assert len(out) == 1, f"kept {len(out)}, expected 1"
    assert stats["kept"] == 1 and stats["unadaptable"] == 2
    assert stats["answer_too_short"] == 1, dict(stats)
    return f"1 kept, {sum(stats.values()) - 1} dropped with reasons"


@test
def convert_dispatches_to_the_preference_rules():
    source = resolve("ultrafeedback", DPO_MIX)
    rows = [{
        "prompt": "Explain attention.",
        "chosen": [{"role": "assistant", "content": LONGER}],
        "rejected": [{"role": "assistant", "content": "Complicated to explain briefly."}],
    }]
    stats: Counter = Counter()
    out = list(convert(source, iter(rows), stats))
    assert len(out) == 1 and "prompt" in out[0], out
    return "DPO sources use the preference ruleset"


@test
def write_split_produces_valid_jsonl():
    import json

    items = [[{"role": "user", "content": f"question {i}"},
              {"role": "assistant", "content": f"answer {i} which is long enough"}]
             for i in range(50)]
    with temp_dir() as tmp:
        counts = write_split(items, tmp, val_fraction=0.1, name="train")
        assert counts == {"train": 45, "val": 5}, counts
        lines = (tmp / "train.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 45
        first = json.loads(lines[0])
        assert "messages" in first and first["messages"][0]["role"] == "user"
        assert not set(l for l in lines) & set(
            (tmp / "train_val.jsonl").read_text(encoding="utf-8").splitlines()), \
            "train and val overlap"
    return "45/5 split, valid JSONL, no overlap"


@test
def preference_pairs_write_without_a_messages_wrapper():
    import json

    pairs = [{"prompt": f"q{i}", "chosen": "a good answer", "rejected": "a poor one"}
             for i in range(20)]
    with temp_dir() as tmp:
        write_split(pairs, tmp, val_fraction=0.1, name="prefs")
        row = json.loads((tmp / "prefs.jsonl").read_text(encoding="utf-8").splitlines()[0])
        assert set(row) == {"prompt", "chosen", "rejected"}, row
    return "DPO rows keep their flat shape"


@test
def both_mixes_are_well_formed():
    validate(SFT_MIX)
    validate(DPO_MIX)
    code = [s for s in SFT_MIX if s.name == "magicoder"]
    assert code and abs(code[0].share - 0.30) < 1e-9, "code share changed"
    assert resolve("ultrafeedback", DPO_MIX).split == "train_prefs", \
        "ultrafeedback needs the train_prefs split, not train"
    return f"{len(SFT_MIX)} SFT + {len(DPO_MIX)} DPO sources, shares sum to 1"


if __name__ == "__main__":
    p, n = suite.run()
    raise SystemExit(0 if p == n else 1)
