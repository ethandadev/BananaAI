"""Sidecar protocol: framing, streaming, cancellation, error handling.

Driven through the real serve() loop with a fake stdin, so the framing and
threading are exercised rather than mocked.
"""

from __future__ import annotations

import io
import json
from dataclasses import asdict

import torch

from tests.harness import Suite, temp_dir

from core.config import ModelConfig
from core.model import Transformer
from core.tokenizer import train as train_tokenizer
from core.train import PRESETS
from server.sidecar import PROTOCOL_VERSION, Engine, Sidecar

suite = Suite("sidecar")
test = suite.test

TINY = ModelConfig(**PRESETS["test"], vocab_size=320)


def build(tmp):
    torch.manual_seed(0)
    model = Transformer(TINY)
    ckpt = tmp / "m.pt"
    torch.save({"model": model.state_dict(), "model_config": asdict(TINY),
                "step": 0, "stage": "sft"}, ckpt)

    texts = [f"the model explains concept {i} in a single clear sentence" for i in range(150)]
    tok = train_tokenizer(texts, vocab_size=320, min_frequency=1)
    tok_path = tmp / "tokenizer.json"
    tok.save(str(tok_path))
    return Engine(ckpt, tok_path, device="cpu")


def drive(engine, requests: list[dict]) -> list[dict]:
    """Run serve() against a scripted stdin and collect the emitted events."""
    out = io.StringIO()
    sidecar = Sidecar(engine)
    sidecar.emit = lambda obj: out.write(json.dumps(obj) + "\n")
    stdin = io.StringIO("".join(json.dumps(r) + "\n" for r in requests))
    sidecar.serve(stdin=stdin)
    return [json.loads(l) for l in out.getvalue().splitlines() if l.strip()]


@test
def ready_event_describes_the_model():
    with temp_dir() as tmp:
        events = drive(build(tmp), [])
    assert events and events[0]["type"] == "ready", events[:1]
    ready = events[0]
    assert ready["protocol"] == PROTOCOL_VERSION
    assert ready["parameters"] == Transformer(TINY).num_params()
    assert ready["stage"] == "sft"
    assert ready["context_len"] == TINY.context_len
    return f"ready: {ready['parameters'] / 1e6:.2f}M params, stage {ready['stage']}"


@test
def info_request_is_answered_with_its_id():
    with temp_dir() as tmp:
        events = drive(build(tmp), [{"id": "x1", "type": "info"}])
    info = [e for e in events if e["type"] == "info"]
    assert len(info) == 1, f"expected one info event, got {len(info)}"
    assert info[0]["id"] == "x1", "response did not carry the request id"
    return "id echoed back on the response"


@test
def generation_streams_tokens_then_done():
    with temp_dir() as tmp:
        events = drive(build(tmp), [{
            "id": "g", "type": "generate",
            "messages": [{"role": "user", "content": "explain a gradient"}],
            "params": {"max_new_tokens": 12, "seed": 0},
        }])

    kinds = [e["type"] for e in events if e.get("id") == "g"]
    assert kinds[0] == "start", kinds[:3]
    assert kinds[-1] == "done", kinds[-3:]
    tokens = [e for e in events if e["type"] == "token"]
    done = [e for e in events if e["type"] == "done"][0]
    assert tokens, "no tokens streamed"
    assert done["tokens"] == 12, f"done reports {done['tokens']} tokens, asked for 12"
    assert done["cancelled"] is False
    return f"start -> {len(tokens)} token events -> done"


@test
def streamed_text_reassembles_into_the_final_text():
    """Incremental decoding must not drop or duplicate characters."""
    with temp_dir() as tmp:
        events = drive(build(tmp), [{
            "id": "g", "type": "generate",
            "messages": [{"role": "user", "content": "hello there"}],
            "params": {"max_new_tokens": 20, "seed": 3},
        }])
    joined = "".join(e["text"] for e in events if e["type"] == "token")
    final = [e for e in events if e["type"] == "done"][0]["text"]
    assert joined == final, f"stream reassembled to {joined!r}, done reported {final!r}"
    return f"{len(joined)} characters match the final text exactly"


@test
def seeded_generation_is_reproducible_through_the_protocol():
    request = {
        "id": "g", "type": "generate",
        "messages": [{"role": "user", "content": "what is a token"}],
        "params": {"max_new_tokens": 16, "seed": 7},
    }
    with temp_dir() as tmp:
        engine = build(tmp)
        a = drive(engine, [request])
        b = drive(engine, [request])
    text_a = [e for e in a if e["type"] == "done"][0]["text"]
    text_b = [e for e in b if e["type"] == "done"][0]["text"]
    assert text_a == text_b, "same seed gave different output over the protocol"
    return "identical output across two runs"


@test
def malformed_json_is_reported_not_fatal():
    with temp_dir() as tmp:
        engine = build(tmp)
        out = io.StringIO()
        sidecar = Sidecar(engine)
        sidecar.emit = lambda obj: out.write(json.dumps(obj) + "\n")
        sidecar.serve(stdin=io.StringIO('{"broken\n{"id":"ok","type":"info"}\n'))
        events = [json.loads(l) for l in out.getvalue().splitlines() if l.strip()]

    errors = [e for e in events if e["type"] == "error"]
    assert errors, "a malformed line produced no error event"
    assert any(e["type"] == "info" for e in events), \
        "the loop died instead of continuing to the next line"
    return "bad line reported, the next request still handled"


@test
def unknown_request_types_are_rejected():
    with temp_dir() as tmp:
        events = drive(build(tmp), [{"id": "q", "type": "teleport"}])
    err = [e for e in events if e["type"] == "error"]
    assert err and "teleport" in err[0]["message"], err
    return "unknown type produces a clear error"


@test
def a_failing_generation_reports_an_error_not_a_crash():
    with temp_dir() as tmp:
        events = drive(build(tmp), [{
            "id": "bad", "type": "generate",
            "messages": [{"role": "wizard", "content": "hi"}],   # invalid role
        }])
    err = [e for e in events if e["type"] == "error"]
    assert err, "an invalid role did not produce an error event"
    assert err[0]["id"] == "bad"
    return "invalid role surfaces as an error event"


@test
def shutdown_is_acknowledged():
    with temp_dir() as tmp:
        events = drive(build(tmp), [{"id": "s", "type": "shutdown"}])
    assert any(e["type"] == "bye" for e in events), events
    return "shutdown answered with bye"


@test
def long_prompts_are_truncated_to_the_context_window():
    with temp_dir() as tmp:
        engine = build(tmp)
        ids = engine.encode("word " * 5000)
    assert ids.size(1) <= TINY.context_len - 1, \
        f"prompt of {ids.size(1)} tokens exceeds the window"
    return f"clipped to {ids.size(1)} of {TINY.context_len} tokens"


@test
def stop_tokens_come_from_the_tokenizer():
    with temp_dir() as tmp:
        engine = build(tmp)
    assert engine.stop_tokens, "no stop tokens configured"
    assert engine.tokenizer.token_to_id("<|eos|>") in engine.stop_tokens
    assert engine.tokenizer.token_to_id("<|end|>") in engine.stop_tokens
    return f"stops on {list(engine.stop_tokens)}"


if __name__ == "__main__":
    p, n = suite.run()
    raise SystemExit(0 if p == n else 1)
