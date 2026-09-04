"""Training sidecar: protocol, stage sequencing, cancellation, progress events.

Driven through the real serve() loop with a fake stdin, and through a real
(very small) training run, so the threading and the event stream are exercised
rather than mocked.
"""

from __future__ import annotations

import io
import json
import threading
import time

import numpy as np

from tests.harness import Suite, temp_dir

from core.settings import DEFAULTS, Settings
from core.train import PRESETS, train
from core.config import ModelConfig, TrainConfig
from server.trainer import DEFAULT_STAGES, PROTOCOL_VERSION, STAGES, Trainer, TrainerServer

suite = Suite("trainer")
test = suite.test


def settings(**overrides) -> Settings:
    data = {k: dict(v) for k, v in DEFAULTS.items()}
    data["hardware"]["device"] = "cpu"
    data["hardware"]["max_hours"] = 1.0
    for section, values in overrides.items():
        data[section].update(values)
    return Settings(data)


def drive(server: TrainerServer, requests: list[dict]) -> list[dict]:
    out = io.StringIO()
    server.emit = lambda obj: out.write(json.dumps(obj, default=str) + "\n")
    server.trainer.emit = server.emit
    server.serve(stdin=io.StringIO("".join(json.dumps(r) + "\n" for r in requests)))
    return [json.loads(l) for l in out.getvalue().splitlines() if l.strip()]


def make_tokens(tmp, vocab=256, n=40_000):
    rng = np.random.default_rng(0)
    base = rng.integers(0, vocab, size=97, dtype=np.uint16)
    data_dir = tmp / "tok"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, count in (("train.bin", n), ("val.bin", n // 8)):
        (data_dir / name).write_bytes(
            np.tile(base, count // 97 + 1)[:count].astype(np.uint16).tobytes())
    (data_dir / "meta.json").write_text(json.dumps(
        {"vocab_size": vocab, "train_tokens": n, "val_tokens": n // 8,
         "dtype": "uint16", "eos_id": 2}), encoding="utf-8")
    return data_dir


@test
def ready_event_describes_the_plan():
    events = drive(TrainerServer(settings()), [])
    assert events and events[0]["type"] == "ready", events[:1]
    ready = events[0]
    assert ready["protocol"] == PROTOCOL_VERSION
    assert ready["stages"] == list(STAGES)
    assert ready["problems"] == [], ready["problems"]
    assert "preset" in ready and ready["parameters"] > 0
    return f"ready: {ready['preset']}, {ready['parameters'] / 1e6:.0f}M params"


@test
def invalid_config_is_reported_in_ready_not_thrown():
    s = settings()
    s.data["model"]["vocab_size"] = 99999          # too big for uint16 bins
    events = drive(TrainerServer(s), [])
    assert events[0]["problems"], "an invalid config produced no problems"
    assert any("uint16" in p for p in events[0]["problems"])
    return "config problems arrive in the ready event"


@test
def plan_request_is_answered():
    events = drive(TrainerServer(settings()), [{"id": "p1", "type": "plan"}])
    plans = [e for e in events if e["type"] == "plan"]
    assert plans and plans[0]["id"] == "p1", plans
    assert "summary" in plans[0] and plans[0]["estimated_hours"] is not None
    return f"plan: {plans[0]['preset']}, {plans[0]['estimated_hours']:.1f}h"


@test
def status_is_idle_before_anything_starts():
    events = drive(TrainerServer(settings()), [{"id": "s", "type": "status"}])
    status = [e for e in events if e["type"] == "status"][0]
    assert status["running"] is False and status["stage"] is None
    return "idle before a run"


@test
def unknown_stages_are_rejected():
    events = drive(TrainerServer(settings()),
                   [{"type": "start", "stages": ["pretrain", "teleport"]}])
    errors = [e for e in events if e["type"] == "error"]
    assert errors and "teleport" in errors[0]["message"], errors
    return "an unknown stage name fails before any work starts"


@test
def unknown_requests_and_bad_json_are_survivable():
    server = TrainerServer(settings())
    out = io.StringIO()
    server.emit = lambda obj: out.write(json.dumps(obj, default=str) + "\n")
    server.serve(stdin=io.StringIO('{ broken\n{"type":"nonsense"}\n{"id":"s","type":"status"}\n'))
    events = [json.loads(l) for l in out.getvalue().splitlines() if l.strip()]
    assert len([e for e in events if e["type"] == "error"]) == 2, events
    assert any(e["type"] == "status" for e in events), "the loop died"
    return "two errors reported, the loop kept going"


@test
def shutdown_is_acknowledged():
    events = drive(TrainerServer(settings()), [{"type": "shutdown"}])
    assert any(e["type"] == "bye" for e in events), events
    return "shutdown answered with bye"


@test
def a_pretrain_stage_streams_progress_then_finishes():
    with temp_dir() as tmp:
        data_dir = make_tokens(tmp)
        s = settings(
            data={"tokenized_dir": str(data_dir)},
            model={"preset": "nano", "vocab_size": 256, "context_len": 64},
            train={"out_dir": str(tmp / "run")},
            hardware={"device": "cpu", "compile": False},
        )
        server = TrainerServer(s)
        # Keep the run tiny; the plan would otherwise size a real one.
        original = server.trainer.stage_pretrain

        def small():
            plan = s.plan()
            plan.model.context_len = 64
            ctx = PRESETS["test"]["context_len"]
            tc = TrainConfig(total_steps=6, warmup_steps=2, peak_lr=3e-3, min_lr=3e-4,
                             micro_batch=2, tokens_per_step=2 * ctx,
                             eval_every=3, ckpt_every=3, log_every=1,
                             sample_every=10**9)
            return train(ModelConfig(**PRESETS["test"], vocab_size=256), tc,
                         data_dir, tmp / "run", device="cpu", compile_model=False,
                         on_event=server.trainer.emit,
                         should_stop=server.trainer.stop_flag.is_set)

        server.trainer.stage_pretrain = small
        events = drive(server, [{"type": "start", "stages": ["pretrain"]}])

    kinds = [e["type"] for e in events]
    assert "setup" in kinds, "no setup event"
    progress = [e for e in events if e["type"] == "progress"]
    assert progress, "no progress events"
    assert all("eta_seconds" in e and "loss" in e for e in progress)
    assert any(e["type"] == "eval" for e in events), "no eval event"
    assert any(e["type"] == "checkpoint" for e in events), "no checkpoint event"
    finished = [e for e in events if e["type"] == "finished"]
    assert finished and not finished[0]["stopped"], finished
    return f"{len(progress)} progress events, eval, checkpoint, finished"


@test
def progress_events_carry_a_usable_eta():
    with temp_dir() as tmp:
        data_dir = make_tokens(tmp)
        seen = []
        tc = TrainConfig(total_steps=8, warmup_steps=2, peak_lr=3e-3, min_lr=3e-4,
                         micro_batch=2, tokens_per_step=2 * 128,
                         eval_every=10**9, ckpt_every=10**9, log_every=1,
                         sample_every=10**9)
        train(ModelConfig(**PRESETS["test"], vocab_size=256), tc, data_dir,
              tmp / "r", device="cpu", compile_model=False,
              on_event=lambda e: seen.append(e))

    progress = [e for e in seen if e["type"] == "progress"]
    assert len(progress) >= 5, f"only {len(progress)} progress events"
    # ETA must shrink as the run proceeds.
    assert progress[-1]["eta_seconds"] < progress[0]["eta_seconds"], \
        "ETA did not decrease"
    assert progress[-1]["step"] > progress[0]["step"]
    assert all(0 <= e["eta_seconds"] for e in progress)
    return f"ETA {progress[0]['eta_seconds']:.1f}s -> {progress[-1]['eta_seconds']:.1f}s"


@test
def stopping_ends_the_run_cleanly_with_a_checkpoint():
    """A stop must checkpoint, not just die -- otherwise the work is lost."""
    with temp_dir() as tmp:
        data_dir = make_tokens(tmp)
        stop = threading.Event()
        seen = []

        def watch(event):
            seen.append(event)
            if event["type"] == "progress" and event["step"] >= 2:
                stop.set()

        tc = TrainConfig(total_steps=500, warmup_steps=2, peak_lr=1e-3, min_lr=1e-4,
                         micro_batch=2, tokens_per_step=2 * 128,
                         eval_every=10**9, ckpt_every=10**9, log_every=1,
                         sample_every=10**9)
        summary = train(ModelConfig(**PRESETS["test"], vocab_size=256), tc, data_dir,
                        tmp / "r", device="cpu", compile_model=False,
                        on_event=watch, should_stop=stop.is_set)
        checkpoints = list((tmp / "r").glob("step-*.pt"))

    assert summary["steps_completed"] < 500, "stop was ignored"
    assert any(e["type"] == "stopped" for e in seen), "no stopped event"
    assert checkpoints, "stopped without writing a checkpoint"
    return f"stopped after {summary['steps_completed']} of 500 steps, checkpoint written"


@test
def the_loss_watchdog_reports_why_it_aborted():
    with temp_dir() as tmp:
        data_dir = make_tokens(tmp)
        seen = []
        # An enormous learning rate reliably diverges within a few steps.
        tc = TrainConfig(total_steps=60, warmup_steps=1, peak_lr=50.0, min_lr=50.0,
                         micro_batch=2, tokens_per_step=2 * 128,
                         eval_every=10**9, ckpt_every=10**9, log_every=1,
                         sample_every=10**9)
        train(ModelConfig(**PRESETS["test"], vocab_size=256), tc, data_dir,
              tmp / "r", device="cpu", compile_model=False,
              on_event=lambda e: seen.append(e), spike_factor=2.0)

    aborted = [e for e in seen if e["type"] == "aborted"]
    assert aborted, "divergence produced no aborted event"
    assert "resume" in aborted[0]["reason"], aborted[0]
    return f"aborted at step {aborted[0]['step']}: {aborted[0]['reason'][:44]}..."


@test
def a_broken_listener_cannot_kill_the_run():
    """on_event is the app's callback; a bug there must not stop training."""
    with temp_dir() as tmp:
        data_dir = make_tokens(tmp)

        def explode(event):
            raise RuntimeError("listener is broken")

        tc = TrainConfig(total_steps=4, warmup_steps=1, peak_lr=1e-3, min_lr=1e-4,
                         micro_batch=2, tokens_per_step=2 * 128,
                         eval_every=10**9, ckpt_every=10**9, log_every=1,
                         sample_every=10**9)
        summary = train(ModelConfig(**PRESETS["test"], vocab_size=256), tc, data_dir,
                        tmp / "r", device="cpu", compile_model=False, on_event=explode)
    assert summary["steps_completed"] == 4, summary
    return "training completed despite every callback raising"


@test
def a_failing_stage_reports_an_error_with_context():
    with temp_dir() as tmp:
        s = settings(data={"tokenized_dir": str(tmp / "missing")},
                     train={"out_dir": str(tmp / "run")})
        events = drive(TrainerServer(s), [{"type": "start", "stages": ["pretrain"]}])
    errors = [e for e in events if e["type"] == "error"]
    assert errors, "a missing dataset produced no error"
    assert errors[0]["stage"] == "pretrain", errors[0]
    assert "traceback" in errors[0]
    return "the failing stage is named and a traceback attached"


@test
def default_stages_stop_before_post_training():
    assert DEFAULT_STAGES == ("data", "tokenizer", "tokenize", "pretrain"), DEFAULT_STAGES
    assert set(DEFAULT_STAGES) <= set(STAGES)
    return "defaults cover data through pretraining"


if __name__ == "__main__":
    p, n = suite.run()
    raise SystemExit(0 if p == n else 1)
