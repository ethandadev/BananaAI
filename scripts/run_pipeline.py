"""Run the whole pipeline from the command line.

    python scripts/run_pipeline.py smoke    # ~15 min, small model, proves the wiring
    python scripts/run_pipeline.py full     # the real run, sized to your machine
    python scripts/run_pipeline.py full --stages pretrain,sft,dpo,export

Same stages the desktop app runs, driven through the same code, so the two
cannot drift apart. Python rather than shell so there is one runbook for
Linux, macOS and Windows.

Run the smoke pass first. A crash at hour 40 of a run you never rehearsed is
the most expensive mistake available here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.settings import DEFAULTS, Settings, load as load_settings  # noqa: E402
from server.trainer import DEFAULT_STAGES, STAGES, Trainer          # noqa: E402

# The smoke pass exists to prove the wiring, not to produce a usable model.
SMOKE_OVERRIDES = {
    "hardware": {"max_hours": 0.25},
    "model": {"preset": "nano", "vocab_size": 8192},
    "data": {"limit_per_source": 20_000, "val_tokens": 200_000},
    "sft": {"target": 2_000, "epochs": 1},
    "dpo": {"target": 800},
    "train": {"total_steps": 60, "micro_batch": 4, "tokens_per_step": 4 * 512},
}


def apply(settings: Settings, overrides: dict) -> Settings:
    for section, values in overrides.items():
        settings.data[section].update(values)
    return settings


def make_printer(verbose: bool):
    last = {"step": -1, "at": 0.0}

    def emit(event: dict) -> None:
        kind = event.get("type")
        if kind == "stage":
            print(f"\n==> {event['name']}: {event['status']}", flush=True)
            if event["status"] == "done" and verbose and event.get("detail"):
                print(f"    {json.dumps(event['detail'], default=str)[:400]}", flush=True)
        elif kind == "setup":
            print(f"    {event['parameters']:,} parameters, {event['precision']}, "
                  f"{event['total_steps']:,} steps on {event['device']}", flush=True)
        elif kind == "progress":
            # Throttle: a long run logs thousands of these.
            now = time.time()
            if now - last["at"] < 2.0 and event["step"] != event["total_steps"] - 1:
                return
            last["at"] = now
            eta = event["eta_seconds"]
            eta_text = (f"{eta / 3600:.1f}h" if eta > 3600
                        else f"{eta / 60:.0f}m" if eta > 90 else f"{eta:.0f}s")
            print(f"    step {event['step']:>7,}/{event['total_steps']:,}  "
                  f"loss {event['loss']:.4f}  "
                  f"{event['tokens_per_second'] / 1000:.1f}k tok/s  eta {eta_text}",
                  flush=True)
        elif kind == "eval":
            print(f"    step {event['step']:>7,}  val {event['val_loss']:.4f}"
                  f"{'  (best)' if event['best'] else ''}", flush=True)
        elif kind in ("note", "advice"):
            print(f"    {event['text']}", flush=True)
        elif kind == "aborted":
            print(f"\n!!  aborted: {event['reason']}", file=sys.stderr, flush=True)
        elif kind == "stopped":
            print(f"\n    stopped at step {event['step']}, checkpoint written", flush=True)
        elif kind == "error":
            print(f"\n!!  {event.get('stage', 'pipeline')}: {event['message']}",
                  file=sys.stderr, flush=True)
        elif kind == "finished":
            print(f"\n==> finished in {event['seconds'] / 60:.1f} minutes", flush=True)

    return emit


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("mode", choices=("smoke", "full"), nargs="?", default="smoke")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--stages", default=None,
                   help=f"comma-separated; default {','.join(DEFAULT_STAGES)}. "
                        f"Available: {','.join(STAGES)}")
    p.add_argument("--max-hours", type=float, default=None)
    p.add_argument("--preset", default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    settings = load_settings(args.config)
    if args.mode == "smoke":
        settings = apply(settings, SMOKE_OVERRIDES)
    if args.max_hours is not None:
        settings.data["hardware"]["max_hours"] = args.max_hours
    if args.preset:
        settings.data["model"]["preset"] = args.preset

    problems = settings.validate()
    if problems:
        for problem in problems:
            print(f"config problem: {problem}", file=sys.stderr)
        return 2

    stages = args.stages.split(",") if args.stages else list(DEFAULT_STAGES)
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        print(f"unknown stage(s): {', '.join(unknown)}", file=sys.stderr)
        return 2

    plan = settings.plan()
    print(f"\n{plan.summary()}\n")
    print(f"stages: {' -> '.join(stages)}")
    if args.mode == "smoke":
        print("(smoke mode: deliberately tiny -- this will not produce a usable model)")

    trainer = Trainer(settings, make_printer(not args.quiet))
    try:
        trainer.run(stages)
    except KeyboardInterrupt:
        print("\ninterrupted; asking the run to checkpoint and stop", file=sys.stderr)
        trainer.stop_flag.set()
        return 130

    failed = not trainer.results or len(trainer.results) < len(stages)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
