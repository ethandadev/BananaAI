"""Training sidecar: run the pipeline and stream progress as JSON lines.

    python -m server.trainer --config bananaai.toml

Same protocol shape as server/sidecar.py -- one request object per line in, a
stream of event objects out -- so the desktop app talks to both the same way.

Requests
    {"type": "plan"}                      what would this machine train?
    {"type": "start", "stages": [...]}    run those stages
    {"type": "stop"}                      finish the current step, checkpoint, exit
    {"type": "status"}
    {"type": "shutdown"}

Events
    {"type": "stage", "name": "pretrain", "status": "start"}
    {"type": "progress", "step": 120, "loss": 4.31, "eta_seconds": 900, ...}
    {"type": "eval", "step": 250, "val_loss": 4.02, "best": true}
    {"type": "stage", "name": "pretrain", "status": "done", "detail": {...}}
    {"type": "finished", "stages": {...}}

Work runs on a thread so a stop request can be read while training is in
progress; a single-threaded loop could not.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Optional

from core.settings import Settings, load as load_settings

PROTOCOL_VERSION = 1

STAGES = ("data", "tokenizer", "tokenize", "pretrain", "sft", "dpo", "export")
DEFAULT_STAGES = ("data", "tokenizer", "tokenize", "pretrain")


class Trainer:
    def __init__(self, settings: Settings, emit):
        self.settings = settings
        self.emit = emit
        self.stop_flag = threading.Event()
        self.current: Optional[str] = None
        self.started_at: Optional[float] = None
        self.results: dict = {}

    # -- individual stages -------------------------------------------------

    def stage_data(self) -> dict:
        from data.custom import assess, documents
        from data.prepare import run as run_pipeline
        from data.sources import MIX, stream_hf

        cfg = self.settings["data"]
        out_dir = Path(cfg["out_dir"])
        custom_dir = cfg["custom_dir"]
        limit = cfg["limit_per_source"] or None

        if custom_dir:
            stats: dict = {}
            docs = documents(Path(custom_dir), "custom", stats=stats)
            manifest = run_pipeline(docs, out_dir, "custom",
                                    window=cfg["dedupe_window"],
                                    threshold=cfg["dedupe_threshold"])
            plan = self.settings.plan()
            manifest["assessment"] = assess(stats.get("characters", 0),
                                            plan.model.n_params())
            self.emit({"type": "advice", "text": manifest["assessment"]["advice"]})
            return manifest

        def chained():
            for source in MIX:
                self.emit({"type": "note", "text": f"downloading {source.name}"})
                yield from stream_hf(source, limit)

        return run_pipeline(chained(), out_dir, "mix",
                            window=cfg["dedupe_window"],
                            threshold=cfg["dedupe_threshold"])

    def stage_tokenizer(self) -> dict:
        from core.tokenizer import compression_ratio, stream_texts_from_shards, train

        cfg = self.settings["data"]
        texts = list(stream_texts_from_shards(Path(cfg["out_dir"]), 200_000))
        if not texts:
            raise RuntimeError(
                f"no documents in {cfg['out_dir']} -- run the data stage first")
        vocab = self.settings["model"]["vocab_size"]
        tok = train(texts, vocab_size=vocab, out_path=Path("tokenizer.json"))
        return {"vocab_size": tok.get_vocab_size(), "documents": len(texts),
                "chars_per_token": round(compression_ratio(tok, texts[:500]), 3)}

    def stage_tokenize(self) -> dict:
        from data.tokenize_corpus import main as tokenize_main

        cfg = self.settings["data"]
        tokenize_main([
            "--shards", cfg["out_dir"], "--tokenizer", "tokenizer.json",
            "--out", cfg["tokenized_dir"], "--val-tokens", str(cfg["val_tokens"]),
        ])
        return json.loads((Path(cfg["tokenized_dir"]) / "meta.json").read_text())

    def stage_pretrain(self) -> dict:
        from core.train import train as run_train

        plan = self.settings.plan()
        self.emit({"type": "plan", "summary": plan.summary(), "preset": plan.preset,
                   "parameters": plan.model.n_params(),
                   "estimated_hours": plan.estimated_hours,
                   "warnings": plan.warnings})

        tc = plan.train
        overrides = self.settings["train"]
        if overrides["peak_lr"]:
            tc.peak_lr = overrides["peak_lr"]
            tc.min_lr = overrides["peak_lr"] / 10
        for key in ("ckpt_every", "total_steps", "micro_batch", "tokens_per_step"):
            if overrides[key]:
                setattr(tc, key, overrides[key])
        tc.grad_clip = overrides["grad_clip"]
        # Warmup and eval cadence must stay sensible relative to a shortened run.
        tc.warmup_steps = max(1, min(tc.warmup_steps, tc.total_steps // 10))
        tc.eval_every = max(1, min(tc.eval_every, max(1, tc.total_steps // 4)))
        tc.ckpt_every = max(1, min(tc.ckpt_every, max(1, tc.total_steps // 2)))
        # grad_accum_steps raises if these do not divide evenly.
        tc.grad_accum_steps(plan.model.context_len)

        return run_train(
            plan.model, tc,
            Path(self.settings["data"]["tokenized_dir"]),
            Path(overrides["out_dir"]),
            device=plan.device,
            compile_model=self.settings["hardware"]["compile"],
            seed=overrides["seed"],
            spike_factor=overrides["spike_factor"],
            tokenizer_path=Path("tokenizer.json"),
            on_event=self.emit,
            should_stop=self.stop_flag.is_set,
        )

    def stage_sft(self) -> dict:
        from data.fetch_post import main as fetch_main
        from post.sft import run_sft

        cfg = self.settings["sft"]
        if not cfg["enabled"]:
            return {"skipped": "sft.enabled is false"}
        data = Path(cfg["data"])
        if not data.exists():
            fetch_main(["sft", "--target", str(cfg["target"]),
                        "--out", str(data.parent)])
        return run_sft(
            Path(self.settings["train"]["out_dir"]) / "best.pt", data,
            Path(cfg["out_dir"]), Path("tokenizer.json"),
            epochs=cfg["epochs"], batch_size=cfg["batch_size"], lr=cfg["lr"],
        )

    def stage_dpo(self) -> dict:
        from data.fetch_post import main as fetch_main
        from post.dpo import run_dpo

        cfg = self.settings["dpo"]
        if not cfg["enabled"]:
            return {"skipped": "dpo.enabled is false"}
        data = Path(cfg["data"])
        if not data.exists():
            fetch_main(["dpo", "--target", str(cfg["target"]),
                        "--out", str(data.parent)])
        return run_dpo(
            Path(self.settings["sft"]["out_dir"]) / "sft.pt", data,
            Path(cfg["out_dir"]), Path("tokenizer.json"),
            beta=cfg["beta"], epochs=cfg["epochs"],
            batch_size=cfg["batch_size"], lr=cfg["lr"],
        )

    def stage_export(self) -> dict:
        from export.to_gguf import convert as to_gguf
        from export.to_safetensors import convert as to_safetensors

        cfg = self.settings["export"]
        # Most-finished stage first, but only among the directories this
        # configuration actually writes to. Scanning fixed paths picked up
        # stale checkpoints from unrelated runs and exported those instead.
        candidates = [
            Path(self.settings["dpo"]["out_dir"]) / "dpo.pt",
            Path(self.settings["sft"]["out_dir"]) / "sft.pt",
            Path(self.settings["train"]["out_dir"]) / "best.pt",
        ]
        for candidate in candidates:
            if candidate.exists():
                ckpt = candidate
                break
        else:
            searched = ", ".join(str(c) for c in candidates)
            raise RuntimeError(f"no checkpoint to export -- looked in {searched}")

        out_dir = Path(cfg["out_dir"])
        out: dict = {"checkpoint": str(ckpt)}
        name = self.settings["project"]["name"]
        if cfg["safetensors"]:
            out["safetensors"] = to_safetensors(
                ckpt, out_dir / "model", cfg["dtype"], Path("tokenizer.json"))
        if cfg["gguf"]:
            gguf_dtype = "float16" if cfg["dtype"] == "bfloat16" else cfg["dtype"]
            out["gguf"] = to_gguf(ckpt, Path("tokenizer.json"),
                                  out_dir / f"{name}-{gguf_dtype}.gguf",
                                  gguf_dtype, name)
        return out

    # -- driving -----------------------------------------------------------

    def run(self, stages) -> None:
        self.started_at = time.time()
        self.results = {}
        handlers = {
            "data": self.stage_data, "tokenizer": self.stage_tokenizer,
            "tokenize": self.stage_tokenize, "pretrain": self.stage_pretrain,
            "sft": self.stage_sft, "dpo": self.stage_dpo, "export": self.stage_export,
        }
        try:
            for name in stages:
                if self.stop_flag.is_set():
                    self.emit({"type": "stage", "name": name, "status": "cancelled"})
                    break
                self.current = name
                self.emit({"type": "stage", "name": name, "status": "start"})
                detail = handlers[name]()
                self.results[name] = detail
                self.emit({"type": "stage", "name": name, "status": "done",
                           "detail": detail})
            self.emit({"type": "finished", "stages": self.results,
                       "seconds": round(time.time() - self.started_at, 1),
                       "stopped": self.stop_flag.is_set()})
        except Exception as e:                     # noqa: BLE001
            self.emit({"type": "error", "stage": self.current,
                       "message": f"{type(e).__name__}: {e}",
                       "traceback": traceback.format_exc()[-2000:]})
        finally:
            self.current = None


class TrainerServer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.lock = threading.Lock()
        self.trainer = Trainer(settings, self.emit)
        self.worker: Optional[threading.Thread] = None

    def emit(self, obj: dict) -> None:
        with self.lock:
            sys.stdout.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
            sys.stdout.flush()

    def status(self) -> dict:
        running = self.worker is not None and self.worker.is_alive()
        return {
            "running": running,
            "stage": self.trainer.current,
            "elapsed": round(time.time() - self.trainer.started_at, 1)
            if running and self.trainer.started_at else 0,
            "completed": sorted(self.trainer.results),
        }

    def serve(self, stdin=None) -> None:
        stdin = stdin or sys.stdin
        problems = self.settings.validate()
        plan = None
        if not problems:
            try:
                plan = self.settings.plan()
            except Exception as e:                 # noqa: BLE001
                problems.append(f"could not plan: {type(e).__name__}: {e}")

        self.emit({
            "type": "ready", "protocol": PROTOCOL_VERSION, "stages": list(STAGES),
            "config_source": str(self.settings.source) if self.settings.source else None,
            "problems": problems,
            **({"device": plan.device.describe(), "preset": plan.preset,
                "parameters": plan.model.n_params(),
                "estimated_hours": plan.estimated_hours,
                "plan_warnings": plan.warnings} if plan else {}),
        })

        shutdown = False
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError as e:
                self.emit({"type": "error", "message": f"malformed request: {e}"})
                continue

            kind = request.get("type")
            if kind == "plan":
                try:
                    p = self.settings.plan()
                    self.emit({"type": "plan", "id": request.get("id"),
                               "summary": p.summary(), "preset": p.preset,
                               "parameters": p.model.n_params(),
                               "estimated_hours": p.estimated_hours,
                               "warnings": p.warnings})
                except Exception as e:             # noqa: BLE001
                    self.emit({"type": "error", "message": str(e)})
            elif kind == "status":
                self.emit({"type": "status", "id": request.get("id"), **self.status()})
            elif kind == "stop":
                self.trainer.stop_flag.set()
                self.emit({"type": "stopping", "id": request.get("id")})
            elif kind == "shutdown":
                shutdown = True
                self.emit({"type": "bye"})
                break
            elif kind == "start":
                if self.worker is not None and self.worker.is_alive():
                    self.emit({"type": "error", "message": "a run is already in progress"})
                    continue
                requested = request.get("stages") or list(DEFAULT_STAGES)
                unknown = [s for s in requested if s not in STAGES]
                if unknown:
                    self.emit({"type": "error",
                               "message": f"unknown stage(s): {', '.join(unknown)}"})
                    continue
                self.trainer.stop_flag.clear()
                self.worker = threading.Thread(
                    target=self.trainer.run, args=(requested,), daemon=True)
                self.worker.start()
            else:
                self.emit({"type": "error", "message": f"unknown request type: {kind!r}"})

        if self.worker is not None and self.worker.is_alive():
            # Shutdown means stop now; plain EOF lets the run finish, matching
            # server/sidecar.py.
            if shutdown:
                self.trainer.stop_flag.set()
                self.worker.join(timeout=120)
            else:
                self.worker.join()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", type=Path, default=None)
    args = p.parse_args(argv)
    TrainerServer(load_settings(args.config)).serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
