"""Inference sidecar: newline-delimited JSON over stdin/stdout.

    python -m server.sidecar --ckpt runs/dpo/dpo.pt --tokenizer tokenizer.json

Electron cannot run PyTorch, so the model lives here in a child process. The
protocol is deliberately thin -- one request object per line in, a stream of
event objects per line out -- which keeps the core usable headlessly and the
UI replaceable.

Requests
    {"id": "1", "type": "info"}
    {"id": "2", "type": "generate", "messages": [...], "params": {...}}
    {"id": "3", "type": "cancel"}

Events
    {"id": "2", "type": "start"}
    {"id": "2", "type": "token", "text": "hel"}
    {"id": "2", "type": "done", "tokens": 42, "seconds": 1.2, "tokens_per_second": 35.0}
    {"id": "2", "type": "error", "message": "..."}

Generation runs on a worker thread so a cancel request can be read and acted
on mid-stream; a single-threaded loop could not read stdin while generating.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import torch

from core.config import ModelConfig
from core.model import Transformer
from core.sample import SamplingConfig, stream
from core.tokenizer import Message, load as load_tokenizer, render

PROTOCOL_VERSION = 1


class Engine:
    def __init__(self, ckpt: Path, tokenizer_path: Path, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(ckpt, map_location=self.device, weights_only=False)
        self.cfg = ModelConfig(**state["model_config"])
        self.model = Transformer(self.cfg)
        self.model.load_state_dict(state["model"])
        self.model = self.model.to(self.device).eval()
        if self.device.startswith("cuda"):
            self.model = self.model.to(torch.bfloat16)
        self.tokenizer = load_tokenizer(tokenizer_path)
        self.stage = state.get("stage", "base")
        self.ckpt = str(ckpt)

        eos = self.tokenizer.token_to_id("<|eos|>")
        end = self.tokenizer.token_to_id("<|end|>")
        self.stop_tokens = tuple(t for t in (eos, end) if t is not None)

    def info(self) -> dict:
        return {
            "protocol": PROTOCOL_VERSION,
            "stage": self.stage,
            "checkpoint": self.ckpt,
            "device": self.device,
            "parameters": self.model.num_params(),
            "context_len": self.cfg.context_len,
            "vocab_size": self.cfg.vocab_size,
            "stop_tokens": list(self.stop_tokens),
        }

    def build_prompt(self, messages: list[dict]) -> str:
        msgs = [Message(m["role"], m["content"]) for m in messages]
        return render(msgs, add_generation_prompt=True)

    def encode(self, text: str) -> torch.Tensor:
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        limit = self.cfg.context_len - 1
        if len(ids) > limit:
            ids = ids[-limit:]      # keep the most recent turns
        return torch.tensor([ids], dtype=torch.long, device=self.device)


class Sidecar:
    def __init__(self, engine: Engine):
        self.engine = engine
        self.cancel = threading.Event()
        self.lock = threading.Lock()

    def emit(self, obj: dict) -> None:
        with self.lock:
            sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            sys.stdout.flush()

    def handle_generate(self, req: dict) -> None:
        rid = req.get("id")
        params = req.get("params") or {}
        try:
            messages = req["messages"]
            prompt = self.engine.build_prompt(messages)
            ids = self.engine.encode(prompt)

            cfg = SamplingConfig(
                max_new_tokens=int(params.get("max_new_tokens", 256)),
                temperature=float(params.get("temperature", 0.8)),
                top_k=params.get("top_k", 50),
                top_p=params.get("top_p", 0.95),
                repetition_penalty=float(params.get("repetition_penalty", 1.1)),
                stop_tokens=self.engine.stop_tokens,
                seed=params.get("seed"),
            )

            self.emit({"id": rid, "type": "start"})
            t0 = time.perf_counter()
            count = 0
            # Decode incrementally: byte-level BPE tokens are not individually
            # valid UTF-8, so decode the accumulated ids and emit the delta.
            produced: list[int] = []
            text_so_far = ""

            for token in stream(self.engine.model, ids, cfg):
                if self.cancel.is_set():
                    break
                produced.append(token)
                count += 1
                decoded = self.engine.tokenizer.decode(produced)
                if len(decoded) > len(text_so_far):
                    self.emit({"id": rid, "type": "token", "text": decoded[len(text_so_far):]})
                    text_so_far = decoded

            dt = time.perf_counter() - t0
            self.emit({
                "id": rid,
                "type": "done",
                "tokens": count,
                "seconds": round(dt, 3),
                "tokens_per_second": round(count / dt, 2) if dt > 0 else 0.0,
                "cancelled": self.cancel.is_set(),
                "text": text_so_far,
            })
        except Exception as e:                      # noqa: BLE001
            self.emit({"id": rid, "type": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            self.cancel.clear()

    def serve(self, stdin=None, ready: bool = True) -> None:
        stdin = stdin or sys.stdin
        worker: Optional[threading.Thread] = None
        shutdown_requested = False

        if ready:
            self.emit({"type": "ready", **self.engine.info()})

        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except json.JSONDecodeError as e:
                self.emit({"type": "error", "message": f"malformed request: {e}"})
                continue

            kind = req.get("type")
            if kind == "info":
                self.emit({"id": req.get("id"), "type": "info", **self.engine.info()})
            elif kind == "cancel":
                self.cancel.set()
                self.emit({"id": req.get("id"), "type": "cancelled"})
            elif kind == "shutdown":
                shutdown_requested = True
                self.emit({"id": req.get("id"), "type": "bye"})
                break
            elif kind == "generate":
                if worker is not None and worker.is_alive():
                    self.emit({
                        "id": req.get("id"), "type": "error",
                        "message": "a generation is already running",
                    })
                    continue
                worker = threading.Thread(target=self.handle_generate, args=(req,), daemon=True)
                worker.start()
            else:
                self.emit({"id": req.get("id"), "type": "error",
                           "message": f"unknown request type: {kind!r}"})

        if worker is not None and worker.is_alive():
            # An explicit shutdown means stop now. Plain EOF on stdin does not:
            # the caller may simply have finished writing requests, and killing
            # a generation mid-stream would drop output it is still waiting for.
            if shutdown_requested:
                self.cancel.set()
                worker.join(timeout=5)
            else:
                worker.join()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, default=Path("tokenizer.json"))
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    engine = Engine(args.ckpt, args.tokenizer, args.device)
    Sidecar(engine).serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
