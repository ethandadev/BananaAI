"""The whole pipeline, start to finish, in one run.

Raw documents -> filter -> dedupe -> shard -> tokenizer -> token bins ->
pretrain -> SFT -> DPO -> safetensors -> GGUF -> sidecar generation.

Unit tests confirm each stage in isolation. This confirms they actually fit
together: that the tokenizer the pipeline trained can read the shards the
pipeline wrote, that the checkpoint the trainer saved is the one SFT can load,
and that a model carried all the way through still generates text.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import torch

from tests.harness import REPO, Suite, ensure_fixtures, temp_dir

ensure_fixtures()

from core.config import ModelConfig, TrainConfig
from core.tokenizer import stream_texts_from_shards
from core.train import PRESETS, train
from data.prepare import run as run_pipeline
from data.sources import stream_local
from data.tokenize_corpus import encode_stream, write_bin
from export.to_gguf import convert as to_gguf
from export.to_safetensors import convert as to_safetensors
from post.dpo import run_dpo
from post.sft import run_sft
from server.sidecar import Engine, Sidecar

suite = Suite("integration")
test = suite.test

FIXTURES = REPO / "tests" / "fixtures"
VOCAB = 512


@test
def full_pipeline_produces_a_generating_model():
    from core.tokenizer import train as train_tokenizer

    with temp_dir() as tmp:
        stages = {}

        # 1. corpus -------------------------------------------------------
        docs = stream_local(FIXTURES / "corpus" / "corpus.jsonl", "fix", "prose")
        manifest = run_pipeline(docs, tmp / "processed", "fix", window=200)
        assert manifest["documents_out"] > 50, manifest
        stages["corpus"] = f"{manifest['documents_out']} docs"

        # 2. tokenizer ----------------------------------------------------
        texts = list(stream_texts_from_shards(tmp / "processed"))
        assert len(texts) == manifest["documents_out"], \
            "shards did not read back as many documents as were written"
        tok = train_tokenizer(texts, vocab_size=VOCAB, min_frequency=1)
        tok_path = tmp / "tokenizer.json"
        tok.save(str(tok_path))
        vocab = tok.get_vocab_size()
        stages["tokenizer"] = f"{vocab} tokens"

        # 3. token bins ---------------------------------------------------
        data_dir = tmp / "tokenized"
        stream = encode_stream(tok, texts)
        head = [next(stream) for _ in range(8)]
        n_val = write_bin(data_dir / "val.bin", head)
        n_train = write_bin(data_dir / "train.bin", stream)
        (data_dir / "meta.json").write_text(json.dumps({
            "vocab_size": vocab, "train_tokens": n_train,
            "val_tokens": n_val, "dtype": "uint16", "eos_id": 2,
        }), encoding="utf-8")
        assert n_train > 5000, f"only {n_train} training tokens"
        stages["bins"] = f"{n_train:,} train / {n_val:,} val"

        # 4. pretrain -----------------------------------------------------
        mc = ModelConfig(**PRESETS["test"], vocab_size=vocab)
        tc = TrainConfig(total_steps=40, warmup_steps=4, peak_lr=3e-3, min_lr=3e-4,
                         micro_batch=8, tokens_per_step=8 * mc.context_len,
                         eval_every=20, ckpt_every=20, log_every=10**9,
                         sample_every=10**9)
        base = train(mc, tc, data_dir, tmp / "base", device="cpu",
                     compile_model=False, seed=0)
        assert base["best_val_loss"] is not None
        stages["pretrain"] = f"val {base['best_val_loss']:.2f}"

        base_ckpt = tmp / "base" / "best.pt"
        assert base_ckpt.exists(), "pretraining produced no best checkpoint"

        # 5. SFT ----------------------------------------------------------
        sft = run_sft(base_ckpt, FIXTURES / "sft.jsonl", tmp / "sft", tok_path,
                      epochs=3, batch_size=4, lr=1e-3, device="cpu", log_every=10**9)
        assert sft["final_loss"] < sft["first_loss"], \
            f"SFT loss rose from {sft['first_loss']:.3f} to {sft['final_loss']:.3f}"
        stages["sft"] = f"{sft['first_loss']:.2f} -> {sft['final_loss']:.2f}"

        # 6. DPO ----------------------------------------------------------
        dpo = run_dpo(Path(sft["checkpoint"]), FIXTURES / "prefs.jsonl", tmp / "dpo",
                      tok_path, beta=0.1, epochs=3, batch_size=4, lr=5e-5,
                      device="cpu", log_every=10**9)
        assert dpo["final_accuracy"] >= 0.5, \
            f"DPO preference accuracy only {dpo['final_accuracy']}"
        stages["dpo"] = f"acc {dpo['final_accuracy']:.2f}"

        # 7. export -------------------------------------------------------
        dpo_ckpt = Path(dpo["checkpoint"])
        st = to_safetensors(dpo_ckpt, tmp / "hf", dtype="float32", tokenizer_path=tok_path)
        gg = to_gguf(dpo_ckpt, tok_path, tmp / "model.gguf", dtype="float16")
        assert st["tensors"] == gg["tensors"], \
            f"safetensors wrote {st['tensors']} tensors, GGUF wrote {gg['tensors']}"
        assert (tmp / "hf" / "config.json").exists()
        stages["export"] = f"{gg['tensors']} tensors, {gg['megabytes']}MB gguf"

        # 8. serve --------------------------------------------------------
        engine = Engine(dpo_ckpt, tok_path, device="cpu")
        out = io.StringIO()
        sidecar = Sidecar(engine)
        sidecar.emit = lambda o: out.write(json.dumps(o) + "\n")
        sidecar.serve(stdin=io.StringIO(json.dumps({
            "id": "1", "type": "generate",
            "messages": [{"role": "user", "content": "Explain what a tokenizer does."}],
            "params": {"max_new_tokens": 24, "seed": 0},
        }) + "\n"))

        events = [json.loads(l) for l in out.getvalue().splitlines() if l.strip()]
        done = [e for e in events if e["type"] == "done"]
        assert done, "the sidecar never completed a generation"
        assert done[0]["tokens"] > 0, "generated nothing"
        stages["serve"] = f"{done[0]['tokens']} tokens"

    return " | ".join(f"{k}: {v}" for k, v in stages.items())


@test
def a_checkpoint_survives_every_stage_boundary():
    """Each stage must load what the previous one saved, with config intact."""
    from core.tokenizer import train as train_tokenizer

    with temp_dir() as tmp:
        tok = train_tokenizer(
            [f"document {i} about models and tokens and training" for i in range(120)],
            vocab_size=VOCAB, min_frequency=1)
        tok_path = tmp / "tok.json"
        tok.save(str(tok_path))

        mc = ModelConfig(**PRESETS["test"], vocab_size=tok.get_vocab_size())
        from dataclasses import asdict
        from core.model import Transformer

        torch.manual_seed(0)
        base = tmp / "base.pt"
        torch.save({"model": Transformer(mc).state_dict(),
                    "model_config": asdict(mc), "step": 0, "stage": "base"}, base)

        sft = run_sft(base, FIXTURES / "sft.jsonl", tmp / "s", tok_path,
                      epochs=1, batch_size=4, lr=1e-4, device="cpu", log_every=10**9)
        dpo = run_dpo(Path(sft["checkpoint"]), FIXTURES / "prefs.jsonl", tmp / "d",
                      tok_path, epochs=1, batch_size=4, lr=1e-6,
                      device="cpu", log_every=10**9)

        shapes = {}
        for label, path in (("base", base), ("sft", Path(sft["checkpoint"])),
                            ("dpo", Path(dpo["checkpoint"]))):
            state = torch.load(path, map_location="cpu", weights_only=False)
            cfg = ModelConfig(**state["model_config"])
            assert cfg.n_layers == mc.n_layers and cfg.vocab_size == mc.vocab_size, \
                f"{label} checkpoint carries a different config"
            shapes[label] = state["model"]["embed.weight"].shape

        assert len({tuple(s) for s in shapes.values()}) == 1, \
            f"embedding shape changed between stages: {shapes}"
    return "base -> sft -> dpo, config and shapes preserved"


if __name__ == "__main__":
    p, n = suite.run()
    raise SystemExit(0 if p == n else 1)
