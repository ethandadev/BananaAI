"""Export: safetensors round-trip and GGUF readback.

An exporter that writes a file without error but permutes a tensor, drops a
layer, or misorders the vocabulary produces a model that loads fine elsewhere
and generates rubbish. So every test here reads the artefact back and compares
it to the source weights.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from tests.harness import Suite, temp_dir

from core.config import ModelConfig
from core.model import Transformer
from core.tokenizer import train as train_tokenizer
from core.train import PRESETS
from export.to_gguf import convert as to_gguf, gguf_name, tokenizer_arrays
from export.to_safetensors import convert as to_safetensors, hf_config, rename

suite = Suite("export")
test = suite.test

TINY = ModelConfig(**PRESETS["tiny"], vocab_size=300)


def checkpoint(tmp: Path) -> tuple[Path, Transformer]:
    torch.manual_seed(0)
    model = Transformer(TINY)
    path = tmp / "ckpt.pt"
    torch.save({"model": model.state_dict(), "model_config": asdict(TINY),
                "step": 0, "stage": "dpo"}, path)
    return path, model


def tokenizer_file(tmp: Path) -> Path:
    texts = [f"sample text number {i} about models and tokens and gradients" for i in range(120)]
    texts += [f"def f{i}(x):\n    return x + {i}\n" for i in range(40)]
    tok = train_tokenizer(texts, vocab_size=300, min_frequency=1)
    path = tmp / "tokenizer.json"
    tok.save(str(path))
    return path


@test
def every_parameter_has_an_hf_name():
    model = Transformer(TINY)
    for key in model.state_dict():
        if key.startswith("rope_"):
            continue
        name = rename(key)
        assert name.startswith(("model.", "lm_head")), f"{key} -> {name}"
    return f"{len(model.state_dict())} tensors mapped"


@test
def every_parameter_has_a_gguf_name():
    model = Transformer(TINY)
    names = set()
    for key in model.state_dict():
        if key.startswith("rope_") or key == "lm_head.weight":
            continue
        name = gguf_name(key)
        assert name not in names, f"duplicate GGUF name {name}"
        names.add(name)
    assert "token_embd.weight" in names and "output_norm.weight" in names
    return f"{len(names)} unique GGUF names"


@test
def unknown_keys_are_rejected_not_silently_dropped():
    for fn in (rename, gguf_name):
        try:
            fn("blocks.0.attn.invented_thing.weight")
        except KeyError:
            continue
        raise AssertionError(f"{fn.__name__} accepted an unknown key")
    return "both mappers raise on an unmapped tensor"


@test
def hf_config_matches_the_model():
    cfg = hf_config(TINY)
    assert cfg["num_attention_heads"] == TINY.n_heads
    assert cfg["num_key_value_heads"] == TINY.n_kv_heads
    assert cfg["intermediate_size"] == TINY.ffn_hidden
    assert cfg["hidden_size"] == TINY.d_model
    assert cfg["tie_word_embeddings"] is True
    assert cfg["architectures"] == ["LlamaForCausalLM"]
    assert cfg["rope_theta"] == TINY.rope_theta
    return "config.json agrees with ModelConfig on every field"


@test
def safetensors_round_trips_bit_exactly():
    from safetensors.torch import load_file

    with temp_dir() as tmp:
        path, model = checkpoint(tmp)
        info = to_safetensors(path, tmp / "hf", dtype="float32")

        loaded = load_file(str(tmp / "hf" / "model.safetensors"))
        original = model.state_dict()

        for key, tensor in original.items():
            if key.startswith("rope_") or key == "lm_head.weight":
                continue
            got = loaded[rename(key)]
            assert torch.equal(got, tensor), f"{key} changed during export"

        assert "lm_head.weight" not in loaded, "tied head written twice"
        assert info["tensors"] == len(loaded)
    return f"{info['tensors']} tensors, every value identical"


@test
def exported_weights_reload_into_the_model():
    """The real test: put the exported tensors back and check the logits match."""
    from safetensors.torch import load_file

    with temp_dir() as tmp:
        path, model = checkpoint(tmp)
        to_safetensors(path, tmp / "hf", dtype="float32")
        loaded = load_file(str(tmp / "hf" / "model.safetensors"))

        reverse = {rename(k): k for k in model.state_dict() if not k.startswith("rope_")}
        rebuilt = Transformer(TINY)
        state = {reverse[name]: tensor for name, tensor in loaded.items()}
        state["lm_head.weight"] = state["embed.weight"]      # tied
        rebuilt.load_state_dict(state, strict=False)

        x = torch.randint(0, TINY.vocab_size, (2, 16))
        model.eval(); rebuilt.eval()
        with torch.no_grad():
            a, _, _ = model(x)
            b, _, _ = rebuilt(x)
        err = (a - b).abs().max().item()
        assert err == 0.0, f"reloaded model differs by {err:.2e}"
    return "logits identical after export and reload"


@test
def gguf_vocabulary_is_written_in_id_order():
    with temp_dir() as tmp:
        tok_path = tokenizer_file(tmp)
        tokens, types, merges = tokenizer_arrays(tok_path)

        raw = json.loads(tok_path.read_text(encoding="utf-8"))
        vocab = raw["model"]["vocab"]
        assert len(tokens) == len(vocab), "token count mismatch"
        for token, idx in vocab.items():
            assert tokens[idx] == token, f"id {idx} holds {tokens[idx]!r}, expected {token!r}"
        assert len(types) == len(tokens)
        assert merges, "no merges extracted; BPE would be broken"
    return f"{len(tokens)} tokens in id order, {len(merges)} merges"


@test
def gguf_readback_matches_the_source_weights():
    import gguf

    with temp_dir() as tmp:
        path, model = checkpoint(tmp)
        tok_path = tokenizer_file(tmp)
        out = tmp / "m.gguf"
        info = to_gguf(path, tok_path, out, dtype="float32")

        reader = gguf.GGUFReader(str(out))
        found = {t.name: t for t in reader.tensors}
        assert len(found) == info["tensors"], "tensor count changed on readback"

        original = model.state_dict()
        checked = 0
        for key, tensor in original.items():
            if key.startswith("rope_") or key == "lm_head.weight":
                continue
            t = found[gguf_name(key)]
            got = np.array(t.data).reshape(tensor.shape)
            assert np.allclose(got, tensor.numpy(), atol=1e-6), f"{key} differs in the GGUF"
            checked += 1
        assert checked == info["tensors"]
    return f"{checked} tensors match after readback"


@test
def gguf_metadata_describes_the_architecture():
    import gguf

    with temp_dir() as tmp:
        path, _ = checkpoint(tmp)
        out = tmp / "m.gguf"
        to_gguf(path, tokenizer_file(tmp), out, dtype="float16")

        reader = gguf.GGUFReader(str(out))
        fields = reader.fields

        def scalar(key):
            f = fields[key]
            return f.parts[f.data[0]][0]

        assert scalar("llama.block_count") == TINY.n_layers
        assert scalar("llama.embedding_length") == TINY.d_model
        assert scalar("llama.attention.head_count") == TINY.n_heads
        assert scalar("llama.attention.head_count_kv") == TINY.n_kv_heads
        assert scalar("llama.context_length") == TINY.context_len
        assert scalar("llama.feed_forward_length") == TINY.ffn_hidden
        arch = bytes(fields["general.architecture"].parts[-1]).decode()
        assert arch == "llama", f"architecture is {arch!r}"
    return "block count, dims, heads, and context all correct"


@test
def float16_export_halves_the_file():
    with temp_dir() as tmp:
        path, _ = checkpoint(tmp)
        tok = tokenizer_file(tmp)
        f32 = to_gguf(path, tok, tmp / "a.gguf", dtype="float32")["megabytes"]
        f16 = to_gguf(path, tok, tmp / "b.gguf", dtype="float16")["megabytes"]
        assert f16 < f32 * 0.6, f"f16 is {f16}MB against f32 {f32}MB"
    return f"{f32}MB f32 -> {f16}MB f16"


if __name__ == "__main__":
    p, n = suite.run()
    raise SystemExit(0 if p == n else 1)
