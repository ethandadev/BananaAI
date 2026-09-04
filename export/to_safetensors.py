"""Convert a training checkpoint to a HuggingFace-layout safetensors model.

    python -m export.to_safetensors --ckpt runs/dpo/dpo.pt --out export/model

Our parameter names are renamed to Llama's, and a config.json declaring
LlamaForCausalLM is written alongside. That is what makes the model loadable
with AutoModelForCausalLM and readable by llama.cpp's converter.

No weight permutation is applied, and none is needed. The permutation in
Meta's own conversion script exists because the original Llama code applies
RoPE to interleaved pairs while HF applies it to split halves. core/model.py
uses the split-halves convention from the start, so the weights are already in
HF form.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from core.config import ModelConfig

# our name -> HF Llama name
DIRECT = {
    "embed.weight": "model.embed_tokens.weight",
    "norm.weight": "model.norm.weight",
    "lm_head.weight": "lm_head.weight",
}
PER_LAYER = {
    "attn_norm.weight": "input_layernorm.weight",
    "attn.wq.weight": "self_attn.q_proj.weight",
    "attn.wk.weight": "self_attn.k_proj.weight",
    "attn.wv.weight": "self_attn.v_proj.weight",
    "attn.wo.weight": "self_attn.o_proj.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "ffn.gate.weight": "mlp.gate_proj.weight",
    "ffn.up.weight": "mlp.up_proj.weight",
    "ffn.down.weight": "mlp.down_proj.weight",
}


def rename(key: str) -> str:
    if key in DIRECT:
        return DIRECT[key]
    if key.startswith("blocks."):
        _, idx, rest = key.split(".", 2)
        if rest not in PER_LAYER:
            raise KeyError(f"no HF name for {key!r}")
        return f"model.layers.{idx}.{PER_LAYER[rest]}"
    raise KeyError(f"no HF name for {key!r}")


def hf_config(mc: ModelConfig, dtype: str = "float32") -> dict:
    return {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": mc.d_model,
        "intermediate_size": mc.ffn_hidden,
        "num_hidden_layers": mc.n_layers,
        "num_attention_heads": mc.n_heads,
        "num_key_value_heads": mc.n_kv_heads,
        "head_dim": mc.head_dim,
        "max_position_embeddings": mc.context_len,
        "rms_norm_eps": mc.norm_eps,
        "rope_theta": mc.rope_theta,
        "vocab_size": mc.vocab_size,
        "tie_word_embeddings": mc.tied_embeddings,
        "hidden_act": "silu",
        "attention_bias": False,
        "mlp_bias": False,
        "torch_dtype": dtype,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
    }


def check_tokenizer_matches(mc, tokenizer_path) -> None:
    """Refuse to export a model paired with a tokenizer it was not trained on.

    Nothing else catches this. The shapes are compatible, the file writes
    cleanly and the result loads in llama.cpp -- it simply generates the wrong
    characters, because token id 4013 in the vocabulary is not what the
    embedding at row 4013 learned. It is exactly the sort of mismatch a
    pipeline produces when a tokenizer is rebuilt without retraining.
    """
    import json as _json
    from pathlib import Path as _Path

    path = _Path(tokenizer_path)
    if not path.exists():
        raise FileNotFoundError(f"no tokenizer at {path}")
    vocab = len(_json.loads(path.read_text(encoding="utf-8"))["model"]["vocab"])
    if vocab > mc.vocab_size:
        raise ValueError(
            f"tokenizer has {vocab} tokens but the model was trained with "
            f"{mc.vocab_size}; these do not belong together"
        )


def convert(ckpt_path: Path, out_dir: Path, dtype: str = "float32",
            tokenizer_path: Path | None = None) -> dict:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mc = ModelConfig(**state["model_config"])
    if tokenizer_path:
        check_tokenizer_matches(mc, tokenizer_path)
    weights = state["model"]

    torch_dtype = {"float32": torch.float32, "float16": torch.float16,
                   "bfloat16": torch.bfloat16}[dtype]

    out: dict[str, torch.Tensor] = {}
    for key, tensor in weights.items():
        if key.startswith("rope_"):
            continue                    # buffers, recomputed at load time
        if mc.tied_embeddings and key == "lm_head.weight":
            continue                    # tie_word_embeddings covers it
        out[rename(key)] = tensor.to(torch_dtype).contiguous().clone()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(out, str(out_dir / "model.safetensors"), metadata={"format": "pt"})
    (out_dir / "config.json").write_text(
        json.dumps(hf_config(mc, dtype), indent=2), encoding="utf-8")

    if tokenizer_path and Path(tokenizer_path).exists():
        import shutil
        shutil.copy(tokenizer_path, out_dir / "tokenizer.json")
        (out_dir / "tokenizer_config.json").write_text(json.dumps({
            "tokenizer_class": "PreTrainedTokenizerFast",
            "bos_token": "<|bos|>",
            "eos_token": "<|eos|>",
            "pad_token": "<|pad|>",
            "model_max_length": mc.context_len,
        }, indent=2), encoding="utf-8")

    return {
        "tensors": len(out),
        "parameters": sum(t.numel() for t in out.values()),
        "dtype": dtype,
        "out_dir": str(out_dir),
        "tied_embeddings": mc.tied_embeddings,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("export/model"))
    p.add_argument("--dtype", default="float32", choices=("float32", "float16", "bfloat16"))
    p.add_argument("--tokenizer", type=Path, default=Path("tokenizer.json"))
    args = p.parse_args(argv)
    print(json.dumps(convert(args.ckpt, args.out, args.dtype, args.tokenizer), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
