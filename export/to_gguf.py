"""Write a GGUF file directly from a training checkpoint.

    python -m export.to_gguf --ckpt runs/dpo/dpo.pt --tokenizer tokenizer.json \
        --out export/model-f16.gguf

Writing GGUF here rather than shelling out to llama.cpp's converter keeps the
export self-contained and testable -- the result can be read back and verified
without cloning a C++ project.

This produces F16 (or F32). Quantisation is llama.cpp's job:

    llama-quantize export/model-f16.gguf export/model-Q8_0.gguf   Q8_0
    llama-quantize export/model-f16.gguf export/model-Q4_K_M.gguf Q4_K_M
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gguf
import numpy as np
import torch

from core.config import ModelConfig

# our name -> GGUF llama name
GGUF_DIRECT = {
    "embed.weight": "token_embd.weight",
    "norm.weight": "output_norm.weight",
    "lm_head.weight": "output.weight",
}
GGUF_PER_LAYER = {
    "attn_norm.weight": "attn_norm.weight",
    "attn.wq.weight": "attn_q.weight",
    "attn.wk.weight": "attn_k.weight",
    "attn.wv.weight": "attn_v.weight",
    "attn.wo.weight": "attn_output.weight",
    "ffn_norm.weight": "ffn_norm.weight",
    "ffn.gate.weight": "ffn_gate.weight",
    "ffn.up.weight": "ffn_up.weight",
    "ffn.down.weight": "ffn_down.weight",
}


def gguf_name(key: str) -> str:
    if key in GGUF_DIRECT:
        return GGUF_DIRECT[key]
    if key.startswith("blocks."):
        _, idx, rest = key.split(".", 2)
        if rest not in GGUF_PER_LAYER:
            raise KeyError(f"no GGUF name for {key!r}")
        return f"blk.{idx}.{GGUF_PER_LAYER[rest]}"
    raise KeyError(f"no GGUF name for {key!r}")


def tokenizer_arrays(tokenizer_path: Path):
    """Extract the vocabulary and merges in the order GGUF expects.

    Tokens must be written in id order -- GGUF stores a flat list and indexes
    it by token id, so an ordering mistake here silently produces a model that
    generates the wrong characters.
    """
    raw = json.loads(Path(tokenizer_path).read_text(encoding="utf-8"))
    vocab = raw["model"]["vocab"]
    merges = raw["model"].get("merges", [])

    tokens = [None] * len(vocab)
    for token, idx in vocab.items():
        tokens[idx] = token
    if any(t is None for t in tokens):
        missing = [i for i, t in enumerate(tokens) if t is None]
        raise ValueError(f"vocabulary has gaps at ids {missing[:5]}")

    added = {a["content"] for a in raw.get("added_tokens", [])}
    # 1 = CONTROL, 4 = USER_DEFINED in GGUF's token-type enum; everything the
    # BPE learned is a normal token (0).
    types = [
        gguf.TokenType.CONTROL if t in added else gguf.TokenType.NORMAL
        for t in tokens
    ]

    if merges and isinstance(merges[0], list):
        merges = [" ".join(m) for m in merges]
    return tokens, types, merges


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


def convert(
    ckpt_path: Path,
    tokenizer_path: Path,
    out_path: Path,
    dtype: str = "float16",
    name: str = "bananaai-327m",
) -> dict:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mc = ModelConfig(**state["model_config"])
    check_tokenizer_matches(mc, tokenizer_path)
    weights = state["model"]

    np_dtype = {"float16": np.float16, "float32": np.float32}[dtype]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(out_path), "llama")

    writer.add_name(name)
    writer.add_context_length(mc.context_len)
    writer.add_embedding_length(mc.d_model)
    writer.add_block_count(mc.n_layers)
    writer.add_feed_forward_length(mc.ffn_hidden)
    writer.add_head_count(mc.n_heads)
    writer.add_head_count_kv(mc.n_kv_heads)
    writer.add_layer_norm_rms_eps(mc.norm_eps)
    writer.add_rope_freq_base(mc.rope_theta)
    writer.add_rope_dimension_count(mc.head_dim)
    writer.add_file_type(gguf.GGMLQuantizationType.F16 if dtype == "float16"
                         else gguf.GGMLQuantizationType.F32)

    tokens, types, merges = tokenizer_arrays(tokenizer_path)
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("default")
    writer.add_token_list(tokens)
    writer.add_token_types(types)
    if merges:
        writer.add_token_merges(merges)
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)
    writer.add_pad_token_id(0)
    writer.add_add_bos_token(True)
    writer.add_add_eos_token(False)

    written = 0
    for key, tensor in weights.items():
        if key.startswith("rope_"):
            continue
        if mc.tied_embeddings and key == "lm_head.weight":
            continue        # llama.cpp falls back to token_embd when output is absent
        array = tensor.to(torch.float32).numpy().astype(np_dtype)
        writer.add_tensor(gguf_name(key), array)
        written += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    return {
        "out": str(out_path),
        "tensors": written,
        "vocab_size": len(tokens),
        "merges": len(merges),
        "dtype": dtype,
        "megabytes": round(out_path.stat().st_size / 1024**2, 2),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, default=Path("tokenizer.json"))
    p.add_argument("--out", type=Path, default=Path("export/model-f16.gguf"))
    p.add_argument("--dtype", default="float16", choices=("float16", "float32"))
    p.add_argument("--name", default="bananaai-327m")
    args = p.parse_args(argv)
    print(json.dumps(convert(args.ckpt, args.tokenizer, args.out, args.dtype, args.name), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
