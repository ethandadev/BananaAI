"""A decoder-only transformer, written from scratch.

The layout deliberately matches Llama (RoPE, RMSNorm, SwiGLU, grouped-query
attention, tied embeddings). Every line here is ours, but keeping the *shape*
conventional is what lets the llama.cpp converter read the weights without a
custom script -- a free win with no cost to quality at this scale.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    """Root-mean-square layer norm. No mean subtraction, no bias."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalise in fp32 regardless of autocast dtype -- the variance of a
        # bf16 sum is where low-precision training quietly loses accuracy.
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x.to(dtype) * self.weight


def build_rope_cache(
    head_dim: int, seq_len: int, theta: float, device=None, dtype=torch.float32
):
    """Precompute the rotary cos/sin tables. Shape: (seq_len, head_dim // 2)."""
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    positions = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(positions, inv_freq)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate the head dimension in halves.

    x is (B, n_heads, T, head_dim); cos/sin are (T, head_dim // 2). This is the
    half-split convention HF Llama uses -- worth matching exactly, since a
    mismatch here produces a model that trains fine and then exports wrong.
    """
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand n_kv_heads up to n_heads for grouped-query attention."""
    if n_rep == 1:
        return x
    b, n_kv, t, hd = x.shape
    return x[:, :, None, :, :].expand(b, n_kv, n_rep, t, hd).reshape(b, n_kv * n_rep, t, hd)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.n_rep = cfg.n_kv_groups
        self.head_dim = cfg.head_dim

        self.wq = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_kv: Optional[tuple] = None,
        use_cache: bool = False,
    ):
        b, t, _ = x.shape

        q = self.wq(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        present = (k, v) if use_cache else None

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # is_causal only holds when query and key share a length. During cached
        # decoding a single query token attends to the whole prefix, which needs
        # no mask at all.
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=(past_kv is None and t > 1)
        )
        out = out.transpose(1, 2).contiguous().view(b, t, -1)
        return self.wo(out), present


class SwiGLU(nn.Module):
    """Gated feed-forward: down(silu(gate(x)) * up(x))."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin, past_kv=None, use_cache=False):
        h, present = self.attn(self.attn_norm(x), cos, sin, past_kv, use_cache)
        x = x + h
        x = x + self.ffn(self.ffn_norm(x))
        return x, present


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tied_embeddings:
            self.lm_head.weight = self.embed.weight

        cos, sin = build_rope_cache(cfg.head_dim, cfg.context_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Scale the residual-path output projections so activation variance does
        # not grow with depth.
        residual_scale = 0.02 / math.sqrt(2 * cfg.n_layers)
        for name, p in self.named_parameters():
            if name.endswith(("attn.wo.weight", "ffn.down.weight")):
                nn.init.normal_(p, mean=0.0, std=residual_scale)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        past_kvs: Optional[list] = None,
        use_cache: bool = False,
    ):
        """Run the model. `targets` must already be shifted by the caller.

        No shift happens here: position i predicts targets[i], so a dataloader
        passes x = tokens[:-1] and y = tokens[1:]. Passing unshifted targets
        trains the model to copy its own input -- and because the embeddings
        are tied, that is easy enough that the loss will drop *below* ln(vocab)
        and look like a suspiciously good start rather than a bug.

        Use -100 in `targets` to mask positions out of the loss (prompt tokens
        during SFT, padding).
        """
        b, t = idx.shape
        start = past_kvs[0][0].shape[2] if past_kvs is not None else 0
        if start + t > self.cfg.context_len:
            raise ValueError(
                f"sequence of {start + t} exceeds context_len {self.cfg.context_len}"
            )

        cos = self.rope_cos[start : start + t]
        sin = self.rope_sin[start : start + t]

        x = self.embed(idx)
        presents = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            past = past_kvs[i] if past_kvs is not None else None
            x, present = block(x, cos, sin, past, use_cache)
            if use_cache:
                presents.append(present)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss, presents

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embed.weight.numel()
        return n
