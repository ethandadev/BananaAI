"""Model and training configuration.

Values here are the spec'd 327M target: Chinchilla-ish at 21 tokens/param,
sized to fit a single 32GB RTX 5090 without gradient checkpointing.
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    # --- architecture ---
    n_layers: int = 26
    d_model: int = 1024
    n_heads: int = 16
    n_kv_heads: int = 4          # grouped-query attention: 4x smaller KV cache
    ffn_hidden: int = 2816       # SwiGLU, 8/3 * d_model rounded to a multiple of 128
    vocab_size: int = 32768
    context_len: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    tied_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def n_kv_groups(self) -> int:
        return self.n_heads // self.n_kv_heads

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must divide evenly into n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be a multiple of n_kv_heads")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for RoPE")

    def n_params(self) -> int:
        """Parameter count, matching what the built model actually allocates."""
        embed = self.vocab_size * self.d_model
        attn = self.d_model * self.d_model * 2                       # q, o
        attn += self.d_model * self.n_kv_heads * self.head_dim * 2   # k, v
        ffn = self.d_model * self.ffn_hidden * 3                     # gate, up, down
        norms = self.d_model * 2
        per_layer = attn + ffn + norms
        total = embed + self.n_layers * per_layer + self.d_model     # + final norm
        if not self.tied_embeddings:
            total += embed
        return total


@dataclass
class TrainConfig:
    # --- schedule ---
    total_steps: int = 13_400        # 7.0B tokens at 524288 tokens/step
    warmup_steps: int = 500
    peak_lr: float = 4e-4
    min_lr: float = 4e-5             # cosine floor, 10% of peak

    # --- optimizer ---
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1        # applied to matmul params only
    grad_clip: float = 1.0

    # --- batching ---
    tokens_per_step: int = 524_288   # 256 sequences of 2048
    micro_batch: int = 16            # x 16 grad-accum steps; capped by the logit tensor,
                                     # not the weights: 16*2048 tokens x 32768 vocab
                                     # is 2.1 GB bf16 + 4.3 GB fp32 for the loss alone

    # --- io ---
    ckpt_every: int = 1000
    eval_every: int = 250
    log_every: int = 10
    sample_every: int = 500          # generate from a fixed prompt to watch coherence emerge

    def grad_accum_steps(self, context_len: int) -> int:
        per_micro = self.micro_batch * context_len
        if self.tokens_per_step % per_micro:
            raise ValueError("tokens_per_step must be divisible by micro_batch * context_len")
        return self.tokens_per_step // per_micro
