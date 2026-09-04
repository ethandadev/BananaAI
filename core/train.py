"""Pretraining loop.

    python -m core.train --data data/tokenized --out runs/base
    python -m core.train --data data/tokenized --out runs/base --resume

Single GPU, bf16 autocast with fp32 master weights, gradient accumulation to
reach a 0.5M-token batch. Everything needed to resume exactly -- model,
optimizer, scaler, step, and both RNG streams -- goes in the checkpoint,
because a run this long will be interrupted at least once.

The loss watchdog matters more than it looks. bf16 pretraining diverges from a
single bad batch, and without a halt the run keeps burning GPU hours producing
NaNs. Stopping on a spike means resuming from the last good checkpoint rather
than from zero.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch

from .config import ModelConfig, TrainConfig
from .dataset import TokenDataset, load_meta
from .hardware import LADDER, Device, detect, recommend
from .model import Transformer
from .sample import SamplingConfig, generate


def lr_at(step: int, tc: TrainConfig) -> float:
    """Linear warmup, then cosine decay to min_lr."""
    if step < tc.warmup_steps:
        return tc.peak_lr * (step + 1) / tc.warmup_steps
    if step >= tc.total_steps:
        return tc.min_lr
    progress = (step - tc.warmup_steps) / max(1, tc.total_steps - tc.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return tc.min_lr + (tc.peak_lr - tc.min_lr) * cosine


def build_optimizer(model: torch.nn.Module, tc: TrainConfig, device: str):
    """Weight-decay the matmuls, not the norms.

    Decaying a RMSNorm gain pulls it toward zero, which scales the whole
    residual stream down -- the opposite of what the parameter is for.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)

    groups = [
        {"params": decay, "weight_decay": tc.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    fused = device.startswith("cuda") and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
    return torch.optim.AdamW(
        groups, lr=tc.peak_lr, betas=(tc.beta1, tc.beta2), eps=tc.eps,
        **({"fused": True} if fused else {}),
    ), len(decay), len(no_decay)


def pick_device(requested=None) -> Device:
    """Resolve to a Device. Accepts a Device, a backend name, or None."""
    if isinstance(requested, Device):
        return requested
    return detect(requested)


def autocast_ctx(device: str, dtype: torch.dtype):
    if device.startswith("cuda"):
        return torch.autocast("cuda", dtype=dtype)
    if dtype is torch.bfloat16 and torch.cpu.is_available():
        return torch.autocast("cpu", dtype=torch.bfloat16)
    return nullcontext()


@torch.no_grad()
def evaluate(model, dataset: TokenDataset, batch_size: int, batches: int, device: str) -> float:
    model.eval()
    losses = []
    for x, y in dataset.deterministic_batches(batch_size, batches, device):
        _, loss, _ = model(x, targets=y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else float("nan")


class Checkpointer:
    def __init__(self, out_dir: Path, keep_last: int = 3):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last

    def save(self, state: dict, step: int, is_best: bool = False) -> Path:
        path = self.out_dir / f"step-{step:07d}.pt"
        tmp = path.with_suffix(".pt.tmp")
        torch.save(state, tmp)
        tmp.replace(path)          # atomic: a killed process cannot leave a half-file
        if is_best:
            best = self.out_dir / "best.pt"
            torch.save(state, best.with_suffix(".pt.tmp"))
            best.with_suffix(".pt.tmp").replace(best)
        self._prune()
        return path

    def _prune(self) -> None:
        ckpts = sorted(self.out_dir.glob("step-*.pt"))
        for old in ckpts[: -self.keep_last] if len(ckpts) > self.keep_last else []:
            old.unlink()

    def latest(self) -> Optional[Path]:
        ckpts = sorted(self.out_dir.glob("step-*.pt"))
        return ckpts[-1] if ckpts else None


def train(
    mc: ModelConfig,
    tc: TrainConfig,
    data_dir: Path,
    out_dir: Path,
    device: Optional[str] = None,
    compile_model: bool = True,
    resume: bool = False,
    seed: int = 1337,
    log_every: Optional[int] = None,
    spike_factor: float = 3.0,
    tokenizer_path: Optional[Path] = None,
) -> dict:
    hw = pick_device(device)
    device = hw.torch_device
    torch.manual_seed(seed)
    if hw.kind == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # bf16 needs Ampere or newer. Everything else trains in fp32, which costs
    # memory but is correct -- silently using bf16 on a card without it
    # produces NaNs a few hundred steps in.
    dtype = torch.bfloat16 if hw.supports_bf16 else torch.float32
    accum = tc.grad_accum_steps(mc.context_len)
    log_every = log_every or tc.log_every

    meta = load_meta(data_dir)
    if meta["vocab_size"] > mc.vocab_size:
        raise ValueError(
            f"tokenizer vocab {meta['vocab_size']} exceeds model vocab {mc.vocab_size}"
        )

    train_ds = TokenDataset(Path(data_dir) / "train.bin", mc.context_len, seed)
    val_ds = TokenDataset(Path(data_dir) / "val.bin", mc.context_len, seed + 1)

    model = Transformer(mc).to(device)
    optimizer, n_decay, n_nodecay = build_optimizer(model, tc, device)

    ckpt = Checkpointer(out_dir)
    start_step, best_val = 0, float("inf")
    history: list[dict] = []

    if resume:
        latest = ckpt.latest()
        if latest is None:
            print(f"[resume] nothing in {out_dir}, starting fresh", file=sys.stderr)
        else:
            state = torch.load(latest, map_location=device, weights_only=False)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            start_step = state["step"] + 1
            best_val = state.get("best_val", float("inf"))
            history = state.get("history", [])
            torch.set_rng_state(state["cpu_rng"])
            train_ds.rng = state.get("data_rng", train_ds.rng)
            print(f"[resume] {latest.name} at step {start_step}", file=sys.stderr)

    raw_model = model
    if compile_model:
        try:
            model = torch.compile(model)
        except Exception as e:                      # noqa: BLE001
            print(f"[compile] unavailable ({type(e).__name__}), running eager", file=sys.stderr)

    tokenizer = None
    if tokenizer_path and Path(tokenizer_path).exists():
        from .tokenizer import load as load_tok
        tokenizer = load_tok(tokenizer_path)

    writer = _tensorboard(out_dir)
    stopping = {"now": False}

    def _stop(signum, frame):   # noqa: ARG001
        print("\n[signal] finishing this step, then checkpointing", file=sys.stderr)
        stopping["now"] = True

    signal.signal(signal.SIGINT, _stop)

    (Path(out_dir) / "config.json").write_text(
        json.dumps({"model": asdict(mc), "train": asdict(tc), "seed": seed}, indent=2),
        encoding="utf-8",
    )

    print(
        f"[setup] {hw.describe()}, {raw_model.num_params() / 1e6:.1f}M params, "
        f"{n_decay} decayed / {n_nodecay} not, accum {accum}, "
        f"{tc.tokens_per_step:,} tokens/step, {len(train_ds):,} train tokens",
        file=sys.stderr,
    )

    model.train()
    recent_losses: list[float] = []
    t_start = time.perf_counter()
    tokens_done = start_step * tc.tokens_per_step

    for step in range(start_step, tc.total_steps):
        t0 = time.perf_counter()
        lr = lr_at(step, tc)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(accum):
            x, y = train_ds.get_batch(tc.micro_batch, device)
            with autocast_ctx(device, dtype):
                _, loss, _ = model(x, targets=y)
            (loss / accum).backward()
            total_loss += loss.item() / accum

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        optimizer.step()

        tokens_done += tc.tokens_per_step
        dt = time.perf_counter() - t0

        if not math.isfinite(total_loss):
            print(f"[abort] non-finite loss at step {step}", file=sys.stderr)
            break

        if recent_losses:
            baseline = sum(recent_losses) / len(recent_losses)
            if total_loss > baseline * spike_factor:
                print(
                    f"[abort] loss spiked {baseline:.3f} -> {total_loss:.3f} at step {step}; "
                    f"resume from the last checkpoint with a lower LR",
                    file=sys.stderr,
                )
                break
        recent_losses.append(total_loss)
        if len(recent_losses) > 50:
            recent_losses.pop(0)

        if step % log_every == 0:
            mfu = _mfu(raw_model, tc, dt, hw)
            msg = (
                f"step {step:>6}  loss {total_loss:.4f}  lr {lr:.2e}  "
                f"gnorm {float(grad_norm):.2f}  {dt * 1000:.0f}ms  "
                f"{tc.tokens_per_step / dt / 1e3:.1f}k tok/s"
            )
            if mfu is not None:
                msg += f"  mfu {mfu * 100:.1f}%"
            print(msg, file=sys.stderr)
            history.append({"step": step, "loss": total_loss, "lr": lr})
            if writer:
                writer.add_scalar("train/loss", total_loss, step)
                writer.add_scalar("train/lr", lr, step)
                writer.add_scalar("train/grad_norm", float(grad_norm), step)

        is_best = False
        if step > 0 and step % tc.eval_every == 0:
            val = evaluate(raw_model, val_ds, tc.micro_batch, 20, device)
            is_best = val < best_val
            best_val = min(best_val, val)
            print(f"step {step:>6}  val {val:.4f}{'  (best)' if is_best else ''}", file=sys.stderr)
            if writer:
                writer.add_scalar("val/loss", val, step)

        if tokenizer and step > 0 and step % tc.sample_every == 0:
            _log_sample(raw_model, tokenizer, device, step, writer)

        if (step > 0 and step % tc.ckpt_every == 0) or is_best or stopping["now"]:
            ckpt.save(
                {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "best_val": best_val,
                    "history": history,
                    "model_config": asdict(mc),
                    "train_config": asdict(tc),
                    "cpu_rng": torch.get_rng_state(),
                    "data_rng": train_ds.rng,
                },
                step,
                is_best,
            )

        if stopping["now"]:
            print("[stop] checkpoint written", file=sys.stderr)
            break

    elapsed = time.perf_counter() - t_start
    final = {
        "steps_completed": step + 1 - start_step,
        "tokens_seen": tokens_done,
        "best_val_loss": best_val if math.isfinite(best_val) else None,
        "final_train_loss": recent_losses[-1] if recent_losses else None,
        "hours": round(elapsed / 3600, 3),
    }
    (Path(out_dir) / "summary.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    if writer:
        writer.close()
    return final


def _tensorboard(out_dir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(str(Path(out_dir) / "tb"))
    except Exception:                               # noqa: BLE001
        return None


def _mfu(model, tc: TrainConfig, dt: float, hw: Device) -> Optional[float]:
    """Model FLOPs utilisation: 6ND per token against the device peak.

    Only reported when the device's real peak is known. Dividing by an assumed
    figure produces confident nonsense -- an early version printed 112%.
    """
    if not hw.peak_tflops:
        return None
    flops = 6 * model.num_params() * tc.tokens_per_step
    return (flops / dt) / (hw.peak_tflops * 1e12)


def _log_sample(model, tokenizer, device: str, step: int, writer) -> None:
    prompt = "The most important thing about"
    ids = torch.tensor(
        [tokenizer.encode(prompt, add_special_tokens=False).ids], dtype=torch.long, device=device
    )
    out = generate(model, ids, SamplingConfig(max_new_tokens=48, temperature=0.8, seed=0))
    text = tokenizer.decode(out[0].tolist())
    print(f"[sample {step}] {text[:300]}", file=sys.stderr)
    if writer:
        writer.add_text("sample", text, step)


# The size ladder lives in core/hardware.py so the trainer, the auto-sizer and
# the app all agree on what "small" means. `test` is extra: far too small to be
# useful, but it keeps the CPU test suite fast.
PRESETS = {name: dict(spec) for name, spec in LADDER}
PRESETS["test"] = dict(n_layers=4, d_model=128, n_heads=4, n_kv_heads=2,
                       ffn_hidden=352, context_len=128)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", type=Path, default=Path("data/tokenized"))
    p.add_argument("--out", type=Path, default=Path("runs/base"))
    p.add_argument("--tokenizer", type=Path, default=Path("tokenizer.json"))
    p.add_argument("--preset", choices=sorted(PRESETS),
                   help="model size; omit with --auto to have one chosen for you")
    p.add_argument("--auto", action="store_true",
                   help="size the model to this machine automatically")
    p.add_argument("--max-hours", type=float, default=None,
                   help="with --auto, the wall-clock budget to plan against")
    p.add_argument("--vocab-size", type=int, default=None)
    p.add_argument("--context-len", type=int, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--micro-batch", type=int, default=None)
    p.add_argument("--tokens-per-step", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--ckpt-every", type=int, default=None)
    p.add_argument("--log-every", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args(argv)

    vocab = args.vocab_size
    if vocab is None:
        try:
            vocab = load_meta(args.data)["vocab_size"]
        except FileNotFoundError:
            vocab = ModelConfig.vocab_size

    if args.auto or (args.preset is None and not args.steps):
        plan = recommend(device=detect(args.device), vocab_size=vocab,
                         max_hours=args.max_hours, preset=args.preset)
        print(os.linesep + plan.summary() + os.linesep, file=sys.stderr)
        mc, tc = plan.model, plan.train
        if args.context_len:
            mc.context_len = args.context_len
    else:
        mc_kwargs = dict(PRESETS.get(args.preset, {}))
        mc_kwargs["vocab_size"] = vocab
        if args.context_len:
            mc_kwargs["context_len"] = args.context_len
        mc = ModelConfig(**mc_kwargs)
        tc = TrainConfig()
    for attr, val in (
        ("total_steps", args.steps), ("warmup_steps", args.warmup), ("peak_lr", args.lr),
        ("micro_batch", args.micro_batch), ("tokens_per_step", args.tokens_per_step),
        ("eval_every", args.eval_every), ("ckpt_every", args.ckpt_every),
    ):
        if val is not None:
            setattr(tc, attr, val)
    if args.lr is not None:
        tc.min_lr = args.lr / 10

    summary = train(
        mc, tc, args.data, args.out,
        device=args.device,
        compile_model=not args.no_compile,
        resume=args.resume,
        seed=args.seed,
        log_every=args.log_every,
        tokenizer_path=args.tokenizer,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
