"""Detect the machine, then size a model that will actually fit on it.

The whole project used to assume one specific GPU. That is fine for one
person and useless for everyone else: the same defaults that keep a 32 GB card
busy will fail to allocate on an 8 GB laptop, and a 96 GB card would sit idle.

So nothing is hardcoded to a device any more. `detect()` reports what is
present, `recommend()` picks the largest model from a ladder that fits inside
a memory budget, and the memory model it uses is written out explicitly below
so its estimates can be checked rather than trusted.
"""

from __future__ import annotations

import ctypes
import math
import platform
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from .config import ModelConfig, TrainConfig

# Dense bf16 throughput, TFLOP/s, matched on a substring of the device name.
# Only used for time estimates and the MFU readout -- a missing entry degrades
# the estimate, never the training itself.
PEAK_BF16 = {
    "5090": 209.0, "4090": 165.2, "4080": 97.5, "4070": 58.0, "4060": 30.0,
    "3090": 71.0, "3080": 59.5, "3070": 40.6, "3060": 25.6,
    "A100": 312.0, "H100": 989.0, "H200": 989.0, "L40": 181.0, "A6000": 155.0,
    "V100": 125.0, "T4": 65.0,
}

# Bytes per parameter held during training: fp32 master weights, fp32
# gradients, and Adam's two fp32 moments. bf16 autocast does not change this --
# the optimizer state stays fp32, which is what keeps the updates stable.
BYTES_PER_PARAM = 16

# Fraction of VRAM left for the allocator, fragmentation, and the driver.
# Going above this is where "it trained for six hours then OOMed" comes from.
SAFETY_MARGIN = 0.80


@dataclass
class Device:
    kind: str                    # 'cuda' | 'mps' | 'cpu'
    name: str
    memory_gb: float
    supports_bf16: bool
    peak_tflops: Optional[float] = None
    compute_capability: Optional[tuple] = None
    notes: list = field(default_factory=list)

    @property
    def torch_device(self) -> str:
        return self.kind

    def describe(self) -> str:
        parts = [self.name, f"{self.memory_gb:.1f} GB"]
        if self.compute_capability:
            parts.append(f"sm_{self.compute_capability[0]}{self.compute_capability[1]}")
        parts.append("bf16" if self.supports_bf16 else "fp32 only")
        return ", ".join(parts)


def peak_for(name: str) -> Optional[float]:
    for key, tflops in PEAK_BF16.items():
        if key.lower() in name.lower():
            return tflops
    return None


def detect(prefer: Optional[str] = None) -> Device:
    """Report the best available device. `prefer` forces a specific backend."""
    import torch

    if prefer == "cpu":
        return _cpu_device()

    if (prefer in (None, "cuda")) and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        cap = (props.major, props.minor)
        notes = []
        # bf16 needs Ampere or newer; older cards must fall back to fp16 or fp32.
        supports_bf16 = cap >= (8, 0)
        if not supports_bf16:
            notes.append("pre-Ampere GPU: bf16 unavailable, training in fp32")
        arch_list = getattr(torch.cuda, "get_arch_list", lambda: [])()
        if arch_list and f"sm_{cap[0]}{cap[1]}" not in arch_list:
            notes.append(
                f"this PyTorch build targets {arch_list} and not sm_{cap[0]}{cap[1]} -- "
                "kernels will fail at launch; install a matching CUDA build"
            )
        return Device(
            kind="cuda",
            name=props.name,
            memory_gb=props.total_memory / 1024**3,
            supports_bf16=supports_bf16,
            peak_tflops=peak_for(props.name),
            compute_capability=cap,
            notes=notes,
        )

    if (prefer in (None, "mps")) and getattr(torch.backends, "mps", None) \
            and torch.backends.mps.is_available():
        # Apple Silicon shares system memory with the GPU. Half of it is a
        # conservative ceiling -- the OS and everything else need the rest.
        total = _system_memory_gb()
        return Device(
            kind="mps",
            name=f"Apple Silicon ({platform.machine()})",
            memory_gb=total / 2,
            supports_bf16=False,     # MPS bf16 support is uneven; fp32 is the safe path
            peak_tflops=None,
            notes=["unified memory: budget is half of system RAM",
                   "training in fp32 -- MPS bf16 coverage is incomplete"],
        )

    return _cpu_device()


def cpu_name() -> str:
    """A readable CPU name.

    platform.processor() returns the marketing name on macOS but a family/model
    string on Windows ("AMD64 Family 26 Model 68 Stepping 0") and often nothing
    at all on Linux, so each platform gets its own lookup.
    """
    system = platform.system()
    try:
        if system == "Windows":
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            with key:
                name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return " ".join(name.split())
        if system == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        if system == "Darwin":
            import subprocess

            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                 capture_output=True, text=True, timeout=5)
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
    except Exception:                              # noqa: BLE001
        pass
    return platform.processor() or platform.machine() or "CPU"


def _cpu_device() -> Device:
    return Device(
        kind="cpu",
        name=cpu_name(),
        memory_gb=_system_memory_gb() / 2,
        supports_bf16=False,
        peak_tflops=None,
        notes=["CPU training is 50-100x slower than a GPU; expect a very small model"],
    )


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


DEFAULT_SYSTEM_MEMORY_GB = 8.0


def _system_memory_gb() -> float:
    """Total physical RAM. Falls back to a conservative guess, never raises."""
    import os

    # POSIX (Linux, macOS)
    if hasattr(os, "sysconf"):
        try:
            if "SC_PHYS_PAGES" in os.sysconf_names:
                return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
        except (ValueError, OSError):
            pass

    # Windows
    if hasattr(ctypes, "windll"):
        try:
            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.ullTotalPhys / 1024**3
        except (OSError, AttributeError):
            pass

    return DEFAULT_SYSTEM_MEMORY_GB


# --------------------------------------------------------------------------
# the model ladder
# --------------------------------------------------------------------------

# Each rung is a complete, sensible architecture rather than one dimension
# scaled arbitrarily: width, depth and head count move together, because a
# very deep narrow model and a very wide shallow one both train badly.
LADDER: tuple[tuple[str, dict], ...] = (
    ("nano",  dict(n_layers=6,  d_model=256,  n_heads=4,  n_kv_heads=2, ffn_hidden=704,  context_len=512)),
    ("micro", dict(n_layers=8,  d_model=384,  n_heads=6,  n_kv_heads=2, ffn_hidden=1024, context_len=512)),
    ("mini",  dict(n_layers=10, d_model=512,  n_heads=8,  n_kv_heads=2, ffn_hidden=1408, context_len=1024)),
    ("small", dict(n_layers=12, d_model=768,  n_heads=12, n_kv_heads=4, ffn_hidden=2048, context_len=1024)),
    ("base",  dict(n_layers=26, d_model=1024, n_heads=16, n_kv_heads=4, ffn_hidden=2816, context_len=2048)),
    ("large", dict(n_layers=24, d_model=1536, n_heads=16, n_kv_heads=4, ffn_hidden=4096, context_len=2048)),
    ("xl",    dict(n_layers=24, d_model=2048, n_heads=16, n_kv_heads=8, ffn_hidden=5632, context_len=2048)),
)

TOKENS_PER_PARAM = 21          # Chinchilla-ish
TARGET_TOKENS_PER_STEP = 524_288

# Default time budget: a long weekend. Selection is time-aware because a
# fully-trained small model beats an undertrained large one at every scale,
# and picking purely on memory recommends runs that take a month.
DEFAULT_MAX_HOURS = 72.0

# Fallback throughput when the device is not in PEAK_BF16, so a time estimate
# always exists. Deliberately conservative -- overestimating speed produces
# recommendations that never finish.
FALLBACK_PEAK = {"cuda": 30.0, "mps": 7.0, "cpu": 0.4}


def effective_peak(device: "Device") -> float:
    """Peak TFLOP/s to plan against, measured or assumed."""
    return device.peak_tflops or FALLBACK_PEAK.get(device.kind, 1.0)


def hours_to_train(mc: ModelConfig, device: "Device", mfu: float,
                   tokens: Optional[int] = None) -> float:
    """Wall-clock for a full Chinchilla-ish run, at 6ND FLOPs per token."""
    tokens = tokens if tokens is not None else mc.n_params() * TOKENS_PER_PARAM
    flops = 6 * mc.n_params() * tokens
    return flops / (effective_peak(device) * 1e12 * mfu) / 3600


def training_memory_gb(mc: ModelConfig, micro_batch: int, bf16: bool) -> dict:
    """Estimated peak memory, broken down so the number can be argued with.

    The logit tensor is the term people forget. At a 32k vocabulary it is
    larger than the entire model, and it is what actually caps the batch size.
    """
    params = mc.n_params()
    weights = params * BYTES_PER_PARAM

    tokens = micro_batch * mc.context_len
    # Logits in compute dtype, plus the fp32 copy cross_entropy materialises.
    logits = tokens * mc.vocab_size * (2 + 4 if bf16 else 4 + 4)
    # Residual stream and attention workspace across layers, measured
    # empirically to sit near this multiple of the widest activation.
    activations = tokens * mc.d_model * mc.n_layers * (2 if bf16 else 4) * 2.5

    total = weights + logits + activations
    return {
        "weights_gb": weights / 1024**3,
        "logits_gb": logits / 1024**3,
        "activations_gb": activations / 1024**3,
        "total_gb": total / 1024**3,
    }


def largest_micro_batch(mc: ModelConfig, budget_gb: float, bf16: bool) -> int:
    """Biggest power-of-two micro-batch that fits, or 0 if even one will not."""
    for mb in (64, 48, 32, 24, 16, 12, 8, 6, 4, 3, 2, 1):
        if training_memory_gb(mc, mb, bf16)["total_gb"] <= budget_gb:
            return mb
    return 0


@dataclass
class Plan:
    device: Device
    preset: str
    model: ModelConfig
    train: TrainConfig
    memory: dict
    estimated_hours: Optional[float]
    warnings: list = field(default_factory=list)

    def summary(self) -> str:
        n = self.model.n_params()
        lines = [
            f"device     {self.device.describe()}",
            f"preset     {self.preset}  ({n / 1e6:.0f}M parameters)",
            f"context    {self.model.context_len}",
            f"batch      {self.train.micro_batch} x "
            f"{self.train.grad_accum_steps(self.model.context_len)} accum "
            f"= {self.train.tokens_per_step:,} tokens/step",
            f"steps      {self.train.total_steps:,} "
            f"({self.train.total_steps * self.train.tokens_per_step / 1e9:.2f}B tokens)",
            f"memory     {self.memory['total_gb']:.1f} GB of "
            f"{self.device.memory_gb:.1f} GB "
            f"(weights {self.memory['weights_gb']:.1f} + "
            f"logits {self.memory['logits_gb']:.1f} + "
            f"activations {self.memory['activations_gb']:.1f})",
        ]
        if self.estimated_hours is not None:
            lines.append(f"estimate   {self.estimated_hours:.1f} hours")
        for w in self.warnings:
            lines.append(f"warning    {w}")
        return "\n".join(lines)


def recommend(
    device: Optional[Device] = None,
    vocab_size: int = 32768,
    max_hours: Optional[float] = None,
    preset: Optional[str] = None,
    mfu: float = 0.35,
) -> Plan:
    """Pick the largest rung that fits, then size the batch and schedule to it."""
    device = device or detect()
    budget = device.memory_gb * SAFETY_MARGIN
    bf16 = device.supports_bf16
    warnings = list(device.notes)

    # Device.notes are filled in by detect(). A Device built any other way --
    # from a config file, or by the app -- carries none, so the warnings that
    # depend only on the device kind are raised here instead.
    if device.kind == "cpu" and not any("slower" in w for w in warnings):
        warnings.append(
            "CPU training is 50-100x slower than a GPU; expect a very small model")
    if device.kind == "mps" and not any("fp32" in w for w in warnings):
        warnings.append("MPS runs in fp32 -- bf16 coverage is incomplete")

    rungs = list(LADDER)
    if preset:
        rungs = [r for r in rungs if r[0] == preset]
        if not rungs:
            raise KeyError(f"unknown preset {preset!r} -- "
                           f"choose from {', '.join(n for n, _ in LADDER)}")

    hours_budget = max_hours if max_hours is not None else DEFAULT_MAX_HOURS

    chosen = None
    memory_rejected = []
    time_rejected = []
    for name, spec in reversed(rungs):
        mc = ModelConfig(vocab_size=vocab_size, **spec)
        mb = largest_micro_batch(mc, budget, bf16)
        if not mb:
            memory_rejected.append(name)
            continue
        if preset is None and hours_to_train(mc, device, mfu) > hours_budget:
            time_rejected.append(name)
            continue
        chosen = (name, mc, mb)
        break

    if chosen is None:
        # Nothing fits both limits. Fall back to the smallest rung and say so,
        # rather than recommending a run that cannot finish.
        name, spec = rungs[0]
        mc = ModelConfig(vocab_size=vocab_size, **spec)
        mb = largest_micro_batch(mc, budget, bf16) or 1
        chosen = (name, mc, mb)
        if memory_rejected and not time_rejected:
            warnings.append(
                f"even {name} needs about "
                f"{training_memory_gb(mc, 1, bf16)['total_gb']:.1f} GB and only "
                f"{budget:.1f} GB is usable -- training will likely run out of memory"
            )
        else:
            warnings.append(
                f"nothing larger than {name} trains inside {hours_budget:g} hours on "
                f"this device -- raise the time budget for a bigger model"
            )
    elif time_rejected:
        warnings.append(
            f"{time_rejected[-1]} would fit in memory but needs "
            f"{hours_to_train(ModelConfig(vocab_size=vocab_size, **dict(LADDER)[time_rejected[-1]]), device, mfu):.0f} "
            f"hours; {chosen[0]} was chosen to stay inside {hours_budget:g}"
        )

    name, mc, micro_batch = chosen
    if preset and micro_batch and name == preset:
        need = training_memory_gb(mc, micro_batch, bf16)["total_gb"]
        if need > budget:
            warnings.append(f"{preset} needs {need:.1f} GB but only {budget:.1f} GB is usable")

    # Round the step size down to something the micro-batch divides evenly.
    per_micro = micro_batch * mc.context_len
    accum = max(1, round(TARGET_TOKENS_PER_STEP / per_micro))
    tokens_per_step = per_micro * accum

    total_tokens = mc.n_params() * TOKENS_PER_PARAM
    steps = max(100, int(total_tokens / tokens_per_step))

    hours = hours_to_train(mc, device, mfu, tokens=steps * tokens_per_step)
    if max_hours is not None and hours > max_hours:
        # Only reachable when a preset was forced; otherwise selection already
        # respected the budget.
        steps = max(100, int(steps * max_hours / hours))
        hours = hours_to_train(mc, device, mfu, tokens=steps * tokens_per_step)
        warnings.append(
            f"shortened to {steps:,} steps to fit {max_hours:g} hours -- "
            f"below the {TOKENS_PER_PARAM} tokens/parameter this size wants"
        )
    if device.peak_tflops is None:
        warnings.append(
            f"{device.name} is not in the throughput table; the time estimate "
            f"assumes {effective_peak(device):g} TFLOP/s and may be well off")

    tc = TrainConfig(
        total_steps=steps,
        warmup_steps=max(10, min(500, steps // 25)),
        micro_batch=micro_batch,
        tokens_per_step=tokens_per_step,
        ckpt_every=max(50, steps // 20),
        eval_every=max(25, steps // 40),
    )
    return Plan(
        device=device,
        preset=name,
        model=mc,
        train=tc,
        memory=training_memory_gb(mc, micro_batch, bf16),
        estimated_hours=hours,
        warnings=warnings,
    )


def main(argv=None) -> int:
    """Print the plan for this machine.

        python -m core.hardware
        python -m core.hardware --max-hours 8
        python -m core.hardware --preset small --device cpu
    """
    import argparse
    import json

    p = argparse.ArgumentParser(description="Show what this machine can train")
    p.add_argument("--device", choices=("cuda", "mps", "cpu"), default=None)
    p.add_argument("--preset", default=None,
                   help=f"force a rung: {', '.join(n for n, _ in LADDER)}")
    p.add_argument("--max-hours", type=float, default=None)
    p.add_argument("--vocab", type=int, default=32768)
    p.add_argument("--mfu", type=float, default=0.35, help="assumed FLOP utilisation")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    plan = recommend(
        device=detect(args.device), vocab_size=args.vocab,
        max_hours=args.max_hours, preset=args.preset, mfu=args.mfu,
    )

    if args.json:
        from dataclasses import asdict
        print(json.dumps({
            "device": asdict(plan.device), "preset": plan.preset,
            "model": asdict(plan.model), "train": asdict(plan.train),
            "memory": plan.memory, "estimated_hours": plan.estimated_hours,
            "warnings": plan.warnings,
        }, indent=2))
    else:
        print()
        print(plan.summary())
        print()
        print("  train with:  python -m core.train --preset " + plan.preset)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
