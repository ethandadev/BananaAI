"""One config file that every stage reads.

    bananaai.toml

TOML rather than JSON because people edit this by hand and comments matter.
Reading uses stdlib tomllib; writing uses the small serialiser below, which
covers exactly the value types this schema uses and refuses anything else
rather than emitting TOML that will not round-trip.

Precedence, lowest to highest: built-in defaults, the config file, environment
variables, then command-line flags. Every layer is optional, so someone who
never creates a config file gets working defaults and someone who wants to
control everything can.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

try:
    import tomllib
except ModuleNotFoundError:                        # Python 3.10
    tomllib = None

CONFIG_NAME = "bananaai.toml"
ENV_PREFIX = "BANANAAI_"

DEFAULTS: dict[str, dict[str, Any]] = {
    "project": {
        "name": "bananaai",
        "description": "a language model trained from scratch",
    },
    "hardware": {
        "device": "auto",           # auto | cuda | mps | cpu
        "max_hours": 72.0,          # time budget the auto-sizer plans against
        "mfu": 0.35,                # assumed FLOP utilisation for estimates
        "compile": True,            # torch.compile; ~30% throughput when it works
    },
    "model": {
        "preset": "auto",           # auto, or a rung: nano micro mini small base large xl
        "vocab_size": 32768,
        "context_len": 0,           # 0 keeps the preset's own context length
    },
    "data": {
        "out_dir": "data/processed",
        "tokenized_dir": "data/tokenized",
        "custom_dir": "",           # a folder of your own documents
        "custom_share": 0.0,        # fraction of the corpus drawn from custom_dir
        "val_tokens": 10_000_000,
        "dedupe_threshold": 0.8,
        "dedupe_window": 50_000,
        "limit_per_source": 0,      # 0 means no limit
    },
    "train": {
        "out_dir": "runs/base",
        "seed": 1337,
        "peak_lr": 0.0,             # 0 lets the preset decide
        "grad_clip": 1.0,
        "spike_factor": 3.0,
        "ckpt_every": 0,            # 0 lets the auto-planner decide
        # All 0 = let the planner size the batch schedule. Setting them is how
        # you get a genuinely quick run: shrinking the step count alone still
        # leaves half a million tokens per step.
        "total_steps": 0,
        "micro_batch": 0,
        "tokens_per_step": 0,
    },
    "sft": {
        "enabled": True,
        "out_dir": "runs/sft",
        "data": "data/sft/train.jsonl",
        "target": 50_000,
        "epochs": 3,
        "batch_size": 8,
        "lr": 1e-5,
    },
    "dpo": {
        "enabled": True,
        "out_dir": "runs/dpo",
        "data": "data/dpo/prefs.jsonl",
        "target": 20_000,
        "beta": 0.1,
        "epochs": 1,
        "batch_size": 4,
        "lr": 5e-7,
    },
    "export": {
        # Not "export": that is the name of the Python package that does the
        # exporting, and writing artifacts there shadows the module.
        "out_dir": "artifacts",
        "safetensors": True,
        "gguf": True,
        "dtype": "float16",
    },
    "app": {
        "temperature": 0.8,
        "max_new_tokens": 256,
        "top_p": 0.95,
        "top_k": 50,
        "repetition_penalty": 1.1,
    },
}


class ConfigError(ValueError):
    pass


# --------------------------------------------------------------------------
# a deliberately small TOML writer
# --------------------------------------------------------------------------


def _format(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format(v) for v in value) + "]"
    raise ConfigError(
        f"cannot serialise {type(value).__name__} to TOML; this schema supports "
        "booleans, numbers, strings and flat lists of them"
    )


def dumps(config: dict) -> str:
    """Serialise a two-level config (sections of scalars) to TOML."""
    lines = [
        "# BananaAI configuration.",
        "# Every value here can be overridden by a command-line flag.",
        "",
    ]
    for section, values in config.items():
        if not isinstance(values, dict):
            raise ConfigError(f"top-level key {section!r} must be a table")
        lines.append(f"[{section}]")
        for key, value in values.items():
            lines.append(f"{key} = {_format(value)}")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def _deep_merge(base: dict, override: dict) -> dict:
    out = {k: dict(v) if isinstance(v, dict) else v for k, v in base.items()}
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _coerce(current: Any, incoming: str) -> Any:
    """Environment variables arrive as strings; match the default's type."""
    if isinstance(current, bool):
        lowered = incoming.strip().lower()
        if lowered in ("1", "true", "yes", "on"):
            return True
        if lowered in ("0", "false", "no", "off"):
            return False
        raise ConfigError(f"expected a boolean, got {incoming!r}")
    if isinstance(current, int) and not isinstance(current, bool):
        return int(incoming)
    if isinstance(current, float):
        return float(incoming)
    return incoming


def from_env(config: dict, environ: Optional[dict] = None) -> dict:
    """Apply BANANAAI_SECTION_KEY overrides.

    Unknown variables are ignored rather than rejected -- the environment is
    shared with everything else on the machine and is not ours to police.
    """
    environ = os.environ if environ is None else environ
    out = {k: dict(v) for k, v in config.items()}
    for name, raw in environ.items():
        if not name.startswith(ENV_PREFIX):
            continue
        remainder = name[len(ENV_PREFIX):].lower()
        for section in out:
            prefix = section + "_"
            if remainder.startswith(prefix):
                key = remainder[len(prefix):]
                if key in out[section]:
                    out[section][key] = _coerce(out[section][key], raw)
                break
    return out


def find(start: Optional[Path] = None) -> Optional[Path]:
    """Search upward from `start` for a config file, like git does for .git."""
    here = Path(start or Path.cwd()).resolve()
    for directory in (here, *here.parents):
        candidate = directory / CONFIG_NAME
        if candidate.is_file():
            return candidate
    return None


def load(path: Optional[Path] = None, use_env: bool = True) -> "Settings":
    """Defaults, then the file if there is one, then the environment."""
    config = {k: dict(v) for k, v in DEFAULTS.items()}

    path = Path(path) if path else find()
    if path and path.is_file():
        if tomllib is None:
            raise ConfigError(
                "reading a config file needs Python 3.11+ for tomllib; "
                "upgrade or delete " + str(path)
            )
        try:
            parsed = tomllib.loads(path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as e:
            raise ConfigError(f"{path}: {e}") from e
        unknown = set(parsed) - set(DEFAULTS)
        if unknown:
            raise ConfigError(
                f"{path}: unknown section(s) {', '.join(sorted(unknown))}; "
                f"valid sections are {', '.join(DEFAULTS)}"
            )
        for section, values in parsed.items():
            stray = set(values) - set(DEFAULTS[section])
            if stray:
                raise ConfigError(
                    f"{path}: unknown key(s) in [{section}]: {', '.join(sorted(stray))}"
                )
        config = _deep_merge(config, parsed)

    if use_env:
        config = from_env(config)
    return Settings(config, source=path)


@dataclass
class Settings:
    data: dict
    source: Optional[Path] = None

    def __getitem__(self, section: str) -> dict:
        return self.data[section]

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self.data.get(section, {}).get(key, default)

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.write_text(dumps(self.data), encoding="utf-8")
        return path

    def validate(self) -> list[str]:
        """Return a list of problems. Empty means the config is usable."""
        problems = []
        hw = self.data["hardware"]
        if hw["device"] not in ("auto", "cuda", "mps", "cpu"):
            problems.append(f"hardware.device must be auto/cuda/mps/cpu, not {hw['device']!r}")
        if hw["max_hours"] <= 0:
            problems.append("hardware.max_hours must be positive")
        if not 0 < hw["mfu"] <= 1:
            problems.append("hardware.mfu must be between 0 and 1")

        model = self.data["model"]
        from .hardware import LADDER
        valid = {"auto", *(n for n, _ in LADDER)}
        if model["preset"] not in valid:
            problems.append(
                f"model.preset must be one of {', '.join(sorted(valid))}, "
                f"not {model['preset']!r}")
        if model["vocab_size"] > 65535:
            problems.append("model.vocab_size above 65535 will not fit the uint16 token bins")
        if model["vocab_size"] < 256:
            problems.append("model.vocab_size below 256 cannot cover a byte-level alphabet")

        data = self.data["data"]
        if not 0.0 <= data["custom_share"] <= 1.0:
            problems.append("data.custom_share must be between 0 and 1")
        if data["custom_share"] > 0 and not data["custom_dir"]:
            problems.append("data.custom_share is set but data.custom_dir is empty")
        if data["custom_dir"] and not Path(data["custom_dir"]).exists():
            problems.append(f"data.custom_dir does not exist: {data['custom_dir']}")
        if not 0 < data["dedupe_threshold"] <= 1:
            problems.append("data.dedupe_threshold must be between 0 and 1")

        if self.data["export"]["dtype"] not in ("float16", "float32", "bfloat16"):
            problems.append("export.dtype must be float16, float32 or bfloat16")
        return problems

    def device_preference(self) -> Optional[str]:
        pref = self.data["hardware"]["device"]
        return None if pref == "auto" else pref

    def preset_preference(self) -> Optional[str]:
        preset = self.data["model"]["preset"]
        return None if preset == "auto" else preset

    def plan(self):
        """Build a hardware Plan from this configuration."""
        from .hardware import detect, recommend, training_memory_gb

        plan = recommend(
            device=detect(self.device_preference()),
            vocab_size=self.data["model"]["vocab_size"],
            max_hours=self.data["hardware"]["max_hours"],
            preset=self.preset_preference(),
            mfu=self.data["hardware"]["mfu"],
        )

        # An explicit context length overrides the preset's. It has to be
        # applied here rather than left to the caller: the batch schedule is
        # derived from it, and a mismatch makes tokens_per_step stop dividing
        # evenly by micro_batch * context_len.
        override = self.data["model"]["context_len"]
        if override and override != plan.model.context_len:
            plan.model.context_len = override
            per_micro = plan.train.micro_batch * override
            accum = max(1, round(plan.train.tokens_per_step / per_micro))
            plan.train.tokens_per_step = per_micro * accum
            plan.memory = training_memory_gb(
                plan.model, plan.train.micro_batch, plan.device.supports_bf16)
            plan.warnings.append(
                f"context length overridden to {override} from the preset's default; "
                f"batch re-derived to {plan.train.tokens_per_step:,} tokens/step")
        return plan


def write_default(path: Path, plan=None) -> Path:
    """Write a starter config, optionally pinned to a detected plan."""
    config = {k: dict(v) for k, v in DEFAULTS.items()}
    if plan is not None:
        config["model"]["preset"] = plan.preset
        config["hardware"]["device"] = plan.device.kind
    return Settings(config).save(Path(path))


def main(argv=None) -> int:
    """Inspect or create the configuration.

        python -m core.settings            # show the active config
        python -m core.settings --init     # write bananaai.toml for this machine
    """
    import argparse
    import json

    p = argparse.ArgumentParser(description="Inspect or create the configuration")
    p.add_argument("--init", action="store_true", help=f"write {CONFIG_NAME}")
    p.add_argument("--path", type=Path, default=None)
    p.add_argument("--json", action="store_true")
    p.add_argument("--force", action="store_true", help="overwrite an existing file")
    args = p.parse_args(argv)

    if args.init:
        target = args.path or Path(CONFIG_NAME)
        if target.exists() and not args.force:
            print(f"{target} already exists; pass --force to overwrite")
            return 1
        from .hardware import detect, recommend
        plan = recommend(detect())
        write_default(target, plan)
        print(f"wrote {target}, pinned to: {plan.preset} on {plan.device.name}")
        return 0

    settings = load(args.path)
    problems = settings.validate()

    if args.json:
        print(json.dumps({"source": str(settings.source) if settings.source else None,
                          "config": settings.data, "problems": problems}, indent=2))
    else:
        print(f"\nsource: {settings.source or 'built-in defaults (no config file found)'}\n")
        print(dumps(settings.data))
        if problems:
            print("problems:")
            for problem in problems:
                print(f"  - {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
