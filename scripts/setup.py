"""Set up BananaAI on any machine.

    python scripts/setup.py

Creates a virtual environment, works out which PyTorch build this computer
needs, installs it, then verifies the result by running real kernels.

Written in Python rather than shell so there is one setup path for Linux,
macOS and Windows instead of three that drift apart. It only needs a Python
interpreter, which you already have if you can run this file.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import venv
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MIN_PYTHON = (3, 10)

# PyTorch publishes one wheel index per accelerator. Picking the wrong one is
# the single most common setup failure: the install succeeds and then every
# kernel launch fails.
TORCH_INDEXES = {
    "cu128": "https://download.pytorch.org/whl/cu128",
    "cu126": "https://download.pytorch.org/whl/cu126",
    "rocm": "https://download.pytorch.org/whl/rocm6.2",
    "cpu": "https://download.pytorch.org/whl/cpu",
}


def say(message: str) -> None:
    print(f"\n==> {message}", flush=True)


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"    $ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run(cmd, check=True, **kwargs)


def venv_python(env_dir: Path) -> Path:
    if os.name == "nt":
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


def nvidia_driver_version() -> str | None:
    """Read the driver version from nvidia-smi, if it is there at all."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return (out.stdout.strip().splitlines() or [None])[0]


def gpu_names() -> list[str]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run([exe, "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def choose_accelerator(force: str | None = None) -> tuple[str, str]:
    """Return (index key, why). Errs toward the CPU build when unsure."""
    if force:
        return force, "forced with --accelerator"

    if platform.system() == "Darwin" and platform.machine() == "arm64":
        # Apple Silicon uses the default wheels, which carry MPS support.
        return "cpu", "Apple Silicon: the default wheel provides MPS"

    driver = nvidia_driver_version()
    if driver:
        names = ", ".join(gpu_names()) or "an NVIDIA GPU"
        try:
            major = int(driver.split(".")[0])
        except ValueError:
            major = 0
        # Blackwell (sm_120) needs CUDA 12.8 wheels, which need a 570+ driver.
        if major >= 570:
            return "cu128", f"{names}, driver {driver}"
        if major >= 525:
            return "cu126", f"{names}, driver {driver} (below 570, so not CUDA 12.8)"
        return "cpu", (f"{names}, but driver {driver} is too old for current "
                       f"CUDA wheels -- update the driver and rerun")

    if shutil.which("rocminfo"):
        return "rocm", "ROCm detected"

    return "cpu", "no GPU detected"


def check_python() -> None:
    if sys.version_info < MIN_PYTHON:
        raise SystemExit(
            f"Python {'.'.join(map(str, MIN_PYTHON))}+ is required; this is "
            f"{platform.python_version()}"
        )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--venv", type=Path, default=REPO / ".venv")
    p.add_argument("--accelerator", choices=sorted(TORCH_INDEXES), default=None,
                   help="override the detected accelerator")
    p.add_argument("--skip-verify", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="report the plan and stop")
    args = p.parse_args(argv)

    check_python()

    say("Working out what this machine needs")
    key, why = choose_accelerator(args.accelerator)
    index = TORCH_INDEXES[key]
    print(f"    system       {platform.system()} {platform.machine()}")
    print(f"    python       {platform.python_version()}")
    print(f"    accelerator  {key}  ({why})")
    print(f"    torch index  {index}")

    if args.dry_run:
        print("\n(dry run, nothing installed)")
        return 0

    if key == "cpu" and "driver" in why and "too old" in why:
        print("\n    Continuing with the CPU build. Training will be very slow.")

    say(f"Creating the virtual environment at {args.venv}")
    if not venv_python(args.venv).exists():
        venv.EnvBuilder(with_pip=True, clear=False).create(args.venv)
    python = venv_python(args.venv)

    say("Upgrading pip")
    run([str(python), "-m", "pip", "install", "--quiet", "--upgrade", "pip", "wheel"])

    say(f"Installing PyTorch ({key})")
    run([str(python), "-m", "pip", "install", "--quiet", "torch",
         "--index-url", index])

    say("Installing the rest of the dependencies")
    run([str(python), "-m", "pip", "install", "--quiet", "-r",
         str(REPO / "requirements.txt")])

    if not args.skip_verify:
        say("Verifying")
        # Not check=True: a failed check should print its report, not raise.
        subprocess.run([str(python), str(REPO / "scripts" / "verify_env.py")], cwd=REPO)

    activate = (r".venv\Scripts\activate" if os.name == "nt"
                else "source .venv/bin/activate")
    print(f"""
Done.

  activate           {activate}
  see your plan      python -m core.hardware
  write a config     python -m core.settings --init
  run the pipeline   python scripts/run_pipeline.py smoke
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
