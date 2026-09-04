#!/usr/bin/env bash
# One-shot toolchain setup inside WSL2 Ubuntu.
#
#   bash scripts/setup_wsl.sh
#
# Do NOT install an NVIDIA driver inside WSL. The Windows driver is shared
# through /dev/dxg; installing a Linux driver on top of it breaks CUDA.

set -euo pipefail

say() { printf "\n\033[1m==> %s\033[0m\n" "$1"; }

say "System packages"
sudo apt-get update -qq
sudo apt-get install -y -qq \
    build-essential git curl \
    python3 python3-venv python3-dev \
    pkg-config cmake

say "Python virtual environment"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --quiet --upgrade pip wheel setuptools

say "PyTorch (CUDA 12.8 - required for Blackwell sm_120)"
pip install --quiet torch --index-url https://download.pytorch.org/whl/cu128

say "Project dependencies"
pip install --quiet -r requirements.txt

say "Verifying the GPU"
python scripts/verify_env.py

cat <<'NOTE'

Keep the repo and all training data on the WSL filesystem (~/llm), never on
/mnt/c. Crossing the 9p filesystem boundary costs roughly 10x on the small
random reads a dataloader makes, which would show up directly as lost
throughput over a 52-hour run.

NOTE
