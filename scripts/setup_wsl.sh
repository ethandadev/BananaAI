#!/usr/bin/env bash
# Kept for muscle memory. Setup is cross-platform now and lives in Python, so
# this just forwards to it rather than being a second copy that drifts.
#
#   bash scripts/setup_wsl.sh
#
# Do NOT install an NVIDIA driver inside WSL. The Windows driver is shared
# through /dev/dxg; installing a Linux one on top of it breaks CUDA.

set -euo pipefail
sudo apt-get update -qq
sudo apt-get install -y -qq build-essential git curl python3 python3-venv python3-dev
exec python3 "$(dirname "$0")/setup.py" "$@"
