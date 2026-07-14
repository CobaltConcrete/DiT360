#!/usr/bin/env bash
# Start an interactive DiT360 podman container with GPU access, the repo
# mounted at /workspace, and the host's Hugging Face cache mounted so
# already-downloaded weights (e.g. FLUX.1-dev) aren't re-fetched.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${IMAGE_NAME:-dit360}"

# If `--device nvidia.com/gpu=all` fails (CDI not set up - see
# `nvidia-ctk cdi generate`), replace it with `--gpus all` instead.
podman run --rm -it \
    --device nvidia.com/gpu=all \
    -v "$SCRIPT_DIR":/workspace \
    "$IMAGE_NAME"
