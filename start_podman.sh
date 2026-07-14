#!/usr/bin/env bash
# Start an interactive DiT360 podman container with GPU access and the repo
# mounted at /workspace.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${IMAGE_NAME:-dit360}"

# :Z relabels the mount for SELinux (Fedora/RHEL) so the container can
# actually read/write it - without it, root-in-container still gets
# "Permission denied" because SELinux blocks the access, not Unix perms.
# If `--device nvidia.com/gpu=all` fails (CDI not set up - see
# `nvidia-ctk cdi generate`), replace it with `--gpus all` instead.
podman run --rm -it \
    --device nvidia.com/gpu=all \
    -v "$SCRIPT_DIR":/workspace:Z \
    "$IMAGE_NAME"
