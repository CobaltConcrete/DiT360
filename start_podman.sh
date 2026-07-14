#!/usr/bin/env bash
# Start an interactive DiT360 podman container with GPU access and the repo
# mounted at /DiT360.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${IMAGE_NAME:-dit360}"

# :Z relabels the mount for SELinux (Fedora/RHEL).
# --userns=keep-id maps container-root to your real host UID, since
# rootless podman's default userns mapping otherwise leaves bind-mounted
# files owned by a UID the container doesn't recognize - "Permission
# denied" even while running as root@container.
# If `--device nvidia.com/gpu=all` fails (CDI not set up - see
# `nvidia-ctk cdi generate`), replace it with `--gpus all` instead.
# HF_HOME override: the image defaults HF_HOME to /root/.cache/huggingface,
# which the container's non-root user can't reach (that path is 700, owned
# by root). Point it at the bind-mounted /DiT360 instead, which the
# non-root user actually owns - this also means downloaded weights persist
# across container restarts on this device.
podman run --rm -it \
    --userns=keep-id \
    --device nvidia.com/gpu=all \
    -e HF_HOME=/DiT360/.cache/huggingface \
    -v "$SCRIPT_DIR":/DiT360:Z \
    -w /DiT360 \
    "$IMAGE_NAME"
