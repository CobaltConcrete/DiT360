#!/usr/bin/env bash
# Start an interactive DiT360 podman container with GPU access and the repo
# mounted at /workspace.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${IMAGE_NAME:-dit360}"

# :Z relabels the mount for SELinux (Fedora/RHEL).
# --userns=keep-id maps your host UID/GID to the same numeric ID inside the
# container, and --user makes the container process actually run as that ID
# (the image has no USER directive, so it'd otherwise run as root, which
# keep-id remaps AWAY from your UID - still "Permission denied" on files
# your host user owns). This pair is the standard rootless-podman recipe
# for bind mounts to have matching ownership.
# If `--device nvidia.com/gpu=all` fails (CDI not set up - see
# `nvidia-ctk cdi generate`), replace it with `--gpus all` instead.
podman run --rm -it \
    --userns=keep-id \
    --user "$(id -u):$(id -g)" \
    --device nvidia.com/gpu=all \
    -v "$SCRIPT_DIR":/workspace:Z \
    "$IMAGE_NAME"
