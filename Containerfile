# Build/runtime image for DiT360 inference and training.
#
# Repo code and the Hugging Face model cache are bind-mounted at run time
# (see the `podman run` command in the README/chat instructions), not baked
# into the image, so editing scripts on the host doesn't require a rebuild.

FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Pinned per the repo's README (torch > 2; tested on 2.6.0/0.21.0). Pulled
# from PyTorch's own index so the CUDA 12.4 build is selected, matching the
# driver already on the host (CUDA 13.1 driver is backward compatible).
RUN pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 \
        --index-url https://download.pytorch.org/whl/cu124

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

CMD ["/bin/bash"]
