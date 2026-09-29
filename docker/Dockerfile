# Training image for Jev-style decision models (Kev 0.8B/4B/9B via Kev's own trainer; Laya in a separate env).
# Runs the same way on SaladCloud, Hugging Face Jobs, Colab-like VMs and local Docker:
#   docker run --gpus all -e JOB=kev -e RUN_NAME=... -e DATA_REPO=... -e HF_REPO=... -e HF_TOKEN=... IMAGE
# See docker/README.md for every variable. No credentials are baked in: keys arrive as environment variables.
FROM nvidia/cuda:12.8.1-base-ubuntu24.04

ARG KEV_REF=f2bb629d670f5b746f712fc05550a098526c836b
ARG TAILSCALE_VERSION=1.102.4
ARG WITH_LAYA=1

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/hf \
    TRITON_CACHE_DIR=/workspace/hf/triton-cache \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    UV_LINK_MODE=copy

# gcc: Triton JIT-compiles a small launcher for the Gated DeltaNet kernels; openssh/tini for access and signals
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git build-essential openssh-server tini jq \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /run/sshd /root/.ssh && chmod 700 /root/.ssh

COPY --from=ghcr.io/astral-sh/uv:0.11.31 /uv /uvx /usr/local/bin/

# Tailscale (static binaries): SSH into nodes behind NAT (SaladCloud nodes accept no inbound connections)
RUN curl -fsSL "https://pkgs.tailscale.com/stable/tailscale_${TAILSCALE_VERSION}_amd64.tgz" | tar xz -C /tmp \
    && install -m 755 /tmp/tailscale_${TAILSCALE_VERSION}_amd64/tailscale /tmp/tailscale_${TAILSCALE_VERSION}_amd64/tailscaled /usr/local/bin/ \
    && rm -rf /tmp/tailscale_*

# Kev: its own trainer at a pinned commit, Python 3.13, torch 2.8 (+cu128 wheels), the pinned DeltaNet kernels
RUN git clone --quiet https://github.com/jaredpalmer/kev.git /opt/kev-src \
    && git -C /opt/kev-src checkout --quiet "${KEV_REF}" \
    && uv venv --quiet --python 3.13 /opt/kev \
    && uv pip install --quiet --python /opt/kev/bin/python "kev[serve] @ file:///opt/kev-src" \
    && uv pip install --quiet --python /opt/kev/bin/python "flash-linear-attention==0.5.2" "triton>=3.7.1" "huggingface_hub>=1.0" \
    && uv cache clean

# Laya: separate env so its transformers requirements never touch Kev's
RUN if [ "${WITH_LAYA}" = "1" ]; then \
        uv venv --quiet --python 3.12 /opt/laya \
        && uv pip install --quiet --python /opt/laya/bin/python laya "transformers>=4.48" datasets safetensors \
           huggingface_hub accelerate scipy pandas \
        && uv cache clean; \
    fi

COPY train_job.py salad.py entrypoint.sh /opt/app/
RUN chmod +x /opt/app/entrypoint.sh

WORKDIR /workspace
ENTRYPOINT ["/usr/bin/tini", "--", "/opt/app/entrypoint.sh"]

LABEL org.opencontainers.image.title="kev-trainer" \
      org.opencontainers.image.description="Fine-tune Jev-style decision models (Kev, Laya) on SaladCloud, HF Jobs, Colab-like VMs or local Docker" \
      org.opencontainers.image.source="https://github.com/jtatman/kev-code-verify" \
      org.opencontainers.image.licenses="Apache-2.0"
