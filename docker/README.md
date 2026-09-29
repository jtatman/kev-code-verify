# kev-trainer image

One container that fine-tunes [Kev](https://github.com/jaredpalmer/kev) decision models (0.8B / 4B / 9B, Kev's own
trainer at a pinned commit) and carries a separate [Laya](https://github.com/NandhaKishorM/laya) environment. It runs
unchanged on SaladCloud, Hugging Face Jobs, rented VMs and local Docker. No credentials are baked in.

| Layer | Contents |
|---|---|
| base | `nvidia/cuda:12.8.1-base-ubuntu24.04`, gcc (Triton JIT), openssh-server, tini, jq |
| `/opt/kev` | Python 3.13 venv: Kev @ `f2bb629`, torch 2.8 (cu128), flash-linear-attention 0.5.2, triton >= 3.7.1, huggingface_hub |
| `/opt/laya` | Python 3.12 venv: laya, transformers, datasets (build with `--build-arg WITH_LAYA=0` to omit) |
| access | Tailscale 1.102 (userspace networking + Tailscale SSH), sshd |
| `/opt/app` | `entrypoint.sh`, `train_job.py` (fine-tune, calibrate, score, publish), `salad.py` (stop own group) |

## Build

```bash
docker build -t ghcr.io/jtatman/kev-trainer:<tag> docker/
docker push ghcr.io/jtatman/kev-trainer:<tag>
```

## Run a fine-tune

```bash
docker run --gpus all --rm \
  -e JOB=kev -e RUN_NAME=code-verify-08b-v2 -e INIT_FROM=jaredpalmer/kev-0.8b \
  -e DATA_REPO=<hf dataset with kev/{train,calibration,development,test_unseen_coder*}.jsonl> \
  -e HF_REPO=<hf model repo to publish to> -e HF_TOKEN=<write token> \
  ghcr.io/jtatman/kev-trainer:<tag>
```

Kev-4B on a 24 GB card (RTX 3090/4090, L4): add `-e WEIGHTS_DTYPE=bf16 -e EVAL_DTYPE=bf16 -e BATCH=1 -e ACCUM=8`.
Smoke test on a small GPU: `-e MAX_RECORDS=40 -e REPLAY=0 -e WEIGHTS_DTYPE=bf16 -e EVAL_DTYPE=bf16 -e BATCH=1`.

## Variables

| Variable | Meaning |
|---|---|
| `JOB` | `kev` = run `train_job.py`; `shell` (default) = stay up for interactive use; or pass a command |
| `RUN_NAME` | run name (required for `JOB=kev`) |
| `INIT_FROM` | Kev checkpoint to start from (Hub id), default `jaredpalmer/kev-0.8b` |
| `EPOCHS`, `REPLAY`, `SEED`, `LR`, `BATCH`, `ACCUM` | training knobs; 0 keeps the checkpoint's own recipe |
| `MAX_STATE` | state tokens per record (default 1664; Kev's default 384 drops long code states) |
| `WEIGHTS_DTYPE`, `EVAL_DTYPE` | `bf16` halves backbone memory for training / scoring |
| `MAX_RECORDS` | trim every split (smoke tests) |
| `DATA_REPO`, `DATA_SUBDIR` | HF dataset holding the splits (default subdir `kev`); else mount files at `/workspace/data` |
| `HF_REPO`, `HF_PRIVATE`, `HF_TOKEN` | publish the checkpoint (repo root) and reports (`run/`); a repo that already holds `run/DONE` is not retrained |
| `TAILSCALE_AUTH_KEY`, `TAILSCALE_HOSTNAME` | join a tailnet with Tailscale SSH (needed on SaladCloud: no inbound connections) |
| `SSH_PUBLIC_KEY` | start sshd with this key (hosts that can reach the container directly) |
| `SALAD_API_KEY`, `SALAD_ORGANIZATION`, `SALAD_PROJECT`, `SALAD_CONTAINER_GROUP_NAME` | on SaladCloud, stop the group when the job ends (Salad restarts exited containers) |
| `KEEP_ALIVE` | `1` = stay up after the job |

## SaladCloud

Salad pulls the image when the container group is created (a new image needs a new group), caches public images for
30 days, and runs on consumer GPUs that can be reallocated mid-run. The job therefore publishes results to the Hub,
skips work that is already published, and stops its own group. Use the **SaladCloud API key** (Account > API Access),
not the Salad AI Gateway key. Pass secrets as environment variables of the container group.
