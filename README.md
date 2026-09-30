# kev-code-verify

Fine-tuning Jev-style decision models to decide whether code written by a cheap model can be trusted, so an agent
router can keep the answer local or escalate it to a bigger model. Everything needed to rebuild the results is here:
the data pipeline, a portable training image, and the scripts that ran each fine-tune.

**Models** (Hugging Face, Apache-2.0, Built with Qwen): [collection](https://huggingface.co/collections/jtatman/kev-code-verify-router-judges-6abb6c34566f58d5e7e663e6)

| model | init | notes |
|---|---|---|
| [jtatman/kev-0.8b-code-verify-v2](https://huggingface.co/jtatman/kev-0.8b-code-verify-v2) | jaredpalmer/kev-0.8b | best local judge; trained with weak-coder data |
| [jtatman/kev-0.8b-code-verify-v1](https://huggingface.co/jtatman/kev-0.8b-code-verify-v1) | jaredpalmer/kev-0.8b | first fine-tune |
| [jtatman/kev-4b-code-verify-v1](https://huggingface.co/jtatman/kev-4b-code-verify-v1) | jaredpalmer/kev-4b | bf16 backbone, one 24 GB GPU |

**Dataset:** [jtatman/kev-code-verify-data](https://huggingface.co/datasets/jtatman/kev-code-verify-data): 4,428
execution-labelled attempts by nine coders, generated tests, evidence, and the Kev/Laya training files
(`load_dataset("jtatman/kev-code-verify-data", "kev")`).

**Training image:** `ghcr.io/jtatman/kev-trainer:0.1.1` ([docker/](docker/README.md)): Kev's own trainer at a pinned
commit, torch 2.8 + CUDA 12.8, a separate Laya env, Tailscale/sshd for remote access. Runs on SaladCloud, Hugging Face
Jobs, rented VMs and local Docker.

## Findings

Labels come from execution: coder attempts at [HumanEvalPack](https://huggingface.co/datasets/bigcode/humanevalpack)
(Python) tasks, judged by each task's hidden tests. The judge never sees the hidden tests; its input is what a router
can compute itself: the request verbatim, the code, and checks.

1. **Deterministic evidence does most of the work.** Run the request's own examples, plus ~8 edge-case asserts a cheap
   model writes from the request alone. On a 3B coder (75% pass) that keeps 77% of attempts local at <=5% error
   (the ceiling is 79%). About 22% of generated asserts are wrong, so score by pass rate, never "all pass".
2. **Thresholds are per cheap-tier coder.** Held-out tasks confirm a threshold's error rate, but moving a threshold to
   a weaker coder roughly doubles its error (the failure base rate changes).
3. **Fine-tuning helps most on unseen weak coders.** On Llama-3.1-8B (never trained on; 57% pass):

   | judge | AUROC | kept local at <=5% error |
   |---|---|---|
   | Kev-0.8B | 0.935 | 0.28 |
   | Kev-0.8B fine-tuned v1 | 0.959 | 0.40 |
   | **Kev-0.8B fine-tuned v2** | **0.970** | **0.51** |
   | Kev-4B | 0.970 | 0.51 |
   | Kev-4B fine-tuned | 0.974 | 0.54 |

   v2 alone is weak at strict (2%) targets; averaging it with the evidence score fixes that. Recommended local judge:
   mean(evidence score, Kev-0.8B v2), threshold tuned per coder.

## Reproduce

```bash
uv sync
# 1. attempts, labelled by hidden tests (local Ollama or any OpenAI-compatible endpoint)
python gen_attempts.py --reference --samples 3                                  # qwen2.5-coder:3b via Ollama
python gen_attempts.py --base-url https://openrouter.ai/api/v1 --api-key-env OPENROUTER_API_KEY \
    --model google/gemma-3-4b-it --name or:gemma-3-4b --samples 3 --workers 6
# 2. evidence: generated edge-case tests from the request, probed against every attempt
python gen_evidence.py
# 3. training files: task-grouped 70/15/15 split, one coder held out
python build_finetune.py --holdout or:llama-3.1-8b         # or use the published splits: DATA_REPO=jtatman/kev-code-verify-data
# 4. fine-tune (any of)
docker run --gpus all -e JOB=kev -e RUN_NAME=my-run -e HF_REPO=you/kev-code-verify-mine -e HF_TOKEN=... \
    -v $PWD/finetune/kev:/workspace/data ghcr.io/jtatman/kev-trainer:0.1.1
python salad/deploy.py create my-run --run-name my-run --hf-repo you/... --data-repo you/...   # SaladCloud
MAX_HOURS=3 colab/run_kev_finetune.sh my-run L4 jaredpalmer/kev-0.8b                         # Google Colab
# 5. evaluate against the base model, alone and blended with the evidence score
python eval_finetune.py my-run
```

Kev-4B on a 24 GB card: `WEIGHTS_DTYPE=bf16 EVAL_DTYPE=bf16 BATCH=1 ACCUM=8`. Router metrics and threshold
validation: `compare.py`, `validate_thresholds.py` (scoring zero-shot models through
[ollaya](https://ollaya.dev) with `run_ollaya.py`).

### SaladCloud notes

Salad runs containers on consumer GPUs (RTX 3090/4090 24 GB from ~$0.09-0.33/hr) that accept no inbound connections
and can be reallocated. The image handles this: Tailscale SSH for access, results pushed to the Hub, work already
published is skipped, and the group is created with `restart_policy=on_failure` and stops itself when the job ends.
Use the **SaladCloud API key** (portal > API Access), not the Salad AI Gateway key.

## Licensing

Code: Apache-2.0. Models: Apache-2.0 (from Kev and the Qwen3.5 bases). Training data includes outputs of
Qwen2.5-Coder-3B-Instruct (Qwen Research License: **Built with Qwen**) and, for v2, Gemma 3 4B (Gemma Terms of Use);
each model card lists every source.

## Credits

[Kev](https://github.com/jaredpalmer/kev) by Jared Palmer, [Laya](https://github.com/NandhaKishorM/laya),
[HumanEvalPack](https://huggingface.co/datasets/bigcode/humanevalpack), Qwen, and the Jev idea from TypeSafe.
