"""Publish fine-tuned Kev checkpoints to the Hugging Face Hub (public by default) with generated model cards.

Each run in runs/finetune/<run>/ becomes one model repo: the Kev checkpoint (LoRA adapter + pointer head + tokenizer)
plus a model card built from the run's own numbers (eval.json, train.log, config.json), and the run's eval/config
files. A collection groups the repos. Token: HF_TOKEN, else HF_API_KEY from .env.

Usage: python publish_hf.py [--namespace jtatman] [--private] [--dry-run] [run ...]
"""
import argparse
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

from build_states import POST_QUESTION
from gen_attempts import env_key

ROOT = Path(__file__).parent
RUNS = {  # run -> (repo name, one-line summary)
    "code-verify-08b-v2": ("kev-0.8b-code-verify-v2", "current best local judge: weak-coder data added"),
    "code-verify-08b-v1": ("kev-0.8b-code-verify-v1", "first fine-tune"),
    "code-verify-4b-v1": ("kev-4b-code-verify-v1", "4B, trained on a bf16 backbone on a single L4"),
}
PROJECT_URL = os.environ.get("PROJECT_URL", "https://github.com/jtatman/kev-code-verify")
COLLECTION = "Kev code-verify router judges"
# training-data sources -> (upstream model, license, obligation); outputs of these models are the attempts judged
SOURCES = {
    "qwen2.5-coder:3b": ("Qwen/Qwen2.5-Coder-3B-Instruct", "Qwen Research License", "Built with Qwen (license s.4b)"),
    "or:mistral-small-3.2-24b": ("mistralai/Mistral-Small-3.2-24B-Instruct-2506", "Apache-2.0", ""),
    "or:qwen3-coder-30b-a3b": ("Qwen/Qwen3-Coder-30B-A3B-Instruct", "Apache-2.0", ""),
    "or:qwen3-235b-a22b": ("Qwen/Qwen3-235B-A22B-Instruct-2507", "Apache-2.0", ""),
    "or:qwen2.5-7b": ("Qwen/Qwen2.5-7B-Instruct", "Apache-2.0", ""),
    "or:ministral-3b": ("mistralai/Ministral-3-3B-Instruct-2512", "Apache-2.0", ""),
    "or:gemma-3-4b": ("google/gemma-3-4b-it", "Gemma Terms of Use",
                      "outputs used as labelled examples; the model is not trained to imitate Gemma"),
    "ternary-qwen3.8-27b": ("local ternary (1-bit) quantization of a Qwen3.8-27B distillation", "unverified", ""),
    "reference-canonical": ("bigcode/humanevalpack canonical solutions", "MIT", ""),
    "reference-buggy": ("bigcode/humanevalpack buggy solutions", "MIT", ""),
}


def metrics_row(name, m):
    return (f"| {name} | {m['auroc']:.3f} | {m['brier']:.3f} | {m['ece']:.2f} | {m['cov@2%']:.2f} | {m['cov@5%']:.2f} "
            f"| {m['cov@10%']:.2f} |")


def results_table(split_metrics):
    head = ("| score | AUROC | Brier | ECE | kept local @2% err | @5% | @10% |\n"
            "|---|---|---|---|---|---|---|")
    rows = [metrics_row(k, v) for k, v in split_metrics.items()]
    return "\n".join([head, *rows])


def card(run, repo_id, summary):
    out = ROOT / "runs" / "finetune" / run
    ev = json.loads((out / "eval.json").read_text())
    config = json.loads((out / "config.json").read_text())
    train_log = (out / "train.log").read_text()
    mixed = re.search(r"replay: (\d+) of \d+ suite training records mixed with (\d+)", train_log)
    replay, n_train = (mixed.group(1), mixed.group(2)) if mixed else ("?", "?")
    init = config["init_from"]
    revision = re.search(r"snapshots/([0-9a-f]{7,})", config.get("init_resolved", ""))
    init_rev = f"{init}@{revision.group(1)[:7]}" if revision else init
    base = config["base"]
    unseen_rows = [json.loads(line) for line in open(out / "preds_finetuned_test_unseen_coder_all_tasks.jsonl")]
    unseen_n, unseen_pass = len(unseen_rows), sum(r["label"] for r in unseen_rows) / len(unseen_rows)
    cfg = config["config"]
    trained_args = json.loads((out / "checkpoint" / "training_config.json").read_text())["args"]   # kev.train's own record
    unseen, dev = ev["test_unseen_coder_all_tasks"], ev["development"]
    counts = {}
    for row_id in (out / "ids" / "train.ids").read_text().split():
        counts[row_id.split("|")[0]] = counts.get(row_id.split("|")[0], 0) + 1
    source_rows = "\n".join(f"| {k} | {SOURCES[k][0]} | {SOURCES[k][1]} | {n} |" for k, n in sorted(counts.items()))
    return f"""---
license: apache-2.0
base_model: {init}
library_name: peft
pipeline_tag: text-classification
tags: [kev, jev, decision-model, calibration, router, code-verification, lora]
---

# {repo_id.split('/')[1]}

A fine-tune of [{init}](https://huggingface.co/{init}) ({summary}): a Jev-style decision model that judges whether
code written by a cheap coding model is correct, so a router can keep the answer local or escalate it. One forward
pass, a calibrated probability, no generated text.

Trained on **execution-labelled** data: coder attempts at HumanEvalPack (Python) tasks, labelled by running each
task's hidden test suite. The model never sees the hidden tests; its input is what a router can compute itself.

## How to ask it

The fine-tune binds these exact strings. One `noul` question (probability that the statement holds):

```json
{{"type": "noul", "instructions": {json.dumps(POST_QUESTION)}}}
```

State (an object; Kev renders it as `key: value` lines):

```json
{{"request": "<the task, verbatim>",
  "code": "<the model's code>",
  "checks": {{"compiles": true, "defines_requested_function": true, "passes_examples_in_request": true,
             "generated_edge_case_tests": "7 of 8 passed",
             "note": "edge-case tests were written by a small model from the request alone; some may be wrong"}}}}
```

`generated_edge_case_tests` comes from a small model writing ~8 asserts from the request only, then running them.
Serve with Kev's runtime (`python -m kev.serve --run {repo_id}`, a TypeSafe `/v1/systemone` endpoint) or
`kev.predictors.LocalPredictor`. The checkpoint carries its fitted temperature.

## Results

Unseen coder (`llama-3.1-8b-instruct`, never in training): {unseen_n} attempts, pass rate {unseen_pass:.2f}. "Kept local @X% err" = share of attempts a router can accept, highest score first, with at most X% of
them wrong. `evidence score` = (request examples passed + generated-assert pass rate) / 2.

{results_table(unseen)}

Development (coders seen in training, tasks not seen):

{results_table(dev)}

Numbers are on small sets; see the project repo for task-bootstrap confidence intervals.

## Recommended use

Average this model's probability with the execution evidence score, and tune the accept threshold **per cheap-tier
coder** on a few hundred of that coder's execution-labelled attempts: a fixed threshold's error rate depends on how
often the coder fails.

## Training

- Init: `{init_rev}` (LoRA r=16 + pointer head, base `{base}`), Kev's own trainer (`kev.train`, commit {config.get('kev_ref', '')[:8]}).
- Data: {n_train} execution-labelled records (task-grouped split; `llama-3.1-8b` held out) + {replay} public
  decision-v7 replay records against forgetting.
- One epoch, lr {cfg['lr']}, batch {cfg['batch']} x accum {cfg['accum']}, gradient checkpointing, bf16 autocast,
  state limit {cfg.get('max_state')} tokens{', frozen backbone in bf16' if trained_args.get('weights_dtype') == 'bf16' else ''}.
- Temperature fitted on a held-out calibration split (min NLL).
- Hardware: one NVIDIA L4 (Google Colab).

## Training data provenance

Code attempts whose correctness this model learned to judge (labels from executing hidden tests):

| source | model | license | training rows |
|---|---|---|---|
{source_rows}

**Built with Qwen.** Training data includes outputs of Qwen2.5-Coder-3B-Instruct, used under the Qwen Research
License Agreement (section 4b). {'Training data also includes code written by Gemma 3 (4B), used as labelled examples of correct and incorrect code; this model judges that code and is not trained to reproduce Gemma. Gemma is provided under and subject to the Gemma Terms of Use at ai.google.dev/gemma/terms. ' if 'or:gemma-3-4b' in counts else ''}The held-out test coder (Llama 3.1 8B Instruct) was never used for training.

## Limitations

- Python function-level tasks only (HumanEvalPack); other languages, repos and multi-file changes are untested.
- The coder pool is 8-10 models of 3B-235B parameters; accept thresholds do not transfer across coders.
- Trained on this question and state layout; other phrasings work less well.

## Credits

Kev by Jared Palmer ([jaredpalmer/kev](https://github.com/jaredpalmer/kev), Apache-2.0); Qwen3.5 base (Apache-2.0);
HumanEvalPack (bigcode). {f'Project: {PROJECT_URL}' if PROJECT_URL else ''}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", default="jtatman")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("runs", nargs="*")
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN") or env_key("HF_API_KEY")
    api = HfApi(token=token)

    published = []
    for run in args.runs or RUNS:
        name, summary = RUNS[run]
        repo_id = f"{args.namespace}/{name}"
        out = ROOT / "runs" / "finetune" / run
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / name
            shutil.copytree(out / "checkpoint", folder)
            (folder / "README.md").write_text(card(run, repo_id, summary))
            for extra in ("eval.json", "config.json", "train.log"):
                shutil.copy(out / extra, folder / f"run_{extra}")
            if args.dry_run:
                print(f"--- {repo_id}: {sorted(p.name for p in folder.iterdir())}")
                print((folder / "README.md").read_text()[:1500])
                continue
            api.create_repo(repo_id, private=args.private, exist_ok=True)
            api.upload_folder(repo_id=repo_id, folder_path=folder, commit_message=f"Upload {run}")
            published.append(repo_id)
            print(f"published https://huggingface.co/{repo_id}")
    if published:
        collection = api.create_collection(COLLECTION, namespace=args.namespace, private=args.private, exists_ok=True,
                                           description="Jev-style decision models fine-tuned to judge cheap-model code")
        for repo_id in published:
            api.add_collection_item(collection.slug, repo_id, "model", exists_ok=True)
        print(f"collection https://huggingface.co/collections/{collection.slug}")


if __name__ == "__main__":
    main()
