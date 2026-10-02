"""Publish the execution-labelled code-verification dataset to the Hugging Face Hub (public by default).

Layout of the dataset repo:
  attempts/attempts.jsonl   every coder attempt: request, raw response, extracted code, evidence, hidden-test label
  attempts/gen_tests.jsonl  edge-case asserts a small model wrote from each request alone
  attempts/evidence.jsonl   each attempt probed against those asserts (pass/fail per assert, outputs per call)
  kev/, laya/               training files in Kev's and Laya's formats (task-grouped split, one coder held out)
  kev/*.ids                 row id per training record, line for line
  runs/<run>/*.ids          the exact rows each published fine-tune was trained and evaluated on
  manifest.json             split settings and counts
Token: HF_TOKEN, else HF_API_KEY from .env.

Usage: python publish_dataset.py [--repo jtatman/kev-code-verify-data] [--private] [--dry-run]
"""
import argparse
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from huggingface_hub import HfApi

from gen_attempts import env_key
from publish_hf import COLLECTION, PROJECT_URL, RUNS, SOURCES

ROOT = Path(__file__).parent
HELD_OUT = {"or:llama-3.1-8b": ("meta-llama/Llama-3.1-8B-Instruct", "Llama 3.1 Community License"),
            "qwen3.5-9b-defiant-iq2m": ("mradermacher/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP-i1-GGUF "
                                        "(IQ2_M; terms of base Qwen/Qwen3.5-9B)", "Apache-2.0")}



def source_table():
    attempts = [json.loads(line) for line in open(ROOT / "runs" / "routing" / "attempts.jsonl")]
    stats = defaultdict(lambda: [0, 0])
    for a in attempts:
        stats[a["model"]][0] += 1
        stats[a["model"]][1] += a["hidden"]["passed"]
    rows = []
    for name, (n, passed) in sorted(stats.items()):
        model, license_ = (SOURCES.get(name) or HELD_OUT.get(name, ("?", "?")))[:2]
        rows.append(f"| {name} | {model} | {license_} | {n} | {passed / n:.2f} |")
    return len(attempts), "\n".join(rows)


def card(repo_id):
    n_attempts, sources = source_table()
    manifest = json.loads((ROOT / "finetune" / "manifest.json").read_text())
    splits = manifest["splits"]
    SPLITS = list(splits)   # train, calibration, development, then every held-out-coder split
    held_out = manifest["holdout"] if isinstance(manifest["holdout"], list) else [manifest["holdout"]]
    split_rows = "\n".join(f"| {s} | {splits[s]['rows']} | {splits[s]['pass_rate']:.2f} |" for s in SPLITS)
    kev_configs = "\n".join(f"      - split: {s}\n        path: kev/{s}.jsonl" for s in SPLITS)
    models = "\n".join(f"- [jtatman/{name}](https://huggingface.co/jtatman/{name})" for name, _ in RUNS.values())
    return f"""---
license: other
license_name: mixed
license_link: LICENSE.md
language: [en, code]
pretty_name: Kev code-verify (execution-labelled)
task_categories: [text-classification]
tags: [code, code-verification, llm-router, decision-model, kev, jev, calibration, humaneval]
size_categories: [1K<n<10K]
configs:
  - config_name: attempts
    data_files: attempts/attempts.jsonl
  - config_name: kev
    data_files:
{kev_configs}
---

# Kev code-verify: execution-labelled coder attempts

{n_attempts} attempts by nine coder models, plus the reference solutions, at the 164 [HumanEvalPack](https://huggingface.co/datasets/bigcode/humanevalpack)
Python tasks, each labelled by **running the task's hidden test suite**. Built to train and evaluate decision models
that tell an agent router whether a cheap model's code can be trusted or should be escalated.

Every attempt also carries evidence a router can compute without the hidden tests: does the code compile, define
the requested function, pass the examples written into the request, and how many of ~8 edge-case asserts (written by
a small model from the request alone, `attempts/gen_tests.jsonl`) it passes.

Models trained on it:
{models}

Code and full pipeline: {PROJECT_URL}

## Sources

| source | model | license | attempts | pass rate |
|---|---|---|---|---|
{sources}

`reference-canonical` / `reference-buggy` are HumanEvalPack's own correct and buggy solutions (a control set).
{", ".join(f"`{h}`" for h in held_out)} {"is" if len(held_out) == 1 else "are"} **held out**: they appear only in the `test_unseen*` splits.

## Files

- `attempts/attempts.jsonl`: `model`, `task_id`, `sample`, `entry_point`, `instruction` (the request, verbatim),
  `response` (raw model output), `code` (extracted), `gen` (token counts, provider), `evidence` (`compiles`,
  `defines_entry_point`, `examples_passed`, `example_error`, `example_traceback_tail`), `hidden` (`passed`,
  `error_type`, `timed_out`).
- `attempts/gen_tests.jsonl`: per task, the generated asserts and the distinct calls inside them. About 22% of
  generated asserts are themselves wrong (they fail on the canonical solution): score by pass rate.
- `attempts/evidence.jsonl`: per attempt, pass/fail per generated assert and the output of each call.
- `kev/`: Kev labelled records (`state` = request + code + checks, one `noul` question `correct`, label = hidden
  tests passed). `laya/`: the same rows for Laya (`state`, `questions`, `gold` as JSON strings).
- `runs/<run>/*.ids`: the rows each published fine-tune used, for exact reproduction.

## Splits (task-grouped, seed 0: no task appears in two splits)

| split | rows | pass rate |
|---|---|---|
{split_rows}

## Licensing and notices

This dataset mixes material under different terms; each row's `model` field names its source.

- **Built with Qwen.** Contains outputs of Qwen2.5-Coder-3B-Instruct (Qwen Research License Agreement, Copyright (c)
  Alibaba Cloud). Its section 4b requires models trained on these outputs to display "Built with Qwen".
- **Built with Llama.** Contains outputs of Llama 3.1 8B Instruct, licensed under the Llama 3.1 Community License,
  Copyright (c) Meta Platforms, Inc. All Rights Reserved.
- Contains outputs of Gemma 3 4B. Gemma is provided under and subject to the Gemma Terms of Use
  (ai.google.dev/gemma/terms).
- Outputs of Mistral Small 3.2, Ministral 3 3B, Qwen3 Coder 30B-A3B, Qwen3 235B-A22B, Qwen2.5 7B and Ternary Bonsai 2
  27B come from Apache-2.0 models.
- Task text, tests and reference solutions: HumanEvalPack (MIT).
- Everything added here (evidence, generated tests, labels, splits, formatting): Apache-2.0.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="jtatman/kev-code-verify-data")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        (folder / "attempts").mkdir()
        for name in ("attempts.jsonl", "gen_tests.jsonl", "evidence.jsonl"):
            shutil.copy(ROOT / "runs" / "routing" / name, folder / "attempts" / name)
        shutil.copytree(ROOT / "finetune" / "kev", folder / "kev")
        shutil.copytree(ROOT / "finetune" / "laya", folder / "laya")
        shutil.copy(ROOT / "finetune" / "manifest.json", folder / "manifest.json")
        for run in RUNS:
            shutil.copytree(ROOT / "runs" / "finetune" / run / "ids", folder / "runs" / run)
        (folder / "README.md").write_text(card(args.repo))
        (folder / "LICENSE.md").write_text("See the Licensing and notices section of README.md: mixed terms per source "
                                           "(Qwen Research License, Llama 3.1 Community License, Gemma Terms of Use, "
                                           "Apache-2.0, MIT); additions by this project are Apache-2.0.\n")
        files = sorted(str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file())
        print(f"{len(files)} files, e.g. {files[:6]}")
        if args.dry_run:
            print((folder / "README.md").read_text()[:3000])
            return
        api = HfApi(token=os.environ.get("HF_TOKEN") or env_key("HF_API_KEY"))
        api.create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)
        api.upload_folder(repo_id=args.repo, repo_type="dataset", folder_path=folder,
                          commit_message="Execution-labelled code-verification data")
        collection = api.create_collection(COLLECTION, namespace=args.repo.split("/")[0], exists_ok=True)
        api.add_collection_item(collection.slug, args.repo, "dataset", exists_ok=True)
        print(f"published https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
