"""Fine-tuning files for the post-verify router question, in Kev's and Laya's training formats.

Every row is the evidence-enriched post-verify state build_states.py --evidence produces (request verbatim, code,
computed checks, generated-test results), so what the model trains on is exactly what the router will send.
The label is the hidden-test result.

Splits are by task, so no task appears on two sides:
  train / calibration / development  - 70 / 15 / 15 of the 164 tasks, every coder except the held-out one,
                                       reference solutions (canonical + buggy) in train only
  test_unseen_coder                  - the first held-out coder on development tasks (unseen coder AND unseen tasks)
  test_unseen_coder_all_tasks        - the first held-out coder on every task (unseen coder only)
  test_unseen_<coder>[_all_tasks]    - the same for each further held-out coder
Identical states within a split are kept once. Rows over Kev's 2048-token request limit are dropped and counted.

Writes finetune/kev/*.jsonl (Kev labelled records), finetune/laya/*.jsonl (Laya rows: state, questions and gold
as JSON strings) and finetune/manifest.json.
Usage: python build_finetune.py [--holdout CODER ...] [--seed 0]   (default holdouts: llama-3.1-8b, the local 9B)
"""
import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from build_states import POST_QUESTION, post_row
from gen_attempts import ATTEMPTS
from gen_evidence import EVIDENCE, load_keyed

OUT = Path(__file__).parent / "finetune"
KEV_REVISION = "9a45d25eb2ab761841196625383fa1dff0e56c1e"  # the round-15 checkpoint ollaya serves
KEV_MAX_TOKENS = 2048
QUESTION_ID = "correct"


def render(v, indent=0):
    """Kev's state rendering (kev.api.render): dicts as `key: value` lines, lists as `- item`."""
    pad = "  " * indent
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return str(v)
    if isinstance(v, list):
        return "\n".join(f"{pad}- {render(x, indent + 1).lstrip()}" for x in v)
    return "\n".join(f"{pad}{k}:\n{render(x, indent + 1)}" if isinstance(x, (dict, list)) else f"{pad}{k}: {render(x)}"
                     for k, x in v.items())


def kev_record(state, passed):
    return {"state": state, "questions": {QUESTION_ID: {"type": "noul", "instructions": POST_QUESTION, "label": passed}}}


def laya_row(row_id, state, passed):
    return {"id": row_id, "workflow": "code-verify", "state": json.dumps(state, ensure_ascii=False),
            "questions": json.dumps({QUESTION_ID: {"type": "noul", "instructions": POST_QUESTION}}),
            "gold": json.dumps({QUESTION_ID: {"probabilities": {"false": 0.0 if passed else 1.0,
                                                                  "true": 1.0 if passed else 0.0}}})}


DEFAULT_HOLDOUTS = ["or:llama-3.1-8b", "qwen3.5-9b-defiant-iq2m"]


def unseen_split_names(coder, index):
    """The first held-out coder keeps the original split names (results stay comparable across runs); each further
    one gets test_unseen_<coder> / test_unseen_<coder>_all_tasks."""
    if index == 0:
        return "test_unseen_coder", "test_unseen_coder_all_tasks"
    slug = re.sub(r"[^a-z0-9]+", "_", coder.split(":", 1)[-1].lower()).strip("_")
    return f"test_unseen_{slug}", f"test_unseen_{slug}_all_tasks"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", action="append",
                        help="coder kept out of training (repeat for several); default: " + ", ".join(DEFAULT_HOLDOUTS))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    holdouts = args.holdout or DEFAULT_HOLDOUTS
    test_splits = {h: unseen_split_names(h, i) for i, h in enumerate(holdouts)}

    tokenizer = Tokenizer.from_file(hf_hub_download("jaredpalmer/kev-0.8b", "tokenizer.json", revision=KEV_REVISION))
    question_tokens = len(tokenizer.encode(POST_QUESTION).ids) + 32  # instructions + template/marker overhead

    attempts = list(map(json.loads, open(ATTEMPTS)))
    probes = load_keyed(EVIDENCE, lambda r: (r["model"], r["task_id"], r["sample"]))
    tasks = sorted({a["task_id"] for a in attempts}, key=lambda t: int(t.split("/")[1]))
    random.Random(args.seed).shuffle(tasks)
    n_train, n_cal = round(0.70 * len(tasks)), round(0.15 * len(tasks))
    split_of = {t: "train" for t in tasks[:n_train]}
    split_of.update({t: "calibration" for t in tasks[n_train:n_train + n_cal]})
    split_of.update({t: "development" for t in tasks[n_train + n_cal:]})

    splits = defaultdict(list)
    skipped = Counter()
    for a in attempts:
        key = (a["model"], a["task_id"], a["sample"])
        if key not in probes:
            skipped["no generated-test probe yet"] += 1
            continue
        split = split_of[a["task_id"]]
        if a["model"] in test_splits:
            unseen, unseen_all = test_splits[a["model"]]
            targets = [unseen_all] + ([unseen] if split == "development" else [])
        elif a["model"].startswith("reference"):
            if split != "train":
                continue
            targets = ["train"]
        else:
            targets = [split]
        row = post_row(a, probes[key])
        state = json.loads(row["state_json"])
        tokens = len(tokenizer.encode(render(state)).ids) + question_tokens
        if tokens > KEV_MAX_TOKENS:
            skipped["over Kev's 2048-token limit"] += 1
            continue
        for target in targets:
            splits[target].append({"id": row["id"], "coder": a["model"], "task_id": a["task_id"], "state": state,
                                   "passed": bool(a["hidden"]["passed"]), "tokens": tokens})

    manifest = {"seed": args.seed, "holdout": holdouts, "kev_revision": KEV_REVISION, "question_id": QUESTION_ID,
                "instructions": POST_QUESTION, "tasks": {s: sorted({t for t in tasks if split_of[t] == s})
                                                         for s in ("train", "calibration", "development")},
                "skipped": dict(skipped), "splits": {}}
    for fmt in ("kev", "laya"):
        (OUT / fmt).mkdir(parents=True, exist_ok=True)
    print(f"{'split':<30}{'rows':>6}{'dupes':>7}{'pass':>7}{'tokens p50/p95/max':>22}  coders")
    for name in ("train", "calibration", "development", *(n for pair in test_splits.values() for n in pair)):
        seen, rows = set(), []
        for r in splits[name]:
            fingerprint = json.dumps(r["state"], sort_keys=True)
            if fingerprint not in seen:
                seen.add(fingerprint)
                rows.append(r)
        # <split>.ids lists our row id for each Kev record, line for line, to map predictions back
        with open(OUT / "kev" / f"{name}.jsonl", "w") as kev, open(OUT / "laya" / f"{name}.jsonl", "w") as laya, \
                open(OUT / "kev" / f"{name}.ids", "w") as ids:
            for r in rows:
                kev.write(json.dumps(kev_record(r["state"], r["passed"]), ensure_ascii=False) + "\n")
                ids.write(r["id"] + "\n")
                laya.write(json.dumps(laya_row(r["id"], r["state"], r["passed"]), ensure_ascii=False) + "\n")
        tokens = sorted(r["tokens"] for r in rows)
        coders = Counter(r["coder"] for r in rows)
        pass_rate = sum(r["passed"] for r in rows) / max(1, len(rows))
        manifest["splits"][name] = {"rows": len(rows), "duplicates_dropped": len(splits[name]) - len(rows),
                                    "pass_rate": round(pass_rate, 3), "coders": dict(coders),
                                    "tokens_p50_p95_max": [tokens[len(tokens) // 2], tokens[int(len(tokens) * 0.95)],
                                                           tokens[-1]] if tokens else []}
        stats = manifest["splits"][name]
        print(f"{name:<30}{len(rows):>6}{stats['duplicates_dropped']:>7}{pass_rate:>7.2f}"
              f"{'/'.join(map(str, stats['tokens_p50_p95_max'])):>22}  {len(coders)}")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"skipped: {dict(skipped)}\n-> {OUT}/kev, {OUT}/laya, {OUT}/manifest.json")


if __name__ == "__main__":
    main()
