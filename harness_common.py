"""Shared pieces for the Open-Jev model comparison: sampling, Jev question mapping, prediction I/O, metrics.

Every runner writes one JSONL per model to runs/, one line per sampled row:
    {"id", "kind", "source", "options", "target", "probs", "input_tokens", "truncated", "latency_ms"}
where `probs` is a distribution over `options` in dataset order, so all models compare row for row.
"""
import json
import os
import random
from pathlib import Path

import numpy

# JEV_RUN_DIR points every script at another run (e.g. runs/routing); its sample.jsonl must already exist there
RUNS = Path(os.environ.get("JEV_RUN_DIR", Path(__file__).parent / "runs"))
SAMPLE_PATH = RUNS / "sample.jsonl"
DATASET = "ZefanCai/Open-Jev-v1.1"
KINDS = ("choice", "noul", "score")


def build_sample(split="validation", per_kind=10, seed=0):
    """`per_kind` rows of each kind, at most one row per group_id (siblings are near-duplicates)."""
    from datasets import load_dataset

    ds = load_dataset(DATASET, split=split)
    rng = random.Random(seed)
    order = list(range(len(ds)))
    rng.shuffle(order)
    picked, groups = {kind: [] for kind in KINDS}, set()
    for i in order:
        row = ds[i]
        kind = row["kind"]
        if len(picked[kind]) >= per_kind or row["group_id"] in groups:
            continue
        if len(set(row["options"])) != len(row["options"]):
            continue  # choice criteria are keyed by option text, so duplicates can't be represented
        picked[kind].append(row)
        groups.add(row["group_id"])
        if all(len(rows) >= per_kind for rows in picked.values()):
            break
    rows = [
        {k: row[k] for k in ("id", "group_id", "source", "kind", "question", "options", "target", "state_json")}
        for kind in KINDS
        for row in picked[kind]
    ]
    RUNS.mkdir(exist_ok=True)
    with open(SAMPLE_PATH, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def load_sample():
    if not SAMPLE_PATH.exists():
        return build_sample()
    return [json.loads(line) for line in open(SAMPLE_PATH)]


def state_of(row):
    """Decoded state: a dict for most rows, a plain string for a few."""
    return json.loads(row["state_json"])


def state_text(row):
    state = state_of(row)
    return state if isinstance(state, str) else json.dumps(state, indent=1, ensure_ascii=False)


def jev_question(row):
    """Open-Jev row -> Jev /v1/systemone question. noul options are always ['no', 'yes'] = [false, true]."""
    kind = row["kind"]
    if kind == "choice":
        return {"type": "choice", "instructions": row["question"], "criteria": {o: None for o in row["options"]}}
    if kind == "score":
        return {"type": "score", "instructions": row["question"], "criteria": list(row["options"])}
    return {"type": "noul", "instructions": row["question"]}


def probs_from_answer(row, answer):
    """Jev answer -> distribution over row["options"] in dataset order."""
    if row["kind"] == "noul":
        p_true = answer["noul"]
        return [1 - p_true, p_true]
    dist = answer["probabilities"]
    if row["kind"] == "score":
        return [dist[str(i)] for i in range(len(row["options"]))]
    return [dist[o] for o in row["options"]]


def write_predictions(name, records):
    RUNS.mkdir(exist_ok=True)
    path = RUNS / f"preds_{name}.jsonl"
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def prediction_record(row, probs, input_tokens, truncated, latency_ms):
    return {
        "id": row["id"],
        "kind": row["kind"],
        "source": row["source"],
        "options": row["options"],
        "target": row["target"],
        "probs": [float(p) for p in probs],
        "input_tokens": input_tokens,
        "truncated": truncated,
        "latency_ms": latency_ms,
    }


def row_metrics(record):
    probs = numpy.asarray(record["probs"], dtype=float)
    probs = probs / probs.sum()
    target = numpy.asarray(record["target"], dtype=float)
    metrics = {
        "correct": float(numpy.argmax(probs) == numpy.argmax(target)),
        "cross_entropy": float(-numpy.sum(target * numpy.log(numpy.clip(probs, 1e-9, 1)))),
        "brier": float(numpy.sum((probs - target) ** 2)),
        "p_target": float(numpy.sum(probs * target)),
        "confidence": float(probs.max()),
    }
    if record["kind"] == "score":
        levels = numpy.arange(len(probs))
        metrics["level_error"] = float(abs(numpy.sum(probs * levels) - numpy.sum(target * levels)))
    return metrics
