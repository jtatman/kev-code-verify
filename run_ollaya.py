"""Poll decision models served by ollaya (localhost:11435, POST /api/decide) on the current run's sample.

Same output as run_laya.py (runs/<dir>/preds_ollaya-<model>.jsonl), so compare.py sets ollaya's fp16/fp32 ONNX
graphs beside the ggmlc GGUF binary. ollaya reports truncation itself (`state_truncated`).
Rows already in an existing predictions file are kept, so after new attempts only the new rows are scored.

Usage: python run_ollaya.py [model ...]     e.g. kev:0.8b laya:typed-decisions
"""
import json
import sys
import urllib.error
import urllib.request

from harness_common import (RUNS, jev_question, load_sample, prediction_record, probs_from_answer, state_of,
                            write_predictions)

OLLAYA = "http://localhost:11435"
MODELS = ["kev:0.8b", "laya:en", "laya:multilingual", "laya:typed-decisions"]


def decide(model, state, questions):
    body = {"model": model, "state": state, "questions": questions, "keep_alive": "10m"}
    req = urllib.request.Request(f"{OLLAYA}/api/decide", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as error:
        return json.loads(error.read())


def run_model(model, rows, existing):
    """Scores rows not already in `existing` (id -> record); returns existing plus new records, in row order."""
    records = []
    for row in rows:
        if row["id"] in existing:
            records.append(existing[row["id"]])
            continue
        response = decide(model, state_of(row), {"q": jev_question(row)})
        if "answers" not in response:
            print(f"  [{model}] {row['id']}: {response.get('code')} {response.get('error')}", file=sys.stderr)
            continue
        probs = probs_from_answer(row, response["answers"]["q"])
        records.append(prediction_record(row, probs, response["usage"]["input_tokens"], response["state_truncated"],
                                         response["eval_duration"] / 1e6))
    unload(model)  # free VRAM for the next model
    return records


def unload(model):
    """A decide request without state and with keep_alive 0 unloads the model."""
    req = urllib.request.Request(f"{OLLAYA}/api/decide", data=json.dumps({"model": model, "keep_alive": 0}).encode(),
                                 headers={"content-type": "application/json"})
    urllib.request.urlopen(req, timeout=60).read()


def main():
    rows = load_sample()
    for model in sys.argv[1:] or MODELS:
        name = "ollaya-" + model.replace(":", "-")
        path = RUNS / f"preds_{name}.jsonl"
        existing = {r["id"]: r for r in map(json.loads, open(path))} if path.exists() else {}
        records = run_model(model, rows, existing)
        path = write_predictions(name, records)
        print(f"{name}: {len(records)}/{len(rows)} rows, truncated={sum(r['truncated'] for r in records)} -> {path}",
              flush=True)


if __name__ == "__main__":
    main()
