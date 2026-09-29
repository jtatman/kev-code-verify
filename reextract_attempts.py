"""Re-apply gen_attempts.extract_code to stored responses after an extraction fix, and relabel what changed.

For every attempt whose extracted code differs from the stored code: re-run the evidence checks and hidden tests,
rewrite the record in place (under the same fcntl lock the generators use), and drop everything downstream that
was computed from the old code - its generated-test probe (evidence.jsonl) and its predictions in runs/routing*/ -
so the next after_generation.sh run recomputes them.

Usage: python reextract_attempts.py [--dry-run]
"""
import argparse
import fcntl
import json
from pathlib import Path

from datasets import load_dataset

from gen_attempts import ATTEMPTS, OUT_DIR, evaluate, extract_code
from gen_evidence import EVIDENCE

ROOT = Path(__file__).parent


def key(r):
    return r["model"], r["task_id"], r["sample"]


def row_id(k):
    return f"{k[0]}|{k[1]}|{k[2]}|post"


def rewrite_jsonl(path, drop):
    """Remove lines whose predicate `drop(record)` is true; returns the number removed."""
    with open(path, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        lines = f.readlines()
        keep = [line for line in lines if not drop(json.loads(line))]
        f.seek(0)
        f.writelines(keep)
        f.truncate()
        fcntl.flock(f, fcntl.LOCK_UN)
    return len(lines) - len(keep)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    tasks = {t["task_id"]: t for t in load_dataset("bigcode/humanevalpack", "python", split="test")}

    changed, flips = set(), 0
    with open(ATTEMPTS, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        records = [json.loads(line) for line in f]
        for r in records:
            if not r["response"].strip():
                continue
            code = extract_code(r["response"])
            if code == r["code"]:
                continue
            changed.add(key(r))
            if args.dry_run:
                continue
            before = r["hidden"]["passed"]
            r.update(code=code, **evaluate(tasks[r["task_id"]], code))
            flips += before != r["hidden"]["passed"]
        if not args.dry_run:
            f.seek(0)
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
            f.truncate()
        fcntl.flock(f, fcntl.LOCK_UN)
    print(f"{len(changed)} attempts with changed code; {flips} hidden-test labels flipped")
    if args.dry_run or not changed:
        return

    removed = rewrite_jsonl(EVIDENCE, lambda e: key(e) in changed)
    print(f"dropped {removed} stale generated-test probes")
    ids = {row_id(k) for k in changed}
    for path in sorted(ROOT.glob("runs/routing*/preds_*.jsonl")):
        n = rewrite_jsonl(path, lambda p: p["id"] in ids)
        if n:
            print(f"dropped {n} stale predictions from {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
