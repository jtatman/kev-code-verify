"""Turn execution-labeled attempts into router-question rows the existing harness reads.

States are built by extraction, never by summarization: the request is copied verbatim, the code is the
code, and the checks are facts code computed. The hidden-test result is only ever the target.

Row kinds (all `noul`, options ["no", "yes"]):
  post  - after the cheap coder answered: "does this code correctly implement the request?"
          state = request + code + checks a router can run itself (compiles, defines the function,
          passes the examples written into the request). target = hidden tests passed.
  pre   - before any coder runs: "will the cheap coder get this right first try?"
          state = request only. target = the coder's pass rate over its samples (a soft label).
Also writes preds_rule-examples.jsonl: the deterministic baseline "accept iff the request's own examples
pass", which any classifier has to beat to earn its place.

Usage: python build_states.py [--coder qwen2.5-coder:3b]  ->  runs/routing/sample.jsonl
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("JEV_RUN_DIR", str(Path(__file__).parent / "runs" / "routing"))
from harness_common import RUNS, prediction_record, write_predictions  # noqa: E402  (reads JEV_RUN_DIR at import)
from gen_attempts import ATTEMPTS  # noqa: E402

POST_QUESTION = ("Does the code correctly and completely implement the request, so it would pass a thorough "
                 "hidden test suite including edge cases? Judge from the request, the code and the checks shown.")
PRE_QUESTION = ("Will a small 3B-parameter code model, given only this request, write a function that passes a "
                "thorough hidden test suite on its first try?")


def post_row(attempt, probe=None):
    """probe: this attempt's gen_evidence result, adding the generated-test outcome to the checks."""
    evidence = attempt["evidence"]
    checks = {
        "compiles": evidence["compiles"],
        "defines_requested_function": evidence["defines_entry_point"],
        "passes_examples_in_request": evidence["examples_passed"],
    }
    if not evidence["examples_passed"]:
        checks["example_error"] = evidence["example_error"]
        checks["error_output_tail"] = evidence["example_traceback_tail"]
    if probe is not None:
        results = probe.get("asserts") or []
        checks["generated_edge_case_tests"] = (f"{sum(results)} of {len(results)} passed" if results
                                               else f"could not run ({probe.get('load_error', 'timeout')})")
        checks["note"] = "edge-case tests were written by a small model from the request alone; some may be wrong"
    passed = attempt["hidden"]["passed"]
    return {
        "id": f"{attempt['model']}|{attempt['task_id']}|{attempt['sample']}|post",
        "group_id": attempt["task_id"],
        "source": attempt["model"],
        "kind": "noul",
        "question": POST_QUESTION,
        "options": ["no", "yes"],
        "target": [0.0, 1.0] if passed else [1.0, 0.0],
        "state_json": json.dumps({"request": attempt["instruction"], "code": attempt["code"], "checks": checks},
                                 ensure_ascii=False),
    }


def pre_row(task_id, instruction, pass_rate, coder):
    return {
        "id": f"{coder}|{task_id}|pre",
        "group_id": task_id,
        "source": f"pre-route/{coder}",
        "kind": "noul",
        "question": PRE_QUESTION,
        "options": ["no", "yes"],
        "target": [1.0 - pass_rate, pass_rate],
        "state_json": json.dumps({"request": instruction}, ensure_ascii=False),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coder", default="qwen2.5-coder:3b")
    parser.add_argument("--evidence", action="store_true",
                        help="add generated-test results (gen_evidence.py) to the checks; set JEV_RUN_DIR to a new dir")
    args = parser.parse_args()

    attempts = [json.loads(line) for line in open(ATTEMPTS)]
    probes = {}
    if args.evidence:
        from gen_evidence import EVIDENCE, load_keyed
        probes = load_keyed(EVIDENCE, lambda r: (r["model"], r["task_id"], r["sample"]))
    key = lambda a: (a["model"], a["task_id"], a["sample"])  # noqa: E731
    rows = [post_row(a, probes.get(key(a)) if args.evidence else None) for a in attempts]

    by_task = defaultdict(list)
    for a in attempts:
        if a["model"] == args.coder:
            by_task[a["task_id"]].append(a)
    for task_id, group in by_task.items():
        rate = sum(a["hidden"]["passed"] for a in group) / len(group)
        rows.append(pre_row(task_id, group[0]["instruction"], rate, args.coder))

    RUNS.mkdir(parents=True, exist_ok=True)
    with open(RUNS / "sample.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # deterministic baseline, post rows only: accept exactly when the request's own examples pass
    by_id = {post_row(a)["id"]: a for a in attempts}
    rule = [prediction_record(row, [0.0, 1.0] if by_id[row["id"]]["evidence"]["examples_passed"] else [1.0, 0.0],
                              0, False, 0.0)
            for row in rows if row["id"].endswith("|post")]
    write_predictions("rule-examples", rule)
    if args.evidence:
        # the graded evidence score from gen_evidence: (examples passed + generated-assert pass rate) / 2
        def score(a):
            results = probes[key(a)].get("asserts") or []
            return (a["evidence"]["examples_passed"] + (sum(results) / len(results) if results else 0.0)) / 2
        by_id = {post_row(a)["id"]: a for a in attempts}
        graded = [prediction_record(row, [1 - score(by_id[row["id"]]), score(by_id[row["id"]])], 0, False, 0.0)
                  for row in rows if row["id"].endswith("|post")]
        write_predictions("rule-evidence", graded)

    posts = [r for r in rows if r["id"].endswith("|post")]
    print(f"{len(posts)} post rows ({sum(r['target'][1] for r in posts):.0f} passed), {len(rows) - len(posts)} pre rows"
          f" -> {RUNS / 'sample.jsonl'}")


if __name__ == "__main__":
    main()
