"""Stronger router evidence for each attempt, using only what a router would have: the request and cheap models.

1. Generated tests: the cheap coder writes edge-case asserts from the request alone (it never sees an attempt),
   once per task -> runs/routing/gen_tests.jsonl.
2. Probing: every attempt (coder samples and references) runs those asserts, and its outputs on the asserts'
   call inputs are recorded -> runs/routing/evidence.jsonl. Comparing outputs across the coder's independent
   samples gives an agreement signal that needs no expected values at all.
Both stages append as they go and skip finished keys, so a killed run resumes. The hidden tests stay the oracle.

Usage: python gen_evidence.py [--model qwen2.5-coder:3b] [--report]
"""
import argparse
import ast
import json
import logging
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy
from datasets import load_dataset

from gen_attempts import ATTEMPTS, OUT_DIR, extract_code, generate, header_imports

TESTS = OUT_DIR / "gen_tests.jsonl"
EVIDENCE = OUT_DIR / "evidence.jsonl"
TEST_PROMPT = """Write 8 Python assert statements that test a function `{entry_point}` for the request below. Cover the
normal case and edge cases (empty inputs, boundaries, negative numbers, duplicates, and so on, whichever apply).
Each assert must call `{entry_point}` directly and compare it with the exact expected value, one assert per line.
Do not implement the function. Return only the asserts in one ```python code block.

Request:
{instruction}"""
PROBE = r'''
import json, signal, sys
def _alarm(*_): raise TimeoutError()
signal.signal(signal.SIGALRM, _alarm)
def norm(v):
    if isinstance(v, float): return round(v, 6)
    if isinstance(v, (list, tuple)): return type(v)(norm(x) for x in v)
    return v
spec = json.load(open("spec.json"))
ns = {}
try:
    signal.alarm(5); exec(compile(open("prog.py").read(), "prog.py", "exec"), ns); signal.alarm(0)
except BaseException as e:
    print("\n" + json.dumps({"load_error": type(e).__name__})); sys.exit(0)
res = {"asserts": [], "outputs": []}
for a in spec["asserts"]:
    try:
        signal.alarm(2); exec(a, ns); signal.alarm(0); res["asserts"].append(True)
    except BaseException:
        signal.alarm(0); res["asserts"].append(False)
for c in spec["calls"]:
    try:
        signal.alarm(2); v = eval(c, ns); signal.alarm(0); res["outputs"].append(repr(norm(v))[:300])
    except BaseException as e:
        signal.alarm(0); res["outputs"].append("EXC:" + type(e).__name__)
print("\n" + json.dumps(res))
'''

log = logging.getLogger("gen_evidence")


def parse_asserts(text, entry_point):
    """Valid single-line asserts that call the entry point, plus the distinct calls inside them."""
    asserts, calls = [], []
    for line in extract_code(text).splitlines():
        line = line.strip()
        if not line.startswith("assert "):
            continue
        try:
            tree = ast.parse(line)
        except SyntaxError:
            continue
        found = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == entry_point]
        if not found:
            continue
        asserts.append(line)
        call = ast.unparse(found[0])
        if call not in calls:
            calls.append(call)
    return asserts, calls


def load_keyed(path, key):
    return {key(r): r for r in map(json.loads, open(path))} if path.exists() else {}


def probe(program, asserts, calls):
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "prog.py").write_text(program)
        (Path(tmp) / "spec.json").write_text(json.dumps({"asserts": asserts, "calls": calls}))
        (Path(tmp) / "probe.py").write_text(PROBE)
        try:
            proc = subprocess.run([sys.executable, "-I", "probe.py"], cwd=tmp, capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            return {"timed_out": True}
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return {"load_error": "NoOutput"}


def run(model):
    tasks = {t["task_id"]: t for t in load_dataset("bigcode/humanevalpack", "python", split="test")}
    tests = load_keyed(TESTS, lambda r: r["task_id"])
    with open(TESTS, "a") as out:
        for task_id, task in tasks.items():
            if task_id in tests:
                continue
            response, _ = generate(model, TEST_PROMPT.format(entry_point=task["entry_point"], instruction=task["instruction"]),
                                   temperature=0.2, seed=0, raw=True)
            asserts, calls = parse_asserts(response, task["entry_point"])
            tests[task_id] = {"task_id": task_id, "model": model, "response": response, "asserts": asserts, "calls": calls}
            out.write(json.dumps(tests[task_id], ensure_ascii=False) + "\n")
            out.flush()
            log.info("%s: %d asserts, %d distinct calls", task_id, len(asserts), len(calls))

    done = load_keyed(EVIDENCE, lambda r: (r["model"], r["task_id"], r["sample"]))
    with open(EVIDENCE, "a") as out:
        for n, attempt in enumerate(map(json.loads, open(ATTEMPTS))):
            key = (attempt["model"], attempt["task_id"], attempt["sample"])
            if key in done:
                continue
            task, spec = tasks[attempt["task_id"]], tests[attempt["task_id"]]
            program = f"{header_imports(task['prompt'])}\n{task['import']}\n{attempt['code']}\n"
            result = probe(program, spec["asserts"], spec["calls"])
            out.write(json.dumps({"model": key[0], "task_id": key[1], "sample": key[2], **result}) + "\n")
            out.flush()
            if n % 50 == 0:
                log.info("probed %d attempts", n)


def agreement(evidence, coder):
    """Per attempt: share of (other coder sample, call) pairs with identical output; None without comparisons."""
    by_task = defaultdict(list)
    for e in evidence.values():
        if e["model"] == coder and "outputs" in e:
            by_task[e["task_id"]].append(e)
    scores = {}
    for key, e in evidence.items():
        others = [o for o in by_task[e["task_id"]] if (o["model"], o["sample"]) != (e["model"], e["sample"])]
        if "outputs" not in e or not others or not e["outputs"]:
            scores[key] = None if "outputs" in e else 0.0
            continue
        same = [a == b for o in others for a, b in zip(e["outputs"], o["outputs"])]
        scores[key] = float(numpy.mean(same)) if same else None
    return scores


def report(coder):
    from compare import coverage_at

    attempts = load_keyed(ATTEMPTS, lambda r: (r["model"], r["task_id"], r["sample"]))
    evidence = load_keyed(EVIDENCE, lambda r: (r["model"], r["task_id"], r["sample"]))
    tests = load_keyed(TESTS, lambda r: r["task_id"])
    agree = agreement(evidence, coder)

    def gen_rate(key):
        results = evidence[key].get("asserts")
        return float(numpy.mean(results)) if results else 0.0

    canon = [k for k in attempts if k[0] == "reference-canonical"]
    n_asserts = sum(len(t["asserts"]) for t in tests.values())
    canon_pass = [x for k in canon for x in evidence[k].get("asserts", [])]
    print(f"generated tests: {n_asserts} asserts over {len(tests)} tasks "
          f"({sum(1 for t in tests.values() if not t['asserts'])} tasks with none); "
          f"{numpy.mean(canon_pass):.1%} of asserts pass on the correct reference solution (the rest are wrong tests)")

    keys = [k for k in attempts if k[0] == coder]
    passed = numpy.array([float(attempts[k]["hidden"]["passed"]) for k in keys])
    ex = numpy.array([float(attempts[k]["evidence"]["examples_passed"]) for k in keys])
    gen = numpy.array([gen_rate(k) for k in keys])
    agr = numpy.array([1.0 if agree[k] is None else agree[k] for k in keys])
    rules = {
        "examples pass": ex,
        "+ all generated asserts pass": ex * (gen == 1),
        "+ >=75% generated asserts pass": ex * (gen >= 0.75),
        "+ all samples agree": ex * (agr == 1),
        "+ >=75% asserts and all agree": ex * (gen >= 0.75) * (agr == 1),
    }
    print(f"\n{coder}: {len(keys)} attempts, {passed.mean():.0%} pass hidden tests")
    print(f"{'accept when':<34}{'kept local':>11}{'error':>8}")
    for name, accept in rules.items():
        kept = accept.astype(bool)
        print(f"{name:<34}{kept.mean():>11.2f}{1 - passed[kept].mean():>8.3f}")
    score = ex + gen + agr
    print(f"{'graded score ex+gen+agree':<34} coverage at 5% error {coverage_at(score, passed, 0.05):.2f}, "
          f"at 10% {coverage_at(score, passed, 0.10):.2f}")

    hidden_bugs = [k for k in attempts if k[0] == "reference-buggy" and attempts[k]["evidence"]["examples_passed"]]
    caught_gen = sum(gen_rate(k) < 1 for k in hidden_bugs)
    caught_agr = sum((agree[k] or 0) < 1 for k in hidden_bugs)
    print(f"\nbuggy references that pass the request's examples: {len(hidden_bugs)}; "
          f"caught by a failing generated assert: {caught_gen}; disagree with the coder's samples: {caught_agr}")
    print(f"correct references flagged by the same checks: "
          f"{sum(gen_rate(k) < 1 for k in canon)}/{len(canon)} (asserts), {sum((agree[k] or 0) < 1 for k in canon)}/{len(canon)} (agreement)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5-coder:3b")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(OUT_DIR / "evidence.log"), logging.StreamHandler()])
    if not args.report:
        run(args.model)
    report(args.model)


if __name__ == "__main__":
    main()
