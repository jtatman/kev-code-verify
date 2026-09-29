"""Generate coder attempts on HumanEvalPack (python) and label them by execution.

For each task x sample: ask a local Ollama coder for the function, then run in a subprocess
  - visible evidence (a router could compute it): does it compile, define the entry point, pass the
    examples written into the request itself (`example_test`)?
  - the hidden oracle (never shown to a classifier): the full `test` suite -> passed / failed.
`--reference` also scores the dataset's canonical (correct) and buggy solutions the same way, as a control set.

Every attempt is appended to runs/routing/attempts.jsonl as soon as it finishes; rerunning skips
(model, task_id, sample) keys already present, so a killed run resumes where it stopped.

Usage: python gen_attempts.py [--model qwen2.5-coder:3b] [--samples 3] [--temperature 0.8] [--reference]
       python gen_attempts.py --base-url https://openrouter.ai/api/v1 --api-key-env OPENROUTER_API_KEY \
           --model qwen/qwen-2.5-coder-32b-instruct --name or:qwen2.5-coder-32b --samples 1 --workers 4
"""
import argparse
import fcntl
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from datasets import load_dataset

OUT_DIR = Path(__file__).parent / "runs" / "routing"
ATTEMPTS = OUT_DIR / "attempts.jsonl"
OLLAMA = "http://localhost:11434"
TIMEOUT_S = 10
PROMPT = "{instruction}\n\nReturn only the complete Python function, including any imports it needs, in one ```python code block."
# fences must start a line: models such as gemma-3-4b echo the prompt's "in one ```python code block." inside the
# docstring, and a mid-line ``` taken as the closing fence cut the function in half (a false SyntaxError label)
FENCE = re.compile(r"^```(?:python|py)?[ \t]*\n(.*?)^```[ \t]*$", re.DOTALL | re.MULTILINE)
LOOSE_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)

log = logging.getLogger("gen_attempts")


def done_keys():
    if not ATTEMPTS.exists():
        return set()
    return {(a["model"], a["task_id"], a["sample"]) for a in map(json.loads, open(ATTEMPTS))}


def env_key(name):
    """An API key from the environment or the project's gitignored .env."""
    if name in os.environ:
        return os.environ[name]
    env = Path(__file__).parent / ".env"
    for line in env.read_text().splitlines() if env.exists() else []:
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip().strip("'\"")
    raise KeyError(f"{name} not set in the environment or .env")


def generate_openai(base_url, api_key, model, content, temperature, seed, max_tokens, extra=None):
    """OpenAI-compatible chat completion (OpenRouter, llama.cpp server). Thinking is switched off where the
    server honours chat_template_kwargs (llama.cpp); other servers ignore the field. `extra` is merged into the
    body, e.g. {"provider": {"ignore": ["Novita"]}} for OpenRouter provider routing."""
    body = {"model": model, "messages": [{"role": "user", "content": content}], "temperature": temperature,
            "seed": seed, "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False},
            **(extra or {})}
    headers = {"content-type": "application/json"}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(f"{base_url.rstrip('/')}/chat/completions", data=json.dumps(body).encode(),
                                 headers=headers)
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as resp:
        out = json.loads(resp.read())
    usage = out.get("usage") or {}
    return out["choices"][0]["message"]["content"] or "", {
        "eval_count": usage.get("completion_tokens"), "prompt_tokens": usage.get("prompt_tokens"),
        "wall_ms": 1000 * (time.perf_counter() - started), "provider": out.get("provider"),
        "cost": usage.get("cost")}


def generate(model, instruction, temperature, seed, raw=False):
    """raw=True sends `instruction` as the whole prompt instead of wrapping it in PROMPT."""
    content = instruction if raw else PROMPT.format(instruction=instruction)
    body = {"model": model, "stream": False, "messages": [{"role": "user", "content": content}],
            "options": {"temperature": temperature, "seed": seed, "num_predict": 1024, "num_ctx": 4096}}
    req = urllib.request.Request(f"{OLLAMA}/api/chat", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        out = json.loads(resp.read())
    return out["message"]["content"], {"eval_count": out.get("eval_count"), "eval_ms": out.get("eval_duration", 0) / 1e6}


def extract_code(text):
    blocks = FENCE.findall(text) or LOOSE_FENCE.findall(text)
    return max(blocks, key=len) if blocks else text


def header_imports(prompt):
    """Import lines from the task prompt, prepended so a model that forgets `from typing import List` still runs."""
    return "\n".join(line for line in prompt.splitlines() if line.startswith(("import ", "from ")))


def run_python(source):
    """Run source in a fresh isolated interpreter; returns (ok, error_type, traceback_tail, timed_out)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "prog.py"
        path.write_text(source)
        try:
            proc = subprocess.run([sys.executable, "-I", str(path)], cwd=tmp, capture_output=True, text=True,
                                  timeout=TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return False, "Timeout", "", True
    if proc.returncode == 0:
        return True, None, "", False
    lines = proc.stderr.strip().splitlines()
    error_line = lines[-1] if lines else ""
    error_type = error_line.split(":", 1)[0].strip() if error_line else f"exit {proc.returncode}"
    return False, error_type, "\n".join(lines[-4:]), False


def evaluate(task, code):
    program = f"{header_imports(task['prompt'])}\n{task['import']}\n{code}\n"
    try:
        compile(program, "prog.py", "exec")
        compiles = True
    except SyntaxError:
        compiles = False
    defines, *_ = run_python(f"{program}\nassert callable({task['entry_point']})\n")
    examples_passed, example_error, example_tail, _ = run_python(f"{program}\n{task['test_setup']}\n{task['example_test']}")
    hidden_passed, hidden_error, _, hidden_timeout = run_python(
        f"{program}\n{task['test_setup']}\n{task['test']}\ncheck({task['entry_point']})\n")
    return {
        "evidence": {"compiles": compiles, "defines_entry_point": defines, "examples_passed": examples_passed,
                     "example_error": example_error, "example_traceback_tail": example_tail},
        "hidden": {"passed": hidden_passed, "error_type": hidden_error, "timed_out": hidden_timeout},
    }


def record(task, model, sample, code, response, stats, result):
    return {"model": model, "task_id": task["task_id"], "sample": sample, "entry_point": task["entry_point"],
            "instruction": task["instruction"], "response": response, "code": code, "gen": stats, **result,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5-coder:3b", help="model id sent to the endpoint")
    parser.add_argument("--name", help="label stored in the records (default: --model)")
    parser.add_argument("--base-url", help="OpenAI-compatible endpoint (e.g. https://openrouter.ai/api/v1); "
                                           "default: local Ollama")
    parser.add_argument("--api-key-env", help="env/.env variable holding the API key, e.g. OPENROUTER_API_KEY")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=1, help="concurrent requests")
    parser.add_argument("--extra-body", type=json.loads, default=None,
                        help='JSON merged into OpenAI-style requests, e.g. \'{"provider": {"ignore": ["Novita"]}}\'')
    parser.add_argument("--reference", action="store_true", help="also score canonical and buggy solutions")
    args = parser.parse_args()
    name = args.name or args.model
    api_key = env_key(args.api_key_env) if args.api_key_env else None

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s [{name}] %(message)s",
                        handlers=[logging.FileHandler(OUT_DIR / "gen.log"), logging.StreamHandler()])
    tasks = load_dataset("bigcode/humanevalpack", "python", split="test")
    done = done_keys()
    log.info("start model=%s base_url=%s samples=%d temp=%.2f workers=%d already_done=%d", args.model,
             args.base_url or "ollama", args.samples, args.temperature, args.workers, len(done))

    lock = threading.Lock()
    counts = {"passed": 0, "total": 0}

    with open(ATTEMPTS, "a") as out:
        def write(rec):
            # several generator processes may append to the same file: lock across threads and processes
            with lock:
                fcntl.flock(out, fcntl.LOCK_EX)
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                fcntl.flock(out, fcntl.LOCK_UN)

        if args.reference:
            for task in tasks:
                for ref, body in (("reference-canonical", task["canonical_solution"]),
                                  ("reference-buggy", task["buggy_solution"])):
                    if (ref, task["task_id"], 0) in done:
                        continue
                    code = task["declaration"] + body
                    write(record(task, ref, 0, code, "", {}, evaluate(task, code)))
            log.info("reference solutions scored")

        def attempt(task, sample):
            # an empty reply is a provider fault (seen on OpenRouter/Novita: tokens billed, no content), not a
            # wrong answer, so it is retried and never recorded as a failed attempt
            for tries in range(3):
                try:
                    if args.base_url:
                        response, stats = generate_openai(args.base_url, api_key, args.model,
                                                          PROMPT.format(instruction=task["instruction"]),
                                                          args.temperature, sample, args.max_tokens, args.extra_body)
                    else:
                        response, stats = generate(args.model, task["instruction"], args.temperature, seed=sample)
                except Exception as error:  # noqa: BLE001 - log and retry; a rerun retries missing keys
                    log.warning("%s sample %d: generation failed: %s", task["task_id"], sample, error)
                    time.sleep(5 * 2 ** tries)   # back off: rate limits (HTTP 429) otherwise burn every try at once
                    continue
                if response.strip():
                    break
                log.warning("%s sample %d: empty response (try %d)", task["task_id"], sample, tries + 1)
            else:
                return
            code = extract_code(response)
            rec = record(task, name, sample, code, response, stats, evaluate(task, code))
            write(rec)
            with lock:
                counts["total"] += 1
                counts["passed"] += rec["hidden"]["passed"]
                log.info("%s s%d hidden=%s examples=%s (%d/%d passed this run)", task["task_id"], sample,
                         "PASS" if rec["hidden"]["passed"] else "fail", rec["evidence"]["examples_passed"],
                         counts["passed"], counts["total"])

        jobs = [(task, s) for task in tasks for s in range(args.samples) if (name, task["task_id"], s) not in done]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(lambda job: attempt(*job), jobs))
    log.info("done: %d new attempts, %d passed hidden tests", counts["total"], counts["passed"])


if __name__ == "__main__":
    main()
