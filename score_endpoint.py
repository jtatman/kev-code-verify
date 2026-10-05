"""Score Kev training-format records against a kev.serve endpoint (local, or SaladCloud via its Container Gateway).
  python score_endpoint.py <endpoint> <out-run> [--splits development ...] [--baseline-run code-verify-4b-v2]
e.g. python score_endpoint.py $(cat runs/salad/kev-4b-v2-serve.endpoint) code-verify-4b-v2-salad-serve
Writes runs/finetune/<out-run>/preds_finetuned_<split>.jsonl (p_true per record, file order) and copies the baseline
run's own fine-tuned predictions in as preds_baseline_<split>.jsonl plus its ids/, so `python eval_finetune.py <out-run>`
compares served vs reference. A *.salad.cloud endpoint gets the Salad-Api-Key header from .env. Reports latency."""
import argparse
import json
import shutil
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from gen_attempts import env_key

QUESTION = "correct"
DATA = Path("finetune/kev")
RUNS = Path("runs/finetune")


def as_bool(label):
    return label is True or str(label).lower() == "true"


def ask(endpoint, headers, record):
    """-> (p_true, seconds). The question is sent without its label; the server applies the checkpoint's temperature."""
    question = {k: v for k, v in record["questions"][QUESTION].items() if k != "label"}
    body = json.dumps({"state": record["state"], "questions": {QUESTION: question}}).encode()
    for attempt in range(5):
        start = time.time()
        try:
            req = urllib.request.Request(endpoint.rstrip("/") + "/v1/systemone", data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=120) as resp:
                answer = json.loads(resp.read())
            return answer["answers"][QUESTION]["noul"], time.time() - start
        except Exception as error:   # spot replicas come and go; retry, then fail loudly
            if attempt == 4:
                raise RuntimeError(f"{endpoint}: {error}") from error
            time.sleep(5 * 2 ** attempt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("endpoint")
    parser.add_argument("out_run")
    parser.add_argument("--splits", nargs="+", default=["development", "test_unseen_coder_all_tasks",
                                                        "test_unseen_qwen3_5_9b_defiant_iq2m_all_tasks"])
    parser.add_argument("--baseline-run", default="code-verify-4b-v2")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    headers = {"Content-Type": "application/json", "User-Agent": "kev-code-verify/0.1"}
    if ".salad.cloud" in args.endpoint:
        headers["Salad-Api-Key"] = env_key("SALAD_API_KEY")
    out, base = RUNS / args.out_run, RUNS / args.baseline_run
    out.mkdir(parents=True, exist_ok=True)
    shutil.copytree(base / "ids", out / "ids", dirs_exist_ok=True)
    latencies = {}
    for split in args.splits:
        records = [json.loads(line) for line in open(DATA / f"{split}.jsonl")]
        start = time.time()
        with ThreadPoolExecutor(args.workers) as pool:
            answers = list(pool.map(lambda r: ask(args.endpoint, headers, r), records))
        rows = [{"p_true": p, "label": as_bool(r["questions"][QUESTION]["label"])} for (p, _), r in zip(answers, records)]
        with open(out / f"preds_finetuned_{split}.jsonl", "w") as f:
            f.writelines(json.dumps(row) + "\n" for row in rows)
        if (base / f"preds_finetuned_{split}.jsonl").exists():   # a split newer than the baseline run has none
            shutil.copy(base / f"preds_finetuned_{split}.jsonl", out / f"preds_baseline_{split}.jsonl")
        seconds = sorted(s for _, s in answers)
        latencies[split] = {"records": len(rows), "wall_s": round(time.time() - start, 1),
                            "median_s": round(statistics.median(seconds), 3), "p95_s": round(seconds[int(0.95 * (len(seconds) - 1))], 3)}
        print(split, latencies[split], flush=True)
    (out / "result.json").write_text(json.dumps({"endpoint": args.endpoint.split("//")[-1].split(".")[0] + "...",
                                                 "baseline_run": args.baseline_run, "latency": latencies}, indent=1))


if __name__ == "__main__":
    main()
