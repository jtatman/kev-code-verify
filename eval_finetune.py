"""Score a Kev fine-tune's downloaded per-row predictions against its baseline, with router metrics.

Reads runs/finetune/<run>/preds_{finetuned,baseline}_<split>.jsonl (from colab/kev_job.py, one line per record in
finetune/kev/<split>.jsonl order) and finetune/kev/<split>.ids. For each split: AUROC, accuracy, Brier, ECE and
coverage at 2/5/10% error, alone and blended with the evidence score (mean of the two), plus a task-bootstrap 95% CI
on the fine-tuned minus baseline difference in coverage at 5% error.

Usage: python eval_finetune.py <run-name>
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy

from compare import auroc, coverage_at, ece

ROOT = Path(__file__).parent
SPLITS = ("development", "test_unseen_coder", "test_unseen_coder_all_tasks")


def load(run, label, split):
    # the ids the run was built from (snapshotted at launch), else the current files
    snapshot = ROOT / "runs" / "finetune" / run / "ids" / f"{split}.ids"
    ids = (snapshot if snapshot.exists() else ROOT / "finetune" / "kev" / f"{split}.ids").read_text().split()
    preds = [json.loads(line) for line in open(ROOT / "runs" / "finetune" / run / f"preds_{label}_{split}.jsonl")]
    if len(ids) != len(preds):
        raise SystemExit(f"{split}: {len(ids)} ids but {len(preds)} predictions")
    return ids, numpy.array([p["p_true"] for p in preds]), numpy.array([float(p["label"]) for p in preds])


def stats(p, y):
    return {"auroc": auroc(p, y), "acc": float(((p >= 0.5) == y).mean()), "brier": float(((p - y) ** 2).mean()),
            "ece": ece(p, y), **{f"cov@{t:.0%}": coverage_at(p, y, t) for t in (0.02, 0.05, 0.10)}}


def main():
    run = sys.argv[1]
    evidence = {r["id"]: r["probs"][1] for r in map(json.loads, open(ROOT / "runs" / "routing_ev" / "preds_rule-evidence.jsonl"))}
    report = {}
    for split in SPLITS:
        ids, p_ft, y = load(run, "finetuned", split)
        _, p_base, _ = load(run, "baseline", split)
        ev = numpy.array([evidence[i] for i in ids])
        scores = {"baseline kev": p_base, "fine-tuned kev": p_ft, "evidence score": ev,
                  "mean(evidence, baseline)": (ev + p_base) / 2, "mean(evidence, fine-tuned)": (ev + p_ft) / 2}
        print(f"\n== {split}: {len(y)} rows, pass rate {y.mean():.2f}")
        print(f"{'score':<28}{'AUROC':>7}{'acc':>7}{'brier':>7}{'ECE':>6}{'cov@2%':>8}{'cov@5%':>8}{'cov@10%':>9}")
        report[split] = {}
        for name, s in scores.items():
            m = stats(s, y)
            report[split][name] = m
            print(f"{name:<28}{m['auroc']:>7.3f}{m['acc']:>7.3f}{m['brier']:>7.3f}{m['ece']:>6.2f}"
                  f"{m['cov@2%']:>8.2f}{m['cov@5%']:>8.2f}{m['cov@10%']:>9.2f}")
        by_task = defaultdict(list)
        for n, row_id in enumerate(ids):
            by_task[row_id.split("|")[1]].append(n)
        tasks, rng = list(by_task), numpy.random.default_rng(0)
        for a, b in (("fine-tuned kev", "baseline kev"), ("mean(evidence, fine-tuned)", "mean(evidence, baseline)")):
            diffs = []
            for _ in range(1000):
                ix = numpy.array([n for t in rng.choice(tasks, len(tasks)) for n in by_task[t]])
                diffs.append(coverage_at(scores[a][ix], y[ix], 0.05) - coverage_at(scores[b][ix], y[ix], 0.05))
            lo, hi = numpy.percentile(diffs, [2.5, 97.5])
            print(f"  cov@5% {a} - {b}: {report[split][a]['cov@5%'] - report[split][b]['cov@5%']:+.2f} "
                  f"[{lo:+.2f}, {hi:+.2f}]{'  significant' if lo > 0 or hi < 0 else ''}")
    (ROOT / "runs" / "finetune" / run / "eval.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
