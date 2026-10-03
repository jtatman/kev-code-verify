"""Per-coder accept thresholds for the router's Kev judge -> a JSON config the router reads.

For each coder with labelled attempts scored by the judge run, and each score (Kev alone, mean(evidence, Kev)):
the threshold is the lowest score that keeps accepted attempts at or under the target error. The score is chosen per
coder by task-fold cross-validation (mean coverage, the realized error must stay at or under the target on average),
then refit on all of that coder's attempts. Coders with too few attempts are pooled into `default`, which is also what
a coder the config does not name gets: the most conservative (highest) of the fitted thresholds for each score.

Usage: python fit_router_thresholds.py [--run code-verify-08b-v2-bf16] [--target 0.05] [--out runs/router/kev_judge.json]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy

from validate_thresholds import apply, fit_threshold

ROOT = Path(__file__).parent
SPLITS = ["development", "test_unseen_coder_all_tasks", "test_unseen_qwen3_5_9b_defiant_iq2m_all_tasks"]
MIN_ATTEMPTS = 150   # below this a coder's own threshold is too noisy; it only contributes to the pooled rows


def load(run):
    evidence = {r["id"]: r["probs"][1] for r in map(json.loads, open(ROOT / "runs" / "routing_ev" / "preds_rule-evidence.jsonl"))}
    rows = defaultdict(list)   # coder -> [(task, kev, evidence, passed)]
    for split in SPLITS:
        snapshot = ROOT / "runs" / "finetune" / run / "ids" / f"{split}.ids"
        ids = (snapshot if snapshot.exists() else ROOT / "finetune" / "kev" / f"{split}.ids").read_text().split()
        preds = [json.loads(line) for line in open(ROOT / "runs" / "finetune" / run / f"preds_finetuned_{split}.jsonl")]
        assert len(ids) == len(preds), split
        for row_id, p in zip(ids, preds):
            coder, task = row_id.split("|")[:2]
            rows[coder].append((task, p["p_true"], evidence[row_id], float(p["label"])))
    return rows


def scores_of(rows):
    tasks = numpy.array([r[0] for r in rows])
    kev, ev, passed = (numpy.array([r[i] for r in rows]) for i in (1, 2, 3))
    return tasks, {"kev": kev, "blend": (kev + ev) / 2}, passed


def cross_validate(tasks, scores, passed, target, folds=5, repeats=20, seed=0):
    rng, unique = numpy.random.default_rng(seed), numpy.unique(tasks)
    covs, errs = [], []
    for _ in range(repeats):
        fold_of = dict(zip(unique, rng.permutation(len(unique)) % folds))
        fold = numpy.array([fold_of[t] for t in tasks])
        for k in range(folds):
            t = fit_threshold(scores[fold != k], passed[fold != k], target)
            cov, err, _ = apply(scores[fold == k], passed[fold == k], t)
            covs.append(cov)
            errs.append(err)
    return float(numpy.mean(covs)), float(numpy.mean(errs))


def fit_coder(rows, target):
    tasks, scores, passed = scores_of(rows)
    entry = {"attempts": len(rows), "pass_rate": round(float(passed.mean()), 3), "scores": {}}
    for name, s in scores.items():
        cov, err = cross_validate(tasks, s, passed, target)
        threshold = fit_threshold(s, passed, target)
        entry["scores"][name] = {"threshold": None if numpy.isinf(threshold) else round(float(threshold), 4),
                                 "cv_coverage": round(cov, 3), "cv_error": round(err, 3)}
    usable = {n: v for n, v in entry["scores"].items() if v["threshold"] is not None and v["cv_error"] <= target + 0.005}   # CV error within noise of the target
    entry["score"] = max(usable, key=lambda n: usable[n]["cv_coverage"]) if usable else "kev"
    entry["threshold"] = entry["scores"][entry["score"]]["threshold"]
    return entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="code-verify-08b-v2-bf16")
    parser.add_argument("--target", type=float, default=0.05)
    parser.add_argument("--out", default="runs/router/kev_judge.json")
    args = parser.parse_args()
    rows = load(args.run)
    coders, pooled = {}, []
    for coder, coder_rows in sorted(rows.items()):
        if coder.startswith("reference"):
            continue
        if len(coder_rows) >= MIN_ATTEMPTS:
            coders[coder] = fit_coder(coder_rows, args.target)
        pooled += coder_rows
    pooled_entry = fit_coder(pooled, args.target)
    # an unnamed coder: the most conservative threshold any fit produced, for the pooled fit's chosen score
    score = pooled_entry["score"]
    candidates = [e["scores"][score]["threshold"] for e in [*coders.values(), pooled_entry] if e["scores"][score]["threshold"]]
    default = {**pooled_entry, "threshold": max(candidates),
               "note": f"unnamed coders: highest {score} threshold over the fitted coders and the pooled fit"}
    config = {"judge": args.run, "target_error": args.target, "question": "correct",
              "fitted_on": SPLITS, "coders": {"default": default, **coders}}
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(config, indent=1) + "\n")
    print(f"{'coder':<28}{'n':>5}{'pass':>6}  {'score':<6}{'thr':>7}  kev cov/err   blend cov/err")
    for name, e in config["coders"].items():
        k, b = e["scores"]["kev"], e["scores"]["blend"]
        print(f"{name:<28}{e['attempts']:>5}{e['pass_rate']:>6.2f}  {e['score']:<6}{e['threshold']:>7.3f}"
              f"  {k['cv_coverage']:.2f}/{k['cv_error']:.3f}   {b['cv_coverage']:.2f}/{b['cv_error']:.3f}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
