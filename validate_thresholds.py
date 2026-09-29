"""Check that accept thresholds hold on data they were not tuned on.

For each accept/escalate score (the evidence score, Kev, their mean) and target error rate:
  - cross-validation on the fit coder: tasks split into k folds; the threshold is chosen on k-1 folds as the
    lowest score that keeps accepted attempts at or under the target error, then applied to the held-out fold;
  - transfer: the threshold fit on all of the fit coder's attempts is applied unchanged to each other coder.
Reported: coverage (share kept local) and realized error, which should stay near or under the target.

Reads runs/routing_ev (build_states.py --evidence, run_ollaya.py). Usage:
  python validate_thresholds.py [--fit qwen2.5-coder:3b] [--kev ollaya-kev-0.8b] [--folds 5] [--repeats 20]
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy

RUN = Path(os.environ.get("JEV_RUN_DIR", Path(__file__).parent / "runs" / "routing_ev"))
TARGETS = (0.02, 0.05, 0.10)


def load(name):
    return {r["id"]: r for r in map(json.loads, open(RUN / f"preds_{name}.jsonl"))}


def fit_threshold(scores, passed, target):
    """Lowest threshold whose accepted set (score >= t) has error <= target; +inf if none qualifies."""
    best = numpy.inf
    for t in numpy.unique(scores)[::-1]:
        accepted = scores >= t
        if 1 - passed[accepted].mean() <= target:
            best = t
    return best


def apply(scores, passed, threshold):
    accepted = scores >= threshold
    coverage = float(accepted.mean())
    error = float(1 - passed[accepted].mean()) if accepted.any() else 0.0
    return coverage, error, int(accepted.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", default="qwen2.5-coder:3b")
    parser.add_argument("--kev", default="ollaya-kev-0.8b")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    evidence, kev = load("rule-evidence"), load(args.kev)
    sources = defaultdict(list)
    for row_id, rec in evidence.items():
        if row_id in kev and not rec["source"].startswith("reference"):
            sources[rec["source"]].append(row_id)

    def arrays(ids):
        ev = numpy.array([evidence[i]["probs"][1] for i in ids])
        kv = numpy.array([kev[i]["probs"][1] / sum(kev[i]["probs"]) for i in ids])
        passed = numpy.array([evidence[i]["target"][1] for i in ids])
        tasks = numpy.array([i.split("|")[1] for i in ids])
        return {"evidence": ev, "kev": kv, "mean(evidence, kev)": (ev + kv) / 2}, passed, tasks

    fit_scores, fit_passed, fit_tasks = arrays(sources[args.fit])
    print(f"fit coder {args.fit}: {len(fit_passed)} attempts, pass rate {fit_passed.mean():.2f}\n")
    print(f"cross-validation ({args.folds} task folds x {args.repeats} repeats): "
          f"mean coverage / mean realized error / share of folds over target")
    print(f"{'score':<22}" + "".join(f"{f'target {t:.0%}':>24}" for t in TARGETS))
    rng = numpy.random.default_rng(0)
    unique_tasks = numpy.unique(fit_tasks)
    for name, scores in fit_scores.items():
        line = f"{name:<22}"
        for target in TARGETS:
            covs, errs = [], []
            for _ in range(args.repeats):
                fold_of = dict(zip(unique_tasks, rng.permutation(len(unique_tasks)) % args.folds))
                folds = numpy.array([fold_of[t] for t in fit_tasks])
                for k in range(args.folds):
                    train, test = folds != k, folds == k
                    threshold = fit_threshold(scores[train], fit_passed[train], target)
                    cov, err, _ = apply(scores[test], fit_passed[test], threshold)
                    covs.append(cov)
                    errs.append(err)
            over = numpy.mean(numpy.array(errs) > target)
            line += f"{numpy.mean(covs):>10.2f} /{numpy.mean(errs):>6.3f} /{over:>5.0%}"
        print(line)

    others = [s for s in sources if s != args.fit]
    if not others:
        return
    print(f"\ntransfer: thresholds fit on all of {args.fit}, applied to unseen coders (coverage / realized error)")
    for source in sorted(others):
        scores, passed, _ = arrays(sources[source])
        print(f"\n{source}: {len(passed)} attempts, pass rate {passed.mean():.2f}")
        for name in scores:
            line = f"  {name:<20}"
            for target in TARGETS:
                threshold = fit_threshold(fit_scores[name], fit_passed, target)
                cov, err, _ = apply(scores[name], passed, threshold)
                flag = " !" if err > target * 1.5 else "  "
                line += f"   {target:.0%}: {cov:.2f} / {err:.3f}{flag}"
            print(line)
    print("\n! = realized error more than 1.5x the target")


if __name__ == "__main__":
    main()
