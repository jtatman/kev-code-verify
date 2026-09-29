"""Compare every runs/preds_*.jsonl on the shared sample: per-model metrics by kind, then per-row spread across models.

Usage: python compare.py
Writes runs/summary.json alongside the printed tables.
"""
import json
from collections import defaultdict

import numpy

from harness_common import KINDS, RUNS, load_sample, row_metrics


def load_predictions():
    preds = {}
    for path in sorted(RUNS.glob("preds_*.jsonl")):
        name = path.stem.removeprefix("preds_")
        preds[name] = {r["id"]: r for r in map(json.loads, open(path))}
    return preds


def model_table(preds):
    summary = {}
    print(f"{'model':<24}{'kind':<8}{'n':>3}{'acc':>7}{'xent':>7}{'brier':>7}{'p(tgt)':>8}{'conf':>7}{'trunc':>7}{'ms':>8}")
    for name, records in preds.items():
        summary[name] = {}
        for kind in KINDS + ("all",):
            rs = [r for r in records.values() if kind in ("all", r["kind"])]
            if not rs:
                continue
            ms = [row_metrics(r) for r in rs]
            stats = {k: float(numpy.mean([m[k] for m in ms])) for k in ms[0] if k != "level_error"}
            if kind == "score":
                stats["level_error"] = float(numpy.mean([m["level_error"] for m in ms]))
            stats["n"] = len(rs)
            stats["truncated"] = sum(r["truncated"] for r in rs)
            stats["latency_ms"] = float(numpy.median([r["latency_ms"] for r in rs]))
            summary[name][kind] = stats
            print(f"{name if kind == 'choice' else '':<24}{kind:<8}{len(rs):>3}{stats['correct']:>7.2f}"
                  f"{stats['cross_entropy']:>7.2f}{stats['brier']:>7.2f}{stats['p_target']:>8.2f}"
                  f"{stats['confidence']:>7.2f}{stats['truncated']:>7}{stats['latency_ms']:>8.0f}")
        print()
    return summary


def spread_table(preds, rows):
    """Per row: how many models got it right, how many distinct answers they gave, mean p(target)."""
    names = list(preds)
    spread = []
    print(f"{'kind':<7}{'source':<42}{'right':>7}{'answers':>9}{'p(tgt)':>8}  per-model top answer (* = correct)")
    for row in rows:
        target = int(numpy.argmax(row["target"]))
        tops, p_target = {}, []
        for name in names:
            record = preds[name].get(row["id"])
            if record is None:
                continue
            tops[name] = int(numpy.argmax(record["probs"]))
            p_target.append(row_metrics(record)["p_target"])
        right = sum(t == target for t in tops.values())
        entry = {"id": row["id"], "kind": row["kind"], "source": row["source"], "right": right, "models": len(tops),
                 "distinct_answers": len(set(tops.values())), "mean_p_target": float(numpy.mean(p_target))}
        spread.append(entry)
        marks = " ".join(f"{t}{'*' if t == target else ' '}" for t in tops.values())
        print(f"{row['kind']:<7}{row['source'][:41]:<42}{right:>4}/{len(tops):<2}{entry['distinct_answers']:>9}"
              f"{entry['mean_p_target']:>8.2f}  {marks}")
    print("\nper-model columns:", " ".join(names))
    by_right = defaultdict(int)
    for entry in spread:
        by_right[entry["right"]] += 1
    print("rows by number of models correct:", dict(sorted(by_right.items())))
    return spread


def coverage_at(p_yes, passed, max_error):
    """Largest share of rows a router can accept (highest p(yes) first) while accepted rows stay under max_error."""
    order = numpy.argsort(-p_yes, kind="stable")
    failures = numpy.cumsum(1 - passed[order])
    error = failures / numpy.arange(1, len(order) + 1)
    # a threshold can only cut between distinct scores, so tied rows are accepted or escalated together
    cut = numpy.append(p_yes[order][1:] != p_yes[order][:-1], True)
    ok = numpy.nonzero((error <= max_error) & cut)[0]
    return float((ok[-1] + 1) / len(order)) if len(ok) else 0.0


def ece(p_yes, passed, bins=10):
    edges = numpy.linspace(0, 1, bins + 1)
    idx = numpy.clip(numpy.digitize(p_yes, edges[1:-1]), 0, bins - 1)
    return float(sum(abs(p_yes[idx == b].mean() - passed[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def auroc(p_yes, passed):
    pos, neg = p_yes[passed == 1], p_yes[passed == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def routing_table(preds, rows):
    """Router metrics on binary accept/escalate rows (ids ending in |post), per source of the attempts."""
    post = [r for r in rows if r["id"].endswith("|post")]
    if not post:
        return {}
    sources = sorted({r["source"] for r in post}, key=lambda s: s.startswith("reference"))
    result = {}
    print(f"\n{'model':<24}{'rows':<22}{'n':>5}{'pass':>6}{'AUROC':>7}{'cov@5%':>8}{'cov@10%':>9}{'ECE':>6}{'trunc':>7}")
    for name, records in preds.items():
        result[name] = {}
        for source in sources + ["reference (both)"]:
            if source == "reference (both)":
                rs = [records[r["id"]] for r in post if r["source"].startswith("reference") and r["id"] in records]
            else:
                rs = [records[r["id"]] for r in post if r["source"] == source and r["id"] in records]
            if not rs:
                continue
            p_yes = numpy.asarray([r["probs"][1] / sum(r["probs"]) for r in rs])
            passed = numpy.asarray([float(numpy.argmax(r["target"]) == 1) for r in rs])
            stats = {"n": len(rs), "pass_rate": float(passed.mean()), "auroc": auroc(p_yes, passed),
                     "coverage_at_5pct": coverage_at(p_yes, passed, 0.05),
                     "coverage_at_10pct": coverage_at(p_yes, passed, 0.10), "ece": ece(p_yes, passed),
                     "truncated": sum(r["truncated"] for r in rs)}
            result[name][source] = stats
            print(f"{name if source == sources[0] else '':<24}{source[:21]:<22}{stats['n']:>5}{stats['pass_rate']:>6.2f}"
                  f"{stats['auroc']:>7.2f}{stats['coverage_at_5pct']:>8.2f}{stats['coverage_at_10pct']:>9.2f}"
                  f"{stats['ece']:>6.2f}{stats['truncated']:>7}")
    print("\ncov@X% = share of rows the router could keep local (highest p(yes) first) with at most X% of them wrong;"
          "\n         the rest escalate. A perfect classifier reaches the pass rate; random stays near 0 when pass < 1-X.")
    return result


def main():
    rows = load_sample()
    preds = load_predictions()
    summary = model_table(preds)
    spread = spread_table(preds, rows)
    routing = routing_table(preds, rows)
    with open(RUNS / "summary.json", "w") as f:
        json.dump({"models": summary, "rows": spread, "routing": routing}, f, indent=1)


if __name__ == "__main__":
    main()
