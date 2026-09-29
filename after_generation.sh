#!/usr/bin/env bash
# Follow-on pipeline for new coder attempts; every step resumes, so rerunning only processes what is new.
#   1. wait for the given generator PIDs to exit (none given: run immediately)
#   2. probe new attempts with the existing generated tests (gen_evidence.py)
#   3. rebuild the evidence-enriched router rows (build_states.py --evidence)
#   4. score only the new rows with ollaya's Kev (run_ollaya.py keeps existing predictions)
#   5. validate thresholds, including transfer from the 3B coder to every other coder
# Usage: nohup ./after_generation.sh [pid ...] > runs/after_generation.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python

for pid in "$@"; do
    while kill -0 "$pid" 2>/dev/null; do sleep 60; done
done
echo "=== $(date +%T) generators finished"

$PY gen_evidence.py
JEV_RUN_DIR=runs/routing_ev $PY build_states.py --evidence
JEV_RUN_DIR=runs/routing_ev $PY run_ollaya.py kev:0.8b
JEV_RUN_DIR=runs/routing_ev $PY validate_thresholds.py | tee runs/routing_ev/validation.txt
echo "=== $(date +%T) done"
