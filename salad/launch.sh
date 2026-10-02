#!/usr/bin/env bash
# Create, start and watch one SaladCloud training run until it publishes and stops, with a hard cost cap.
#   salad/launch.sh <group> <hf-repo> [--wait-for LOGFILE MARKER] -- <salad/deploy.py create args after the group name>
# e.g. salad/launch.sh kev-4b-v2 jtatman/kev-4b-code-verify-v2 --wait-for runs/chain_9b.log "chain done" -- \
#        --run-name code-verify-4b-v2 --hf-repo jtatman/kev-4b-code-verify-v2 --data-repo jtatman/kev-code-verify-data \
#        --init jaredpalmer/kev-4b --bf16
# Steps: optional wait for a marker line (aborts if the log reports a Traceback or the writer dies without it),
# create the group, wait for the image pull, start, poll until the job's HF repo has run/DONE and the group is
# stopped. MAX_RUN_HOURS (default 3) after start the group is stopped no matter what. Log: runs/salad/<group>.log
# ATTACH=1 salad/launch.sh <group> <hf-repo>: skip create/start and watch a running group (cap counts from now).
# The cap includes the node's image download (26 min on one home node), so budget for it.
set -uo pipefail
cd "$(dirname "$0")/.."
GROUP=${1:?group}; HF_REPO=${2:?hf repo}; shift 2
WAIT_LOG=""; WAIT_MARKER=""
if [[ "${1:-}" == --wait-for ]]; then WAIT_LOG=$2; WAIT_MARKER=$3; shift 3; fi
[[ "${1:-}" == -- ]] && shift
MAX_RUN_HOURS=${MAX_RUN_HOURS:-3}
PY=.venv/bin/python
mkdir -p runs/salad
exec >> "runs/salad/$GROUP.log" 2>&1
log() { echo "[$(date '+%F %T')] $*"; }
state() {   # "<group status> <running instances> <hf done>"
    HF_TOKEN=$(grep '^HF_API_KEY=' .env | cut -d= -f2- | tr -d "'\"") $PY - "$GROUP" "$HF_REPO" <<'EOF' 2>/dev/null | tail -1
import sys
import salad.deploy as d
from huggingface_hub import HfApi
g = d.call("GET", d.group_path(sys.argv[1])).get("current_state", {})
try:
    done = HfApi().file_exists(sys.argv[2], "run/DONE")
except Exception:
    done = False
print(g.get("status"), g.get("instance_status_counts", {}).get("running_count"), done)
EOF
}

if [[ -n "${ATTACH:-}" ]]; then   # re-attach to a group that is already running (e.g. to extend MAX_RUN_HOURS)
    log "attaching to running $GROUP (MAX_RUN_HOURS=$MAX_RUN_HOURS from now)"
elif [[ -n "$WAIT_LOG" ]]; then
    log "waiting for '$WAIT_MARKER' in $WAIT_LOG"
    until grep -q "$WAIT_MARKER" "$WAIT_LOG" 2>/dev/null; do
        if grep -q -E 'Traceback|Error:' "$WAIT_LOG" 2>/dev/null; then log "upstream log reports an error; not launching"; exit 1; fi
        sleep 60
    done
fi

if [[ -z "${ATTACH:-}" ]]; then
    log "creating $GROUP"
    $PY salad/deploy.py create "$GROUP" "$@" || { log "create failed"; exit 1; }
    for _ in $(seq 120); do read -r status _ _ <<<"$(state)"; [[ "$status" != pending ]] && break; sleep 30; done
    log "group status after image pull: $status"
    [[ "$status" == stopped ]] || { log "unexpected status; not starting"; exit 1; }
    $PY salad/deploy.py start "$GROUP"
fi
started=$(date +%s); last=""
while true; do
    read -r status running done <<<"$(state)"
    now="$status running=$running hf_done=$done"
    [[ "$now" != "$last" ]] && log "$now"; last=$now
    if [[ "$status" == stopped && "$done" == True ]]; then log "finished: published and stopped"; break; fi
    if [[ "$status" == stopped && $(( $(date +%s) - started )) -gt 900 ]]; then log "stopped without run/DONE: check the job"; break; fi
    if (( $(date +%s) - started > MAX_RUN_HOURS * 3600 )); then
        log "MAX_RUN_HOURS=$MAX_RUN_HOURS reached: stopping the group"; $PY salad/deploy.py stop "$GROUP"; break
    fi
    sleep 60
done
log "billed window (start request -> end): $(( ($(date +%s) - started) / 60 )) min"
