#!/usr/bin/env bash
# Bring up a short-lived SaladCloud inference endpoint for evals, then stop it after a fixed time.
#   salad/serve.sh <group> <hours> -- <salad/deploy.py serve args after the group name>
# e.g. salad/serve.sh kev-4b-v2-serve 1 -- --run jtatman/kev-4b-code-verify-v2 --replicas 2
# Creates the group (unless it exists), waits for the image pull, starts it, logs the endpoint once a replica is ready
# (also written to runs/salad/<group>.endpoint), and stops the group <hours> after the start request whatever happens.
# Callers send the Salad-Api-Key header (salad gateway auth). Log: runs/salad/<group>.log
set -uo pipefail
cd "$(dirname "$0")/.."
GROUP=${1:?group}; HOURS=${2:?hours}; shift 2
[[ "${1:-}" == -- ]] && shift
PY=.venv/bin/python
mkdir -p runs/salad
exec >> "runs/salad/$GROUP.log" 2>&1
log() { echo "[$(date '+%F %T')] $*"; }
state() {   # "<group status> <ready instances> <dns>"
    $PY - "$GROUP" <<'EOF' 2>/dev/null | tail -1
import sys
import salad.deploy as d
g = d.call("GET", d.group_path(sys.argv[1]))
s = g.get("current_state", {})
print(s.get("status"), s.get("instance_status_counts", {}).get("running_count", 0), g.get("networking", {}).get("dns", "-"))
EOF
}

read -r status _ _ <<<"$(state)"
if [[ -z "$status" ]]; then
    log "creating $GROUP"
    $PY salad/deploy.py serve "$GROUP" "$@" || { log "create failed"; exit 1; }
    for _ in $(seq 120); do read -r status _ _ <<<"$(state)"; [[ "$status" != pending ]] && break; sleep 30; done
    log "group status after image pull: $status"
fi
$PY salad/deploy.py start "$GROUP"
started=$(date +%s); last=""; announced=""
while (( $(date +%s) - started < HOURS * 3600 )); do
    read -r status running dns <<<"$(state)"
    now="$status running=$running"
    [[ "$now" != "$last" ]] && log "$now"; last=$now
    if [[ -z "$announced" && "$running" -gt 0 && "$dns" != - ]]; then
        echo "https://$dns" > "runs/salad/$GROUP.endpoint"; announced=1
        log "endpoint https://$dns (send Salad-Api-Key); stopping at $(date -d @$((started + HOURS * 3600)) '+%T')"
    fi
    sleep 60
done
log "time cap ($HOURS h) reached: stopping"
$PY salad/deploy.py stop "$GROUP"
rm -f "runs/salad/$GROUP.endpoint"
