#!/usr/bin/env bash
# Container entrypoint: optional remote access, then the job, then a clean finish for the platform it runs on.
#   access  TAILSCALE_AUTH_KEY -> join the tailnet with Tailscale SSH (works behind NAT, e.g. SaladCloud)
#           SSH_PUBLIC_KEY     -> sshd on port 22 with that key (for hosts that can reach the container directly)
#   job     JOB=kev  -> /opt/app/train_job.py in the Kev env;  JOB=shell or unset -> stay up for interactive use;
#           any command arguments -> run them instead
#   finish  on SaladCloud (SALAD_API_KEY + group name set) stop this container group, because Salad restarts
#           containers whose process exits; with KEEP_ALIVE=1 stay up afterwards; otherwise exit with the job status
set -uo pipefail

log() { echo "[entrypoint $(date +%H:%M:%S)] $*"; }

if [[ -n "${TAILSCALE_AUTH_KEY:-}" ]]; then
    log "starting tailscale (userspace networking)"
    tailscaled --tun=userspace-networking --state=mem: --socket=/tmp/tailscaled.sock >/tmp/tailscaled.log 2>&1 &
    for _ in $(seq 20); do tailscale --socket=/tmp/tailscaled.sock status >/dev/null 2>&1 && break; sleep 1; done
    tailscale --socket=/tmp/tailscaled.sock up --auth-key="${TAILSCALE_AUTH_KEY}" --ssh --accept-dns=false \
        --hostname="${TAILSCALE_HOSTNAME:-kev-trainer-${SALAD_MACHINE_ID:-$(hostname)}}" \
        && log "tailscale up: $(tailscale --socket=/tmp/tailscaled.sock ip -4 | head -1)" \
        || log "tailscale failed; continuing without it"
fi

if [[ -n "${SSH_PUBLIC_KEY:-}" ]]; then
    printf '%s\n' "${SSH_PUBLIC_KEY}" > /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys
    ssh-keygen -A >/dev/null 2>&1
    /usr/sbin/sshd -o PasswordAuthentication=no -o PermitRootLogin=prohibit-password && log "sshd listening on 22"
    printenv | grep -v -E '^(SSH_PUBLIC_KEY|TAILSCALE_AUTH_KEY)=' > /etc/environment   # same env in ssh sessions
fi

status=0
if [[ $# -gt 0 ]]; then
    "$@"; status=$?
elif [[ "${JOB:-shell}" == kev ]]; then
    /opt/kev/bin/python /opt/app/train_job.py; status=$?
elif [[ "${JOB:-shell}" == shell ]]; then
    log "no job: staying up for interactive use"; exec sleep infinity
else
    log "unknown JOB=${JOB}"; status=2
fi
log "job finished with status ${status}"

if [[ -n "${SALAD_API_KEY:-}" && -n "${SALAD_CONTAINER_GROUP_NAME:-}" ]]; then
    /opt/kev/bin/python /opt/app/salad.py stop || log "could not stop the container group; stop it in the portal"
    exec sleep infinity   # until Salad stops us; exiting would make Salad restart the job
fi
[[ "${KEEP_ALIVE:-0}" == 1 ]] && exec sleep infinity
exit "${status}"
