#!/usr/bin/env bash
# Fine-tune Kev on a Colab GPU from this machine, then bring the results home and release the VM.
#   colab/run_kev_finetune.sh <run-name> [gpu=A100] [init=jaredpalmer/kev-0.8b] [extra KEY=VALUE env for kev_job.py ...]
# e.g. colab/run_kev_finetune.sh code-verify-08b-v1
#      colab/run_kev_finetune.sh code-verify-4b-v1 A100 jaredpalmer/kev-4b
# Uploads finetune/kev/*.jsonl and docker/train_job.py (the training image's job, with Colab paths), starts the job in the background on the VM, polls its log every
# minute (a dropped connection cannot kill the job), downloads /content/out/<run> as a tarball into runs/finetune/,
# and stops the session on exit, success or failure. Local log: runs/finetune/<run>.log
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=${1:?usage: run_kev_finetune.sh <run-name> [gpu] [init] [KEY=VALUE ...]}
GPU=${2:-A100}
INIT=${3:-jaredpalmer/kev-0.8b}
shift $(( $# < 3 ? $# : 3 ))
EXTRA_ENV="$*"
SESSION="kev-${NAME//[^A-Za-z0-9-]/-}"
LOCAL=runs/finetune
mkdir -p "$LOCAL"
exec > >(tee -a "$LOCAL/$NAME.log") 2>&1

MAX_HOURS=${MAX_HOURS:-4}   # hard wall-clock limit: the session is stopped after this whatever the job is doing
remote() {   # run Python on the VM; fails only if colab exec fails (an empty output is fine)
    local out
    # colab exec can hang on a dead websocket; the outer timeout guarantees the poll loop keeps moving
    out=$(timeout $(( ${2:-120} + 60 )) colab exec -s "$SESSION" --timeout "${2:-120}" 2>&1 <<<"$1") || { echo "$out"; return 1; }
    grep -v -E '^\[colab\] (A new version|You can run|Run .uv tool|To silence)' <<<"$out" || true
}
cleanup() { echo "=== $(date +%T) stopping session $SESSION"; colab stop -s "$SESSION" || true; }

echo "=== $(date +%T) run $NAME on $GPU from $INIT ${EXTRA_ENV}"
mkdir -p "$LOCAL/$NAME/ids" && cp finetune/kev/*.ids "$LOCAL/$NAME/ids/"   # row ids this run's predictions map to
colab new --gpu "$GPU" -s "$SESSION"
trap cleanup EXIT

remote "import os; os.makedirs('/content/data', exist_ok=True); os.makedirs('/content/out', exist_ok=True)"
for file in finetune/kev/*.jsonl; do   # train/calibration/development + every held-out-coder split
    timeout 600 colab upload -s "$SESSION" "$file" "/content/data/$(basename "$file")"
done
timeout 600 colab upload -s "$SESSION" docker/train_job.py /content/kev_job.py   # the same job the training image runs

remote "import subprocess
subprocess.Popen('RUN_NAME=$NAME INIT_FROM=$INIT BOOTSTRAP_VENV=/content/kevenv KEV_ROOT=/content/kev DATA_DIR=/content/data OUT_ROOT=/content/out $EXTRA_ENV nohup python /content/kev_job.py > /content/job.log 2>&1 &', shell=True)
print('job started')"

SEEN=0
DEADLINE=$(( $(date +%s) + MAX_HOURS * 3600 ))
while true; do
    sleep 60
    if (( $(date +%s) > DEADLINE )); then
        echo "=== $(date +%T) MAX_HOURS=$MAX_HOURS reached; collecting what exists and stopping"
        STATE=timeout
        break
    fi
    STATUS=$(remote "import os, subprocess
lines = open('/content/job.log').read().splitlines() if os.path.exists('/content/job.log') else []
for line in lines[$SEEN:]: print('LOG', line)
done = os.path.exists('/content/out/$NAME/DONE')
alive = subprocess.run(['pgrep', '-f', 'kev_job.py'], capture_output=True).returncode == 0
print('STATE', len(lines), 'done' if done else ('running' if alive else 'dead'))" 300) || { echo "poll failed (retrying): $STATUS"; continue; }
    grep '^LOG ' <<<"$STATUS" | sed 's/^LOG //' || true
    read -r _ SEEN STATE < <(grep '^STATE ' <<<"$STATUS" | tail -1)
    [[ "$STATE" == running ]] && continue
    echo "=== $(date +%T) job $STATE"
    break
done

remote "import subprocess; subprocess.run(['tar', 'czf', '/content/out_$NAME.tgz', '-C', '/content/out', '$NAME', '/content/job.log'], check=False); print('packed')" 600
timeout 900 colab download -s "$SESSION" "/content/out_$NAME.tgz" "$LOCAL/$NAME.tgz"
tar xzf "$LOCAL/$NAME.tgz" -C "$LOCAL" && echo "=== results in $LOCAL/$NAME/"
[[ "$STATE" == done ]]
