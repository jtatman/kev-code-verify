"""Run a kev-trainer fine-tune on SaladCloud through its public API (create, start, watch, stop, delete).

Needs the **SaladCloud API key** (portal > account > API Access; not the Salad AI Gateway key) plus organization and
project names, from the environment or .env: SALAD_API_KEY, SALAD_ORGANIZATION, SALAD_PROJECT. The training job's own
secrets (HF_TOKEN, optional TAILSCALE_AUTH_KEY) are passed to the container as environment variables.

  python salad/deploy.py gpus                                   # GPU classes available to the organization
  python salad/deploy.py create <name> --run-name code-verify-08b-v3 --hf-repo you/kev-...-v3 \
        --data-repo jtatman/kev-code-verify-data [--init jaredpalmer/kev-4b --bf16] [--gpu "RTX 3090 (24 GB)"]
  python salad/deploy.py start|status|stop|delete <name>

The group uses restart_policy=on_failure (a finished job is not rerun) and passes its own coordinates so the
container also stops the group when the job ends; the job skips work already published to --hf-repo, so a
reallocated node never trains twice. Untested against a live account until an API key is available: API validation
errors are printed as returned.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

API = "https://api.salad.com/api/public"
# Cloudflare in front of the API rejects Python's default urllib user-agent (error 1010)
USER_AGENT = "kev-code-verify/0.1 (+https://github.com/jtatman/kev-code-verify)"
IMAGE = "ghcr.io/jtatman/kev-trainer:0.1.2"


def env(name, required=True):
    if os.environ.get(name):
        return os.environ[name]
    dotenv = Path(__file__).resolve().parent.parent / ".env"
    for line in dotenv.read_text().splitlines() if dotenv.exists() else []:
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip().strip("'\"")
    if required:
        raise SystemExit(f"{name} is not set (environment or .env)")
    return ""


def call(method, path, body=None):
    url = f"{API}/organizations/{env('SALAD_ORGANIZATION')}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Salad-Api-Key": env("SALAD_API_KEY"), "accept": "application/json", "content-type": "application/json",
        "user-agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as error:
        raise SystemExit(f"{method} {path}: HTTP {error.code}: {error.read().decode(errors='replace')}")
    return json.loads(raw) if raw else {}


def project():
    """API project names are lowercase slugs; the portal may show a capitalised display name ("Default")."""
    return env("SALAD_PROJECT").lower()


def group_path(name=""):
    return f"/projects/{project()}/containers" + (f"/{name}" if name else "")


def gpu_classes():
    return call("GET", "/gpu-classes").get("items", [])


def create(args):
    by_name = {g["name"]: g["id"] for g in gpu_classes()}
    missing = [g for g in args.gpu if g not in by_name]
    if missing:
        raise SystemExit(f"unknown GPU classes {missing}; available: {sorted(by_name)}")
    job_env = {"JOB": "kev", "RUN_NAME": args.run_name, "INIT_FROM": args.init, "HF_REPO": args.hf_repo,
               "DATA_REPO": args.data_repo, "HF_TOKEN": env("HF_TOKEN", required=False) or env("HF_API_KEY"),
               # lets the container stop its own group when the job ends
               "SALAD_API_KEY": env("SALAD_API_KEY"), "SALAD_ORGANIZATION": env("SALAD_ORGANIZATION"),
               "SALAD_PROJECT": project(), "SALAD_CONTAINER_GROUP_NAME": args.name}
    if args.bf16:
        job_env.update(WEIGHTS_DTYPE="bf16", EVAL_DTYPE="bf16", BATCH="1", ACCUM="8")
    if env("TAILSCALE_AUTH_KEY", required=False):
        job_env["TAILSCALE_AUTH_KEY"] = env("TAILSCALE_AUTH_KEY")
    for item in args.env:
        key, _, value = item.partition("=")
        job_env[key] = value
    body = {"name": args.name, "display_name": args.name, "replicas": 1, "autostart_policy": False,
            "restart_policy": "on_failure", "priority": args.priority,   # group-level field in the API
            "container": {"image": args.image, "environment_variables": job_env,
                          # vCPU count and memory in MB, as in SaladCloud's API quickstart
                          "resources": {"cpu": args.cpu, "memory": args.memory_mb,
                                        "gpu_classes": [by_name[g] for g in args.gpu]}}}
    result = call("POST", group_path(), body)
    print(json.dumps({k: result.get(k) for k in ("name", "id", "current_state")}, indent=1))
    print("created (stopped until the image is pulled; then: deploy.py start", args.name + ")")


def serve(args):
    """A short-lived inference group: kev.serve on a published checkpoint behind SaladCloud's Container Gateway.
    The gateway gives https://<dns>; with auth on, callers send the Salad-Api-Key header. kev.serve must bind IPv6 (::).
    No self-stop: the group serves until `stop` (salad/serve.sh stops it after a fixed number of hours)."""
    by_name = {g["name"]: g["id"] for g in gpu_classes()}
    missing = [g for g in args.gpu if g not in by_name]
    if missing:
        raise SystemExit(f"unknown GPU classes {missing}; available: {sorted(by_name)}")
    port = 8000
    server_env = {"HF_TOKEN": env("HF_TOKEN", required=False) or env("HF_API_KEY"), "KEV_DTYPE": args.dtype}
    for item in args.env:
        key, _, value = item.partition("=")
        server_env[key] = value
    body = {"name": args.name, "display_name": args.name, "replicas": args.replicas, "autostart_policy": False,
            "restart_policy": "always", "priority": args.priority,
            "networking": {"protocol": "http", "port": port, "auth": True},
            "readiness_probe": {"http": {"path": "/v1/models", "port": port, "scheme": "http", "headers": []},
                                "initial_delay_seconds": 20, "period_seconds": 10, "timeout_seconds": 5,
                                "success_threshold": 1, "failure_threshold": 3},
            "container": {"image": args.image, "environment_variables": server_env,
                          "command": ["/opt/kev/bin/python", "-m", "kev.serve", "--run", args.run, "--host", "::",
                                      "--port", str(port)],
                          "resources": {"cpu": args.cpu, "memory": args.memory_mb,
                                        "gpu_classes": [by_name[g] for g in args.gpu]}}}
    result = call("POST", group_path(), body)
    print(json.dumps({k: result.get(k) for k in ("name", "id", "current_state", "networking")}, indent=1))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("gpus")
    c = sub.add_parser("create")
    c.add_argument("name")
    c.add_argument("--run-name", required=True)
    c.add_argument("--hf-repo", required=True)
    c.add_argument("--data-repo", required=True)
    c.add_argument("--init", default="jaredpalmer/kev-0.8b")
    c.add_argument("--bf16", action="store_true", help="bf16 backbone recipe (Kev-4B on a 24 GB card)")
    c.add_argument("--gpu", action="append", default=[], help="GPU class name; repeat for several")
    c.add_argument("--priority", default="high", choices=["high", "medium", "low", "batch"])
    c.add_argument("--cpu", type=int, default=4)
    c.add_argument("--memory-mb", type=int, default=30720)
    c.add_argument("--image", default=IMAGE)
    c.add_argument("--env", action="append", default=[], help="extra KEY=VALUE for the job")
    v = sub.add_parser("serve", help="create an inference group (kev.serve behind the Container Gateway)")
    v.add_argument("name")
    v.add_argument("--run", required=True, help="checkpoint: HF repo id, e.g. jtatman/kev-4b-code-verify-v2")
    v.add_argument("--replicas", type=int, default=1)
    v.add_argument("--dtype", default="bf16", help="KEV_DTYPE for kev.serve (bf16 | fp32)")
    v.add_argument("--gpu", action="append", default=[], help="GPU class name; repeat for several")
    v.add_argument("--priority", default="low", choices=["high", "medium", "low", "batch"])
    v.add_argument("--cpu", type=int, default=4)
    v.add_argument("--memory-mb", type=int, default=16384)
    v.add_argument("--image", default=IMAGE)
    v.add_argument("--env", action="append", default=[], help="extra KEY=VALUE for the server")
    for action in ("start", "status", "stop", "delete"):
        sub.add_parser(action).add_argument("name")
    args = parser.parse_args()

    if args.action == "gpus":
        for g in sorted(gpu_classes(), key=lambda g: g["name"]):
            print(f"{g['name']:<28} {g['id']}")
    elif args.action == "create":
        if not args.gpu:
            args.gpu = ["RTX 3090 (24 GB)", "RTX 3090 Ti (24 GB)", "RTX 4090 (24 GB)", "RTX A5000 (24 GB)"]
        create(args)
    elif args.action == "serve":
        if not args.gpu:
            args.gpu = ["RTX 3090 (24 GB)", "RTX 3090 Ti (24 GB)", "RTX 4090 (24 GB)", "RTX A5000 (24 GB)"]
        serve(args)
    elif args.action == "start":
        call("POST", group_path(args.name) + "/start")
        print("start requested")
    elif args.action == "stop":
        call("POST", group_path(args.name) + "/stop")
        print("stop requested")
    elif args.action == "delete":
        call("DELETE", group_path(args.name))
        print("deleted")
    else:
        group = call("GET", group_path(args.name))
        print(json.dumps(group.get("current_state", {}), indent=1))
        if group.get("networking", {}).get("dns"):
            print("endpoint: https://" + group["networking"]["dns"])
        instances = call("GET", group_path(args.name) + "/instances").get("instances", [])
        for i in instances:
            print(f"  instance {i.get('machine_id', i.get('instance_id'))}: {i.get('state')} {i.get('update_time', '')}")


if __name__ == "__main__":
    sys.exit(main())
