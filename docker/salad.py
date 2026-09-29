"""Minimal SaladCloud public-API client used inside the container: stop the container group this job runs in.

SaladCloud restarts a container whose process exits, so a finished training job must stop its own group or it
would train again. Needs SALAD_API_KEY (the SaladCloud API key from Account > API Access, not the AI Gateway key),
SALAD_ORGANIZATION, SALAD_PROJECT and SALAD_CONTAINER_GROUP_NAME.

Usage: python salad.py stop | status
"""
import json
import os
import sys
import urllib.request

API = "https://api.salad.com/api/public"


def call(method, path):
    url = (f"{API}/organizations/{os.environ['SALAD_ORGANIZATION']}/projects/{os.environ['SALAD_PROJECT']}"
           f"/containers/{os.environ['SALAD_CONTAINER_GROUP_NAME']}{path}")
    req = urllib.request.Request(url, method=method, headers={"Salad-Api-Key": os.environ["SALAD_API_KEY"],
                                                              "accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read()
    return json.loads(body) if body else {}


def main():
    action = sys.argv[1] if len(sys.argv) > 1 else "status"
    if action == "stop":
        call("POST", "/stop")
        print("container group stop requested", flush=True)
    else:
        print(json.dumps(call("GET", "").get("current_state", {}), indent=1))


if __name__ == "__main__":
    main()
