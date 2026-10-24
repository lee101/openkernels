#!/usr/bin/env python3
"""RunPod H200 with hard cost guards: TTL'd pods, daily minute budget, always-terminate.
rp.py up [--minutes N]   create pod (sglang judge image), print ssh
rp.py ls | down [id] | reap | budget"""
import argparse, datetime as dt, json, os, sys, time, urllib.request
from pathlib import Path

API = "https://api.runpod.io/graphql"
TAG = "okc-h200"
IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
GPU = "NVIDIA H200"
BUDGET = Path(__file__).with_name("notes") / "rp_budget.json"
CAP_MIN = 120  # per-day GPU-minute cap; raise deliberately


def load_env(p=Path(__file__).with_name(".env")):
    if not p.exists(): return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


load_env()


def key():
    k = os.environ.get("RUNPOD_API_KEY", "").strip()
    if k: return k
    sys.exit("no RUNPOD_API_KEY")


def gql(q, v=None):
    r = urllib.request.Request(API, data=json.dumps({"query": q, "variables": v or {}}).encode(),
                               headers={"content-type": "application/json", "authorization": f"Bearer {key()}", "user-agent": "okc/1"})
    out = json.load(urllib.request.urlopen(r, timeout=60))
    if out.get("errors"): raise RuntimeError(json.dumps(out["errors"])[:600])
    return out["data"]


def budget(add=0.0):
    b = json.loads(BUDGET.read_text()) if BUDGET.exists() else {"cap": CAP_MIN, "spent": {}}
    d = dt.date.today().isoformat(); s = b["spent"].get(d, 0.0)
    if add:
        if s + add > b["cap"]: sys.exit(f"budget: {s:.0f}+{add:.0f} > {b['cap']} min today")
        b["spent"][d] = s + add; BUDGET.write_text(json.dumps(b, indent=1))
    return s, b["cap"]


def pods():
    return [p for p in gql("query{myself{pods{id name desiredStatus costPerHr env runtime{uptimeInSeconds ports{ip publicPort privatePort type}}}}}")["myself"]["pods"] or []
            if (p.get("name") or "").startswith(TAG)]


def down(pid): gql("mutation($id:String!){podTerminate(input:{podId:$id})}", {"id": pid}); print("terminated", pid)


def reap():
    now = time.time()
    for p in pods():
        ttl = next((int(e.split("=", 1)[1]) for e in p.get("env") or [] if isinstance(e, str) and e.startswith("TTL_EPOCH=")), 0)
        if not ttl or now > ttl: down(p["id"])


def up(minutes, cloud, image, gpu=GPU):
    budget(minutes)
    home = Path.home()
    pub = next((home / f".ssh/{k}.pub").read_text().strip() for k in ("id_ed25519", "id_rsa") if (home / f".ssh/{k}.pub").exists())
    ttl = int(time.time()) + minutes * 60
    inp = {"name": f"{TAG}-{int(time.time())}", "imageName": image, "gpuTypeId": gpu, "cloudType": cloud, "gpuCount": 1,
           "containerDiskInGb": 60, "volumeInGb": 0, "ports": "22/tcp", "startSsh": True,
           "env": [{"key": "PUBLIC_KEY", "value": pub}, {"key": "TTL_EPOCH", "value": str(ttl)}]}
    p = gql("mutation($i:PodFindAndDeployOnDemandInput){podFindAndDeployOnDemand(input:$i){id costPerHr}}", {"i": inp})["podFindAndDeployOnDemand"]
    print("pod", p["id"], "$/hr", p["costPerHr"], "ttl", minutes, "min", flush=True)
    # local watchdog terminates the pod at TTL even if this session dies
    import subprocess
    subprocess.Popen([sys.executable, __file__, "killat", p["id"], str(ttl)], start_new_session=True,
                     stdout=open(BUDGET.with_name("rp_watchdog.log"), "a"), stderr=subprocess.STDOUT)
    for _ in range(90):
        time.sleep(10)
        q = [x for x in pods() if x["id"] == p["id"]]
        rt = q and q[0].get("runtime")
        if rt and rt.get("ports"):
            for pt in rt["ports"]:
                if pt["privatePort"] == 22 and pt["type"] == "tcp":
                    print(f"ssh -o StrictHostKeyChecking=no root@{pt['ip']} -p {pt['publicPort']}"); return
    print("not ready after 15 min; terminating"); down(p["id"])


def killat(pid, ttl):
    while time.time() < ttl: time.sleep(20)
    try: down(pid)
    except Exception as e: print("killat", pid, e)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("id", nargs="?"); ap.add_argument("--minutes", type=int, default=30)
    ap.add_argument("--cloud", default="SECURE"); ap.add_argument("--gpu", default=GPU); ap.add_argument("--image", default=IMAGE); ap.add_argument("ttl", nargs="?")
    a = ap.parse_args()
    if a.cmd == "up": up(a.minutes, a.cloud, a.image, a.gpu)
    elif a.cmd == "killat": killat(a.id, int(a.ttl))
    elif a.cmd == "ls": [print(p["id"], p["name"], p["desiredStatus"], p["costPerHr"], (p.get("runtime") or {}).get("uptimeInSeconds")) for p in pods()]
    elif a.cmd == "down": [down(p["id"]) for p in pods() if not a.id or p["id"] == a.id]
    elif a.cmd == "reap": reap()
    elif a.cmd == "budget": print("%.0f/%d min today" % budget())
