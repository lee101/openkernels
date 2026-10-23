#!/usr/bin/env python3
"""RunPod H200 with hard cost guards: TTL'd pods, daily minute budget, always-terminate.
rp.py up [--minutes N]   create pod (sglang judge image), print ssh
rp.py ls | down [id] | reap | budget"""
import argparse, datetime as dt, json, os, sys, time, urllib.request
from pathlib import Path

API = "https://api.runpod.io/graphql"
TAG = "okc-h200"
IMAGE = "lmsysorg/sglang:dev-muse-spark-open"
GPU = "NVIDIA H200"
BUDGET = Path(__file__).with_name("notes") / "rp_budget.json"
CAP_MIN = 120  # per-day GPU-minute cap; raise deliberately


def key():
    k = os.environ.get("RUNPOD_API_KEY", "").strip()
    if k: return k
    for l in open("/vfast/data/code/app-site/.env"):
        if l.startswith("RUNPOD_API_KEY="): return l.split("=", 1)[1].strip().strip('"')
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


def up(minutes, cloud):
    budget(minutes)
    pub = Path.home().joinpath(".ssh/id_ed25519.pub").read_text().strip() if Path.home().joinpath(".ssh/id_ed25519.pub").exists() else Path.home().joinpath(".ssh/id_rsa.pub").read_text().strip()
    ttl = int(time.time()) + minutes * 60
    # pod self-destructs at TTL even if this machine dies
    cmd = (f"mkdir -p ~/.ssh && echo '{pub}' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys; "
           "(apt-get install -y openssh-server >/dev/null 2>&1 || true); service ssh start || /usr/sbin/sshd; "
           f"sleep {minutes * 60}; kill 1")
    inp = {"name": f"{TAG}-{int(time.time())}", "imageName": IMAGE, "gpuTypeId": GPU, "cloudType": cloud, "gpuCount": 1,
           "containerDiskInGb": 80, "volumeInGb": 0, "ports": "22/tcp", "startSsh": True,
           "dockerArgs": f"bash -lc \"{cmd}\"", "env": [{"key": "TTL_EPOCH", "value": str(ttl)}]}
    p = gql("mutation($i:PodFindAndDeployOnDemandInput){podFindAndDeployOnDemand(input:$i){id costPerHr}}", {"i": inp})["podFindAndDeployOnDemand"]
    print("pod", p["id"], "$/hr", p["costPerHr"], "ttl", minutes, "min")
    for _ in range(120):
        time.sleep(10)
        q = [x for x in pods() if x["id"] == p["id"]]
        rt = q and q[0].get("runtime")
        if rt and rt.get("ports"):
            for pt in rt["ports"]:
                if pt["privatePort"] == 22 and pt["type"] == "tcp": print(f"ssh -o StrictHostKeyChecking=no root@{pt['ip']} -p {pt['publicPort']}"); return
    print("not ready after 20 min; terminating"); down(p["id"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("id", nargs="?"); ap.add_argument("--minutes", type=int, default=30)
    ap.add_argument("--cloud", default="SECURE")
    a = ap.parse_args()
    if a.cmd == "up": up(a.minutes, a.cloud)
    elif a.cmd == "ls": [print(p["id"], p["name"], p["desiredStatus"], p["costPerHr"], (p.get("runtime") or {}).get("uptimeInSeconds")) for p in pods()]
    elif a.cmd == "down": [down(p["id"]) for p in pods() if not a.id or p["id"] == a.id]
    elif a.cmd == "reap": reap()
    elif a.cmd == "budget": print("%.0f/%d min today" % budget())
