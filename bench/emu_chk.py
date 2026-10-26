#!/usr/bin/env python3
"""attention correctness of the sm_90a verify path (attnw) on sm_86: wgmma emulated in software (-DWGEMU=1, small ring).
emu_chk.py <submission.py> [--seeds N] [--only substr] [--all] [--dbg "-DX=1 ..."] [--stress N]"""
import argparse, importlib.util, json, re, sys
from pathlib import Path
import torch

R = Path(__file__).resolve().parent.parent
D = R / "muse_attention_decode"
ap = argparse.ArgumentParser()
ap.add_argument("sub"); ap.add_argument("--seeds", type=int, default=1); ap.add_argument("--only", default=None)
ap.add_argument("--all", action="store_true"); ap.add_argument("--dbg", default=""); ap.add_argument("--stress", type=int, default=0)
ap.add_argument("--native", action="store_true"); ap.add_argument("--ovr", default=""); ap.add_argument("--vovr", default=""); ap.add_argument("--kind", default=""); ap.add_argument("--dfl", type=int, default=-1); ap.add_argument("--l1", action="store_true"); ap.add_argument("--daovr", default="")
a = ap.parse_args()
sys.path.insert(0, str(D))


def load(p, name):
    spec = importlib.util.spec_from_file_location(name, p); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


ref = load(D / "reference.py", "reference")
sub = load(Path(a.sub).resolve(), "sub_mod")
if not a.native:
    sub._DBG = ["-DWGEMU=1", "-DRSW=2"] + a.dbg.split()
    sub._hopper = lambda: True
elif a.dbg:
    sub._DBG = a.dbg.split()
for kv in filter(None, a.ovr.split(";")):
    k, v = kv.split("="); kk = k.split(","); sub.OVR[(kk[0], int(kk[1]), bool(int(kk[2])))] = int(v)
if a.dfl >= 0: sub.DFLOVR = a.dfl
if a.vovr: sub.VOVR = tuple(int(x) for x in a.vovr.split(","))
if a.daovr: sub.DAOVR = tuple(int(x) for x in a.daovr.split(","))
meta = json.load(open(R / "notes" / "muse_attention_decode.json"))


def parse(line):
    d = {}
    for kv in line.split(","):
        k, v = kv.strip().split(None, 1); d[k] = int(v) if re.fullmatch(r"-?\d+", v) else (float(v) if re.fullmatch(r"-?\d*\.\d+", v) else v)
    return d


blk = re.search(r"must pass these \d+ tests.*?```\n(.*?)```", (D / "TASK.md").read_text(), re.S)
tests = [parse(l) for l in blk.group(1).strip().splitlines()]
cases = [("test", t) for t in tests] + [("bench", b) for b in meta["benchmarks"]]
if not a.all: cases = [c for c in cases if c[1].get("q_len", 1) == 8]
if a.l1: cases = [c for c in cases if c[1].get("q_len", 1) == 1]
if a.kind: cases = [c for c in cases if c[0] == a.kind]
if a.only: cases = [c for c in cases if a.only in json.dumps(c[1])]
args_of = lambda c: {k: v for k, v in c.items() if k not in ("load", "weight", "repeats")}
bad = 0
for kind, c in cases:
    for i in range(a.seeds):
        kw = args_of(c); kw["seed"] = c.get("seed", 1000 * 7 + 13 * i) + 101 * i
        data = ref.generate_input(**kw)
        out = sub.kernel_fn(data); torch.cuda.synchronize()
        ok, msg = ref.check_implementation(data, ref.collect(data, out))
        bad += not ok
        print(f"{'OK ' if ok else 'BAD'} {kind:5} {json.dumps(args_of(c))[:110]} s{kw['seed']}: {msg}", flush=True)
        del data, out
if a.stress:
    for kind, c in cases:
        if kind != "bench": continue
        data = ref.generate_input(**args_of(c)); fails = 0
        st_ = torch.cuda.Stream(); st_.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st_):
            for _ in range(2): sub.kernel_fn(data)
        torch.cuda.current_stream().wait_stream(st_); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): out = sub.kernel_fn(data)
        for r in range(a.stress):
            kw = args_of(c); kw["seed"] = 100000 + 7919 * r; fresh = ref.generate_input(**kw)
            for k, v in data.items():
                if isinstance(v, torch.Tensor): v.copy_(fresh[k])
            g.replay(); torch.cuda.synchronize()
            ok, msg = ref.check_implementation(data, ref.collect(data, out.clone()))
            if not ok: fails += 1; print(f"  STRESS FAIL {c['load']} rep {r}: {msg}")
        bad += fails; print(f"stress {c['load']:10} {a.stress - fails}/{a.stress} ok", flush=True)
        del g, data, out
print("correctness:", "PASS" if not bad else f"{bad} FAIL")
sys.exit(1 if bad else 0)
