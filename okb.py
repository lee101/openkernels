#!/usr/bin/env python3
"""local judge: okb.py <problem_dir> [-s sub.py] [-b other.py] [--seeds N] [--no-time] [--only LOAD]"""
import argparse, importlib.util, json, re, statistics as st, sys, time
from pathlib import Path
import torch

ap = argparse.ArgumentParser()
ap.add_argument("dir"); ap.add_argument("-s", "--sub", default="submission.py"); ap.add_argument("-b", "--base", default=None)
ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--no-time", action="store_true"); ap.add_argument("--no-check", action="store_true")
ap.add_argument("--only", default=None); ap.add_argument("--experts", type=int, default=0); ap.add_argument("--reps", type=int, default=50); ap.add_argument("--stress", type=int, default=0); ap.add_argument("--kprof", action="store_true")
a = ap.parse_args()
D = Path(a.dir).resolve(); sys.path.insert(0, str(D))

def load(p, name):
    p = Path(p); p = p if p.is_absolute() else (D / p if (D / p).exists() else Path.cwd() / p)
    spec = importlib.util.spec_from_file_location(name, p); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

ref = load(D / "reference.py", "reference"); sub = load(a.sub, "sub_mod")
import os
if os.environ.get("OKB_DBG"): sub._DBG = os.environ["OKB_DBG"].split()
base = load(a.base, "base_mod") if a.base else None
meta = json.load(open(D.parent / "notes" / f"{D.name}.json"))
def parse(line):
    d = {}
    for kv in line.split(","):
        k, v = kv.strip().split(None, 1); d[k] = int(v) if re.fullmatch(r"-?\d+", v) else (float(v) if re.fullmatch(r"-?\d*\.\d+", v) else v)
    return d
blk = re.search(r"must pass these \d+ tests.*?```\n(.*?)```", (D / "TASK.md").read_text(), re.S)
tests = [parse(l) for l in blk.group(1).strip().splitlines()] if blk else []
benches = meta["benchmarks"]
if a.only: benches = [b for b in benches if a.only in b["load"]]

ws = None
if hasattr(ref, "generate_weights"):
    if a.experts and hasattr(ref, "N_EXPERTS"):
        ne = ref.N_EXPERTS; ref.N_EXPERTS = a.experts; ws = ref.generate_weights(ref.WEIGHTS_SEED); ref.N_EXPERTS = ne
    else: ws = ref.generate_weights(ref.WEIGHTS_SEED)
def prep(m):
    if ws is None or not hasattr(m, "prepare"): return None
    t = time.time(); s = m.prepare(ws); torch.cuda.synchronize(); print(f"prepare {m.__name__} {time.time()-t:.1f}s"); return s
S = prep(sub); B = prep(base) if base else None
import inspect
def call(m, data, st):
    return m.kernel_fn(data, st) if len(inspect.signature(m.kernel_fn).parameters) > 1 else m.kernel_fn(data)
def args_of(c): return {k: v for k, v in c.items() if k not in ("load", "weight", "repeats")}
def check(data, out):
    if hasattr(ref, "collect"): out = ref.collect(data, out)
    return ref.check_implementation(data, out, ws) if ws is not None else ref.check_implementation(data, out)

bad = 0
if not a.no_check:
    cases = [("test", t) for t in tests] + [("bench", b) for b in benches]
    for kind, c in cases:
        for i in range(a.seeds):
            kw = args_of(c); kw["seed"] = kw.get("seed", 1000) * 7 + 13 * i + (0 if i else 0)
            if i == 0 and "seed" in c: kw["seed"] = c["seed"]
            data = ref.generate_input(**kw)
            out = call(sub, data, S); torch.cuda.synchronize()
            ok, msg = check(data, out)
            bad += not ok
            if not ok or i == 0: print(f"{'OK ' if ok else 'BAD'} {kind:5} {c.get('load','')[:10]:10} {json.dumps(args_of(c))[:90]} s{kw['seed']}: {msg}")
            del data, out
    print(f"correctness: {'PASS' if not bad else f'{bad} FAIL'}")
def fill(dst, src):
    if isinstance(dst, torch.Tensor): dst.copy_(src)
    elif isinstance(dst, dict):
        for k in dst: fill(dst[k], src[k])
if a.stress:  # judge-style: capture once, refresh inputs in place, replay, check every repetition
    for c in benches:
        data = ref.generate_input(**args_of(c)); fails = 0
        st_ = torch.cuda.Stream(); st_.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st_):
            for _ in range(2): call(sub, data, S)
        torch.cuda.current_stream().wait_stream(st_); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): out = call(sub, data, S)
        for r in range(a.stress):
            kw = args_of(c); kw["seed"] = 100000 + 7919 * r; fresh = ref.generate_input(**kw); fill(data, fresh)
            g.replay(); torch.cuda.synchronize()
            ok, msg = check(data, tuple(o.clone() for o in out) if isinstance(out, tuple) else out.clone())
            if not ok: fails += 1; print(f"  STRESS FAIL {c['load']} rep {r}: {msg}")
        bad += fails; print(f"stress {c['load']:10} {a.stress - fails}/{a.stress} ok")
        del g, data, out
if a.no_time: sys.exit(1 if bad else 0)

flush = torch.ones(int(0.5 * 2**30) // 4, dtype=torch.float32, device="cuda"); sink = torch.empty((), device="cuda")  # read-only L2 flush
def graph(m, s, data):
    st_ = torch.cuda.Stream(); st_.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st_):
        for _ in range(2): call(m, data, s)
    torch.cuda.current_stream().wait_stream(st_); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): call(m, data, s)
    return g
def tm(g):
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    sink.copy_(flush.sum()); e0.record(); g.replay(); e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) * 1e3
NB = load(D / "_bytes.py", "nb") if (D / "_bytes.py").exists() else None
empty = torch.cuda.CUDAGraph()
with torch.cuda.graph(empty): pass
loads = {}
for c in benches:
    data = ref.generate_input(**args_of(c))
    gs = graph(sub, S, data); gb = graph(base, B, data) if base else None
    for _ in range(3): tm(gs); gb and tm(gb)
    ts, tb, r = [], [], []
    for i in range(a.reps):
        x = tm(gs); y = tm(gb) if gb else None
        ts.append(x); y is not None and (tb.append(y), r.append(y / x))
    ms = st.median(ts); line = f"{c['load']:10} {json.dumps({k: v for k, v in args_of(c).items() if k != 'seed'})[:50]:50} {ms:9.1f}us"
    if NB: line += f" {NB.nbytes(data) / ms / 1e3:7.1f}GB/s"
    if gb: line += f"  base {st.median(tb):9.1f}us  x{st.median(r):.3f}"
    print(line)
    if a.kprof:
        from torch.profiler import profile, ProfilerActivity
        sink.copy_(flush.sum()); torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as pr:
            gs.replay(); torch.cuda.synchronize()
        evs = [e for e in pr.events() if e.device_type.name == "CUDA"]
        evs.sort(key=lambda e: e.time_range.start)
        t0 = evs[0].time_range.start if evs else 0
        for e in evs: print(f"    {(e.time_range.start - t0):8.1f} +{e.time_range.elapsed_us():7.1f}us  {e.name[:90]}")
    L = loads.setdefault(c["load"], [0.0, 0.0]); L[0] += ms * c["weight"]; L[1] += (st.median(tb) if gb else 0) * c["weight"]
    del gs, gb, data; torch.cuda.empty_cache()
if base:
    sp = [b / s for s, b in loads.values()]
    gm = 1.0
    for x in sp: gm *= x
    print(f"geomean speedup x{gm ** (1 / len(sp)):.3f}  per-load " + " ".join(f"{k}:{b/s:.2f}" for k, (s, b) in loads.items()))
else:
    print("load times " + " ".join(f"{k}:{s:.0f}" for k, (s, _) in loads.items()))
