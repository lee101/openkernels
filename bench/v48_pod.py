#!/usr/bin/env python3
"""verify attention variants on a Hopper pod: v48_pod.py [prof]  (judge-like: graph replay, cold L2, median)"""
import importlib.util, os, sys, torch
sys.path.insert(0, "muse_attention_decode")
import reference as R
spec = importlib.util.spec_from_file_location("sub", os.environ.get("SUB", "muse_attention_decode/submission_v48.py"))
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
PROF = "prof" in sys.argv[1:]
FL = torch.ones(64 << 20, dtype=torch.float32, device="cuda")


def gt(fn, reps=30):
    s_ = torch.cuda.Stream(); s_.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s_):
        for _ in range(2): fn()
    torch.cuda.current_stream().wait_stream(s_); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    ts = []
    for _ in range(reps + 3):
        FL.sum()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record(); g.replay(); e1.record(); e1.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3)
    ts = sorted(ts[3:]); return ts[len(ts) // 2]


def timeline(data):
    S._VPROF = torch.zeros((65536 + 2048,), dtype=torch.int64, device="cuda")
    for _ in range(3):
        S._VPROF.zero_(); FL.sum(); S.kernel_fn(data)
    torch.cuda.synchronize()
    V = S._VPROF.cpu()
    P = [r for r in V[:65536].view(-1, 8).tolist() if r[0]]
    W = [r for r in V[65536:].view(-1, 4).tolist() if r[2]]
    S._VPROF = None
    wl = ""
    if W:
        med = lambda k: sorted(r[k] for r in W)[len(W) // 2] / 1e3
        wl = " | lanes %d full-wait %.1f empty-wait %.1f cons-loop %.1f prod-loop %.1f us (p50)" % (len(W), med(0), med(1), med(2), med(3))
    t0 = min(r[0] for r in P); out = []
    for k, nm in enumerate(["start", "q", "prolog", "loop", "parts", "ranked", "all-in", "end"]):
        c = sorted((r[k] - t0) / 1e3 for r in P if r[k])
        if c: out.append("%s %.1f/%.1f" % (nm, c[len(c) // 2], c[-1]))
    return " ".join(out) + wl


if PROF: S._DBG = S._DBG + ["-DPROF"]
big = 1 << 30
if PROF:
    SH = {(1, True): [(-1, 128, 8, big, 8, 1), (-1, 129, 8, big, 136, 1), (-1, 129, 8, big, 200, 1)],
          (16, True): [(-1, 256, 8, 32, 0, 1), (-1, 192, 8, 32, 108, 1), (-1, 256, 8, 32, 136, 1), (-1, 192, 8, 32, 200, 1)],
          (32, True): [(-1, 256, 8, 32, 0, 1), (-1, 256, 8, 32, 136, 1), (-1, 256, 8, 32, 200, 1)],
          (16, False): [(-1, 256, 8, 32, 0, 1), (-1, 256, 8, 32, 64, 1)]}
else:
    SH = {(1, True): [(-1, 128, 8, big, 8, 1)] + [(-1, k, 8, big, x, 1) for k in (129, 172, 258) for x in (136, 140, 200, 204)] + [(-1, 129, 8, big, x, 1) for x in (172, 236)],
          (16, True): [(-1, k, 8, 32, x, 1) for k in (176, 192, 208, 224) for x in (0, 64, 108)] + [(-1, k, 8, 32, x, 1) for k in (137, 192, 256) for x in (136, 200, 204)],
          (32, True): [(-1, 256, 8, 32, x, 1) for x in (0, 64, 136, 200, 204, 236)] + [(-1, k, 8, 32, x, 1) for k in (192, 137) for x in (200, 204)],
          (16, False): [(-1, 256, 8, 32, x, 1) for x in (0, 64, 72, 96, 0, 64)],
          (32, False): [(-1, 256, 8, 32, x, 1) for x in (0, 64, 72, 96, 0, 64)],
          (1, False): [(520, 607, 128, big, 0, 1), (-1, 607, 128, big, 64, 1)]}
if os.environ.get("CFG"): SH = eval(os.environ["CFG"])
if os.environ.get("DBG"): S._DBG = S._DBG + os.environ["DBG"].split()
DEC = eval(os.environ.get("DEC", "[(16, False)]"))
only = os.environ.get("ONLY")
for (B, sl), cfgs in SH.items():
    tag = "v%d%s" % (B, "s" if sl else "f")
    if only and tag not in only.split(","): continue
    d = R.generate_input(batch=B, ctx=80000, heads="tp8", layer="sliding" if sl else "full", seed=7, q_len=8,
                         spread="agentic" if B > 1 else "fixed", dist="soft" if sl else "coherent", page=1)
    for c in cfgs:
        S.VOVR = c
        try:
            ok, msg = R.check_implementation(d, R.collect(d, S.kernel_fn(d)))
            if PROF: print("VD %s %s %s" % (tag, c, timeline(d)), flush=True)
            else: print("T %s %s %.2f %s" % (tag, c, gt(lambda: S.kernel_fn(d)), "" if ok else "BAD " + msg[:80]), flush=True)
        except Exception as e:
            print("T %s %s ERR %s" % (tag, c, str(e)[:100]), flush=True)
    S.VOVR = None
    del d; torch.cuda.empty_cache()
if not PROF:
    for e in DEC:
        B, sl = e[:2]
        d = R.generate_input(batch=B, ctx=80000, heads="tp8", layer="sliding" if sl else "full", seed=7, q_len=1, spread="agentic" if B > 1 else "fixed", dist="soft" if sl else "coherent", page=1)
        for fl in (e[2:] or (None,)):
            S.DAOVR = None
            if isinstance(fl, tuple) and fl[0] == "a": S.DAOVR = (fl[1], fl[2]); fl = None
            elif isinstance(fl, tuple): S.OVR[("d", B, bool(sl))] = fl[1]; fl = fl[0]
            else: S.OVR.pop(("d", B, bool(sl)), None)
            S.DFLOVR = fl
            ok, msg = R.check_implementation(d, R.collect(d, S.kernel_fn(d)))
            print("T d%d%s fl%s ns%s ad%s %.2f %s" % (B, "s" if sl else "f", fl, S.OVR.get(("d", B, bool(sl))), S.DAOVR, gt(lambda: S.kernel_fn(d)), "" if ok else "BAD " + msg[:80]), flush=True)
        S.DFLOVR = None; S.DAOVR = None
