import sys, os, time, importlib.util, torch
D = sys.argv[1]
sys.path.insert(0, D)
os.chdir(D)
import reference as ref
spec = importlib.util.spec_from_file_location("sub", os.path.join(D, sys.argv[2] if len(sys.argv) > 2 else "submission_v34.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
cfgs = [tuple(int(v) if v.isdigit() else v for v in c.split(",")) for c in os.environ.get("CFGS", "fs,0,3").split()]
Ms = [int(v) for v in os.environ.get("MS", "1,4,8,16,32").split(",")]
m.TUNE = False
if os.environ.get("ONLYF"):
    m._extL = lambda nx1=0: None
    m._extS = lambda k: None
    m.V17_KSF = ()
ne = ref.N_EXPERTS; ref.N_EXPERTS = 32; ws = ref.generate_weights(ref.WEIGHTS_SEED); ref.N_EXPERTS = ne
t = time.time(); st = m.prepare(ws); torch.cuda.synchronize(); print("prepare %.1fs" % (time.time() - t), flush=True)
fl = torch.ones(2 ** 27, device="cuda"); sink = torch.empty((), device="cuda")
for c in cfgs:
    if c[0] == "fs": m._extF(*c[1:])
if not os.environ.get("NOCHK"):
    for M in Ms:
        for dist in ("normal", "skewed"):
            data = ref.generate_input(M, "ep8", 1 + 17 * M, dist=dist)
            for c in [c for c in cfgs if c[0] == "fs" or os.environ.get("CHKALL")]:
                m.FORCE = c
                for rep in range(2):
                    out = m.kernel_fn(data, st); torch.cuda.synchronize()
                    ok, msg = ref.check_implementation(data, out, ws)
                    if not ok or rep == 1: print(f"M{M} {dist} {c} rep{rep}: {'OK ' if ok else 'BAD'} {msg}", flush=True)
if os.environ.get("TIME"):
    for M in Ms:
        res = {}
        for sd in (5, 6, 7):
            data = ref.generate_input(M, "ep8", sd)
            for c in cfgs:
                m.FORCE = c
                for _ in range(2): m.kernel_fn(data, st)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g): m.kernel_fn(data, st)
                ts = []
                for _ in range(15):
                    sink.copy_(fl.sum()); e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                    e0.record(); g.replay(); e1.record(); torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1) * 1e3)
                res.setdefault(c, []).append(sorted(ts)[7]); del g
        print(f"time M{M}: " + "  ".join(f"{'.'.join(map(str, c))}={sum(v) / len(v):.1f}" for c, v in res.items()), flush=True)
if os.environ.get("PROFF"):
    m._DBG = ["-DPROF"]; m._modF.clear(); st["profF"] = torch.zeros(4096, dtype=torch.int64, device="cuda")
    for M in Ms:
        data = ref.generate_input(M, "ep8", 5)
        for c in [c for c in cfgs if c[0] == "fs"]:
            for _ in range(3):
                st["profF"].zero_(); sink.copy_(fl.sum()); m._fs(data, st, c)
            torch.cuda.synchronize()
            P = st["profF"].view(-1, 16).cpu()
            n = (P[:, 0] > 0).sum().item(); P = P[:n].double(); t0 = P[:, 0].min()
            q = lambda k, f: (P[:, k] - t0).quantile(f).item() / 1e3
            mm = n
            print(f"prof M{M} {c} ctas {n}: pro p50/max {q(1,.5):.1f}/{q(1,1):.1f} A p10/50/max {q(2,.1):.1f}/{q(2,.5):.1f}/{q(2,1):.1f} "
                  f"B {q(3,.1):.1f}/{q(3,.5):.1f}/{q(3,1):.1f} bar {((P[:mm, 4] - t0).max().item() / 1e3):.1f} fin {((P[:, 7] > 0).nonzero().flatten().tolist()[:3])} c4 {((P[:, 4] - t0) * (P[:, 7] > 0)).max().item() / 1e3:.1f} c1 {((P[:mm, 6] - t0).max().item() / 1e3):.1f} c2 {((P[:mm, 7] - t0).max().item() / 1e3):.1f} end {((P[:mm, 5] - t0).max().item() / 1e3):.1f}", flush=True)
