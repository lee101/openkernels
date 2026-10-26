"""ring-path (CUDA_L) timeline + ablations: python bench/ringdiag.py <moe_dir> <submission.py> [M,...]
env FLAGS='base;-DFAKEDQ;-DFAKEDQ -DNOMMA;-DNOCOMP'  NX=1  KS=1"""
import os, sys, time, importlib.util, torch
D = sys.argv[1]; sys.path.insert(0, D); os.chdir(D)
import reference as ref
spec = importlib.util.spec_from_file_location("sub", os.path.join(D, sys.argv[2])); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
Ms = [int(v) for v in (sys.argv[3] if len(sys.argv) > 3 else "16,32").split(",")]
flags = os.environ.get("FLAGS", "base;-DFAKEDQ;-DFAKEDQ -DNOMMA;-DNOCOMP").split(";")
nx, ks = int(os.environ.get("NX", "1")), int(os.environ.get("KS", "1"))
m.TUNE = False; m._extS = lambda k: None; m.V17_KSF = (); m.FS_CFG = ()
ne = ref.N_EXPERTS; ref.N_EXPERTS = 32; ws = ref.generate_weights(ref.WEIGHTS_SEED); ref.N_EXPERTS = ne
t = time.time(); st = m.prepare(ws); torch.cuda.synchronize(); print("prepare %.1fs" % (time.time() - t), flush=True)
fl = torch.ones(2 ** 27, device="cuda"); sink = torch.empty((), device="cuda")
from concurrent.futures import ThreadPoolExecutor
jobs = [tuple(([] if f == "base" else f.split()) + (["-DPROF"] if pr else [])) for f in flags for pr in (False, True)]
t = time.time()
with ThreadPoolExecutor(8) as ex: list(ex.map(lambda x: m._extL(nx, x), jobs))
print("compiled %d in %.0fs" % (len(jobs), time.time() - t), flush=True)
nsm = torch.cuda.get_device_properties(0).multi_processor_count
NTH = 9  # warps per CTA (8 consumers + producer)


def timed(fn, reps=15):
    for _ in range(2): fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    ts = []
    for _ in range(reps):
        sink.copy_(fl.sum()); e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record(); g.replay(); e1.record(); torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1) * 1e3)
    return sorted(ts)[reps // 2]


for f in flags:
    fx = [] if f == "base" else f.split()
    for prof in (False, True):
        xf = tuple(fx + (["-DPROF"] if prof else []))
        NTH = 1 + next((int(x.split("=")[1]) for x in fx if x.startswith("-DNCW=")), 8)
        st["prof"] = torch.zeros((nsm * NTH * 16 if prof else 0,), dtype=torch.int64, device="cuda")
        for M in Ms:
            data = ref.generate_input(M, "ep8", 403 if M == 32 else 402)
            ids = data["topk_ids"]; l = ids[ids < 32]; c = torch.bincount(l.long(), minlength=32)
            fn = lambda: m._ring(data, st, ks, nx, xf)
            if not prof:
                print(f"time {f} M{M} na {int((c > 0).sum())} maxT {int(c.max())}: {timed(fn):.1f} us", flush=True)
                continue
            for _ in range(3):
                sink.copy_(fl.sum()); fn()
            torch.cuda.synchronize()
            P = st["prof"].view(nsm, NTH, 16).double().cpu()
            t0 = P[:, :, 0][P[:, :, 0] > 0].min()
            mhz = float(((P[:, 0, 9] - P[:, 0, 8]) / (P[:, 0, 7] - P[:, 0, 0]).clamp(min=1)).median()) * 1e3
            pr = P[:, NTH - 1]; co = P[:, :NTH - 1]
            us = lambda x: (x - t0) / 1e3
            q = lambda x, f_: float(x.flatten().quantile(f_))
            span_p = (pr[:, 2] - pr[:, 1]).clamp(min=1); span_c = (co[:, :, 2] - pr[:, 1:2]).clamp(min=1)
            wemp = pr[:, 3] / mhz * 1e3 / span_p; wdep = pr[:, 4] / mhz * 1e3 / span_p
            wful = co[:, :, 3] / mhz * 1e3 / span_c
            print(f"prof {f} M{M}: mhz {mhz:.0f} pro {q(us(pr[:, 1]), .5):.1f} prodEnd p50/max {q(us(pr[:, 2]), .5):.1f}/{q(us(pr[:, 2]), 1):.1f} "
                  f"consEnd p10/50/max {q(us(co[:, :, 2]), .1):.1f}/{q(us(co[:, :, 2]), .5):.1f}/{q(us(co[:, :, 2]), 1):.1f} "
                  f"w2wait {q(us(P[:, :, 6]), .5):.1f} end {q(us(P[:, :, 7]), 1):.1f} | prod waitEmpty {q(wemp, .5):.2f} dep {q(wdep, .5):.2f} | "
                  f"cons waitFull {q(wful, .5):.2f} units p50/max {q(co[:, :, 5], .5):.0f}/{q(co[:, :, 5], 1):.0f}", flush=True)
