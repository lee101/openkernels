"""per-warp phase timeline of the MoE kernel: python prof.py [bs ...]"""
import os, sys, statistics as st
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reference as R, submission as S

S._DBG = ["-DPROF"] + os.environ.get("OKB_DBG", "").split()
S.DIAG = False
R.N_EXPERTS = 32; ws = R.generate_weights(R.WEIGHTS_SEED); R.N_EXPERTS = 256
stt = S.prepare(ws)
ghz = float(os.environ.get("GHZ", "1.98"))
flush = torch.empty(2**28, dtype=torch.float32, device="cuda")
bss = [int(b) for b in sys.argv[1:]] or [1, 8, 16, 32, 128, 256]
seeds = {1: 401, 16: 402, 32: 403, 8: 404, 128: 405, 256: 406}
pct = lambda v, q: sorted(v)[min(len(v) - 1, int(q * len(v)))]
for bs in bss:
    d = R.generate_input(bs=bs, sharding="ep8", seed=seeds.get(bs, 1))
    for _ in range(3): S.kernel_fn(d, stt)
    torch.cuda.synchronize()
    flush.add_(1); torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record(); S.kernel_fn(d, stt); e1.record(); torch.cuda.synchronize()
    P = stt["prof"].view(-1, 16).cpu()
    nw = torch.cuda.get_device_properties(0).multi_processor_count * int(next((d[6:].split("x")[0] for d in S._DBG if d.startswith("-DCFG=")), 16 if S._big() else 8))
    P = P[:nw]
    t0 = int(P[:, 0].min())
    us = lambda col: [(int(v) - t0) / 1e3 for v in P[:, col]]
    cyc = lambda col: [int(v) / ghz / 1e3 for v in P[:, col]]
    items = P[:, 8].tolist()
    act = [i for i in range(len(items)) if items[i] > 0]
    print(f"bs {bs}: event {e0.elapsed_time(e1) * 1e3:.1f}us  span {max(us(7)):.1f}us  na {int(P[0, 11])} ks {int(P[0, 12])} warps {len(items)} active {len(act)} items/warp max {max(items)}")
    if not act: act = list(range(len(items)))
    mhz = [(int(P[i, 14]) - int(P[i, 13])) / max(1, int(P[i, 7]) - int(P[i, 0])) * 1e3 for i in act]
    print(f"  sm clock MHz p10 {pct(mhz, .1):.0f} p50 {pct(mhz, .5):.0f}")
    print(f"  start  p50 {pct(us(0), .5):.1f} p99 {pct(us(0), .99):.1f} | routed p50 {pct(us(1), .5):.1f} max {max(us(1)):.1f}")
    print(f"  items done p10 {pct(us(5), .1):.1f} p50 {pct(us(5), .5):.1f} max {max(us(5)):.1f} | comb wait end max {max(us(6)):.1f} | end max {max(us(7)):.1f}")
    for nm, col in (("g1 busy", 2), ("g2 busy", 3), ("g2 dep-wait", 4), ("tma wait", 9), ("epilogue", 15)):
        v = [cyc(col)[i] for i in act]
        print(f"  {nm:12} per-warp us p10 {pct(v, .1):6.2f} p50 {pct(v, .5):6.2f} p90 {pct(v, .9):6.2f} max {max(v):6.2f}")
    E = stt["prof"][65536 * 16:].view(64, 96, 2).cpu()
    sym = {1: "a", 2: "A", 3: "r", 4: "e", 5: "E", 6: "x", 7: "R", 8: "I", 9: "N", 10: "T", 11: "D", 12: "P"}
    for wv in (0, 1, 5, 17, 40):
        ev = [(int(a), (int(b) - t0) / 1e3) for a, b in E[wv].tolist() if a]
        print(f"  w{wv}: " + " ".join(f"{sym.get(a, a)}{t:.1f}" for a, t in ev[:40]))
