import sys, os, importlib.util, torch
sys.path.insert(0, "muse_attention_decode")
import reference as R
spec = importlib.util.spec_from_file_location("sub", os.environ.get("SUB", "muse_attention_decode/submission_v34.py"))
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S); S._DBG += sys.argv[1:]
flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
def tm(fn, it=30):
    g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        fn(); torch.cuda.synchronize()
        with torch.cuda.graph(g): fn()
    ts = []
    for _ in range(it):
        flush.zero_(); a = torch.cuda.Event(True); b = torch.cuda.Event(True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1e3)
    ts.sort(); return ts[len(ts) // 2]
e0 = tm(lambda: None)
sweep = {("d", 1): [4, 8, 16], ("d", 16): [2, 4, 8], ("d", 32): [2, 4], ("v", 1): [4, 9, 16, 32], ("v", 16): [4, 9, 16], ("v", 32): [4, 8]}
for (kind, B), vals in sweep.items():
    d = R.generate_input(batch=B, ctx=80000, heads="tp8", layer="sliding", seed=11, q_len=8 if kind == "v" else 1, spread="agentic" if B > 1 else "fixed", dist="soft", page=1)
    res = []
    for v in [None] + vals:
        S.OVR.clear()
        if v: S.OVR[(kind, B, True)] = v
        try:
            ok, msg = R.check_implementation(d, R.collect(d, S.kernel_fn(d)))
            t = tm(lambda: S.kernel_fn(d)) - e0
            res.append(f"{v or 'def'}:{t:.1f}{'' if ok else '!BAD'}")
        except Exception as e:
            res.append(f"{v}:ERR {str(e)[:60]}")
    print(f"sliding {kind} B={B}: " + "  ".join(res), flush=True)
