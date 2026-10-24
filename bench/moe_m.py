import sys, os, importlib.util, torch
sys.path.insert(0, "muse_moe_w4a16")
import reference as R
spec = importlib.util.spec_from_file_location("sub", os.environ.get("SUB", "muse_moe_w4a16/submission.py"))
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S); S._DBG += sys.argv[1:]
W = R.generate_weights(1); st = S.prepare(W); S.WGLMAX = 1024
flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
def tm(fn, it=10):
    g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        fn(); torch.cuda.synchronize()
        with torch.cuda.graph(g): fn()
    ts = []
    for _ in range(it):
        flush.zero_(); a = torch.cuda.Event(True); b = torch.cuda.Event(True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1e3)
    ts.sort(); return ts[len(ts) // 2]
for M in map(int, os.environ.get("MS", "128,256,512,753,1024").split(",")):
    d = R.generate_input(bs=M, sharding="ep8", seed=406)
    res = []
    for wgl in (False, True):
        S.WGL = wgl
        try:
            ok, msg = R.check_implementation(d, S.kernel_fn(d, st), W)
            res.append(f"{'wgl' if wgl else 'moe'} {tm(lambda: S.kernel_fn(d, st)):7.1f}{'' if ok else ' BAD ' + str(msg)[:50]}")
        except Exception as e:
            res.append(f"{wgl} ERR {str(e)[:80]}")
    print(f"M={M}: " + " | ".join(res), flush=True)
