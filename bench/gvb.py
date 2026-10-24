import sys, os, importlib.util, torch
sub = os.environ.get("SUB", "muse_bf16_proj/submission_v10.py")
spec = importlib.util.spec_from_file_location("sub", sub)
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S); S._DBG += sys.argv[1:]
torch.manual_seed(0)
shp = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096)}
W = {k: (torch.randn(v, device="cuda") / v[1] ** .5).bfloat16() for k, v in shp.items()}
W["router"] = torch.randn(256, 8192, device="cuda") / 90
st = S.prepare(W); ext = S._ext()
flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
def tm(fn, it=int(os.environ.get("IT", "40"))):
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
cases = [(l, int(m)) for l, m in (c.split(":") for c in os.environ.get("CASES", "o:1,o:16,o:32,qkv_gate:1,qkv_gate:32,pooler:16,unpooler:32").split(","))]
for layer, m in cases:
    n = 1024 if layer == "unpooler" and m <= 8 else W[layer].shape[0]
    x = torch.randn(m, W[layer].shape[1], device="cuda").bfloat16()
    out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    A = st[layer][0]
    r = {"gv": tm(lambda: ext.run4(A, x, out, st["wp"], st["cnt"], n, S.XB))}
    ref = x.float() @ W[layer][:n].float().T
    ext.run4(A, x, out, st["wp"], st["cnt"], n, S.XB); err = ((out.float() - ref).norm() / ref.norm()).item()
    if not os.environ.get("ONLYGV"):
        r["sk"] = tm(lambda: ext.run(A, st["none"], x, out, st["ws"], st["cnt"], n))
        r["cb"] = tm(lambda: x @ W[layer][:n].T)
    print(f"{layer:9s} m={m:3d} " + " ".join(f"{k} {v - e0:6.1f}" for k, v in r.items()) + f" err {err:.1e}", flush=True)
