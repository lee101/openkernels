import sys, os, importlib.util, torch
spec = importlib.util.spec_from_file_location("sub", os.environ.get("SUB", "muse_bf16_proj/submission_v10.py"))
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S); S._DBG += ["-DDBGT=1"] + sys.argv[1:]
torch.manual_seed(0)
shp = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096)}
W = {k: (torch.randn(v, device="cuda") / v[1] ** .5).bfloat16() for k, v in shp.items()}
W["router"] = torch.randn(256, 8192, device="cuda") / 90
st = S.prepare(W); ext = S._ext(); flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
for c in os.environ.get("CASES", "o:1,o:32,qkv_gate:32").split(","):
    layer, m = c.split(":"); m = int(m)
    n = 1024 if layer == "unpooler" and m <= 8 else W[layer].shape[0]
    x = torch.randn(m, W[layer].shape[1], device="cuda").bfloat16(); out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    tb = st["wp"].view(torch.int64)[(16 << 20) // 2 - 32768:]
    res = []
    for it in range(5):
        tb.zero_(); flush.zero_(); torch.cuda.synchronize()
        ext.run4(st[layer][0], x, out, st["wp"], st["cnt"], n, S.XB); torch.cuda.synchronize()
        t = tb[:132 * 16 * 4].view(-1, 4).double()
        t = t[t[:, 0] > 0]; b = t[:, 0].min()
        t = ((t - b) / 1e3).cpu()
        res.append(t)
    t = res[-1]
    q = lambda v: " ".join(f"{x:5.1f}" for x in torch.quantile(v, torch.tensor([0., .5, .9, 1.], dtype=torch.float64)).tolist())
    print(f"{layer} m={m} warps={len(t)} | start {q(t[:,0])} | xready {q(t[:,1])} | streamend {q(t[:,2])} | end {q(t[:,3])}", flush=True)
