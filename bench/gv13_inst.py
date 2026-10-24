import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../muse_bf16_proj")
import submission_v13 as s
m = s._gvext()
ws = torch.zeros(1 << 22, device="cuda"); cnt = torch.zeros(1024, dtype=torch.int32, device="cuda")
esc = torch.full((1 << 20,), -1, dtype=torch.int32, device="cuda")
Ns = {k: (1024 if k == "unpooler_s" else {"pooler": 4096, "qkv_gate": 2304, "o": 8192, "unpooler": 8192}[k]) for k in s.GVGEO}
for i, c in enumerate(s.GVI):
    fmt, mt, st, wr, wk, ku, S = c
    K = S * wk * ku * 64
    keys = [k for k, g in s.GVGEO.items() if (wr, wk, ku, S) in g and (k != "unpooler_s" or mt == 1) and (k != "unpooler" or mt > 1)]
    N = Ns[keys[0]]
    W = torch.zeros(N * K * 2, dtype=torch.uint8, device="cuda")
    x = torch.randn(8 * mt, K, device="cuda").bfloat16()
    out = torch.empty(8 * mt, N, dtype=torch.bfloat16, device="cuda")
    r = m.gvrun(i, W, x, out, esc, ws, cnt, 0)
    torch.cuda.synchronize()
    if r: print("FAIL", i, c, keys, r)
print("done")
