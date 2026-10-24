"""per-shape local timing: submission (graph replay, cold L2) vs torch.matmul; usage: gv13_time.py [sub.py] [reps]"""
import sys, os, importlib.util, torch
D = os.path.dirname(os.path.abspath(__file__)) + "/../muse_bf16_proj"
sys.path.insert(0, D)
spec = importlib.util.spec_from_file_location("sub", sys.argv[1] if len(sys.argv) > 1 else D + "/submission_v13.py")
s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
import reference as ref
R = int(sys.argv[2]) if len(sys.argv) > 2 else 30
ws = ref.generate_weights(ref.WEIGHTS_SEED)
st = s.prepare(ws)
flush = torch.empty(64 << 20, dtype=torch.float32, device="cuda")
pr = torch.cuda.get_device_properties(0)
BW = {"NVIDIA GeForce RTX 3090 Ti": 1008e9, "NVIDIA H200": 4.8e12}.get(pr.name, 1e12)


def graph(fn):
    s_ = torch.cuda.Stream(); s_.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s_):
        fn(); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): fn()
    torch.cuda.synchronize()
    return g


def pair(ga, gb):
    ra, rb = [], []
    for i in range(R):
        for g, r in ((ga, ra), (gb, rb)) if i % 2 else ((gb, rb), (ga, ra)):
            flush.add_(1); a = torch.cuda.Event(True); b = torch.cuda.Event(True)
            a.record(); g.replay(); b.record(); torch.cuda.synchronize(); r.append(a.elapsed_time(b) * 1e3)
    ra.sort(); rb.sort()
    return ra[R // 2], rb[R // 2]


print(f"{pr.name}; median of {R} cold-L2 graph replays (us, incl. launch); eff = weight+io bytes / time / {BW / 1e9:.0f} GB/s")
print(f"{'layer':9} {'m':>4} {'path':24} {'ours':>7} {'torch':>7} {'x':>6} {'eff':>5}")
rows = []
for layer in ("qkv_gate", "o", "pooler", "unpooler", "router"):
    for m in (1, 8, 16, 32, 128, 256):
        data = ref.generate_input(m, layer, 7)
        x, n = data["x"], data["config"]["n"]
        W = ws[layer][:n]
        ga = graph(lambda: s.kernel_fn(data, st))
        gb = graph((lambda: x.float() @ W.T) if layer == "router" else (lambda: x @ W.T))
        a, b = pair(ga, gb)
        p = st["gv"].get((layer, s._bk(m)))
        path = str(s.GVI[p[0]]) if p is not None and p[4] == n else "v12"
        byt = W.numel() * W.element_size() + x.numel() * 2 + m * n * (4 if layer == "router" else 2)
        rows.append((layer, m, a, b))
        print(f"{layer:9} {m:4} {path:24} {a:7.2f} {b:7.2f} {b / a:6.3f} {byt / (a * 1e-6) / BW:5.2f}", flush=True)
import math
for loadm in (1, 8, 16, 32, 128, 256):
    r = [x for x in rows if x[1] == loadm]
    print(f"load m={loadm}: ours {sum(x[2] for x in r):.1f} torch {sum(x[3] for x in r):.1f} -> {sum(x[3] for x in r) / sum(x[2] for x in r):.3f}")
