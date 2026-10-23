"""muse_bf16_proj/sweep_check.py: submission + config sweep timed in prepare()"""
sub = open("muse_bf16_proj/submission.py").read()
extra = r'''

def _sweep(st, weights):
    import statistics
    global _mod, _DBG
    fl = torch.ones(2 ** 27, device="cuda"); sink = torch.empty((), device="cuda")
    def gt(fn, reps=25):
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn(); fn()
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): fn()
        ts = []
        for _ in range(reps):
            sink.copy_(fl.sum()); a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record(); g.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1e3)
        return statistics.median(ts)
    base = gt(lambda: None)
    cases = []
    for name, M in (("qkv_gate", 1), ("pooler", 1), ("o", 1), ("qkv_gate", 8), ("unpooler", 1)):
        W = weights[name]; n = 1024 if (name == "unpooler" and M <= 8) else W.shape[0]
        x = torch.randn((M, W.shape[1]), device="cuda").bfloat16()
        cases.append((name, M, {"x": x, "config": {"m": M, "n": n, "k": W.shape[1], "layer": name}}, W[:n]))
    print("SW cublas " + " ".join("%s%d:%.1f" % (c[0][:3], c[1], gt(lambda: c[2]["x"] @ c[3].T) - base) for c in cases), flush=True)
    for cfg in (["-DRSN=3", "-DCUN=2"], ["-DRSN=2", "-DCUN=2"], ["-DRSN=2", "-DCUN=3"], ["-DWPC=8", "-DRSN=3", "-DCUN=4"], ["-DWPC=8", "-DRSN=4", "-DCUN=3"], ["-DWPC=12", "-DRSN=3", "-DCUN=2"]):
        _mod, _DBG = None, cfg
        try:
            r = " ".join("%s%d:%.1f" % (c[0][:3], c[1], gt(lambda: kernel_fn(c[2], st)) - base) for c in cases)
        except Exception as e:
            r = "ERR " + str(e)[:80]
        print("SW " + " ".join(cfg) + " " + r, flush=True)
    _mod, _DBG = None, []
'''
sub = sub.replace("    _ext()\n    return st", "    _ext()\n    _sweep(st, weights)\n    return st")
open("muse_bf16_proj/sweep_check.py", "w").write(sub + extra)
