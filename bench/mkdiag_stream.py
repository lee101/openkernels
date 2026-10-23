"""muse_bf16_proj/stream_check.py = submission + single-launch streaming calibration (graph replay, cold L2)"""
sub = open("muse_bf16_proj/submission.py").read()
cu = open("bench/tma_stream.cu").read()
extra = '''

_TS = r"""''' + cu + '''"""


def _calib(weights):
    import statistics
    m = load_inline("tma_stream_cal", cpp_sources="double run(torch::Tensor,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);",
                    cuda_sources=_TS, functions=["run"], extra_cuda_cflags=["-O3"], verbose=False)
    fl = torch.ones(2 ** 27, device="cuda"); sink = torch.empty((), device="cuda")
    def gt(fn, reps=30):
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
    print("CAL empty-graph %.2f" % base, flush=True)
    for mb in (8.4, 16.8, 37.7, 67.1):
        buf = torch.empty(int(mb * 1e6), dtype=torch.uint8, device="cuda").random_(0, 255)
        row = []
        for wpc, S, stg in ((16, 2, 4608), (16, 6, 2048), (8, 4, 4608), (8, 6, 4096), (4, 6, 8192), (4, 3, 16384), (2, 6, 16384), (1, 12, 16384)):
            row.append("%dx%dx%d:%.1f" % (wpc, S, stg, gt(lambda: m.run(buf, wpc, S, stg, 0, 0, 0)) - base))
        n = {8.4: 1024, 16.8: 8192, 37.7: 2304, 67.1: 4096}[mb]
        k = int(mb * 1e6 / 2 / n)
        W = torch.randn((n, k), device="cuda").bfloat16(); x = torch.randn((1, k), device="cuda").bfloat16()
        row.append("cublas:%.1f" % (gt(lambda: x @ W.T) - base))
        print("CAL %.1fMB " % mb + " ".join(row), flush=True)
'''
sub = sub.replace("    _ext()\n    return st", "    _ext()\n    _calib(weights)\n    return st")
open("muse_bf16_proj/stream_check.py", "w").write(sub + extra)
