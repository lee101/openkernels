"""build muse_moe_w4a16/diag_check.py = submission.py + an H200 microbenchmark sweep run from prepare()"""
import sys
sub = open("muse_moe_w4a16/submission.py").read()
cu = open("bench/tma_stream.cu").read()
extra = '''

_TMA_SRC = r"""''' + cu + '''"""


def _tma_sweep():
    m = load_inline("tma_stream_bench", cpp_sources="double run(torch::Tensor,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);",
                    cuda_sources=_TMA_SRC, functions=["run"], extra_cuda_cflags=["-O3"], verbose=False)
    buf = torch.randint(0, 255, (1 << 29,), dtype=torch.uint8, device="cuda")
    for spin in (0, 600):
        for wpc, S, stg in ((16, 2, 4608), (16, 3, 4608), (8, 4, 4608), (8, 6, 4608), (8, 2, 9216), (8, 3, 9216), (4, 4, 9216),
                            (4, 2, 18432), (4, 3, 18432), (2, 4, 18432), (2, 2, 36864), (1, 4, 36864), (32, 1, 4608)):
            if wpc * S * stg > 225000: continue
            r = [m.run(buf, wpc, S, stg, mode, spin, 5) for mode in (0, 1)]
            print(f"TMA spin {spin} wpc {wpc} S {S} stg {stg}: contig {r[0]:.0f} GB/s  interleaved {r[1]:.0f} GB/s", flush=True)
    del buf
'''
sub = sub.replace("    if DIAG:\n        _diag(st, weights)", "    _tma_sweep()\n    if DIAG:\n        _diag(st, weights)")
open("muse_moe_w4a16/diag_check.py", "w").write(sub + extra)
