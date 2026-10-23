sub = open("muse_attention_decode/submission.py").read()
cu = open("bench/gather.cu").read()
extra = '''

_GS = r"""''' + cu + '''"""


def _gather_bench():
    m = load_inline("gather_bench", cpp_sources="double run(torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t);", cuda_sources=_GS, functions=["run"], extra_cuda_cflags=["-O3"], verbose=False)
    for mb in (176, 352, 704):
        base = torch.empty(mb * 2**20, dtype=torch.uint8, device="cuda")
        for R in (64, 128, 256):
            rows = base.numel() // R
            warps, per = 1056, 4096 * 64 // R
            idx = torch.randint(0, rows, (warps * per,), dtype=torch.int32, device="cuda")
            seq = (torch.arange(warps * per, device="cuda", dtype=torch.int64) % rows).int()
            print("GATHER %dMB R=%d random %.0f GB/s  sequential %.0f GB/s" % (mb, R, m.run(base, idx, R, per, 5), m.run(base, seq, R, per, 5)), flush=True)
        del base


_gb_done = False
'''
sub = sub.replace("def kernel_fn(data):\n", "def kernel_fn(data):\n    global _gb_done\n    if not _gb_done:\n        _gb_done = True\n        _gather_bench()\n", 1)
open("muse_attention_decode/gather_check.py", "w").write(sub + extra)
