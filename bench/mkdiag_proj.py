"""muse_bf16_proj/diag_check.py = submission + fixed-cost microbench printed from prepare()"""
sub = open("muse_bf16_proj/submission.py").read()
extra = r'''

_FC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
__global__ void k_empty() {}
__global__ void k_one(float* p) { if (threadIdx.x == 0) p[blockIdx.x] = 1.f; }
__global__ void k_atom(float* p, int* c) { if (threadIdx.x == 0) { atomicAdd(p + blockIdx.x, 1.f); __threadfence(); atomicAdd(c, 1); } }
void empty(int64_t g) { k_empty<<<g, 32, 0, at::cuda::getCurrentCUDAStream()>>>(); }
void one(torch::Tensor p, int64_t g) { k_one<<<g, 32, 0, at::cuda::getCurrentCUDAStream()>>>(p.data_ptr<float>()); }
void atom(torch::Tensor p, torch::Tensor c, int64_t g) { k_atom<<<g, 32, 0, at::cuda::getCurrentCUDAStream()>>>(p.data_ptr<float>(), c.data_ptr<int>()); }
"""


def _gt(fn, reps=50):
    import statistics
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn(); fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    fl = torch.ones(2 ** 27, device="cuda"); sink = torch.empty((), device="cuda")
    ts = []
    for _ in range(reps):
        sink.copy_(fl.sum()); a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1e3)
    return statistics.median(ts)


def _fixed(st, weights):
    m = load_inline("fixedcost", cpp_sources="void empty(int64_t); void one(torch::Tensor,int64_t); void atom(torch::Tensor,torch::Tensor,int64_t);",
                    cuda_sources=_FC, functions=["empty", "one", "atom"], verbose=False)
    p = torch.zeros(4096, device="cuda"); c = torch.zeros(1, dtype=torch.int32, device="cuda")
    print("FIX empty-graph", round(_gt(lambda: None), 2), flush=True)
    for g in (1, 132, 1056):
        print("FIX empty-kernel g", g, round(_gt(lambda: m.empty(g)), 2), " one", round(_gt(lambda: m.one(p, g)), 2), " atom", round(_gt(lambda: m.atom(p, c, g)), 2), flush=True)
    for name, M in (("o", 1), ("unpooler", 1), ("router", 8), ("qkv_gate", 1), ("o", 32)):
        W = weights[name]; n = 1024 if (name == "unpooler" and M <= 8) else W.shape[0]
        x = torch.randn((M, W.shape[1]), device="cuda").bfloat16()
        ref = (lambda: x.float() @ W.T) if name == "router" else (lambda: x @ W[:n].T)
        d = {"x": x, "config": {"m": M, "n": n, "k": W.shape[1], "layer": name}}
        print("FIX", name, M, "torch", round(_gt(ref), 2), "ours", round(_gt(lambda: kernel_fn(d, st)), 2), flush=True)
'''
sub = sub.replace("    _ext()\n    return st", "    _ext()\n    _fixed(st, weights)\n    return st")
open("muse_bf16_proj/diag_check.py", "w").write(sub + extra)
