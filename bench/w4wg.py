import os, sys, time, torch
from torch.utils.cpp_extension import load_inline
m = load_inline("w4wg", cpp_sources="void run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor); void run2(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t);",
                cuda_sources=open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "w4wg.cu")).read(), functions=["run", "run2"],
                extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"] + [f for f in os.environ.get("W4FLAGS", "").split() if f], verbose=False)
torch.manual_seed(0)
E, N, K = 32, int(sys.argv[1]) if len(sys.argv) > 1 else 1024, 4096
mc = int(sys.argv[2]) if len(sys.argv) > 2 else 100
cnt = torch.randint(0, 2 * mc, (E,)); cnt[5] = 0
offs = torch.cumsum(cnt, 0).int().cuda(); R = int(cnt.sum())
A = torch.randn(R, K, device="cuda").bfloat16()
W = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device="cuda")
SF = (torch.rand(E, N, K // 16, device="cuda") * 2 + 0.25).to(torch.float8_e4m3fn)
gs = (torch.rand(E, device="cuda") * 0.01 + 0.001).float()
grid = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device="cuda")
def deq(e):
    p = W[e]; c = torch.stack((p & 15, p >> 4), -1).view(N, K).long()
    v = torch.where(c >= 8, -grid[c & 7], grid[c & 7])
    return (v * SF[e].float().repeat_interleave(16, 1)).bfloat16().float() * gs[e]
C = torch.empty(R, N, device="cuda", dtype=torch.bfloat16)
o = [0] + offs.tolist()
for name, fn in (("v1", lambda: m.run(A, W, SF, gs, offs, C)), ("v2x1", lambda: m.run2(A, W, SF, gs, offs, C, 1)), ("v2x2", lambda: m.run2(A, W, SF, gs, offs, C, 2))):
    C.zero_(); fn(); torch.cuda.synchronize(); err = 0
    for e in range(E):
        if o[e + 1] > o[e]:
            ref = A[o[e]:o[e + 1]].float() @ deq(e).T
            err = max(err, ((C[o[e]:o[e + 1]].float() - ref).norm() / ref.norm()).item())
    for _ in range(3): fn()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(20): fn()
    torch.cuda.synchronize(); dt = (time.time() - t) / 20
    print(f"{name}: err {err:.2e} {dt*1e6:.1f} us  {2*R*N*K/dt/1e12:.1f} TFLOPs  weights {E*N*K/2/dt/1e9:.0f} GB/s", flush=True)
