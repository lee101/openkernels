import os, torch
from torch.utils.cpp_extension import load_inline
m = load_inline("wgt", cpp_sources="void run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t);",
                cuda_sources=open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "wgt.cu")).read(), functions=["run"],
                extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
torch.manual_seed(0)
q, k, v = (torch.randn(64, 64, device="cuda") * 0.3).half(), (torch.randn(64, 64, device="cuda") * 0.3).half(), torch.randn(64, 64, device="cuda").half()
Sr = q.float() @ k.float().T
Or = Sr.half().float() @ v.float()
for vtr in (1, 0):
    for vl, vs in ((16, 1024), (1024, 16), (2048, 1024), (1024, 2048), (8192, 1024), (1024, 8192)):
        S = torch.zeros(64, 64, device="cuda"); O = torch.zeros(64, 64, device="cuda")
        m.run(q, k, v, S, O, vl, vs, vtr); torch.cuda.synchronize()
        es = ((S - Sr).norm() / Sr.norm()).item(); eo = ((O - Or).norm() / Or.norm()).item()
        print(f"vtr {vtr} lbo {vl} sbo {vs}: S err {es:.2e}  O err {eo:.2e}", flush=True)
