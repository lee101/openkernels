import sys, os, torch
from torch.utils.cpp_extension import load_inline
m = load_inline("inner_bench", cpp_sources="double run(torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t);",
                cuda_sources=open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "inner.cu")).read(), functions=["run"],
                extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)
buf = torch.randint(0, 255, (1 << 29,), dtype=torch.uint8, device="cuda")
xb = torch.randn((8, 4096), device="cuda").bfloat16()
cfgs = [int(c) for c in (sys.argv[1] if len(sys.argv) > 1 else "0,1,2,3").split(",")]
names = {0: "16w x2", 1: "8w x4", 2: "12w x3", 3: "8w x2"}
for cfg in cfgs:
    print(names[cfg], " ".join(f"V{v}:{m.run(buf, xb, v, cfg, 5):.0f}" for v in range(7)), flush=True)
