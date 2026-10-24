import sys, os, importlib.util, torch
sys.path.insert(0, "muse_prefill_layer")
import reference as R
spec = importlib.util.spec_from_file_location("sub", "muse_prefill_layer/submission_v2.py")
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
W = R.generate_weights(1); st = S.prepare(W)
cap = {}
_orig = S.M__ext
class _Wrap:
    def __init__(self, m): self.m = m
    def moe(self, *a):
        cap["loc"] = a[1].detach().clone(); return self.m.moe(*a)
S.M__ext = lambda large=False: _Wrap(_orig(large))
for layer, B, L in (("full", 1, 753), ("sliding", 4, 753)):
    d = R.generate_input(batch=B, ctx=80000, layer=layer, seed=5, q_len=L, page=1)
    for _ in range(2): S.kernel_fn(d, st)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as pr:
        S.kernel_fn(d, st); torch.cuda.synchronize()
    print(f"== {layer} B={B} L={L}")
    print(pr.key_averages().table(sort_by="cuda_time_total", row_limit=8, max_name_column_width=50))
    if "loc" in cap:
        loc = cap["loc"].flatten().long()
        h = torch.bincount(loc[(loc >= 0) & (loc < 32)], minlength=32)
        print("local expert counts", sorted(h.tolist(), reverse=True)[:12], "sum", int(h.sum()), "dtype", cap["loc"].dtype, cap["loc"].shape)
