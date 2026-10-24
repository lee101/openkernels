import sys, os, importlib.util, torch
sys.path.insert(0, "muse_attention_decode")
import reference as R
from sgl_kernel.flash_attn import flash_attn_with_kvcache
spec = importlib.util.spec_from_file_location("sub", "muse_attention_decode/submission_v35.py")
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
def tm(fn, it=10):
    g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        fn(); torch.cuda.synchronize()
        with torch.cuda.graph(g): fn()
    ts = []
    for _ in range(it):
        flush.zero_(); a = torch.cuda.Event(True); b = torch.cuda.Event(True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1e3)
    ts.sort(); return ts[len(ts) // 2]
for layer, L, B in (("full", 753, 1), ("sliding", 753, 1), ("full", 753, 4), ("sliding", 753, 4), ("full", 8, 16), ("full", 8, 1), ("sliding", 8, 1), ("full", 1, 16)):
    d = R.generate_input(batch=B, ctx=80000, heads="tp8", layer=layer, seed=1, q_len=L, page=1, spread="agentic" if B > 1 else "fixed", dist="coherent" if layer == "full" else "soft")
    cfg = d["config"]; hq = cfg["num_q_heads"]
    q8 = (torch.randn(B * L, hq, 64, device="cuda") * 2).to(torch.float8_e4m3fn)
    win = cfg["window"]
    fa = lambda: flash_attn_with_kvcache(q=q8, k_cache=d["k_cache"], v_cache=d["v_cache"], page_table=d["page_table"], cache_seqlens=d["cache_seqlens"],
                                         cu_seqlens_q=d["cu_seqlens_q"], max_seqlen_q=L, softmax_scale=cfg["sm_scale"], causal=True,
                                         window_size=(win - 1, 0) if win else (-1, -1), scheduler_metadata=d["scheduler_metadata"])
    t_fa = tm(fa); t_us = tm(lambda: S.kernel_fn(d))
    print(f"{layer:7s} L={L:4d} B={B:2d}: ours {t_us:8.1f}  fa3 {t_fa:8.1f}", flush=True)
