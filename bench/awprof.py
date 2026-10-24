import sys, os, importlib.util, torch
sys.path.insert(0, "muse_attention_decode")
import reference as R
spec = importlib.util.spec_from_file_location("sub", "muse_attention_decode/submission_v33p.py")
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S); S._DBG += ["-DAWPROF"] + sys.argv[1:]
for B in (16, 32):
    d = R.generate_input(batch=B, ctx=80000, heads="tp8", layer="full", seed=7, q_len=8, spread="agentic", dist="coherent", page=1)
    for _ in range(3): S.kernel_fn(d)
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record(); S.kernel_fn(d); e1.record(); torch.cuda.synchronize()
    P = S._ext().awp_read().view(-1, 8).double()
    P = P[P[:, 7] > 0]
    tot = P[:, :7].sum(1)
    print(f"B={B} {e0.elapsed_time(e1)*1e3:.1f} us warps={len(P)} steps med {P[:,7].median().item():.0f} cyc/warp med {tot.median().item():.0f} max {tot.max().item():.0f} per-step {(tot/P[:,7]).median().item():.0f}")
    print("  share wait/iss/vconv/qk/softmax/pv/epi", [round(x, 3) for x in (P[:, :7].sum(0) / P[:, :7].sum()).tolist()])
