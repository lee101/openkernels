import sys, os, importlib.util, torch
sys.path.insert(0, "muse_attention_decode")
import reference as R
spec = importlib.util.spec_from_file_location("sub", "muse_attention_decode/submission_v35p.py")
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S); S._DBG += ["-DAWT"] + sys.argv[1:]
for layer, B in (("sliding", 1), ("sliding", 32), ("full", 1), ("full", 16)):
    d = R.generate_input(batch=B, ctx=80000, heads="tp8", layer=layer, seed=7, q_len=8, spread="agentic" if B > 1 else "fixed", dist="soft" if layer == "sliding" else "coherent", page=1)
    for _ in range(3): S.kernel_fn(d)
    torch.cuda.synchronize()
    S._ext().awt_read().zero_() if False else None
    S.kernel_fn(d); torch.cuda.synchronize()
    T = S._ext().awt_read().view(-1, 8).double()
    T = T[T[:, 0] > 0]
    t0 = T[:, 0].min(); T = (T - t0) / 1e3; T[T < 0] = float("nan")
    q = lambda c: f"{torch.nanmedian(T[:, c]).item():5.1f}/{T[:, c][~T[:, c].isnan()].max().item() if (~T[:, c].isnan()).any() else float('nan'):5.1f}"
    print(f"{layer} B={B} ctas={len(T)} med/max: start {q(0)} prolog {q(1)} loop {q(2)} wo {q(3)} mwait {q(4)} end {q(5)}", flush=True)
