import sys, importlib.util, torch
sys.path.insert(0, "muse_moe_w4a16")
import reference as R
spec = importlib.util.spec_from_file_location("b", "muse_moe_w4a16/submission_v18.py"); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
S._DBG += ["-DSKT"]
W = R.generate_weights(1); st = S.prepare(W)
prof = torch.zeros(132 * 16 * 8, dtype=torch.int64, device="cuda"); S._ext().skprof(prof)
for bs in (1, 8):
    d = R.generate_input(bs=bs, sharding="ep8", seed=406)
    for _ in range(3): S.kernel_fn(d, st)
    torch.cuda.synchronize(); prof.zero_(); S.kernel_fn(d, st); torch.cuda.synchronize()
    T = prof.view(-1, 8).double(); T = T[T[:, 0] > 0]; T = (T - T[:, 0].min()) / 1e3
    q = lambda c: f"{T[:, c].median().item():5.1f}/{T[:, c].max().item():5.1f}"
    ids = d["topk_ids"]; na = int(((ids - d["config"]["local_start"]) % 256 < 32).any(0).sum()) if False else -1
    print(f"bs {bs}: routed {q(1)} p1start {q(2)} p1end {q(4)} barrier {q(3)} p2end {q(5)} end {q(6)}", flush=True)
