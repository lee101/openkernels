import sys, torch
sys.path.insert(0, "muse_moe_w4a16")
import reference as R, submission as S
from nvfp4 import dequantize
W = R.generate_weights(1); st = S.prepare(W)
bs = int(sys.argv[1]) if len(sys.argv) > 1 else 128
d = R.generate_input(bs=bs, sharding="ep8", seed=405)
x = d["hidden_states"].contiguous(); M = x.shape[0]
ids = d["topk_ids"].to(torch.int32).contiguous(); tw = d["topk_weights"].float().contiguous()
inter = torch.zeros((M * 8, 4096), dtype=torch.bfloat16, device="cuda"); y = torch.zeros_like(inter)
ss = torch.zeros((M * 8,), device="cuda"); out = torch.empty_like(x); xp = torch.empty_like(x)
lo = d["config"]["local_start"]
S._ext(True).moel(x, ids, tw, st["w1"], st["w2"], st["g1"], st["g2"], st["nw"], inter, y, ss, st["ctr"], xp, out, lo)
torch.cuda.synchronize()
# inverse permutation within 64-chunks: logical L <- true q
q = torch.arange(64); t, jh, jl, hi, lo_ = q // 16, (q // 8) % 2, (q // 4) % 2, (q // 2) % 2, q % 2
L = (2 * jh + jl) * 16 + hi * 8 + 2 * t + lo_
Lfull = (torch.arange(64).repeat(64) * 0 + L.repeat(64) + torch.arange(64).repeat_interleave(64) * 64).cuda()
xt = xp[:, Lfull]
print("xp ok", (xt == x).all().item())
it = inter[:, Lfull]
rows = (ids.long() - lo)
for e in sorted(set(rows[(rows >= 0) & (rows < 32)].flatten().tolist()))[:4]:
    tok, slot = ((ids.long() - lo) == e).nonzero(as_tuple=True)
    r = tok * 8 + slot
    h = x[tok].float() @ dequantize(W["w13"][lo + e], W["w13_sf"][lo + e], W["w13_gs"][lo + e]).T
    ri = torch.nn.functional.silu(h[:, :4096]) * h[:, 4096:]
    gi = it[r].float()
    err = ((gi - ri).norm() / ri.norm()).item()
    yy = ri.bfloat16().float() @ dequantize(W["w2"][lo + e], W["w2_sf"][lo + e], W["w2_gs"][lo + e]).T
    ry = yy * (1 + W["post_norm_w"].float())
    gy = y[r].float()
    ey = ((gy - ry).norm() / ry.norm()).item()
    print(f"e {e} cnt {len(r)} inter err {err:.3e} y err {ey:.3e}  col-err first tile {((gi[:, :8]-ri[:, :8]).norm()/ri[:, :8].norm()).item():.2e} ss {ss[r][:3].tolist()} ref {ry.pow(2).sum(-1)[:3].tolist()}")
