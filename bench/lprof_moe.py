import sys, torch
sys.path.insert(0, "muse_moe_w4a16")
import reference as R, submission as S
S._DBG = ["-DLPROF"] + sys.argv[2:]; S.WGL = True
W = R.generate_weights(1); st = S.prepare(W)
bs = int(sys.argv[1]) if len(sys.argv) > 1 else 256
d = R.generate_input(bs=bs, sharding="ep8", seed=406)
for _ in range(3): S.kernel_fn(d, st)
torch.cuda.synchronize(); st["prof"].zero_()
e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
e0.record(); S.kernel_fn(d, st); e1.record(); torch.cuda.synchronize()
nsm = torch.cuda.get_device_properties(0).multi_processor_count
P = st["prof"][: nsm * 8 * 8].view(nsm * 8, 8)[:, :4].double()
ms = e0.elapsed_time(e1)
print(f"bs {bs} {ms*1e3:.1f} us; per-warp cycles median top/wait/stage/epi:", [round(x) for x in P.median(0).values.tolist()], "max", [round(x) for x in P.max(0).values.tolist()])
print("sum share", [round(x, 3) for x in (P.sum(0) / P.sum()).tolist()])
