"""run kernel_fn a few times at one bs (for ncu/nsys): python one.py BS [reps]"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reference as R, submission as S
S._DBG = os.environ.get("OKB_DBG", "").split()
R.N_EXPERTS = 32; ws = R.generate_weights(R.WEIGHTS_SEED); R.N_EXPERTS = 256
st = S.prepare(ws)
bs = int(sys.argv[1]); n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
d = R.generate_input(bs=bs, sharding="ep8", seed={1: 401, 16: 402, 32: 403, 8: 404, 128: 405, 256: 406}.get(bs, 1))
for _ in range(n): S.kernel_fn(d, st)
torch.cuda.synchronize(); print("ok")
