"""sanitizer-sized run: 8 experts, ids folded mod 8"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reference as R, submission as S
S._DBG = os.environ.get("OKB_DBG", "").split()
R.N_EXPERTS = 8; ws = R.generate_weights(R.WEIGHTS_SEED); R.N_EXPERTS = 256
st = S.prepare(ws)
bs = int(sys.argv[1])
d = R.generate_input(bs=bs, sharding="ep8", seed=1)
ids = d["topk_ids"]
d["topk_ids"] = torch.where(ids < 32, ids % 8, ids).int()
S.kernel_fn(d, st); torch.cuda.synchronize(); print("ok")
