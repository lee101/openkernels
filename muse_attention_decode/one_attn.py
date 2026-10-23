import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reference as R, submission as S
S._DBG = os.environ.get("OKB_DBG", "").split()
b, L = int(sys.argv[1]), int(sys.argv[2]); layer = sys.argv[3] if len(sys.argv) > 3 else "full"
d = R.generate_input(batch=b, ctx=80000, heads="tp8", layer=layer, seed=611, q_len=L, spread="agentic", dist="soft", page=1)
for _ in range(3): S.kernel_fn(d)
torch.cuda.synchronize(); print("ok")
