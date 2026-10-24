"""okc_case.py <problem> <submission.py> <bench_index|json-kwargs> [--calls N]: run one scored shape of a problem once
(after a warmup call that builds the extension), for NVBit tracing / ncu. Prints the kernel_fn wall time (not meaningful)."""
import importlib.util, inspect, json, sys, time
from pathlib import Path
import torch

R = Path(__file__).resolve().parent.parent.parent
prob, sub = sys.argv[1], sys.argv[2]
D = R / prob; sys.path.insert(0, str(D))
def load(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
ref = load(D / "reference.py", "reference"); m = load(Path(sub).resolve(), "sub")
meta = json.load(open(R / "notes" / f"{prob}.json"))
arg = sys.argv[3]
c = meta["benchmarks"][int(arg)] if arg.isdigit() else json.loads(arg)
kw = {k: v for k, v in c.items() if k not in ("load", "weight", "repeats")}
ws = ref.generate_weights(ref.WEIGHTS_SEED) if hasattr(ref, "generate_weights") else None
st = m.prepare(ws) if (ws is not None and hasattr(m, "prepare")) else None
call = lambda d: m.kernel_fn(d, st) if len(inspect.signature(m.kernel_fn).parameters) > 1 else m.kernel_fn(d)
data = ref.generate_input(**kw)
import os
if not os.environ.get("OKC_NOWARM"): call(data); torch.cuda.synchronize()  # warmup / build (OKC_NOWARM=1 when tracing a cached build)
torch.cuda.synchronize(); t = time.time(); call(data); torch.cuda.synchronize()
print(json.dumps(kw), f"{(time.time() - t) * 1e6:.0f} us wall (local)")
