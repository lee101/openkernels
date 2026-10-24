import sys, time, importlib.util, os
import torch
D = os.path.dirname(os.path.abspath(__file__)) + "/../muse_bf16_proj"
sys.path.insert(0, D)
spec = importlib.util.spec_from_file_location("sub", os.environ.get("SUB", D + "/submission_v13.py"))
s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
import reference as ref
mode = sys.argv[1] if len(sys.argv) > 1 else "compile"
t = time.time()
s._gvext()
print(f"gv compile {time.time() - t:.1f}s, {len(s.GVI)} instances", flush=True)
if mode == "compile":
    sys.exit()
ws = ref.generate_weights(ref.WEIGHTS_SEED)
if mode == "pack":
    for k in ("qkv_gate", "o", "pooler", "unpooler"):
        t = time.time(); c = s._c12(s._pack(ws[k]))
        print(k, "c12 ok" if c else "FAIL", c and hex(c[1] & (2 ** 64 - 1)), c and len(c[2]), f"{time.time() - t:.1f}s")
    sys.exit()
t = time.time()
st = s.prepare(ws)
torch.cuda.synchronize()
print(f"prepare {time.time() - t:.1f}s")
for r in st["gvlog"]:
    print(*r)
print({k: s._name(v[0]) if hasattr(s, "_name") else s.GVI[v[0]] for k, v in st["gv"].items()})
