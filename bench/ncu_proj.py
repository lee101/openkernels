import sys, torch
sys.path.insert(0, "muse_bf16_proj")
import reference as R, submission as S
W = R.generate_weights(1); st = S.prepare(W)
for layer, m in (("router", 1), ("router", 32), ("qkv_gate", 1), ("unpooler", 1)):
    d = R.generate_input(m, layer, 3)
    for _ in range(2): S.kernel_fn(d, st)
    torch.cuda.synchronize()
