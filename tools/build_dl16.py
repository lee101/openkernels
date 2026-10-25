"""build muse_decode_layer/submission_v10.py: build_dl.py with bf16_proj v16 (c12 register-direct GEMV) as P_ + glue16.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); import compose
R = Path(__file__).resolve().parent.parent
out_dl = R / (sys.argv[1] if len(sys.argv) > 1 else "muse_decode_layer/submission_v10.py")
out_pf = R / sys.argv[2] if len(sys.argv) > 2 else None
plain = (R / "muse_decode_layer/submission_plain.py").read_text().splitlines(True)
cls = "".join(plain[15:75])
head = '''# okc: problem=muse_decode_layer
"""Decoder layer: c12 register-direct GEMV / cuBLAS projections + fused split-KV FP8 attention + fused residual norms + NVFP4 W4A16 MoE."""
from typing import TypedDict
import torch
from torch.utils.cpp_extension import load_inline

'''
glue = (R / "muse_decode_layer/glue.py").read_text()
rc = 'P__ext().run(st["rA"], st["rA2"], h, lg, st["pws"], st["pcnt"], 256)'
assert glue.count(rc) == 1
glue = glue.replace(rc, "_router(h, lg, st)")
parts = [head, compose.rename(R / "muse_attention_decode/submission.py", "A_"), "\n",
         compose.rename(R / "muse_moe_w4a16/submission.py", "M_"), "\n",
         compose.rename(R / "muse_bf16_proj/submission_v16.py", "P_"), "\n", cls, "\n", glue, "\n",
         (R / "muse_prefill_layer/glue_pf.py").read_text(), "\n", (R / "muse_decode_layer/glue16.py").read_text()]
out = "".join(parts)
compile(out, out_dl.name, "exec")
out_dl.write_text(out)
if out_pf:
    out_pf.write_text(out.replace("# okc: problem=muse_decode_layer", "# okc: problem=muse_prefill_layer", 1))
print("wrote", out_dl, len(out))
