"""build muse_decode_layer/submission.py from the attention + moe submissions and glue.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); import compose
R = Path(__file__).resolve().parent.parent
plain = (R / "muse_decode_layer/submission_plain.py").read_text().splitlines(True)
cls = "".join(plain[15:75])  # Weights, Config, Inputs
head = '''# okc: problem=muse_decode_layer
"""Decoder layer: cuBLAS projections + fused split-KV FP8 attention + fused residual norms + NVFP4 W4A16 MoE."""
from typing import TypedDict
import torch
from torch.utils.cpp_extension import load_inline

'''
parts = [head, compose.rename(R / "muse_attention_decode/submission.py", "A_"), "\n",
         compose.rename(R / "muse_moe_w4a16/submission.py", "M_"), "\n",
         compose.rename(R / "muse_bf16_proj/submission.py", "P_"), "\n", cls, "\n", (R / "muse_decode_layer/glue.py").read_text(), "\n", (R / "muse_prefill_layer/glue_pf.py").read_text()]
out = "".join(parts)
compile(out, "submission.py", "exec")
(R / "muse_decode_layer/submission.py").write_text(out)
pl = out.replace("# okc: problem=muse_decode_layer", "# okc: problem=muse_prefill_layer", 1)
(R / "muse_prefill_layer/submission.py").write_text(pl)
print("wrote", len(out))
