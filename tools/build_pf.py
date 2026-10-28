"""build a muse_prefill_layer submission: build_dl.py's composition + muse_prefill_layer/glue_pf2.py (prefill-only kernel_fn)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); import compose
R = Path(__file__).resolve().parent.parent
out = R / "muse_prefill_layer" / (sys.argv[1] if len(sys.argv) > 1 else "submission_pf.py")
plain = (R / "muse_decode_layer/submission_plain.py").read_text().splitlines(True)
cls = "".join(plain[15:75])
head = '''# okc: problem=muse_prefill_layer
"""Prefill layer: cuBLAS projections + FA3 with fused prologue/epilogue + fused residual norms + Marlin / dequant + grouped GEMM MoE with fused glue."""
from typing import TypedDict
import torch
from torch.utils.cpp_extension import load_inline

'''
# attention / MoE modules pinned to the versions prefill was tuned on (pf32: attention v69f, MoE v48); master moved on for decode
parts = [head, compose.rename(R / "muse_attention_decode/submission_v69f.py", "A_"), "\n",
         compose.rename(R / "muse_moe_w4a16/submission_v48.py", "M_"), "\n",
         compose.rename(R / "muse_bf16_proj/submission.py", "P_"), "\n", compose.rename(R / "muse_moe_w4a16/submission_v23.py", "Q_"), "\n", cls, "\n", (R / "muse_decode_layer/glue.py").read_text(), "\n",
         (R / "muse_prefill_layer/glue_pf.py").read_text(), "\n", (R / "muse_prefill_layer/glue_pf2.py").read_text()]
src = "".join(parts)
compile(src, out.name, "exec")
out.write_text(src)
print("wrote", out, len(src))
