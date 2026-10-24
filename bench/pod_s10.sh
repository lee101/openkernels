#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0a GHZ=1.8
cd /workspace/ok
echo "== fdiag"; (cd muse_attention_decode && python -c "
import torch, submission as S
S._diag(torch.device('cuda'))
" 2>&1 | grep FDIAG)
echo "== proj v6"; python okb.py muse_bf16_proj --no-check --reps 30 2>&1 | grep -E "^(decode|verify)" | awk '{print $1,$2,$3,$4,$NF}'
sed 's/^V6 = True/V6 = False/' muse_bf16_proj/submission.py > muse_bf16_proj/sub_v5.py
echo "== proj v5"; python okb.py muse_bf16_proj -s sub_v5.py --no-check --reps 30 2>&1 | grep -E "^(decode|verify)" | awk '{print $1,$2,$3,$4,$NF}'
echo "== moe prof"; (cd muse_moe_w4a16 && python prof.py 1 16 2>&1 | grep -E "^bs|busy|epilogue|tma|dep|w0:|w1:|w17:")
echo "== verify VDB0"; OKB_DBG="-DVDB=0 -DRSW=5" python okb.py muse_attention_decode --no-check --reps 20 --only verify 2>&1 | grep -E '^verify' | awk '{print $1,$2,$NF}' | tr '\n' ' '; echo
