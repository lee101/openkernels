#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0a GHZ=1.8
cd /workspace/ok
echo "== attention correctness"; python okb.py muse_attention_decode --seeds 2 --stress 2 --no-time --only decode 2>&1 | grep -E "^BAD|correctness|FAIL|rror" | cut -c1-200
echo "== attention timing"; python okb.py muse_attention_decode --no-check --reps 30 --only decode 2>&1 | grep -E '^decode' | awk '{print $1,$2,$NF}' | tr '\n' ' '; echo
OKB_DBG="-DCLUS=0" python okb.py muse_attention_decode --no-check --reps 30 --only decode 2>&1 | grep -E '^decode' | awk '{print $1,$2,$NF}' | tr '\n' ' '; echo
(cd muse_attention_decode && python -c "
import torch, submission as S
S._diag(torch.device('cuda'))
" 2>&1 | grep FDIAG)
