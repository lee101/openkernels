#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0a GHZ=1.8
cd /workspace/ok
echo "== moe"; python okb.py muse_moe_w4a16 --experts 32 --no-check --reps 30 2>&1 | grep -E '^(decode|verify)' | awk '{print $1,$2,$(NF-1)}' | tr '\n' ' '; echo
(cd muse_moe_w4a16 && OKB_DBG=-DEVONLY python prof.py 1 16 2>&1 | grep -E '^bs|w0:|w1:')
echo "== fdiag"; (cd muse_attention_decode && python -c "
import torch, submission as S
S._diag(torch.device('cuda'))
" 2>&1 | grep FDIAG)
echo "== attn"; python okb.py muse_attention_decode --no-check --reps 30 2>&1 | grep -E '^(decode|verify)' | awk '{print $1,$2,$NF}' | tr '\n' ' '; echo
echo "== prefill"; python okb.py muse_prefill_layer --seeds 1 --reps 3 --kprof 2>&1 | grep -v -i warn | grep -E "^(OK|BAD)|correctness|^turn|^first|load times|  +[0-9.]+ \+" | cut -c1-140 | head -150
