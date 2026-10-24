#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0a GHZ=1.8
cd /workspace/ok
for v in "" "-DFD_EMPTY" "-DFD_NOLOAD" "-DFD_NOCOMP" "-DFD_NOMERGE" "-DFD_NOLOAD -DFD_NOCOMP" "-DFD_NOLOAD -DFD_NOCOMP -DFD_NOMERGE"; do
  echo "== attn [$v]"; OKB_DBG="$v" python okb.py muse_attention_decode --no-check --reps 30 --only decode 2>&1 | grep -E '^decode' | awk '{print $1,$2,$NF}' | tr '\n' ' '; echo
done
echo "== moe"; python okb.py muse_moe_w4a16 --experts 32 --no-check --reps 30 2>&1 | grep -E '^(decode|verify)' | awk '{print $1,$2,$(NF-1)}' | tr '\n' ' '; echo
(cd muse_moe_w4a16 && OKB_DBG=-DEVONLY python prof.py 1 16 2>&1 | grep -E '^bs|w0:|w1:')
