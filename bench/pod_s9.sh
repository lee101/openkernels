#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0a
cd /workspace/ok
echo "== verify fused correctness"; python okb.py muse_attention_decode --seeds 2 --stress 2 --no-time --only verify 2>&1 | grep -E "^BAD|correctness|stress|rror" | cut -c1-200
for v in "" "-DFUSEV=0" "-DVDB=0 -DRSW=5"; do
  echo "== [$v]"; OKB_DBG="$v" python okb.py muse_attention_decode --no-check --reps 20 --only verify 2>&1 | grep -E '^verify' | awk '{print $1,$2,$NF}' | tr '\n' ' '; echo
done
