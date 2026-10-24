#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0a
cd /workspace/ok
for v in "-DPFF=0 -DPFD=0" "-DPFF=2 -DPFD=2" "-DPFF=4 -DPFD=4" "-DPFF=8 -DPFD=8" "-DPFF=12 -DPFD=6"; do
  echo "== [$v]"; OKB_DBG="$v" python okb.py muse_attention_decode --no-check --reps 20 2>&1 | grep -E '^(decode|verify)' | awk '{print $1,$2,$NF}' | tr '\n' ' '; echo
done
echo "== inner ladder"; python bench/inner.py 0,1 2>&1 | grep -v -i warn
