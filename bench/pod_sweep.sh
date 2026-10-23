#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0a
cd /workspace/ok
for v in "-DCHK=1" "-DCHK=2" "-DCHK=4" "-DCHK=2 -DNSLOT=40 -DNACC=12" "-DCHK=8 -DNSLOT=40 -DNACC=12"; do
  echo "== [$v]"; OKB_DBG="$v" python okb.py muse_moe_w4a16 --experts 32 --no-check --reps 30 2>&1 | grep -E "^(decode|verify|load)" | awk '{print $1,$2,$(NF-1)}' | tr '\n' ' '; echo
done
