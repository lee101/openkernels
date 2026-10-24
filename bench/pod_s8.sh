#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0a GHZ=1.8
cd /workspace/ok
for v in "" "-DKSF=2" "-DKSF=1" "-DCFG=8x4" "-DCFG=8x4 -DKSF=2"; do
  echo "== moe [$v]"; OKB_DBG="$v" python okb.py muse_moe_w4a16 --experts 32 --no-check --reps 30 2>&1 | grep -E "^(decode|verify)" | awk '{print $1,$2,$NF}' | tr '\n' ' '; echo
done
cd muse_moe_w4a16; for v in "" "-DKSF=2"; do echo "== prof [$v]"; OKB_DBG="$v" python prof.py 1 16 32 2>&1 | grep -E "^bs|busy|epilogue|tma|dep"; done
