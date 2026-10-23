#!/bin/bash
# one-shot H200 session: inner-loop ladder + real-kernel config variants (timeline spans)
export PATH=/usr/local/cuda/bin:$PATH GHZ=1.8 TORCH_CUDA_ARCH_LIST=9.0a
cd /workspace/ok
echo "== inner ladder (GB/s)"; python bench/inner.py 0,1,2,3 2>&1 | grep -v -i warn
cd muse_moe_w4a16
for v in "" "-DBD1=3" "-DCFG=12x3" "-DCFG=8x4" "-DCFG=8x2"; do
  echo "== kernel variant [$v]"
  OKB_DBG="$v" python prof.py 1 8 32 256 2>&1 | grep -E "^bs|g1 busy|g2 busy|dep|tma"
done
