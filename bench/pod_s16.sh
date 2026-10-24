#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0a
cd /workspace/ok
echo "== w4wg"; for a in "8192 20" "8192 40" "8192 140" "4096 140" "1024 100"; do echo "[$a]"; timeout 300 python bench/w4wg.py $a 2>&1 | grep -v -i warn | tail -4; done
echo "== moe"; timeout 600 python okb.py muse_moe_w4a16 --reps 30 2>&1 | tail -14
