#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0a
cd /workspace/ok
echo "== moe M sweep"; timeout 900 python bench/moe_m.py 2>&1 | grep -E "^M=|rror" 
