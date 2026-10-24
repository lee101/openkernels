#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0a
cd /workspace/ok
timeout 900 python bench/sk_t.py 2>&1 | grep -E "^bs|rror"
