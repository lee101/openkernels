#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0a
cd /workspace/ok
echo "== wgt"; python bench/wgt.py 2>&1 | grep -v -i warn
echo "== verify attnw"; python okb.py muse_attention_decode --seeds 1 --only verify --reps 20 2>&1 | grep -E "^(OK|BAD)|correctness|^verify|rror" | cut -c1-160
