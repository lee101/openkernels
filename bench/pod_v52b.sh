#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=9.0
cd /workspace/ok
pip install -q ninja 2>&1 | tail -1
B=1073741824
export SUB=muse_attention_decode/submission_v52.py
CFG="{(32, False): [(-2, 256, 8, $B, x, 1) for x in (324, 2372)]}" DEC="[(16, False, None, ('a', 256, 64), ('a', 256, 0)), (32, False, None, ('a', 256, 64)), (1, False, None, ('a', 256, 64)), (16, True, None, ('a', 128, 64)), (32, True, None, ('a', 128, 64)), (1, True, None, ('a', 128, 64), ('a', 64, 64))]" timeout 420 python3 bench/v48_pod.py > t6.log 2>&1; echo "t6 $?"
