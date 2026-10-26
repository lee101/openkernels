#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=9.0
cd /workspace/ok
pip install -q ninja 2>&1 | tail -1
B=1073741824
SUB=muse_attention_decode/submission_v51.py CFG="{(16, False): [(-2, 256, 8, $B, 68, 1), (-2, 256, 8, $B, 324, 1)], (32, False): [(-2, 256, 8, $B, 68, 1), (-2, 256, 8, $B, 324, 1)], (16, True): [(-2, 160, 8, $B, 68, 1), (-2, 160, 8, $B, 324, 1)], (32, True): [(-2, 256, 8, $B, 68, 1), (-2, 256, 8, $B, 324, 1)]}" DEC="[]" timeout 400 python3 bench/v48_pod.py > v51t.log 2>&1; echo "t rc $?"
SUB=muse_attention_decode/submission_v51.py CFG="{(32, False): [(-2, 256, 8, $B, 68, 1), (-2, 256, 8, $B, 324, 1)], (16, True): [(-2, 160, 8, $B, 324, 1)]}" timeout 300 python3 bench/v48_pod.py prof > v51p.log 2>&1; echo "p rc $?"
