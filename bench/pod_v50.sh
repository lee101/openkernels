#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=9.0
cd /workspace/ok
pip install -q ninja 2>&1 | tail -1
B=1073741824
SUB=muse_attention_decode/submission_v50.py CFG="{(16, False): [(-1, 256, 8, $B, 332, 1), (-2, 256, 8, $B, 68, 1), (-2, 256, 8, $B, 4, 1), (-1, 256, 8, $B, 332, 1), (-2, 256, 8, $B, 68, 1)], (32, False): [(-1, 256, 8, $B, 332, 1), (-2, 256, 8, $B, 68, 1), (-2, 256, 8, $B, 4, 1), (-1, 256, 8, $B, 332, 1), (-2, 256, 8, $B, 68, 1)], (16, True): [(-1, 160, 8, 32, 76, 1), (-2, 160, 8, $B, 68, 1), (-2, 128, 8, $B, 68, 1)], (32, True): [(-1, 256, 8, 32, 76, 1), (-2, 256, 8, $B, 68, 1)], (1, True): [(-1, 128, 8, $B, 8, 1), (-2, 128, 8, $B, 68, 1), (-2, 64, 8, $B, 68, 1)], (1, False): [(520, 607, 128, $B, 0, 1), (-2, 320, 8, $B, 68, 1)]}" DEC="[]" timeout 900 python3 bench/v48_pod.py > v50t.log 2>&1; echo "t rc $?"
SUB=muse_attention_decode/submission_v50.py CFG="{(16, False): [(-2, 256, 8, $B, 68, 1)], (32, False): [(-2, 256, 8, $B, 68, 1)], (16, True): [(-2, 160, 8, $B, 68, 1)]}" timeout 900 python3 bench/v48_pod.py prof > v50p.log 2>&1; echo "p rc $?"
