#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=9.0
cd /workspace/ok
pip install -q ninja 2>&1 | tail -1
B=1073741824
CFG="{(16, False): [(-1, 256, 8, $B, x, 1) for x in (76, 332, 320, 264, 76, 332)], (32, False): [(-1, 256, 8, $B, x, 1) for x in (76, 332, 320, 264, 76, 332)], (1, False): [(520, 607, 128, $B, 0, 1)], (16, True): [(-1, 160, 8, 32, 76, 1), (-1, 176, 8, 32, 76, 1)], (32, True): [(-1, 256, 8, 32, 76, 1)], (1, True): [(-1, 128, 8, $B, 8, 1)]}" DEC="[(16, False, 5, 13, 5, 13), (32, False, 5, 13, 5, 13), (1, False, 7)]" timeout 900 python3 bench/v48_pod.py > v48t6.log 2>&1; echo "t6 rc $?"
CFG="{(16, False): [(-1, 256, 8, $B, x, 1) for x in (76, 332)], (32, False): [(-1, 256, 8, $B, x, 1) for x in (76, 332)]}" timeout 900 python3 bench/v48_pod.py prof > v48p6.log 2>&1; echo "p6 rc $?"
