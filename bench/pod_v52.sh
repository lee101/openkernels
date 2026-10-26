#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=9.0
cd /workspace/ok
pip install -q ninja 2>&1 | tail -1
B=1073741824
export SUB=muse_attention_decode/submission_v52.py
CFG="{(16, False): [(-2, 256, 8, $B, x, 1) for x in (324, 1348, 324, 1348)], (32, False): [(-2, 256, 8, $B, x, 1) for x in (324, 1348, 324, 1348)], (16, True): [(-2, 160, 8, $B, x, 1) for x in (68, 1092)], (32, True): [(-2, 256, 8, $B, x, 1) for x in (68, 1092)]}" DEC="[(16, False, 5, 5), (32, False, 5, 5)]" timeout 400 python3 bench/v48_pod.py > t1.log 2>&1; echo "t1 $?"
CFG="{(32, False): [(-2, 256, 8, $B, 324, 1)]}" DEC="[]" DBG="-DAP_NOCVT=1" timeout 300 python3 bench/v48_pod.py > t2.log 2>&1; echo "t2 $?"
CFG="{(32, False): [(-2, 256, 8, $B, 324, 1)]}" DEC="[]" DBG="-DAP_NOEXP=1" timeout 300 python3 bench/v48_pod.py > t3.log 2>&1; echo "t3 $?"
CFG="{}" DEC="[(16, False, 5), (32, False, 5)]" DBG="-DRBF=6" timeout 300 python3 bench/v48_pod.py > t4.log 2>&1; echo "t4 $?"
CFG="{}" DEC="[(16, False, 5), (32, False, 5)]" DBG="-DRBF=4" timeout 300 python3 bench/v48_pod.py > t5.log 2>&1; echo "t5 $?"
CFG="{(32, False): [(-2, 256, 8, $B, x, 1) for x in (324, 1348)]}" DEC="[]" timeout 300 python3 bench/v48_pod.py prof > p1.log 2>&1; echo "p1 $?"
