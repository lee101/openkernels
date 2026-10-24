#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0a
cd /workspace/ok
for f in "" "-DRSW=3" "-DRSW=5" "-DAW_H2=0 -DAW_Z=0"; do echo "== awprof $f"; timeout 600 python bench/awprof.py $f 2>&1 | grep -E "^B=|share|rror" ; done
