#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=9.0
cd /workspace/ok
pip install -q ninja 2>&1 | tail -1
timeout 600 python3 -c "import sys; sys.path.insert(0,'bench'); import gbench; gbench.run()" > gb.log 2>&1; echo "gb rc $?"
B=1073741824
SUB=muse_attention_decode/submission_v49.py CFG="{(16, False): [(-1, 256, 8, $B, x, 1) for x in (76, 588, 332, 584, 76, 588)], (32, False): [(-1, 256, 8, $B, x, 1) for x in (76, 588, 332, 584, 76, 588)]}" DEC="[]" timeout 900 python3 bench/v48_pod.py > v49t.log 2>&1; echo "t rc $?"
SUB=muse_attention_decode/submission_v49.py CFG="{(16, False): [(-1, 256, 8, $B, x, 1) for x in (76, 588)], (32, False): [(-1, 256, 8, $B, x, 1) for x in (76, 588)]}" timeout 900 python3 bench/v48_pod.py prof > v49p.log 2>&1; echo "p rc $?"
