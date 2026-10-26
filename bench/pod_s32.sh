#!/bin/bash
# bf16_proj v22 on an H100: correctness (prepare log = every candidate's time) then paired v22 vs v21 timing
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
cd /workspace/ok
pip install -q ninja 2>&1 | tail -1
nvidia-smi --query-gpu=name,clocks.max.sm --format=csv,noheader
timeout 1200 python3 okb.py muse_bf16_proj -s submission_v22.py --no-time --seeds 1 --stress 2 > v22chk.log 2>&1
echo "chk rc $?"
timeout 1200 python3 okb.py muse_bf16_proj -s submission_v22.py -b submission_v21.py --no-check --reps 30 > cmp.log 2>&1
echo "cmp rc $?"
