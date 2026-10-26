#!/bin/bash
# bf16_proj v23 on an H100: correctness (prepare log = every candidate's time) then paired v23 vs v22 timing
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
cd /workspace/ok
pip install -q ninja 2>&1 | tail -1
python3 -c "import torch,sys;sys.exit(torch.__version__ < \"2.9\")" || pip install -q torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -1
nvidia-smi --query-gpu=name,clocks.max.sm --format=csv,noheader
timeout 1200 python3 okb.py muse_bf16_proj -s submission_v23.py --no-time --seeds 1 --stress 2 > v23chk.log 2>&1
echo "chk rc $?"
timeout 1200 python3 okb.py muse_bf16_proj -s submission_v23.py -b submission_v22.py --no-check --reps 30 > cmp.log 2>&1
echo "cmp rc $?"
