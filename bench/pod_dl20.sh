#!/bin/bash
# H100: decode_layer v20 (bf16_proj v22 composed) vs v17: correctness, graph stress, graph-replay timing
export PATH=/usr/local/cuda/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0a
cd /workspace/ok
pip install -q ninja 2>&1 | tail -1
python okb.py muse_decode_layer -s submission_v20.py -b submission_v17.py --seeds 1 --stress 3 --reps 30 2>&1 | grep -vE "^\s*$"
