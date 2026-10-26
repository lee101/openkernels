#!/bin/bash
# H100: decode_layer v21 (attention v50 + MoE v34 + bf16_proj v22) vs v17: correctness, graph stress, graph-replay timing,
# then the kernels each verify/decode bench actually runs (one profiled graph replay)
export PATH=/usr/local/cuda/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0a
cd /workspace/ok
pip install -q ninja 2>&1 | tail -1
python okb.py muse_decode_layer -s submission_v21.py -b submission_v17.py --seeds 1 --stress 5 --reps 30 --kprof 2>&1 | grep -vE "^\s*$"
