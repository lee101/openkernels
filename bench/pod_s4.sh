#!/bin/bash
# H200 session 4: MoE timeline, FP8 verify attention, decode-layer kernel timeline + prefetch diag
export PATH=/usr/local/cuda/bin:$PATH GHZ=1.8 TORCH_CUDA_ARCH_LIST=9.0a
cd /workspace/ok
echo "== attention v12"; python okb.py muse_attention_decode --no-check --reps 30 2>&1 | grep -E "^(decode|verify|load)"
echo "== moe prof"; (cd muse_moe_w4a16 && python prof.py 1 16 32 2>&1 | grep -v -i warn)
echo "== decode layer"; python okb.py muse_decode_layer --no-check --reps 20 --kprof 2>&1 | grep -v -i warn
echo "== pfdiag"; python okb.py muse_decode_layer -s submission_pfdiag.py --no-check --no-time 2>&1 | grep PFDIAG
