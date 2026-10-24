#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0a
cd /workspace/ok
echo "== dbg"; timeout 600 python bench/dbg_moel.py 128 2>&1 | grep -v -i warn | tail -5
echo "== moe default"; timeout 900 python okb.py muse_moe_w4a16 --reps 30 2>&1 | grep -E '^(decode|verify)|BAD|PASS|FAIL'
for f in "-DLN=32" "-DLN=32 -DLS_=4" "-DNOWGL"; do echo "== moe $f"; OKB_DBG="$f" timeout 900 python okb.py muse_moe_w4a16 --no-check --reps 30 2>&1 | grep -E '^verify (16|32)'; done
echo "== proj"; python okb.py muse_bf16_proj --no-check --reps 30 2>&1 | grep -E 'router'
echo "== attn"; timeout 900 python okb.py muse_attention_decode --reps 30 2>&1 | grep -E '^(decode|verify)|BAD|PASS|FAIL' | awk '{print $1,$2,$3,$4,$NF}'
