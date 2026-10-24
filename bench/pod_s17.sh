#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0a
cd /workspace/ok
echo "== proj"; python okb.py muse_bf16_proj --no-check --reps 30 2>&1 | grep -E '^(decode|verify)' | awk '{print $1,$2,$3,$5,$NF}'
echo "== moe base"; python okb.py muse_moe_w4a16 --no-check --reps 30 2>&1 | grep -E '^(decode|verify)'
echo "== moe NOBLD"; OKB_DBG="-DNOBLD" python okb.py muse_moe_w4a16 --no-check --reps 30 2>&1 | grep -E '^(decode|verify)'
echo "== moe FAKEDQ"; OKB_DBG="-DFAKEDQ" python okb.py muse_moe_w4a16 --no-check --reps 30 2>&1 | grep -E '^(decode|verify)'
echo "== moe NOBLD FAKEDQ"; OKB_DBG="-DNOBLD -DFAKEDQ" python okb.py muse_moe_w4a16 --no-check --reps 30 2>&1 | grep -E '^(decode|verify)'
