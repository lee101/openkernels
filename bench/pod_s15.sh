#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0a GHZ=1.8
cd /workspace/ok
echo "== w4wg"; for a in "1024 100" "8192 20" "8192 40" "8192 140" "4096 140" "8192 600"; do echo "[$a]"; python bench/w4wg.py $a 2>&1 | grep -v -i warn; done
echo "== attn"; python okb.py muse_attention_decode --no-check --reps 30 2>&1 | grep -E '^(decode|verify)' | awk '{print $1,$2,$NF}' | tr '\n' ' '; echo
echo "== proj"; python okb.py muse_bf16_proj --no-check --reps 30 2>&1 | grep -E '^(decode|verify)' | awk '{print $1,$2,$3,$4,$NF}' | tr '\n' ' '; echo
