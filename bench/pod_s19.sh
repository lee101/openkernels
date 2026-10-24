#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0a
cd /workspace/ok
echo "== moe lprof"; for b in 256 128; do timeout 400 python bench/lprof_moe.py $b 2>&1 | grep -v -i warn | tail -2; done
echo "== moe"; timeout 600 python okb.py muse_moe_w4a16 --reps 30 2>&1 | grep -E '^(verify)|BAD|PASS|FAIL'
echo "== proj"; timeout 600 python okb.py muse_bf16_proj --reps 30 2>&1 | grep -E 'router|BAD|PASS|FAIL'
for f in "" "-DAW_NOLOAD" "-DAW_NOMMA" "-DAW_NOLOAD -DAW_NOMMA"; do echo "== attn $f"; OKB_DBG="$f" timeout 900 python okb.py muse_attention_decode --no-check --reps 30 2>&1 | grep -E '^verify (16|32)' | awk '{print $1,$2,$3,$4,$NF}'; done
