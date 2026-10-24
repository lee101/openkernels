#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0a
cd /workspace/ok
(apt-get update -qq && apt-get install -y -qq libnuma1) >/dev/null 2>&1
pip install -q "sgl-kernel==0.3.9.post2" --no-deps 2>&1 | grep -v notice | tail -1; python -c "from sgl_kernel.flash_attn import flash_attn_with_kvcache; print('sgl ok')" 2>&1 | tail -1
echo "== fa3 vs ours"; timeout 900 python bench/fa3_vs.py 2>&1 | grep -E "ours|rror"
echo "== prefill profile"; timeout 900 python bench/pf_prof.py 2>&1 | grep -v -i warn | grep -E "==|Self CUDA|moe|attn|nvjet|local expert|Error|---" | head -40
