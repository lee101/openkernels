#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0a
cd /workspace/ok
echo "== sweep"; timeout 900 python bench/attn_sweep.py 2>&1 | grep -E "^sliding|rror"
echo "== diag"; timeout 300 python -c "
import sys;sys.path.insert(0,'muse_attention_decode')
import importlib.util,reference as R
spec=importlib.util.spec_from_file_location('sub','muse_attention_decode/submission_v34.py');S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)
S.DIAG=True
d=R.generate_input(batch=1,ctx=80000,heads='tp8',layer='sliding',seed=3,q_len=1,dist='soft',page=1)
S.kernel_fn(d)" 2>&1 | grep -v -i warn | tail -30
