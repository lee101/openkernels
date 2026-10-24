#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0a
cd /workspace/ok
echo "== awt"; timeout 600 python bench/awt.py 2>&1 | grep -E "B=|rror"
echo "== moe diag v9"; timeout 600 python -c "
import sys;sys.path.insert(0,'muse_moe_w4a16')
import importlib.util,reference as R
spec=importlib.util.spec_from_file_location('sub','muse_moe_w4a16/submission_v9.py');S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)
S.DIAG=True
W=R.generate_weights(1); st=S.prepare(W)" 2>&1 | grep -E "DIAG|rror" | tail -8
