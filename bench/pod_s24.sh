#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0a
cd /workspace/ok
for f in "-DFDGAP" ""; do echo "== diag $f (FDGAP: cta-merge=after wait_group, ranked=after sync1, last-end=after partial stores+sync2, all-in=after merge)"; timeout 400 python -c "
import sys;sys.path.insert(0,'muse_attention_decode')
import importlib.util,reference as R
spec=importlib.util.spec_from_file_location('sub','muse_attention_decode/submission_v34.py');S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)
S._DBG += '$f'.split(); S.DIAG=True
d=R.generate_input(batch=1,ctx=80000,heads='tp8',layer='sliding',seed=3,q_len=1,dist='soft',page=1)
S.kernel_fn(d)" 2>&1 | grep -E "FDIAG|rror" | tail -8; done
