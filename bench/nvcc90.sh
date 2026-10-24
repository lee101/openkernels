#!/bin/bash
# cross-compile a torch-extension .cu for sm_90a: nvcc90.sh <cu> <outdir>
T=$(python3 -c "import torch,os;print(os.path.dirname(torch.__file__)+'/include')")
P=$(python3 -c "import sysconfig;print(sysconfig.get_paths()['include'])")
mkdir -p "$2"
s=$(date +%s)
/usr/local/cuda-12/bin/nvcc -isystem "$T" -isystem "$T/torch/csrc/api/include" -isystem "$P" -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -gencode=arch=compute_90a,code=sm_90a --compiler-options -fPIC -O3 -lineinfo -std=c++17 -Xptxas -v ${EXTRA} -c "$1" -o "$2/k.o" > "$2/ptxas.log" 2>&1
echo "rc $? $(( $(date +%s) - s ))s"
grep -c "bytes spill" "$2/ptxas.log"
grep "bytes spill" "$2/ptxas.log" | grep -v " 0 bytes spill stores, 0 bytes spill loads" | head
grep -o "Used [0-9]* registers" "$2/ptxas.log" | sort | uniq -c | sort -rn | head -20
