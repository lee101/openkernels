#!/bin/bash
# NVBit 1.8 + Accel-Sim tracer (runs on the local sm_86 GPU; emits sm_86 SASS traces)
set -e
H=$(cd "$(dirname "$0")/.." && pwd); E=$H/ext; P=$E/prefix/usr; T=$E/accel-sim-framework/util/tracer_nvbit
export CUDA_INSTALL_PATH=${CUDA_INSTALL_PATH:-/usr/local/cuda-12} PATH=/usr/local/cuda-12/bin:$PATH
export CPATH=$P/include:$CPATH LIBRARY_PATH=$P/lib/x86_64-linux-gnu:$LIBRARY_PATH
[ -d $T/nvbit_release ] || (cd $T && ./install_nvbit.sh)
make -C $T -j8 > $E/tracer_build.log 2>&1 || { rg -n -m5 "error" $E/tracer_build.log; exit 1; }
ls $T/tracer_tool/tracer_tool.so $T/tracer_tool/traces-processing/post-traces-processing
