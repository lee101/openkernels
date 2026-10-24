#!/bin/bash
# trace.sh <outdir> <cmd...>: NVBit SASS trace of the kernels a command launches (local sm_86 GPU), post-processed for Accel-Sim.
# KR="<id range>@<regex>" limits tracing (e.g. KR="0-@.*fdec.*"). CUDA_INJECTION64_PATH (not LD_PRELOAD) so PyTorch processes work.
set -e
H=$(cd "$(dirname "$0")/.." && pwd); T=$H/ext/accel-sim-framework/util/tracer_nvbit/tracer_tool
O=$(realpath -m "$1"); shift; rm -rf "$O"; mkdir -p "$O"
SL=$H/ext/accel-sim-framework/util/tracer_nvbit/others/spinlock_tool/spinlock_tool.so
if [ "${SPIN:-1}" = 1 ]; then  # detect spin loops (2 runs) so the trace keeps 1 iteration of each (cross-CTA waits, mbarrier polls)
  mkdir -p "$O/spinlock_detection"
  for ph in 0 1; do env TRACES_FOLDER="$O" SPINLOCK_PHASE=$ph ${KR:+DYNAMIC_KERNEL_RANGE=$KR} CUDA_INJECTION64_PATH=$SL "$@" > "$O/spin$ph.log" 2>&1; done
  SPINENV="SPINLOCK_HANDLING_MODE=1 SPINLOCK_ITER_TO_KEEP=1"
fi
env $SPINENV USER_DEFINED_FOLDERS=1 TRACES_FOLDER="$O" TOOL_VERBOSE=0 ${KR:+DYNAMIC_KERNEL_RANGE=$KR} CUDA_INJECTION64_PATH=$T/tracer_tool.so "$@" > "$O/app.log" 2>&1
$T/traces-processing/post-traces-processing "$O"/traces/kernelslist_ctx_* > "$O/post.log" 2>&1
ls "$O/traces" | head; du -sh "$O/traces" | cut -f1
