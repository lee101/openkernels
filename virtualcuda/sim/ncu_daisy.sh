#!/bin/bash
# ncu_daisy.sh <file.cu> [args...]: build a standalone CUDA program for sm_86 on daisy (RTX 3090, ncu under sudo) and
# collect a small metric set per kernel (DRAM/L2 bytes, executed instructions per pipe, time). CSV -> stdout.
# Light GPU use only (daisy serves production): one launch per kernel, few replay passes. Needs DAISY_SUDO_PW in the env.
# Feed the CSV to `python3 vc.py ncu2spec <csv>` for an H200 estimate from the measured (sm_86) instruction mix.
set -e
[ -n "$DAISY_SUDO_PW" ] || { echo "set DAISY_SUDO_PW (sudo password on daisy)" >&2; exit 1; }
H=${DAISY_HOST:-lee@daisy}; C=/usr/local/cuda-12.8/bin; D=/tmp/vc_ncu_$USER
SRC=$1; shift; B=$(basename "$SRC" .cu)
M="gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,lts__t_sector_hit_rate.pct,sm__inst_executed.sum,\
smsp__inst_executed_pipe_fma.sum,smsp__inst_executed_pipe_alu.sum,smsp__inst_executed_pipe_fp16.sum,smsp__inst_executed_pipe_tensor.sum,\
smsp__inst_executed_pipe_xu.sum,smsp__inst_executed_pipe_lsu.sum,smsp__inst_executed_pipe_uniform.sum,sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__grid_size,launch__block_size,launch__registers_per_thread,smsp__average_warp_latency_per_inst_issued.ratio"
ssh -o BatchMode=yes $H "mkdir -p $D"
scp -q "$SRC" $H:$D/$B.cu
ssh -o BatchMode=yes $H "cd $D && $C/nvcc -O3 -arch=sm_86 -lineinfo $B.cu -o $B && sudo -S -p '' $C/ncu --csv --metrics $M ./$B $*" <<< "$DAISY_SUDO_PW" | rg '^"'
