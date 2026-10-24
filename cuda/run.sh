#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
make -s
{
  echo "# local 3090 Ti kerneless experiments"
  echo "# $(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=name,compute_cap,clocks.sm,clocks.mem,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader
  echo
  ./gather
  echo
  ./cvt
  echo
  ./reuse
  echo
  ./split
} | tee results.txt
