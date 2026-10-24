#!/bin/bash
# run_sim.sh <tracedir> [cfg=SM90_H200] [extra gpgpusim flags...]: Accel-Sim trace-driven run; prints cycles + derived time
set -e
H=$(cd "$(dirname "$0")/.." && pwd); E=$H/ext; G=$E/accel-sim-framework/gpu-simulator
D=$(realpath "$1"); C=${2:-SM90_H200}; shift; shift || true
W=$D/sim_$C; rm -rf $W; mkdir -p $W; cp $E/gpgpu-sim/configs/tested-cfgs/$C/* $W/; cp $G/configs/tested-cfgs/$C/trace.config $W/
cd $W
export LD_LIBRARY_PATH=$E/gpgpu-sim/lib/gcc-11.4.0/cuda-12090/release:$LD_LIBRARY_PATH
/usr/bin/time -f "wall %es rss %MKB" $G/bin/release/accel-sim.out -trace $D/traces/kernelslist.g -config ./gpgpusim.config -config ./trace.config "$@" > sim.log 2>&1 || { tail -20 sim.log; exit 1; }
python3 - <<'PY'
import re
cfg = open("gpgpusim.config").read(); log = open("sim.log").read()
clk = int(re.search(r"-gpgpu_clock_domains (\d+)", cfg).group(1))
for k, cyc in zip(re.findall(r"kernel_name = (\S+)", log), re.findall(r"gpu_sim_cycle = (\d+)", log)):
    print(f"kernel {k[:60]} cycles {cyc} -> {int(cyc) / clk:.2f} us at {clk} MHz")
tot = int(re.findall(r"gpu_tot_sim_cycle = (\d+)", log)[-1]); l2 = re.findall(r"L2_BW_total\s*=\s*([\d.]+)", log)
print(f"total {tot / clk:.2f} us, L2_BW_total {l2[-1] if l2 else '?'} GB/s, sim rate {re.findall(r'gpgpu_simulation_rate = (\d+) \(cycle', log)[-1]} cycle/s")
PY
