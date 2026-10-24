# virtualcuda: a local "virtual H200"

Estimate how long a CUDA kernel (or a kernel idea that doesn't exist yet) would take on the judge's H200 (sm_90a, 132 SMs), without renting one. Use it to triage ideas before spending `okc check` runs or pod money.

Three tools, ranked by how much to trust them:

| tool | what it is | use it for |
|---|---|---|
| `vc.py est`: analytical model (`vc/model.py`) | roofline per phase, plus latency chains, grid syncs, wave/queue tails and launch/cold-code costs. 7 constants fitted to 40 real H200 measurements, 3 pinned from direct H200 measurements. | absolute and relative estimates for any design, including wgmma/TMA/mbarrier/cluster kernels |
| `vc.py sass`: static SASS analyzer (`vc/sass.py`) | compiles for sm_90a (nvcc 12.9), parses `cuobjdump` SASS with control bits, finds natural loops, sorts instructions by H200 pipe, and simulates one warp's in-order latency from the encoded stall counts and scoreboards | instructions per loop iteration, the 1-warp latency-bound cycles per iteration, tensor cycles: the compute inputs of the model |
| Accel-Sim `SM90_H200` (`sim/`) | cycle-level simulator replaying NVBit SASS traces from the local RTX 3090 Ti (sm_86) | standalone sm_86-compatible microkernels only (see the limits below). It is a cross-check, not a predictor for our kernels. |

Supporting tools: `sim/ncu_daisy.sh` / `sim/ncu_daisy_py.sh` run ncu on daisy's RTX 3090 (sm_86, sudo) and return executed instructions per pipe plus DRAM/L2 bytes. `vc.py ncu2spec` turns that CSV into an H200 estimate.

## Quick start (new kernel idea)

```sh
cd virtualcuda
# 1. compute side: loop instruction count, 1-warp cycles/iteration, tensor cycles of a prototype (any .cu or submission .py)
python3 vc.py sass ../muse_attention_decode/submission_v38.py -D RB=8 -D RB2=6 -D RB3=20 -D RBF=5 --func 'attnw<true, true, true>'
# 2. describe the design: copy examples/new_idea_template.py (or a builder in vc/cases.py) and fill bytes/instrs/syncs
python3 vc.py est examples/attnw_variants.py          # variants side by side, ratio to the first one
python3 vc.py est examples/new_idea_template.py -v    # per-phase breakdown: mem / issue / tensor / serial / tail / latency
# 3. sanity: calibration still holds (40 H200 cases, about 2 s)
python3 vc.py calib --cv
```

A spec is either JSON (`examples/gemv_stream.json`) or a Python file whose `kernels()` returns a list of `Kernel`s, or a dict `{variant: [Kernel, ...]}` for comparisons. Use `--set k=v` to override a constant, e.g. `python3 vc.py est spec.py bw_hbm=4.0e12`.

## The model (`vc/model.py`)

A timed call is the per-call graph overhead plus a sum over kernels. Each kernel costs its launch/ramp, its cold-code cost (`icache` × static SASS size) and its straight-line cost (`fstraight` × the longest 1-warp path outside loops, from `vc sass`), plus the sum of its phases. Each phase is:

```
t_mem    = hbm/BW_stream + g64/BW_64B + g128/BW_128B + g256/BW_256B   (scaled by min(1, 1.6*sms/132); Little's law if `inflight` is given)
           max'd with l2/BW_L2
t_tp     = max(issue/(4*sms*ipc*clk), pipe cycles, tensor SM-cycles/sms/clk)
t_thr    = max( (t_mem^3 + t_tp^3)^(1/3),  serial_cycles*imbalance/clk )   # memory vs throughput overlap; latency-bound warps hide memory
t_phase  = tail(t_thr; units, workers, dynamic|static) + fill(1 DRAM latency) + chain*lat + l2chain*l2lat + syncs*t_sync + l2_sm/BW_L2_per_SM
```

Inputs per phase: bytes by level and access type (stream/TMA, random rows of 64/128/256 B, L2 hits), warp-instructions, tensor SM-cycles (dense bf16 wgmma: FLOPs/4096 per SM-cycle; FP8: /8192; mma.sync: HMMA.16816 = 1.52 SM-cycles, measured 645 TF), the critical warp's serial cycles (iterations × `vc sass` 1-warp cycles), work units over workers (wave quantization or a dynamic-queue tail), dependent DRAM/L2 round trips, grid syncs, and the bytes a single SM must move alone (last-arriver merges).

Constants (`vc/params.json` stores the fitted ones):

| constant | value | source |
|---|---:|---|
| `bw_hbm`, streamed DRAM read | 4.25 TB/s | fit (prior 4.2: pod `cuk bw`) |
| `bw_g64/128/256`, random row gathers | 2.25 / 3.0 / 3.3 TB/s | H200 `bench/gather.cu` (fixed) |
| `bw_l2` | 6.1 TB/s | fit |
| `bw_l2_sm`, one SM's L2 bandwidth | 79 GB/s | fit (prior 64 B/clk) |
| `lat`, dependent DRAM round trip | 0.89 µs | fit |
| `l2lat` | 0.35 µs | fixed |
| `t_kernel`, launch/ramp per kernel | 0.91 µs | fit |
| `t_sync`, grid sync or flag merge | 0.92 µs | fit |
| `t_graph`, judge time minus summed kernel time | 0.64 µs | fit |
| `fstraight` | 0.23 | fit |
| `ipc` | 1.0 | pinned: H200 awprof, attnw at 4 WG/SM runs ~2850 cycles per 2940-instruction round |
| `nsm`, mem/issue overlap exponent | 3 | pinned: attnw "no KV loads" ablation (−11.6%) |
| `icache`, µs per 1000 static instructions | 0.6 | pinned: MoE first-iteration stall of ~5 µs (8.4K instructions) |

`vc sass` latencies: global load waited inside a loop 250 cycles (an L2 hit), LDS 33, MUFU/XU 18, SHFL 24, HMMA 32, plus 96 cycles at `WARPGROUP.DEPBAR` after a GMMA. Fixed-latency ops are timed from the stall counts the compiler encodes. Check: for the attnw v38 step the analyzer gives 2118 cycles for one warp, and H200 clock64 measured 2230 cycles per step at 2 WG/SM, so −5%.

## Calibration (H200 ground truth)

Measured values are the `okc check` judge medians (`yours`, µs, graph replay minus 4 µs) from `muse_*/profiles`, snapshotted in `data/h200_measured.json` (`python3 -m vc.collect`), plus pod microbenchmarks (TMA stream, graph-empty). SASS features come from the judged sources (`data/sass_db.json`, `python3 -m vc.sassdb --refresh`). Shapes, grids and ring depths come from each submission's host code. No per-case tuning was done.

The fit uses 40 cases. MoE is held out: its per-stage stalls are not modeled (see Limits), and fitting it would bias the global constants. Error columns:
- **error**: the fitted model.
- **LOO**: the case left out of the fit.
- **LOGO**: its whole family left out of the fit, an honest proxy for a new kernel family.

| family | case | H200 measured µs | predicted µs | error | LOO | LOGO | bound |
|---|---|---:|---:|---:|---:|---:|---|
| ubench | TMA stream 16.8 MB | 5.7 | 5.2 | -9% | -11% | -11% | mem |
| ubench | TMA stream 37.7 MB | 10.0 | 10.1 | +1% | +2% | -1% | mem |
| ubench | TMA stream 67.1 MB | 16.5 | 17.1 | +3% | +5% | +2% | mem |
| cublas | o m1 | 6.8 | 7.7 | +13% | +13% | +15% | mem |
| cublas | o m16 | 6.9 | 7.7 | +11% | +11% | +13% | mem |
| cublas | o m32 | 6.8 | 7.7 | +13% | +14% | +15% | mem |
| cublas | o m8 | 6.9 | 7.7 | +11% | +12% | +13% | mem |
| cublas | qkv_gate m16 | 13.9 | 14.4 | +4% | +4% | +5% | mem |
| cublas | qkv_gate m32 | 14.0 | 14.6 | +4% | +4% | +6% | mem |
| cublas | qkv_gate m8 | 14.2 | 14.3 | +1% | +1% | +3% | mem |
| cublas | pooler m16 | 21.3 | 21.4 | +0% | +0% | +1% | mem |
| cublas | pooler m32 | 22.1 | 21.6 | -2% | -2% | -1% | mem |
| cublas | pooler m8 | 21.4 | 21.3 | -1% | -1% | +1% | mem |
| cublas | unpooler m16 | 18.3 | 19.5 | +7% | +7% | +7% | mem |
| cublas | unpooler m32 | 18.5 | 19.6 | +6% | +6% | +7% | mem |
| cublas | o m128 | 7.7 | 7.9 | +3% | +4% | +5% | mem |
| cublas | unpooler m128 | 20.0 | 20.6 | +3% | +3% | +4% | mem |
| cublas | pooler m128 | 24.9 | 20.8 | -16% | -17% | -16% | mem |
| cublas | o m256 | 9.9 | 9.0 | -9% | -9% | -7% | tensor |
| cublas | unpooler m256 | 24.1 | 24.9 | +3% | +3% | +4% | tensor |
| cublas | pooler m256 | 30.7 | 25.2 | -18% | -18% | -17% | tensor |
| cublas | qkv_gate m256 | 19.6 | 16.1 | -18% | -18% | -17% | mem |
| bf16 ours | skinny qkv_gate m1 | 13.5 | 13.0 | -3% | -4% | -2% | mem |
| bf16 ours | skinny unpooler m1 | 5.8 | 6.0 | +3% | +4% | +7% | mem |
| bf16 ours | rtr router m1 | 5.5 | 6.3 | +14% | +17% | +25% | mem |
| bf16 ours | rtr router m16 | 6.0 | 6.3 | +4% | +5% | +14% | mem |
| bf16 ours | rtr router m32 | 6.9 | 6.3 | -10% | -12% | -1% | mem |
| bf16 ours | rtr router m8 | 5.7 | 6.3 | +10% | +12% | +21% | mem |
| attention | fdec decode B1 sliding | 8.9 | 12.0 | +35% | +43% | +38% | latency |
| attention | attnw verify B1 sliding | 16.2 | 16.3 | +1% | +1% | -6% | latency |
| attention | fdec decode B1 full | 21.0 | 24.0 | +14% | +21% | +6% | latency |
| attention | attnw verify B1 full | 26.0 | 20.8 | -20% | -22% | -21% | issue |
| attention | fdec decode B16 sliding | 12.6 | 13.2 | +5% | +6% | +9% | mem |
| attention | attnw verify B16 sliding | 21.4 | 16.3 | -24% | -28% | -29% | latency |
| attention | fdec decode B16 full | 113.0 | 124.6 | +10% | +10% | +11% | mem |
| attention | attnw verify B16 full | 156.9 | 161.1 | +3% | +3% | +1% | issue |
| attention | fdec decode B32 sliding | 17.0 | 15.9 | -6% | -7% | -3% | mem |
| attention | attnw verify B32 sliding | 24.7 | 19.3 | -22% | -24% | -26% | issue |
| attention | fdec decode B32 full | 205.5 | 238.7 | +16% | +16% | +16% | mem |
| attention | attnw verify B32 full | 293.8 | 303.6 | +3% | +3% | +3% | issue |
| moe (held out) | moe bs1 | 52.4 | 30.9 | -41% | | | mem+tail |
| moe (held out) | moe bs16 | 136.2 | 87.9 | -35% | | | mem+tail |
| moe (held out) | moe bs32 | 174.9 | 110.3 | -37% | | | mem+tail |
| moe (held out) | moe bs8 | 105.5 | 69.2 | -34% | | | mem+tail |
| moe (held out) | moe bs128 | 302.1 | 253.1 | -16% | | | mem+tail |
| moe (held out) | moe bs256 | 424.4 | 454.4 | +7% | | | mem+tail |

On the 40 fitted cases: median |error| is **8.6%** in the fit, **9.5%** leave-one-out and **8.1%** leave-one-family-out. 80% of LOO predictions land within 20%. Per family (LOGO median): TMA stream 2%, cuBLAS 7%, our bf16 kernels 11%, attention 10%. MoE, never fitted, is 16 to 41% too fast at small batch.

The model also predicts variant-over-baseline ratios. Measured values are H200 pod ablations and judge A/B runs:

| change | measured ratio | predicted ratio |
|---|---:|---:|
| attnw b16 verify full: no KV loads | 0.884 | 0.844 |
| attnw b16 verify full: no MMA | 0.939 | 0.993 |
| attnw b16 verify full: neither | 0.848 | 0.834 |
| attnw b32 verify full: no KV loads | 0.885 | 0.838 |
| attnw b32 verify full: no MMA | 0.939 | 0.992 |
| attnw b32 verify full: neither | 0.843 | 0.827 |
| attnw b16 verify full: loop 786 to 735 SASS (v37 to v38, judge) | 0.976 | 0.964 |
| attnw b32 verify full: loop 786 to 735 SASS (v37 to v38, judge) | 0.972 | 0.962 |
| moe bs1: FAKEDQ (no dequant ALU) | 0.820 | 0.927 |

The model gets the direction right in every case. It overstates savings from removed instructions or bytes by about 1.5×. It misses that MMA and dequant work also sits on the critical path (it gives MMA almost no weight and the dequant ALU about 40% of its measured weight).

## Simulators and other tools

**Accel-Sim** (dev, built here by `sim/build_accelsim.sh`, which fetches bison, flex, m4 and zstd headers as debs without sudo). It ships `SM90_H100`/`SM90_H200` configs and parses Hopper SASS (HGMMA/QGMMA/UTMALDG/SYNCS). But traces come from NVBit running on real hardware, and the only local GPUs are sm_86. So it can replay sm_86 traces on an H200-like memory system and SM count. It cannot replay our sm_90a paths (wgmma, TMA, clusters).

`sim/trace.sh` handles tracing:
- it uses `CUDA_INJECTION64_PATH`, because `LD_PRELOAD` silently traces nothing under PyTorch;
- it runs two-phase spinlock detection plus fast-forward;
- `KR=` filters kernels.

`sim/run_sim.sh` runs the simulation. Speed is about 150 to 3000 simulated cycles per second, so 1 to 10 minutes per microbenchmark.

| microkernel (sm_86 trace, 264 CTAs) | H200 measured | Accel-Sim SM90_H200 | error |
|---|---:|---:|---:|
| stream read, 128 MB (LDG.128) | ~4.2 TB/s (`cuk bw`) | 3.93 TB/s | −7% |
| random 256 B rows, cp.async ring | 3.2–3.4 TB/s | 3.38 TB/s | ~0% |
| random 128 B rows | 2.9–3.1 TB/s | 3.50 TB/s | +17% |
| random 64 B rows | 2.1–2.4 TB/s | 3.42 TB/s | +52% |
| mma.sync m16n8k16 bf16 | 645 TF | 755 TF | +17% |
| fdec decode B16 sliding (v38 sm_86 build), raw trace | 10.7 µs kernel | 318.6 µs | ×30 |
| same, spinlock fast-forward (`trace.sh` default) | 10.7 µs | 22.7 µs | ×2.1 |
| fdec decode B32 sliding, raw trace | 14.7 µs | 229.1 µs | ×16 |
| fdec decode B1 full, raw trace | 18.8 µs | 84.8 µs | ×4.5 |

The simulator gets streams roughly right. It does not model the HBM row/sector cost of small random rows, so the 64 B row case comes out 1.5× too fast. It is unusable for our real kernels:
1. Grids sized for 132 SMs with 1 CTA/SM run in 2 waves on the 82-SM 3090. The cross-CTA merge spin-waits recorded on the 3090 (NANOSLEEP/CCTL loops, 26K iterations) are replayed as work. Spinlock fast-forward removes most of it (318 → 22.7 µs) but not the rest.
2. The sm_86 build is a different kernel (RBF=2 ring, no wgmma or TMA).
3. A 100 µs kernel takes 10+ minutes to simulate.

**GPGPU-Sim PTX mode** has no `mma.sync`, `ldmatrix`, `cp.async`, `mbarrier`, `wgmma` or `cp.async.bulk` in its PTX front end (`src/cuda-sim/opcodes.def`), so it cannot run even our sm_86 paths. Not pursued.

**gpuroofperf-toolkit** collects metrics with nvprof, and nvprof does not support compute capability ≥ 8.0, so it cannot profile the 3090s. Its model is a two-term roofline (ops vs DRAM bytes), which `vc/model.py` already covers. Instead, `sim/ncu_daisy.sh` collects the equivalent kernel parameters with ncu on daisy. For example, on the stream microbench, `vc.py ncu2spec` gives 32.9 µs on H200 for 128 MB, about 4.0 TB/s. On fdec B16 sliding it gives 12.0 µs judge-style vs 12.6 measured. But the 3090 counts include the 2-wave spin, so treat ncu numbers from 132-SM-sized grids with suspicion.

## Limits (what the analytical model cannot see)

- **Dynamic paths.** SASS counts are static. Branches inside loops count as always taken, and straight-line cost is a longest path, not a profile. Short, latency-dominated kernels (decode B1 sliding, verify sliding) are off by 20 to 35% in either direction.
- **Stalls the static analyzer can't see.** MoE's 2-deep per-warp TMA ring runs at a measured ~3.9 µs per 4.6 KB stage vs ~1.4 µs predicted from SASS, which makes the small-batch MoE 35 to 41% too fast. The same blind spot hides wgmma/DEPBAR waits in the attention step loop. Rule: if a design relies on shallow per-warp rings, fences per tile or many joins, add measured per-unit latency (`unit_cyc`, `unit_sync`) or trust only ratios.
- **cuBLAS internals** (tile shapes, split-K efficiency at m = 128 to 256) are unknown: ±18%.
- **Modeled as aggregates only:** HBM row-buffer and partition effects beyond the measured gather table, L2 partition crossing, bank conflicts, register spills (`vc sass` shows LDL/STL counts) and clock throttling.
- **Clusters and DSMEM:** only whatever the user writes down as syncs or bytes.

## Recommended workflow

1. **New idea:** write the phases (bytes by access type, instructions, syncs) using `vc/cases.py` builders or `examples/new_idea_template.py`. If a prototype exists, take the loop instructions and 1-warp cycles from `vc.py sass` (compiles sm_90a locally, no GPU needed). Run `vc.py est`.
2. **Decide by ratio vs the current kernel**, modeled the same way. Pursue ideas the model says are at least 10 to 15% faster; the calibration noise is about 10%. Treat predicted ratios as optimistic, since savings were overstated about 1.5× above. Drop ideas that are memory-bound already at the gather or stream roofline.
3. **Correctness:** check locally (`okb.py`, 3090 Ti). For an independent instruction and byte count of the sm_86 path, run `sim/ncu_daisy_py.sh` (light use; daisy serves production).
4. **Last step:** `okc check` or a short pod run. Add the new H200 number to `data/h200_measured.json` (`python3 -m vc.collect`) and a case in `vc/cases.py`, then refit (`vc.py calib --fit --cv --save`) so the model keeps learning.

## Files

- `vc.py`: CLI.
- `vc/model.py`: the model.
- `vc/sass.py`: the SASS analyzer.
- `vc/sassdb.py`: cached SASS features (`data/sass_db.json`).
- `vc/cases.py`: calibration cases and kernel builders (`fdec`, `attnw`, `moe`, `skinny`, `rtr`, `cublas_gemv`, `tma_stream`).
- `vc/calib.py`: fitting and cross-validation.
- `vc/collect.py`: profile snapshot.
- `vc/params.json`: fitted constants.
- `data/`: measured H200 numbers, SASS features, the calibration output, Accel-Sim results (`accelsim_results.json`).
- `sim/`:
  - `build_accelsim.sh`, `build_tracer.sh`
  - `trace.sh`, `run_sim.sh`
  - `ubench.cu`: microkernels
  - `okc_case.py`: run one scored shape
  - `ncu_daisy.sh`, `ncu_daisy_py.sh`
- `examples/`: spec templates.
- `tests/test_vc.py`: tests; run with `python3 tests/test_vc.py`.
- `ext/`: simulator sources and builds (gitignored).
- `.cache/`: cubins and traces (gitignored).
