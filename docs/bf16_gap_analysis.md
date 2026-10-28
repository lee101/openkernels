# muse_bf16_proj: where is the slack after v22?

Ours: v22, judge 1.459 (CI 1.437-1.481). Leader: 1.45. The margin is inside the CI, so we need about +1.5-2% more.

## 1. Shapes, bytes, floors

Sources: `muse_bf16_proj/reference.py`, `profiles/submission_v22.66a91507ca9a4603.{txt,log}` (judge H200, per-shape SGLang and ours, kernel names), `bench/wgr_h100_run.txt` / `bench/wgd_h100_run.txt` (H100 prepare-time candidate times).

- Weights dominate every shape. x + out is below 0.8 MB for m <= 32 and 2.2-6.3 MB at m 128/256.
- `streamed` is what our chosen path actually reads: c12 (lossless 12-bit, about 0.75 of BF16) for the GEMV shapes (m <= 32, except o m32, which runs cuBLAS), BF16 for cuBLAS shapes, BF16 hi + lo (= the fp32 bytes) for the router.
- Floors: bytes / 3.9 TB/s (the realistic H200 streaming rate; 4.8 is the spec peak), and the dense-BF16 tensor floor 2mnK / 989 TFLOP/s. `ours/max(floor)` uses the streamed bytes and the larger of the two floors.
- The judge subtracts about 4 µs of graph-replay cost from both sides, but a single small kernel still carries roughly 1.5-2.5 µs of launch, first-byte and tail latency. That shows up as the 2-2.8x ratios on the 6-8 MB shapes (unpooler m1/m8, router).

Per shape (µs):

| load | layer | m | N x K | W MB | streamed MB (path) | x+out MB | t@3.9 W | t@3.9 streamed | t@989TF | SGLang | ours | ours/max(floor) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| decode 1 | qkv_gate | 1 | 2304x8192 | 37.7 | 28.3 (c12) | 0.02 | 9.68 | 7.26 | 0.04 | 13.79 | 9.69 | 1.33 |
| decode 1 | o | 1 | 8192x1024 | 16.8 | 12.6 (c12) | 0.02 | 4.31 | 3.23 | 0.02 | 6.66 | 4.84 | 1.50 |
| decode 1 | pooler | 1 | 4096x8192 | 67.1 | 50.3 (c12) | 0.02 | 17.21 | 12.91 | 0.07 | 21.55 | 13.95 | 1.08 |
| decode 1 | unpooler | 1 | 1024x4096 | 8.4 | 6.3 (c12) | 0.01 | 2.15 | 1.62 | 0.01 | 7.04 | 4.25 | 2.63 |
| decode 1 | router | 1 | 256x8192 | 8.4 | 8.4 (fp32) | 0.02 | 2.16 | 2.16 | 0.00 | 10.46 | 3.96 | 1.84 |
| decode 16 | qkv_gate | 16 | 2304x8192 | 37.7 | 28.3 (c12) | 0.34 | 9.77 | 7.35 | 0.61 | 14.03 | 9.62 | 1.31 |
| decode 16 | o | 16 | 8192x1024 | 16.8 | 12.6 (c12) | 0.29 | 4.38 | 3.30 | 0.27 | 6.78 | 5.62 | 1.70 |
| decode 16 | pooler | 16 | 4096x8192 | 67.1 | 50.3 (c12) | 0.39 | 17.31 | 13.01 | 1.09 | 21.23 | 13.70 | 1.05 |
| decode 16 | unpooler | 16 | 8192x4096 | 67.1 | 50.3 (c12) | 0.39 | 17.31 | 13.01 | 1.09 | 18.54 | 14.59 | 1.12 |
| decode 16 | router | 16 | 256x8192 | 8.4 | 8.4 (bf16 hi+lo) | 0.28 | 2.22 | 2.22 | 0.07 | 14.69 | 4.63 | 2.08 |
| decode 32 | qkv_gate | 32 | 2304x8192 | 37.7 | 28.3 (c12) | 0.67 | 9.85 | 7.43 | 1.22 | 14.30 | 13.67 | 1.84 |
| decode 32 | o | 32 | 8192x1024 | 16.8 | 16.8 (bf16 cuBLAS) | 0.59 | 4.45 | 4.45 | 0.54 | 6.62 | 6.62 | 1.49 |
| decode 32 | pooler | 32 | 4096x8192 | 67.1 | 50.3 (c12) | 0.79 | 17.41 | 13.11 | 2.17 | 22.14 | 17.02 | 1.30 |
| decode 32 | unpooler | 32 | 8192x4096 | 67.1 | 50.3 (c12) | 0.79 | 17.41 | 13.11 | 2.17 | 18.61 | 16.57 | 1.26 |
| decode 32 | router | 32 | 256x8192 | 8.4 | 8.4 (bf16 hi+lo) | 0.56 | 2.29 | 2.29 | 0.14 | 15.15 | 6.01 | 2.62 |
| verify 1 | qkv_gate | 8 | 2304x8192 | 37.7 | 28.3 (c12) | 0.17 | 9.72 | 7.30 | 0.31 | 13.98 | 10.10 | 1.38 |
| verify 1 | o | 8 | 8192x1024 | 16.8 | 12.6 (c12) | 0.15 | 4.34 | 3.26 | 0.14 | 6.69 | 5.30 | 1.62 |
| verify 1 | pooler | 8 | 4096x8192 | 67.1 | 50.3 (c12) | 0.20 | 17.26 | 12.96 | 0.54 | 21.31 | 14.10 | 1.09 |
| verify 1 | unpooler | 8 | 1024x4096 | 8.4 | 6.3 (c12) | 0.08 | 2.17 | 1.63 | 0.07 | 7.04 | 4.54 | 2.78 |
| verify 1 | router | 8 | 256x8192 | 8.4 | 8.4 (bf16 hi+lo) | 0.14 | 2.19 | 2.19 | 0.03 | 26.34 | 4.44 | 2.03 |
| verify 16 | qkv_gate | 128 | 2304x8192 | 37.7 | 37.7 (bf16 cuBLAS) | 2.69 | 10.37 | 10.37 | 4.89 | 16.74 | 15.35 | 1.48 |
| verify 16 | o | 128 | 8192x1024 | 16.8 | 16.8 (bf16 cuBLAS) | 2.36 | 4.91 | 4.91 | 2.17 | 7.15 | 6.65 | 1.36 |
| verify 16 | pooler | 128 | 4096x8192 | 67.1 | 67.1 (bf16 cuBLAS) | 3.15 | 18.01 | 18.01 | 8.69 | 24.72 | 19.61 | 1.09 |
| verify 16 | unpooler | 128 | 8192x4096 | 67.1 | 67.1 (bf16 cuBLAS) | 3.15 | 18.01 | 18.01 | 8.69 | 19.97 | 18.67 | 1.04 |
| verify 16 | router | 128 | 256x8192 | 8.4 | 8.4 (bf16 hi+lo) | 2.23 | 2.72 | 2.72 | 0.54 | 16.13 | 7.09 | 2.60 |
| verify 32 | qkv_gate | 256 | 2304x8192 | 37.7 | 37.7 (bf16 cuBLAS) | 5.37 | 11.06 | 11.06 | 9.77 | 19.81 | 17.70 | 1.60 |
| verify 32 | o | 256 | 8192x1024 | 16.8 | 16.8 (bf16 cuBLAS) | 4.72 | 5.51 | 5.51 | 4.34 | 9.41 | 9.07 | 1.65 |
| verify 32 | pooler | 256 | 4096x8192 | 67.1 | 67.1 (bf16 cuBLAS) | 6.29 | 18.82 | 18.82 | 17.37 | 30.48 | 24.23 | 1.29 |
| verify 32 | unpooler | 256 | 8192x4096 | 67.1 | 67.1 (bf16 cuBLAS) | 6.29 | 18.82 | 18.82 | 17.37 | 24.34 | 24.09 | 1.28 |
| verify 32 | router | 256 | 256x8192 | 8.4 | 8.4 (bf16 hi+lo) | 4.46 | 3.29 | 3.29 | 1.09 | 17.50 | 9.17 | 2.78 |

Per load (weight 62 on every shape, so a load is the plain sum of its five shapes; the score is the geomean over the six loads of SGLang sum / ours):


| load | SGLang sum | ours sum | speedup | floor sum (bf16 bytes) | floor sum (streamed) | ours/floor |
|---|---|---|---|---|---|---|
| decode 1 | 59.5 | 36.7 | 1.622 | 35.5 | 27.2 | 1.35 |
| decode 16 | 75.3 | 48.2 | 1.563 | 51.0 | 38.9 | 1.24 |
| decode 32 | 76.8 | 59.9 | 1.283 | 51.4 | 40.4 | 1.48 |
| verify 1 | 75.4 | 38.5 | 1.958 | 35.7 | 27.3 | 1.41 |
| verify 16 | 84.7 | 67.4 | 1.257 | 54.0 | 54.0 | 1.25 |

Recomputed from the per-shape medians this gives gm 1.459, matching the judge.

## 2. What a microsecond is worth

The geomean gain from saving 1 µs on one shape is ln(L / (L - 1)) / 6, with L the load's sum, so cheap loads pay most:

| load | ours sum | gm gain per µs saved |
|---|---|---|
| decode 1 | 36.7 | 0.46% |
| verify 1 | 38.5 | 0.44% |
| decode 16 | 48.2 | 0.35% |
| decode 32 | 59.9 | 0.28% |
| verify 16 | 67.4 | 0.25% |
| verify 32 | 84.3 | 0.20% |

Examples: qkv m32 13.7 to 11.0 gives +0.77%; o m32 6.6 to 5.0 gives +0.45%; both together give +1.24%. qkv m128 15.35 to 13.0 gives +0.59%. unpooler m256 minus 1 µs gives +0.20%. Half a µs off every d1 shape gives +1.18%, and the same on v1 gives +1.12%.

## 3. Where the slack is

1. **decode 32 is the worst load (1.48x its streamed floor), and qkv m32 holds most of it: 13.7 µs against a 7.4 µs c12 floor, 6.2 µs of slack.** The c12 GEMV is fine up to m16 (9.6 µs, 1.31x). From m16 to m32 qkv gains 4 µs, pooler 3.3 and unpooler 2. The mma.sync swap-AB GEMV re-reads x once per 16-row tile: qkv's 512 KB x is read 144 times (74 MB through L1/L2, 2.6x the weight bytes). MT=4 also needs more than 128 registers: v22's push + uncapped variant cut 15.2 to 12.5 µs on the H100, but the judge's re-time picked V9, which gave a 12.35 µs prepare time and 13.67 µs on the judge. Fix direction: a CTA-shared x tile and wgmma.
   - **Swap-AB wgmma (v24 `wgs`).** W rows become the M side, 64 per warpgroup from smem via TMA. The 32 tokens are N. x is read N / BM = 9-36 times instead of 144, and a 64-row x 32-token x k16 tile is one instruction. Plain BF16 costs 9.4 MB (2.4 µs) more DRAM than c12 for qkv, so the realistic target is about 11 µs for qkv m32 and about 5 µs for o m32 (BF16 floor 4.45 µs). c12 decoded into the smem A tile, as in the v17 `wgc` decoder warps, would be the next step if `wgs` streams well.
2. **o m32 runs cuBLAS** (`nvjet_sm90_tst_64x32_64x16_2x1`: 4.7 µs kernel, 6.62 µs on the judge, x1.00 against SGLang, which runs the same kernel). Our c12 GEMV gets 5.89 µs prepare time on the H200 against 5.63 for cuBLAS, so it is not picked. The c12 floor is 3.3 µs. With K = 1024 there are only 16 k-chunks per row tile: the GEMV is latency- and launch-bound and its x re-reads are 512 tiles x 64 KB = 33 MB. Same fix as (1).
3. **m128/m256 dense shapes are close to what cuBLAS can do.** pooler and unpooler m256 have a tensor floor of 17.4 µs at the 989 TF peak; cuBLAS runs them at 24.1-24.2 µs, which is about 72% of peak and near the practical limit for a BF16 GEMM of this size. qkv m128/256 and o m256 have more relative slack (1.5-1.65x) but are hard: the v17 c12 wgmma GEMM (`wgc`), the v19 `wgb` and the v23 one-wave `wgd` (BN 64-256, workspace split-K) all lost to nvjet by 1.3-2x on the H100. For `wgd` the configuration with the least x re-reading (BN 256) was the slowest, so the limit is the TMA ring's streaming rate and not the x traffic or the split-K reduction.
   - **unpooler m256** runs torch's default cuBLAS heuristic: `nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT`, 24.0 µs. The prepare cuBLASLt search tried 440 configurations and kept torch. SGLang runs the same kernel, hence x1.01. A persistent wgmma kernel would have to beat nvjet at 72% of tensor peak. Our best hand-written attempt (`wgd` 128.6.1.2) was 30.8 µs against 23.7 on the H100, so this is not a near-term lever. Cheaper: give the cuBLASLt search a larger budget or more seeds for unpooler m256 (`LTW` 0.7).
4. **d1 / v1 small shapes carry per-launch latency.** unpooler m1/m8 (6.3 MB) take 4.3-4.5 µs, router m1/m8 4.0-4.4 µs, o m1/m8 4.8-5.3 µs: 1.5-2.9 µs above their floors, and each µs here is worth 0.44-0.46%. Levers: earlier first bytes (the GEMV prologue issues the escape-list load and the first weight ring before anything else), fewer CTAs with deeper rings at m1, and checking that the 6.3 MB unpooler slice is not limited by too few CTAs.
5. **The router is done.** m128 7.1 µs and m256 9.2 µs (x2.27 and x1.91), m8-32 4.4-6.0 µs.

## 4. Plan (v24)

- `wgs` (swap-AB wgmma on plain BF16 W, split-K over S plain CTAs through the wgr workspace + counter, S 1 stores BF16 directly) as prepare-timed candidates for o / qkv / unpooler / pooler at m32 and o / qkv at m16 (v24diag adds m8 and wider geometry lists). A candidate is adopted only if it is 2% faster than the best v22 candidate after re-timing (WSMARGIN 0.98). v24diag prints a `WD <layer> m<bucket> pick <name> | v12 <us> old <best v22 candidate> <us> | new <best wgs> <us>` line for every shape.
- If `wgs` streams well on the judge: a c12 variant (decoder warps writing the SW128 A tile) for qkv m32 and o m32, which would also open the m16 shapes.
- Not worth it: hand-written m256 GEMMs; any more router work.

## 5. Results of v24 / v25 and status (parked)

Judge prepare-time numbers (µs, new kernel vs best v22 candidate):

| shape | bf16 wgs (v24diag) | c12 wgq (v25diag) | v22 best |
|---|---|---|---|
| qkv m8 | 11.71 | 13.76 | 9.38-9.63 |
| qkv m16 | 11.78 | 13.86 | 8.86-9.02 |
| qkv m32 | 12.03 | 14.34 | 11.26-11.74 |
| o m8 | 4.96 | 6.78 | 4.22-4.32 |
| o m16 | 5.02 | 7.01 | 4.45-4.67 |
| o m32 | 5.22-5.38 | - | v12 5.31-5.66 |
| pooler m32 | 18.72 | 21.73 | 15.42-15.87 |
| unpooler m32 | 19.33 | 21.25 | 15.42-15.74 |

Judge scores: v24diag 1.4527, v25diag 1.4379, both noise against v22's 1.459. v22 stays.

- bf16 wgs does not depend on m (qkv 11.7-12.0 µs from m8 to m32) and streams BF16 at about 3.1 TB/s. The c12 GEMV reaches about 3.0 TB/s on 0.75x the bytes and wins everywhere except o m32, where the margin is marginal.
- c12 wgq is 2-4 µs slower than bf16 wgs even though it reads 25% fewer bytes, so it is decode-bound. Four decoder warps per CTA give one warp per scheduler and no latency hiding. Per 16x64 unit: 3 lds.v4, 4 x 9-op decode, escape patch, 16 swizzled sts32, which is roughly 150+ cycles of dependent latency, and each warp handles 2 units per chunk.
- Best remaining idea: wgq with 8-12 decoder warps (or decoders double-buffered across 2 units). If that made it purely stream-bound at the wgs rate, the estimate is qkv m32 about 9.5 µs (-1.8 µs vs 11.26) and o m32 about 4.3 µs (-1.0 µs). That is about +0.75% geomean, all on decode 32. m8/m16 would not win: the GEMV is already at the c12 stream rate there. **Below the 1% bar, so not proposed.**
- No other lever found above 1%. m128/256 are at cuBLAS (about 72% of tensor peak at m256), the router is done, and the d1/v1 small shapes sit 1.5-2.9 µs above their floors from per-kernel cold-start latency, which SGLang pays as well. The per-call judge offset (about 1.3-1.5 µs over prepare time) is the same for one-kernel SGLang calls, so it is not launch count. v22 already issues exactly one kernel per call on all 30 shapes.

## 6. Against the new leader (1.637, 12% ahead of v22's 1.459)

### Precision: no FP8 route

TASK.md, Rules:
> **Weights.** ... don't store them at a different precision: ... BF16 weights stay BF16, or smaller, losslessly. ... The router's FP32 weights may run as a BF16 hi + lo pair with FP32 accumulation.
> **Activations.** BF16 wherever SGLang's kernel keeps BF16.

Probe rows (x has a single power-of-two entry) must reproduce weight columns within 0.1%, so any lossy weight format fails outright. FP8 x is also out (activations stay BF16). The m128/m256 tensor floor is therefore the BF16 one (989 TFLOP/s dense). Beating cuBLAS there has to come from bytes (lossless compression) and from occupancy and tiling, not from precision.

### Lossless compression is already near the limit

Exponent entropy of the generator's weights (all five matrices): 2.54 bits. That gives an ideal of (8 raw sign+mantissa bits + 2.54) / 16 = **0.659 of BF16**. Our c12 is 0.75 (3-bit offset over a 16-exponent window, 99.993% coverage). A fixed 11-bit code (8-exponent window) covers only 98.65%, so 1.35% of values become escapes: about 14 per 16x64 unit, against 0.07 today. A Huffman/DFloat11-style code reaches about 0.67, but needs table decode at more than 3.9 TB/s of output. Worth at most 0.75 to 0.67 = -11% of bytes on the stream-bound shapes, and it would have to come with a decoder at least as fast as today's 9-op one. Not the first lever.

### What 1.637 implies (first-principles model, scratch scen2.py / scen3.py)

Model: m <= 32 non-router shapes at c12 bytes / 3.9 TB/s + 1.5 µs; router at its FP32 bytes / 3.9 TB/s + 1.5 µs (+1 µs at m >= 128); m128/m256 at max(c12 bytes / 3.9 TB/s, 2mnK at 75% of 989 TF) + 2 µs. The 1.5 µs fixed cost matches our best shapes (pooler m1: 13.95 µs = 50.3 MB at 4.0 TB/s + 1.5).

| scenario | gm | d1 | d16 | d32 | v1 | v16 | v32 |
|---|---|---|---|---|---|---|---|
| ours v22 | 1.459 | 1.62 | 1.56 | 1.28 | 1.96 | 1.26 | 1.21 |
| m <= 32 shapes at c12 floor + 1.5 | 1.556 | 1.72 | 1.62 | 1.57 | 2.14 | 1.26 | 1.21 |
| + router at its floor | 1.599 | 1.74 | 1.65 | 1.64 | 2.19 | 1.29 | 1.25 |
| + m128 c12 GEMM | 1.660 | 1.74 | 1.65 | 1.64 | 2.19 | 1.62 | 1.25 |
| + m256 c12 GEMM | 1.673 | 1.74 | 1.65 | 1.64 | 2.19 | 1.62 | 1.32 |
| everything, a 0.70-ratio code | 1.733 | | | | | | |

Single levers from v22 (everything else unchanged):

| lever | gm |
|---|---|
| qkv m32 13.67 to 9.3 | 1.478 (+1.3%) |
| pooler + unpooler m32 17.0/16.6 to 14.6 | 1.478 (+1.3%) |
| o m32 6.62 to 4.7 | 1.467 (+0.5%) |
| all four m32 shapes at the floor | 1.509 (+3.4%) |
| qkv m128 15.35 to 10 | 1.479 (+1.4%) |
| pooler/unpooler m128 19.6/18.7 to 15 | 1.491 (+2.2%) |
| router at its floor (all m) | 1.497 (+2.6%) |
| qkv m256 17.7 to 12.5 | 1.475 (+1.1%) |

**1.637 is what a submission gets by running every shape within about 1.5 µs of its c12 / FP32 streaming floor and beating cuBLAS on m128 with compressed weights.** The leader must be winning mainly on:
- d32 (we are 1.28; the model gives 1.64),
- v16 (we are 1.26; 1.62 with an m128 compressed GEMM),
- the router everywhere (each router shape is 1-5 µs above 1.5 µs + bytes),
- less so v32, which is tensor-bound at m256: about 1.32 at most.

Nothing needs FP8 or an exploit.

### Why our m32 and m128 kernels miss, and the common fix

- The c12 GEMV (`gv`, mma.sync, decoded A in registers) streams at 3.5-4.0 TB/s up to m16. At m32, each warp does 16 HMMA and loads 4 KB of x per 1.5 KB weight unit from L1 (x re-read N/16 times), so it falls to 2.3-3.3 TB/s.
- The wgmma kernels lost for separate reasons:
  - `wgs` (bf16) streams at only 3.1 TB/s on 1.33x the bytes.
  - `wgq` (c12) decodes in 4 separate warps through smem and is latency-bound.
  - `wgd`/`wgc` at m128: TMA ring with 128 B-row boxes and a single producer.
- Fix: **wgmma RS**. Every consumer warp keeps the gv weight path (register-direct c12 loads, 9-op decode into mma-order A registers: exactly the wgmma RS A-fragment layout for its 16 rows). The tokens are the wgmma N side (n32 at m32, n128 at m128), from a TMA x tile in smem that the whole CTA shares. One `wgmma m64nNk16` per k16 step per warpgroup replaces 4 (m32) or 16 (m128) HMMAs per warp. x is read once per CTA per chunk instead of once per 16-row tile, and weights never touch smem. This also avoids wgq's decoder hand-off.

### Proposed builds (not started; estimates are judge µs)

1. **`gvw` m32: wgmma-RS c12 GEMV for qkv / pooler / unpooler / o at m32.** CTA = NWG x 64 weight rows (each warp 16 rows, gv-style register ring of c12 units), x tile 32 tokens x 64 k via TMA (one 4 KB stage per chunk, ring of 4-6), split-K by the wgr workspace + counter where N / BM < 132.
   - Per-SM work at qkv m32: 140 units x (3 loads + 36 decode ops + 4 wgmma issues) is about 1.4 µs of issue. Stream: 28.3 MB at 3.6-3.9 TB/s.
   - Estimate: qkv 9.5 µs (now 13.67), pooler/unpooler 14.8 (17.0/16.6), o 4.8 (6.62). **+3.0-3.4% gm.**
   - Risk: the wgmma RS operand layout (A registers must stay untouched until wgmma.wait; ptxas C7515-style serialization) and register pressure (ring + 16 accumulators at n32). Medium.
2. **`gvw` m128 (same kernel, TN = 128, accumulators 64 per thread): pooler / unpooler / qkv / o at m128.** Tensor time at 75% peak: pooler/unpooler 11.7 µs; c12 stream 12.9 µs.
   - Estimate: pooler/unpooler about 15 µs (now 19.6/18.7), qkv about 10 (15.35, split-K over S = 7 with 18 row blocks of 128), o about 5.5 (6.65). **+3.0-3.6% gm.**
   - Risk: higher. It needs 75% of tensor peak from RS wgmma with decode in the same warps (decode for 508K values per SM is about 3.4 µs of issue), and three previous GEMM attempts lost to nvjet.

Together that is about +6-7% (to about 1.55-1.56). The router to its floor (+2.6%) is the next lever after these. The c12 hi + lo router lost on the H200 in v14 (6-8 vs 4.7 µs) as a mma.sync GEMV; the same wgmma-RS path would be the retry.

## 7. RS-wgmma 12-bit GEMM (v26-v30diag): parked at v22

Judge diag runs: v26/v27 (after the sm_90 build fix), v28diag, v29diag and v30diag all scored 1.443-1.458, which is noise against v22's 1.459-1.463. `wv` (12-bit weights decoded straight into RS-wgmma A registers, x tile shared by the CTA, split-K via workspace + counter) is correct but slower than the paths it competes with:
- **m128** (v30diag span, µs): qkv 25.4 vs 15.35 (cuBLAS), pooler 31.9 vs 19.6, unpooler 32.0 vs 18.7, o 10.5 vs 6.65. That is 1.6-1.7x off cuBLAS and about 2.5-3x off the 12-bit floors (8.4 / 13.6 / 13.6 / 4.7).
- **m32 / router:** at best marginal. qkv m32 11.2 vs 11.5. Router m32 4.70 and m256 7.14 were picked.
- **Consumer cycles:** spread over stage wait, decode, wgmma issue (warpgroup convergence), wgmma wait and the split-K reduction. Deep decode-ahead (4 A sets, wait_group 3) did not change the picture, and no single phase dominates.
- **Stragglers:** CTA start spread is under 0.15 µs. Shapes with split-K end with an 8-9.5 µs end spread that sits in the counter + reduction tail; the same kernel without split-K ends within 1.5-2.2 µs. Any further attempt must avoid the global split-K reduction at these sizes (cluster DSMEM push, or a schedule with no split).

Status: bf16_proj parked at v22. The analysis in section 6 stands: 1.637 implies about 50% of each shape's slack recovered, mostly at m128 (memory-bound, arithmetic intensity 112-122 flop/B, below H200's 206 ridge) and m32. Our hand-written GEMMs do not reach cuBLAS-level streaming at m128. The tools stay for a future attempt: bench/wb_compile_chk.py (sm_90 build gate + stub prepare), bench/wv_esc_chk.py, and the WP/WC profiles in submission_v30diag.py.
