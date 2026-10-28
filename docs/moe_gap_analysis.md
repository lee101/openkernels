# muse_moe_w4a16: how does anyone reach 2.33?

Ours: v34, judge 1.5745. Leader: 2.33, which is 1.48x faster than us in geomean.

## 1. Shapes, routing, byte floors

Sources: `muse_moe_w4a16/reference.py`, `notes/muse_moe_w4a16.json`, `_bytes.py`.

- H = I = 4096, 256 experts, top-8, ep8 (local experts 0-31). One expert is W13 [8192, 4096] + W2 [4096, 4096] = 50.3M NVFP4 values, which is 25.2 MB of codes + 3.1 MB of e4m3 scales = **28.3 MB**.
- `generate_input`: `scores = rand(bs, 256)` plus a per-expert popularity term `0.03 * t2` (Student-t with 2 degrees of freedom, so heavy-tailed), then `topk(8)`. `rank="busiest"` rotates ids by whole GPUs so that the 32 local experts hold the most pairs. Each token's 8 draft positions are routed independently, with no correlation inside a verify block.
- Judge seeds: random 64-bit values, new on every run and every repetition (judge record: `seed 6380544118842228819`, ...). The `benchmarks` seeds 401-406 in the metadata are not what gets timed. Every one of the 100 repetitions gets a fresh routing.
- The TASK floor uses `ROUTING_SHARE = 0.75` times the mean number of distinct experts, so it understates the real mean-routing byte floor by 1.33x.

Monte Carlo over 400 seeds of the real generator (scratch `sim.py`):

| load | M | distinct local experts mean [p10 p50 p90] | local pairs | max tok/expert p50 [p90] | weight MB | t@3.9 TB/s | t@4.8 | SGLang | ours |
|---|---|---|---|---|---|---|---|---|---|
| decode1 | 1 | 2.58 [2 2 3] | 2.6 | 1 | 73 | 18.7 | 15.2 | 80.7 | 50.4 |
| decode16 | 16 | 12.1 [9 12 15] | 25.8 | 6 [16] | 343 | 87.9 | 71.4 | 187.3 | 126.2 |
| decode32 | 32 | 16.1 [12 16 20] | 49.6 | 11 [31] | 457 | 117.1 | 95.1 | 214.3 | 175.0 |
| verify1 | 8 | 8.4 [6 8 11] | 13.9 | 4 [8] | 237 | 60.7 | 49.3 | 150.9 | 101.4 |
| verify16 | 128 | 22.6 [19 23 26] | 195 | 47 [128] | 640 | 164.1 | 133.4 | 469.0 | 259.3 |
| verify32 | 256 | 24.7 [22 25 28] | 388 | 99 [255] | 700 | 179.5 | 145.8 | 633.5 | 340.5 |

Activation bytes are below 4 MB and are ignored. The weight codes are uniform random (`randint(0,256)`), so the bytes listed are the floor except for the scales (see 4c).

Geomean of SGLang time over time (using our per-load numbers, ours comes out at 1.563, not the judge's 1.5745, because the judge uses paired-ratio medians):

| profile | d1 | d16 | d32 | v1 | v16 | v32 | gm |
|---|---|---|---|---|---|---|---|
| ours | 50.4 | 126.2 | 175.0 | 101.4 | 259.3 | 340.5 | 1.56 |
| weight floor @3.9 TB/s | 18.7 | 87.9 | 117.1 | 60.7 | 164.1 | 179.5 | **2.74** |
| weight floor @4.8 TB/s | 15.2 | 71.4 | 95.1 | 49.3 | 133.4 | 145.8 | 3.37 |
| floor@3.9 / 0.85 | 22.1 | 103.4 | 137.7 | 71.4 | 193.1 | 211.1 | 2.33 |
| bytes/3.7 TB/s + 5 us | 24.8 | 97.6 | 128.4 | 69.0 | 178.0 | 194.2 | 2.41 |
| ours uniformly / 1.48 | 34.1 | 85.3 | 118.3 | 68.5 | 175.2 | 230.1 | 2.31 |

- A 2.33 score means running every load at about 85% of the 3.9 TB/s weight-streaming roofline (3.3 TB/s effective, end to end), or 69% of the 4.8 TB/s peak. That is physically plausible, so the score needs no exploit.
- A uniform 1.48x over us is not possible. d16 would need 85 µs, below its 3.9 TB/s floor (88 µs), and d32 would need 118 µs, at its floor (117 µs; our NOCOMP skeleton alone takes 137 µs). The leader must therefore be much further ahead of us where we are furthest from the floor. Our time divided by the floor is: **d1 2.7x**, v32 1.9x, v1 1.67x, v16 1.58x, d32 1.49x, d16 1.44x.

## 2. Timing loop (judge)

The judge's `harness/eval.py`, `runner.py` and `worker.py` are not in the repo; we only have their hashes in `notes/moe_check*.json`. The evidence below is TASK.md plus the judge's result records.

TASK.md, "How a run is timed":
> Each shape is captured as a CUDA graph and timed by replaying it, in the same run and on the same GPU as SGLang's kernel, your replays and SGLang's taking turns on the same fresh inputs.
> Every timed replay starts with a cold L2: the judge reads a buffer larger than L2 just before it.
> The judge subtracts a graph replay's fixed cost (about 4 µs) from both sides.

and under Rules:
> **Fresh inputs.** Every call gets new inputs; don't cache anything derived from `data`.

`reference.py`: `MAX_REPEATS = 100  # Timed repetitions per shape; each is checked.`

Judge record (`notes/moe_check11.json`, decode1):
`"replay_overhead": {"gate_ns": 299136, "one_ns": 4992, "overhead_ns": 3904, "parent_one_ns": 304064, ...}`, `"runs": 100`, setup `"baseline_copies": 4, "copies": 1, "prepare_pad": 7215104`.

- **L2 flush:** yes, before every timed replay. The `gate` is a read of a large buffer that takes about 299 µs, which is roughly 1 GB, far more than the 50 MB L2.
- **Inputs:** the weights (the prepare state) persist. hidden_states, topk_ids and topk_weights are refilled for every repetition, and SGLang and we get the same values in each pair. Baseline per-repetition times swing widely (d16: best 139 µs, median 190, worst 264) because the routing changes on every repetition.
- **L2 residency:** not possible. The flush runs before every replay, the set of hit experts changes on every repetition, and even decode1 touches 73 MB, which is more than L2. `copies` / `baseline_copies` / `prepare_pad` suggest the judge also rotates or pads state copies where memory allows. We get 1 copy because our state is large.
- **CUDA graphs:** yes, capture once and replay. `parent_*` and `outside_bursts_ns` show the judge also times work outside the graph.

## 3. What is legal

`checks.py` is numerics only: relative L2 at most 1.5e-2 over the whole output, each row at most 5x that (against the row norm, floored at 0.1x the mean row norm), and no NaN or Inf. It says nothing about caching. The rules come from TASK.md, AGENTS.md and the static checks.

Legal:
- Any repack, reorder or interleave of weights in `prepare()`. It runs untimed, for up to 300 s.
- **Lossless compression** of the weights ("compress them losslessly if you like"; in bf16_proj, c12 took us from 1.146 to 1.366).
- Dequantizing inside the kernel. Branching on shape or config. Per-M autotuning in prepare (we already do this).
- Any device-side, data-dependent control flow inside the kernel.

Not legal:
- Caching or memoizing anything derived from `data` (topk, x, outputs) across calls; AGENTS.md: "cache or replay outputs" gets you removed.
- Storing weights at a different precision. For example, BF16-expanded weights are out, because "NVFP4 stays NVFP4, or smaller losslessly".
- Activations below BF16 where SGLang keeps BF16.
- Host syncs or allocation inside kernel_fn.
- Patching timing, reading harness frames or I/O, editing loaded modules, importing reference, starting processes, network access (static checks).
- Final review reruns on held-out and nearby shapes, so any trick tuned to the generator gets a closer look.

Note: a hi+lo FP8 split of x (to use FP8 wgmma) is grey at best. It is also blocked on its own terms: an e2m1 x e4m3 block-scale product needs up to 6 significant bits, so the scaled weight is not exact in FP8, and the 16-wide block scale cannot be applied inside a k=32 FP8 wgmma. Not a path.

## 4. Likely mechanisms (ranked)

### (a) wgmma RS large-M path: weights in registers, tokens in smem (M >= 8; biggest gm lever)

This is the Humming design (`docs/upstream_ideas.md` item 1). Dequant warpgroups produce the A fragment (64 W rows x k16, BF16, in registers; the same per-warp layout as the mma.sync A fragment). B is the expert's gathered x rows, loaded into smem once per CTA per K chunk, with N = ceil8(tokens) capped at 128 per wgmma.

What changes against our mma.sync ring:
- **Tensor issue:** one wgmma per 64x16 weights for all of the expert's tokens, instead of 4 warps x N/8 HMMAs.
- **B traffic:** about 4x less smem/L1 traffic. For v32 that is about 0.77 GB of smem reads, versus about 3.1 GB of B fragments with m16n8k16.
- **Hot experts:** experts with 47 / 99 / 128-255 tokens (p50/p90 at v16/v32) cost about the same issue per weight as cold ones. Today they cost 2.5x per granule (v36d: nt4 13.9 vs nt1 6.3 µs per granule).
- **Clock:** fewer instructions means less power, which recovers some of the 1979 to 1808 MHz clock drop.

Consumer cost then becomes dequant only: 24.7 experts x 50.3M values at about 17 instructions per 8 values comes to about 87 µs at IPC 1, which is below v32's 180 µs stream time. So v16 and v32 become stream-bound.

Estimates: v16 about 185-195 µs, v32 about 200-210 µs, v1 about 70-75 µs, d16 about 105 µs, d32 about 135-145 µs. The floor under d32 is the skeleton, NOCOMP at 137 µs, so d32 also needs the prologue and tail trimmed.

### (b) decode1: a single-pass, latency-hidden GEMV (biggest single-load lever, 2.7x off the floor)

v34d FD M1 breakdown: prologue 3.1, W13 done 29.4 (49 MB, about 1.9 TB/s, consumer-latency-bound at IPC about 0.27), W2 done 39.2, barrier 41.5, exchange 49.2, end 51.2.

Target: about 25 µs, which is 73 MB at 3.9 TB/s plus about 5 µs of prologue and tail. Ingredients:
- **Prefetch W2 into L2 during W13.** The hit experts' W2 is 24.5 MB (28 MB at p90), which fits in the 50 MB L2. Use `cp.async.bulk.prefetch.L2` for the contiguous slabs with an evict_last policy, and evict_first for the W13 stream. HBM then never idles across the W13 to W2 dependency, and phase B runs from L2.
- **Raise consumer IPC at M=1.** Options: split K so each warp keeps its x slice in registers (this removes the 5 µs of activation ldg); deeper unroll / software pipelining of LDS, dequant and HMMA; fewer stages per mbarrier. The issue budget is only about 8 µs at IPC 1, so the problem is latency, not instruction count.
- **Combine in one stage.** Replace the barrier plus column-split exchange (about 10 µs) with a last-arriver RMSNorm on the 2-3 rows. LASTA got the exchange to 45.8, so the remaining tail is in the finisher passes.

Alone this is worth gm 1.56 to 1.76. Combined with (a) it reaches **gm about 2.30** (d1 25, d16 105, d32 135, v1 72, v16 190, v32 205).

### (c) Lossless scale compression (about 4-6% fewer bytes on every load)

Codes are incompressible: uniform, 4.0 bits, or 3.875 bits if ±0 are merged, which is not worth exploiting. The e4m3 block scales (11.1% of bytes) are log-normal with σ = 0.4 octave. Measured on the generator: **32 distinct codes (95..126), entropy 3.71 bits**, top-16 coverage 98.7%.

Options:
- **5-bit fixed width** (code minus a per-expert base), verified lossless in prepare with a raw-8-bit fallback per expert: saves 4.2% of bytes.
- **4-bit with a per-16/32-group base:** the range fits in 15 steps for 95% / 85% of groups, and the rest need an escape. That is about 5.5%, at the cost of complexity.

On top of (a) and (b): gm 2.30 to about 2.40.

## Conclusion

2.33 is reachable without exploits. It corresponds to about 85% of the 3.9 TB/s weight roofline on every load, and the leader most plausibly has:
1. A near-roofline decode1 (about 25 µs vs our 50).
2. A wgmma-RS (or otherwise B-reuse-heavy) large-M path that makes v16/v32/v1 stream-bound (about 190 / 205 / 72 µs vs our 259 / 341 / 101).
3. Possibly lossless scale compression for the last few percent.

d16 and d32 cannot carry the gap: they are already within 1.44x and 1.49x of their floors.

What to build, in order:
1. A wgmma RS expert GEMM for M >= 8, with gathered x in smem, N per expert up to 128, and dequant warpgroups feeding A registers. Validate the layouts with bench/wgt.py first.
2. A decode1 path: W2 L2 prefetch during W13, split-K x in registers, and a single last-arriver combine.
3. A 5-bit scale pack in prepare (lossless check plus fallback), shared by both paths.

## Instructions per weight word (v51 analysis)

v51d: gk consumers issue at ~0.67-0.71 of 4 slots/clk at M8/M16 (0.62 at M32) and 1.5-1.7 GHz; more warps (NW 20) only lowered the clock (1497 MHz), more accumulator chains (ACCN 4) tied, r1 with 16 consumer warps lost. The consumers are near the practical issue ceiling, so time per byte = instructions per byte / (issue rate x clock).

One word = 32 bits of codes = 8 weights (4.5 bytes with scales). sm_90a SASS of the gk piece loop (two pieces per slot, per 32 words): LOP3 200, IMAD.SHL 114, SHF 55, HMUL2 128, HMMA 32, LDS 31, IMAD 29, LEA 22, PRMT 16, IADD3 11, BRA 11, VIADD 9, ISETP 8, uniform/misc 12 = 22.0 per word:

| part | per word | what |
|---|---|---|
| bit placement | 11 | 6 LOP3 + 5 shifts (3.6 as IMAD.SHL on the FMA pipe, 1.4 SHF): a0 = q & M (1), a2 = (q << 6) & M (2), a1 / a3 = ((q << s1) & M1) \| ((q >> or << s2) & M2) (4 each) |
| scale apply | 4 | HMUL2 (bf16x2 fma with -0): applies the block scale and fixes the e2m1 exponent bias, subnormal codes included |
| mma | 1 | HMMA m16n8k16 |
| scale prep | 1 | per 64-K chunk and row pair: PRMT (replicate the e4m3 byte) + LEA (<< 4, + bias) for sA and sB = 4 per 4 words |
| smem loads | 1.0 | codes 0.25, scales 0.03, activation B 0.5 (gk; g1 keeps B in registers), rest addressing |
| loop / slot / issue / reduction | ~3 | IMAD / LEA / IADD3 / ISETP / BRA addressing, mbarrier wait + TMA issue per slot, group reduction per tile slice |

What a lossless repack could remove:
- Bit placement: a brute force over every per-half layout (any shift or rotation per sign / magnitude field, bf16 targets {15, 8:6} and fp16 targets {15, 11:9}) gives 11 integer ops per word as the minimum: three single-shift fields fit a 16-bit half ({0, 3, 6}), the fourth then has no 3 contiguous free bits, so two fields need a two-part merge. Storing signs apart from magnitudes costs the same merge. Byte-permute tables (20-24) and LUT GEMV (>= 2 per weight) are worse.
- HMUL2: needed for every block. Only 14.5% of the generator's block scales are exact powers of two (mantissa 0), and even for those an integer exponent add is wrong for the subnormal codes (0, +-0.5), so the multiply cannot be dropped; the per-row / per-expert global scale is already applied once in the epilogue (0 per word).
- Scale prep: the e4m3 byte must become a replicated bf16 pair with the scale's exponent moved by 4 bits; byte permutes move whole bytes, so a shift (or LEA) per register remains. Pre-expanded 16-bit scales would remove it (-1 per word) but add 11% of bytes. Pairing two blocks into one A register (K permutation) turns 4 ops per 4 words into ~4 per 8 at best (-0.5), with a permuted activation staging.
- Loop / slot / issue overhead (~3): halves with two-piece slots (measured: gk PPS 2 at M1 3.03 TB/s at 1305 MHz vs PPS 1 3.12 at 1475 MHz: no gain), 4-piece slots need 2x ring smem (only M <= 2 fits).
- Activation B loads (0.5, gk): registers instead of smem need a 256-K slice per warp (16-way reductions per tile, the g1 shape), which g1 already is (no faster at M1).

Best layout estimate: 16 (core) + ~0.5 (paired scale prep) + ~0.8 (loads) + ~1.5 (overhead) = ~18.8 per word vs 22.0 now: -14.5%, below the 15% bar, and its largest part (slot overhead) already failed to convert to speed in PPS 2. Not built. The scale entropy is 3.71 bits (31 distinct values, top 16 cover 98.7%), codes are exactly 4.00 bits: a 4-bit scale index saves ~5.5% of bytes but costs ~0.5-1 instruction per word on an issue-bound consumer.

## Load-path floor and dequant alternatives (v54d / v55 notes)

v54d stream bench (no compute, cold L2, hit experts' W13 + W2): 1D cp.async.bulk units >= 16 KB reach 4.29-4.36 TB/s (d32, 538 MB: 123-125 us) and 4.39-4.43 (v16, 793 MB: 179-181 us) at any depth 3-8 and 1 or 2 CTAs/SM; 2D TMA boxes 16-64 KB 3.96-4.02 (about 8% less); a single producer thread with 2-4 KB units 1.0-2.0 (issue-bound on one thread: not the gk / g1 shape, where 16 warps each run their own ring). wx's NOCOMP (no dequant, no mma) is within ~4% of the 1D floor (M32 129 vs 124, M128 185 vs 179): its load path is fine; the gap is consumer issue / energy. v55d adds the 16-warp small-unit shape (16w: 2.3-9.2 KB units, 1-4 in flight per warp) to answer whether gk's load pattern itself loses bandwidth, and the wx consumers' SM clock.

Dequant alternatives:
- Block scale into the activations: impossible (the scale is per output row x 16-K block; activations are per K). Into the accumulator: needs the k16 steps aligned to blocks and 4 FFMA per mma per n-tile (N / 2 per wgmma thread), and the per-register multiply is still needed for the e2m1 exponent bias: placed e2m1 bits are only an affine image of the value for normal codes (the 0 / 0.5 codes break any OR-in exponent bias), and the 15 distinct values {0, +-0.5, ..., +-6} are not an affine image of any 4-bit index (span 24 half-steps > 16), so no magic-number integer trick either. HMUL2 x 4 per word (bias + block scale in one multiply) is the minimum.
- PRMT LUT: a 3-bit magnitude selects a byte of an 8-byte table in two registers; bf16 needs a high byte and a low byte per value (2 tables), then interleaving into bf16x2 (1 PRMT per register), sign bits merged (1 LOP3 per register) and the block scale still multiplied (HMUL2 per register): ~17-18 per 8 weights vs 15 now. Marlin's fp4 -> bf16 (mask, shift, merge with the sign: ~3 per register) lands at the same ~11-12 integer ops plus the scale multiply.
- Moving dequant to the producer warp(s): the same instructions issue from the same 4 schedulers; it only adds warps, and gk NW 20 showed more warps lowering the clock (1497 vs ~1650 MHz) under the power cap. Expected neutral unless the consumers are latency-bound rather than issue / power bound (v55d's wx clock line checks this for wx).

## What the 2.62 leader implies, and formulations below ~17 instructions per word (v55d notes)

v55d: 16-warp rings of 2.2-9 KB units stream at 4.21-4.35 TB/s (gk's load shape is fine); wx consumers run at 1755-1771 MHz (not power-throttled). Our consumers sit at ~75% of the stream rate everywhere at ~22 instructions per weight word.

Floors at 4.35 TB/s (d1 17, d16 79, d32 124, v1 54, v16 179, v32 161 us) against SGLang (79.5, 192.9, 205.6, 141.7, 476.8, 635.2) give a floor geomean of ~2.84. The leader's 2.62 is ~92% of that, i.e. roughly every load at ~92% of the stream floor: about d1 18.5, d16 86, d32 135, v1 59, v16 195, v32 175 us (we: 30.0, 118, 163, 81, 233, 273). Some compute-heavy load could be slower only if others sit at the floor; there is no slack for a 75% load.

Issue arithmetic at 1.77 GHz: 22 instructions per word on 4 schedulers is ~6.1 TB/s at 100% issue efficiency; we reach ~3.3-3.6, i.e. ~55-60% of issue capacity, not pipe-bound (ALU-pipe bound ~7 TB/s) and not memory-bound. More warps (gk NW 20, r1 c16) did not raise it.

Candidates for < ~17 per word:
- fp16 instead of bf16 MMA: the dequant cost is identical (the 11-op placement and the per-register multiply are the same for fp16 targets; brute force gives 11 for both), and converting activations away from bf16 conflicts with "Activations BF16 wherever SGLang keeps BF16" (Marlin keeps bf16). No gain.
- Producer-side dequant into smem + SS wgmma: total instructions do not drop (same 15 dequant + an STS per 2 words + proxy fence / barrier), and smem traffic becomes write 2 B + tensor read 2 B per weight = ~7x the HBM bytes: at 4.3 TB/s that is ~130 B/clk/SM, the whole smem bandwidth. Not viable.
- 2 CTAs/SM: same instruction count, more warps; equivalent to the NW 20 / c16 experiments (no gain).
- No lossless re-encoding, LUT (PRMT) or accumulator-scale formulation found below 15 integer + multiply ops per word (see the sections above).
So the remaining gap is issue efficiency (scheduling / dependency stalls at ~22 per word), not instruction count or bytes; closing it needs per-instruction stall data (ncu) we do not have on the judge GPU.

## Closing summary (MoE parked at v56, judge 1.9749)

**Final per-load numbers (v56d), µs.** The floor is the 4.35 TB/s 1D-bulk stream rate on each load's bytes.

| load | v56d | floor | % of floor | path |
|---|---|---|---|---|
| d1 | 28.8 | 17 | 59% | gk KS2 1 MAXM 8 |
| d16 | 108.5 | 79 | 73% | gk KS2 2 |
| d32 | 155.0 | 124 | 80% | r1 (gk51 M32 / wx NCW16 tie at about 156) |
| v1 | 73.9 | 54 | 73% | gk KS2 1 MAXM 8 |
| v16 | 218.0 | 179 | 82% | wx NCW16 UDESC |
| v32 | 258.1 | 161 (bench) / 200 (wx NOCOMP) | 62% / 78% | wx NCW16 (N8STEP) |

v56 contributed the 8-op e2m1 placement: about 3 fewer instructions per word on every path, +4.9% on the judge.

None of v57–v60 beat it:

| version | judge | change |
|---|---|---|
| v57 | 1.969 | gk producer warp: 22.6 -> 16.7 instructions per word, but slower in the tune |
| v58 | 1.954 | v51 gk incl. M32 shapes: ties r1 / wx at d32 |
| v59 | 1.969 | WB load variants |
| v60 | 1.973 | bf16 / bf16x2 scale copies: lose at M1 / M8 (36.0 / 39.6 and 69.3 / 79.2 vs 35.0 / 67.2) |

**Evidence that the consumer loop is issue-bound (WB, H100, no TMA / mbarrier, ns per wgmma per SM; r1w4 = 16 warps, r2w2 = RPW 2 on 8 warps).**
- A wgmma alone costs 7.5-8.7 ns at N 8-16, 9.2-10.5 at N 32 and 18.2-19.2 at N 64. At N 64 the tensor pipe alone runs at 4.0-4.2 TB/s equivalent, which is the stream rate: hot experts are tensor-bound at BF16 peak. Padding is not the cause; N8 rounding falls almost entirely on experts with N <= 32, where wgmma cost is flat.
- Register dequant + wgmma (D, after fixing the v57 / v58 bug where ptxas computed one dequant per 4 chunks) against the same loop with the wx shared-memory loads + sc2 (L):

| N | D r1w4 / r2w2 | L r1w4 / r2w2 |
|---|---|---|
| 16 | - / 13.8 | 15.6 / 16.2 |
| 32 | 16.2 / 16.8 | 17.4 / 17.7 |
| 64 | 21.9 / 21.0 | 22.4 / 24.0 |

  Loads add only 1-2 ns over D.
- One-stage prefetch (P, Q) and per-group hoisting (H) gain at most 0.6 ns. bf16x2 scale pairs (S) gain about 1.2 ns at N 16 and nothing at N 64, and as separate copies (+11% / +33% bytes) they lose in the real gk.
- L runs about 18 instructions per word at about 0.68 IPC per scheduler. The real wx loop is 19.9 instructions per word, so the consumers sit at the dequant + mma issue bound plus about 30% dependency stalls.
- Instruction cuts that left the critical path untouched did not pay: the gk producer warp removed 26% of the loop instructions and lost to its polling cost.

**Levers left for a future attempt.**
1. **Free registers so a load prefetch survives in the real wx.**
   - What fails now: every register prefetch (next chunk at chunk start, after the commit, or the whole slot at its top) makes ptxas serialize the wgmmas (C7512) or spill. NCW16 is at 96 of 120 registers with a 16 B spill; RPW2 is at 167.
   - Fold the producer warp into a consumer lane: 512 threads raises the cap to 128.
   - Or shrink the A double buffer to k32 groups.
   - WB says the payoff is small (<= 0.6 ns per wgmma); it matters only together with lever 2.
2. **RPW2 at 3 warps per scheduler.**
   - The shape: 3 consumer warpgroups x RPW2 = 384-row units, plus a 4-warp producer warpgroup that gives registers to the consumers with setmaxnreg (consumers up to 160).
   - Needs: RPW2 trimmed from 167 to 160 registers, and ragged unit tails (W13 64 = 21 x 3 + 1 blocks, W2 32 = 10 x 3 + 2).
   - Why: RPW2's per-wgmma descriptor amortization (16.9 vs 19.9 instructions per word) currently loses to NCW16's latency hiding (2 vs 4 warps per scheduler).
3. **Per-instruction stall data (ncu source view) on a host that allows GPU performance counters.** RunPod H100 pods return ERR_NVGPUCTRPERM even as root. scratchpad/podprof (pro.py + run.sh) builds only the picked modules and profiles wx M128 / M256 / M32, gk M16 / M1, g1 M1 and r1 M32 in about 10 minutes.
4. **d1 (59% of floor), the weakest load relative to its floor.** gk / g1 at M1 are tail- and barrier-bound (8 experts, 2 grid barriers, clock-capped at 1.1-1.5 GHz), not dequant-bound; a single-pass, latency-hidden GEMV (section 4b) remains the structural option.
