# muse_prefill_layer: what does 1.70 take?

Ours: pf12, judge 1.358 (pf13 1.353, flat). Leader: 1.70, which is 1.26x faster than us in geomean.

## 1. Loads, shapes, routing

Sources: `muse_prefill_layer/TASK.md`, `reference.py` (`speed_of_light_ns`, `weights_fields`, `_router`), judge profiles
`muse_prefill_layer/profiles/submission_pf13.194c59208aa144d1.{txt,json,log}`, scratch scripts `pfx/pfgap.py` (this table).

- There are three loads, and each counts once in the geomean:
  - turn 1: 46 x sliding + 16 x full, B1 L753, ctx 80K.
  - turn 4: the same with B4 (3012 tokens).
  - first turn: 16 x (ctx 0 full) + 138 x (ctx 32K sliding) + 16 x (ctx 32K full) + 16 x (ctx 64K full), L16384.
- **Routing:** the judge picks a router seed so that each load has its target `router_load` / `router_distinct` (task.yml). Backing the
  load out of the TASK.md floors (every shape is flops-bound there) gives **~1.8 local routes/token** on the busiest 32-expert range at every
  load: T753 1314-1359, T3012 5330-5543, 16K 28.9-30.5K.
- Running the reference router with seeds picked for that load (on reference `h`) gives 12-23 distinct local experts with extreme skew:
  - T753: e.g. [400, 314, 241, 174, 105, 57, 19, 15, 9, ...] or [635, 184, 181, 154, 69, ...].
  - T3012: [1571, 1248, 749, 490, 488, 280, 255, 120, ...].
  - 16K: [10543, 8504, 3315, 2228, 1774, 1545, 852, 661, 306, ...].
  - The top expert takes 30-80% of the tokens. These profiles are `MOE_PROF` in `glue_pf2.py`.
- **Floors** below use 3.9 TB/s for bytes, 650 TFLOP/s BF16 for GEMM math, and 640 TFLOP/s for FA3 FP8 hdim 64 (what FA3 measures; its exp
  throughput caps it near ~990).
  - MoE bytes: 28.3 MB per touched expert (NVFP4 codes + e4m3 scales), plus activations.
  - MoE FLOPs: routes x 3 x 4096² x 2.
  - Dense: qkv / o / pooler / unpooler / router [hi|lo].
  - Residual: resa + resb, 22 B per element of T x 8192.

## 2. Per component, per shape (µs; ours = pf13 judge profile scaled to the timed median)

SGLang FA3 is our prepare() timing of FA3 on the paged cache (the kernel the baseline calls). SGLang MoE is our prepare() timing of SGLang's
Marlin path on the scored count profiles (T753 749, T3012 2221). At 16K SGLang's MoE is not separable. SGLang dense GEMMs are the same
cuBLAS calls as ours, except where our cuBLASLt picks win 3-10%.

| shape | w | SGLang | ours | attn: ours / SGL / floor | MoE: ours / SGL / floor | dense: ours / floor@650 | resid: ours / floor | other |
|---|---|---|---|---|---|---|---|---|
| turn 1: B1 ctx80K sliding L753 | 46 | 999 | 810 | 42 / 35 / 10 | 514 / 749 / 210 | 204 / 228 | 44 / 35 | 6 |
| turn 1: B1 ctx80K full L753 | 16 | 1405 | 1216 | 408 / 441 / 387 | 494 / 749 / 210 | 265 / 228 | 43 / 35 | 7 |
| turn 4: B4 ctx80K sliding L753 | 46 | 3302 | 2409 | 63 / 66 / 39 | 1319 / 2221 / 839 | 750 / 913 | 162 / 139 | 115 |
| turn 4: B4 ctx80K full L753 | 16 | 4894 | 3920 | 1511 / 1638 / 1549 | 1349 / 2221 / 839 | 746 / 913 | 160 / 139 | 154 |
| first: B1 ctx0 full L16384 | 16 | 17799 | 11447 | 843 / ~843 / 859 | 5637 / ? / 4646 | 3994 / 4969 | 808 / 757 | 166 |
| first: B1 ctx32K sliding L16384 | 138 | 16946 | 10556 | 280 / 307 / 215 | 5289 / ? / 4646 | 4002 / 4969 | 819 / 757 | 166 |
| first: B1 ctx32K full L16384 | 16 | 20941 | 14578 | 4255 / 4465 / 4295 | 5283 / ? / 4646 | 4055 / 4969 | 814 / 757 | 170 |
| first: B1 ctx64K full L16384 | 16 | 24578 | 18246 | 7696 / ~7700 / 7731 | 5520 / ? / 4646 | 4066 / 4969 | 826 / 757 | 138 |

How the components break down:
- **MoE**:
  - T753: ring (`moe_kernel`) 521 µs.
  - T3012: hybrid; ring for the light experts ~395-505, grouped GEMMs ~620-674, deq3 ~210-240, glue.
  - 16K: big path; grouped GEMMs 4.2-4.7 ms, deq3 0.52-0.69 ms, swiglu/gnorm/comb ~0.45 ms.
- **attn**: FA3 plus the combine, qkprep, the gather and the epilogue.
- **dense**: cuBLAS/nvjet already runs at ~750-830 TFLOP/s (above the 650 "realistic" rate, so ours sits under that column's floor).
- **other**: profile lines not listed (route2, rmsg2, rlist, msplit, small kernels) and stream overlap.

Per load (ms, weighted; components as ours (floor)):

| load | SGLang | ours | speedup | floor sum | ours needed for 1.70 | MoE / attn / dense / resid / other |
|---|---|---|---|---|---|---|
| turn 1 | 68.4 | 56.7 | 1.206 | 36.0 (1.90x) | 40.2 (-16.5, -29%) | 31.5 (13.0) / 8.4 (6.7) / 13.6 (14.2) / 2.7 (2.2) / 0.4 |
| turn 4 | 230.2 | 173.5 | 1.326 | 143.9 (1.60x) | 135.4 (-38.1, -22%) | 82.3 (52.0) / 27.1 (26.6) / 46.4 (56.6) / 10.0 (8.6) / 7.8 |
| first turn | 3351.7 | 2165.0 | 1.548 | 2165 (1.55x) | 1971.6 (-193.5, -9%) | 992.9 (864.2) / 243.3 (235.8) / 746.1 (924.2) / 152.1 (140.8) / 30.5 |

Where we are against the floors:
- **Turn 1** is 56% MoE, and the MoE runs 2.4x its floor (ring 514 vs 210). Everything else on turn 1 is within 1.0-1.3x of its floor.
- **Turn 4** is 47% MoE, at 1.6x its floor (1319 vs 839). The rest is at its floor.
- **First turn** is at the "realistic" floor sum already. 1.70 there needs better-than-650-TF GEMMs: dense already runs ~800, and
  the MoE grouped GEMM runs ~640 plus 0.6 ms of dequant.

## 3. What has to shrink to reach 1.70

Scenarios below are computed from the table above (`pfgap.py`; each row includes the previous ones unless marked):

| scenario | turn 1 | turn 4 | first turn | geomean |
|---|---|---|---|---|
| now (pf13) | 1.21 | 1.33 | 1.55 | 1.353 |
| turn MoE -> 1.35x its floor (T753 ~285, T3012 ~1130 µs) | 1.60 | 1.43 | 1.55 | 1.523 |
| (alt) turn MoE -> 1.15x floor | 1.71 | 1.52 | 1.55 | 1.590 |
| + 16K MoE dequant gone (fused W4A16 at the grouped-GEMM rate, -640 µs) | 1.60 | 1.43 | 1.64 | 1.552 |
| + dense at 800 TFLOP/s everywhere | 1.69 | 1.43 | 1.64 | 1.580 |
| + resid at its floor, "other" 0 | 1.73 | 1.52 | 1.67 | 1.635 |
| + FA3 1.3x faster on every shape | 1.81 | 1.58 | 1.72 | 1.702 |

Conclusions:
- The **turn MoE** is the only component far from its floor, and alone it is worth 1.353 -> 1.52-1.59. A leader at 1.70 almost
  certainly runs T753 MoE in ~250-300 µs (we run 514-521) and T3012 in ~1.0-1.1 ms (we run 1.32-1.35).
- After that, 1.70 needs everything else together: the 16K dequant gone (fused W4A16 GEMM), turn-4 glue (other: 115-154 µs) and resid at
  their floors, and an FA3 that beats SGLang's by ~1.3x.
  - We share SGLang's FA3 (ours is only 3-10% faster, through contiguous KV / splits). An FP8 hdim-64 FA kernel is exp-bound near 990 TF
    and FA3 runs 640, so 1.3x is possible in principle but is a large separate project.
- Uniform "1.26x on everything" is not possible: first turn would need 1.95 ms of weighted time, which is below its floor sum at 650 TF.
  The leader must be well ahead of us where we are furthest from the floor, which is the turn-size MoE.

## 4. Why the turn MoE is slow (H100 PROF, scored profiles)

Ring v23 at T753: 558 µs. Ablations:
- NOCOMP (no dequant, no mma): 344 µs.
- FAKEDQ + NOMMA: 437 µs.
- NOMMA: 501 µs. FAKEDQ: 498 µs.

What the profile shows:
- The producer warp almost never waits for a free slot.
- ~60% of the time is the ring's data-movement skeleton. Each 256-row unit streams its weights and re-gathers the unit's 64 activation rows
  from L2 every stage, so the activation bytes equal the weight bytes.
- The rest is mma.sync plus in-register dequant, which is not overlapped with the movement.
- XBULK (bulk copies for the activation rows) made it slower (583 µs). TP=128 is 1.4x slower (RPW=1 halves weight rows per stage).
- At T753 the ring's math runs ~240-260 TFLOP/s.

The heavy experts (400-635 tokens on one expert at T753, 1.2-2.5K at T3012) are compute-bound territory. SGLang's answer (Marlin, mma.sync)
is 749 µs there. The dequant + cuBLAS-grouped-GEMM answer (our "big" path) loses at T753 (1232 µs) because it streams 126 MB of BF16 per
expert through HBM twice.

## 5. The pick: a fused W4A16 grouped wgmma GEMM, dequant-once per CTA tile

Design: `w4g` in `glue_pf2.py`, used by the pf14 "w4" MoE path.
- **Inputs:** a drop-in for `_gmm(A, W_bf16, offs)`. A is the expert-contiguous gathered and normed activation (`gnorm`, already used by
  the big path), so the token tiles are dense rows and no gather sits in the inner loop. W stays NVFP4: codes and e4m3 scales repacked
  per 64-K step.
- **Tile:** BM = 256 weight rows x BN <= 128 tokens x BK = 64.
  - 5 warpgroups: WG0 loads and dequantizes; WG1-4 run wgmma, each on one m64 row block x n128.
  - A = dequantized weights in SW128 smem, B = token rows in SW128 smem.
  - 4-stage full/empty mbarrier ring (empty = one arrival per consumer warp), dequant lookahead 2, staging ring of 3 TMA-bulk NVFP4 chunks.
  - As built: two m64 blocks per WG at n128 needed > 168 registers (ptxas spilled and serialized the wgmmas), so it is 4 WGs x 1 block
    (93 registers, no spills).
- **Token tiles:** one wgmma shape per kernel, because mixing n16/32/64/128 in one kernel spilled. So each GEMM is two launches:
  - n128: full 128-token tiles plus remainders > 32;
  - n32: remainders <= 32, including all the light experts.
- **Epilogue:** writes `H = A @ W13^T` (gate | up, natural row order) or `Y = I @ W2^T` with the expert's global scale. `act2` does SwiGLU
  and the existing `comb` the post-expert norm and the weighted sum, i.e. the big path's glue with dequant + cuBLAS replaced.
- **Watchdog:** each mbarrier wait gives up after ~2^22 polls and sets a device flag; all later waits return at once, so a pipeline bug
  drains with garbage instead of hanging the run. prepare() reads the flag (`w4g_err`) and drops the w4 paths if it fired.
- **Why this differs from w4wg2 and v24-v26:**
  - w4wg2 targeted decode, where per-step memory time is ~270 cycles, so dequant ALU (1000 cycles per 64-K step) could not hide.
    Here a 256x128x64 step is ~1600 cycles of wgmma at 650 TF, and its 16K-element dequant is ~260 instructions per WG0 thread,
    so dequant hides under the math.
  - v24-v26 were RS wgmma (A in registers) with a fence/commit/wait every k16. This is SS with a 4-stage ring: one commit per 64-K stage
    and wait_group 1, so wgmma overlaps the next stage's dequant.
  - The movement differs from the ring:
    - Per stage per CTA it moves 9.2 KB of NVFP4 weights + 16 KB of tokens for 4.2 MFLOP (166 FLOP/B, ring 120).
    - Tokens come from contiguous rows, not gathered ones.
    - Weights are dequantized once per (256 rows x 128 tokens) instead of once per 64 tokens.
- **First-principles T753** (scored profile, 1353 routes on 15 experts):
  - W13 is 91 GF of useful math.
  - With the n128 / n32 tile classes, padding waste is ~10%.
  - Heavy tiles are compute-bound at ~4.9 TFLOP/s per SM: ~55 µs per 128-token tile.
  - Light-expert tiles are bandwidth-bound: ~22 µs per 256-row block.
  - Estimate: W13 ~180 µs + W2 ~90 µs + gnorm/comb ~25 µs, so **~300 µs vs the ring's 521 (1.7x)**.
  - At T3012, 545 GF at ~600 TF is ~0.95 ms, vs the hybrid's 1.32-1.35 ms.
  - At 16K it removes the 0.52-0.69 ms dequant.
- **Diagnostics:** pf14diag times each piece in prepare() (gnorm, w4g W13, W2, comb; ring and big for reference) on the scored profiles and
  prints per-tile-class rates.
