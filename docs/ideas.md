# Ideas to improve the Muse kernels

A prioritized, evidence-backed backlog for `muse_attention_decode`,
`muse_moe_w4a16`, `muse_bf16_proj`, and the composed decode/prefill layers.
The goal is real upstreamable speed, not harness wins. Every item names the
code it touches, the measured reason to believe it, and the smallest test.

## Status snapshot

Recorded `okc check` results on H200 (see `notes/`):

| problem | best recorded | main remaining bottleneck |
|---|---:|---|
| `muse_moe_w4a16` | **1.339x** (v4) | weight streaming / scheduling, not dequant ALU |
| `muse_bf16_proj` | **0.94x** | m<=32 GEMV; non-router m>32 falls back to torch; router stays hi+lo |
| `muse_attention_decode` | **0.86x** | scattered FP8 gather, redundant FP8->F16, split merge |
| `muse_decode_layer` | not in notes | glue + attention + MoE composition |
| `muse_prefill_layer` | not in notes | grouped NVFP4 MoE, long-context attention |

The latest commits (`e77e9c9`, `758cbc6`, `c373c19`) are **unmeasured** here.
Run `okc check submission.py` before believing any idea in this file.

## Status 2026-10-24 (latest judge checks; supersedes the snapshot below)

| problem | best check | leader | notes |
|---|---:|---:|---|
| attention | 1.36x (v21) | 2.73 | v24-v31 pending: cluster merge (b1), early loads, page-table smem rings |
| moe | 1.37x (v9) | 2.35 | v11 bs1 1.61x but lookahead-3 hurt balance; v12 tagged slots pending |
| decode layer | 1.22x (v1) | 2.10 | v2 pending (fused verify, moe large-M) |
| bf16_proj | 0.94x (v5) | 1.50 | v6 deep CTA ring was worse (0.83) |
| prefill | fails->passes on pod | 1.745 | v1 pending; MoE (1.35 ms at T=753) and long-ctx attention dominate |

Measured corrections to the items below:
- A1: `mma.sync ... e4m3` is emulated on sm_90 (ptxas expands to F2FP+HMMA): no gain. Real FP8 needs wgmma
  (done in `attnw`: FP8 QK from the raw cp.async ring, f16 PV).
- L2 bulk prefetch ahead of the ring: much slower on full shapes (tested PFF/PFD 2-12).
- A3: NSDIV 128 won for sliding b1 (16 splits); clusters only for B*NS <= 64 (GPC fragmentation => 2nd wave).
- fdec ablation (b1 sliding, graph us): full 14.5, no-loads 14.5, no-compute 12.2, no-merge 12.8,
  skeleton 10.0, empty 4.9 => sliding is prologue/merge/latency bound, not memory. Full b32: skeleton 98 us
  from dependent page-table loads => smem page-table rings (v30/31).
- MoE: ptxas warp-aggregates `atomicAdd` (atomic + shfl) and the shfl waits on the contended queue atomic;
  use `atom.global.inc` asm. First-iteration stall ~5 us at kernel start (cold code? unresolved).

Tooling: `python3 tools/cu.py extract` writes each submission's CUDA to `cu/*.cu` (for nvcc, `cuk res/sass`,
compute-sanitizer, reviews); `python3 tools/cu.py embed <problem>` writes edits back (submission stays one file).
Reviews from other models go to `docs/reviews/` and must be checked against measurements.

## Guardrails

- Time graph replays, cold L2, paired ratios; the judge subtracts ~4 µs.
- One variable per check. The 95% interval is at least ±1.5%; ignore smaller
  deltas unless they repeat.
- Optimize the weighted loads, not just the geomean: decode 1 has weight 46 or
  62 on most problems.
- Correctness first: FP8/FP4 changes must pass the task's own error bound.
- Do not cache outputs or move work outside the timed call.

## Local 3090 Ti evidence

`cuda/` contains four small probes; raw output is `cuda/results.txt` and the
interpretation is `cuda/README.md`. These are sm_86 mechanism checks, not H200
numbers: no TMA, no FP8 tensor cores, 6 MB L2, 100 KB shared.

| probe | result | implication |
|---|---|---|
| gather 128 B rows | random 829, runs 878 GB/s | A2 is only ~6% for the current 128 B token row |
| gather 64 B rows | random 531, runs 896 GB/s | if a path splits per head, run coalescing is ~1.7x |
| FP8->F16 + MMA | 4x conversion ~6.7x the 1x conv+MMA time (emulated on sm86) | measure H200's native cvt; convert once |
| 36 KB tile, 4 passes | stream 0.06 ms vs resident 0.04 ms, **1.47x** memory-only | M1 can pay, but not 4x: re-streamed passes hit L2 |
| split count | B=1/80K ns=32/64/132 ~588 GB/s, ns=16 401; B=32 ns=1 922, ns=4 870 | A3 needs a sweep; 32 splits may be enough for B=1 long |

## Priority table

| # | idea | touches | expected | effort | confidence |
|---|---|---|---|---|---|
| A1 | FP8 QK, convert V once | `muse_attention_decode/submission.py` (`fdec`, `attnw`) | high for verify/full | med | med |
| A2 | coalesce paged-cache row gathers | `fdec` `issue`, `attnw` | med (128B rows already fast) | med | med |
| A3 | split-count sweep, global merge | `kernel_fn`, `run_fdec` | med | low | med |
| A5 | WGMMA only where the row count fits | `attn9`/`attnw` | med | low | high |
| A6 | validate the page-table smem ring | `fdec`/`attnw` `ptseg` | low-med | low | low |
| A7 | keep every decode on 128 B fused rows | `kernel_fn` dispatch | med | low | med |
| M1 | full weight tile residency across token passes | `muse_moe_w4a16` `Ctx`/`run` | high for bs128/256 | high | med |
| M2 | `NTM=16` / occupancy sweep | `muse_moe_w4a16` `_ext`, `dispatch` | high for large M | low-med | med |
| M3 | precompute the local route list | moe routing prologue | med for bs256 | med | med |
| P1 | direct stores for fully owned stream-K tiles | `muse_bf16_proj` `skinny` | med for m<=32 | low | med |
| P2 | CTA/ring sweep for GEMV | `_ext`, `run` | med | low | med |
| L1 | fuse projection epilogue + prep | decode/prefill `prep` | med | med | med |
| L2 | fused W4A16 grouped GEMM for prefill | prefill MoE path | high for 16K prefill | high | med |
| L3 | bigger prefill query tiles | prefill attention | high for long prefill | high | med |

---

## A. Attention decode

Current path: fused `fdec` (one CTA per request/split, 8 warps, both KV heads),
cp.async ring `RBF`, online softmax, CTA merge, then global or cluster split
merge. The gather is 64 B per head row / 128 B per token row of FP8 cache.

### A1. Use FP8 QK; stop converting K to F16

`fdec` still does `mma16(..., e2h(kk[j]), ...)` for QK, i.e. converts each FP8
K word to F16 before HMMA. `attn8` already uses
`mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`; `attn9` uses the
WGMMA `m64n64k32.e4m3.e4m3` form. The measured verify
bottleneck was F2FP conversion (~376 µs at b32) exceeding HMMA (~145 µs).
The same conversion is present in `fdec`; verify the size of the win with a
check before rewriting the whole path.

Do:
- QK: use `m16n8k32` E4M3 MMA directly. Query is already allowed to be FP8;
  K is E4M3 in the cache. Remove `e2h()` on K.
- PV: keep F16 V (or convert V once per ring slot), accumulate FP32.
- Keep the same softmax scale (`p.qk = sm_scale * log2(e) * kv_scale`).

Risk: QK accuracy. The check allows FP8 queries; run the full 16-test set.
Local sm86 probe: 4x conversion was ~6.7x the 1x conv+MMA time, but sm86
emulates FP8 conversion; on H200 measure the native cvt cost and the check
accuracy. Expected: verify/full layers; decode may also win because QK is
~half the MMA.

### A2. Coalesce 64/128 B cache rows into runs

Measured random gather (H200, `bench/gather.cu`):
64 B rows 2.1–2.4 TB/s, 128 B 2.9–3.1, 256 B 3.2–3.4. The scored page size is
1, but adjacent tokens may be contiguous in the cache; measure the actual run
lengths before relying on it.

Do:
- In `issue`, inspect the four `rw` rows. If they are consecutive, issue one
  32/64 B copy per lane group instead of four 16 B copies, or use a single
  `cp.async.bulk` of 128/256 B per row.
- Keep the 16 B path as the fallback for gaps.
- Measure the actual run-length distribution in the scored distributions
  (`soft`, `coherent`, `agentic`, `mixed`). Do not assume.
- For page size 64, a page is 64 contiguous 128 B rows; use 256/512 B bulk
  copies instead of per-row 16 B copies. Local 256 B random rows hit 864 GB/s
  vs 829 for 128 B.

Local 3090 Ti: 128 B random 829 -> runs 878 GB/s (+6%); 64 B random 531 ->
runs 896 GB/s (+69%). `fdec` already loads 128 B per token row, so A2 is
mostly a 64 B-path win. Coalesce runs only after measuring the distribution.

### A3. Sweep `NS`/`NSDIV` instead of hard-coding 128

`kernel_fn` uses `NS = min(132//B, span/128)`. For sliding 2048 that is 16
CTAs for B=1; for full 80K it is 132. The cluster path rounds small NS to a
power of two and only runs when `B*min(NS,16) <= 64` (latest `c373c19`); it was
restricted because clusters fragmented GPCs and forced a second wave.

Do:
- For B=1 sliding, try 32/64 splits (64/32-key chunks) with the global
  `comb2` merge. The merge is tiny compared with filling 16 SMs.
- For B=1 full, 132 splits is already one wave; try 66/132 with 64-key chunks.
- For B=16/32, NS=8/4 already gives 128 CTAs (one wave); sweep 4/8/16 to
  confirm more splits do not create a second wave.
- Record the winner per `(B, window, q_len)`; branch on shape is allowed.

Local 3090 Ti read-only model: B=1/80K ns=32/64/132 all ~588 GB/s, ns=16
401; B=1/1M ns=32 898, ns=64/132 ~857-876; B=32/80K ns=1 922, ns=4 870,
ns=8 760. This is KV read + merge only; attention math changes the optimum.
Expected: med on decode-1 loads; measure per-load, not just geomean.

### A4. Tune `RBF`, warps, and stage size

`_ext()` sets `RBF=5` for big L2 and 2 otherwise; `PFF=0` because L2 prefetch
hurt. Test `RBF=4/6`, 4 vs 8 warps per CTA, and 8 KB vs 4 KB stages. Do not
retry `cp.async.bulk.prefetch.L2`; it lost before.

### A5. WGMMA only where the row count fits

`attn9`/`attnw` use WGMMA for verify (8 tokens × 16 heads = 128 rows), where
`m64` does not waste rows. Decode has 16 rows, so `mma.sync m16n8` is the right
shape; do not force WGMMA on decode. Ensure verify/prefill dispatch actually
sets `WGV` and uses the WGMMA path on H200.

### A6. Validate the page-table smem ring

The latest uncommitted change streams page-table entries into shared memory
(`ptseg`/`rowss` in `fdec`, `ptseg`/`rowr` in `attnw`). The page table is only
~4 B/token: 80K tokens = 320 KB, 1M = 4 MB. It is likely L2-resident.

Do:
- A/B the smem ring against direct `__ldg(pt + idx)` with the same
  three-block lookahead. The ring removes a dependent load chain but adds a
  copy and shared-memory pressure.
- Keep it only if it wins on the scored shapes, especially page size 1 and the
  1M-token test.
- Test page size 64 separately; the lookup/run pattern differs.

Local 3090 Ti: not measured directly; the gather probe shows row bytes, not
table lookups. Add a `ptable.cu` probe before deciding to keep the complexity.

### A7. Keep every decode on 128 B fused rows

`fdec` loads both KV heads as one 128 B token row. The non-fused `attn` path
loads 64 B per head. Local gather: 64 B random 531 GB/s, 128 B random
829 GB/s (+56%). If any decode case still dispatches to the per-head path,
route it through `fdec` or coalesce the two heads into one 128 B copy.

The scored shapes are hq=16/hkv=2 and already take the fused path; check the
`heads dp` and other test shapes before changing dispatch.

---

## B. MoE W4A16

Current best v4: persistent kernel, global tile queue, 16 warps × 2 TMA stages
for small M, 8 warps × 4 stages for large M (`LARGE=True`), split-K in shared
for M<=64, per-quarter G2 dependencies, release-add tile counts.
Measured: FAKEDQ (no dequant ALU) only -12%; FAKEDQ+NOMMA at bs32 still 181 µs.
So this is **memory/scheduling-bound**, not ALU-bound.

### B1. Hold one full weight tile across token passes (highest large-M idea)

`run()` is called once per 64-token pass (`np = ceil(cnt/(8*NTM))`), and each
call re-streams the same 8 × 4.5 KB = 36 KB tile (32 KB weights + 4 KB
scales). For bs256 that is 4x the weight traffic. The tile is small enough to
keep in shared memory.

Do:
- Stage the full 16-row × 4096-K tile (8 × 4.5 KB = 36 KB: 32 KB weights +
  4 KB scales) once per CTA, double-buffered (72 KB), and loop all token
  passes over it.
- Split the 8 K-stages across warps as today; keep the partials in shared.
- Replace the per-warp ring with a CTA-level ring/mbarrier for this tile.
- Budget: 72 KB tile + scratch < 228 KB; one CTA is comfortable, two are tight.

Local 3090 Ti memory-only model: streaming the 36 KB tile 4x (151 MB issued)
vs resident (37.7 MB) gave **1.47x**; re-streamed passes hit L2. So the real
H200 gain is bounded by L2/TMA overhead, not 4x. Expected: high for
bs128/256, neutral for bs1; measure before the full redesign.

### B2. `NTM=16` and occupancy sweep (cheap first test)

Before the full-tile redesign, test `NTM=16` (128 tokens per pass) with the
existing ring. It halves the number of weight passes for bs128/256; register
cost is ~2x accumulators, so watch spills. Then sweep:

| WPC | stages | CTAs/SM | notes |
|---:|---:|---:|---|
| 16 | 2 | 1 | current small-M |
| 8 | 4 | 1 | current large-M (only 8 warps/SM) |
| 8 | 2 | 2 | new; needs `g_grid=2*SM` and smem budget |
| 12 | 2 | 1 | middle ground |

The current large-M config runs only 8 warps/SM (12.5% occupancy) with a
147 KB ring. Re-test 16×2 first.

### B3. Queue and routing overhead

- Latest `758cbc6` uses tagged slots and lookahead 2 (`LA3`). Try lookahead
  3/4 and per-group queues; keep the static first three tiles.
- Every CTA scans all `M*8` routes and builds a shared local list. For bs256
  that is ~2K shared atomics per CTA duplicated across ~132 CTAs. Precompute
  the compressed local route list once (small kernel or one CTA) and have
  consumers read it; only worth it for large M.
- `atom.global.inc` is used because `atomicAdd`+shfl waited on the contended
  atomic; do not revert to the warp-aggregated atomic without measuring.

### B4. Stage size / L2

- Try 8 KB stages (fewer TMA issues) with the same total ring bytes; 2 KB was
  slower, 16 KB was good for the projection GEMV.
- Do not add L2 persistence assuming routing reuse: each weight tile is
  assigned once per call. The reuse is across token passes (B1), not across
  CTAs.
- Prefetching the next weight tile may help now that slots are tagged; test
  after B1/B2, not before.

### B5. Cluster split-K / DSMEM

Only worth it if cross-CTA split-K is needed. Current split-K is inside one
CTA. For small M the CTA already has 16 warps; cluster overhead likely loses.
For large M there is no split-K. Low priority.

---

## C. BF16 projections

`kernel_fn` uses the custom stream-K GEMV for non-router `m <= 32`;
non-router `m > 32` falls back to `x @ W.T` (torch/cuBLAS); the router keeps
the custom hi+lo path at every m. `V6=False`: the deep per-CTA TMA ring was slower
than v5 on H200.

### C1. Direct stores for owned tiles

`skinny()` accumulates all partial tiles into `ws` with FP32 `atomicAdd`, then
`fin()` converts and clears. A warp whose range covers a full tile can write
the output directly; only split tiles need the workspace/atomics. This removes
output traffic and atomic latency for the common m<=32 case.

### C2. CTA/ring sweep

`_ext()` uses `WPC=16` (big L2), `RSN=2`, `CUN=2`. Test 8 warps × 4 stages with
2 CTAs/SM, and 12×3. Do not revive V6 (deep per-CTA ring), which lost.

### C3. Dispatch m>32 to a tuned GEMM

For m=33..1000 the custom path already calls torch, so the only win is a tuned
CUTLASS/cuBLASLt kernel with the right epilogue; the scored shapes are 128/256.
This is lower priority because cuBLAS is strong at these shapes; measure a
WGMMA 128×128/128×256 kernel before writing one.

### C4. Router

The router always uses the custom BF16 hi + lo path, at every m (the
`layer != "router"` guard only sends the other projections to torch above 32).
The hi+lo path passes the 0.01% bound, but at m=128/256 it was still slower
than SGLang split2. Try dispatching the router to torch for large m, or a
dedicated GEMV, then measure. Do not switch to BF16-only. If touching this,
fuse top-k into the epilogue to avoid materializing scores.

---

## D. Composed decode/prefill layers

### D1. Fuse projection epilogue + prep

`qkv_gate` is written BF16, then `prep` reads it, does QK-norm/RoPE/FP8 cache
write, and writes the query. For decode this is a small write; for prefill
(753–16384 tokens) it is 2304×T BF16 written and read. Fuse the norm/RoPE/
cache-write epilogue into the projection kernel, or at least keep the qkv tile
in shared.

### D2. Fused W4A16 grouped GEMM for prefill

`muse_prefill_layer` dequantizes routed experts to BF16 and runs grouped GEMMs
above `MOE_SMALL_MAX` (2048 on Hopper). For 16K-token turns this writes/reads a
large BF16 weight copy. Use a W4A16 grouped GEMM (dequant in the GEMM epilogue,
CUTLASS NVFP4/Marlin-style) so the BF16 weights never hit HBM. This is the
largest prefill opportunity after attention.

### D3. Bigger prefill query tiles

Prefill attention uses 8-token query blocks (`PF` path). For 753/16384-token
prefills, use 64/128-token query tiles with WGMMA and TMA. The verify path
already proves 128-row WGMMA works; the prefill shape gives even better tensor
utilization and more KV reuse.

### D4. Fuse router/topk and norms

Fuse the router GEMM + top-k + bias into one epilogue; fuse residual norms and
the final weighted sum into the MoE epilogue. These are launch/traffic wins,
not roofline changes; do them after D1–D3.

---

## E. What not to retry

| experiment | measured result | reason |
|---|---|---|
| MoE warp-specialized producer/consumer ring (v6) | 1.188x vs 1.339x | extra complexity, no stall to remove |
| 12 warps × 3 TMA stages | slower than 16×2 | stage/occupancy tradeoff |
| Full-stage B prefetch | no change | latency already covered |
| Interleave G2 right after G1 | worse | same-round dependency waits |
| Attention L2 prefetch (`PFF`) | hurt | competes for bandwidth |
| 2 KB TMA stages | slower | per-stage overhead |
| bf16_proj V6 deep ring | slower (`V6=False`) | per-CTA ring overhead |
| Cluster merge for large grids | second wave (GPC fragmentation) | `c373c19` |
| FAKEDQ / NOMMA | only -12%; still 181 µs at bs32 | not ALU-bound |
| Prompt-lookup spec on CPU | 20.4 → 19.5 tok/s | compute-bound |

## F. First checks (reordered after local probes)

1. **MoE B2** — `NTM=16` and 8×2×2-CTA occupancy. Cheap, no accuracy risk;
   check bs128/256 first (weight 60 each). If this wins, do **B1/M1**.
2. **MoE M1** — full-tile residency. Local memory-only bound is ~1.47x; the
   real gain depends on TMA/L2 overhead. This is the highest-upside redesign.
3. **Attention A1/A6** — FP8 QK + convert once, then A/B the page-table smem
   ring against direct lookups. Run the full 16-test check and compare the
   error bound.
4. **BF16 P1** — direct stores for owned stream-K tiles. Low risk, easy to
   revert.
5. **Attention A3** — 32 vs 64 vs 132 splits for B=1 long; B=32 needs the
   compute path, not just the read probe.

A2 is now a 64 B-path optimization; do it only if the gather distribution
shows 64 B rows and non-contiguous runs.

## G. Acceptance

- `okc check` passes all tests, no flagged/rejected.
- Each load improves or stays within noise; no shape regresses >2%.
- Latest commits are measured, not assumed. Record the exact content hash and
  `profiles/submission.<job>.json`.
- If a change only wins the geomean but loses a high-weight load, it is not a
  win.

## Status 2026-10-24 evening
- Judge profiles (`<problem>/profiles/*.txt`) include a speed-of-light column. MoE v11/v12: bs1 41 µs vs SOL 11, bs256 452 vs SOL 109 → large-M path runs ~1.2-1.7 TB/s.
- MoE large-M hypothesis: every 16-row tile re-reads its expert's activations from L2 each 64-k chunk → L2 traffic ≈ cnt/4 × weight bytes (≈3x at 13 tok/expert). Fix: share x across adjacent tiles (smem x ring per warp group, B via lds/ldmatrix; or wgmma RS with A = dequantized weights in registers, B = x in smem). Ablation macro NOBLD.
- bf16_proj: cuBLAS beats our skinny kernel for m 8-32 on qkv/o/pooler/unpooler(m>8) → v8 dispatches those to torch. Router: dedicated `rtr` kernel (CTA = 16-row tile × 16 K-chunks, one-shot register loads, smem warp reduce, last-arriver across 8 K-splits; no atomics, no fin kernel).
- torch fp32 matmul for the router is slow (35-55 µs at m 128/256): keep ours.
- w4wg2 (wgmma dequant-to-smem grouped GEMM) wrong (err 5.4) and 2x slower than v1; v1 is dequant-ALU bound (~1000 cycles/64-k step vs 270 memory budget). prmt-LUT dequant (DQL) written, untested.
- RunPod pods also block GPU perf counters (ERR_NVGPUCTRPERM). daisy (sudo) is the only ncu box (sm_86).
- Judge image ships DeepGEMM, CUTLASS, CuTe DSL, FlashInfer, TileLang, Triton: possible to launch a pod on lmsysorg/sglang:dev-muse-spark-open for exact env and to read their sm90 kernels.

## 2026-10-25 H200 session (pod, ~27 min)
- bf16_proj gv (submission_v10.py: x K-slice staged in smem, stream-K over (slice, tile, chunk), in-kernel reduction): judge 0.74-0.92, loses to skinny+fin. Globaltimer per-warp trace (bench/gvt.py, -DDBGT=1):
  - x staging wait 2-5 us before first MMA (o m=1 xready p50 2.0, m=32 5.1) even with x bulk issued first and 160 KB/SM weight prefetch.
  - in-kernel reduction tail: atomics (red.v4) or partial slots + last-arriver gather add 5-18 us (qkv m=32 end 36 vs streamend 19). fp32 L2 atomics at M=32 volume (S*M*N) dominate: no-atomic ablation o m=32 26.1 -> 11.5.
  - pure stream (no atomics, no staging) ~= cuBLAS (pooler 25.2 vs 27.9 bench-us, qkv 16.0 vs 18.8).
  - ncu unavailable in RunPod containers (ERR_NVGPUCTRPERM).
- bf16_proj router m=256: one cuBLAS mm on [hi;lo]^T (N=512) + add: 11.4 vs 17.7 us (1.54x). Kept (RTRM=128).
- attention verify b16 layer: 164 us; -DAW_NOLOAD 145, -DAW_NOMMA 154, both 139. ~85% is neither KV loads nor MMA: softmax/exp, page table, barriers, combine. b32: 313 / 277 / 294 / 264.
- attention attnw verify (H200 clock64 per phase, bench/awprof.py, -DAWPROF): ~2850 cycles per 64-key step per warpgroup, shares wait 3% / issue 17% / V convert 16% / QK 11% / softmax 35% / PV 18%. Step loop ~840 SASS instrs; issue-bound with 4 WGs/SM (RSW=5 -> 1 CTA/SM: steps 2230 cycles but 260 vs 218 us). ex2.approx.f16x2 is emulated (2x MUFU.EX2.F16 + PRMT): no gain. Lazy rescale (FA4-style, threshold 2^8): -1..2%.
- ptxas C7515 (wgmma serialized) in every attnw: zero-filled PV accumulators reaching the loop header. Fix: no o=0 init, first PV with runtime scale-d predicate (wg_rsz), zero o after loop if nb == 0. 6 -> 2 WARPGROUP.DEPBAR per step (no measurable time change: not the bottleneck).
- attention split sweeps (bench/attn_sweep.py, OVR knobs): sliding B=1 time flat in nsplit/NS (fixed overheads dominate).
- fdec timeline (sliding B=1): prologue 3.2 us (seq -> page table -> KV: 3 dependent DRAM trips), loop ~1 us, CTA merge 2.7 us of which 2.2 us was 16-way bank-conflicted scalar st.shared of the warps' partials (fixed in v35: float4 + padded swizzled rows), cross-CTA merge ~2 us.
- MoE (H200 DIAG, v9): bs1 span ~55 us: routing 2.4, phase-1 per-warp 23-27 us (~3.9 us per 4.6 KB stage, TMA wait <1 us), epilogue 7.5 us/warp. NOBLD: no change at bs<=32; FAKEDQ -18%; NOFENCE -4/-11/-13 us (bs1/8/32); KSF=1/2 worse at bs1. v13 deferred tile release: 1.320 (worse than v9 1.371: phase-2 waits longer). Stage loop NT=1: 679 SASS (LOP3 199, SHF 160, HMUL2 128, HMMA 32): dequant ~15 ops/8 values; no single-rotation bit layout exists for 11 ops.
- prefill v2 (judge): turn 1 full 2910 vs sgl 1490: MoE 1567 us (52%, ~0.58 TB/s: x rows re-read from L2 per 16-row tile, ~4.7 GB L2 at T=753), attnw 1117 us (37%: 8-token q-blocks re-read KV ~94x). Needs multi-tile x reuse in MoE and 64+ token q-blocks in attention.
- MoE v14 heavy-first expert order: 1.400 (best). v15 per-(tile, pass) queue items (needs a stable per-CTA token list: atomicAdd list order differs per CTA, so split passes covered different tokens -> wrong): 1.284 (worse: duplicated dequant across SMs, serial list build). v16 = v15 + red.release.gpu add instead of fence.acq_rel + atomicAdd: small-bs -3..4 us vs v15 -> v17 = v14 + release add.
- prefill v6 1.034: FA3 prefill attention (sgl_kernel, judge env only) + SGLang Marlin MoE for 512 < T <= 8192 (turn loads ~0.96), our dequant + grouped GEMM at 16K (first turns 1.17-1.26). Marlin at 16K was 0.94-0.96.
- bf16_proj v12: prepare()-time cuBLASLt algorithm autotune (graph replays, cold L2, on the judge H200) for every cuBLAS-served shape: 1.146 = v11 (default heuristic already best). Judge nvcc (-std=c++20) ICEs on STL containers in .cu: put host-only code in cpp_sources.
- MoE v17 = v14 + red.release add: 1.418 (best). decode_layer v7 (moe v17): 1.348; v8 router via rtr/concat GEMM: 1.341 (router overlaps pooler on a side stream, no gain; reverted).
- attention v38: P row sums via an m64n8k16 wgmma against a 1 KB all-ones smem tile (same f16 P fragments as PV; no FADD sums, no quad shuffles): loop 786 -> 735 SASS, judge 1.599 (verify full b16 160.8 -> 156.9, b32 302.2 -> 293.7 us). Hoisting wgmma descriptors (v37): -6 SASS only (compiler already hoists). Next: drop the per-step row max (P with the running max, packed HSETP2 > 2^8 check, exact-max slow path).
- MoE stream-K small-batch path (moe_sk, contiguous per-warp (expert, tile, stage) ranges, fp32 ws partials, last-arriver finalize, stable token list): v18 1.372 (bs1 51.0, bs8 128.8 vs v17 52.4/105.5). Timeline (H200, bench/sk_t.py): median warp streams phase 1 at ~3.5 TB/s but max warp +7 us per phase; barrier waits on stragglers. v19 dynamic 4-stage chunks (queue fetched 2 ahead): 1.272 (bs1 68: per-chunk flush/acq_rel/prefetch restart). v20 no in-stream atomics, grid barriers + parallel finalize passes: 1.333 (bs1 61). All worse than v17 (1.418): not pursued. Bug notes: full-mask __shfl inside a lane-divergent loop hangs; never init a shared static (g_grid) that another entry point keys its cudaFuncSetAttribute on.
- bf16_proj wgmma GEMM prototype (bench/wgg.py: 128 W rows x all tokens per CTA, W as prepare-time SW128 tiles via 1D bulk copies, x via TMA tensor map with OOB zero rows, split-K over a cluster reduced in DSMEM): correct first try, but slower than cuBLAS (H200, graph + cold L2, incl ~3 us offset): qkv m128 26.1 vs 22.8, o 13.0 vs 10.2, pooler 47.5 vs 28.4, unpooler 30.2 vs 26.9. Lessons: cluster size 7 does not pack GPCs (2 waves), use 1/2/4/8; DSMEM reduction must be token-major float4 with all KS loads in flight (scalar per-element map_shared_rank loop cost ~30 us); main loop ~3x below BW: every CTA re-reads x (half of each 32 KB stage), 4 stages too shallow. Next if revisited: TMA multicast of x across a row-block cluster, 6+ stages, persistent tile loop.
- attention v39: attnw MRG arrival = __syncthreads + atom.acq_rel (no fence.sc), ld.acquire spin, two-pass split merge with independent loads; fdec: dropped fence.sc after the acquire spin. decode_layer v9 (attention v39): 1.335 vs v7 1.348 (noise; verify full b32 820 -> 796). Attention-only judge pending.
- decode_layer verify32 full (860 us): moe 378, attnw 304, cuBLAS projections ~93, router skinny+fin 31 + route 26 on the side stream (route's GPU time mostly queueing behind the pooler GEMM).
- bf16_proj v13 (submission_v13.py, local only so far): own lossless 12-bit BF16 format (per lane per 16x64 unit: 32 low bytes + 4 code words; nibble = 3-bit offset of the upper 7 exponent bits from a per-matrix 8-value window + sign bit placed so the sign ORs into the table-lookup word: hl = prmt(T0,T1,C&0x77777777) | ((C<<4)&0x80808080), hh = prmt(T0,T1,cm>>16) | (C&0x80808080), then 4 merge prmts = 11 ops / 8 values), natural [tile][chunk] layout so every geometry reads one pack; per-warp escape lists (cap 32, ~1.4e-4 of values on randn weights) patched in registers; exactness proved in prepare by an independent torch decoder plus one-hot runs of every column through each candidate kernel. Register-direct swap-AB mma.sync GEMV (WR x WK warps, KU chunks per warp, ST-deep register ring, split-K over S CTAs via cluster DSMEM on sm_90 / last-arriver ws elsewhere); prepare() times every verified candidate against the v12 path on the GPU (graph, cold L2) and keeps the fastest. 3090 Ti: pooler m1 63.5 vs v12 87 us (1.05x the bf16 DRAM floor), qkv m1 38.9 vs 54, unpooler m8 12.3 vs 21.5; load m1/8/16/32 vs torch 1.36/1.49/1.41/1.20.
- v13 failed knobs: __launch_bounds__ min-blocks to cap MT=4 at 128 regs spills (2-2.5x slower, c12 hit worst); 1024-thread CTAs with ST=2 spill on sm_90 (kept only MT=1 ST=1); WK=1 / WR=4-8 geometries (warps share x chunks through L1) only win at m=32 for o/pooler on the 3090 (DRAM-bound there; kept as H200 candidates since per-warp x re-reads at m=32 are ~2.5x the weight bytes).
- bf16_proj v14 (local only): judge v13 1.304. Prepare timer now flushes L2 with a read-only pass (torch.sum over 256 MB) instead of flush.add_ (dirty L2 -> writeback traffic inside the timed kernel; v13 picked fmt0 qkv m32 at 20.9 vs v12 21.8 in prepare, judge 15.8 vs 14.3), logs times minus a 1-kernel graph's cost, re-times the top 3 alternating with v12, margin 0.97. GEMV generalised: RT row tiles per warp, XS=1 x slab per CTA in swizzled smem via cp.async (B fragments by ld.shared, no x register ring; m32 per-warp x re-reads were ~2.7x the weight bytes), FMT 2/3 = router BF16 hi + lo (c12 per matrix, one escape list with a matrix bit), token blocks in grid.x. Router c12 hi+lo m<=32 passes (1e-4, one-hot exact). Failed: m=128/256 as MT=4 token blocks (weights re-decoded per block): 1.6-2x slower than cuBLAS on the 3090; needs a decode-once smem tile + wgmma/mma GEMM.
- bf16_proj v15 (local only): judge v14 1.320. Changes: (a) add-based c12 decode, 9 ops / 8 values (code byte i = off(v_i) | sign(v_{4+(i+3)%4})<<3 | off(v_{4+i})<<4 | sign(v_i)<<7; high bytes = (C & 0x87878787) + base*0x01010101 and the same on rotr(C, 4); no table prmt); (b) XS=2 GEMV mode: x loaded straight from global at use (L1), no x register ring, __launch_bounds__ min 2 CTAs (<=128 regs): on the 3090 it beats every other m32 variant (qkv 37 vs v12 53, pooler 66 vs 82, unpooler 62 vs 84 us); (c) rolled K loop (#pragma unroll 1 over ST-deep groups): pooler m1 kernel 2052 -> 665 static SASS (vc model: cold-L2 icache ~0.6 us / 1K instrs), same speed on the 3090; (d) FP32 router for m=1: CTA per 1-2 full rows, 512 threads, fp32 weights read once, no cross-CTA reduction (rr1 9.2 vs v12 11.3 us on the 3090); (e) margin 0.99; (f) every candidate's time printed in the prepare log. vc est (pooler m32): L2->SM traffic is the dominant term for per-warp x rings (134 MB of x re-reads); wgmma (A from registers) only trims tensor/issue time (~0.1 us) once x traffic is fixed, so it is not worth building before the H200 logs show XS2 numbers.
- bf16_proj v16 (local only): judge v15 1.3188 (v14 1.3200). v15 H200 log: the rolled-loop + 9-op-decode kernel (v15 global change) lost 0.4-2.3 us vs the v14 kernel at deep K (qkv m1 8.90 -> 11.23, pooler m16 13.60 -> 15.78, unpooler m1 3.46 -> 3.84), while it won m32 via x mode 2 (pooler 17.0, unpooler 17.2) and rr1 router m1 (3.46). Also on the H200: x-slab (XS 1) kernels 22-30 us at m16/m32 (lose), c12 hi+lo router 6-8 us (loses to v12 rtr 4.7: dropped), rr m8 7.7-14.9 us (dropped). v16: kernel variant V as a template parameter (bit 0: 9-op add decode vs 11-op table decode, each with its own pack; bit 1: rolled vs fully unrolled K loop); top geometries get V 0/1/3, so the next log separates the decode effect (V1 vs V0) from the loop effect (V3 vs V1). sm_90a: unrolled MT=4 x-mode-2 variants spill 8-80 B under the 128-register cap (rolled twins don't).
- bf16_proj v17 m128/256 wgmma work (bench/wgc.py, H100 80GB pod, 33 min): c12 tiles decoded once per CTA into a SW128 smem ring used as wgmma B (W rows = N), x as wgmma A via TMA (128 tokens = 2 consumer warpgroups), split-K over a z-cluster reduced in DSMEM; natural-k c12 pack (pack_nat), escapes patched before the smem store. Correct first try (err 1.7e-3 = bf16 output rounding, one-hot probes exact; router hi+lo 3.3e-6). Iterations: v1 register-direct weights + 2-deep prefetch: 2-3x slower than cuBLAS (H100 pooler m128 45 vs 29 us); v2 one bulk copy of c12 rows + x TMA per stage (5-6 deep): no better; v3 warp-specialized (2 wgmma WGs, 3-4 decoder warps, producer warp, full/empty/bfull/bempty mbarriers): pooler m128 30 vs 28.8, unpooler m128 25.4 vs 28.0 (only win), o m128 10.2 vs 7.8, qkv m128 25.9 vs 16.7, every m256 ~1.7-2.5x slower. DIAG (no decode, no wgmma, loads only) is already at or above cuBLAS: the x tile is re-read through L2 by every 64/128-row block (pooler m256 BN=64: 256 MB of x traffic), so the kernel is L2-bandwidth-bound before any compute. x multicast over a 2-4 CTA row cluster (cross-CTA stage-empty arrivals): .release.cluster remote arrives made it 7-10x slower, .relaxed.cluster 2x slower than no multicast. BN=256 needs >128 regs/thread (416 threads cap at 128: ptxas C7602) and leaves too few CTAs. Clusters of 8 with 1 CTA/SM do not pack the GPCs (2 waves). Needed next: CTA tile 128 tok x 256 rows (acc 128 regs, setmaxnreg to rebalance producer/decoder vs consumer warps) plus 2-CTA x multicast without per-stage cross-CTA handshakes (e.g. pair-level empty barrier arrived once per stage by an elected thread), persistent tile loop.
- bf16_proj v18 (local only): judge v17 1.3606 (v16 1.366). m128/256 + router m128/256: prepare() cuBLASLt search over four descriptor layouts (bit 0 row-major with x as A, bit 1 W pre-transposed [K, N]), the heuristic list of each layout, seeded algo-66 configs (CUBLASLT_ALGO_CONFIG_* order: ID, TILE_ID, SPLITK_NUM, REDUCTION_SCHEME, CTA_SWIZZLING, CUSTOM_OPTION, STAGES_ID, INNER_SHAPE_ID, CLUSTER_SHAPE_ID; 7-8 are uint16), an algo-66 tile sweep (stages 35, clusters 1x1x1 / 1x2x1) and custom-option / cluster / split-K (compute-type reduction) / stages refinement of the four leaders; accumulator-32F / no-TF32 numerical flags, cublasLtMatmulAlgoCheck, numeric check, judge-style timing, 26 s cumulative deadline. 3090 (no algo 66): packed layouts alone win pooler m256 233 vs 291, unpooler m128 140 vs 148, o m128 40 vs 43, router m256 43 vs 49 us. Competitor's m128/256 wgmma kernels run on plain BF16 weights (TMA for W and x with L2 cache-hint policies: evict_first for weights, evict_last for x) and push split-K partials with st.async into the leader CTA.
- bf16_proj v19 (local only): judge v18 1.405. Router m8/16/32: GEMV FMT 2 (plain BF16 hi + lo fragments, no decode) with every warp's single unit in flight and split-K over a 16-CTA non-portable cluster (geometries 1.8.1.16 / 1.4.2.16 / 1.8.2.8); 3090: m8 10.3 vs v12 11.3, m32 13.3 vs 15.4 us (the H200 c12 hi+lo router at S=8 / ST=2 lost: 6-8 us). Algo search budget weighted toward qkv / router (o 0.3x). bench/wgb.py (compile-checked for sm_90a, not yet run): warp-specialized TMA + wgmma on plain BF16 weights, x as wgmma A (64 tokens per consumer warpgroup, rows >= m zero-filled by TMA), W as B with L2 evict_first and x with evict_last cache-hint policies, split-K over up to 16 CTAs in DSMEM; router hi + lo as two B tiles into one fp32 accumulator; for router m8..256 and qkv / o m32..256, to be validated on the next H100 session.
- bf16_proj v20 (local only): judge v19 1.4155. Competitor m32 qkv (m32x) = c12, MT 4, register ring depth 1, x straight from global, 128 threads, (1, 4, 8, 4) + push reduction: added a 7th geometry field forcing the ring depth and x-mode-2 ring-1 variants for qkv / o m16 / m32 (3090: they win all four, qkv m32 36.9 vs 37.9, o m16 15.4 vs 16.4 us). Router m128/256 as bf16 hi + lo GEMV token blocks (MT 4): 1.6x slower than v12's algo search on the 3090 (35 vs 22.5 us), left disabled (GVRB). Algo search budget 26 -> 22 s (12 more GEMV instances to compile).
- attention v40 (local only, judge pending): attnw rewrite. (1) bench/emu_chk.py: wgmma emulated in software on sm_86 (-DWGEMU=1, RSW=2; A-from-registers via smem, SW128/MN-major descriptors decoded as on H200) so attnw is correctness-tested locally; a planted fast-path bug fails it. (2) prologue: page-table + first RSW-1 KV steps issued before the query RMSNorm/RoPE; the block's own new keys are written straight into the prefetched ring slots (cp.async skipped for those rows). (3) copy mapping: thread = 2 keys x 32 B sector (2 lookups, 1 64-bit address per key per step instead of 4). (4) softmax fast path: P against the running max (no row max, no shuffles), overflow (P > 2^8) checked on packed f16 with VHMNMX + vote, rare per-warp rescale by 1/Pmax; exact path only for rows still at -inf. (5) QK issued first, loads/rows/V-convert overlap it; 1 WARPGROUP.DEPBAR per step (was 2). Dynamic step ~350 SASS vs ~440 (static 732 -> 910: two softmax paths). (6) merge: last nmg arrivers (up to 128 when the whole grid is co-resident, else 8) with split groups in smem; FU for every nsplit (verify B1 full: no prep/comb2). (7) sliding verify: 64-key splits (B1 33, B16 16, B32 8). vc est: B1 full 20.8 -> 16.8, B1 sl 13.0 -> 9.1, B16 sl 13.0 -> 11.5, B32 sl 16.3 -> 14.9, B16 full 155 -> 138, B32 full 301 -> 269 (savings ~1.5x optimistic) => judge ~1.74 expected.
- Failed/notes v40: ptxas 12.9 segfaults on attnw<PS1=true, FUSE=false> (prefill/non-fused); those paths now use PS1=false. Many spinning mergers deadlock when the grid is not co-resident (3090 emulation, 1 CTA/SM): nmg capped by an occupancy query. f16 P max via 3-input VHMNMX but ptxas re-packs P twice (11 extra F2FP; asm barrier does not stop it).
- prefill glue v7 (muse_prefill_layer/glue_pf2.py, tools/build_pf.py -> submission_pf7.py; prefill-only kernel_fn overriding glue.py's for T > 256): Marlin path: SGLang post-expert Triton kernel replaced by the same comb kernel in slot-major mode (Y rows t*8+k, lid < 0 skipped, fp32 (1+w) instead of bf16); FA3 prologue as one warp per token (all 20 head rows in flight; bitwise = A_ prep), FA3 epilogue (rms * sigmoid(gate)) one kernel instead of ~10 torch ops; resid1 writes h + per-row rms factor only, resid2 recomputes s1 from (s, a, ra) (22 vs 24 B/elem, s1 never stored); router as one [hi|lo] N=512 GEMM, the hi+lo add folded into routing (route2 also emits Marlin's local ids and the per-expert counts: no loc.long()/expert_map gather/rcount); moe_align on the side stream under the pooler; pooler launched first like SGLang; big path: rmsg+gather fused (gnorm, persistent, next row prefetched), vectorized swiglu, MoE combine as a per-token weighted gather of the Y rows (no fp32 atomics, no memset, no tobf), deq2 (arithmetic e2m1 decode, 32 values/thread; old deq had a stack LUT: 3090 130 -> 890 GB/s at w2, bitwise equal); early frees of w13/A/H/I. 3090 per-kernel (T=3012 old -> new us): prep 79 -> 31, attn epilogue 228 -> 26, rmsg+gather 190 -> 83, combine (post+tobf+memset) 537 -> 73, rmsg 106 -> 58.
- prefill glue v7 knobs that did NOT help on the 3090: resid as 256x32 or 512x16 per row (2-3 rows per SM in flight) is slower than 1024x8 there (651-734 vs 845-913 GB/s; 3090 is DRAM-bound at 1 CTA/SM). H200 old resid ran at only ~2.5 TB/s (1 CTA of 1024 thr/SM at 59-64 regs), so prepare() now times all four shapes on the judge GPU and picks per size (RV_TUNE). Same for dense GEMMs: prepare() times torch vs each cuBLASLt heuristic algo per prefill M (LT_TUNE; on the 3090 o / unpooler win 10-25%).
- not tried (reasoned out): deq on a side stream overlapping qkv/attention/pooler. nvjet and FA3 CTAs take the whole register file (setmaxnreg), so deq CTAs cannot co-reside; a persistent limited-CTA deq would delay the persistent GEMM's missing CTAs by the whole deq time, and a many-CTA deq launched first just serializes. Graph-captured stream priorities are dropped without cudaGraphInstantiateFlagUseNodePriority (torch does not pass it).
- prefill pf7 judge 1.152 (v6 1.034): 16K loads 1.29-1.53, turn loads ~1.01-1.02 (Marlin 64-70% there). H200 per-kernel at 16K: resa 275 us = 1074 MB (a bf16 + s fp32 in, h bf16 out; hidden is 8192, not 4096) = 3.9 TB/s; resb 550 us = 1878 MB (f, a, s in; s2, hid out) = 3.4 TB/s. The real outliers were gnorm 131 / swiglu 206 / comb 122 us at ~1.4-2 TB/s: soft routing gives only ~0.73 local routes per token (T=753: 547 routes, one expert 306), and the persistent grids were sized nsm*8 while only 4-5 CTAs/SM fit (two waves of fixed-size work).
- prefill pf8 (glue_pf2.py, local only): occupancy-sized persistent grids (swiglu); cp.async ring variants (each thread prefetches exactly the 16-B chunks it later reads, so the ring needs no barrier) for gnorm / comb (6 x 8 KB rows) and both residual kernels (resa 4 x 48 KB rows, resb 3 x 64 KB rows, 1 CTA/SM; templated on row width so a 4096-wide instance validates on the 3090's 99 KB smem); prepare() times 7 resid, 2 gnorm, 2 comb variants per size. Marlin path: prepare() times pooler + side-stream router against one fp32-output GEMM on [pooler; hi; lo] (N 4608; pooler columns rounded to bf16 in rmsg2f, routing reads the strided hi/lo columns), and Marlin moe_block_size 16/32/48/64 against our dequant + grouped GEMM on synthetic uniform and hot (40% of tokens on one local expert) routings; default kept unless 3% faster. Dense GEMMs: bf16_proj v19-style cuBLASLt search (4 layouts incl. prepare-time W^T, algo-66 tile x cluster sweep, custom option / cluster / split-K / stages refinement), 70 s budget, prepare deadline 255 s. On the 3090 the ring variants tie the register versions (DRAM-bound there).
- prefill pf8 judge 1.130 (pf7 1.152): FAILED pieces. (a) Fused [pooler; router hi; lo] fp32 GEMM at T=753: prepare's isolated segment said 91.5 vs 97.3 us, but in the layer the 81.5 us fused GEMM replaces a 51 us pooler with the router overlapped; T753 regressed 1052 -> 1120. (b) Marlin moe_block_size from synthetic routing (T753 bm32 620 vs bm64 807 us in prepare); the judge profile shows Marlin 764 vs 794 (sliding) but 840 vs 706 (full), so not attributable; pf9b isolates it. (c) cp.async ring resid/gnorm/comb variants: never won on the H200 (ring resid 10-25% slower; comb ring 5% at 16K only). (d) Full v19-style cuBLASLt search (~600 configs/shape): only o m3012 (-6%), o/pooler/unpooler 16K (-3..-10%), same as the cheap pf7 search; with 7 resid templates + rings, compile hit 214 s and prepare 255 s. (e) The 16K glue (swiglu 206, gnorm 138, comb 122 us) is NOT slow: CPU estimate of the soft routing gives ~0.6-0.75 local routes/token with one expert taking ~45% of tokens, but grouped-GEMM throughput and prepare's synthetic timings (3.3-3.8 TB/s at ~23K rows) imply ~25-28K local rows at 16K, i.e. these kernels already run at ~3.3 TB/s. Occupancy-sized grids changed nothing (swiglu 206 -> 206).
- prefill pf9: pf7 behaviour (side-stream router, Marlin heuristic block size) + resid fixed to the judge's picks (resa<512,2>, resb<256,2>, no tuner) + LT only for the 4 shapes that won, seeded with the pf8 judge configs + layout-0 heuristics; lean prepare on Hopper (FA3 + Marlin present): the decode-only attention / MoE / bf16_proj / glue extensions are not compiled (prefill shapes never reach them; built lazily if a T <= 256 call ever arrives), pf2 + cuBLASLt extensions compiled in parallel threads.
- attention v40 judge 1.581 (v39 1.642): verify B1 sliding 16.0 -> 13.7 (win), B1 full 25.9 -> 28.0, B16 sl 20.5 -> 20.7, B16 full 153 -> 160, B32 sl 24.2 -> 28.5, B32 full 289 -> 350 (+21%). Decode B32 full 192 -> 204 is noise: fdec SASS is byte-identical to v39. vc est predicted the opposite sign on full shapes: the loop changes (copy mapping / QK-first overlap / fast softmax / merge) hurt bandwidth-bound verify; not yet attributed.
- attention v41: v39 verify kernel verbatim (attnw39, var -1) for every shape except verify B1 sliding = v40 config (V15, 64-key splits, merger cap 128). attnw<.., V> template bits: 1 copy, 2 QK-first overlap, 4 fast softmax, 8 prologue overlap; launch-time kps / merger cap / fused-split limit from VCFG[(B, sliding)]. ptxas segfaults on V=6 (skipped). submission_v41diag.py: all variants instantiated (-DAWV_ALL), times each on the first eager call of every scored verify shape (cold L2, CUDA events, median of 20) and prints `VD <shape> V<bits> k<kps> m<mergers> f<fused allowed> <us>`.
- attention v41 judge 1.653 (best). v41diag: v40 loop micro-opts do not help bandwidth-bound verify (v16f V-1 156.1 vs V14 162.8, v32f 287.2 vs 293.2); full shapes sit at the random 128 B-row gather ceiling (~2.7 TB/s).
- Physical-order KV scan for full layers (pending premise check, submission_v42pd.py prints `PD` lines): coverage of the pool by the batch is 0.918 at B16 (1376016 rows, 1262581 live) and 1.000 at B1 in the generator. Full layers have no RoPE, so cached keys are position independent; only the new rows need causal masking (verify). Caveat: the win relies on a densely used pool (one randperm over batch x max_pages); real SGLang pools are sized to free memory, so a scan needs a shape-only switch (page-table entries vs pool rows) and may read as workload-specific in review.
- v41diag full-shape bits (us, v16f / v32f, v39 = 156.2 / 288.8): my rewrite with all bits off 167.1 / 296.5 (slower than v39); bit 1 (2 keys x 32 B sector copy) 167.6 / 375.1 (+27% at B32: memory system, keep 4 keys x 16 B); bit 2 (QK first, loads after) 178.4 / 320.9 (+7-8%: the next step's loads must issue early, verify is load-lead sensitive); bit 4 fast softmax -1.5% / 0; bit 8 prologue overlap -1.5% / -0.6%.
- v42diag: verify ablations are not a clean skeleton (the no-wgmma ablation adds 32 LDS + LOP per thread per step for fake S); v16f: no loads -24 us, no wgmma -11, no V convert -13, no exp -9, no barriers -2. FDIAG decode B1 full: all CTAs merged at 14.0 us, last end 19.1: fdec's cross-CTA merge loops (runtime trip count, not unrolled) wait one L2 round trip per split (16 mergers x ~8 splits x 2 passes).
- attention v43: attnw39<.., X> = v39 verify kernel (X = 0 byte-identical SASS) + grafts: 4 fast softmax, 8 prologue overlap, 16 deep ring (RSW + 1 stages with a single f16 V buffer: same 110 KB, 2 CTAs/SM). fdec<CL, MU>: MU unrolls the cross-CTA merge loops (default on). Verify B1 full default: new kernel, 132 fused splits, 128 mergers (diag 29.3 vs v39 30.0). submission_v43diag.py prints `XD` lines (X grafts per verify shape, MU 0/1 per decode shape).
- attention v43 judge 1.655. v43diag: MU (unrolled decode merge loops) no effect (d1f 24.5 -> 24.3); FDIAG d1f all-in 14.6 -> last-end 19.5 unchanged, so the tail is not the loop trip count. Verify grafts <= 1 us (v1s X8 16.8 vs X0 17.4). X16 (deep ring) slower everywhere, and X24/X28 crashed: BUG, with RS = 5 the page-table smem ring is read one stage before cp.async.wait_group(RS - 2) guarantees it landed (issued RS + 3 ahead, read RS ahead; needs RS + 3 >= 2 RS - 1, so only RS <= 4 is safe). X16 alone likely read stale page-table entries silently. Removed; defaults (RS = 4) are exact.
- attention v44: fdec flags (DFL per decode shape): 1 gate rows prefetched at kernel start into smem (no DRAM trip in the merge tail), 2 seq-independent split ranges for full layers (page-table loads issue before seq lands; d1f only by default), 4 one-pass cross-CTA merge (running max per split group: one L2 round trip and one barrier fewer). Verify merge: first-pass gate values loaded before the spin. FDIAG now stamps the prologue issue and each merge step (pass1, pass2, sync2, norm, gate); VDIAG stamps verify per CTA (start, issued, prologue, loop, partials, ranked, all-in, end).
- attention v44 judge 1.689 (d1f 20.5 -> 17.4). v44diag: d1f tail all-in 13.7 -> pass1 15.9: merge loops over splits with a runtime bound keep a guard per unrolled iteration, and a guarded load is not hoisted above its guard, so each split costs one dependent L2 round trip (also v1f merge all-in 17.7 -> end 22.3, and v39's verify merge: 2 x nsplit serial trips per thread).
- attention v45: mrg8 = chunked one-pass online merge (8 splits' (m, l) + O loads per chunk with clamped indices, no guard before any load) in fdec (fl & 4), the v39 verify kernel and the new kernel (mfl & 1, default on). New-kernel V bit 512: seq-independent split ranges for full layers (v1f default V 520 = prologue overlap + that, 132 fused splits). v1s default moved to the v39 kernel + prologue overlap with 128-key fused splits (v43diag 16.8 vs 17.6). attnx: warp-specialized verify (2 consumer warpgroups + 1 producer warp, 288 threads, 2 CTAs/SM, 96 regs with 12-56 B spills): producer does page-table lookups and the fdec-pattern gather of whole 128 B K/V rows into a 4-slot ring, mbarrier full/emptyR/emptyV handshakes, no warpgroup barriers in the consumer loop; var & 8 producer also converts V (risk: ~500 instr/step on one warp, both CTAs' producer warps may share a scheduler), else consumers convert their head's half. ptxas 12.9 now segfaults on new-kernel V 7/14/15/527 (dropped).
- prefill pf9 judge 1.156 (prepare 48.6 s); pf9b (Marlin bm32 at T753) 1.127: the bm32 synthetic-routing win was the pf8 T753 regression.
- prefill pf10 (local only): (a) Marlin inactive-route skip without a new Marlin build: the pinned `moe_align_block_size` already has `ignore_invalid_expert`; align(lid, bm, 32, ignore_invalid_expert=True) drops the non-local routes and uses 33 instead of 257 buckets, so the pinned is_ep Marlin sees no -1 blocks (its per-CTA serial walk past ~86 / ~340 invalid blocks at T753 / 3012 disappears; the remaining is_ep count loop is a few dozen L1 reads). is_ep=False would need a second JIT Marlin specialization (compile time), not done. Flag ALIGN_LOCAL. (b) FA3 on a contiguous KV copy: kvg kernel gathers each request's visible keys (full: all seq+L, sliding: its 2047+L window, so the sliding copy is ~0.7 MB) into [B, maxk, 2, 64] fp8, FA3 called without page_table (TMA path); prepare() times paged vs gather+contiguous on synthetic caches for (B, L, sliding) in {1,4} x 753 x 80K and 16384 x 32K and adopts contiguous only if >2% faster (flag KVC_TUNE). Emulated-FA3 check: identical errors to the paged path on all 5 tests (incl. the 64-token-page test).
- attention v45 judge 1.711. attnx (warp-specialized verify) lost everywhere: v16f 236-276 vs 155, v32f 454-532 vs 290, v1f 37-50 vs 26; VDIAG: its loop itself is slow (one producer warp cannot feed two consumer warpgroups; producer-converts-V is worse). Dropped in v46 (code stays in submission_v45.py). Picks from v45diag: B16/B32 sliding verify grafts 4+8 (23.7 vs 25.3, 27.8 vs 28.5).
- Diag bug: template launch helpers' function-local statics (attribute set once) merge across two extension modules in one process (vague linkage), so the PROF module launched without its smem attribute ("invalid argument"); helpers are static (internal linkage) in v46.
- attention v46scan (prototype, decode full, page 1, B <= 16): physical-order scan. scan_hist (132 CTAs histogram live page-table entries by owning scan CTA), scan_scatter (prefix sums of the histograms in smem, packed (row offset | request) entries into per-bin runs; also stores the new token's K/V), scan_dec (CTA streams its contiguous pool chunk through a 3-tile cp.async ring, smem row -> request table from its bin, ballot compaction per warp/request, fdec math per 16-row group, CTA merge across a request's warps), scan_merge (one CTA per (request, q head) row, mrg8 over the 132 partials). No memset, no epoch, nothing kept across calls. A spinning last-arriver merge deadlocked on the 3090 (fewer resident CTAs than the grid; the merger cap patch never applied), hence the separate merge kernel. submission_v46scandiag.py prints `SD` lines (gather vs scan stages) on d1f / d16f.
- attention v46 judge 1.694 (noise vs v45 1.711). Physical-order scan prototype on the H200: d16f gather 113.3 vs scan 272.0 (hist 13.1, hist+scatter 39.7, scan_dec 230.7, merge 10.7); d1f 21.3 vs 33.5; outputs match. scan_dec is ~2x slower than the gather (per-tile, per-request ballot compaction + one warp per request + 16-row mma groups with smem row gathers); the map build alone (~40 us) eats a third of the gain the stream promised. Shelved (track B stopped).
- VDIAG v46 defaults: v1s prolog 3.1/4.1, loop 6.1/6.8, all-in 8.3, end 11.4 (3 us merge tail with chunked loads already on); v1f issued 2.4, prolog <= 4.5, loop 13.5/14.8, all-in 16.3, end 19.2.
- attention v47: FA4-style exp2 offload on the v39 verify kernel (x39 bits 64/128/192 = 1/4, 1/2, 3/4 of each thread's 32 scores via ex2p on the FMA pipe: clamp at -125, Cody-Waite split at the nearest integer with the 1.5 * 2^23 trick, degree-3 minimax on [-0.5, 0.5] (rel err 1.4e-4), integer part shifted into the exponent). P is already packed with one cvt per 2 scores and masking already runs only on boundary steps. Expectation from instruction counts: each offloaded exp costs ~8 issue slots instead of 1 MUFU issue, so it only helps if the MUFU pipe, not issue, is the limiter. submission_v47diag.py prints `ED` lines (exp split x {plain, grafts 4+8}) for every verify shape.
- v47diag (H200): exp2 offload loses everywhere: v16f X0 155.8, X64 164.7, X128 177.2, X192 187.7; v32f 294.7 -> 310 / 336 / 353; small shapes +0.5-1.5 us. The verify loop is issue-bound, not MUFU-bound. Attention stays at v45 (judge 1.711).
- Attention status (end of this round, best v45 1.711): decode full B16/B32 at the random 128 B-row gather ceiling (~2.9-3.2 TB/s; the physical scan that could stream at 4.25 TB/s lost to its per-row request bookkeeping); decode B1 latency-bound (d1f data phase near stream rate, ~2.4 us merge tail left); verify full issue-bound in the step loop (~0.6 IPC at 4 warpgroups/SM): 2-key copy, QK-first overlap, deep ring, producer-warp specialization, fast softmax, prologue overlap, chunked merge and exp offload all lost or gave <= 2%; what remains would need a structural change (more q rows per K/V tile, e.g. several 8-token blocks or both KV heads per warpgroup in larger CTAs). Verify B1/sliding keep 2-3 us merge tails.
- Tooling left for the next round: bench/emu_chk.py (wgmma emulation on sm_86, --vovr/--ovr/--dfl/--kind/--l1), VDIAG/FDIAG globaltimer stamps (PROF builds), submission_v4xdiag.py pattern (graph-replay timings with L2 flush on the first eager call, printed lines for okc check).
- bf16_proj v21 (H100 pod 20:01-20:03 UTC, ~3 min; results in bench/wgb_h100_run{1,2}.txt): bench/wgb.py correct everywhere (router err 2.6-3.4e-6, one-hot exact; bf16 1.7e-3). Router (vs a cuBLAS N=512 bf16 GEMM + add): m8/16/32 7.7/7.9/8.1 vs 14.9-16.0; m128 9.9 (64.6.16.1.2) vs 18.8; m256 12.4 (64.6.8.1.2) vs 18.6; 128-CTA cluster-16 configs at m256 ~20 (cluster packing). qkv / o via wgb lose everywhere (qkv m128 21.7 vs 17.0, o m128 22.5 vs 7.8): plain-BF16 W through TMA with x evict_last does not fix the per-row-block x re-read at these tile sizes. v21: wgb router folded in as prepare()-timed candidates for router m8/16/32/128/256 (body compiled only for sm_90a via __CUDA_ARCH_FEAT_SM90_ALL, so a compute_90 pass yields an empty kernel that fails the exactness check); the three extensions now build concurrently in threads (3090 cold prepare 263 -> 128 s).
- bf16_proj v22 (H100 pod 01:13-01:26 UTC 10-26, ~13 min, ~$0.9; bench/wgr_h100_run.txt; judge pending): (a) router m128/256 `wgr`: wgb's TMA + wgmma main loop without a cluster, split-K over S plain CTAs (grid <= SMs x occupancy, checked on the host since CTAs spin), fp32 partials to a global workspace (st.cg), one fence.acq_rel + atom per CTA on a per-tile monotonic counter (target = next multiple of S: no reset), then every CTA reduces TM/S tokens of the tile from L2 with all S/G partial loads in flight (G thread groups + smem). H100 prepare us: m128 (BN 32, ring 8, S 16, 2 WG) 6.98 vs v21 wb 9.50; m256 (64, 6, 16, 2 WG, 2 token blocks) 9.12 vs 11.84. Big BN (128) or 4-WG / S 32-64 configs lose (10.9-16.4): more partial traffic. (b) gv V bit 4 = push split-K reduction (competitor style): leader CTA inits an mbarrier + expect_tx((S-1)*NI*16) and every CTA does barrier.cluster.arrive at kernel start; non-leaders wait on the cluster barrier only after their K loop, st.async their WK-reduced partial into the leader's smem (mbarrier complete_tx) and exit; leader sums and emits. V bit 8 = no min-blocks launch bound (MT 4 x-mode-2 kernels spilled 16-32 B stores / 112-168 B loads under the 128-reg cap; uncapped 138-145 regs). H100: qkv m32 (1,4,16,2) V13 (push + uncapped) 12.54 vs v21 pick 15.23 (push alone 14.02, uncapped alone 13.66); qkv m16 (1,4,8,4) V5 11.74 vs 12.16; router m8/16 (FMT 2, S 8) push 3.81/4.00 vs 4.96/5.28; router m32 push no gain. o m32: push / uncapped / 512-thread / ring-4 variants all lose (6.9-11.3 vs 6.2-6.3), o m32 gap to the competitor (6.70 vs 5.81 judge) still open. okb paired H100 (incl ~5 us graph cost) v22/v21: d32 qkv 18.3/20.2, v16 router 12.5/15.0, v32 router 14.7/16.9, d16 router 9.8/10.5, v1 router 9.3/10.4, geomean x1.024. Estimate from v21 judge x H100 deltas: router m128 ~7.1 (comp 6.19), m256 ~8.1 (8.44), qkv m32 ~11.5 (12.16), qkv m16 ~10.2, router m8 ~4.0, m16 ~3.7 => judge ~1.46-1.47 if the H200 deltas match.
## 2026-10-25 MoE CTA-ring redesign (v21-v27): judge 0.74-0.88 vs v17 1.418, not pursued as is
- Design: 132 CTAs x (1 producer warp + 8 consumer warps); CTA ring of 4 x 53 KB stages (16 tiles x 256-K NVFP4 by TMA bulk + the unit's activation rows by cp.async, shared by all consumers from smem); units (W13|W2, expert, 256-row block, K part, 32-token pass) from one queue; K split at M<=64 via fp32 red.add into a self-cleaning workspace (last-arriving warp finishes, check deferred one stage); no per-tile fences. v23/v24: prefill-scale (global routing lists via 2 grid barriers, compact rows, TP=64). v24/v25/v26: wgmma RS (dequantized A in registers, SW128 K-major activation tile as B). All correct locally and on the judge.
- Judge (us, v17 in parens): v22 0.883: bs1 61.9 (52.4), bs16 272.0 (136.2), bs32 382.0 (174.9), bs8 145.8 (105.5), bs128 497.1 (302.2), bs256 573.5 (424.4). v27 (v22 + v23 for M>64) 0.869: 63.2/259.2/389.3/149.7/505.3/621.2. v26 (wgmma) 0.743: 71.4/304.6/435.0/172.5/671.5/737.4.
- Root cause (H100 PROF, bench/prof21.py, logs in muse_moe_w4a16/bench/logs/): the producer blocked on the W13->W2 dependency. LAG=1 queue order (W2 of expert a right after W13 of a+1) handed out W2 units while W13 of a was still in flight in the same round; the producer spins with the ring empty. Median producer dep-wait 114/211 us at bs16, 161/310 at bs32, 246/520 at bs128. Not the producer issue rate, ring depth, TMA or consumer count (v22n16 = 16 consumer warps: same; cpb2: slightly better).
- With LAG=64 (all W13 before any W2, as v17), H100 vs v17: bs128 258.6 vs 324.4 (1.26x), bs256 345.6 vs 542.9 (1.57x); small bs still lose (bs1 0.80, bs8 0.77, bs16 0.65, bs32 0.77): coarse-unit tails (bs16 consumers done p10 68 / max 162 us) and K-split atomics; finer K split (16/8/4) is worse. Ablations on LAG=1 v22: NOCOMP 1.5x v17 at bs256, FAKEDQ 1.16x, NOMMA ~0.
- wgmma RS (v24/v25/v26) was worse than mma.sync everywhere (per-k16 fence/commit/wait, descriptor ALU); v24 prefill T=753 897 vs v23 754 us (LAG=1).
- v29 = v28 (v17 per-warp kernel, rotation-free dequant: 4 of 5 shifts as IMAD) for M <= 64 + v22 with LAG=64 for M > 64, one weight layout. H100 vs v17: bs1 1.034, bs8 1.027, bs16 1.035, bs32 1.011, bs128 1.269, bs256 1.572 (est judge ~1.62). v28 alone ~+1%.
- Prefill MoE (v23 = global lists + TP=64, LAG=64, H100): T=753 521 us (1088 local routes; Marlin on the H200 judge 644-730), T=3012 1427 us (Marlin 2093); T=16384 7392 us with LAG=1 (current 16K MoE ~8.1 ms on H200).
- virtualcuda missed by 3x because it has no queue-order/dependency model: any persistent MoE design needs the dependency-wait term.
- prefill pf10 judge 1.153 (flat vs pf9 1.156).
- prefill pf11 (local only): composes the MoE agent's prefill-scale fused W4A16 kernel (muse_moe_w4a16/submission_v23.py as Q_, any M; v29's M>64 path is capped at M*8 <= 4096, i.e. M <= 512, so it cannot serve T >= 753). Q_prepare (repack + compile) runs in the prepare() thread pool. prepare() times, per T in (753, 3012, 16384), rmsg + v23 against rmsg + Marlin (turn sizes) or dequant + grouped GEMM (16K) on skewed synthetic routes (42% of tokens on one local expert, ~0.75 local routes/token, the shape of the scored soft routing) and picks v23 only if >5% faster and within 2e-2 of the default path; other T use the nearest tuned size. 3090: v23 path passes all 5 tests and both turn-1 loads (errors slightly lower than the dequant path).
## 2026-10-26 MoE v29-v33 (judge) and the small-M floor
- v29 (v28 for M<=64 + ring with all-W13-first for M>64): 1.543. Judge sgl/ours us: bs1 77.2/51.1, bs16 186.5/125.7, bs32 209.3/177.0, bs8 141.6/95.4, bs128 457.5/250.8, bs256 623.8/331.1.
- v30 = v29 + ring lookahead-1 queue fetch (NX1) + prepare()-time autotune per M bucket (v17 path with KSF auto/1/4, ring ks 1/2/4/8 x NX1; synthetic busiest-GPU routings, graph replays, cold L2; default kept unless >2% better): **1.558, the submission**. bs1 52.5, bs16 128.4, bs32 163.6 (1.27x, was 1.18), bs8 100.4, bs128 259.0, bs256 337.2. Tune (H200, us): M1 v0 57 v4 56 r2n 55 r4n 55; M8 v0 105 r1n 100; M16 v0 118 r1n 114; M32 v0 184 r1 179; M128 r1 261; M256 r1 369; M512 r1 499.
- v31 (+ ring "h": 2 CTAs/SM x 16 consumer warps, 32 dequant warps/SM): 1.580 within noise of v30, h never picked (M1 r2h 78, r4h 62 vs 55; M8 r1h 126 vs 97; M16 138 vs 111): more warps per SM does not help.
- v32 (ring W13->W2 dependency per 512-column chunk of inter, checked per W2 stage, W2 units K-part-major): 1.522; ring slower at every M (M1 r2n 60 vs 55, M128 269 vs 261, M512 511 vs 499). Reverted.
- v33 (v17 path: no group barrier, last arriving warp finishes a K-split tile, 3 rotating smem partial buffers, tagged 8-deep tile-id ring, cached W2 dependency, early queue heads, combine routes from smem): 1.559 flat; v17 path itself regressed (M1 63 vs 56, M16 138 vs 118; spill loads 104 -> 192 B) and the tune switched to the ring. Reverted.
- v30d MD (H200, us; full / W13 only / W2 only / skeleton = all tiles without weight TMA or math / prologue+combine with no tiles / empty kernel / ring ks2+NX1):
  M1 56.3/25.8/27.7/25.7/10.6/4.7/49.5; M2 62.5/38.5/31.1/25.5/10.8/4.8/65.5; M4 68.9/39.9/34.8/25.7/11.1/4.7/77.6; M8 98.7/62.8/47.7/32.2/11.6/4.8/108.1; M16 124.0/79.4/55.3/38.0/11.6/4.7/149.1; M32 174.9/104.4/75.0/42.4/12.9/4.7/208.1 (empty graph 2.9). v33 MD M1: 59.4/27.8/30.4/28.6.
- What the M1 floor (~50-56 us vs 11.4 SOL) is made of: ~5 us launch + graph; ~6 us routing prologue + final combine (DRAM-cold ids/tw/x, queue atomics, completion poll, y/ss reads); ~15 us per-tile machinery that does not need weight bytes (queue, dependency bookkeeping, K-split partial exchange, epilogue stores; ~25.7 skeleton); and two fully serial phases (W13 alone 25.8 + W2 alone 27.7 ~ full), each latency-bound at ~1.3-3 TB/s with only 2-3 experts active. H200 DIAG bs1: per warp ~8 us per K-split tile (2 stages + epilogue), TMA wait 0.7 us, dep wait 0.9/4.7 us.
- Not yet tried: (a) prologue trim: compute queue order/ks from counts before the list build and start the first W13 TMA loads before the lists exist; keep ids/tw from the prologue in smem for the combine (the v33 attempt bundled this with the barrier removal, so its own effect is unknown); (b) fuse the combine into the W2 epilogue for M <= 4 (per-route norm known after the expert's last W2 tile: last finisher scales y into fp32 atomics on out, last expert per token converts) to drop the final grid-wide wait; (c) W2 of expert a overlapping W13 of expert b at M1 with only 2-3 experts: needs finer W13->W2 deps that cost less than v32's per-stage checks (per-unit check of the K range, no queue reorder); (d) PDL / early launch so the routing prologue overlaps the previous kernel's tail in decode_layer.
## 2026-10-26 decode_layer v20 (bf16_proj v22 composed), H100 pod ~10 min
- tools/build_dl20.py = build_dl15 b + bf16_proj v22: wb ext gets PDL (W ring stages issued before griddepcontrol.wait, x after; wgb triggers after its main loop, wgr only after its counter wait so dependents cannot take SMs from a not-yet-resident wgr CTA); router on the side stream gets its own gv/wgr workspace + counters (st["p16r"]); gv/wgr router candidates at T128/256 in _rcands; glue19 P_GVC + v22 qkv m16/m32 push+uncapped and router push variants.
- H100 (bench/dl20_h100_run.txt, okb paired vs v17): correct, graph stress ok, geomean x1.000 (decode32 sliding 0.98, decode16 full 1.008, rest +-0.5%). Standalone prepare picks: qkv m32 V13 12.45 vs v12 14.85, qkv m16 11.46 vs 12.54, router m8/16 push 4.06/4.26 vs 5.09/5.18, wgr m128 8.70 / m256 10.43 vs v12 14.3 / 21.5. But in the layer's overlapped timing (router on the side stream next to the pooler) lt wins: T128 lt 35.5 vs wgr 38.5, T256 53.6 vs 59.7: wgr needs all 128 CTAs resident while the pooler GEMM holds SMs. Router m<=32 is hidden behind the pooler anyway, so v22's router wins buy nothing here; qkv m16/m32 savings (~1-2.4 us) are within noise. Not worth a judge check on its own.
