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
