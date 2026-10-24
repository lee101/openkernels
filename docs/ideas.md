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
