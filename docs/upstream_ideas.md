# Upstream ideas (SGLang 5971f47, vLLM 444d691, DeepInfer) for the Muse problems on H200

Sources: the pinned judge SGLang `okc-serve/sglang` (a55ab7d), `sglang-latest`, `vllm-latest`, `DeepInfer`
(including vendor/DeepGEMM and FlashMLA), and the `humming-kernels==0.1.18` wheel, extracted to the scratchpad and read only.
Nothing here was run. Every µs figure is an estimate, to be checked with `okc check`.
Items that ideas.md already records as failed are excluded or flagged.

## TL;DR ranking

| # | idea | problem / shape | in pinned SGLang? | est. gain |
|---|---|---|---|---|
| 1 | Humming-style W4A16 kernel: wgmma RS with dequantized weights in registers as A and tokens as N=8..64; mma.sync m8 path for ≤8-row expert tiles | moe bs16-256, prefill T=753/3012 MoE (replaces Marlin), decode_layer MoE | no (vLLM ranks Humming above Marlin on sm90) | prefill turn loads -300..-450 µs at T=753 (Marlin 730 µs at ~1.1 TB/s); moe bs128/256 -100..-250 µs |
| 2 | Drop non-local routes before Marlin's align and call Marlin with `is_ep=False` | prefill T=753/3012 (Marlin path) | **no**: pinned Marlin scans the -1 blocks serially per CTA; latest SGLang fixed it, so this is a free win against the pinned baseline | 5-20 µs per layer at T=753, more at T=3012 |
| 3 | PDL on every kernel of the decode layer: independent prologue (weight/page-table prefetch) before `griddepcontrol.wait`, early `launch_dependents` | decode_layer (all loads, mostly bs1/16) | no (FA3, cuBLAS and Marlin do not chain) | 1.5-3 µs per boundary x 8-10 = 10-20 µs at bs1 (~10%) |
| 4 | One-pass FP32 router at m=128/256: hi + lo/256 into one fp32 accumulator, split-K with a self-cleaning flag | bf16_proj router m128/256; decode_layer verify | no (HPC-Ops path exists, opt-in, needs the `hpc` pip package) | 10.3/11.1 -> ~5-6 µs |
| 5 | Prefill FA3 on gathered contiguous KV (TMA path) instead of page-1 cp.async gathers | prefill full layers T=753 (FA3 419 µs), 16K first turn | no | -60..-150 µs per full T=753 layer; measure first |
| 6 | DeepGEMM sm90 `m_grouped_bf16_gemm_nt_contiguous` vs torch `_grouped_mm` for the dequantized 16K MoE | prefill 16K (grouped GEMM 4.3 ms of 11.3 ms) | not used by Muse | 0-15% of 4.3 ms |
| 7 | vLLM CuTe skinny GEMM (M=1..16, H200 tables incl. N=2304) as a reference against our `gv` at m=16 | bf16_proj qkv/o m16 | no | 0-1.5 µs per shape |

Negative or no-op findings are listed at the end. Read them before building items 1, 4 or 5.

---

## 1. Humming W4A16 MoE design (biggest MoE and prefill lever)

What: `vllm-project/humming` (Apache-2.0) is a JIT CUDA GEMM library: headers plus an NVRTC launcher.
On sm90 vLLM puts it ahead of Marlin for NVFP4 W4A16 MoE:
- `vllm-latest/vllm/model_executor/layers/quantization/utils/humming/priority.py`: `_HUMMING_PREFERRED_CAPABILITIES = {90}`.
- `vllm-latest/vllm/model_executor/layers/fused_moe/oracle/nvfp4.py:425`: the `NvFp4MoeBackend.HUMMING` backend.
- `vllm-latest/vllm/model_executor/layers/fused_moe/experts/fused_humming_moe.py`: INDEXED, GROUPED_CONTIGUOUS and GROUPED_MASKED GEMM types.

The source is only in the wheel; `pip download humming-kernels==0.1.18` fetches it. The files that matter:
- `humming/include/humming/mma/wgmma.cuh` (`WGMMA::issue`): calls `MmaOpClass::fma(desc, regs_b[..], regs_c, scale_d)`.
  The activation tile is the smem descriptor; the dequantized weight fragments are the register operand.
  So the output-feature dimension is wgmma M=64 and tokens are wgmma N (8..64). Small expert tiles need no M padding.
  This is the "wgmma RS with A = dequantized weights" idea in ideas.md (2026-10-24 evening). It is not the
  dequant-to-smem `w4wg2` that failed.
- `humming/tune/sm90_policies.py`:
  - `select_sm90_small_m_schedule`, expert tiles of ≤8 rows: `mma.sync`, block (8, 128, 64), 4 CTAs/SM, 4 stages,
    stream-K, no TMA, no warp specialization.
  - 16-32-row tiles: still `mma.sync` with stream-K when the grid is short ("removed producer and TMA overhead from a
    short expert grid").
  - `get_compact_moe_config` (WNA16 grouped): WGMMA, 2 CTAs/SM, warp specialization and TMA dropped, ≥6 stages.
  - Warp specialization and TMA only for block_m 32-64 with a narrow N.
  - This matches our measurement that the TMA-ring plus producer-warp redesign was 1.2-2.2x slower: Humming also drops
    TMA and the producer warp for short expert grids.
- `humming/include/humming/kernel/humming_ws.cuh`, `scheduler.cuh` (stream-K / raster), `epilogue/smem_reducer.cuh`.
- `humming/include/humming/datatype/dequant_fused.cuh`: the fused dequant is for MXFP4 only (the e8m0 scale is
  added into the exponent with one prmt). NVFP4 still needs a multiply, so expect no ALU gain over our ~15 ops / 8 values.

Targets:
- **Prefill T=753/3012.** Marlin is 729.5 µs for the two GEMMs at T=753 (67% of the layer). The 32 local experts are
  ~805 MB of NVFP4, so that is ~1.1 TB/s. A weight-streaming kernel at ~3 TB/s would be ~270 µs. Turn loads are 1.01x
  today, so this is the single largest prefill item; at T=3012, Marlin's 2056 µs scales the same way.
- **moe bs16-256** and **decode_layer MoE** (45-51% of decode_layer time).

Plan: port the schedule choices, not the library:
1. m8 `mma.sync` tiles with 4 CTAs/SM and stream-K for bs≤32.
2. The WGMMA RS path (weights dequantized straight into A-fragment registers, x tile in SW128 smem as B, n8/n16/n32
   per expert tile) for bs128/256 and prefill.

Read their tune tables for block_k / stages per (N, K) before sweeping.

## 2. Marlin inactive-route skip (free win for the prefill Marlin path)

Our glue (`muse_prefill_layer/glue_pf2.py`) and the baseline both do two things:
- `align(lid, bm, 256)`, where -1 routes go to bucket 0 and become a prefix of `expert_id == -1` blocks.
- Marlin with `is_ep=True`.

In the pinned kernel (`okc-serve/sglang/python/sglang/kernels/jit/csrc/gemm/marlin_moe/marlin_template.h`):
- Every thread first loops over all `parallel` block ids to count the -1 blocks.
- Each CTA's first `update_next_moe_block_data` then walks `expert_ids` serially, with a data-dependent `break`,
  past every -1 block to find its `par_id`-th valid block.

At T=753 with bm=64 that is ~86 invalid blocks (~0.73 local routes/token); at T=3012 it is ~340.

The fix is upstream:
- **SGLang latest** (`sglang-latest/.../marlin_moe/marlin_template.h`, `moe_wna16_marlin.cuh`, `kernels/jit/csrc/moe/moe_align_kernel.cu`):
  - New template flags `kExpertIdsAreSorted` and `kHasInactivePrefixCount`.
  - The aligner writes an `int2 [padded, inactive_prefix_blocks]`, and Marlin skips the prefix in O(1) (binary
    search when only sorted).
  - `sglang-latest/.../layers/muse_spark/muse_spark_fp4_moe.py:_marlin_expert_outputs` maps to local ids first, aligns
    over `num_local_experts`, and passes `is_ep=False, expert_ids_are_sorted=True, has_inactive_prefix_count=True`.
- **vLLM**: `moe_align_block_size(..., expert_map, ignore_invalid_experts=True)`
  (`vllm-latest/vllm/model_executor/layers/fused_moe/moe_align_block_size.py`). Its Marlin
  (`vllm-latest/csrc/moe/marlin_moe_wna16/marlin_template.h`) has no EP scan at all.

To do it with the pinned JIT Marlin (no vendoring needed):
- Have `route2` (which already emits per-expert counts) write a block-padded `sorted_token_ids` / `expert_ids` /
  `npad` that contains only local routes, over 32 buckets.
- Call `moe_wna16_marlin_gemm(..., is_ep=False)`.
- Non-local rows of c1/c3 are never written; `silu_and_mul(expert_ids=lid)` and our `comb` already skip `lid < 0`.

Gains:
- The scan: ~2-10 µs per GEMM.
- The aligner: 1/8 of the routes and 32 instead of 256 buckets (a few µs, and it may move off the side stream).

Risk is low. Verify bitwise against the current path.

## 3. PDL across the decode layer (DeepInfer-style)

DeepInfer enables PDL globally: `deepinfer/worker/runtime_config.py:65-66` calls `deep_gemm.set_pdl(True)` and
`tile_kernels.set_pdl(True)`. SGLang's JIT kernels take `is_arch_support_pdl()`; examples:
- `sglang-latest/python/sglang/kernels/ops/gemm/hopper_bf16_gemv.py`
- `muse_spark_router_gemv_cutedsl.py` (`griddepcontrol_wait` then `launch_dependents` at entry)
- the Muse norm and routing JIT kernels

vLLM's `csrc/fp32_router_gemm.cu` fires `cudaTriggerProgrammaticLaunchCompletion()` right after its own
`cudaGridDependencySynchronize()`. A consumer that doesn't read its output (there, the NVFP4 quant) then overlaps the
whole kernel.

We use PDL only for skinny+fin in bf16_proj. The decode layer is ~10 kernels per step (profile v13, bs1 sliding:
moe 53.5, gv x4 33.5, resid x2 9.0, fdec 8.1, route 6.6, rr 4.1, rmsg 2.8 µs) against a 40 µs floor.
Prologues worth hoisting above `griddepcontrol.wait`:
- `gv`: the first weight stages, which are input-independent. This is a different case from the attention `PFF`
  prefetch that hurt: here it only fills the previous kernel's tail.
- `fdec` / attnw: `seq` / page-table loads (3 dependent DRAM trips, 3.2 µs prologue measured).
- The moe kernel's expert-queue init.
- `resid` / `rmsg`: weight vectors.

Launch every kernel with `cudaLaunchAttributeProgrammaticStreamSerialization`, as `muse_bf16_proj` already does.
Graph capture keeps the programmatic edges. Leave cuBLAS-served shapes (verify m≥128) alone.

Estimate: 10-20 µs at bs1/bs16, worth ~5-10% on decode_layer. The same trick helps prefill glue only marginally.

## 4. FP32 router at m=128/256 in one pass

The reference design is SGLang's HPC-Ops bf16xfp32 path:
- `sglang-latest/python/sglang/kernels/ops/gemm/bf16_fp32.py` (also present in pinned, opt-in via
  `SGLANG_OPT_BF16_FP32_GEMM_ALGO=hpc`, needs Tencent `hpc-ops`, sm90a).
- The weight is split once into `w_hi = bf16(w)` and `w_lo = bf16((w - w_hi) * 256)`.
- `hpc.gemm_bf16xfp32(x, w_hi, w_lo, 1/256, use_fp32_output=True, use_splitk=True, split_flag=ws)` is one kernel:
  both B tiles accumulate into one fp32 accumulator, and a split-K flag workspace is left zeroed by the reducer.

Alternative: `SGLANG_OPT_BF16_FP32_GEMM_ALGO=deep_gemm` routes to `deep_gemm.bf16_gemm_nt` with fp32 D.
- DeepGEMM sm90 BF16 accepts `c` (accumulate), so hi then lo with c=d is two launches and no add.
- Caveat: in DeepInfer's vendored DeepGEMM (`csrc/apis/gemm.hpp:520`), `bf16_gemm_nt` goes to cuBLASLt whenever it
  is available and the epilogue is plain. Check the judge's sgl-deep-gemm 0.2.1.post1 before expecting anything.

Ours: 10.32 / 11.06 µs (one cuBLAS N=512 GEMM plus add) against a floor of 2.2 / 2.7.
bench/wgb.py already plans hi + lo as two B tiles into one accumulator with DSMEM split-K; HPC-Ops confirms the shape
of that design. Target ~5-6 µs, taking router m128/256 from ~1.58x to ~3x.

## 5. Prefill attention: gather KV contiguous, then FA3 with TMA

FA3 cannot use TMA for page-size-1 KV and falls back to cp.async gathers. At T=753, full layer:
- FA3 takes 419 µs for ~246 GFLOP (~590 TFLOP/s with an FP8 Q/K/V path; FP8 peak is ~1979).
- Gathering each request's 80K live rows into a contiguous buffer is ~20 MB per request, ~8 µs at the measured
  2.9 TB/s for 128 B rows.
- Then call FA3 varlen without a page table (or with page = kBlockN), so it takes the TMA path.

Upstream analogues: vLLM `gather_and_maybe_dequant_cache` / `cp_gather_cache` (`vllm-latest/csrc/cache_kernels.cu`,
`vllm/_custom_ops.py:3067`), used for MLA chunked prefill.

First test: time FA3 on a contiguous copy against page-1 at T=753 and T=3012 before writing anything.
For the 16K first-turn loads, all KV is new, so write it contiguous in `qkprep` and run FA3 on that.
Not applicable to decode/verify: the gather would cost as much as the attention.

## 6. DeepGEMM sm90 grouped BF16 for the 16K MoE

`deep_gemm.m_grouped_bf16_gemm_nt_contiguous` is sm90-capable (`DeepInfer/vendor/DeepGEMM/csrc/apis/gemm.hpp:567`);
SGLang uses it in `sglang-latest/python/sglang/srt/layers/moe/moe_runner/deep_gemm.py:670`. The judge image ships
DeepGEMM.

It needs per-expert row blocks padded to block_m (128) plus an m_indices vector (-1 on pad rows); our `offs` layout
gives that cheaply.
- Compare against the `at::cuda::detail` cutlass `_grouped_mm` (4.30 ms, 2 calls, at 16K).
- DeepGEMM sm90 heuristics: block_m 64/128 (`jit_kernels/heuristics/sm90.hpp:18-36`), persistent, TMA multicast,
  no split-K.

Expected 0-15%. Do it after 1.

## 7. Small-M BF16 GEMV references (low)

- **vLLM CuTe DSL skinny GEMM**, `vllm-latest/vllm/model_executor/kernels/linear/cute_dsl/{skinny_gemm.py,_skinny_gemm.py}`:
  M=1..16 fully unrolled, 2-tile prefetch, PDL. The H200 table is `KIMI_K3_PROJECTIONS_SM90` in
  `vllm/models/kimi_k3/nvidia/low_latency_gemm.py` (~L535, has N=2304). Benchmark it against `gv` at m=16
  (qkv 10.97 vs floor 7.93, o 5.60 vs 3.56).
- **vLLM `csrc/dsv3_fused_a_gemm.cu`** (TRT-LLM): swap-AB `mma.sync`, 128 load threads plus 128 MMA threads,
  4-warp split-K, PDL; M≤16. A load/MMA warp split is the one structural lever left for qkv/o m=32, which ties
  cuBLAS today.
- **vLLM `csrc/fp32_router_gemm.cu`**: exact (H=8192, E=256) instance for M≤48, `kTGroups` token groups re-reading
  weights from L1. Our router m≤32 is already 2.5-5x, so this is a check only.
- **Not useful**:
  - SGLang `hopper_bf16_gemv` is m=1 only, opt-in, not the judge default; ours already beats it at ~3.9 TB/s.
  - SGLang `router_gemv_cutedsl` is gated to SM100.
  - DeepGEMM sm90 BF16 has no split-K and small-M tiles of only 16/32. At m=32 and N=2304 it fills ~18 CTAs.

---

## Negative findings: don't build these

- **Per-row `cp.async.bulk` gathers for page-1 attention.**
  `sglang-latest/python/sglang/kernels/jit/csrc/kvcacheio/hicache_tma.cuh` (header comment): "the TMA unit processes
  bulk ops at a fixed per-op rate, so a source run must move as one op ... never one op per row".
  A producer warp issuing one 128 B bulk op per K/V row will hit the TMA op rate, not help issue-bound verify.
  Keep 16 B cp.async, or bulk copies only for runs.
- **Nothing upstream for page-size-1 FP8 decode/verify on Hopper.**
  - FlashInfer XQA (the sm90 FP8 default in vLLM) needs page 16/32/64 (`vllm-latest/vllm/utils/flashinfer.py` ~L690).
  - TRITON_ATTN needs a block size that is a multiple of 16.
  - HPC_ATTN is head_dim 128 / page 64.
  - FlashMLA dropped Hopper; MLA only.
  - TMA gather4 is SM100.
  - FA3 (the baseline) is unchanged in Muse's call path.
  The attention gap to the leader has to come from our own kernel.
- **Mega-MoE / mega-gate / mega-mHC** (DeepGEMM `bf16_mega_moe`, `fp8_fp4_mega_moe`, `bf16_mega_gate`, `mega_mhc`):
  `arch_major == 10` only.
  - `sglang-latest/.../layers/muse_spark/muse_spark_megamoe.py` needs SM100 plus `docker/muse_spark/deep_gemm.patch`.
  - SGLang's sm90 FP8 mega-MoE (`srt/layers/moe/mega_moe_sm90.py`, also in pinned) needs FP8 weights and
    `deep_gemm.fp8_mega_moe`; not applicable to NVFP4.
- **DeepInfer itself**: its CUDA work is DeepGEMM / FlashMLA / DeepEP / TileKernels plus TileLang ops (`deepinfer/model_executor/layers/tilelang_ops/cuda/*.py`, tested on SM100 only).
  Neither SGLang nor vLLM imports `deepinfer` or `tile_kernels`. What they share is DeepGEMM (BF16/FP32 GEMM,
  `tf32_hc_prenorm_gemm` sm90), DeepSelect (SGLang JITs it for 9.0) and PDL everywhere. For us only PDL (item 3) and
  the DeepGEMM sm90 grouped BF16 (item 6) carry over.
- **FlashInfer CUTLASS fused MoE on sm90**: W4A16 only for MXFP4 (e8m0, group 32;
  `vllm-latest/vllm/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py`). NVFP4 group-16 e4m3 scales
  cannot be represented losslessly, and its fused finalize has no per-expert post-norm.
  triton_kernels `matmul_ogs` is also MXFP4 only.
- **Marlin's NVFP4 dequant** (`vllm-latest/csrc/quantization/marlin/dequant.h` ~L434: mask, shift by 6, 2^126 bias
  folded into the scale) is ~the same op count as ours. No gain.
- **Fused QK-norm + RoPE + FP8 KV insert** (vLLM `csrc/fused_muse_spark_qknorm_rope_kv_insert_kernel.cu`,
  `csrc/fused_qknorm_rope_cache_kernel.cu`; SGLang `kernels/ops/attention/muse_spark_qk_norm_rope_store.py`):
  we already fuse this into the attention prologue and `qkprep`.

## Upstream changes to the Muse Spark path since a55ab7d

Muse ops, file by file (all checked against pinned):
- **Kernel files moved, logic unchanged.** `kernels/ops/muse_spark_v12/*` was split into
  `kernels/ops/{attention,layernorm,moe,gemm,communication}/muse_spark_*.py`, and `kernels/jit/csrc/muse_spark_v12/`
  became `.../muse_spark/`. Byte-identical: `router_gemm.cuh`, `norm_gate.cuh`, `norm_residual.cuh`,
  `all_gather*.cuh`. `routing.cuh` and `row_sum.cuh` differ in comments only.
- **`norm_fp4_quant.cuh`**: new static-global-scale mode (`fused_norm_fp4_quant_static`) for the SM100 NVFP4
  activation path. Not used on H200.
- **Helion megakernels removed**: lm-head verify, TP4 MoE verify and attention verify megakernels, plus
  `muse_spark_expert_tp.py`. They were TP4 / Helion-only; the judge baseline never used them.
- **Router**: `MuseSpark1xRouter` adds `router_gemv_cute` (CuTe DSL FP32 GEMV, T≤8, all weight loads issued before
  the first FMA, 1-2 experts per CTA, butterfly plus fixed-order smem reduce, PDL). Gated to SM100 by
  `_is_sm100_family`, so H200 is unchanged.
- **NVFP4 Marlin MoE** (`layers/muse_spark/muse_spark_fp4_moe.py`): local-id remap before align, align over local
  experts with an inactive-prefix count, Marlin sorted/prefix specialisation (item 2). The finalize gains
  `slot_expert_ids` / `round_product` (`kernels/ops/moe/muse_spark_post_expert_finalize_triton.py`). The Marlin
  output buffer is now `torch.zeros` when the kernel allocates it.
- **`moe_align`**: sentinel bucket for invalid ids, optional `int2` count output, `-1` padding of the expert-id tail.
- **Model** (`models/muse_spark.py`):
  - `MuseSpark1xForCausalLM` was renamed `MuseSparkTextForCausalLM`.
  - `_fused_norm_boundary` helper (residual-gate flag); `all_gather_norm_residual` called directly.
  - A unified-memory KV-pool fallback.
  - `o_proj` moved back inside attention forward.
  - The ROCm (`rocm_muse_spark`) entry is gone.
- **`unquant.py`**: the Hopper GEMV backend now also accepts sm12x, adds 16-byte alignment guards and a PDL arg, and
  excludes K=5120/N≤512. The backend is still opt-in, so H200 dense projections stay on cuBLAS in both trees.
- **Dependency bumps**: `sglang-kernel` 0.4.9 -> 0.4.10 (FA3 lives here; no source in either tree to diff),
  `flashinfer` 0.7.0.post1 -> 0.7.2, `sgl-deep-gemm` 0.2.1.post1 -> 0.2.3.post2.
  The FA3 attention backend diff is FA4 / unified-verify metadata only. Nothing changes the Muse FA3 call.

Net: the only upstream change on an H200 Muse path the judge times is the Marlin inactive-block skip (item 2), and
we use Marlin in prefill. Adopting that behaviour against the pinned baseline is a free win.
