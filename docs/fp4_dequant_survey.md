# FP4/INT4 to BF16/FP16 dequant on sm_90: survey and a cheaper e2m1 sequence

Context: `moe_gap_analysis.md`, "Load-path floor". Our consumer issues about 22 instructions per 32-bit code word (8 weights). Dequant is 11 integer ops plus 4 HMUL2 (`cu/muse_moe_w4a16__CUDA.cu:209-212`, `:577-580`). The leader implies 17 or fewer.

All counts below are per 8 weights (one 32-bit word), integer or permute ops first, then float ops ("+ N mul").

## 1. What the open kernels do

| source | path | sequence | per 8 | applies to us? |
|---|---|---|---|---|
| vLLM / SGLang Marlin, Marlin-MoE NVFP4 (bf16) | `vllm-latest/csrc/quantization/marlin/dequant.h:434` (`dequant<nv_bfloat162, kFE2M1f>`), call site `marlin_template.h:1034`; SGLang JIT copy `python/sglang/kernels/jit/csrc/gemm/marlin/dequant.h` is identical | `b0 = q << 8`; per call `Out1 = (q & 0x80008000) \| ((q & 0x70007000) >> 6); q <<= 4; Out2 = ...` | 1 + 2 x (3 + 1 + 3) = **15** + 4 mul | Same target bits as ours ({15, 8:6}). The 2^126 exponent bias goes into the fp32 global scale (`nvfp4_marlin_process_global_scale`), so the per-register HMUL2 applies only the block scale (prepared as S0E5M3: `scale * 2^7`, `<< 1`). Worse than our 11. **This is the SGLang baseline we are timed against.** |
| Marlin W4A8-FP8 (fp4 to e4m3) | `dequant.h:474` | same mask/shift on 0x70707070 / 0x80808080, 4 values per register | **7**, then FP8 MMA | e2m1 is exact in e4m3, but it needs FP8 activations. Not allowed here (see section 3). |
| Marlin u4b8 to fp16 | `dequant.h:107-140` | `lop3((q & 0x000f000f) \| 0x64006400)`, the same for `0x00f000f0`, `q >>= 8`, then HSUB2 and HFMA2 with magic numbers | 5 + 4 (scale fused into the HFMA2 only with zero-points) | Integer only: an affine map exists, which e2m1 lacks. |
| CUTLASS sm90 mixed-input u4/s4 to bf16 (Machete uses this) | `cutlass/detail/collective/mixed_input_utils.hpp` `LayoutAwareConvertImpl<uint4b_t,bfloat16_t,(2,4):(4,1)>`; Machete adds the offline "shuffle" interleave (`machete_prepacked_layout.cuh`) so that one LDS.32 holds the 8 values in converter order | `r[i] = q >> 4i; lop3(r & 0x000F000F \| 0x43004300); hsub2(r, 128)` | 3 SHF + 4 LOP3 = **7** + 4 sub + 4 scale mul | Integer only. |
| TRT-LLM fpA_intB `FastInterleavedAndBiasedNumericArrayConverter<bf16, uint4b_t>` | `cpp/tensorrt_llm/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h:328` | the same magic 0x43004300 plus `fma.rn.bf16x2(x, 1, -136)`; nibble-interleaved columns | 3 + 4 = 7 + 4 fma (+ scale) | Integer only. |
| CUTLASS generic e2m1 to bf16 (sm < 100) | `cutlass/numeric_conversion.h` `_e2m1_to_bf16_x4` (about line 3796): two PRMT LUTs (`0x03020100FFFEFC00`, `0xc0804000`), byte reshuffles, `shl 6`, `or` | 13 per 4 | **26** + 4 scale mul | Exact without a bias multiply, but far worse. |
| TRT-LLM WFP4A16 (mxfp4 x bf16 MoE on Hopper) | `cutlass_extensions/.../sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input_.hpp` (`IsMXFP4`; a `UseFP4ToBF16LookupTable` flag is declared but unused), conversion through the CUTLASS converter above | wgmma RS, converter plus scale multiply | about 26 + 4 | Uses the wgmma-RS structure (our plan 4a), but the dequant is not better. |
| CUTLASS / TRT-LLM scale-folded LUT (int4 or e2m1 to fp8) | `mixed_input_utils.hpp` `lookup_table_convert` (also in SGLang AOT `cutlass_extensions/detail/collective/mixed_input_utils.hpp:146`) | `lut_idx = q & 0x77777777`, sign via lop3/shr; per 4 values: `prmt neg; prmt pos; prmt sel(sign)`. The table is the 8 magnitudes times the group scale (8-bit), built once per group | 2 + 6 = **8**, with the scale included | Only for 8-bit outputs: 1 byte per value, 4 values per register. For bf16 each value needs 2 bytes (the mantissa sits in the low byte), so it needs 2 tables and a merge. With per-16 scales a thread owns only 4 weights per (row, block), so table setup (about 8 ops) cannot be amortized. No. |
| CUTLASS `_e2m1_to_fp8_x4` | `numeric_conversion.h` about line 4022 | and, shr, and, prmt, prmt, or per 4 | 12 | FP8 output only. |
| HF kernels-community `nvfp4-gemm` GEMV, FlashInfer fp4 GEMMs | `nvfp4/nvfp4_gemv_kernels.cu` (`__CUDA_ARCH__ >= 1000`: `__nv_cvt_fp4x2_to_halfraw2`), FlashInfer `fp4_gemm.py` / trtllm-gen | hardware `cvt.rn.f16x2.e2m1x2` | 4 cvt (fp16) | **Blackwell only.** sm_90 has `cvt.rn.f16x2.e4m3x2` (sm_89+, F2FP...UNPACK_B) but no e2m1 or bf16 variant. |
| SGLang AOT/JIT | the Marlin and CUTLASS copies above | | | Nothing new. |

Summary: no open sm_90 kernel converts e2m1 to bf16 in fewer than our 11 integer ops plus 4 HMUL2. Marlin, which is the SGLang baseline, uses 15 + 4.

## 2. A new 8-op placement (verified)

The target is the same as now. Per bf16x2 register the bits are {15 sign, 8:6 e1 e0 m} and {31, 24:22}. The value is `bf16(bits) * 2^126`, which is exact for all 16 codes including the 0.5 subnormal (bf16 denormal). The existing HMUL2 by `scale * 2^126` (with our LEA bias) applies the scale, so nothing changes downstream.

```
M = 0x81C081C0
a0 = q & M                                   // LOP3
a1 = rotl(q, 10) & M                         // SHF.L.W + LOP3
a2 = rotl(q, 13) & M                         // SHF.L.W + LOP3
a3 = ((q & 0x001B001B) * 0x1084u) & M        // LOP3 + IMAD + LOP3
```

That is **8 integer ops per word instead of 11 (-3, about -14% of 22)**. `0x1084 = 2^2 + 2^7 + 2^12`. The three shifted copies of the 8 leftover bits {0,1,3,4,16,17,19,20} land on M exactly once and cannot carry into M. Checks:
- the IMAD path was checked exhaustively over all 256 leftover inputs (`scratchpad s1.py`);
- end to end, 20k random words decode bit-exactly to the e2m1 values;
- nvcc 12.9 `-arch=sm_90a` SASS for 4 words: 20 LOP3 + 8 SHF + 4 IMAD (plus addressing) + 16 HMUL2/HFMA2, i.e. 8 integer + 4 mul per word.

Repack (`prepare`) bit table. Bit i of q holds:

```
 0 a3.e0_lo   1 a3.e1_lo   2 a2.s_lo    3 a3.s_lo    4 a3.m_lo    5 a1.s_lo    6 a0.m_lo    7 a0.e0_lo
 8 a0.e1_lo   9 a2.m_hi   10 a2.e0_hi  11 a2.e1_hi  12 a1.m_hi   13 a1.e0_hi  14 a1.e1_hi  15 a0.s_lo
16 a3.e0_hi  17 a3.e1_hi  18 a2.s_hi   19 a3.s_hi   20 a3.m_hi   21 a1.s_hi   22 a0.m_hi   23 a0.e0_hi
24 a0.e1_hi  25 a2.m_lo   26 a2.e0_lo  27 a2.e1_lo  28 a1.m_lo   29 a1.e0_lo  30 a1.e1_lo  31 a0.s_hi
```

Here `lo` and `hi` are the two bf16 halves of each register, and s, e1, e0, m are the e2m1 code bits 3..0. The register-to-fragment assignment is free: keep a0..a3 as the current mma A slots, with sA for a0/a2 and sB for a1/a3. Equivalent alternatives: rotations (10, 29), (13, 26) or (26, 29), and the constant 0x4108 (shifts 3, 8, 14).

Searched without finding anything at 7 or fewer (`scratchpad s3/s5/s6/s8/s9`; this is not a proof):
- any 3 shift, rotate or zero-filling PRMT versions of q combined by select-LOP3;
- 2-word funnel windows plus a multiply;
- IMAD.WIDE producing two registers from one premask;
- an IMAD without the premask.

Translates of M cannot tile Z_32 or Z_64, which is why one register always needs the 3-op multiply.

## 3. Folding the bias and the scale into fewer multiplies (item c)

- Marlin already folds the e2m1 bias into the fp32 global scale, so 1 HMUL2 per bf16x2 remains for the block scale. Every sm_90 bf16 path surveyed (Marlin, CUTLASS/Machete ConvertAndScale, TRT-LLM WFP4A16) keeps that per-register multiply. The only multiply-free variants are the 8-bit-output LUTs (scale times value tables), which do not fit bf16.
- Scale on the accumulator per k16 block: Marlin-style placement leaves bf16 values times 2^-126 (denormal for code 0.5), so the bias would have to move into the activations (x * 2^119 is exact for |x| < 2^9, applied once at staging). The block scale then costs one FFMA per accumulator per k16: 4 per word per n8 tile (`N_pad / 2` per word for wgmma m64nN). That ties HMUL2 at N <= 8 and loses for larger N. It also depends on the tensor core keeping bf16 denormal inputs. No gain.
- K permutation (offline weights plus the same K permutation of x at staging) can make each thread's 16 K per row in a k64 chunk a single scale block. This cuts scale prep from 4 to 2-3 ops per 4 words: **-0.25 to -0.5 per word**.
- FP8 wgmma: e2m1 to e4m3 costs 7 ops per 8 (Marlin), but it needs x split into hi + lo e4m3 (grey under "activations BF16"). The 16-K block scale also cannot go inside a k32 FP8 wgmma without zero-padding K and per-block accumulator FFMAs. Not a path, as `moe_gap_analysis.md` already concludes.

## 4. Other consumer cuts seen in these kernels (item d)

- Marlin-style prepack, where one LDS.128 feeds the fragments of 4 MMAs, is what we already do (codes cost 0.25 per word).
- wgmma RS (CUTLASS mixed input, TRT-LLM WFP4A16, Machete) removes the per-n8 HMMA and the 0.5 per word of activation-B LDS. Dequant registers feed the A operand directly. This is our plan 4a.
- fp16 targets do not help. The placement is the same cost, and the sm_90 hardware `cvt.f16x2.e4m3x2` route (e2m1 to e4m3 at 3.5 ops, then 4 F2FP) is about 11 ops and fp16-only.

## 5. Recommendations

1. **Swap in the 8-op placement** (section 2): a repack table change in prepare plus 4 lines in the consumer. 22 to **about 19 per word**. At the measured issue efficiency (about 75% of stream at 22) this is roughly 86% of stream, if the consumers stay issue-bound. Check that ptxas keeps the IMAD (it does at -O3 on 12.9) and stress-test exactness on all 16 codes.
2. **wgmma RS consumer with this dequant**: drops the activation LDS (0.5) and most of the HMMA, loop and reduction overhead (about 2 to 2.5). That gives about 8 + 4 + 0.5 (scale) + 0.3 (loads) + about 1.5 (wgmma, fence, loop) = **about 14-15 per word**, below the leader's implied 17.
3. **K-permuted scale blocks** (one block per thread per row per k64): -0.25 to -0.5 per word on top of either.
