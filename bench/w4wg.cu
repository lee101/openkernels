// W4A16 grouped GEMM on Hopper wgmma: C[r, n] = sum_k A[r, k] * deq(W_e[n, k]) * gs[e] for rows r of group e.
// A: [R, K] bf16 (rows grouped by expert, offs = inclusive prefix of counts). W: NVFP4 [E, N, K/2] codes,
// [E, N, K/16] e4m3 block scales, gs[E]. CTA = 2 consumer warpgroups (64 rows of A each) x 128 W rows, K step 64.
// Raw stages (A 16 KB + W codes 4 KB + scales 512 B) stream by cp.async; all 256 threads dequantize the W step
// into a bf16 128 B-swizzled tile (exact: e2m1 x e4m3 fits bf16), then each warpgroup runs 4 x m64n128k16.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#ifndef WS
#define WS 3
#endif
#define STA 16384
#define STW 4096
#define STS 512
#define STB 21504  // STA + STW + STS rounded up to 1 KB: each stage's A tile starts on a 128 B-swizzle atom
__device__ __forceinline__ uint64_t gdesc(uint32_t a, uint32_t lbo, uint32_t sbo) {
  return (uint64_t)((a & 0x3FFFF) >> 4) | ((uint64_t)(lbo >> 4) << 16) | ((uint64_t)(sbo >> 4) << 32) | (1ull << 62);
}
#define D64 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}"
#define R64(d) "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]), \
  "+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]), \
  "+f"(d[32]),"+f"(d[33]),"+f"(d[34]),"+f"(d[35]),"+f"(d[36]),"+f"(d[37]),"+f"(d[38]),"+f"(d[39]),"+f"(d[40]),"+f"(d[41]),"+f"(d[42]),"+f"(d[43]),"+f"(d[44]),"+f"(d[45]),"+f"(d[46]),"+f"(d[47]), \
  "+f"(d[48]),"+f"(d[49]),"+f"(d[50]),"+f"(d[51]),"+f"(d[52]),"+f"(d[53]),"+f"(d[54]),"+f"(d[55]),"+f"(d[56]),"+f"(d[57]),"+f"(d[58]),"+f"(d[59]),"+f"(d[60]),"+f"(d[61]),"+f"(d[62]),"+f"(d[63])
template <int ACC>
__device__ __forceinline__ void wg128(float* d, uint64_t da, uint64_t db) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 " D64 ", %64, %65, %66, 1, 1, 0, 0;\n" : R64(d) : "l"(da), "l"(db), "n"(ACC));
#endif
}
__device__ __forceinline__ void wfence() {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void wcommit() {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void wwait0() {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void cpa(uint32_t s, const void* g, int n) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(s), "l"(g), "r"(n)); }
__device__ __forceinline__ uint32_t pkb(float a, float b) { __nv_bfloat162 h = __floats2bfloat162_rn(a, b); return *(uint32_t*)&h; }

__global__ void __launch_bounds__(256, 1) w4wg(const __nv_bfloat16* A, const uint8_t* W, const uint8_t* SF, const float* gs,
                                              const int* offs, __nv_bfloat16* C, int N, int K) {
  extern __shared__ __align__(1024) uint8_t smem[];
  __shared__ int ti[2];
  const int tid = threadIdx.x, wg = tid >> 7, wt = tid & 127, w = wt >> 5, lane = tid & 31;
  if (tid == 0) {
    int t = blockIdx.x, e = 0, lo = 0;
    for (; e < 32; e++) { const int hi = offs[e], nt = (hi - lo + 127) >> 7; if (t < nt) break; t -= nt; lo = hi; }
    ti[0] = e; ti[1] = lo + t * 128;
  }
  __syncthreads();
  const int e = ti[0];
  if (e >= 32) return;
  const int r0 = ti[1], rend = offs[e], n0 = blockIdx.y * 128;
  const uint32_t sb = ((uint32_t)__cvta_generic_to_shared(smem) + 1023) & ~1023u;
  const uint32_t bt0 = sb + WS * STB;  // bf16 W tiles: 2 x 16 KB (128 rows x 64 k, 128 B swizzled rows)
  const uint8_t* We = W + ((size_t)e * N + n0) * (K / 2);
  const uint8_t* Se = SF + ((size_t)e * N + n0) * (K / 16);
  auto load = [&](int st, int k0) {
    const uint32_t s = sb + st * STB;
#pragma unroll
    for (int i = 0; i < 4; i++) {  // A: 128 rows x 128 B
      const int idx = i * 256 + tid, r = idx >> 3, c = idx & 7;
      const bool ok = r0 + r < rend;
      cpa(s + r * 128 + ((c ^ (r & 7)) << 4), A + (size_t)(ok ? r0 + r : r0) * K + k0 + c * 8, ok ? 16 : 0);
    }
    {  // W codes: 128 rows x 32 B (64 k)
      const int r = tid >> 1, c = tid & 1;
      cpa(s + STA + r * 32 + c * 16, We + (size_t)r * (K / 2) + k0 / 2 + c * 16, 16);
    }
    if (tid < 32) {  // scales: 128 rows x 4 B
      const int r = tid * 4;
#pragma unroll
      for (int j = 0; j < 4; j++) asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(s + STA + STW + (r + j) * 4), "l"(Se + (size_t)(r + j) * (K / 16) + k0 / 16));
    }
    asm volatile("cp.async.commit_group;\n");
  };
  float acc[64];
#pragma unroll
  for (int i = 0; i < 64; i++) acc[i] = 0.f;
  const int nk = K / 64;
  for (int i = 0; i < WS - 1; i++) load(i, i * 64);
  const float grid8[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  for (int kt = 0; kt < nk; kt++) {
    if (kt + WS - 1 < nk) load((kt + WS - 1) % WS, (kt + WS - 1) * 64); else asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_group %0;\n" :: "n"(WS - 1) : "memory");
    __syncthreads();
    const uint32_t s = sb + (kt % WS) * STB, bt = bt0 + (kt & 1) * 16384;
    {  // dequantize: thread -> W row tid >> 1, 32 k (16 B of codes, 2 block scales)
      const int r = tid >> 1, hf = tid & 1;
      uint4 q; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(q.x), "=r"(q.y), "=r"(q.z), "=r"(q.w) : "r"(s + STA + r * 32 + hf * 16));
      uint32_t sc; asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(sc) : "r"(s + STA + STW + r * 4));
      const uint32_t qq[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
      for (int c2 = 0; c2 < 2; c2++) {  // 16 k per block scale
        __nv_fp8_e4m3 f8; f8.__x = (uint8_t)(sc >> (8 * (hf * 2 + c2)));
        const float scl = (float)f8;
        uint32_t o[8];
#pragma unroll
        for (int j = 0; j < 8; j++) {
          const uint32_t wv = qq[c2 * 2 + (j >> 2)] >> (8 * (j & 3));
          const uint32_t lo = wv & 15, hi = (wv >> 4) & 15;
          const float a = (lo & 8 ? -grid8[lo & 7] : grid8[lo & 7]) * scl, b = (hi & 8 ? -grid8[hi & 7] : grid8[hi & 7]) * scl;
          o[j] = pkb(a, b);
        }
        const int ch = hf * 4 + c2 * 2;  // 16 B chunks (8 bf16) of the 128 B row
        asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" :: "r"(bt + r * 128 + ((ch ^ (r & 7)) << 4)), "r"(o[0]), "r"(o[1]), "r"(o[2]), "r"(o[3]));
        asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" :: "r"(bt + r * 128 + (((ch + 1) ^ (r & 7)) << 4)), "r"(o[4]), "r"(o[5]), "r"(o[6]), "r"(o[7]));
      }
    }
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    __syncthreads();
    wfence();
    const uint32_t as = s + wg * 64 * 128;
#pragma unroll
    for (int kk = 0; kk < 4; kk++) wg128<1>(acc, gdesc(as + kk * 32, 16, 1024), gdesc(bt + kk * 32, 16, 1024));
    wcommit(); wwait0();
  }
  const float g = gs[e];
  const int gg = lane >> 2, t = lane & 3;
#pragma unroll
  for (int j = 0; j < 16; j++)
#pragma unroll
    for (int hh = 0; hh < 2; hh++) {
      const int r = r0 + wg * 64 + w * 16 + gg + hh * 8;
      if (r < rend) *(uint32_t*)(C + (size_t)r * N + n0 + j * 8 + 2 * t) = pkb(acc[4 * j + 2 * hh] * g, acc[4 * j + 2 * hh + 1] * g);
    }
}
void run(torch::Tensor A, torch::Tensor W, torch::Tensor SF, torch::Tensor gs, torch::Tensor offs, torch::Tensor C) {
  const int N = W.size(1), K = W.size(2) * 2, R = A.size(0);
  const int smem = WS * STB + 32768 + 1024;
  static bool once = false;
  if (!once) { cudaFuncSetAttribute(w4wg, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); once = true; }
  w4wg<<<dim3((R + 127) / 128 + 32, N / 128), 256, smem, at::cuda::getCurrentCUDAStream()>>>((const __nv_bfloat16*)A.data_ptr(), W.data_ptr<uint8_t>(),
      (const uint8_t*)SF.data_ptr(), gs.data_ptr<float>(), offs.data_ptr<int>(), (__nv_bfloat16*)C.data_ptr(), N, K);
}

// ---- v2: WGN consumer warpgroups (64 A rows each), WS2-deep raw ring, dequant of step k+1 overlaps wgmma of step k
#ifndef DQL
#define DQL 1
#endif
#ifndef WS2
#define WS2 6
#endif
template <int WGN>
__global__ void __launch_bounds__(WGN * 128, WGN == 1 ? 2 : 1) w4wg2(const __nv_bfloat16* A, const uint8_t* W, const uint8_t* SF, const float* gs,
                                                                   const int* offs, __nv_bfloat16* C, int N, int K) {
  constexpr int SA = WGN * 8192, SBYTES = ((SA + STW + STS + 1023) / 1024) * 1024, NT = WGN * 128, MT = WGN * 64;
  extern __shared__ __align__(1024) uint8_t smem[];
  __shared__ int ti[2];
  const int tid = threadIdx.x, wg = tid >> 7, wt = tid & 127, w = wt >> 5, lane = tid & 31;
  if (tid == 0) {
    int t = blockIdx.x, e = 0, lo = 0;
    for (; e < 32; e++) { const int hi = offs[e], nt = (hi - lo + MT - 1) / MT; if (t < nt) break; t -= nt; lo = hi; }
    ti[0] = e; ti[1] = lo + t * MT;
  }
  __syncthreads();
  const int e = ti[0];
  if (e >= 32) return;
  const int r0 = ti[1], rend = offs[e], n0 = blockIdx.y * 128;
  const uint32_t sb = ((uint32_t)__cvta_generic_to_shared(smem) + 1023) & ~1023u;
  const uint32_t bt0 = sb + WS2 * SBYTES;
  const uint8_t* We = W + ((size_t)e * N + n0) * (K / 2);
  const uint8_t* Se = SF + ((size_t)e * N + n0) * (K / 16);
  const int nrow = min(MT, rend - r0);
  auto load = [&](int st, int k0) {
    const uint32_t s = sb + st * SBYTES;
    for (int idx = tid; idx < MT * 8; idx += NT) {  // A rows (only the valid ones are fetched)
      const int r = idx >> 3, c = idx & 7;
      if (r < nrow) cpa(s + r * 128 + ((c ^ (r & 7)) << 4), A + (size_t)(r0 + r) * K + k0 + c * 8, 16);
    }
    for (int idx = tid; idx < 256; idx += NT) { const int r = idx >> 1, c = idx & 1; cpa(s + SA + r * 32 + c * 16, We + (size_t)r * (K / 2) + k0 / 2 + c * 16, 16); }
    for (int r = tid; r < 128; r += NT) asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(s + SA + STW + r * 4), "l"(Se + (size_t)r * (K / 16) + k0 / 16));
    asm volatile("cp.async.commit_group;\n");
  };
  // rows past nrow must be zero for the MMA: clear them once in every stage (cp.async never writes them)
  for (int st = 0; st < WS2; st++)
    for (int idx = tid; idx < MT * 8; idx += NT) { const int r = idx >> 3; if (r >= nrow) asm volatile("st.shared.v4.u32 [%0], {%1,%1,%1,%1};\n" :: "r"(sb + st * SBYTES + r * 128 + ((idx & 7) << 4)), "r"(0)); }
  auto dequant = [&](int st, uint32_t bt) {  // 128 W rows x 64 k: NT threads, (128 * 4 / NT) chunks of 16 k each
    const uint32_t s = sb + st * SBYTES;
    for (int it = tid; it < 512; it += NT) {
      const int r = it >> 2, q4 = it & 3;  // row, 16-k block
      uint2 q; asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];\n" : "=r"(q.x), "=r"(q.y) : "r"(s + SA + r * 32 + q4 * 8));
      uint32_t sc; asm volatile("ld.shared.u8 %0, [%1];\n" : "=r"(sc) : "r"(s + SA + STW + r * 4 + q4));
      __nv_fp8_e4m3 f8; f8.__x = (uint8_t)sc;
      const float scl = (float)f8;
      uint32_t o[8];
#if DQL
      const __nv_bfloat162 s2 = __float2bfloat162_rn(scl);
#pragma unroll
      for (int hw = 0; hw < 4; hw++) {  // 4 elements (nibbles) per step: LUT bytes via prmt, sign by xor, exact bf16 scale
        const uint32_t wv = hw < 2 ? q.x : q.y, x = (hw & 1) ? wv >> 16 : wv, sel = x & 0x7777u;
        const uint32_t lo4 = __byte_perm(0xC0800000u, 0xC0804000u, sel), hi4 = __byte_perm(0x3F3F3F00u, 0x40404040u, sel);
        uint32_t a = __byte_perm(lo4, hi4, 0x5140), b = __byte_perm(lo4, hi4, 0x7362);
        a ^= ((x << 12) & 0x8000u) | ((x << 24) & 0x80000000u);
        b ^= ((x << 4) & 0x8000u) | ((x << 16) & 0x80000000u);
        __nv_bfloat162 av = __hmul2(*(__nv_bfloat162*)&a, s2), bv = __hmul2(*(__nv_bfloat162*)&b, s2);
        o[2 * hw] = *(uint32_t*)&av; o[2 * hw + 1] = *(uint32_t*)&bv;
      }
#else
#pragma unroll
      for (int j = 0; j < 8; j++) {
        const uint32_t wv = (j < 4 ? q.x : q.y) >> (8 * (j & 3));
        const uint32_t lo = wv & 15, hi = (wv >> 4) & 15;
        const float va = ((lo & 7) < 4 ? (lo & 7) * 0.5f : (float)(1 << ((lo & 7) - 2)) * (((lo & 7) & 1) ? 1.5f : 1.f)) * ((lo & 8) ? -scl : scl);
        const float vb = ((hi & 7) < 4 ? (hi & 7) * 0.5f : (float)(1 << ((hi & 7) - 2)) * (((hi & 7) & 1) ? 1.5f : 1.f)) * ((hi & 8) ? -scl : scl);
        o[j] = pkb(va, vb);
      }
#endif
      const int ch = q4 * 2;
      asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" :: "r"(bt + r * 128 + ((ch ^ (r & 7)) << 4)), "r"(o[0]), "r"(o[1]), "r"(o[2]), "r"(o[3]));
      asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" :: "r"(bt + r * 128 + (((ch + 1) ^ (r & 7)) << 4)), "r"(o[4]), "r"(o[5]), "r"(o[6]), "r"(o[7]));
    }
  };
  float acc[64];
#pragma unroll
  for (int i = 0; i < 64; i++) acc[i] = 0.f;
  const int nk = K / 64;
  for (int i = 0; i < WS2 - 1; i++) { if (i < nk) load(i, i * 64); else asm volatile("cp.async.commit_group;\n"); }
  asm volatile("cp.async.wait_group %0;\n" :: "n"(WS2 - 2) : "memory");
  __syncthreads();
  dequant(0, bt0);
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  __syncthreads();
  for (int kt = 0; kt < nk; kt++) {
    const uint32_t bt = bt0 + (kt & 1) * 16384, as = sb + (kt % WS2) * SBYTES + wg * 8192;
    wfence();
#pragma unroll
    for (int kk = 0; kk < 4; kk++) wg128<1>(acc, gdesc(as + kk * 32, 16, 1024), gdesc(bt + kk * 32, 16, 1024));
    wcommit();
    // step kt - 1's MMAs are done (at most 1 group pending): its raw slot and bf16 buffer are free
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
#endif
    __syncthreads();
    if (kt + WS2 - 1 < nk) load((kt + WS2 - 1) % WS2, (kt + WS2 - 1) * 64); else asm volatile("cp.async.commit_group;\n");
    if (kt + 1 < nk) {
      asm volatile("cp.async.wait_group %0;\n" :: "n"(WS2 - 2) : "memory");
      __syncthreads();
      dequant((kt + 1) % WS2, bt0 + ((kt + 1) & 1) * 16384);
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
      __syncthreads();
    }
  }
  wwait0();
  const float g = gs[e];
  const int gg = lane >> 2, t = lane & 3;
#pragma unroll
  for (int j = 0; j < 16; j++)
#pragma unroll
    for (int hh = 0; hh < 2; hh++) {
      const int r = r0 + wg * 64 + w * 16 + gg + hh * 8;
      if (r < rend) *(uint32_t*)(C + (size_t)r * N + n0 + j * 8 + 2 * t) = pkb(acc[4 * j + 2 * hh] * g, acc[4 * j + 2 * hh + 1] * g);
    }
}
void run2(torch::Tensor A, torch::Tensor W, torch::Tensor SF, torch::Tensor gs, torch::Tensor offs, torch::Tensor C, int64_t wgn) {
  const int N = W.size(1), K = W.size(2) * 2, R = A.size(0);
  auto st = at::cuda::getCurrentCUDAStream();
  if (wgn == 1) {
    constexpr int SBY = ((8192 + STW + STS + 1023) / 1024) * 1024;
    const int smem = WS2 * SBY + 32768 + 1024;
    static bool o1 = false; if (!o1) { cudaFuncSetAttribute(w4wg2<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); o1 = true; }
    w4wg2<1><<<dim3((R + 63) / 64 + 32, N / 128), 128, smem, st>>>((const __nv_bfloat16*)A.data_ptr(), W.data_ptr<uint8_t>(), (const uint8_t*)SF.data_ptr(), gs.data_ptr<float>(), offs.data_ptr<int>(), (__nv_bfloat16*)C.data_ptr(), N, K);
  } else {
    constexpr int SBY = ((16384 + STW + STS + 1023) / 1024) * 1024;
    const int smem = WS2 * SBY + 32768 + 1024;
    static bool o2 = false; if (!o2) { cudaFuncSetAttribute(w4wg2<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); o2 = true; }
    w4wg2<2><<<dim3((R + 127) / 128 + 32, N / 128), 256, smem, st>>>((const __nv_bfloat16*)A.data_ptr(), W.data_ptr<uint8_t>(), (const uint8_t*)SF.data_ptr(), gs.data_ptr<float>(), offs.data_ptr<int>(), (__nv_bfloat16*)C.data_ptr(), N, K);
  }
}
