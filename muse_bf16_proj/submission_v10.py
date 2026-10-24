# okc: problem=muse_bf16_proj
"""Stream-K skinny GEMM: out[m, n] = x @ W[:n].T for m <= 32 (and the router at any m).
W is repacked into mma.sync A-fragment order (2 KB contiguous per warp per 64-wide K chunk); every warp streams an
equal contiguous range of (tile, chunk) units; tiles split across warps meet through fp32 atomics and the last
arriver writes the output. The FP32 router runs as a BF16 hi + lo pair with FP32 accumulation."""
from typing import TypedDict
import torch
from torch.utils.cpp_extension import load_inline


class Weights(TypedDict):
    qkv_gate: torch.Tensor
    o: torch.Tensor
    pooler: torch.Tensor
    unpooler: torch.Tensor
    router: torch.Tensor


class Config(TypedDict):
    m: int
    n: int
    k: int
    layer: str


class Inputs(TypedDict):
    x: torch.Tensor
    config: Config


CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>
#ifndef WPC
#define WPC 16
#endif
#ifndef V7
#define V7 0
#endif
#ifndef GVNOST
#define GVNOST 0
#endif
#ifndef GVNOAT
#define GVNOAT 0
#endif
#ifndef RSN
#define RSN 2
#endif
#ifndef CUN
#define CUN 2
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define TMA 1
#else
#define TMA 0
#endif

__device__ __forceinline__ void mma(float* c, const uint4& a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint4 lds(uint32_t a) { uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a)); return v; }
__device__ __forceinline__ void mbinit(uint32_t b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(b), "r"(n)); }
__device__ __forceinline__ void mbwait(uint32_t b, uint32_t ph) {
#if TMA
  asm volatile("{\n.reg .pred p;\nW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W;\n}\n" :: "r"(b), "r"(ph) : "memory");
#else
  asm volatile("{\n.reg .pred p;\nW: mbarrier.test_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W;\n}\n" :: "r"(b), "r"(ph) : "memory");
#endif
}
__device__ __forceinline__ void cp16(uint32_t s, const void* g) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(g)); }

__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
#ifndef DBGT
#define DBGT 0
#endif
struct Q {
  const uint8_t* A; const uint8_t* A2; const __nv_bfloat16* x; void* out; float* ws; int* cnt;
  int M, N, K, MT, KC, TB, f32;
};

__device__ __forceinline__ void put(const Q& q, int tok, int row, float v) {
  if (q.f32) ((float*)q.out)[(size_t)tok * q.N + row] = v; else ((__nv_bfloat16*)q.out)[(size_t)tok * q.N + row] = __float2bfloat16(v);
}

// Each warp streams an equal contiguous range of (tile, unit) work through its own RS-deep TMA ring; a unit is CU
// consecutive 64-wide K chunks (2 KB of A fragments each; the router's lo half rides along). Partial tiles are
// summed with fp32 red.add into ws; a PDL-launched finalize kernel converts ws to the output and clears it.
template <int NT, bool HL, int RS, int CU>
__global__ void __launch_bounds__(WPC * 32, 1) skinny(Q q) {
  constexpr int SB = (HL ? 2 : 1) * CU * 2048;
  constexpr int CH = NT == 1 ? 4 : (NT == 2 ? 2 : 1);
  extern __shared__ __align__(128) uint8_t smem[];
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int W = gridDim.x * WPC, gw = blockIdx.x * WPC + wid;
  const uint32_t ring = (uint32_t)__cvta_generic_to_shared(smem) + wid * RS * SB;
  const uint32_t bars = (uint32_t)__cvta_generic_to_shared(smem) + WPC * RS * SB + wid * RS * 8;
#if TMA
  if (lane < RS) mbinit(bars + lane * 8, 1);
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");  // let the finalize kernel get scheduled early
#else
  if (lane < RS) mbinit(bars + lane * 8, 32);
#endif
  __syncwarp();
  const int KU = q.KC / CU;
  const long long U = (long long)q.TB * q.MT * KU;
  const long long L0 = U * gw / W, Le = U * (gw + 1) / W;
  if (L0 >= Le) return;
  long long pu = L0;
  auto issue = [&](int slot) {
    if (pu < Le) {
      const long long tile = pu / KU; const int cu = (int)(pu - tile * KU), mt = (int)(tile % q.MT);
      const size_t off = ((size_t)mt * q.KC + cu * CU) * 2048;
      const uint32_t d = ring + slot * SB, bar = bars + slot * 8;
#if TMA
      if (lane == 0) {
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(bar), "n"(SB) : "memory");
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" :: "r"(d), "l"(q.A + off), "n"(CU * 2048), "r"(bar) : "memory");
        if (HL) asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" :: "r"(d + CU * 2048), "l"(q.A2 + off), "n"(CU * 2048), "r"(bar) : "memory");
      }
#else
#pragma unroll
      for (int k = 0; k < 4 * CU; k++) cp16(d + k * 512 + lane * 16, q.A + off + k * 512 + lane * 16);
      if (HL) {
#pragma unroll
        for (int k = 0; k < 4 * CU; k++) cp16(d + CU * 2048 + k * 512 + lane * 16, q.A2 + off + k * 512 + lane * 16);
      }
      asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" :: "r"(bar) : "memory");
#endif
    } else {
#if !TMA
      asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(bars + slot * 8) : "memory");
#endif
    }
    pu++;
  };
#pragma unroll
  for (int i = 0; i < RS - 1; i++) issue(i);
  uint4 b0[NT][2], b1[NT][2];
  auto ldb = [&](long long cg, uint4 (&b)[NT][2]) {  // cg: global chunk index tile * KC + c
    const long long tile = cg / q.KC; const int c = (int)(cg - tile * q.KC), tb = (int)(tile / q.MT);
#pragma unroll
    for (int n = 0; n < NT; n++) {
      const int tok = min(tb * 32 + n * 8 + g, q.M - 1);
      const uint4* xr = (const uint4*)(q.x + (size_t)tok * q.K) + t * 2 + c * 8;
      b[n][0] = __ldg(xr); b[n][1] = __ldg(xr + 1);
    }
  };
  const long long CE = Le * CU;  // chunk range end
  ldb(L0 * CU, b0);
  float acc[NT][CH][4];
  int qn = 0;
  long long u = L0;
  while (u < Le) {
    const long long tile = u / KU; const int u0 = (int)(u - tile * KU);
    const int u1 = (int)min((long long)KU, u0 + (Le - u));
#pragma unroll
    for (int n = 0; n < NT; n++)
#pragma unroll
      for (int h = 0; h < CH; h++) acc[n][h][0] = acc[n][h][1] = acc[n][h][2] = acc[n][h][3] = 0.f;
    for (int uu = u0; uu < u1; uu++, u++, qn++) {
      const int slot = qn % RS;
      mbwait(bars + slot * 8, (qn / RS) & 1);
      __syncwarp();
      issue((qn + RS - 1) % RS);
#pragma unroll
      for (int i = 0; i < CU; i++) {
        const long long cg = u * CU + i;
        if (cg + 1 < CE) ldb(cg + 1, b1);
        const uint32_t sa = ring + slot * SB + i * 2048 + lane * 16;
#pragma unroll
        for (int j = 0; j < 4; j++) {
          const uint4 a = lds(sa + j * 512);
          uint4 h; if (HL) h = lds(sa + CU * 2048 + j * 512);
#pragma unroll
          for (int n = 0; n < NT; n++) {
            const uint32_t* bb = (const uint32_t*)&b0[n][j >> 1];
            mma(acc[n][j % CH], a, bb[(j & 1) * 2], bb[(j & 1) * 2 + 1]);
            if (HL) mma(acc[n][(j + 1) % CH], h, bb[(j & 1) * 2], bb[(j & 1) * 2 + 1]);
          }
        }
#pragma unroll
        for (int n = 0; n < NT; n++) { b0[n][0] = b1[n][0]; b0[n][1] = b1[n][1]; }
      }
    }
    const int tb = (int)(tile / q.MT), mt = (int)(tile - (long long)tb * q.MT), row0 = mt * 16 + g, mrem = min(32, q.M - tb * 32);
    float* w = q.ws + (size_t)tb * 32 * q.N;
#pragma unroll
    for (int n = 0; n < NT; n++) {
      float f[4];
#pragma unroll
      for (int i2 = 0; i2 < 4; i2++) { f[i2] = 0.f;
#pragma unroll
        for (int h = 0; h < CH; h++) f[i2] += acc[n][h][i2]; }
      const int tk = n * 8 + 2 * t;
      if (tk < mrem) { atomicAdd(w + (size_t)tk * q.N + row0, f[0]); atomicAdd(w + (size_t)tk * q.N + row0 + 8, f[2]); }
      if (tk + 1 < mrem) { atomicAdd(w + (size_t)(tk + 1) * q.N + row0, f[1]); atomicAdd(w + (size_t)(tk + 1) * q.N + row0 + 8, f[3]); }
    }
#if V7
    {  // the last warp to add into this tile writes it out and clears ws (no finalize kernel)
      auto wof = [&](long long uu) { return (int)(((uu + 1) * W - 1) / U); };  // warp streaming unit uu
      const long long t0u = tile * KU, t1u = t0u + KU - 1;
      const int nc = wof(t1u) - wof(t0u) + 1;
      __syncwarp();
      int last = 0;
      if (lane == 0) { int v; asm volatile("atom.add.acq_rel.gpu.global.s32 %0, [%1], 1;\n" : "=r"(v) : "l"(q.cnt + tile) : "memory"); last = v == nc - 1; }
      last = __shfl_sync(0xffffffffu, last, 0);
      if (last) {
        for (int e = lane; e < 16 * mrem; e += 32) {
          const int tok = e >> 4, row = mt * 16 + (e & 15);
          if (row < q.N) { float* wp = w + (size_t)tok * q.N + row; put(q, tb * 32 + tok, row, __ldcg(wp)); *wp = 0.f; }
        }
        if (lane == 0) q.cnt[tile] = 0;
      }
    }
#endif
  }
}

// ws (fp32 sums, [M][N]) -> out; ws cleared for the next call. Launched as a programmatic dependent.
__global__ void __launch_bounds__(256) fin(Q q) {
#if TMA
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
  const int n4 = q.M * q.N / 4;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) {
    float4* p = (float4*)q.ws + i;
    const float4 v = __ldcg(p);
    *p = make_float4(0.f, 0.f, 0.f, 0.f);
    if (q.f32) ((float4*)q.out)[i] = v;
    else { __nv_bfloat162 o[2] = {__floats2bfloat162_rn(v.x, v.y), __floats2bfloat162_rn(v.z, v.w)}; ((uint2*)q.out)[i] = *(uint2*)o; }
  }
}

static int g_grid = 0;
template <int NT, bool HL> static void go(const Q& q, cudaStream_t st) {
  constexpr int RS = HL ? 3 : RSN, CU = HL ? 1 : CUN;
  const size_t sm = (size_t)WPC * RS * (HL ? 2 : 1) * CU * 2048 + WPC * RS * 8;
  static bool once = false;
  if (!once) { cudaFuncSetAttribute(skinny<NT, HL, RS, CU>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm); once = true; }
  skinny<NT, HL, RS, CU><<<g_grid, WPC * 32, sm, st>>>(q);
}
void run(torch::Tensor A, torch::Tensor A2, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int64_t N) {
  if (!g_grid) { int dev; cudaGetDevice(&dev); cudaDeviceGetAttribute(&g_grid, cudaDevAttrMultiProcessorCount, dev); }
  Q q;
  q.A = (const uint8_t*)A.data_ptr(); q.A2 = A2.numel() ? (const uint8_t*)A2.data_ptr() : nullptr;
  q.x = (const __nv_bfloat16*)x.data_ptr(); q.out = out.data_ptr(); q.ws = ws.data_ptr<float>(); q.cnt = cnt.data_ptr<int>();
  q.M = x.size(0); q.K = x.size(1); q.N = (int)N; q.MT = (int)((N + 15) / 16); q.KC = q.K / 64; q.TB = (q.M + 31) / 32;
  q.f32 = out.scalar_type() == torch::kFloat32;
  auto st = at::cuda::getCurrentCUDAStream();
  const int nt = q.M >= 32 ? 4 : (q.M + 7) / 8;
  if (q.A2) { if (nt <= 1) go<1, true>(q, st); else if (nt <= 2) go<2, true>(q, st); else go<4, true>(q, st); }
  else { if (nt <= 1) go<1, false>(q, st); else if (nt <= 2) go<2, false>(q, st); else go<4, false>(q, st); }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(std::min(g_grid * 4, (q.M * q.N / 4 + 255) / 256)); cfg.blockDim = dim3(256); cfg.stream = st;
  cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[0].val.programmaticStreamSerializationAllowed = 1;
  static int major = 0; if (!major) { int dev; cudaGetDevice(&dev); cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev); }
  cfg.attrs = at; cfg.numAttrs = major >= 9 ? 1 : 0;
  if (!V7) cudaLaunchKernelEx(&cfg, fin, q);
}

// ---------------------------------------------------------------- v10: stream-K over (K slice, tile, chunk) units, x slice in smem
// CTA c streams units [U c / G, U (c + 1) / G) of the (slice, tile, chunk-unit) order, so it touches at most two K
// slices; their x fragments are staged once in smem. Each warp streams an equal sub-range through its own TMA ring.
// Partials meet in ws with fp32 red.add; a per-tile chunk counter picks the last arriver, which writes the output
// and clears ws and the counter (one launch, no finalize kernel).
template <int NT, int RS, int CU>
__global__ void __launch_bounds__(WPC * 32, 1) gv(Q q, int S, int CK, int XC) {
  constexpr int SB = CU * 2048;
  constexpr int CH = NT == 1 ? 4 : (NT == 2 ? 2 : 1);
  extern __shared__ __align__(128) uint8_t smem[];
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const unsigned long long T0 = DBGT ? gtime() : 0;
  unsigned long long* TB = (unsigned long long*)(q.ws + (16 << 20) - 65536) + (blockIdx.x * WPC + wid) * 4;
  const uint32_t sb = (uint32_t)__cvta_generic_to_shared(smem);
  const uint32_t ring = sb + wid * RS * SB, bars = sb + WPC * RS * SB + wid * RS * 8, xs = sb + WPC * RS * SB + WPC * RS * 8 + 16;
#if TMA
  if (lane < RS) mbinit(bars + lane * 8, 1);
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#else
  if (lane < RS) mbinit(bars + lane * 8, 32);
#endif
  __syncwarp();
  const int KU = CK / CU, SU = q.MT * KU;
  const long long U = (long long)S * SU;
  const long long C0 = U * blockIdx.x / gridDim.x, C1 = U * (blockIdx.x + 1) / gridDim.x;
  const long long L0 = C0 + (C1 - C0) * wid / WPC, Le = C0 + (C1 - C0) * (wid + 1) / WPC;
  long long pu = L0;
  auto issue = [&](int slot) {
    if (pu < Le) {
      const int s = (int)(pu / SU), r = (int)(pu - (long long)s * SU), mt = r / KU, cu = r - mt * KU;
      const size_t off = ((size_t)mt * q.KC + s * CK + cu * CU) * 2048;
      const uint32_t d = ring + slot * SB, bar = bars + slot * 8;
#if TMA
      if (lane == 0) {
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(bar), "n"(SB) : "memory");
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" :: "r"(d), "l"(q.A + off), "n"(SB), "r"(bar) : "memory");
      }
#else
#pragma unroll
      for (int k = 0; k < 4 * CU; k++) cp16(d + k * 512 + lane * 16, q.A + off + k * 512 + lane * 16);
      asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" :: "r"(bar) : "memory");
#endif
    } else {
#if !TMA
      asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(bars + slot * 8) : "memory");
#endif
    }
    pu++;
  };
  // stage the M token rows of x for the CTA's chunks [xc0, xc0 + XC): row stride XR = XC * 128 + 16 (conflict-free)
  const int xc0 = (int)(C0 / SU) * CK, xn = min(XC, q.KC - xc0), XR = XC * 128 + 16;
#if TMA
  const uint32_t xbar = sb + WPC * RS * SB + WPC * RS * 8;  // after the warp barriers
  if (threadIdx.x == 0) {
    mbinit(xbar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(xbar), "r"(q.M * xn * 128) : "memory");
    for (int r0 = 0; r0 < (GVNOST ? 0 : q.M); r0++) {  // rotated start row: CTAs do not all hit the same lines first
      const int r = (r0 + blockIdx.x) % q.M;
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                   :: "r"(xs + r * XR), "l"(q.x + (size_t)r * q.K + xc0 * 64), "r"(xn * 128), "r"(xbar) : "memory");
    }
  }
#endif
#pragma unroll
  for (int i = 0; i < RS - 1; i++) issue(i);
#if TMA
  __syncthreads();
  if (!GVNOST) mbwait(xbar, 0);
#else
  for (int e = threadIdx.x; e < (GVNOST ? 0 : q.M * xn * 8); e += WPC * 32) {
    const int r = e / (xn * 8), c = e - r * xn * 8;
    cp16(xs + r * XR + c * 16, q.x + (size_t)r * q.K + xc0 * 64 + c * 8);
  }
  asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::: "memory");
  __syncthreads();
#endif
  const unsigned long long T1 = DBGT ? gtime() : 0;
  if (L0 >= Le) return;
  // deferred epilogue: counter atomics (blocking) and last-arriver sums run after the warp's stream ends
  const int MP = (q.M + 3) & ~3;
  auto cntadd = [&](int mt, int mine) {  // true if this add completes the tile
    int v; asm volatile("atom.add.acq_rel.gpu.global.s32 %0, [%1], %2;\n" : "=r"(v) : "l"(q.cnt + mt * 32), "r"(mine) : "memory");
    return v + mine == q.KC;
  };
  auto gather = [&](int mt) {
    {  // sum the tile's partial slots: one per segment start (a (slice, tile) start or a warp's first unit)
      const int G = gridDim.x, n4 = 4 * MP;  // float4s per slot
      float4 sum[8];
#pragma unroll
      for (int i = 0; i < 8; i++) sum[i] = make_float4(0.f, 0.f, 0.f, 0.f);
      for (int b0 = 0; b0 < S * KU; b0 += 32) {  // lane tests unit b0 + lane; ballot gives the valid slots
        bool st = false;
        const int bi = b0 + lane;
        if (bi < S * KU) {
          const int ss = bi / KU, k = bi - ss * KU;
          const long long uu = (long long)ss * SU + (long long)mt * KU + k;
          st = k == 0;
          if (!st) {
            const int c = (int)(((uu + 1) * G - 1) / U);
            const long long c0 = U * c / G, len = U * (c + 1) / G - c0;
            const int w = (int)(((uu - c0 + 1) * WPC - 1) / len);
            st = c0 + len * w / WPC == uu;
          }
        }
        unsigned msk = __ballot_sync(0xffffffffu, st);
        while (msk) {
          int ids[4], nb = 0;
#pragma unroll
          for (int j = 0; j < 4; j++) { ids[j] = -1; if (msk) { ids[j] = b0 + __ffs(msk) - 1; msk &= msk - 1; nb++; } }
          float4 a[4][8];
#pragma unroll
          for (int j = 0; j < 4; j++) {
            const int bj = ids[j] < 0 ? 0 : ids[j], ss = bj / KU, k = bj - ss * KU;
            const float4* sp = (const float4*)(q.ws + ((size_t)(ss * q.MT + mt) * KU + k) * 16 * MP);
#pragma unroll
            for (int i = 0; i < 8; i++) a[j][i] = (ids[j] >= 0 && lane + 32 * i < n4) ? __ldcg(sp + lane + 32 * i) : make_float4(0.f, 0.f, 0.f, 0.f);
          }
#pragma unroll
          for (int j = 0; j < 4; j++)
#pragma unroll
            for (int i = 0; i < 8; i++) { sum[i].x += a[j][i].x; sum[i].y += a[j][i].y; sum[i].z += a[j][i].z; sum[i].w += a[j][i].w; }
          (void)nb;
        }
      }
#pragma unroll
      for (int i = 0; i < 8; i++) {
        const int e = (lane + 32 * i) * 4;
        if (e < 16 * MP) {
          const int rr = e / MP, tk = e - rr * MP, row = mt * 16 + rr;
          const float* fv = &sum[i].x;
#pragma unroll
          for (int j = 0; j < 4; j++) if (tk + j < q.M) put(q, tk + j, row, fv[j]);
        }
      }
      if (lane == 0) q.cnt[mt * 32] = 0;
    }
  };
  int segm[8], segc[8], nseg = 0;
  float acc[NT][CH][4];
  int qn = 0;
  long long u = L0;
  while (u < Le) {
    const int s = (int)(u / SU), r = (int)(u - (long long)s * SU), mt = r / KU, u0 = r - mt * KU;
    const int u1 = (int)min((long long)KU, u0 + (Le - u));
#pragma unroll
    for (int n = 0; n < NT; n++)
#pragma unroll
      for (int h = 0; h < CH; h++) acc[n][h][0] = acc[n][h][1] = acc[n][h][2] = acc[n][h][3] = 0.f;
    for (int uu = u0; uu < u1; uu++, u++, qn++) {
      const int slot = qn % RS;
      mbwait(bars + slot * 8, (qn / RS) & 1);
      __syncwarp();
      issue((qn + RS - 1) % RS);
#pragma unroll
      for (int i = 0; i < CU; i++) {
        const uint32_t xa = xs + (uint32_t)(s * CK + uu * CU + i - xc0) * 128 + t * 32 + g * XR;
        uint4 b[NT][2];
#pragma unroll
        for (int n = 0; n < NT; n++) {
          if (n * 8 + g < q.M) { b[n][0] = lds(xa + n * 8 * XR); b[n][1] = lds(xa + n * 8 * XR + 16); }
          else b[n][0] = b[n][1] = make_uint4(0, 0, 0, 0);
        }
        const uint32_t sa = ring + slot * SB + i * 2048 + lane * 16;
#pragma unroll
        for (int j = 0; j < 4; j++) {
          const uint4 a = lds(sa + j * 512);
#pragma unroll
          for (int n = 0; n < NT; n++) {
            const uint32_t* bb = (const uint32_t*)&b[n][j >> 1];
            mma(acc[n][j % CH], a, bb[(j & 1) * 2], bb[(j & 1) * 2 + 1]);
          }
        }
      }
    }
    if (GVNOAT) { if (acc[0][0][0] == 123.f) q.ws[0] = 1.f; continue; }
    float* slot = q.ws + ((size_t)(s * q.MT + mt) * KU + u0) * 16 * MP;
#pragma unroll
    for (int n = 0; n < NT; n++) {
      float f[4];
#pragma unroll
      for (int i2 = 0; i2 < 4; i2++) { f[i2] = 0.f;
#pragma unroll
        for (int h = 0; h < CH; h++) f[i2] += acc[n][h][i2]; }
      const bool od = t & 1;
      const float r0 = __shfl_xor_sync(0xffffffffu, od ? f[0] : f[2], 1), r1 = __shfl_xor_sync(0xffffffffu, od ? f[1] : f[3], 1);
      const int rr = g + (od ? 8 : 0), tok = n * 8 + (t & 2) * 2;
      if (tok < MP) __stcg((float4*)(slot + rr * MP + tok), od ? make_float4(r0, r1, f[2], f[3]) : make_float4(f[0], f[1], r0, r1));
    }
    if (nseg < 8) { segm[nseg] = mt; segc[nseg] = (u1 - u0) * CU; nseg++; }
    else {
      __syncwarp();
      int last = 0; if (lane == 0) last = cntadd(mt, (u1 - u0) * CU);
      if (__shfl_sync(0xffffffffu, last, 0)) gather(mt);
    }
  }
  const unsigned long long T2 = DBGT ? gtime() : 0;
  __syncwarp();
  int lm = 0, my = 0;
  if (lane < nseg) { for (int i = 0; i < 8; i++) if (i == lane) { lm = segm[i]; my = segc[i]; } }
  const unsigned lst = __ballot_sync(0xffffffffu, lane < nseg && cntadd(lm, my));
  for (int i = 0; i < nseg; i++) if (lst >> i & 1) gather(__shfl_sync(0xffffffffu, lm, i));
  if (DBGT && lane == 0) { TB[0] = T0; TB[1] = T1; TB[2] = T2; TB[3] = gtime(); }
}

template <int NT> static void go4(const Q& q, cudaStream_t st, int xb) {
  constexpr int RS = RSN, CU = CUN;
  const int xmax = (xb - 16 * q.M) / (q.M * 128);
  int S, CK, XC;
  if (q.KC <= xmax) { S = 1; CK = q.KC; XC = q.KC; } else { CK = 1; while (CK * 4 <= xmax) CK *= 2; S = q.KC / CK; XC = 2 * CK; }
  const size_t sm = (size_t)WPC * RS * CU * 2048 + WPC * RS * 8 + 16 + (size_t)q.M * (XC * 128 + 16);
  static bool once = false;
  if (!once) { int dev, mx; cudaGetDevice(&dev); cudaDeviceGetAttribute(&mx, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev); cudaFuncSetAttribute(gv<NT, RS, CU>, cudaFuncAttributeMaxDynamicSharedMemorySize, mx); once = true; }
  gv<NT, RS, CU><<<g_grid, WPC * 32, sm, st>>>(q, S, CK, XC);
}
void run4(torch::Tensor A, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int64_t N, int64_t xb) {
  if (!g_grid) { int dev; cudaGetDevice(&dev); cudaDeviceGetAttribute(&g_grid, cudaDevAttrMultiProcessorCount, dev); }
  Q q;
  q.A = (const uint8_t*)A.data_ptr(); q.A2 = nullptr;
  q.x = (const __nv_bfloat16*)x.data_ptr(); q.out = out.data_ptr(); q.ws = ws.data_ptr<float>(); q.cnt = cnt.data_ptr<int>();
  q.M = x.size(0); q.K = x.size(1); q.N = (int)N; q.MT = (int)((N + 15) / 16); q.KC = q.K / 64; q.TB = 1; q.f32 = 0;
  auto st = at::cuda::getCurrentCUDAStream();
  const int nt = (q.M + 7) / 8;
  if (nt <= 1) go4<1>(q, st, (int)xb); else if (nt <= 2) go4<2>(q, st, (int)xb); else go4<4>(q, st, (int)xb);
}

// ---------------------------------------------------------------- v6: deep per-CTA TMA ring (stream over 16 KB stages)
// A stage is 8 consecutive 2 KB units (one 16-row tile, 8 K chunks); CTA c streams stages [ST c / G, ST (c + 1) / G)
// through an NS-deep ring filled by a producer warp, so most of a CTA's bytes are in flight at once. Consumer warp w
// takes unit w of every stage; at a tile's end the 8 partials meet in smem, and a tile shared by CTAs is summed in
// ws with fp32 atomics, the last arriver writing the output.
#ifndef NSG
#define NSG 12
#endif
__device__ __forceinline__ void mbarrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(b) : "memory"); }
template <int NT, bool HL>
__global__ void __launch_bounds__(288, 1) gemv(Q q) {
  constexpr int SB = (HL ? 2 : 1) * 16384, NS = HL ? NSG / 2 : NSG, CH = NT == 1 ? 4 : (NT == 2 ? 2 : 1);
  extern __shared__ __align__(128) uint8_t smem[];
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const uint32_t base = (uint32_t)__cvta_generic_to_shared(smem);
  const uint32_t full = base + NS * SB, empty = full + NS * 8;
  float* red = (float*)(smem + NS * SB + NS * 16);
  if (threadIdx.x < NS) { mbinit(full + threadIdx.x * 8, TMA ? 1 : 32); mbinit(empty + threadIdx.x * 8, 8); }
#if TMA
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#endif
  __syncthreads();
  const int SPT = q.KC / 8;  // stages per tile
  const int ST = q.MT * SPT, G = gridDim.x, c = blockIdx.x;
  const int s0 = (int)((long long)ST * c / G), s1 = (int)((long long)ST * (c + 1) / G);
  if (wid == 8) {  // producer
#if !TMA
    for (int i = s0; i < s1; i++) {
      const int k = i - s0, slot = k % NS;
      if (k >= NS) mbwait(empty + slot * 8, ((k / NS) - 1) & 1);
      for (int o = lane * 16; o < 16384; o += 512) {
        cp16(base + slot * SB + o, q.A + (size_t)i * 16384 + o);
        if (HL) cp16(base + slot * SB + 16384 + o, q.A2 + (size_t)i * 16384 + o);
      }
      asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" :: "r"(full + slot * 8) : "memory");
    }
    asm volatile("cp.async.wait_all;\n" ::: "memory");
#endif
    if (lane == 0) {
      for (int i = s0; i < s1 && TMA; i++) {
        const int k = i - s0, slot = k % NS;
        if (k >= NS) mbwait(empty + slot * 8, ((k / NS) - 1) & 1);
#if TMA
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(full + slot * 8), "n"(SB) : "memory");
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" :: "r"(base + slot * SB), "l"(q.A + (size_t)i * 16384), "n"(16384), "r"(full + slot * 8) : "memory");
        if (HL) asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" :: "r"(base + slot * SB + 16384), "l"(q.A2 + (size_t)i * 16384), "n"(16384), "r"(full + slot * 8) : "memory");
#endif
      }
    }
    return;
  }
  const int tb = 0;
  float acc[NT][CH][4];
#pragma unroll
  for (int n = 0; n < NT; n++)
#pragma unroll
    for (int h = 0; h < CH; h++) acc[n][h][0] = acc[n][h][1] = acc[n][h][2] = acc[n][h][3] = 0.f;
  uint4 b0[NT][2];
  auto ldb = [&](int i, uint4 (&b)[NT][2]) {
    const int ch = (i % SPT) * 8 + wid;
#pragma unroll
    for (int n = 0; n < NT; n++) {
      const int tok = min(tb * 32 + n * 8 + g, q.M - 1);
      const uint4* xr = (const uint4*)(q.x + (size_t)tok * q.K) + t * 2 + ch * 8;
      b[n][0] = __ldg(xr); b[n][1] = __ldg(xr + 1);
    }
  };
  if (s0 < s1) ldb(s0, b0);
  for (int i = s0; i < s1; i++) {
    const int k = i - s0, slot = k % NS;
    uint4 b1[NT][2];
    if (i + 1 < s1) ldb(i + 1, b1);
    mbwait(full + slot * 8, (k / NS) & 1);
    const uint32_t sa = base + slot * SB + wid * 2048 + lane * 16;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const uint4 a = lds(sa + j * 512);
      uint4 h; if (HL) h = lds(sa + 16384 + j * 512);
#pragma unroll
      for (int n = 0; n < NT; n++) {
        const uint32_t* bb = (const uint32_t*)&b0[n][j >> 1];
        mma(acc[n][j % CH], a, bb[(j & 1) * 2], bb[(j & 1) * 2 + 1]);
        if (HL) mma(acc[n][(j + 1) % CH], h, bb[(j & 1) * 2], bb[(j & 1) * 2 + 1]);
      }
    }
    __syncwarp();
    if (lane == 0) mbarrive(empty + slot * 8);
#pragma unroll
    for (int n = 0; n < NT; n++) { b0[n][0] = b1[n][0]; b0[n][1] = b1[n][1]; }
    const int tile = i / SPT;
    if (i + 1 == s1 || (i + 1) % SPT == 0) {  // tile ends here (for this CTA): reduce the 8 warps' partials
#pragma unroll
      for (int n = 0; n < NT; n++) {
        float f[4];
#pragma unroll
        for (int i2 = 0; i2 < 4; i2++) { f[i2] = 0.f;
#pragma unroll
          for (int h = 0; h < CH; h++) f[i2] += acc[n][h][i2]; }
        ((float4*)red)[(n * 8 + wid) * 32 + lane] = make_float4(f[0], f[1], f[2], f[3]);
#pragma unroll
        for (int h = 0; h < CH; h++) acc[n][h][0] = acc[n][h][1] = acc[n][h][2] = acc[n][h][3] = 0.f;
      }
      asm volatile("bar.sync 1, 256;\n" ::: "memory");
      const int t0s = tile * SPT, t1s = t0s + SPT;
      const bool own = t0s >= s0 && t1s <= s1;
      const int row0 = tile * 16 + g;
      for (int n = wid; n < NT; n += 8) {
        float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int w2 = 0; w2 < 8; w2++) { const float4 v = ((float4*)red)[(n * 8 + w2) * 32 + lane]; sum.x += v.x; sum.y += v.y; sum.z += v.z; sum.w += v.w; }
        const int tk = n * 8 + 2 * t;
        const float fv[4] = {sum.x, sum.y, sum.z, sum.w};
#pragma unroll
        for (int e = 0; e < 4; e++) {
          const int tok = tk + (e & 1), row = row0 + (e >> 1) * 8;
          if (tok >= q.M || row >= q.N) continue;
          if (own) put(q, tok, row, fv[e]);
          else atomicAdd(q.ws + (size_t)tok * q.N + row, fv[e]);
        }
      }
      if (!own) {
        __threadfence();
        asm volatile("bar.sync 1, 256;\n" ::: "memory");
        __shared__ int lastf;
        if (threadIdx.x == 0) {
          auto cta = [&](int st2) { return (int)(((long long)(st2 + 1) * G - 1) / ST); };  // CTA streaming stage st2
          const int nc = cta(t1s - 1) - cta(t0s) + 1;
          lastf = atomicAdd(q.cnt + tile, 1) == nc - 1;
        }
        asm volatile("bar.sync 1, 256;\n" ::: "memory");
        if (lastf) {
          __threadfence();
          for (int e = threadIdx.x; e < 16 * q.M; e += 256) {
            const int tok = e >> 4, row = tile * 16 + (e & 15);
            if (row < q.N) { float* wp = q.ws + (size_t)tok * q.N + row; put(q, tok, row, __ldcg(wp)); *wp = 0.f; }
          }
          if (threadIdx.x == 0) q.cnt[tile] = 0;
        }
      }
      asm volatile("bar.sync 1, 256;\n" ::: "memory");
    }
  }
}
template <int NT, bool HL> static void go2(const Q& q, cudaStream_t st) {
  constexpr int NS = HL ? NSG / 2 : NSG;
  const size_t sm = (size_t)NS * (HL ? 32768 : 16384) + NS * 16 + 8 * 32 * 16 * NT;
  static bool once = false;
  if (!once) { cudaFuncSetAttribute(gemv<NT, HL>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm); once = true; }
  gemv<NT, HL><<<g_grid, 288, sm, st>>>(q);
}
void run2(torch::Tensor A, torch::Tensor A2, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int64_t N) {
  if (!g_grid) { int dev; cudaGetDevice(&dev); cudaDeviceGetAttribute(&g_grid, cudaDevAttrMultiProcessorCount, dev); }
  Q q;
  q.A = (const uint8_t*)A.data_ptr(); q.A2 = A2.numel() ? (const uint8_t*)A2.data_ptr() : nullptr;
  q.x = (const __nv_bfloat16*)x.data_ptr(); q.out = out.data_ptr(); q.ws = ws.data_ptr<float>(); q.cnt = cnt.data_ptr<int>();
  q.M = x.size(0); q.K = x.size(1); q.N = (int)N; q.MT = (int)((N + 15) / 16); q.KC = q.K / 64; q.TB = 1;
  q.f32 = out.scalar_type() == torch::kFloat32;
  auto st = at::cuda::getCurrentCUDAStream();
  const int nt = (q.M + 7) / 8;
  if (q.A2) { if (nt <= 1) go2<1, true>(q, st); else if (nt <= 2) go2<2, true>(q, st); else go2<4, true>(q, st); }
  else { if (nt <= 1) go2<1, false>(q, st); else if (nt <= 2) go2<2, false>(q, st); else go2<4, false>(q, st); }
}
// ---------------------------------------------------------------- router (N 256, K 8192, m <= 32)
// CTA (ks, mt) owns row tile mt and 16 consecutive 64-wide K chunks, one per warp. Every warp loads its hi + lo A
// fragments and its x chunk straight into registers in one round trip (64 KB per CTA in flight); the 16 warp partials
// meet in smem and the last of the KS CTAs on a tile sums their partials (no atomics on the output, no second kernel).
#define RPART (1 << 20)
#define RCNT 2048
__device__ __forceinline__ uint4 ldnc(const void* p) {
  uint4 v; asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}
template <int NT>
__global__ void __launch_bounds__(512, 1) rtr(Q q) {
  __shared__ float red[16 * 512];
  __shared__ int last;
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, t = lane & 3;
  const int ks = blockIdx.x, mt = blockIdx.y, KS = gridDim.x, c = ks * 16 + w;
  const uint8_t* ah = q.A + ((size_t)mt * q.KC + c) * 2048 + lane * 16;
  const uint8_t* al = q.A2 + ((size_t)mt * q.KC + c) * 2048 + lane * 16;
  uint4 a[4], h[4], b[NT][2];
#pragma unroll
  for (int j = 0; j < 4; j++) { a[j] = ldnc(ah + j * 512); h[j] = ldnc(al + j * 512); }
  const int ntg = (q.M + 31) >> 5;  // 32-token groups (the A fragments stay in registers)
  float* P = q.ws + RPART + (size_t)mt * KS * ntg * 512;
  for (int tg = 0; tg < ntg; tg++) {
#pragma unroll
  for (int n = 0; n < NT; n++) {
    const uint4* xr = (const uint4*)(q.x + (size_t)min(tg * 32 + n * 8 + g, q.M - 1) * q.K + c * 64) + t * 2;
    b[n][0] = __ldg(xr); b[n][1] = __ldg(xr + 1);
  }
  float acc[NT][4];
#pragma unroll
  for (int n = 0; n < NT; n++) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f;
#pragma unroll
  for (int j = 0; j < 4; j++)
#pragma unroll
    for (int n = 0; n < NT; n++) {
      const uint32_t* bb = (const uint32_t*)&b[n][j >> 1];
      mma(acc[n], a[j], bb[(j & 1) * 2], bb[(j & 1) * 2 + 1]);
      mma(acc[n], h[j], bb[(j & 1) * 2], bb[(j & 1) * 2 + 1]);
    }
#pragma unroll
  for (int n = 0; n < NT; n++) {
    float* r = red + w * 512 + g * 32 + n * 8 + 2 * t;
    *(float2*)r = make_float2(acc[n][0], acc[n][1]);
    *(float2*)(r + 8 * 32) = make_float2(acc[n][2], acc[n][3]);
  }
  __syncthreads();
  {
    float s = 0.f;
    if ((tid & 31) < NT * 8) {
#pragma unroll
      for (int ww = 0; ww < 16; ww++) s += red[ww * 512 + tid];
    }
    P[((size_t)ks * ntg + tg) * 512 + tid] = s;
  }
  __syncthreads();  // red is reused by the next group
  }
  const int row = tid >> 5, tok = tid & 31;  // this thread's output (row of the tile, token within a group)
  __threadfence();
  __syncthreads();
  if (tid == 0) last = atomicAdd(q.cnt + RCNT + mt, 1) == KS - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();
  for (int tg = 0; tg < ntg; tg++) {
    float tot = 0.f;
    for (int k2 = 0; k2 < KS; k2++) tot += __ldcg(P + ((size_t)k2 * ntg + tg) * 512 + tid);
    if (tg * 32 + tok < q.M) ((float*)q.out)[(size_t)(tg * 32 + tok) * q.N + mt * 16 + row] = tot;
  }
  if (tid == 0) q.cnt[RCNT + mt] = 0;
}
void run3(torch::Tensor A, torch::Tensor A2, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int64_t N) {
  Q q;
  q.A = (const uint8_t*)A.data_ptr(); q.A2 = (const uint8_t*)A2.data_ptr();
  q.x = (const __nv_bfloat16*)x.data_ptr(); q.out = out.data_ptr(); q.ws = ws.data_ptr<float>(); q.cnt = cnt.data_ptr<int>();
  q.M = x.size(0); q.K = x.size(1); q.N = (int)N; q.MT = (int)(N / 16); q.KC = q.K / 64; q.TB = 1; q.f32 = 1;
  const dim3 grid(q.KC / 16, q.MT);
  auto st = at::cuda::getCurrentCUDAStream();
  const int nt = q.M > 32 ? 4 : (q.M + 7) / 8;
  if (nt <= 1) rtr<1><<<grid, 512, 0, st>>>(q); else if (nt <= 2) rtr<2><<<grid, 512, 0, st>>>(q); else rtr<4><<<grid, 512, 0, st>>>(q);
}
"""

_mod = None
_DBG = []
RTR = True
RTRM = 128
XB = 65536
V6 = False  # deep per-CTA ring GEMV: slower than v5 on H200


def _big():
    pr = torch.cuda.get_device_properties(0)
    sm = getattr(pr, "shared_memory_per_block_optin", None)
    return sm >= 180000 if sm else (pr.major, pr.minor) in ((9, 0), (10, 0))


def _ext():
    global _mod
    if _mod is None:
        wpc = next((int(d[6:]) for d in _DBG if d.startswith("-DWPC=")), 16 if _big() else 8)
        _mod = load_inline("skinny_v10_%d" % wpc + "".join(_DBG).replace("-", "_").replace("=", ""),
                           cpp_sources="void run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t); void run2(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t); void run3(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t); void run4(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t);",
                           cuda_sources=CUDA, functions=["run", "run2", "run3", "run4"], extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo", "-DWPC=%d" % wpc] + ([] if _big() else ["-DNSG=4"]) + [d for d in _DBG if not d.startswith("-DWPC=")],
                           verbose=False)
    return _mod


def _pack(w):
    """bf16 [N, K] -> A fragments [N/16, K/64, 32 lanes, 4 steps, 8] (K permuted so each lane's 16 k are contiguous)."""
    N, K = w.shape
    W = w.view(N // 16, 16, K // 64, 4, 4, 4)  # mt, i, ch, t, j, e
    lo, hi = W[:, :8], W[:, 8:]
    fr = torch.stack((lo[..., 0], lo[..., 1], hi[..., 0], hi[..., 1], lo[..., 2], lo[..., 3], hi[..., 2], hi[..., 3]), -1)
    return fr.permute(0, 2, 4, 1, 3, 5).contiguous()  # mt, ch, j, g, t, 8: per step j the 32 lanes are contiguous


def prepare(weights):
    st = {}
    for k, w in weights.items():
        if k == "router":
            hi = w.to(torch.bfloat16)
            lo = (w - hi.float()).to(torch.bfloat16)
            st[k] = (_pack(hi), _pack(lo))
            st["router_hl_t"] = torch.cat((hi, lo)).t().contiguous()  # [K, 2N] for one cuBLAS mm at m > RTRM
        else:
            st[k] = (_pack(w), None)
        st[k + "_raw"] = w
    dev = weights["qkv_gate"].device
    st["ws"] = torch.zeros((256 * 8192,), dtype=torch.float32, device=dev)
    st["wp"] = torch.empty((16 << 20,), dtype=torch.float32, device=dev)  # gv partial slots
    st["cnt"] = torch.zeros((512 * 32,), dtype=torch.int32, device=dev)
    st["none"] = torch.empty(0, dtype=torch.bfloat16, device=dev)
    global XB
    XB = 73728 if _big() else 32768
    _ext()
    return st


def kernel_fn(data, state):
    cfg = data["config"]
    layer, n = cfg["layer"], cfg["n"]
    x = data["x"]
    m = x.shape[0]
    if layer != "router":
        if m > 32:
            return x @ state[layer + "_raw"][:n].T
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        _ext().run4(state[layer][0], x.contiguous(), out, state["wp"], state["cnt"], n, XB)
        return out
    A, A2 = state[layer]
    out = torch.empty((m, n), dtype=torch.float32, device=x.device)
    if m > RTRM:
        a = torch.mm(x, state["router_hl_t"], out_dtype=torch.float32)
        return a[:, :n] + a[:, n:]
    if layer == "router" and m <= 256 and RTR:
        _ext().run3(A, A2, x.contiguous(), out, state["ws"], state["cnt"], n)
        return out
    (_ext().run2 if V6 and m <= 32 else _ext().run)(A, A2 if A2 is not None else state["none"], x.contiguous(), out, state["ws"], state["cnt"], n)
    return out
