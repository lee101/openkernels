# okc: problem=muse_bf16_proj
"""Stream-K skinny GEMM: out[m, n] = x @ W[:n].T for m <= 32 (and the router at any m).
W is repacked into mma.sync A-fragment order (2 KB contiguous per warp per 64-wide K chunk); every warp streams an
equal contiguous range of (tile, chunk) units; tiles split across warps meet through fp32 atomics and the last
arriver writes the output. The FP32 router runs as a BF16 hi + lo pair with FP32 accumulation.
v13/v14: up to 32 tokens, the BF16 projections and the router (12-bit hi + lo) run a register-direct GEMV on losslessly 12-bit-packed weights (see GVCU),
launch geometry picked per shape in prepare() by timing against the v12 path.
v22: router m128/256 wgr (TMA + wgmma, workspace + counter split-K, no cluster); GEMV V bit 4 = st.async push split-K, bit 8 = no register cap.
v23: wgd (dense BF16 one-wave wgmma GEMM, workspace split-K) for m32..256: loses to cuBLASLt on the H100, kept unbuilt (WDI empty).
v24: wgs swap-AB wgmma (W rows = M from smem via TMA, tokens = N) for m16/32 o / qkv / pooler / unpooler, adopted only at a 2% margin (WSMARGIN).
v27: wgv HL (router BF16 hi + lo, c12 or plain fragments, FP32 out) for router m8/16/32/128/256 at the 2% margin.
v26: wgv = RS wgmma with c12 decoded straight into the A-fragment registers by the consumer warps, x tile shared via TMA
(m32 and m128 for qkv / o / pooler / unpooler), adopted at the 2% margin; wgq unbuilt.
v25: wgq = wgs on the lossless 12-bit weights (decoder warps fill the SW128 A ring); every path is one kernel launch per call
(no torch ops besides torch.empty, counters self-resetting, cuBLASLt split-K configs excluded for the BF16 layers)."""
from typing import TypedDict
import time
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
#pragma unroll
  for (int n = 0; n < NT; n++) {
    const uint4* xr = (const uint4*)(q.x + (size_t)min(n * 8 + g, q.M - 1) * q.K + c * 64) + t * 2;
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
  const int row = tid >> 5, tok = tid & 31;  // this thread's output (row of the tile, token)
  float s = 0.f;
  if (tok < NT * 8) {
#pragma unroll
    for (int ww = 0; ww < 16; ww++) s += red[ww * 512 + tid];
  }
  float* P = q.ws + RPART + (size_t)mt * KS * 512;
  P[ks * 512 + tid] = s;
  __threadfence();
  __syncthreads();
  if (tid == 0) last = atomicAdd(q.cnt + RCNT + mt, 1) == KS - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();
  float tot = 0.f;
  for (int k2 = 0; k2 < KS; k2++) tot += __ldcg(P + k2 * 512 + tid);
  if (tok < q.M) ((float*)q.out)[(size_t)tok * q.N + mt * 16 + row] = tot;
  if (tid == 0) q.cnt[RCNT + mt] = 0;
}
void run3(torch::Tensor A, torch::Tensor A2, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int64_t N) {
  Q q;
  q.A = (const uint8_t*)A.data_ptr(); q.A2 = (const uint8_t*)A2.data_ptr();
  q.x = (const __nv_bfloat16*)x.data_ptr(); q.out = out.data_ptr(); q.ws = ws.data_ptr<float>(); q.cnt = cnt.data_ptr<int>();
  q.M = x.size(0); q.K = x.size(1); q.N = (int)N; q.MT = (int)(N / 16); q.KC = q.K / 64; q.TB = 1; q.f32 = 1;
  const dim3 grid(q.KC / 16, q.MT);
  auto st = at::cuda::getCurrentCUDAStream();
  const int nt = (q.M + 7) / 8;
  if (nt <= 1) rtr<1><<<grid, 512, 0, st>>>(q); else if (nt <= 2) rtr<2><<<grid, 512, 0, st>>>(q); else rtr<4><<<grid, 512, 0, st>>>(q);
}
"""

# ---------------------------------------------------------------- v13/v14: register-direct swap-AB GEMV on 12-bit-packed BF16
# A unit is one 16-row tile x one 64-wide K chunk in mma.sync A-fragment order, per lane 3 x 16 B (c12) or 4 x 16 B (bf16):
# c12 keeps each value's low byte (exponent LSB + mantissa) and a nibble (sign + 3-bit offset of the upper 7 exponent bits
# from a per-matrix window); out-of-window values are fixed from a per-warp escape list (exact). FMT 2/3: the router's
# BF16 hi + lo pair (bf16 / c12), fp32 out. Warp (wr, wk) of CTA (bx, by) streams tiles (bx * WR + wr) * RT + j, chunks
# [(by * WK + wk) * KU, + KU) straight into registers; x comes from registers (XS 0) or a CTA-shared smem slab (XS 1);
# partials meet in smem, then across the S CTAs of a cluster through DSMEM (sm_90) or a last-arriver workspace (older GPUs).
GVCU = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>

struct G { const uint4* W; const uint4* W2; const __nv_bfloat16* x; void* out; const uint32_t* esc; float4* ws; int* cnt; int m, N, K, KC; uint32_t T0, T1, U0, U1; };

__device__ __forceinline__ uint4 gnc(const void* p) {
  uint4 v; asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}
__device__ __forceinline__ uint4 gx(const void* p) {
  uint4 v; asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}
__device__ __forceinline__ uint4 lds4(uint32_t a) {
  uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a)); return v;
}
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t s) { uint32_t d; asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(s)); return d; }
__device__ __forceinline__ void hmma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// V & 1 == 0: nibble v = offset of value v (byte-permute table T0:T1 = base .. base + 7), signs at bits 3, 11, 19, 27 (values 0-3)
// and 7, 15, 23, 31 (values 4-7): 11 ops / 8 values
__device__ __forceinline__ void dec0(uint32_t L0, uint32_t L1, uint32_t C, uint32_t T0, uint32_t T1, uint32_t* a) {
  const uint32_t cm = C & 0x77777777u;
  const uint32_t hl = prmt(T0, T1, cm) | ((C << 4) & 0x80808080u), hh = prmt(T0, T1, cm >> 16) | (C & 0x80808080u);
  a[0] = prmt(L0, hl, 0x5140); a[1] = prmt(L0, hl, 0x7362); a[2] = prmt(L1, hh, 0x5140); a[3] = prmt(L1, hh, 0x7362);
}
// V & 1 == 1: C byte i: bits 0-2 offset of value i, bit 7 its sign, bits 4-6 offset of value 4 + i, bit 3 the sign of value 4 + (i + 3) % 4,
// so (C & 0x87878787) + base and the same on C rotated by 4 are the high bytes of values 0-3 and 4-7 (window base <= 120)
__device__ __forceinline__ void dec1(uint32_t L0, uint32_t L1, uint32_t C, uint32_t Tb, uint32_t, uint32_t* a) {
  const uint32_t hl = (C & 0x87878787u) + Tb, hh = (__funnelshift_r(C, C, 4) & 0x87878787u) + Tb;
  a[0] = prmt(L0, hl, 0x5140); a[1] = prmt(L0, hl, 0x7362); a[2] = prmt(L1, hh, 0x5140); a[3] = prmt(L1, hh, 0x7362);
}
__device__ __forceinline__ void sto(__nv_bfloat16* d, float v) { *d = __float2bfloat16(v); }
__device__ __forceinline__ void sto(float* d, float v) { *d = v; }
template <typename O>
__device__ __forceinline__ void emit(const G& p, int idx, int row0, float4 v, int MT) {
  const int l = idx & 31, r = idx >> 5, n = r % MT, row = row0 + (r / MT) * 16 + (l >> 2), tok = n * 8 + 2 * (l & 3);
  O* o = (O*)p.out + (size_t)tok * p.N + row;
  sto(o, v.x); sto(o + 8, v.z);
  if (tok + 1 < p.m) { sto(o + p.N, v.y); sto(o + p.N + 8, v.w); }
}
__device__ __forceinline__ void add4(float4& a, const float4& b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }
__device__ __forceinline__ bool live(int idx, int MT, int m) { return ((idx >> 5) % MT) * 8 + 2 * (idx & 3) < m; }

template <int FMT, int MT, int ST, int WR, int WK, int KU, int S, int RT, int XS, int V>
__global__ void __launch_bounds__(WR * WK * 32, XS == 2 && !(V & 8) && WR * WK * 32 < 512 ? 512 / (WR * WK * 32) : 1) gv(G p) {
  constexpr bool C12 = FMT & 1, HL = FMT >= 2;
  constexpr int NL = C12 ? 3 : 4, NA = HL ? 2 : 1, NW = WR * WK, NT = NW * 32, NI = WR * RT * MT * 32, CH = WK * KU, TK = MT * 8;
  constexpr int NC = MT * RT * NA >= 2 ? 1 : 2, XB = XS ? 1 : ST;  // XS 2: x straight from global at use (L1), no ring
  using O = typename std::conditional<HL, float, __nv_bfloat16>::type;
  extern __shared__ __align__(16) uint8_t smem[];
  float4* red = (float4*)(smem + (XS == 1 ? CH * TK * 128 : 0));
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, wr = w % WR, wk = w / WR, g = lane >> 2, t = lane & 3;
  const int bz = blockIdx.x, bx = blockIdx.y, by = blockIdx.z, GX = gridDim.y;
  p.x += (size_t)bz * TK * p.K; p.out = (uint8_t*)p.out + (size_t)bz * TK * p.N * sizeof(O); p.m = min(TK, p.m - bz * TK);
  const int cb = by * CH, c0 = cb + wk * KU, row0 = bx * WR * RT * 16;
  const size_t wo = ((size_t)(bx * WR + wr) * RT * p.KC + c0) * (NL * 32) + lane, js = (size_t)p.KC * NL * 32;
  const __nv_bfloat16* xp[MT];
#pragma unroll
  for (int n = 0; n < MT; n++) xp[n] = p.x + (size_t)min(n * 8 + g, p.m - 1) * p.K + c0 * 64 + t * 16;
  uint4 wb[ST][RT][NA][NL], xb[XB][MT][2];
  auto ld = [&](int s, int u) {
#pragma unroll
    for (int j = 0; j < RT; j++)
#pragma unroll
      for (int h = 0; h < NA; h++)
#pragma unroll
        for (int i = 0; i < NL; i++) wb[s][j][h][i] = gnc((h ? p.W2 : p.W) + wo + j * js + (u * NL + i) * 32);
    if (!XS) {
#pragma unroll
      for (int n = 0; n < MT; n++) { xb[s % XB][n][0] = gx(xp[n] + u * 64); xb[s % XB][n][1] = gx(xp[n] + u * 64 + 8); }
    }
  };
#pragma unroll
  for (int s = 0; s < ST; s++) if (s < KU) ld(s, s);
  const uint32_t sb = (uint32_t)__cvta_generic_to_shared(smem);
  if (XS == 1) {
    for (int e = threadIdx.x; e < CH * TK * 8; e += NT) {
      const int s8 = e & 7, tok = (e >> 3) % TK, ch = e / (TK * 8);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sb + ((ch * TK + tok) * 8 + (s8 ^ (tok & 7))) * 16),
                   "l"(p.x + (size_t)min(tok, p.m - 1) * p.K + (cb + ch) * 64 + s8 * 8));
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_all;\n" ::: "memory");
  }
  uint32_t e = ~0u, nx = ~0u;
  int ep = 0;
  if (C12) { e = p.esc[((size_t)(by * GX + bx) * NW + w) * 32 + lane]; nx = __shfl_sync(~0u, e, 0); }
  if (XS == 1) __syncthreads();
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  constexpr bool PUSH = (V & 4) && S > 1;  // split-K partials pushed with st.async into the leader CTA (no pull, non-leaders exit)
  const uint32_t pbar = (uint32_t)__cvta_generic_to_shared(red + (WK + 1 + (S - 1)) * NI);
  uint32_t crk = 0;
  if constexpr (PUSH) {
    asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(crk));
    if (crk == 0 && threadIdx.x == 0) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\nfence.mbarrier_init.release.cluster;\n" :: "r"(pbar) : "memory");
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(pbar), "n"((S - 1) * NI * 16) : "memory");
    }
    asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory");
  }
#endif
  float acc[RT][MT][NC][4];
#pragma unroll
  for (int j = 0; j < RT; j++)
#pragma unroll
    for (int n = 0; n < MT; n++)
#pragma unroll
      for (int c = 0; c < NC; c++) acc[j][n][c][0] = acc[j][n][c][1] = acc[j][n][c][2] = acc[j][n][c][3] = 0.f;
  auto step = [&](const int u, const int s) {
    uint32_t A[RT][NA][16], B[MT][8];
    uint4 xd[XS == 2 ? MT : 1][2];
    if (XS == 2) {
#pragma unroll
      for (int n = 0; n < MT; n++) { xd[n][0] = gx(xp[n] + u * 64); xd[n][1] = gx(xp[n] + u * 64 + 8); }
    }
#pragma unroll
    for (int j = 0; j < RT; j++)
#pragma unroll
      for (int h = 0; h < NA; h++) {
        const uint4* q = wb[s][j][h];
        if (C12) {
          const uint32_t T0 = h ? p.U0 : p.T0, T1 = h ? p.U1 : p.T1;
          if (V & 1) {
            dec1(q[0].x, q[0].y, q[2].x, T0, T1, A[j][h]); dec1(q[0].z, q[0].w, q[2].y, T0, T1, A[j][h] + 4);
            dec1(q[1].x, q[1].y, q[2].z, T0, T1, A[j][h] + 8); dec1(q[1].z, q[1].w, q[2].w, T0, T1, A[j][h] + 12);
          } else {
            dec0(q[0].x, q[0].y, q[2].x, T0, T1, A[j][h]); dec0(q[0].z, q[0].w, q[2].y, T0, T1, A[j][h] + 4);
            dec0(q[1].x, q[1].y, q[2].z, T0, T1, A[j][h] + 8); dec0(q[1].z, q[1].w, q[2].w, T0, T1, A[j][h] + 12);
          }
        } else {
#pragma unroll
          for (int i = 0; i < 4; i++) { A[j][h][4 * i] = q[i].x; A[j][h][4 * i + 1] = q[i].y; A[j][h][4 * i + 2] = q[i].z; A[j][h][4 * i + 3] = q[i].w; }
        }
      }
#pragma unroll
    for (int n = 0; n < MT; n++) {
      uint4 b0, b1;
      if (XS == 1) {
        const uint32_t a = sb + ((wk * KU + u) * TK + n * 8 + g) * 128;
        b0 = lds4(a + (((2 * t) ^ g) << 4)); b1 = lds4(a + (((2 * t + 1) ^ g) << 4));
      } else if (XS == 2) { b0 = xd[n % (XS == 2 ? MT : 1)][0]; b1 = xd[n % (XS == 2 ? MT : 1)][1]; }
      else { b0 = xb[s % XB][n][0]; b1 = xb[s % XB][n][1]; }
      B[n][0] = b0.x; B[n][1] = b0.y; B[n][2] = b0.z; B[n][3] = b0.w; B[n][4] = b1.x; B[n][5] = b1.y; B[n][6] = b1.z; B[n][7] = b1.w;
    }
    if (u + ST < KU) ld(s, u + ST);
    if (C12) {
#pragma unroll
      for (int j = 0; j < RT; j++)
        while ((nx >> 20) == (uint32_t)(u * RT + j)) {
          if (lane == (int)((nx >> 15) & 31)) {
            const uint32_t r = (nx >> 11) & 15, hm = (nx >> 9) & 1, xr = (nx & 0x7fu) << (8 + ((nx >> 10) & 1) * 16);
#pragma unroll
            for (int h = 0; h < NA; h++)
#pragma unroll
              for (int i = 0; i < 16; i++) if (i == (int)r && h == (int)hm) A[j][h][i] ^= xr;
          }
          nx = ++ep < 32 ? __shfl_sync(~0u, e, ep) : ~0u;
        }
    }
#pragma unroll
    for (int q = 0; q < 4; q++)
#pragma unroll
      for (int j = 0; j < RT; j++)
#pragma unroll
        for (int n = 0; n < MT; n++)
#pragma unroll
          for (int h = 0; h < NA; h++) hmma(acc[j][n][(q * NA + h) % NC], A[j][h] + 4 * q, B[n][2 * q], B[n][2 * q + 1]);
  };
  if (V & 2) {  // rolled: small code (cold instruction fetch), no cross-group load hoisting
#pragma unroll 1
    for (int u0 = 0; u0 < KU; u0 += ST)
#pragma unroll
      for (int s = 0; s < ST; s++) if (u0 + s < KU) step(u0 + s, s);
  } else {
#pragma unroll
    for (int u = 0; u < KU; u++) step(u, u % ST);
  }
  float4 f[RT][MT];
#pragma unroll
  for (int j = 0; j < RT; j++)
#pragma unroll
    for (int n = 0; n < MT; n++) {
      f[j][n] = make_float4(acc[j][n][0][0], acc[j][n][0][1], acc[j][n][0][2], acc[j][n][0][3]);
#pragma unroll
      for (int c = 1; c < NC; c++) add4(f[j][n], make_float4(acc[j][n][c][0], acc[j][n][c][1], acc[j][n][c][2], acc[j][n][c][3]));
    }
  if (WK == 1 && S == 1) {
#pragma unroll
    for (int j = 0; j < RT; j++)
#pragma unroll
      for (int n = 0; n < MT; n++) if (n * 8 + 2 * t < p.m) emit<O>(p, ((wr * RT + j) * MT + n) * 32 + lane, row0, f[j][n], MT);
    return;
  }
  float4* cs = red + WK * NI;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if constexpr (PUSH) {
#pragma unroll
    for (int j = 0; j < RT; j++)
#pragma unroll
      for (int n = 0; n < MT; n++) red[wk * NI + ((wr * RT + j) * MT + n) * 32 + lane] = f[j][n];
    asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory");
    __syncthreads();
    const uint32_t rv = (uint32_t)__cvta_generic_to_shared(cs + NI);
    uint32_t rrv = 0, rbar = 0;
    if (crk) {
      asm volatile("mapa.shared::cluster.u32 %0, %1, 0;\n" : "=r"(rrv) : "r"(rv + (crk - 1) * NI * 16));
      asm volatile("mapa.shared::cluster.u32 %0, %1, 0;\n" : "=r"(rbar) : "r"(pbar));
    }
    for (int idx = threadIdx.x; idx < NI; idx += NT) {
      float4 v = red[idx];
#pragma unroll
      for (int q = 1; q < WK; q++) add4(v, red[q * NI + idx]);
      if (crk == 0) cs[idx] = v;
      else asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];\n"
                        :: "r"(rrv + idx * 16), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(rbar) : "memory");
    }
    if (crk) return;
    __syncthreads();
    asm volatile("{\n.reg .pred P;\nPW: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], 0;\n@!P bra PW;\n}\n" :: "r"(pbar) : "memory");
    const float4* rvp = cs + NI;
    for (int idx = threadIdx.x; idx < NI; idx += NT) {
      if (!live(idx, MT, p.m)) continue;
      float4 v = cs[idx];
#pragma unroll
      for (int q = 0; q < S - 1; q++) add4(v, rvp[q * NI + idx]);
      emit<O>(p, idx, row0, v, MT);
    }
    return;
  }
#endif
#pragma unroll
  for (int j = 0; j < RT; j++)
#pragma unroll
    for (int n = 0; n < MT; n++) red[wk * NI + ((wr * RT + j) * MT + n) * 32 + lane] = f[j][n];
  __syncthreads();
  for (int idx = threadIdx.x; idx < NI; idx += NT) {
    if (!live(idx, MT, p.m)) continue;
    float4 v = red[idx];
#pragma unroll
    for (int q = 1; q < WK; q++) add4(v, red[q * NI + idx]);
    if (S == 1) emit<O>(p, idx, row0, v, MT); else cs[idx] = v;
  }
  if (S == 1) return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
  uint32_t rk; asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(rk));
  constexpr int PER = (NI + S - 1) / S;
  const uint32_t ca = (uint32_t)__cvta_generic_to_shared(cs);
  for (int idx = rk * PER + threadIdx.x; idx < min(NI, (int)(rk + 1) * PER); idx += NT) {
    if (!live(idx, MT, p.m)) continue;
    float4 v[S];
#pragma unroll
    for (int q = 0; q < S; q++) {
      uint32_t ra; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(ra) : "r"(ca + idx * 16), "r"(q));
      asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];\n" : "=f"(v[q].x), "=f"(v[q].y), "=f"(v[q].z), "=f"(v[q].w) : "r"(ra) : "memory");
    }
#pragma unroll
    for (int q = 1; q < S; q++) add4(v[0], v[q]);
    emit<O>(p, idx, row0, v[0], MT);
  }
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
#else
  float4* wsp = p.ws + (size_t)(bz * GX + bx) * S * NI;
  for (int idx = threadIdx.x; idx < NI; idx += NT) wsp[by * NI + idx] = cs[idx];
  __threadfence();
  __syncthreads();
  __shared__ int last;
  if (threadIdx.x == 0) last = atomicAdd(p.cnt + bz * GX + bx, 1) == S - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();
  for (int idx = threadIdx.x; idx < NI; idx += NT) {
    if (!live(idx, MT, p.m)) continue;
    float4 v = __ldcg(wsp + idx);
    for (int q = 1; q < S; q++) add4(v, __ldcg(wsp + q * NI + idx));
    emit<O>(p, idx, row0, v, MT);
  }
  if (threadIdx.x == 0) p.cnt[bz * GX + bx] = 0;
#endif
}

template <int FMT, int MT, int ST, int WR, int WK, int KU, int S, int RT, int XS, int V>
static int64_t gvl(const G& p, cudaStream_t st) {
  constexpr int NI = WR * RT * MT * 32;
  auto k = gv<FMT, MT, ST, WR, WK, KU, S, RT, XS, V>;
  if (p.N % (16 * WR * RT) || p.K != S * WK * KU * 64 || p.m < 1) return -1;
  const size_t sm = (XS == 1 ? (size_t)WK * KU * MT * 8 * 128 : 0) + ((WK > 1 || S > 1) ? (size_t)(WK + 1) * NI * 16 : 0) + ((V & 4) && S > 1 ? (size_t)(S - 1) * NI * 16 + 16 : 0);
  static bool once = false;
  if (!once) {
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    if (S > 8) { cudaFuncSetAttribute(k, cudaFuncAttributeNonPortableClusterSizeAllowed, 1); cudaGetLastError(); }
    once = true;
  }
  static int cc = 0;
  if (!cc) { int dev; cudaGetDevice(&dev); cudaDeviceGetAttribute(&cc, cudaDevAttrComputeCapabilityMajor, dev); }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((p.m + 8 * MT - 1) / (8 * MT), p.N / (16 * WR * RT), S); cfg.blockDim = dim3(WR * WK * 32); cfg.dynamicSmemBytes = sm; cfg.stream = st;
  cudaLaunchAttribute at[1];
  if (S > 1 && cc >= 9) { at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = 1; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = S; cfg.attrs = at; cfg.numAttrs = 1; }
  const cudaError_t r = cudaLaunchKernelEx(&cfg, k, p);
  if (r != cudaSuccess) cudaGetLastError();
  return (int64_t)r;
}

// FP32 router, small m: CTA = R full rows (no cross-CTA reduction), 512 threads split K, fp32 weights read once
template <int M, int R>
__global__ void __launch_bounds__(512) rr(const float4* W, const __nv_bfloat16* x, float* out, int m, int N, int K) {
  __shared__ float red[16][R * M];
  const int t = threadIdx.x, lane = t & 31, w = t >> 5, K4 = K >> 2, row0 = blockIdx.x * R;
  float4 wv[R][4];
#pragma unroll
  for (int r = 0; r < R; r++)
#pragma unroll
    for (int i = 0; i < 4; i++) { const uint4 u = gnc(W + (size_t)(row0 + r) * K4 + t + i * 512); wv[r][i] = make_float4(__uint_as_float(u.x), __uint_as_float(u.y), __uint_as_float(u.z), __uint_as_float(u.w)); }
  float acc[R][M];
#pragma unroll
  for (int r = 0; r < R; r++)
#pragma unroll
    for (int q = 0; q < M; q++) acc[r][q] = 0.f;
#pragma unroll
  for (int i = 0; i < 4; i++)
#pragma unroll
    for (int q = 0; q < M; q++) {
      if (q < m) {
        const uint2 xv = __ldg((const uint2*)(x + (size_t)q * K) + t + i * 512);
        const float2 a = __bfloat1622float2(*(const __nv_bfloat162*)&xv.x), b = __bfloat1622float2(*(const __nv_bfloat162*)&xv.y);
#pragma unroll
        for (int r = 0; r < R; r++) acc[r][q] = fmaf(wv[r][i].w, b.y, fmaf(wv[r][i].z, b.x, fmaf(wv[r][i].y, a.y, fmaf(wv[r][i].x, a.x, acc[r][q]))));
      }
    }
#pragma unroll
  for (int r = 0; r < R; r++)
#pragma unroll
    for (int q = 0; q < M; q++) {
      float v = acc[r][q];
#pragma unroll
      for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
      if (lane == 0) red[w][r * M + q] = v;
    }
  __syncthreads();
  if (t < R * M && t % M < m) {
    float v = 0.f;
#pragma unroll
    for (int i = 0; i < 16; i++) v += red[i][t];
    out[(size_t)(t % M) * N + row0 + t / M] = v;
  }
}

int64_t rrrun(int64_t inst, torch::Tensor W, torch::Tensor x, torch::Tensor out) {
  const int m = x.size(0), K = x.size(1), N = out.size(1);
  auto st = at::cuda::getCurrentCUDAStream();
  const float4* w = (const float4*)W.data_ptr(); const __nv_bfloat16* xp = (const __nv_bfloat16*)x.data_ptr(); float* o = out.data_ptr<float>();
  if (K != 8192) return -1;
  switch (inst) {
    case 0: if (m > 1) return -1; rr<1, 1><<<N, 512, 0, st>>>(w, xp, o, m, N, K); break;
    case 1: if (m > 1) return -1; rr<1, 2><<<N / 2, 512, 0, st>>>(w, xp, o, m, N, K); break;
    case 2: if (m > 8) return -1; rr<8, 2><<<N / 2, 512, 0, st>>>(w, xp, o, m, N, K); break;
    case 3: if (m > 8) return -1; rr<8, 4><<<N / 4, 512, 0, st>>>(w, xp, o, m, N, K); break;
    default: return -2;
  }
  const cudaError_t r = cudaGetLastError();
  return (int64_t)r;
}

int64_t gvrun(int64_t inst, torch::Tensor W, torch::Tensor W2, torch::Tensor x, torch::Tensor out, torch::Tensor esc, torch::Tensor ws, torch::Tensor cnt, int64_t T, int64_t U) {
  G p;
  p.W = (const uint4*)W.data_ptr(); p.W2 = (const uint4*)W2.data_ptr(); p.x = (const __nv_bfloat16*)x.data_ptr(); p.out = out.data_ptr();
  p.esc = (const uint32_t*)esc.data_ptr(); p.ws = (float4*)ws.data_ptr(); p.cnt = cnt.data_ptr<int>();
  p.m = x.size(0); p.K = x.size(1); p.N = out.size(1); p.KC = p.K / 64;
  p.T0 = (uint32_t)(T & 0xffffffff); p.T1 = (uint32_t)((uint64_t)T >> 32); p.U0 = (uint32_t)(U & 0xffffffff); p.U1 = (uint32_t)((uint64_t)U >> 32);
  auto st = at::cuda::getCurrentCUDAStream();
  switch (inst) {
@@CASES@@
  }
  return -2;
}
"""
# candidate launch geometries (WR, WK, KU, S, RT, XS) per (layer key, MT); FMT is c12 (1/3) with a bf16 (0/2) twin of the
# first geometry as the fallback when a 12-bit pack fails
# candidates per (layer key, MT): (WR, WK, KU, S, RT, XS) -> kernel variants V (bit 0: 9-op add decode, else 11-op table
# decode; bit 1: rolled K loop); H200 v15 judge: the v14 kernel (V 0) was faster at deep K, the rolled/9-op one (V 3) at m32
_0, _1, _01 = (0,), (1,), (0, 1)
GVC = {  # H200 v16 log: V1 (unrolled, 9-op decode) wins most shapes, V3 (rolled) only pooler m1 at 1024 threads and qkv m16
    ("qkv_gate", 1): [((1, 8, 16, 1, 1, 0), _1), ((1, 4, 16, 2, 1, 0), _01), ((1, 4, 8, 4, 1, 0), _1)],
    ("qkv_gate", 2): [((1, 4, 16, 2, 1, 0), _1), ((1, 4, 8, 4, 1, 2), (1, 3)), ((1, 4, 16, 2, 1, 2), _01), ((1, 4, 16, 2, 1, 2, 1), (1, 5, 13)), ((1, 4, 8, 4, 1, 2, 1), (1, 5, 13))],
    ("qkv_gate", 4): [((1, 8, 16, 1, 1, 2), _01), ((1, 4, 16, 2, 1, 2), _01), ((2, 4, 16, 2, 1, 2), _1), ((1, 4, 8, 4, 1, 2, 1), (0, 1, 5, 13)), ((1, 4, 8, 4, 1, 2), _1),
                      ((1, 4, 16, 2, 1, 2, 1), (1, 5, 9, 13)), ((2, 4, 16, 2, 1, 2, 1), (13,))],
    ("o", 1): [((1, 8, 2, 1, 1, 0), _01), ((2, 4, 4, 1, 1, 0), _1), ((1, 4, 4, 1, 1, 0), _1)],
    ("o", 2): [((1, 8, 2, 1, 1, 0), _01), ((2, 4, 4, 1, 1, 0), _1), ((2, 4, 4, 1, 1, 2), _1), ((1, 8, 2, 1, 1, 2), _1), ((1, 4, 4, 1, 1, 2, 1), _1)],
    ("o", 4): [((1, 4, 4, 1, 1, 2), _01), ((2, 4, 4, 1, 1, 2), _01), ((1, 8, 2, 1, 1, 0), _1), ((1, 4, 4, 1, 1, 2, 1), _01), ((1, 8, 2, 1, 1, 2), _1), ((1, 8, 2, 1, 1, 2, 1), _1),
               ((1, 4, 4, 1, 1, 2, 4), _1), ((1, 8, 2, 1, 1, 0, 1), _1)],
    ("pooler", 1): [((2, 8, 16, 1, 1, 0), _01), ((2, 16, 8, 1, 1, 0), (3,)), ((1, 8, 16, 1, 1, 0), _1)],
    ("pooler", 2): [((2, 8, 16, 1, 1, 0), _01), ((1, 8, 16, 1, 1, 2), _1), ((1, 8, 16, 1, 1, 0), _1)],
    ("pooler", 4): [((1, 8, 16, 1, 1, 2), _01), ((1, 4, 16, 2, 1, 2), _1), ((2, 4, 16, 2, 1, 2), _1)],
    ("unpooler_s", 1): [((1, 16, 4, 1, 1, 0), _01), ((1, 8, 2, 4, 1, 0), _1), ((1, 8, 4, 2, 1, 0), _1)],
    ("unpooler", 2): [((1, 4, 16, 1, 1, 2), _01), ((1, 4, 16, 1, 1, 0), _1), ((1, 8, 8, 1, 1, 0), _1)],
    ("unpooler", 4): [((1, 4, 16, 1, 1, 2), _01), ((2, 4, 16, 1, 1, 2), _1), ((1, 8, 8, 1, 1, 2), _1)],
    # router: plain BF16 hi + lo fragments (FMT 2), every warp's units in flight at once, split-K over a 16-CTA cluster
    ("router", 1): [((1, 8, 1, 16, 1, 0), _0), ((1, 4, 2, 16, 1, 0), (0, 4)), ((1, 8, 2, 8, 1, 0), (0, 4))],
    ("router", 2): [((1, 8, 1, 16, 1, 0), _0), ((1, 4, 2, 16, 1, 0), (0, 4)), ((1, 8, 2, 8, 1, 0), (0, 4))],
    ("router", 4): [((1, 8, 1, 16, 1, 0), _0), ((1, 4, 2, 16, 1, 0), (0, 4)), ((1, 8, 2, 8, 1, 0), (0, 4))],
}
GVRB = [(128, 4), (256, 4)]  # router only: wgmma router candidates (the GEMV token blocks lost: 1.6x v12)
GVB = {1: 1, 8: 1, 16: 2, 32: 4}  # bucket m -> MT (the kernel also runs m > 8 MT as token blocks in grid.x; slower than cuBLAS)


def _gvst(geo):  # a 7th geometry field forces the register-ring depth
    if len(geo) == 7:
        return geo[6]
    wr, wk, ku, s, rt, xs = geo
    return 1 if wr * wk >= 32 else 3 if xs == 1 and rt == 1 else 2


def _gvinst():
    out = []
    for (key, mt), geos in GVC.items():
        for k, (geo, vs) in enumerate(geos):
            if key == "router":
                out += [(2, mt, _gvst(geo)) + geo[:6] + (v,) for v in vs]
                continue
            out += [(1, mt, _gvst(geo)) + geo[:6] + (v,) for v in vs] + ([(0, mt, _gvst(geo)) + geo[:6] + (0,)] if k == 0 else [])
    return list(dict.fromkeys(out))


GVI = _gvinst()
_gvmod = None


def _gvext():
    global _gvmod
    if _gvmod is None:
        cases = "\n".join("    case %d: return gvl<%s>(p, st);" % (i, ",".join(map(str, c))) for i, c in enumerate(GVI))
        _gvmod = load_inline("gv21_%x" % (hash(tuple(GVI)) & 0xffffffff), cpp_sources="int64_t gvrun(int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t); int64_t rrrun(int64_t, torch::Tensor, torch::Tensor, torch::Tensor);",
                             cuda_sources=GVCU.replace("@@CASES@@", cases), functions=["gvrun", "rrrun"], extra_cuda_cflags=["-O3", "-lineinfo"], verbose=False)
    return _gvmod


# code-word bit of each value's 3-bit offset and of its sign, per decode variant (see dec0 / dec1)
_OS = ((0, 4, 8, 12, 16, 20, 24, 28), (0, 8, 16, 24, 4, 12, 20, 28))
_SS = ((3, 11, 19, 27, 7, 15, 23, 31), (7, 15, 23, 31, 11, 19, 27, 3))


def _c12(fr, dv=1):
    """bf16 fragments [T, KC, 4, 32, 8] -> (uint8 [T, KC, 3, 32, 16], table, escapes [E, 6] (tile, ch, j, lane, v, xor)) or None"""
    T, KC = fr.shape[:2]
    dev = fr.device
    b = fr.reshape(T, KC, 4, 32, 8).view(torch.int16).int() & 0xffff
    h7, sg = (b >> 8) & 127, b >> 15
    win = torch.bincount(h7.flatten(), minlength=128).unfold(0, 8, 1).sum(1)
    B = int(win.argmax())
    d = h7 - B
    inw = (d >= 0) & (d < 8)
    d = torch.where(inw, d, 0).long()
    C = ((d << torch.tensor(_OS[dv], device=dev)) | (sg.long() << torch.tensor(_SS[dv], device=dev))).sum(-1)
    C = torch.where(C >= 2 ** 31, C - 2 ** 32, C).int()
    lo = (b & 255).to(torch.uint8)
    p0 = lo[:, :, 0:2].permute(0, 1, 3, 2, 4).reshape(T, KC, 32, 16)
    p1 = lo[:, :, 2:4].permute(0, 1, 3, 2, 4).reshape(T, KC, 32, 16)
    p2 = C.permute(0, 1, 3, 2).contiguous().view(torch.uint8).view(T, KC, 32, 16)
    data = torch.stack((p0, p1, p2), 2).contiguous()
    ei = (~inw).nonzero()
    esc = torch.cat((ei, (h7[tuple(ei.t())] ^ B)[:, None].long()), 1)
    tab = B * 0x01010101 if dv else sum((B + i) << (8 * i) for i in range(8))
    if not torch.equal(_unc12(data, tab, esc, dv), b):
        return None
    return data, tab - 2 ** 64 if tab >= 2 ** 63 else tab, esc


def _unc12(data, tab, esc, dv):
    """independent decoder (mirrors the kernel's byte permutes) for the exactness proof"""
    T, KC = data.shape[:2]
    dev = data.device
    w = data.view(torch.int32).view(T, KC, 3, 32, 4).long() & 0xffffffff
    L = torch.stack((w[:, :, 0, :, 0], w[:, :, 0, :, 1], w[:, :, 0, :, 2], w[:, :, 0, :, 3], w[:, :, 1, :, 0], w[:, :, 1, :, 1], w[:, :, 1, :, 2], w[:, :, 1, :, 3]), 2).view(T, KC, 4, 2, 32)
    C = w[:, :, 2].permute(0, 1, 3, 2)
    off = (C[..., None] >> torch.tensor(_OS[dv], device=dev)) & 7
    sgn = (C[..., None] >> torch.tensor(_SS[dv], device=dev)) & 1
    tb = torch.tensor([(tab >> (8 * i)) & 255 for i in range(8)], device=dev)
    hi = ((tab & 255) + off if dv else tb[off]) | (sgn << 7)
    lob = (L.permute(0, 1, 2, 4, 3)[..., None] >> (8 * torch.arange(4, device=dev))) & 255
    val = (hi << 8) | lob.reshape(T, KC, 4, 32, 8)
    if len(esc):
        t, c, j, l, vv, x = esc.t()
        val[t, c, j, l, vv] ^= x << 8
    return val.int()


def _gvesc(esc, T, geo):
    """per-warp escape lists for one launch geometry (32 entries per warp, sorted by (unit, tile)); None if a list overflows.
    esc: [E, 7] (tile, chunk, step, lane, value, xor, matrix)"""
    wr, wk, ku, s, rt, xs = geo
    gx, nw = T // (wr * rt), wr * wk
    out = torch.full((gx * s * nw * 32,), -1, dtype=torch.int64, device=esc.device)
    e = esc[esc[:, 0] < T]
    if len(e):
        t, c, j, l, v, x, h = e.t()
        gw = ((c // (wk * ku)) * gx + t // (wr * rt)) * nw + ((c // ku) % wk) * wr + (t // rt) % wr
        seq = (c % ku) * rt + t % rt
        o = torch.argsort(gw * 4096 + seq)
        gw, ent = gw[o], ((seq << 20) | (l << 15) | ((j * 4 + v // 2) << 11) | ((v % 2) << 10) | (h << 9) | x)[o]
        rank = torch.arange(len(gw), device=gw.device) - torch.searchsorted(gw, gw)
        if int(rank.max()) >= 32:
            return None
        out[gw * 32 + rank] = ent
    return torch.where(out >= 2 ** 31, out - 2 ** 32, out).int()


def _bk(m):
    return 1 if m == 1 else 8 if m <= 8 else 16 if m <= 16 else 32 if m <= 32 else 128 if m == 128 else 256 if m == 256 else 0


def _tm(fn, flush, it=15):
    """median replay time (ms) of a CUDA graph of fn, each replay after a read-only pass over a buffer larger than L2
    (the judge's cold-L2 method; a write flush leaves L2 dirty and penalises bandwidth-heavy kernels unevenly)"""
    g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn(); torch.cuda.synchronize()
        with torch.cuda.graph(g): fn()
    if _FL[0] is not flush:
        _FL[0], _FL[1] = flush, torch.empty((), dtype=flush.dtype, device=flush.device)
    ts = []
    for _ in range(it):
        torch.sum(flush, 0, out=_FL[1]); a = torch.cuda.Event(True); b = torch.cuda.Event(True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    ts.sort(); return ts[len(ts) // 2]


_FL = [None, None]


def _gvok(st, plan, exact, w, mr):
    """exact on every one-hot column, close on random rows"""
    ext, n, K = _gvext(), exact.shape[0], exact.shape[1]
    mm = 8 if plan[0] >= 2000 else (1 if plan[0] < 1002 else 8) if plan[0] >= 1000 else 8 * GVI[plan[0]][1]
    dt = exact.dtype
    eye = torch.eye(K, dtype=torch.bfloat16, device=w.device).view(K // mm, mm, K)
    got = torch.empty((K // mm, mm, n), dtype=dt, device=w.device)
    for i in range(K // mm):
        if _gvcall(st, plan, eye[i], got[i]):
            return False
    if not torch.equal(got.view(K, n), exact.t()):
        return False
    x = torch.randn((max(mm, mr), K), device=w.device).to(torch.bfloat16)
    out = torch.empty((max(mm, mr), n), dtype=dt, device=w.device)
    _gvcall(st, plan, x, out)
    ref = x.double() @ w.double().T
    return bool(torch.isfinite(out).all()) and float((out.double() - ref).norm() / ref.norm()) < (2e-5 if dt == torch.float32 else 4e-3)



# ---------------------------------------------------------------- v21: wgmma router (sm_90a) for m 8..256, see bench/wgb.py
# fp32 router as BF16 hi + lo: x (wgmma A, TMA, L2 evict_last) x [hi | lo] tiles (wgmma B, TMA, evict_first) into one fp32
# accumulator; split-K over a z-cluster reduced in DSMEM. Body compiled only for sm_90a (an empty compute_90 kernel fails
# the prepare() checks and is never picked). H100 (bench, us): m128 9.9 vs 18.8, m256 12.4 vs 18.6 for a cuBLAS N=512 GEMM + add.
def _wb_fn(n):
    regs = n // 2
    outs = ", ".join(f'"+f"(d[{i}])' for i in range(regs))
    lst = ",".join(f"%{i}" for i in range(regs))
    return f'''__device__ __forceinline__ void wb{n}(float* d, uint64_t da, uint64_t db, int sd) {{
  asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, %{regs + 2}, 0;\\n"
               "wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{lst}}}, %{regs}, %{regs + 1}, p, 1, 1, 0, 0;\\n}}\\n"
               : {outs} : "l"(da), "l"(db), "r"(sd));
}}
'''


WBCU = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <stdint.h>

struct BP { void* out; int m, N, K, KC, NH; float* ws; int* cnt; const uint32_t* esc; uint32_t T0, U0; unsigned long long* prof; };

__device__ __forceinline__ uint64_t bdesc(uint32_t s) {
  return (uint64_t)((s & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)64 << 32) | (1ull << 62);
}
__device__ __forceinline__ void bwait(uint32_t bar, uint32_t ph) {
  asm volatile("{\n.reg .pred P;\nWT: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n@!P bra WT;\n}\n" :: "r"(bar), "r"(ph) : "memory");
}
__device__ __forceinline__ void barr(uint32_t bar) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(bar) : "memory"); }
__device__ __forceinline__ void btma(uint32_t dst, const CUtensorMap* m, int c0, int c1, uint32_t bar, uint64_t pol) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3}], [%4], %5;\n"
               :: "r"(dst), "l"(m), "r"(c0), "r"(c1), "r"(bar), "l"(pol) : "memory");
}
WBFNS

template <int BN, int QS, int S, int HL, int NWG>
__global__ void __launch_bounds__(NWG * 128 + 32, 1) wgb(const __grid_constant__ CUtensorMap xm, const __grid_constant__ CUtensorMap wm, BP p) {
#if defined(__CUDA_ARCH__) && defined(__CUDA_ARCH_FEAT_SM90_ALL)
  constexpr int NA = HL ? 2 : 1, WT = BN * 128, XT = NWG * 8192, SB = XT + NA * WT, SBA = (SB + 1023) & ~1023, LD = BN + 4, TM = 64 * NWG;
  constexpr int NT = NWG * 128 + 32;
  using O = typename std::conditional<HL, float, __nv_bfloat16>::type;
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw), sb = (sraw + 1023) & ~1023u;
  float* red = (float*)(smem_raw + (sb - sraw));
  const uint32_t full = sb + QS * SBA, empty = full + QS * 8;
  const int tid = threadIdx.x, wg = tid >> 7, w = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int rb = blockIdx.x, tb = blockIdx.y, ks = blockIdx.z;
  const int c0 = p.KC * ks / S, nk = p.KC * (ks + 1) / S - c0;
  if (tid == 0) {
    for (int s = 0; s < QS; s++) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(full + s * 8));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(empty + s * 8), "n"(NWG));
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  float d[BN / 2];
  if (w == NWG * 4) {
    if (lane == 0) {
      uint64_t pw, px;
      asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pw));
      asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(px));
      for (int c = 0; c < nk; c++) {
        const int s = c % QS;
        if (c >= QS) bwait(empty + s * 8, ((c / QS) - 1) & 1);
        const uint32_t bar = full + s * 8, dst = sb + s * SBA;
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(bar), "n"(SB) : "memory");
        btma(dst, &xm, (c0 + c) * 64, tb * TM, bar, px);
        btma(dst + XT, &wm, (c0 + c) * 64, rb * BN, bar, pw);
        if (HL) btma(dst + XT + WT, &wm, (c0 + c) * 64, p.NH + rb * BN, bar, pw);
      }
    }
  } else {
    for (int c = 0; c < nk; c++) {
      const int s = c % QS;
      bwait(full + s * 8, (c / QS) & 1);
      asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
      const uint32_t xa = sb + s * SBA + wg * 8192, wa = sb + s * SBA + XT;
#pragma unroll
      for (int j = 0; j < 4; j++)
#pragma unroll
        for (int h = 0; h < NA; h++) WBN(d, bdesc(xa) + 2 * j, bdesc(wa + h * WT) + 2 * j, (c | j | h) != 0);
      asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
      asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
      if (c > 0 && (tid & 127) == 0) barr(empty + ((c - 1) % QS) * 8);
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    if (nk == 0) for (int i = 0; i < BN / 2; i++) d[i] = 0.f;
  }
  __syncthreads();
  const int tk0 = wg * 64 + (w & 3) * 16 + g;
  O* out = (O*)p.out;
  const int mt = min(TM, p.m - tb * TM);
  if (S == 1) {
    if (w < NWG * 4) {
#pragma unroll
      for (int i = 0; i < BN / 8; i++) {
        const int col = rb * BN + 8 * i + 2 * t;
#pragma unroll
        for (int hh = 0; hh < 2; hh++) {
          const int tk = tk0 + 8 * hh;
          if (tk >= mt) continue;
          O* o = out + (size_t)(tb * TM + tk) * p.N + col;
          if (HL) *(float2*)o = make_float2(d[4 * i + 2 * hh], d[4 * i + 2 * hh + 1]);
          else *(__nv_bfloat162*)o = __floats2bfloat162_rn(d[4 * i + 2 * hh], d[4 * i + 2 * hh + 1]);
        }
      }
    }
    return;
  }
  if (w < NWG * 4) {
#pragma unroll
    for (int i = 0; i < BN / 8; i++) {
      *(float2*)(red + tk0 * LD + 8 * i + 2 * t) = make_float2(d[4 * i], d[4 * i + 1]);
      *(float2*)(red + (tk0 + 8) * LD + 8 * i + 2 * t) = make_float2(d[4 * i + 2], d[4 * i + 3]);
    }
  }
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
  const uint32_t rbase = (uint32_t)__cvta_generic_to_shared(red);
  constexpr int C4 = BN / 4;
  const int tpr = (mt + S - 1) / S, ta = ks * tpr, tz = min(mt, ta + tpr);
  for (int e2 = tid; e2 < (tz - ta) * C4; e2 += NT) {
    const int tk = ta + e2 / C4, c4 = (e2 % C4) * 4;
    float4 v[S];
#pragma unroll
    for (int q = 0; q < S; q++) {
      uint32_t ra; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(ra) : "r"(rbase + (tk * LD + c4) * 4), "r"(q));
      asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];\n" : "=f"(v[q].x), "=f"(v[q].y), "=f"(v[q].z), "=f"(v[q].w) : "r"(ra) : "memory");
    }
#pragma unroll
    for (int q = 1; q < S; q++) { v[0].x += v[q].x; v[0].y += v[q].y; v[0].z += v[q].z; v[0].w += v[q].w; }
    O* o = out + (size_t)(tb * TM + tk) * p.N + rb * BN + c4;
    if (HL) *(float4*)o = v[0];
    else { __nv_bfloat162 o2[2] = {__floats2bfloat162_rn(v[0].x, v[0].y), __floats2bfloat162_rn(v[0].z, v[0].w)}; *(uint2*)o = *(uint2*)o2; }
  }
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
#endif
}

static PFN_cuTensorMapEncodeTiled_v12000 g_benc = nullptr;
static bool bmap(CUtensorMap* m, const void* p, int K, int rows, int box) {
  cuuint64_t dims[2] = {(cuuint64_t)K, (cuuint64_t)rows}, strides[1] = {(cuuint64_t)K * 2};
  cuuint32_t bx[2] = {64, (cuuint32_t)box}, es[2] = {1, 1};
  return g_benc(m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(p), dims, strides, bx, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}
template <int BN, int QS, int S, int HL, int NWG>
static int64_t wbl(BP p, const void* x, const void* W, int wrows, cudaStream_t st) {
  constexpr int NA = HL ? 2 : 1, TM = 64 * NWG;
  if (p.N % BN || S > 16) return -1;
  if (!g_benc) {
    cudaDriverEntryPointQueryResult q;
    if (cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", (void**)&g_benc, cudaEnableDefault, &q) != cudaSuccess || !g_benc) return -3;
  }
  CUtensorMap xm, wm;
  if (!bmap(&xm, x, p.K, p.m, TM) || !bmap(&wm, W, p.K, wrows, BN)) return -4;
  const size_t SBA = ((size_t)NWG * 8192 + NA * BN * 128 + 1023) / 1024 * 1024, ring = QS * SBA + QS * 16, part = (size_t)TM * (BN + 4) * 4;
  const size_t sm = 1024 + (ring > part ? ring : part);
  auto k = wgb<BN, QS, S, HL, NWG>;
  static bool once = false;
  if (!once) {
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    if (S > 8) cudaFuncSetAttribute(k, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    once = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(p.N / BN, (p.m + TM - 1) / TM, S); cfg.blockDim = dim3(NWG * 128 + 32); cfg.dynamicSmemBytes = sm; cfg.stream = st;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = 1; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = S;
  cfg.attrs = at; cfg.numAttrs = S > 1 ? 1 : 0;
  const cudaError_t r = cudaLaunchKernelEx(&cfg, k, xm, wm, p);
  if (r != cudaSuccess) cudaGetLastError();
  return (int64_t)r;
}


// v22 router m128/256 (wgr): wgb's TMA + wgmma main loop, no cluster: split-K over S plain CTAs (grid <= one wave, checked on
// the host), fp32 partials to a global workspace, one release/acquire counter per output tile, then every CTA reduces
// TM / S tokens of the tile from L2 (all S partials of a slice in flight at once).
template <int BN, int QS, int S, int NWG>
__global__ void __launch_bounds__(NWG * 128 + 32, 1) wgr(const __grid_constant__ CUtensorMap xm, const __grid_constant__ CUtensorMap wm, BP p) {
#if defined(__CUDA_ARCH__) && defined(__CUDA_ARCH_FEAT_SM90_ALL)
  constexpr int WT = BN * 128, XT = NWG * 8192, SB = XT + 2 * WT, SBA = (SB + 1023) & ~1023, TM = 64 * NWG, NC = NWG * 128;
  constexpr int TPS = TM / S, E4 = TPS * BN / 4, EC = E4 < NC ? E4 : NC, G = NC / EC;
  static_assert(TM % S == 0 && E4 % EC == 0 && NC % EC == 0 && S % G == 0, "wgr geometry");
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw), sb = (sraw + 1023) & ~1023u;
  float4* red = (float4*)(smem_raw + (sb - sraw));
  const uint32_t full = sb + QS * SBA, empty = full + QS * 8;
  const int tid = threadIdx.x, wg = tid >> 7, w = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int rb = blockIdx.x, tb = blockIdx.y, ks = blockIdx.z, ti = tb * gridDim.x + rb;
  const int c0 = p.KC * ks / S, nk = p.KC * (ks + 1) / S - c0;
  if (tid == 0) {
    for (int s = 0; s < QS; s++) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(full + s * 8));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(empty + s * 8), "n"(NWG));
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  float d[BN / 2];
  const int mt = min(TM, p.m - tb * TM);
  float* wsp = p.ws + (size_t)ti * S * TM * BN;
  if (w == NWG * 4) {
    if (lane == 0) {
      uint64_t pw, px;
      asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pw));
      asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(px));
      for (int c = 0; c < nk; c++) {
        const int s = c % QS;
        if (c >= QS) bwait(empty + s * 8, ((c / QS) - 1) & 1);
        const uint32_t bar = full + s * 8, dst = sb + s * SBA;
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(bar), "n"(SB) : "memory");
        btma(dst + XT, &wm, (c0 + c) * 64, rb * BN, bar, pw);
        btma(dst + XT + WT, &wm, (c0 + c) * 64, p.NH + rb * BN, bar, pw);
        btma(dst, &xm, (c0 + c) * 64, tb * TM, bar, px);
      }
    }
  } else {
    for (int c = 0; c < nk; c++) {
      const int s = c % QS;
      bwait(full + s * 8, (c / QS) & 1);
      asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
      const uint32_t xa = sb + s * SBA + wg * 8192, wa = sb + s * SBA + XT;
#pragma unroll
      for (int j = 0; j < 4; j++)
#pragma unroll
        for (int h = 0; h < 2; h++) WBN(d, bdesc(xa) + 2 * j, bdesc(wa + h * WT) + 2 * j, (c | j | h) != 0);
      asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
      asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
      if (c > 0 && (tid & 127) == 0) barr(empty + ((c - 1) % QS) * 8);
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    if (nk == 0) for (int i = 0; i < BN / 2; i++) d[i] = 0.f;
    const int tk0 = wg * 64 + (w & 3) * 16 + g;
    float* o = wsp + (size_t)ks * TM * BN;
#pragma unroll
    for (int i = 0; i < BN / 8; i++) {
      if (tk0 < mt) __stcg((float2*)(o + tk0 * BN + 8 * i + 2 * t), make_float2(d[4 * i], d[4 * i + 1]));
      if (tk0 + 8 < mt) __stcg((float2*)(o + (tk0 + 8) * BN + 8 * i + 2 * t), make_float2(d[4 * i + 2], d[4 * i + 3]));
    }
  }
  __syncthreads();
  if (tid == 0) {
    int old, v;
    asm volatile("fence.acq_rel.gpu;\natom.add.relaxed.gpu.global.s32 %0, [%1], 1;\n" : "=r"(old) : "l"(p.cnt + ti) : "memory");
    const int tgt = (old / S + 1) * S;
    do { asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(p.cnt + ti) : "memory"); } while (v - tgt < 0);
  }
  __syncthreads();
  if (tid < NC) {
    const int gi = tid / EC, el = tid % EC;
#pragma unroll
    for (int e0 = 0; e0 < E4; e0 += EC) {
      const int e = e0 + el, tk = ks * TPS + e / (BN / 4), c4 = (e % (BN / 4)) * 4;
      float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
      if (tk < mt) {
        float4 u[S / G];
#pragma unroll
        for (int q = 0; q < S / G; q++) u[q] = __ldcg((const float4*)(wsp + ((size_t)(gi + q * G) * TM + tk) * BN + c4));
#pragma unroll
        for (int q = 0; q < S / G; q++) { v.x += u[q].x; v.y += u[q].y; v.z += u[q].z; v.w += u[q].w; }
      }
      if (G > 1) {
        red[gi * EC + el] = v;
        asm volatile("bar.sync 1, %0;\n" :: "n"(NC) : "memory");
        if (gi == 0) {
#pragma unroll
          for (int q = 1; q < G; q++) { const float4 r = red[q * EC + el]; v.x += r.x; v.y += r.y; v.z += r.z; v.w += r.w; }
        }
        asm volatile("bar.sync 1, %0;\n" :: "n"(NC) : "memory");
      }
      if (gi == 0 && tk < mt) *(float4*)((float*)p.out + (size_t)(tb * TM + tk) * p.N + rb * BN + c4) = v;
    }
  }
#endif
}

template <int BN, int QS, int S, int NWG>
static int64_t wrl(BP p, const void* x, const void* W, int wrows, cudaStream_t st) {
  constexpr int TM = 64 * NWG;
  if (p.N % BN || p.KC < S) return -1;
  if (!g_benc) {
    cudaDriverEntryPointQueryResult q;
    if (cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", (void**)&g_benc, cudaEnableDefault, &q) != cudaSuccess || !g_benc) return -3;
  }
  CUtensorMap xm, wm;
  if (!bmap(&xm, x, p.K, p.m, TM) || !bmap(&wm, W, p.K, wrows, BN)) return -4;
  const size_t SBA = ((size_t)NWG * 8192 + 2 * BN * 128 + 1023) / 1024 * 1024, ring = QS * SBA + QS * 16, part = (size_t)NWG * 128 * 16;
  const size_t sm = 1024 + (ring > part ? ring : part);
  auto k = wgr<BN, QS, S, NWG>;
  static int cap = -1;
  if (cap < 0) {
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    int dev, sms, occ = 0; cudaGetDevice(&dev); cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k, NWG * 128 + 32, sm);
    cudaGetLastError();
    cap = sms * occ;
  }
  const dim3 grid(p.N / BN, (p.m + TM - 1) / TM, S);
  if ((int)(grid.x * grid.y * grid.z) > cap) return -5;  // the counter wait needs every CTA resident at once
  wgr<BN, QS, S, NWG><<<grid, NWG * 128 + 32, sm, st>>>(xm, wm, p);
  return (int64_t)cudaGetLastError();
}

// v23 dense BF16 projections at m 32..256 (wgd): plain BF16 W (wgmma B, TMA, evict_first) x x (wgmma A, TMA, evict_last),
// one wave of CTAs (host-checked), BN up to 256 rows per CTA so x is re-read N / BN times; split-K over S CTAs through an
// fp32 workspace + per-tile counter as wgr (S 1: direct BF16 stores), every CTA reduces ceil(TM / S) tokens of its tile.
template <int BN, int QS, int S, int NWG>
__global__ void __launch_bounds__(NWG * 128 + 32, 1) wgd(const __grid_constant__ CUtensorMap xm, const __grid_constant__ CUtensorMap wm, BP p) {
#if defined(__CUDA_ARCH__) && defined(__CUDA_ARCH_FEAT_SM90_ALL)
  constexpr int WT = BN * 128, XT = NWG * 8192, SB = XT + WT, SBA = (SB + 1023) & ~1023, TM = 64 * NWG, NC = NWG * 128;
  constexpr int TPS = (TM + S - 1) / S, E4 = TPS * BN / 4, EC = E4 < NC ? E4 : NC, G = NC / EC, QG = (S + G - 1) / G;
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw), sb = (sraw + 1023) & ~1023u;
  float4* red = (float4*)(smem_raw + (sb - sraw));
  const uint32_t full = sb + QS * SBA, empty = full + QS * 8;
  const int tid = threadIdx.x, wg = tid >> 7, w = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int rb = blockIdx.x, tb = blockIdx.y, ks = blockIdx.z, ti = tb * gridDim.x + rb;
  const int c0 = p.KC * ks / S, nk = p.KC * (ks + 1) / S - c0;
  if (tid == 0) {
    for (int s = 0; s < QS; s++) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(full + s * 8));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(empty + s * 8), "n"(NWG));
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  float d[BN / 2];
  const int mt = min(TM, p.m - tb * TM);
  float* wsp = p.ws + (size_t)ti * S * TM * BN;
  __nv_bfloat16* out = (__nv_bfloat16*)p.out;
  if (w == NWG * 4) {
    if (lane == 0) {
      uint64_t pw, px;
      asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pw));
      asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(px));
      for (int c = 0; c < nk; c++) {
        const int s = c % QS;
        if (c >= QS) bwait(empty + s * 8, ((c / QS) - 1) & 1);
        const uint32_t bar = full + s * 8, dst = sb + s * SBA;
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(bar), "n"(SB) : "memory");
        btma(dst + XT, &wm, (c0 + c) * 64, rb * BN, bar, pw);
        btma(dst, &xm, (c0 + c) * 64, tb * TM, bar, px);
      }
    }
  } else {
    for (int c = 0; c < nk; c++) {
      const int s = c % QS;
      bwait(full + s * 8, (c / QS) & 1);
      asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
      const uint32_t xa = sb + s * SBA + wg * 8192, wa = sb + s * SBA + XT;
#pragma unroll
      for (int j = 0; j < 4; j++) { constexpr int h = 0; WBN(d, bdesc(xa) + 2 * j, bdesc(wa + h * WT) + 2 * j, (c | j | h) != 0); }
      asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
      asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
      if (c > 0 && (tid & 127) == 0) barr(empty + ((c - 1) % QS) * 8);
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    if (nk == 0) for (int i = 0; i < BN / 2; i++) d[i] = 0.f;
    const int tk0 = wg * 64 + (w & 3) * 16 + g;
    if (S == 1) {
#pragma unroll
      for (int i = 0; i < BN / 8; i++) {
        __nv_bfloat16* o = out + (size_t)(tb * TM + tk0) * p.N + rb * BN + 8 * i + 2 * t;
        if (tk0 < mt) *(__nv_bfloat162*)o = __floats2bfloat162_rn(d[4 * i], d[4 * i + 1]);
        if (tk0 + 8 < mt) *(__nv_bfloat162*)(o + 8 * p.N) = __floats2bfloat162_rn(d[4 * i + 2], d[4 * i + 3]);
      }
    } else {
      float* o = wsp + (size_t)ks * TM * BN;
#pragma unroll
      for (int i = 0; i < BN / 8; i++) {
        if (tk0 < mt) __stcg((float2*)(o + tk0 * BN + 8 * i + 2 * t), make_float2(d[4 * i], d[4 * i + 1]));
        if (tk0 + 8 < mt) __stcg((float2*)(o + (tk0 + 8) * BN + 8 * i + 2 * t), make_float2(d[4 * i + 2], d[4 * i + 3]));
      }
    }
  }
  if (S == 1) return;
  __syncthreads();
  if (tid == 0) {
    int old, v;
    asm volatile("fence.acq_rel.gpu;\natom.add.relaxed.gpu.global.s32 %0, [%1], 1;\n" : "=r"(old) : "l"(p.cnt + ti) : "memory");
    const int tgt = (old / S + 1) * S;
    do { asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(p.cnt + ti) : "memory"); } while (v - tgt < 0);
  }
  __syncthreads();
  if (tid < NC) {
    const int gi = tid / EC, el = tid % EC;
    for (int e0 = 0; e0 < E4; e0 += EC) {
      const int e = e0 + el, tk = ks * TPS + e / (BN / 4), c4 = (e % (BN / 4)) * 4;
      const bool on = gi < G && e < E4 && e / (BN / 4) < TPS && tk < mt;
      float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
      if (on) {
        float4 u[QG];
#pragma unroll
        for (int q = 0; q < QG; q++)
          if (gi + q * G < S) u[q] = __ldcg((const float4*)(wsp + ((size_t)(gi + q * G) * TM + tk) * BN + c4));
          else u[q] = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int q = 0; q < QG; q++) { v.x += u[q].x; v.y += u[q].y; v.z += u[q].z; v.w += u[q].w; }
      }
      if (G > 1) {
        if (gi < G) red[gi * EC + el] = v;
        asm volatile("bar.sync 1, %0;\n" :: "n"(NC) : "memory");
        if (gi == 0) {
#pragma unroll
          for (int q = 1; q < G; q++) { const float4 r = red[q * EC + el]; v.x += r.x; v.y += r.y; v.z += r.z; v.w += r.w; }
        }
        asm volatile("bar.sync 1, %0;\n" :: "n"(NC) : "memory");
      }
      if (gi == 0 && on) {
        __nv_bfloat162 o2[2] = {__floats2bfloat162_rn(v.x, v.y), __floats2bfloat162_rn(v.z, v.w)};
        *(uint2*)(out + (size_t)(tb * TM + tk) * p.N + rb * BN + c4) = *(uint2*)o2;
      }
    }
  }
#endif
}

template <int BN, int QS, int S, int NWG>
static int64_t wdl(BP p, const void* x, const void* W, int wrows, cudaStream_t st) {
  constexpr int TM = 64 * NWG;
  if (p.N % BN || p.KC < S) return -1;
  if (!g_benc) {
    cudaDriverEntryPointQueryResult q;
    if (cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", (void**)&g_benc, cudaEnableDefault, &q) != cudaSuccess || !g_benc) return -3;
  }
  CUtensorMap xm, wm;
  if (!bmap(&xm, x, p.K, p.m, TM) || !bmap(&wm, W, p.K, wrows, BN)) return -4;
  const size_t SBA = ((size_t)NWG * 8192 + BN * 128 + 1023) / 1024 * 1024, ring = QS * SBA + QS * 16, part = (size_t)NWG * 128 * 16;
  const size_t sm = 1024 + (ring > part ? ring : part);
  auto k = wgd<BN, QS, S, NWG>;
  static int cap = -1;
  if (cap < 0) {
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    int dev, sms, occ = 0; cudaGetDevice(&dev); cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k, NWG * 128 + 32, sm);
    cudaGetLastError();
    cap = sms * occ;
  }
  const dim3 grid(p.N / BN, (p.m + TM - 1) / TM, S);
  if (S > 1 && (int)(grid.x * grid.y * grid.z) > cap) return -5;  // the counter wait needs every CTA resident at once
  wgd<BN, QS, S, NWG><<<grid, NWG * 128 + 32, sm, st>>>(xm, wm, p);
  return (int64_t)cudaGetLastError();
}

// v24 swap-AB GEMV-ish (wgs) for m <= 32 on plain BF16 W: W rows are the wgmma M side (A from smem, 64 rows per consumer
// warpgroup, TMA evict_first), the TN tokens are N (x tile as B, TMA evict_last, rows >= m zero-filled); a CTA covers
// BM = 64 * NWG rows and K / S, so x is read N / BM times; split-K over S plain CTAs through the fp32 workspace + per-tile
// counter of wgr (grid <= one wave, host-checked), S 1 stores BF16 directly.
template <int TN, int QS, int S, int NWG>
__global__ void __launch_bounds__(NWG * 128 + 32, 1) wgs(const __grid_constant__ CUtensorMap xm, const __grid_constant__ CUtensorMap wm, BP p) {
#if defined(__CUDA_ARCH__) && defined(__CUDA_ARCH_FEAT_SM90_ALL)
  constexpr int BM = 64 * NWG, WTB = NWG * 8192, SB = WTB + TN * 128, SBA = (SB + 1023) & ~1023, NC = NWG * 128;
  constexpr int TPS = (TN + S - 1) / S, E4 = TPS * BM / 4, EC = E4 < NC ? E4 : NC, G = NC / EC, QG = (S + G - 1) / G;
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw), sb = (sraw + 1023) & ~1023u;
  float4* red = (float4*)(smem_raw + (sb - sraw));
  const uint32_t full = sb + QS * SBA, empty = full + QS * 8;
  const int tid = threadIdx.x, wg = tid >> 7, w = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int rb = blockIdx.x, tb = blockIdx.y, ks = blockIdx.z, ti = tb * gridDim.x + rb;
  const int c0 = p.KC * ks / S, nk = p.KC * (ks + 1) / S - c0;
  if (tid == 0) {
    for (int s = 0; s < QS; s++) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(full + s * 8));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(empty + s * 8), "n"(NWG));
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  float d[TN / 2];
  const int mt = min(TN, p.m - tb * TN);
  float* wsp = p.ws + (size_t)ti * S * TN * BM;
  __nv_bfloat16* out = (__nv_bfloat16*)p.out;
  if (w == NWG * 4) {
    if (lane == 0) {
      uint64_t pw, px;
      asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pw));
      asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(px));
      for (int c = 0; c < nk; c++) {
        const int s = c % QS;
        if (c >= QS) bwait(empty + s * 8, ((c / QS) - 1) & 1);
        const uint32_t bar = full + s * 8, dst = sb + s * SBA;
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(bar), "n"(SB) : "memory");
        btma(dst, &wm, (c0 + c) * 64, rb * BM, bar, pw);
        btma(dst + WTB, &xm, (c0 + c) * 64, tb * TN, bar, px);
      }
    }
  } else {
    for (int c = 0; c < nk; c++) {
      const int s = c % QS;
      bwait(full + s * 8, (c / QS) & 1);
      asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
      const uint32_t wa = sb + s * SBA + wg * 8192, xa = sb + s * SBA + WTB;
#pragma unroll
      for (int j = 0; j < 4; j++) { WSN(d, bdesc(wa) + 2 * j, bdesc(xa) + 2 * j, (c | j) != 0); }
      asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
      asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
      if (c > 0 && (tid & 127) == 0) barr(empty + ((c - 1) % QS) * 8);
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    if (nk == 0) for (int i = 0; i < TN / 2; i++) d[i] = 0.f;
    const int r0 = wg * 64 + (w & 3) * 16 + g;  // W row inside the CTA's BM rows; d[4i + 2h + e] = (row r0 + 8h, token 8i + 2t + e)
    if (S == 1) {
#pragma unroll
      for (int i = 0; i < TN / 8; i++)
#pragma unroll
        for (int e = 0; e < 2; e++) {
          const int c = 8 * i + 2 * t + e;
          if (c < mt) {
            __nv_bfloat16* o = out + (size_t)(tb * TN + c) * p.N + rb * BM + r0;
            o[0] = __float2bfloat16_rn(d[4 * i + e]);
            o[8] = __float2bfloat16_rn(d[4 * i + 2 + e]);
          }
        }
    } else {
      float* o = wsp + (size_t)ks * TN * BM;
#pragma unroll
      for (int i = 0; i < TN / 8; i++)
#pragma unroll
        for (int e = 0; e < 2; e++) {
          const int c = 8 * i + 2 * t + e;
          if (c < mt) { __stcg(o + c * BM + r0, d[4 * i + e]); __stcg(o + c * BM + r0 + 8, d[4 * i + 2 + e]); }
        }
    }
  }
  if (S == 1) return;
  __syncthreads();
  if (tid == 0) {
    int old, v;
    asm volatile("fence.acq_rel.gpu;\natom.add.relaxed.gpu.global.s32 %0, [%1], 1;\n" : "=r"(old) : "l"(p.cnt + ti) : "memory");
    const int tgt = (old / S + 1) * S;
    do { asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(p.cnt + ti) : "memory"); } while (v - tgt < 0);
  }
  __syncthreads();
  if (tid < NC) {
    const int gi = tid / EC, el = tid % EC;
    for (int e0 = 0; e0 < E4; e0 += EC) {
      const int e = e0 + el, tk = ks * TPS + e / (BM / 4), c4 = (e % (BM / 4)) * 4;
      const bool on = gi < G && e < E4 && tk < min(mt, (ks + 1) * TPS);
      float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
      if (on) {
        float4 u[QG];
#pragma unroll
        for (int q = 0; q < QG; q++)
          if (gi + q * G < S) u[q] = __ldcg((const float4*)(wsp + ((size_t)(gi + q * G) * TN + tk) * BM + c4));
          else u[q] = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int q = 0; q < QG; q++) { v.x += u[q].x; v.y += u[q].y; v.z += u[q].z; v.w += u[q].w; }
      }
      if (G > 1) {
        if (gi < G) red[gi * EC + el] = v;
        asm volatile("bar.sync 1, %0;\n" :: "n"(NC) : "memory");
        if (gi == 0) {
#pragma unroll
          for (int q = 1; q < G; q++) { const float4 r = red[q * EC + el]; v.x += r.x; v.y += r.y; v.z += r.z; v.w += r.w; }
        }
        asm volatile("bar.sync 1, %0;\n" :: "n"(NC) : "memory");
      }
      if (gi == 0 && on) {
        __nv_bfloat162 o2[2] = {__floats2bfloat162_rn(v.x, v.y), __floats2bfloat162_rn(v.z, v.w)};
        *(uint2*)(out + (size_t)(tb * TN + tk) * p.N + rb * BM + c4) = *(uint2*)o2;
      }
    }
  }
#endif
}

template <int TN, int QS, int S, int NWG>
static int64_t wsl(BP p, const void* x, const void* W, int wrows, cudaStream_t st) {
  constexpr int BM = 64 * NWG;
  if (p.N % BM || p.KC < S) return -1;
  if (!g_benc) {
    cudaDriverEntryPointQueryResult q;
    if (cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", (void**)&g_benc, cudaEnableDefault, &q) != cudaSuccess || !g_benc) return -3;
  }
  CUtensorMap xm, wm;
  if (!bmap(&xm, x, p.K, p.m, TN) || !bmap(&wm, W, p.K, wrows, BM)) return -4;
  const size_t SBA = ((size_t)NWG * 8192 + TN * 128 + 1023) / 1024 * 1024, ring = QS * SBA + QS * 16, part = (size_t)NWG * 128 * 16;
  const size_t sm = 1024 + (ring > part ? ring : part);
  auto k = wgs<TN, QS, S, NWG>;
  static int cap = -1;
  if (cap < 0) {
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    int dev, sms, occ = 0; cudaGetDevice(&dev); cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k, NWG * 128 + 32, sm);
    cudaGetLastError();
    cap = sms * occ;
  }
  const dim3 grid(p.N / BM, (p.m + TN - 1) / TN, S);
  if (S > 1 && (int)(grid.x * grid.y * grid.z) > cap) return -5;  // the counter wait needs every CTA resident at once
  wgs<TN, QS, S, NWG><<<grid, NWG * 128 + 32, sm, st>>>(xm, wm, p);
  return (int64_t)cudaGetLastError();
}

// v25 swap-AB on lossless 12-bit weights (wgq): wgs with the W tile produced by 4 decoder warps. Per K chunk the producer
// bulk-copies the CTA's BM / 16 c12 units (natural-k A-fragment pack, 1536 B each, one contiguous run) and TMA-loads the x
// tile; each decoder warp decodes its units (9-op add decode + per-warp escape list, as bench/wgc3.py, H100-validated)
// into a 4-deep SW128 K-major ring that the consumer warpgroups read as the wgmma A operand (64 W rows each).
__device__ __forceinline__ void qdec(uint32_t L0, uint32_t L1, uint32_t C, uint32_t Tb, uint32_t* a) {
  const uint32_t hl = (C & 0x87878787u) + Tb, hh = (__funnelshift_r(C, C, 4) & 0x87878787u) + Tb;
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, 0x5140;\n" : "=r"(d) : "r"(L0), "r"(hl)); a[0] = d;
  asm("prmt.b32 %0, %1, %2, 0x7362;\n" : "=r"(d) : "r"(L0), "r"(hl)); a[1] = d;
  asm("prmt.b32 %0, %1, %2, 0x5140;\n" : "=r"(d) : "r"(L1), "r"(hh)); a[2] = d;
  asm("prmt.b32 %0, %1, %2, 0x7362;\n" : "=r"(d) : "r"(L1), "r"(hh)); a[3] = d;
}
template <int TN, int QS, int S, int NWG>
__global__ void __launch_bounds__(NWG * 128 + 160, 1) wgq(const __grid_constant__ CUtensorMap xm, const uint8_t* __restrict__ wq, BP p) {
#if defined(__CUDA_ARCH__) && defined(__CUDA_ARCH_FEAT_SM90_ALL)
  constexpr int BM = 64 * NWG, UPC = BM / 16, U = UPC / 4, NB = 4, DT = NWG * 8192, RAW = UPC * 1536, XT = TN * 128;
  constexpr int SB = XT + RAW, SBA = (SB + 1023) & ~1023, NC = NWG * 128;
  constexpr int TPS = (TN + S - 1) / S, E4 = TPS * BM / 4, EC = E4 < NC ? E4 : NC, G = NC / EC, QG = (S + G - 1) / G;
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw), sb = (sraw + 1023) & ~1023u;
  float4* red = (float4*)(smem_raw + (sb - sraw));
  const uint32_t dec0 = sb, st0 = sb + NB * DT, bars = st0 + QS * SBA;
  const uint32_t full = bars, sempty = bars + QS * 8, bfull = bars + 2 * QS * 8, bempty = bfull + NB * 8;
  const int tid = threadIdx.x, wg = tid >> 7, w = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int rb = blockIdx.x, tb = blockIdx.y, ks = blockIdx.z, ti = tb * gridDim.x + rb;
  const int c0 = p.KC * ks / S, nk = p.KC * (ks + 1) / S - c0;
  if (tid == 0) {
    for (int s = 0; s < QS; s++) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(full + s * 8));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(sempty + s * 8), "n"(4 + NWG));
    }
    for (int b = 0; b < NB; b++) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 4;\n" :: "r"(bfull + b * 8));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(bempty + b * 8), "n"(NWG));
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  float d[TN / 2];
  const int mt = min(TN, p.m - tb * TN);
  float* wsp = p.ws + (size_t)ti * S * TN * BM;
  __nv_bfloat16* out = (__nv_bfloat16*)p.out;
  if (w == NWG * 4 + 4) {
    if (lane == 0) {
      uint64_t px;
      asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(px));
      const uint8_t* wsrc = wq + ((size_t)rb * p.KC + c0) * RAW;
      for (int c = 0; c < nk; c++) {
        const int s = c % QS;
        if (c >= QS) bwait(sempty + s * 8, ((c / QS) - 1) & 1);
        const uint32_t bar = full + s * 8, dst = st0 + s * SBA;
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(bar), "n"(SB) : "memory");
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                     :: "r"(dst + XT), "l"(wsrc + (size_t)c * RAW), "n"(RAW), "r"(bar) : "memory");
        btma(dst, &xm, (c0 + c) * 64, tb * TN, bar, px);
      }
    }
  } else if (w >= NWG * 4) {
    const int dw = w - NWG * 4;
    const uint32_t e = p.esc[((size_t)(rb * S + ks) * 4 + dw) * 32 + lane];
    uint32_t nx = __shfl_sync(~0u, e, 0);
    int ep = 0;
    for (int c = 0; c < nk; c++) {
      const int s = c % QS, b = c % NB;
      bwait(full + s * 8, (c / QS) & 1);
      if (c >= NB) bwait(bempty + b * 8, ((c / NB) - 1) & 1);
      const uint32_t stg = st0 + s * SBA + XT, base0 = dec0 + b * DT;
#pragma unroll
      for (int i = 0; i < U; i++) {
        const int uu = dw + 4 * i;
        uint32_t A[16];
        const uint32_t ra = stg + uu * 1536 + lane * 16;
        uint4 q0, q1, q2;
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(q0.x), "=r"(q0.y), "=r"(q0.z), "=r"(q0.w) : "r"(ra));
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(q1.x), "=r"(q1.y), "=r"(q1.z), "=r"(q1.w) : "r"(ra + 512));
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(q2.x), "=r"(q2.y), "=r"(q2.z), "=r"(q2.w) : "r"(ra + 1024));
        qdec(q0.x, q0.y, q2.x, p.T0, A); qdec(q0.z, q0.w, q2.y, p.T0, A + 4);
        qdec(q1.x, q1.y, q2.z, p.T0, A + 8); qdec(q1.z, q1.w, q2.w, p.T0, A + 12);
        while ((nx >> 20) == (uint32_t)(c * U + i)) {
          if (lane == (int)((nx >> 15) & 31)) {
            const uint32_t rr = (nx >> 11) & 15, xr = (nx & 0x7fu) << (8 + ((nx >> 10) & 1) * 16);
#pragma unroll
            for (int k = 0; k < 16; k++) if (k == (int)rr) A[k] ^= xr;
          }
          nx = ++ep < 32 ? __shfl_sync(~0u, e, ep) : ~0u;
        }
        const uint32_t base = base0 + (uu * 16 + g) * 128;
#pragma unroll
        for (int j = 0; j < 4; j++) {
          const uint32_t c0a = (((2 * j) ^ g) << 4) + 4 * t, c1a = (((2 * j + 1) ^ g) << 4) + 4 * t;
          asm volatile("st.shared.b32 [%0], %1;\n" :: "r"(base + c0a), "r"(A[4 * j]) : "memory");
          asm volatile("st.shared.b32 [%0], %1;\n" :: "r"(base + 1024 + c0a), "r"(A[4 * j + 1]) : "memory");
          asm volatile("st.shared.b32 [%0], %1;\n" :: "r"(base + c1a), "r"(A[4 * j + 2]) : "memory");
          asm volatile("st.shared.b32 [%0], %1;\n" :: "r"(base + 1024 + c1a), "r"(A[4 * j + 3]) : "memory");
        }
      }
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
      __syncwarp();
      if (lane == 0) { barr(bfull + b * 8); barr(sempty + s * 8); }
    }
  } else {
    for (int c = 0; c < nk; c++) {
      const int s = c % QS, b = c % NB;
      bwait(full + s * 8, (c / QS) & 1);
      bwait(bfull + b * 8, (c / NB) & 1);
      asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
      const uint32_t wa = dec0 + b * DT + wg * 8192, xa = st0 + s * SBA;
#pragma unroll
      for (int j = 0; j < 4; j++) { WSN(d, bdesc(wa) + 2 * j, bdesc(xa) + 2 * j, (c | j) != 0); }
      asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
      asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
      if (c > 0 && (tid & 127) == 0) { barr(bempty + ((c - 1) % NB) * 8); barr(sempty + ((c - 1) % QS) * 8); }
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    if (nk == 0) for (int i = 0; i < TN / 2; i++) d[i] = 0.f;
    const int r0 = wg * 64 + (w & 3) * 16 + g;
    if (S == 1) {
#pragma unroll
      for (int i = 0; i < TN / 8; i++)
#pragma unroll
        for (int e = 0; e < 2; e++) {
          const int c = 8 * i + 2 * t + e;
          if (c < mt) {
            __nv_bfloat16* o = out + (size_t)(tb * TN + c) * p.N + rb * BM + r0;
            o[0] = __float2bfloat16_rn(d[4 * i + e]);
            o[8] = __float2bfloat16_rn(d[4 * i + 2 + e]);
          }
        }
    } else {
      float* o = wsp + (size_t)ks * TN * BM;
#pragma unroll
      for (int i = 0; i < TN / 8; i++)
#pragma unroll
        for (int e = 0; e < 2; e++) {
          const int c = 8 * i + 2 * t + e;
          if (c < mt) { __stcg(o + c * BM + r0, d[4 * i + e]); __stcg(o + c * BM + r0 + 8, d[4 * i + 2 + e]); }
        }
    }
  }
  if (S == 1) return;
  __syncthreads();
  if (tid == 0) {
    int old, v;
    asm volatile("fence.acq_rel.gpu;\natom.add.relaxed.gpu.global.s32 %0, [%1], 1;\n" : "=r"(old) : "l"(p.cnt + ti) : "memory");
    const int tgt = (old / S + 1) * S;
    do { asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(p.cnt + ti) : "memory"); } while (v - tgt < 0);
  }
  __syncthreads();
  if (tid < NC) {
    const int gi = tid / EC, el = tid % EC;
    for (int e0 = 0; e0 < E4; e0 += EC) {
      const int e = e0 + el, tk = ks * TPS + e / (BM / 4), c4 = (e % (BM / 4)) * 4;
      const bool on = gi < G && e < E4 && tk < min(mt, (ks + 1) * TPS);
      float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
      if (on) {
        float4 u[QG];
#pragma unroll
        for (int q = 0; q < QG; q++)
          if (gi + q * G < S) u[q] = __ldcg((const float4*)(wsp + ((size_t)(gi + q * G) * TN + tk) * BM + c4));
          else u[q] = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int q = 0; q < QG; q++) { v.x += u[q].x; v.y += u[q].y; v.z += u[q].z; v.w += u[q].w; }
      }
      if (G > 1) {
        if (gi < G) red[gi * EC + el] = v;
        asm volatile("bar.sync 1, %0;\n" :: "n"(NC) : "memory");
        if (gi == 0) {
#pragma unroll
          for (int q = 1; q < G; q++) { const float4 r = red[q * EC + el]; v.x += r.x; v.y += r.y; v.z += r.z; v.w += r.w; }
        }
        asm volatile("bar.sync 1, %0;\n" :: "n"(NC) : "memory");
      }
      if (gi == 0 && on) {
        __nv_bfloat162 o2[2] = {__floats2bfloat162_rn(v.x, v.y), __floats2bfloat162_rn(v.z, v.w)};
        *(uint2*)(out + (size_t)(tb * TN + tk) * p.N + rb * BM + c4) = *(uint2*)o2;
      }
    }
  }
#endif
}

template <int TN, int QS, int S, int NWG>
static int64_t wql(BP p, const void* x, const void* W, cudaStream_t st) {
  constexpr int BM = 64 * NWG;
  if (p.N % BM || p.KC < S) return -1;
  if (!g_benc) {
    cudaDriverEntryPointQueryResult q;
    if (cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", (void**)&g_benc, cudaEnableDefault, &q) != cudaSuccess || !g_benc) return -3;
  }
  CUtensorMap xm;
  if (!bmap(&xm, x, p.K, p.m, TN)) return -4;
  const size_t SBA = ((size_t)TN * 128 + BM / 16 * 1536 + 1023) / 1024 * 1024;
  const size_t ring = (size_t)4 * NWG * 8192 + QS * SBA + (2 * QS + 8) * 8, part = (size_t)NWG * 128 * 16;
  const size_t sm = 1024 + (ring > part ? ring : part);
  auto k = wgq<TN, QS, S, NWG>;
  static int cap = -1;
  if (cap < 0) {
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    int dev, sms, occ = 0; cudaGetDevice(&dev); cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k, NWG * 128 + 160, sm);
    cudaGetLastError();
    cap = sms * occ;
  }
  const dim3 grid(p.N / BM, (p.m + TN - 1) / TN, S);
  if (S > 1 && (int)(grid.x * grid.y * grid.z) > cap) return -5;  // the counter wait needs every CTA resident at once
  wgq<TN, QS, S, NWG><<<grid, NWG * 128 + 160, sm, st>>>(xm, (const uint8_t*)W, p);
  return (int64_t)cudaGetLastError();
}

// c12 swap-AB: W = wq pack [N / BM, KC, BM / 16, 1536] bytes; esc = per-decoder-warp escape lists; T = decode base
int64_t wqrun(int64_t inst, torch::Tensor W, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, torch::Tensor esc, int64_t T, int64_t N) {
  BP p;
  p.out = out.data_ptr(); p.m = x.size(0); p.K = x.size(1); p.N = (int)N; p.KC = p.K / 64; p.NH = p.N;
  p.ws = ws.data_ptr<float>(); p.cnt = cnt.data_ptr<int>() + 9216 + inst * 256;
  p.esc = (const uint32_t*)esc.data_ptr(); p.T0 = (uint32_t)(T & 0xffffffff);
  auto st = at::cuda::getCurrentCUDAStream();
  const void* xp = x.data_ptr(); const void* wp = W.data_ptr();
  switch (inst) {
@@QCASES@@
  }
  return -2;
}

// v26 wgv: c12 GEMV/GEMM on RS wgmma. Each consumer warp owns 16 W rows (one c12 unit per K chunk): it reads its unit's 48 B
// per lane from the stage (bulk-copied with the CTA's other units), decodes the two k32 halves into two A-fragment register
// sets (mma order = the RS A layout) and issues m64nTNk16 wgmmas against the x tile (TMA, SW128, TN tokens) that the whole CTA
// shares. Half 0 / half 1 alternate with wait_group 1 (the prefill w4g pattern), so a half decodes while the other is in flight.
// Split-K over S plain CTAs through the wgr workspace + counter (one wave, host-checked); S 1 stores BF16 directly.
template <int N> __device__ __forceinline__ void wgv_mma(float* d, const uint32_t* a, uint64_t db, int sd);
__device__ __forceinline__ void sto(__nv_bfloat16* d, float v) { *d = __float2bfloat16_rn(v); }
__device__ __forceinline__ void sto(float* d, float v) { *d = v; }
@@WVMMA@@
// v27: HL = router BF16 hi + lo (two units per warp per chunk into one fp32 accumulator, FP32 output); C12 0 = plain BF16
// fragments (2048 B units, no decode). Escape entries carry the matrix in bit 9.
template <int TN, int QS, int S, int NWG, int HL, int C12, int PF, int DA>
__global__ void __launch_bounds__(NWG * 128 + 32, 1) wgv(const __grid_constant__ CUtensorMap xm, const uint8_t* __restrict__ wq, BP p) {
#if defined(__CUDA_ARCH__) && defined(__CUDA_ARCH_FEAT_SM90_ALL)
  constexpr int NA = HL ? 2 : 1, UB = C12 ? 1536 : 2048;
  constexpr int BM = 64 * NWG, UPC = BM / 16, RAW = NA * UPC * UB, XT = TN * 128, SB = XT + RAW, SBA = (SB + 1023) & ~1023, NC = NWG * 128;
  using O = typename std::conditional<HL, float, __nv_bfloat16>::type;
  constexpr int TPS = (TN + S - 1) / S, E4 = TPS * BM / 4, EC = E4 < NC ? E4 : NC, G = NC / EC, QG = (S + G - 1) / G;
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw), sb = (sraw + 1023) & ~1023u;
  float4* red = (float4*)(smem_raw + (sb - sraw));
  const uint32_t full = sb + QS * SBA, empty = full + QS * 8;
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int rb = blockIdx.x, tb = blockIdx.y, ks = blockIdx.z, ti = tb * gridDim.x + rb;
  const int c0 = p.KC * ks / S, nk = p.KC * (ks + 1) / S - c0;
  if (tid == 0) {
    for (int s = 0; s < QS; s++) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(full + s * 8));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(empty + s * 8), "n"(NWG * 4));
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  // v28 PF: clock64 phase profile (per warp, summed into p.prof by lane 0): 0 total, 1 wait full, 2 lds + decode half 0,
  // 3 wgmma issue, 4 wait_group, 5 decode half 1, 6 epilogue store, 7 split-K counter barrier, 8 reduction, 9 chunks,
  // 10 producer wait empty, 11 producer issue, 12 first full wait (in 1 too), 15 warp count; 32-35 globaltimer start min/max, end min/max; 36/37 CTA cycles / ns (thread 0)
  unsigned long long PA[13] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  const unsigned long long pc0 = PF ? (unsigned long long)clock64() : 0ull;
  unsigned long long pg0 = 0, pt = pc0, pgl = 0;
  if (PF) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(pg0));
  auto CK = [&](int i) { if (PF) { const unsigned long long n = clock64(); PA[i] += n - pt; pt = n; } };
  auto PFL = [&]() {
    if (PF) {
      const unsigned long long c1 = clock64();
      unsigned long long g1; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1));
      PA[0] = c1 - pc0;
      if (lane == 0) {
        const int base = w < NWG * 4 ? 0 : 16;
        for (int i = 0; i < 13; i++) atomicAdd(p.prof + base + i, PA[i]);
        atomicAdd(p.prof + base + 15, 1ull);
      }
      if (tid == 0) {
        atomicMin(p.prof + 32, pg0); atomicMax(p.prof + 33, pg0); atomicMin(p.prof + 34, g1); atomicMax(p.prof + 35, g1);
        atomicAdd(p.prof + 36, c1 - pc0); atomicAdd(p.prof + 37, g1 - pg0);
        // v30: per-CTA record (consumer warp 0): sm, total, full, decode, issue, wgwait, epi, ctr, red, chunks, first full, start, loop end, end
        uint32_t smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
        unsigned long long* r = p.prof + 64 + (size_t)(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * 16;
        r[0] = smid; r[1] = PA[0]; r[2] = PA[1]; r[3] = PA[2] + PA[5]; r[4] = PA[3]; r[5] = PA[4]; r[6] = PA[6]; r[7] = PA[7]; r[8] = PA[8];
        r[9] = PA[9]; r[10] = PA[12]; r[11] = pg0; r[12] = pgl; r[13] = g1;
      }
    }
  };
  float d[TN / 2];
  const int mt = min(TN, p.m - tb * TN);
  float* wsp = p.ws + (size_t)ti * S * TN * BM;
  O* out = (O*)p.out;
  if (w == NWG * 4) {
    if (lane == 0) {
      uint64_t px;
      asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(px));
      const uint8_t* wsrc = wq + ((size_t)rb * p.KC + c0) * RAW;
      for (int c = 0; c < nk; c++) {
        const int s = c % QS;
        pt = PF ? (unsigned long long)clock64() : 0ull;
        if (c >= QS) bwait(empty + s * 8, ((c / QS) - 1) & 1);
        CK(10);
        const uint32_t bar = full + s * 8, dst = sb + s * SBA;
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(bar), "n"(SB) : "memory");
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                     :: "r"(dst + XT), "l"(wsrc + (size_t)c * RAW), "n"(RAW), "r"(bar) : "memory");
        btma(dst, &xm, (c0 + c) * 64, tb * TN, bar, px);
        CK(11);
      }
    }
  } else {
    // escape list: loaded at start, first shuffled after the chunk-0 full wait (no exposed DRAM round trip before the ring starts)
    const uint32_t e = C12 ? p.esc[((size_t)(rb * S + ks) * UPC + w) * 32 + lane] : ~0u;
    uint32_t nx = ~0u;
    int ep = 0;
    // DA 2: two A sets, wait_group 1 after each half (v26). DA 4: four A sets over a chunk pair, wait_group 3 before each decode,
    // so up to 3 half-chunk wgmma groups stay in flight while the next half decodes; chunk c - 2 is released at chunk c's half 1.
    uint32_t A[DA][NA][8];
    uint4 q[NA][4];
    auto ldq = [&](uint32_t xa) {
#pragma unroll
      for (int h = 0; h < NA; h++) {
        const uint32_t ra = xa + XT + (h * UPC + w) * UB + lane * 16;
#pragma unroll
        for (int i = 0; i < (C12 ? 3 : 4); i++)
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(q[h][i].x), "=r"(q[h][i].y), "=r"(q[h][i].z), "=r"(q[h][i].w) : "r"(ra + i * 512));
      }
    };
    auto dech = [&](uint32_t (&D)[NA][8], int hf, int c) {
#pragma unroll
      for (int h = 0; h < NA; h++) {
        if (C12) {
          const uint32_t tb0 = h ? p.U0 : p.T0;
          if (hf == 0) { qdec(q[h][0].x, q[h][0].y, q[h][2].x, tb0, D[h]); qdec(q[h][0].z, q[h][0].w, q[h][2].y, tb0, D[h] + 4); }
          else { qdec(q[h][1].x, q[h][1].y, q[h][2].z, tb0, D[h]); qdec(q[h][1].z, q[h][1].w, q[h][2].w, tb0, D[h] + 4); }
        } else {
          const uint4 u0 = q[h][2 * hf], u1 = q[h][2 * hf + 1];
          D[h][0] = u0.x; D[h][1] = u0.y; D[h][2] = u0.z; D[h][3] = u0.w; D[h][4] = u1.x; D[h][5] = u1.y; D[h][6] = u1.z; D[h][7] = u1.w;
        }
      }
      if (C12) {
        while ((nx >> 20) == (uint32_t)c && (hf || ((nx >> 11) & 15) < 8)) {
          if (lane == (int)((nx >> 15) & 31)) {
            const uint32_t rr = (nx >> 11) & 7, hm = (nx >> 9) & 1, xr = (nx & 0x7fu) << (8 + ((nx >> 10) & 1) * 16);
#pragma unroll
            for (int h = 0; h < NA; h++)
#pragma unroll
              for (int k = 0; k < 8; k++) if (k == (int)rr && h == (int)hm) D[h][k] ^= xr;
          }
          nx = ++ep < 32 ? __shfl_sync(~0u, e, ep) : ~0u;
        }
      }
    };
    auto issue = [&](uint32_t (&D)[NA][8], uint64_t db, int first) {
      asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
      for (int j2 = 0; j2 < 2; j2++)
#pragma unroll
        for (int h = 0; h < NA; h++) wgv_mma<TN>(d, D[h] + 4 * j2, db + 2 * j2, (first | j2 | h) != 0);
      asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    };
    auto fullw = [&](int c) {
      pt = PF ? (unsigned long long)clock64() : 0ull;
      bwait(full + (c % QS) * 8, (c / QS) & 1);
      if (PF && c == 0) PA[12] += (unsigned long long)clock64() - pt;
      if (C12 && c == 0) nx = __shfl_sync(~0u, e, 0);
      CK(1);
      if (PF) PA[9]++;
    };
    if constexpr (DA == 4) {
      for (int cp = 0; cp < nk; cp += 2) {
#pragma unroll
        for (int u = 0; u < 2; u++) {
          const int c = cp + u;
          if (c < nk) {
            fullw(c);
            const uint32_t xa = sb + (c % QS) * SBA;
            const uint64_t db = bdesc(xa);
            ldq(xa);
            asm volatile("wgmma.wait_group.sync.aligned 3;\n" ::: "memory");
            CK(4);
            dech(A[2 * u], 0, c);
            CK(2);
            issue(A[2 * u], db, c != 0);
            CK(3);
            asm volatile("wgmma.wait_group.sync.aligned 3;\n" ::: "memory");
            if (c >= 2 && lane == 0) barr(empty + ((c - 2) % QS) * 8);
            CK(4);
            dech(A[2 * u + 1], 1, c);
            CK(5);
            issue(A[2 * u + 1], db + 4, 1);
            CK(3);
          }
        }
      }
    } else {
      for (int c = 0; c < nk; c++) {
        fullw(c);
        const uint32_t xa = sb + (c % QS) * SBA;
        const uint64_t db = bdesc(xa);
        ldq(xa);
        dech(A[0], 0, c);
        CK(2);
        issue(A[0], db, c != 0);
        CK(3);
        if (c > 0) {
          asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
          if (lane == 0) barr(empty + ((c - 1) % QS) * 8);
        }
        CK(4);
        dech(A[1], 1, c);
        CK(5);
        issue(A[1], db + 4, 1);
        CK(3);
        asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
        CK(4);
      }
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    CK(4);
    if (PF) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(pgl));
    const int r0 = w * 16 + g;  // W row inside the CTA's BM rows (warp w = 16 rows); d[4i + 2h + e] = (row r0 + 8h, token 8i + 2t + e)
    if (S == 1) {
#pragma unroll
      for (int i = 0; i < TN / 8; i++)
#pragma unroll
        for (int e2 = 0; e2 < 2; e2++) {
          const int c = 8 * i + 2 * t + e2;
          if (c < mt) {
            O* o = out + (size_t)(tb * TN + c) * p.N + rb * BM + r0;
            sto(o, d[4 * i + e2]);
            sto(o + 8, d[4 * i + 2 + e2]);
          }
        }
    } else {
      float* o = wsp + (size_t)ks * TN * BM;
#pragma unroll
      for (int i = 0; i < TN / 8; i++)
#pragma unroll
        for (int e2 = 0; e2 < 2; e2++) {
          const int c = 8 * i + 2 * t + e2;
          if (c < mt) { __stcg(o + c * BM + r0, d[4 * i + e2]); __stcg(o + c * BM + r0 + 8, d[4 * i + 2 + e2]); }
        }
    }
  }
  CK(6);
  if (S == 1) { PFL(); return; }
  __syncthreads();
  if (tid == 0) {
    int old, v;
    asm volatile("fence.acq_rel.gpu;\natom.add.relaxed.gpu.global.s32 %0, [%1], 1;\n" : "=r"(old) : "l"(p.cnt + ti) : "memory");
    const int tgt = (old / S + 1) * S;
    do { asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(p.cnt + ti) : "memory"); } while (v - tgt < 0);
  }
  __syncthreads();
  CK(7);
  if (tid < NC) {
    const int gi = tid / EC, el = tid % EC;
    for (int e0 = 0; e0 < E4; e0 += EC) {
      const int e = e0 + el, tk = ks * TPS + e / (BM / 4), c4 = (e % (BM / 4)) * 4;
      const bool on = gi < G && e < E4 && tk < min(mt, (ks + 1) * TPS);
      float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
      if (on) {
        float4 u[QG];
#pragma unroll
        for (int q = 0; q < QG; q++)
          if (gi + q * G < S) u[q] = __ldcg((const float4*)(wsp + ((size_t)(gi + q * G) * TN + tk) * BM + c4));
          else u[q] = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int q = 0; q < QG; q++) { v.x += u[q].x; v.y += u[q].y; v.z += u[q].z; v.w += u[q].w; }
      }
      if (G > 1) {
        if (gi < G) red[gi * EC + el] = v;
        asm volatile("bar.sync 1, %0;\n" :: "n"(NC) : "memory");
        if (gi == 0) {
#pragma unroll
          for (int q = 1; q < G; q++) { const float4 r = red[q * EC + el]; v.x += r.x; v.y += r.y; v.z += r.z; v.w += r.w; }
        }
        asm volatile("bar.sync 1, %0;\n" :: "n"(NC) : "memory");
      }
      if (gi == 0 && on) {
        O* o = out + (size_t)(tb * TN + tk) * p.N + rb * BM + c4;
        if (HL) *(float4*)o = v;
        else { __nv_bfloat162 o2[2] = {__floats2bfloat162_rn(v.x, v.y), __floats2bfloat162_rn(v.z, v.w)}; *(uint2*)o = *(uint2*)o2; }
      }
    }
  }
  CK(8);
  PFL();
#endif
}

template <int TN, int QS, int S, int NWG, int HL, int C12, int PF, int DA>
static int64_t wvl(BP p, const void* x, const void* W, cudaStream_t st) {
  constexpr int BM = 64 * NWG;
  if (p.N % BM || p.KC < S) return -1;
  if (!g_benc) {
    cudaDriverEntryPointQueryResult q;
    if (cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", (void**)&g_benc, cudaEnableDefault, &q) != cudaSuccess || !g_benc) return -3;
  }
  CUtensorMap xm;
  if (!bmap(&xm, x, p.K, p.m, TN)) return -4;
  const size_t SBA = ((size_t)TN * 128 + (HL ? 2 : 1) * BM / 16 * (C12 ? 1536 : 2048) + 1023) / 1024 * 1024;
  const size_t ring = QS * SBA + QS * 16, part = (size_t)NWG * 128 * 16;
  const size_t sm = 1024 + (ring > part ? ring : part);
  auto k = wgv<TN, QS, S, NWG, HL, C12, PF, DA>;
  static int cap = -1;
  if (cap < 0) {
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    int dev, sms, occ = 0; cudaGetDevice(&dev); cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k, NWG * 128 + 32, sm);
    cudaGetLastError();
    cap = sms * occ;
  }
  const dim3 grid(p.N / BM, (p.m + TN - 1) / TN, S);
  if (S > 1 && (int)(grid.x * grid.y * grid.z) > cap) return -5;  // the counter wait needs every CTA resident at once
  wgv<TN, QS, S, NWG, HL, C12, PF, DA><<<grid, NWG * 128 + 32, sm, st>>>(xm, (const uint8_t*)W, p);
  return (int64_t)cudaGetLastError();
}

int64_t wvrun(int64_t inst, torch::Tensor W, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, torch::Tensor esc, int64_t T, int64_t U, int64_t N, torch::Tensor prof) {
  BP p;
  p.out = out.data_ptr(); p.m = x.size(0); p.K = x.size(1); p.N = (int)N; p.KC = p.K / 64; p.NH = p.N;
  p.ws = ws.data_ptr<float>(); p.cnt = cnt.data_ptr<int>() + 16384 + inst * 128;
  p.esc = (const uint32_t*)esc.data_ptr(); p.T0 = (uint32_t)(T & 0xffffffff); p.U0 = (uint32_t)(U & 0xffffffff);
  p.prof = (unsigned long long*)prof.data_ptr();
  auto st = at::cuda::getCurrentCUDAStream();
  const void* xp = x.data_ptr(); const void* wp = W.data_ptr();
  switch (inst) {
@@VCASES@@
  }
  return -2;
}

// W: [N, K] (or [2N, K] hi | lo for HL); out: [m, N]
int64_t wbrun(int64_t inst, torch::Tensor W, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt) {
  BP p;
  p.out = out.data_ptr(); p.m = x.size(0); p.K = x.size(1); p.N = out.size(1); p.KC = p.K / 64; p.NH = p.N;
  p.ws = ws.data_ptr<float>(); p.cnt = cnt.data_ptr<int>() + (inst >= 300 ? 4096 + (inst - 300) * 256 : inst >= 200 ? 1024 + (inst - 200) * 256 : inst >= 100 ? (inst - 100) * 64 : 0);
  auto st = at::cuda::getCurrentCUDAStream();
  const void* xp = x.data_ptr(); const void* wp = W.data_ptr(); const int wr = W.size(0);
  switch (inst) {
@@CASES@@
  }
  return -2;
}
'''


WBI = [(64, 6, 16, 1, 1), (64, 6, 16, 1, 2), (64, 6, 8, 1, 2), (64, 4, 16, 1, 2)]  # (BN, ring, split-K, hi+lo, warpgroups)
# v22 wgr (no cluster, workspace + counter split-K, one wave): (BN, ring, split-K, warpgroups); plan id 2100 + index
# H100 (prepare us): m128 wr (32,8,16,2) 6.98 / (32,6,16,2) 7.07 / (64,4,32,2) 7.94 vs wb 9.50; m256 (64,6,16,2) 9.12 /
# (32,4,16,4) 9.44 / (32,5,16,4) 9.22 vs wb 11.84; dropped (losers there): (128,2,64,2) 10.9, (64,4,32,4) 11.1, (128,4,32,2) 11.3, (128,2,64,4) 16.4
WRI = [(64, 4, 32, 2), (32, 8, 16, 2), (32, 6, 16, 2), (64, 6, 16, 2), (32, 4, 16, 4), (32, 5, 16, 4)]
WRB = {8: (0,), 16: (0,), 32: (0,), 128: (1, 3, 100, 101, 102), 256: (2, 103, 104, 105)}
# v23 wgd (dense BF16, one wave, workspace split-K): (BN, ring, split-K, warpgroups); plan id 2200 + index. H100 (prepare us, v12 =
# cuBLASLt pick): qkv m128 19.6-24.7 vs 14.9, m256 24.8-25.7 vs 17.7, o m32 7.65/11.1 vs gv 6.1, o m128 9.1-18.2 vs 7.1, o m256 12.4-21.8
# vs 8.1, pooler m128/256 29.1-45.2 vs 23.1/27.3, unpooler m128/256 29.8-51.1 vs 23.4/23.7 (BN 256 worst): never wins, not built.
WDI = []
_WD = {w: i for i, w in enumerate(WDI)}
WDB = {}
# v24 wgs (swap-AB: W rows = wgmma M from smem, tokens = N): (TN tokens, ring, split-K, warpgroups); plan id 2300 + index.
# Selected only when 2% faster than the best of the v22 candidates (WSMARGIN). DIAG adds the wider list.
DIAG = True
WSMARGIN = 0.98
# judge v24diag (prepare us, wgs vs best v22 candidate): qkv m8 11.71 vs 9.63, m16 11.78 vs 8.86, m32 12.03 vs 11.26; o m8 4.96 vs 4.22,
# m16 5.02 vs 4.45, m32 5.22 vs 5.57 (v12 5.31 kept); pooler m32 18.72 vs 15.42, unpooler m32 19.33 vs 15.42: only o m32 kept
WSB = {("o", 32): [(32, 8, 1, 1), (32, 8, 2, 2), (32, 4, 4, 4)]}
WSI = sorted({c for v in WSB.values() for c in v})
# v25 wgq (swap-AB on c12 weights, decoder warps -> SW128 A ring): (TN tokens, ring, split-K, warpgroups); plan id 2400 + index
# judge v25diag (prepare us, wq vs best old): qkv m8 13.76/9.38, m16 13.86/9.02, m32 14.34/11.74; o m8 6.78/4.32, m16 7.01/4.67;
# pooler m32 21.73/15.87, unpooler m32 21.25/15.74: the 4 decoder warps are latency-bound; not built (WQB emptied below)
WQB = {("qkv_gate", 32): [(32, 6, 7, 2), (32, 8, 3, 1)],
       ("o", 32): [(32, 8, 1, 1), (32, 6, 2, 2)],
       ("pooler", 32): [(32, 6, 4, 2)],
       ("unpooler", 32): [(32, 6, 2, 2)],
       ("qkv_gate", 16): [(16, 6, 7, 2)],
       ("o", 16): [(16, 8, 1, 1)]}
if DIAG:
    for k, v in {("qkv_gate", 32): [(32, 4, 7, 2), (32, 3, 14, 4)], ("o", 32): [(32, 3, 4, 4)], ("qkv_gate", 8): [(8, 6, 7, 2)],
                 ("o", 8): [(8, 8, 1, 1)], ("qkv_gate", 16): [(16, 8, 3, 1)]}.items():
        WQB[k] = WQB.get(k, []) + v
WQB = {}
WQI = sorted({c for v in WQB.values() for c in v})
_WQ = {w: i for i, w in enumerate(WQI)}
# v26 wgv (RS wgmma, c12 decoded straight into the A-fragment registers, x tile shared by the CTA): (TN tokens, ring, split-K,
# warpgroups); plan id 2500 + index; adopted at the 2% margin like wgs
WVB = {("qkv_gate", 32): [(32, 8, 7, 2), (32, 8, 3, 1)],
       ("o", 32): [(32, 8, 1, 1), (32, 8, 2, 2)],
       ("pooler", 32): [(32, 8, 4, 2), (32, 8, 2, 1)],
       ("unpooler", 32): [(32, 8, 2, 2), (32, 8, 1, 1)],
       ("qkv_gate", 128): [(128, 6, 7, 2), (128, 6, 3, 1)],
       ("o", 128): [(128, 6, 2, 2), (128, 6, 1, 1)],
       ("pooler", 128): [(128, 6, 4, 2), (128, 6, 2, 1)],
       ("unpooler", 128): [(128, 6, 2, 2), (128, 6, 1, 1)]}
if DIAG:
    for k, v in {("qkv_gate", 32): [(32, 6, 14, 4)], ("o", 32): [(32, 6, 4, 4)], ("qkv_gate", 16): [(16, 8, 7, 2), (16, 8, 3, 1)],
                 ("o", 16): [(16, 8, 1, 1)], ("qkv_gate", 8): [(8, 8, 7, 2)], ("qkv_gate", 128): [(128, 4, 7, 2)],
                 ("pooler", 128): [(128, 4, 4, 2)], ("qkv_gate", 256): [(128, 6, 3, 1)]}.items():
        WVB[k] = WVB.get(k, []) + v
# v27 router (HL): (TN, ring, split-K, warpgroups, 1, c12); BM 64 = 4 row blocks x 32 (m256: 2 token blocks x 16) K splits
for _m, _tn, _s in ((8, 8, 32), (16, 16, 32), (32, 32, 32), (128, 128, 32), (256, 128, 16)):
    WVB[("router", _m)] = [(_tn, 4, _s, 1, 1, 1), (_tn, 4, _s, 1, 1, 0)]
WVB = {k: [(g if len(g) == 6 else g + (0, 1)) + (0, 2) for g in v] for k, v in WVB.items()}
# v29 (judge v28diag WP: consumer-bound ring, ~880 cyc/chunk, ~2000 cyc exposed escape-list load at start, reduction 4-8K cyc):
# DA 4 = deep decode-ahead (4 A register sets, wait_group 3) and 2-CTA/SM geometries (ring 4, S doubled)
for _k, _v in {("qkv_gate", 32): [(32, 8, 7, 2, 0, 1, 0, 4), (32, 4, 14, 2, 0, 1, 0, 2), (32, 4, 14, 2, 0, 1, 0, 4), (32, 8, 3, 1, 0, 1, 0, 4)],
               ("qkv_gate", 128): [(128, 6, 7, 2, 0, 1, 0, 4), (128, 6, 3, 1, 0, 1, 0, 4)],
               ("o", 128): [(128, 6, 1, 1, 0, 1, 0, 4), (128, 6, 2, 2, 0, 1, 0, 4)],
               ("o", 32): [(32, 8, 1, 1, 0, 1, 0, 4)], ("o", 16): [(16, 8, 1, 1, 0, 1, 0, 4)],
               ("pooler", 32): [(32, 8, 4, 2, 0, 1, 0, 4)], ("unpooler", 32): [(32, 8, 2, 2, 0, 1, 0, 4)],
               ("router", 16): [(16, 4, 32, 1, 1, 1, 0, 4)], ("router", 128): [(128, 4, 32, 1, 1, 1, 0, 4)],
               ("pooler", 128): [(128, 6, 4, 2, 0, 1, 0, 4), (128, 6, 2, 1, 0, 1, 0, 4)],
               ("unpooler", 128): [(128, 6, 2, 2, 0, 1, 0, 4), (128, 6, 1, 1, 0, 1, 0, 4)]}.items():
    WVB[_k] = WVB.get(_k, []) + _v
# v28 diag: clock64 phase profiles (PF 1 twins, run after selection; not candidates)
WVP = [("qkv_gate", 32, (32, 8, 7, 2, 0, 1, 1, 2)), ("qkv_gate", 32, (32, 8, 7, 2, 0, 1, 1, 4)), ("qkv_gate", 32, (32, 4, 14, 2, 0, 1, 1, 4)),
       ("qkv_gate", 128, (128, 6, 7, 2, 0, 1, 1, 4)), ("o", 128, (128, 6, 1, 1, 0, 1, 1, 2)), ("o", 128, (128, 6, 1, 1, 0, 1, 1, 4)),
       ("o", 16, (16, 8, 1, 1, 0, 1, 1, 4)), ("pooler", 128, (128, 6, 4, 2, 0, 1, 1, 4)), ("unpooler", 128, (128, 6, 1, 1, 0, 1, 1, 4))] if DIAG else []
WVI = sorted({c for v in WVB.values() for c in v} | {g for _, _, g in WVP})
_WV = {w: i for i, w in enumerate(WVI)}
_WS = {w: i for i, w in enumerate(WSI)}
_wbmod = [None]


def _wbsrc():
    """(extension name, CUDA source) of the sm_90a wgmma extension; also compiled for sm_90a on any box by bench/wb_compile_chk.py"""
    ns = sorted({i[0] for i in WBI} | {i[0] for i in WRI} | {i[0] for i in WDI} | {i[0] for i in WSI} | {i[0] for i in WQI})
    cases = "\n".join("    case %d: return wbl<%s>(p, xp, wp, wr, st);" % (k, ",".join(map(str, c))) for k, c in enumerate(WBI))
    cases += "\n" + "\n".join("    case %d: return wrl<%s>(p, xp, wp, wr, st);" % (100 + k, ",".join(map(str, c))) for k, c in enumerate(WRI))
    cases += "\n" + "\n".join("    case %d: return wdl<%s>(p, xp, wp, wr, st);" % (200 + k, ",".join(map(str, c))) for k, c in enumerate(WDI))
    cases += "\n" + "\n".join("    case %d: return wsl<%s>(p, xp, wp, wr, st);" % (300 + k, ",".join(map(str, c))) for k, c in enumerate(WSI))
    src = WBCU.replace("WBFNS", "".join(_wb_fn(n) for n in ns))
    call = "".join(f"if constexpr (BN == {n}) wb{n}(d, bdesc(xa) + 2 * j, bdesc(wa + h * WT) + 2 * j, (c | j | h) != 0); " for n in ns)
    src = src.replace("WBN(d, bdesc(xa) + 2 * j, bdesc(wa + h * WT) + 2 * j, (c | j | h) != 0);", "{ " + call + "}")
    src = src.replace("@@QCASES@@", "\n".join("    case %d: return wql<%s>(p, xp, wp, st);" % (k, ",".join(map(str, c))) for k, c in enumerate(WQI)))
    src = src.replace("WSN(d, bdesc(wa) + 2 * j, bdesc(xa) + 2 * j, (c | j) != 0);",
                      "".join(f"if constexpr (TN == {n}) wb{n}(d, bdesc(wa) + 2 * j, bdesc(xa) + 2 * j, (c | j) != 0); " for n in ns))
    src = src.replace("@@WVMMA@@", _wv_mma(sorted({i[0] for i in WVI})))
    src = src.replace("@@VCASES@@", "\n".join("    case %d: return wvl<%s>(p, xp, wp, st);" % (k, ",".join(map(str, c))) for k, c in enumerate(WVI)))
    name = "wb27_%x" % (hash(tuple(WBI) + tuple(WRI) + tuple(WDI) + tuple(WSI) + tuple(WQI) + tuple(WVI)) & 0xffffffff)
    return name, src.replace("@@CASES@@", cases)


WBCPP = ("int64_t wbrun(int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);"
         "int64_t wqrun(int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t);"
         "int64_t wvrun(int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, torch::Tensor);")


def _wbext():
    if _wbmod[0] is None:
        try:
            if torch.cuda.get_device_capability()[0] != 9:
                raise RuntimeError("sm_90 only")
            name, src = _wbsrc()
            _wbmod[0] = load_inline(name, cpp_sources=WBCPP, cuda_sources=src, functions=["wbrun", "wqrun", "wvrun"],
                                    extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
        except Exception as e:
            print("wb unavailable:", str(e)[:200], flush=True)
            _wbmod[0] = False
    return _wbmod[0]


RRB = {1: (0, 1)}  # fp32 router kernels per bucket (m8 rr: 7.7-14.9 us on the H200, v12 4.7)


def _name(i):
    return "wv%s" % ".".join(map(str, WVI[i - 2500])) if i >= 2500 else "wq%s" % ".".join(map(str, WQI[i - 2400])) if i >= 2400 else "ws%s" % ".".join(map(str, WSI[i - 2300])) if i >= 2300 else "wd%s" % ".".join(map(str, WDI[i - 2200])) if i >= 2200 else "wr%d" % (i - 2100) if i >= 2100 else "wb%d" % (i - 2000) if i >= 2000 else "rr%d" % (i - 1000) if i >= 1000 else ".".join(map(str, GVI[i]))


def _gvcall(st, p, x, out):
    if p[0] >= 2500:
        return _wbext().wvrun(p[0] - 2500, p[1], x, out, st["gvws"], st["wrcnt"], p[2], p[3], p[6], p[4], st["wvprof"])
    if p[0] >= 2400:
        return _wbext().wqrun(p[0] - 2400, p[1], x, out, st["gvws"], st["wrcnt"], p[2], p[3], p[4])
    if p[0] >= 2000:
        return _wbext().wbrun(p[0] - 2000, p[1], x, out, st["gvws"], st["wrcnt"])
    if p[0] >= 1000:
        return _gvext().rrrun(p[0] - 1000, p[1], x, out)
    return _gvext().gvrun(p[0], p[1], p[5], x, out, p[2], st["gvws"], st["gvcnt"], p[3], p[6])


def _pack_nat(w):
    """bf16 [N, K] -> natural-k A fragments [N/16, K/64, 4 steps, 32 lanes, 8] (lane (g, t), step j: rows g, g + 8 at k 16 j + 2 t (+1)
    and 16 j + 8 + 2 t (+1), mma register order); bench/wgc3.py"""
    N, K = w.shape
    W = w.view(N // 16, 2, 8, K // 64, 4, 2, 4, 2)
    fr = torch.stack((W[:, 0, :, :, :, 0, :, 0], W[:, 0, :, :, :, 0, :, 1], W[:, 1, :, :, :, 0, :, 0], W[:, 1, :, :, :, 0, :, 1],
                      W[:, 0, :, :, :, 1, :, 0], W[:, 0, :, :, :, 1, :, 1], W[:, 1, :, :, :, 1, :, 0], W[:, 1, :, :, :, 1, :, 1]), -1)
    return fr.permute(0, 2, 3, 1, 4, 5).contiguous()


def _wqplan(st, key, w, geo):
    """c12 swap-AB plan: natural-k c12 pack regrouped as one contiguous [BM / 16 units x 1536 B] run per (row block, chunk) and the
    per-decoder-warp escape lists of this (BM, S); None if the pack or an escape list fails"""
    TN, QS, S, NWG = geo
    BM, N, K = 64 * NWG, w.shape[0], w.shape[1]
    KC, T, tpb = K // 64, N // 16, NWG * 4
    if (key, N) not in st["wq"]:
        c = _c12(_pack_nat(w), 1)
        st["wq"][(key, N)] = c
        if c is not None:
            st["wq"][(key, N, "esc")] = torch.cat((c[2], torch.zeros_like(c[2][:, :1])), 1)
    c = st["wq"][(key, N)]
    if c is None or N % BM:
        return None
    if (key, N, BM) not in st["wq"]:
        st["wq"][(key, N, BM)] = c[0].view(T // tpb, tpb, KC, 1536).permute(0, 2, 1, 3).contiguous()
    esc = st["wq"][(key, N, "esc")]
    out = torch.full((T // tpb * S * 4 * 32,), -1, dtype=torch.int64, device=w.device)
    if len(esc):
        t, ch, j, l, v, x, h = esc.t()
        c0 = torch.tensor([KC * k // S for k in range(S)], device=w.device)
        kk = torch.searchsorted(c0, ch, right=True) - 1
        uu = t % tpb
        gw = ((t // tpb) * S + kk) * 4 + uu % 4
        seq = (ch - c0[kk]) * (tpb // 4) + uu // 4
        o = torch.argsort(gw * 4096 + seq)
        gw, ent = gw[o], ((seq << 20) | (l << 15) | ((j * 4 + v // 2) << 11) | ((v % 2) << 10) | x)[o]
        rank = torch.arange(len(gw), device=gw.device) - torch.searchsorted(gw, gw)
        if int(rank.max()) >= 32:
            return None
        out[gw * 32 + rank] = ent
    e = torch.where(out >= 2 ** 31, out - 2 ** 32, out).int()
    return (2400 + _WQ[geo], st["wq"][(key, N, BM)], e, c[1], N, w, 0)


def _wv_mma(ns):
    out = []
    for N in ns:
        n = N // 2
        regs = ",".join(f"%{i}" for i in range(n))
        cons = ", ".join(f'"+f"(d[{i}])' for i in range(n))
        out.append(f'''template <> __device__ __forceinline__ void wgv_mma<{N}>(float* d, const uint32_t* a, uint64_t db, int sd) {{
  asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, %{n + 5}, 0;\\nwgmma.mma_async.sync.aligned.m64n{N}k16.f32.bf16.bf16 {{{regs}}}, {{%{n},%{n + 1},%{n + 2},%{n + 3}}}, %{n + 4}, p, 1, 1, 0;\\n}}\\n"
               : {cons} : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(sd));
}}''')
    return "\n".join(out)


def _wvplan(st, key, w, geo):
    """RS-wgmma plan: units [N / BM, KC, NA, BM / 16, 1536 (c12) or 2048 (plain BF16) B] and per-consumer-warp escape lists (one
    unit per matrix per chunk; sorted by (chunk, k32 half, matrix, register) so half-0 patches come first; bit 9 = matrix);
    the router (HL) runs its FP32 weights as BF16 hi + lo. None if a pack fails or a list overflows"""
    TN, QS, S, NWG, HL, C12 = geo[:6]
    BM, N, K = 64 * NWG, w.shape[0], w.shape[1]
    KC, T, tpb = K // 64, N // 16, NWG * 4
    if N % BM:
        return None
    mats = [w.to(torch.bfloat16), (w - w.to(torch.bfloat16).float()).to(torch.bfloat16)] if HL else [w]
    ck = (key, N, C12)
    if ck not in st["wq"]:
        if C12:
            cs = [_c12(_pack_nat(mm), 1) for mm in mats]
            st["wq"][ck] = None if any(c is None for c in cs) else (
                [c[0] for c in cs], [c[1] for c in cs], torch.cat([torch.cat((c[2], torch.full_like(c[2][:, :1], h)), 1) for h, c in enumerate(cs)]))
        else:
            st["wq"][ck] = ([_pack_nat(mm).view(torch.uint8).view(T, KC, 2048) for mm in mats], [0, 0], torch.zeros((0, 7), dtype=torch.long, device=w.device))
    c = st["wq"][ck]
    if c is None:
        return None
    if ck + (BM,) not in st["wq"]:
        st["wq"][ck + (BM,)] = torch.stack([d.reshape(T // tpb, tpb, KC, -1).permute(0, 2, 1, 3) for d in c[0]], 2).contiguous()
    esc = c[2]
    out = torch.full((T // tpb * S * tpb * 32,), -1, dtype=torch.int64, device=w.device)
    if len(esc):
        t, ch, j, l, v, x, h = esc.t()
        c0 = torch.tensor([KC * k // S for k in range(S)], device=w.device)
        kk = torch.searchsorted(c0, ch.contiguous(), right=True) - 1
        gw = ((t // tpb) * S + kk) * tpb + t % tpb
        seq = ch - c0[kk]
        rr = j * 4 + v // 2
        o = torch.argsort((gw * 4096 + seq) * 64 + (rr // 8) * 32 + h * 16 + rr)
        gw, seq = gw[o], seq[o]
        ent = ((seq << 20) | (l[o] << 15) | (rr[o] << 11) | ((v[o] % 2) << 10) | (h[o] << 9) | x[o])
        rank = torch.arange(len(gw), device=gw.device) - torch.searchsorted(gw, gw)
        if int(rank.max()) >= 32:
            return None
        out[gw * 32 + rank] = ent
    e = torch.where(out >= 2 ** 31, out - 2 ** 32, out).int()
    return (2500 + _WV[geo], st["wq"][ck + (BM,)], e, c[1][0], N, w, c[1][-1])


def _wvprofile(st, flush):
    """v28 diag: run each WVP geometry's clock64-profiled twin cold 5 times and print one WP line of per-warp phase cycles"""
    if not WVP or not _wbext():
        return
    names = ["total", "full", "dec0", "issue", "wgwait", "dec1", "epi", "ctr", "red", "", "", "", "first-full"]
    for layer, bm, geo in WVP:
        w = st[layer + "_raw"]
        n = w.shape[0]
        plan = _wvplan(st, layer, w, geo)
        if plan is None:
            print("WP %s m%d %s no plan" % (layer, bm, ".".join(map(str, geo))), flush=True); continue
        x = torch.randn((bm, w.shape[1]), device=w.device).to(torch.bfloat16)
        out = torch.empty((bm, n), dtype=torch.bfloat16, device=w.device)
        acc = torch.zeros(64, dtype=torch.float64, device=w.device)
        ext, ok = 0, True
        span = []
        for r in range(6):
            pr = st["wvprof"]
            pr.zero_(); pr[32] = 2 ** 62; pr[34] = 2 ** 62
            torch.sum(flush, 0, out=_FL[1] if _FL[0] is flush else torch.empty((), device=w.device))
            rc = _gvcall(st, plan, x, out)
            torch.cuda.synchronize()
            if rc:
                ok = False; break
            if r:
                acc += pr[:64].double()
                span.append(((pr[35] - pr[32]).item(), (pr[33] - pr[32]).item(), (pr[35] - pr[34]).item()))
        if not ok:
            print("WP %s m%d %s launch rc %d" % (layer, bm, ".".join(map(str, geo)), rc), flush=True); continue
        cw, pw = max(acc[15].item(), 1), max(acc[31].item(), 1)
        ph = " ".join("%s %.0f" % (nm, acc[i].item() / cw) for i, nm in enumerate(names) if nm)
        mhz = acc[36].item() / max(acc[37].item(), 1) * 1e3
        sp = [sum(v[i] for v in span) / len(span) / 1e3 for i in range(3)]
        print("WP %s m%d %s | span %.2f us, CTA start spread %.2f, end spread %.2f | SM %.0f MHz | per consumer warp cycles: %s | chunks %.1f"
              " | producer: total %.0f wait-empty %.0f issue %.0f" % (layer, bm, ".".join(map(str, geo)), sp[0], sp[1], sp[2], mhz, ph,
                                                                       acc[9].item() / cw, acc[16].item() / pw, acc[26].item() / pw, acc[27].item() / pw), flush=True)
        # v30: per-CTA records of the last run (consumer warp 0): where do the 8-10 us of end spread come from?
        TN, S, NWG = geo[0], geo[2], geo[3]
        nrb, nt = n // (64 * NWG), (bm + TN - 1) // TN
        ncta = nrb * nt * S
        R = st["wvprof"][64:64 + 16 * ncta].view(ncta, 16).cpu().double()
        g0 = R[:, 11].min()
        loop_end, end = (R[:, 12] - g0) / 1e3, (R[:, 13] - g0) / 1e3
        q = lambda v: "/".join("%.2f" % float(v.quantile(t)) for t in (0.0, 0.1, 0.5, 0.9, 1.0))
        qc = lambda v: "/".join("%.0f" % float(v.quantile(t)) for t in (0.0, 0.1, 0.5, 0.9, 1.0))
        print("WC %s m%d %s | %d CTAs | end us min/p10/p50/p90/max %s | loop end us %s | total cyc %s | full %s | issue %s | wgwait %s | red %s | chunks %s"
              % (layer, bm, ".".join(map(str, geo)), ncta, q(end), q(loop_end), qc(R[:, 1]), qc(R[:, 2]), qc(R[:, 4]), qc(R[:, 5]), qc(R[:, 8]), qc(R[:, 9])), flush=True)
        o = end.argsort(descending=True)[:8].tolist()
        print("WC   slowest (rb.tb.ks sm: loop-end/end us, total full dec issue wgwait ctr red cyc): " + "  ".join(
            "%d.%d.%d sm%d: %.2f/%.2f, %d %d %d %d %d %d %d" % (i % nrb, (i // nrb) % nt, i // (nrb * nt), int(R[i, 0]), loop_end[i], end[i], R[i, 1], R[i, 2], R[i, 3],
                                                                 R[i, 4], R[i, 5], R[i, 7], R[i, 8]) for i in o), flush=True)
        ks_end = [float(loop_end[torch.arange(ncta) // (nrb * nt) == k].mean()) for k in range(S)]
        rb_end = [float(loop_end[torch.arange(ncta) % nrb == k].mean()) for k in range(nrb)]
        sm = R[:, 0].long()
        print("WC   loop-end by ks %s | by rb min/max %.2f/%.2f (slowest rb %d) | sm ids of the 8 slowest %s | end-vs-sm corr %.2f" % (
            " ".join("%.1f" % v for v in ks_end), min(rb_end), max(rb_end), max(range(nrb), key=lambda k: rb_end[k]),
            ",".join(str(int(R[i, 0])) for i in o), float(torch.corrcoef(torch.stack((end, sm.double())))[0, 1])), flush=True)


def _gvprep(st, flush):
    dev = flush.device
    st["gvws"] = torch.zeros((1 << 24,), dtype=torch.float32, device=dev)
    st["gvcnt"] = torch.zeros((1 << 16,), dtype=torch.int32, device=dev)
    st["wrcnt"] = torch.zeros((32768,), dtype=torch.int32, device=dev)
    st["wvprof"] = torch.zeros((64 + 16 * 1024,), dtype=torch.int64, device=dev)
    st["wq"] = {}
    noesc = torch.full((1 << 16,), -1, dtype=torch.int32, device=dev)
    st["gvlog"] = []
    ok = {}
    z = torch.zeros(1, device=dev)
    off = _tm(lambda: z.zero_(), flush, 31)
    for layer in ("qkv_gate", "o", "pooler", "unpooler", "router"):
        hl = layer == "router"
        frs = list(st[layer]) if hl else [st[layer][0]]
        css = [[_c12(f, dv) for f in frs] if GVC12 and not hl else [None] for dv in (0, 1)]
        c12 = all(c is not None for cs in css for c in cs)
        if c12:
            esc = torch.cat([torch.cat((c[2], torch.full_like(c[2][:, :1], h)), 1) for h, c in enumerate(css[1])])
        Tf, KC = frs[0].shape[:2]
        w = st[layer + "_raw"]
        for bm, mt in list(GVB.items()) + GVRB:  # router m128/256 as token blocks: 1.6x slower than v12 on the 3090
            key = "unpooler_s" if layer == "unpooler" and bm <= 8 else layer
            n = 1024 if key == "unpooler_s" else Tf * 16
            T, wn = n // 16, w[:n]
            exact = (wn.to(torch.bfloat16).float() + (wn - wn.to(torch.bfloat16).float()).to(torch.bfloat16).float()) if hl else wn
            cands = []
            for k, (geo, vs) in enumerate(GVC.get((key, mt), []) if bm <= 32 else []):
                if not c12 and k and not hl:
                    break
                if geo[1] * geo[2] * geo[3] != KC or T % (geo[0] * geo[4]):
                    continue
                e = _gvesc(esc, T, geo[:6]) if c12 else noesc
                for v in vs if c12 or hl else (0,):
                    fmt = 1 if c12 else 2 if hl else 0
                    i = GVI.index((fmt, mt, _gvst(geo)) + geo[:6] + (v,))
                    if e is None:
                        continue
                    cs = css[v & 1]
                    src = [c[0] for c in cs] if fmt & 1 else frs
                    plan = (i, src[0], e, cs[0][1] if fmt & 1 else 0, n, src[-1], cs[-1][1] if fmt & 1 else 0)
                    if (i, n, bm > 8 * mt) not in ok:
                        ok[(i, n, bm > 8 * mt)] = _gvok(st, plan, exact, wn, bm)
                    if ok[(i, n, bm > 8 * mt)]:
                        cands.append(plan)
            for j in WRB.get(bm, ()) if hl and _wbext() else ():
                plan = (2000 + j, st["router_hl"], noesc, 0, n, wn, 0)  # j >= 100: wgr
                if (plan[0], n, bm) not in ok:
                    ok[(plan[0], n, bm)] = _gvok(st, plan, exact, wn, bm)
                if ok[(plan[0], n, bm)]:
                    cands.append(plan)
            for wd in WDB.get((key, bm), ()) if not hl and _wbext() else ():
                plan = (2200 + _WD[wd], w[:n], noesc, 0, n, wn, 0)
                if (plan[0], n, bm) not in ok:
                    ok[(plan[0], n, bm)] = _gvok(st, plan, exact, wn, bm)
                if ok[(plan[0], n, bm)]:
                    cands.append(plan)
            for ws in WSB.get((key, bm), ()) if not hl and _wbext() else ():
                plan = (2300 + _WS[ws], w[:n], noesc, 0, n, wn, 0)
                if (plan[0], n, bm) not in ok:
                    ok[(plan[0], n, bm)] = _gvok(st, plan, exact, wn, bm)
                if ok[(plan[0], n, bm)]:
                    cands.append(plan)
            for wv in WVB.get((key, bm), ()) if _wbext() else ():
                plan = _wvplan(st, key, w[:n], wv)
                if plan is None:
                    continue
                if (plan[0], n, bm) not in ok:
                    ok[(plan[0], n, bm)] = _gvok(st, plan, exact, wn, bm)
                if ok[(plan[0], n, bm)]:
                    cands.append(plan)
            for wq in WQB.get((key, bm), ()) if not hl and _wbext() else ():
                plan = _wqplan(st, key, w[:n], wq)
                if plan is None:
                    continue
                if (plan[0], n, bm) not in ok:
                    ok[(plan[0], n, bm)] = _gvok(st, plan, exact, wn, bm)
                if ok[(plan[0], n, bm)]:
                    cands.append(plan)
            if not cands and bm > 32:
                continue
            for i in RRB.get(bm, ()) if hl else ():
                plan = (1000 + i, wn.contiguous(), noesc, 0, n, wn, 0)
                if (plan[0], n, 0) not in ok:
                    ok[(plan[0], n, 0)] = _gvok(st, plan, wn, wn, bm)
                if ok[(plan[0], n, 0)]:
                    cands.append(plan)
            x = torch.randn((bm, KC * 64), device=dev).to(torch.bfloat16)
            dt = torch.float32 if hl else torch.bfloat16
            out = torch.empty((bm, n), dtype=dt, device=dev)
            t12 = _tm(lambda: _v12(x, layer, n, st), flush)
            res = [(layer, bm, "v12", t12 - off)]
            tt = []
            pen = lambda p: 1.0 / WSMARGIN if p[0] >= 2300 else 1.0  # v24: a new swap-AB candidate must win by 2%
            for p in cands:
                tt.append((_tm(lambda: _gvcall(st, p, x, out), flush, 9) * pen(p), p))
                res.append((layer, bm, _name(p[0]), tt[-1][0] / pen(p) - off))
            tt.sort(key=lambda r: r[0])
            best = (t12, None)
            old = [r for r in tt if r[1][0] < 2300][:3]
            new = [r for r in tt if r[1][0] >= 2300][:2]
            rt = {}
            for _, p in old + new:  # re-time the leaders (best 3 old, best 2 new), alternating with v12
                a = _tm(lambda: _gvcall(st, p, x, out), flush, 21)
                rt[_name(p[0])] = a - off
                t12 = min(t12, _tm(lambda: _v12(x, layer, n, st), flush, 21))
                if a * pen(p) < best[0]:
                    best = (a * pen(p), p)
            if best[1] is not None and best[0] > t12 * GVMARGIN:
                best = (t12, None)
            st["gvlog"] += res
            bo = min(((a, k) for k, a in rt.items() if not k.startswith(("ws", "wq", "wv"))), default=(t12 - off, "v12"))
            bn = min(((a, k) for k, a in rt.items() if k.startswith(("ws", "wq", "wv"))), default=(float("nan"), "-"))
            if bn[1] != "-" or DIAG:
                print("WD %s m%d pick %s | v12 %.2f old %s %.2f | new %s %.2f" % (layer, bm, _name(best[1][0]) if best[1] is not None else "v12",
                      (t12 - off) * 1e3, bo[1], bo[0] * 1e3, bn[1], bn[0] * 1e3), flush=True)
            print("gv %s m%d v12 %.2f best %s %.2f" % (layer, bm, (t12 - off) * 1e3, _name(best[1][0]) if best[1] is not None else "v12", (best[0] / (pen(best[1]) if best[1] is not None else 1.0) - off) * 1e3), flush=True)
            print("  " + " ".join("%s:%.2f" % (r[2], r[3] * 1e3) for r in res), flush=True)
            if best[1] is not None:
                st["gv"][(layer, bm)] = best[1]

_mod = None
# host-only (g++): the judge's nvcc crashes on these STL containers under -std=c++20
LTCPP = r"""
// ---------------------------------------------------------------- cuBLASLt with an algorithm picked in prepare()
// layout bit 0: row-major descriptors with x as A (else column-major, W as A); bit 1: W pre-transposed [K, N] (packed)
// algorithm handle = layout * 100000 + index into that plan's list (heuristic results, then configs added by lt_add)
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>
#include <map>
#include <tuple>
#include <vector>
struct LtPlan { cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t a, b, c; std::vector<cublasLtMatmulHeuristicResult_t> algos; };
static cublasLtHandle_t g_lt = nullptr;
static void* g_ltws = nullptr;
static const size_t LTWS = 64ull << 20;
static std::map<std::tuple<int, int, int, int, int>, LtPlan> g_plans;
static cublasLtHandle_t lth() { if (!g_lt) { cublasLtCreate(&g_lt); cudaMalloc(&g_ltws, LTWS); } return g_lt; }
static LtPlan& ltplan(int m, int n, int k, int f32, int lay) {
  auto key = std::make_tuple(m, n, k, f32, lay);
  auto it = g_plans.find(key);
  if (it != g_plans.end()) return it->second;
  lth();
  const bool row = lay & 1, pk = lay & 2;
  const cudaDataType_t ct = f32 ? CUDA_R_32F : CUDA_R_16BF;
  LtPlan p;
  cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  cublasOperation_t ta = !row && !pk ? CUBLAS_OP_T : CUBLAS_OP_N, tb = row && !pk ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
  cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
  if (row) {
    cublasLtMatrixLayoutCreate(&p.a, CUDA_R_16BF, m, k, k);
    cublasLtMatrixLayoutCreate(&p.b, CUDA_R_16BF, pk ? k : n, pk ? n : k, pk ? n : k);
    cublasLtMatrixLayoutCreate(&p.c, ct, m, n, n);
    cublasLtOrder_t o = CUBLASLT_ORDER_ROW;
    for (auto d : {p.a, p.b, p.c}) cublasLtMatrixLayoutSetAttribute(d, CUBLASLT_MATRIX_LAYOUT_ORDER, &o, sizeof(o));
  } else {
    cublasLtMatrixLayoutCreate(&p.a, CUDA_R_16BF, pk ? n : k, pk ? k : n, pk ? n : k);
    cublasLtMatrixLayoutCreate(&p.b, CUDA_R_16BF, k, m, k);
    cublasLtMatrixLayoutCreate(&p.c, ct, n, m, n);
  }
  cublasLtMatmulPreference_t pref;
  cublasLtMatmulPreferenceCreate(&pref);
  size_t ws = LTWS;
  cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
  p.algos.resize(32);
  int cnt = 0;
  if (cublasLtMatmulAlgoGetHeuristic(g_lt, p.op, p.a, p.b, p.c, p.c, pref, 32, p.algos.data(), &cnt) != CUBLAS_STATUS_SUCCESS) cnt = 0;
  p.algos.resize(cnt);
  cublasLtMatmulPreferenceDestroy(pref);
  return g_plans.emplace(key, p).first->second;
}
int64_t lt_algos(int64_t m, int64_t n, int64_t k, int64_t f32, int64_t lay) { return (int64_t)ltplan(m, n, k, f32, lay).algos.size(); }
std::vector<int64_t> lt_caps(int64_t f32, int64_t id) {
  cublasLtMatmulAlgo_t a;
  std::vector<int64_t> r;
  if (cublasLtMatmulAlgoInit(lth(), CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF,
                             f32 ? CUDA_R_32F : CUDA_R_16BF, f32 ? CUDA_R_32F : CUDA_R_16BF, (int)id, &a) != CUBLAS_STATUS_SUCCESS) return r;
  for (auto cap : {CUBLASLT_ALGO_CAP_TILE_IDS, CUBLASLT_ALGO_CAP_STAGES_IDS}) {
    size_t bytes = 0;
    cublasLtMatmulAlgoCapGetAttribute(&a, cap, nullptr, 0, &bytes);
    std::vector<uint32_t> v(bytes / 4);
    if (bytes) cublasLtMatmulAlgoCapGetAttribute(&a, cap, v.data(), bytes, &bytes);
    r.push_back((int64_t)v.size());
    for (auto x : v) r.push_back(x);
  }
  for (auto cap : {CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK}) {
    uint32_t v = 0; size_t w = 0;
    cublasLtMatmulAlgoCapGetAttribute(&a, cap, &v, sizeof(v), &w);
    r.push_back(v);
  }
  return r;
}
static bool ltset(cublasLtMatmulAlgo_t& a, int attr, int64_t v) {
  if (attr >= 7) { uint16_t x = (uint16_t)v; return cublasLtMatmulAlgoConfigSetAttribute(&a, (cublasLtMatmulAlgoConfigAttributes_t)attr, &x, sizeof(x)) == CUBLAS_STATUS_SUCCESS; }
  uint32_t x = (uint32_t)v; return cublasLtMatmulAlgoConfigSetAttribute(&a, (cublasLtMatmulAlgoConfigAttributes_t)attr, &x, sizeof(x)) == CUBLAS_STATUS_SUCCESS;
}
// cfg: CUBLASLT_ALGO_CONFIG_{ID, TILE_ID, SPLITK_NUM, REDUCTION_SCHEME, CTA_SWIZZLING, CUSTOM_OPTION, STAGES_ID, INNER_SHAPE_ID, CLUSTER_SHAPE_ID}
int64_t lt_add(int64_t m, int64_t n, int64_t k, int64_t f32, int64_t lay, std::vector<int64_t> cfg) {
  if (cfg.size() != 9) return -1;
  LtPlan& p = ltplan(m, n, k, f32, lay);
  cublasLtMatmulAlgo_t a;
  const cudaDataType_t ct = f32 ? CUDA_R_32F : CUDA_R_16BF;
  if (cublasLtMatmulAlgoInit(g_lt, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, ct, ct, (int)cfg[0], &a) != CUBLAS_STATUS_SUCCESS) return -1;
  for (int i = 1; i < 9; i++) if (!ltset(a, i, cfg[i]) && cfg[i]) return -1;
  if (cfg[2] > 1 && cfg[3] != CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE) return -1;
  uint64_t fl = 0; size_t w = 0;
  if (cublasLtMatmulAlgoCapGetAttribute(&a, CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS, &fl, sizeof(fl), &w) != CUBLAS_STATUS_SUCCESS) return -1;
  if ((fl & CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_TYPE_MASK) != CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32F || (fl & CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_TF32)) return -1;
  cublasLtMatmulHeuristicResult_t r{};
  if (cublasLtMatmulAlgoCheck(g_lt, p.op, p.a, p.b, p.c, p.c, &a, &r) != CUBLAS_STATUS_SUCCESS || r.state != CUBLAS_STATUS_SUCCESS || r.workspaceSize > LTWS) return -1;
  r.algo = a;
  p.algos.push_back(r);
  return lay * 100000 + (int64_t)p.algos.size() - 1;
}
std::vector<int64_t> lt_cfg(int64_t m, int64_t n, int64_t k, int64_t f32, int64_t code) {
  LtPlan& p = ltplan(m, n, k, f32, code / 100000);
  std::vector<int64_t> r;
  const auto& a = p.algos.at(code % 100000).algo;
  for (int i = 0; i < 9; i++) {
    size_t w = 0;
    if (i >= 7) { uint16_t v = 0; cublasLtMatmulAlgoConfigGetAttribute(&a, (cublasLtMatmulAlgoConfigAttributes_t)i, &v, sizeof(v), &w); r.push_back(v); }
    else { uint32_t v = 0; cublasLtMatmulAlgoConfigGetAttribute(&a, (cublasLtMatmulAlgoConfigAttributes_t)i, &v, sizeof(v), &w); r.push_back(v); }
  }
  return r;
}
int64_t lt_mm(torch::Tensor x, torch::Tensor w, torch::Tensor wt, torch::Tensor out, int64_t n, int64_t code) {
  const int m = x.size(0), k = x.size(1), f32 = out.scalar_type() == torch::kFloat32, lay = (int)(code / 100000);
  LtPlan& p = ltplan(m, (int)n, k, f32, lay);
  const float alpha = 1.f, beta = 0.f;
  const void* wp = (lay & 2) ? wt.data_ptr() : w.data_ptr();
  const void* a = (lay & 1) ? x.data_ptr() : wp;
  const void* b = (lay & 1) ? wp : x.data_ptr();
  const auto& al = p.algos.at(code % 100000);
  return (int64_t)cublasLtMatmul(g_lt, p.op, &alpha, a, p.a, b, p.b, &beta, out.data_ptr(), p.c, out.data_ptr(), p.c,
                 &al.algo, g_ltws, LTWS, at::cuda::getCurrentCUDAStream());
}
"""
_DBG = []
RTR = True
GV = True
GVC12 = True
GVMARGIN = 0.99
TUNE = True  # prepare(): time torch vs cuBLASLt algorithms per cuBLAS-served shape
V6 = False  # deep per-CTA ring GEMV: slower than v5 on H200


def _big():
    pr = torch.cuda.get_device_properties(0)
    sm = getattr(pr, "shared_memory_per_block_optin", None)
    return sm >= 180000 if sm else (pr.major, pr.minor) in ((9, 0), (10, 0))


def _ext():
    global _mod
    if _mod is None:
        wpc = next((int(d[6:]) for d in _DBG if d.startswith("-DWPC=")), 16 if _big() else 8)
        _mod = load_inline("skinny_v21_%d" % wpc + "".join(_DBG).replace("-", "_").replace("=", ""),
                           cpp_sources=LTCPP + "void run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t); void run2(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t); void run3(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t);",
                           cuda_sources=CUDA, functions=["run", "run2", "run3", "lt_algos", "lt_mm", "lt_caps", "lt_add", "lt_cfg"], extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo", "-DWPC=%d" % wpc] + ([] if _big() else ["-DNSG=4"]) + [d for d in _DBG if not d.startswith("-DWPC=")], extra_ldflags=_ltflags(),
                           verbose=False)
    return _mod


def _ltflags():
    """link the process's cuBLASLt (torch's nvidia-cublas wheel ships only the versioned soname)"""
    import glob, os
    try:
        import nvidia.cublas as nc
        d = os.path.join(list(nc.__path__)[0], "lib")
    except Exception:
        d = os.path.join(os.environ.get("CUDA_HOME", "/usr/local/cuda"), "lib64")
    so = sorted(glob.glob(os.path.join(d, "libcublasLt.so*")))
    return ["-L" + d, "-l:" + os.path.basename(so[0]) if so else "-lcublasLt", "-Wl,-rpath," + d]


def _tune(st, flush):
    """pick, per cuBLAS-served shape, the fastest of torch's default and every cuBLASLt heuristic algorithm (timed here,
    on this GPU, as the judge times: CUDA graph replays from a cold L2); m 128/256 get the pinned-config search below"""
    ext = _ext()
    tm = lambda fn, it=12: _tm(fn, flush, it)
    dev = flush.device
    jobs = []
    for layer, (N, K) in (("qkv_gate", (2304, 8192)), ("o", (8192, 1024)), ("pooler", (4096, 8192)), ("unpooler", (8192, 4096))):
        for m in (1, 8, 16, 32, 128, 256):
            if _cublas(layer, m):
                jobs.append((layer, m, 1024 if layer == "unpooler" and m <= 8 else N, K, st[layer + "_raw"], st.get(layer + "_t"), torch.bfloat16))
    for m in (128, 256):
        jobs.append(("router", m, 512, 8192, st["router_hl"], st["router_hl_t"], torch.float32))
    t0 = time.time()
    for layer, m, n, K, W, Wt, dt in jobs:
        x = torch.randn((m, K), device=dev).to(torch.bfloat16)
        out = torch.empty((m, n), dtype=dt, device=dev)
        ref = torch.mm(x.float(), W[:n].float().T)
        f32 = int(dt == torch.float32)
        torch_fn = (lambda: torch.mm(x, W[:n].T, out_dtype=dt)) if f32 else (lambda: torch.mm(x, W[:n].T))
        ok = lambda c: ext.lt_mm(x, W, Wt if Wt is not None else W, out, n, c) == 0 and _close(out, ref)
        run = lambda c: ext.lt_mm(x, W, Wt if Wt is not None else W, out, n, c)
        if m < 128 or not LTSEARCH or Wt is None:
            best = (tm(torch_fn), -1)
            for i in range(ext.lt_algos(m, n, K, f32, 0)):
                if ok(i):
                    best = min(best, (tm(lambda: run(i)), i))
            if best[1] >= 0:
                st["lt"][(layer, m)] = best[1]
            continue
        st["lt"][(layer, m)] = _ltsearch(ext, layer, m, n, K, f32, ok, run, torch_fn, flush, t0)


LTSEARCH = True
LTBUDGET = 22.0  # seconds for the whole m128/256 search (10 shapes)
LTW = {"qkv_gate": 1.6, "o": 0.3, "pooler": 1.0, "unpooler": 0.7, "router": 1.6}  # v18 H200: o never beat torch
# H200 picks reported for another submission's pinned search: (layout, epilogue, CUBLASLT_ALGO_CONFIG_* fields) -> seeds
_LTSEED = [(2, [66, 17, 1, 0, 0, 0, 35, 0, 2]), (2, [66, 18, 1, 0, 0, 3, 35, 0, 5]), (3, [66, 20, 1, 0, 0, 4, 35, 0, 2]),
           (0, [66, 92, 1, 0, 0, 0, 35, 0, 5]), (0, [66, 124, 1, 0, 0, 0, 35, 0, 2]), (2, [66, 15, 1, 0, 0, 4, 35, 0, 2]),
           (3, [66, 18, 1, 0, 0, 0, 35, 0, 2]), (2, [66, 15, 1, 0, 0, 0, 35, 0, 2]), (2, [66, 15, 1, 0, 0, 1, 35, 0, 2]),
           (2, [66, 15, 1, 0, 0, 0, 35, 0, 4]), (2, [66, 18, 1, 0, 0, 4, 35, 0, 2]), (3, [66, 18, 1, 0, 0, 2, 35, 0, 5]),
           (3, [66, 18, 1, 0, 0, 5, 35, 0, 2]), (2, [66, 15, 1, 0, 0, 2, 35, 0, 5]), (2, [66, 18, 1, 0, 0, 0, 35, 0, 2]),
           (2, [66, 18, 1, 0, 0, 6, 35, 0, 5])]


def _close(out, ref):
    torch.cuda.synchronize()
    return bool(torch.isfinite(out).all()) and float((out.float() - ref).norm() / ref.norm()) < 1e-2


def _ltsearch(ext, layer, m, n, K, f32, ok, run, torch_fn, flush, t0):
    """heuristic algorithms of all four layouts + seeds + an algo-66 (sm90 nvjet) sweep over tile / stages / cluster, then
    split-K / custom option / cluster / stages refinement of the leaders; all checked numerically and timed (judge timer)"""
    if not _LTDONE and not _LT0:
        _LT0.append(time.time())
    t0 = _LT0[0]
    order = [(l, mm) for l in ("qkv_gate", "o", "pooler", "unpooler", "router") for mm in (128, 256)]
    wsum = sum(LTW[l] for l, _ in order)
    dl = t0 + LTBUDGET * sum(LTW[l] for l, mm in order[:order.index((layer, m)) + 1]) / wsum  # cumulative, weighted
    njob = wsum / LTW[layer]
    tried, res = set(), []
    def tryc(code):
        if code < 0 or code in tried:
            return
        tried.add(code)
        if layer != "router" and ext.lt_cfg(m, n, K, f32, code)[2] > 1:  # v25: split-K may add a reduction kernel; one launch per call
            return
        if ok(code):
            res.append((_tm(lambda: run(code), flush, 5), code))
    def addc(lay, cfg):
        c = ext.lt_add(m, n, K, f32, lay, cfg)
        if c >= 0:
            cfg2 = tuple(ext.lt_cfg(m, n, K, f32, c))
            if (lay, cfg2) in seen:
                return
            seen.add((lay, cfg2))
            tryc(c)
    seen = set()
    t_torch = _tm(torch_fn, flush, 9)
    for lay in range(4):
        for i in range(min(ext.lt_algos(m, n, K, f32, lay), 8)):
            seen.add((lay, tuple(ext.lt_cfg(m, n, K, f32, lay * 100000 + i))))
            tryc(lay * 100000 + i)
    for lay, cfg in _LTSEED:
        addc(lay, cfg)
    caps = ext.lt_caps(f32, 66)
    if caps:
        nt = caps[0]; tiles = caps[1:1 + nt]; ns = caps[1 + nt]; stages = caps[2 + nt:2 + nt + ns]
        cmax, ksup, swz = caps[2 + nt + ns], caps[3 + nt + ns], caps[4 + nt + ns]
        stg = [35] if 35 in stages else stages[:1] or [0]
        for lay in (2, 0, 3, 1):
            for tile in tiles:
                if time.time() > dl - 0.4 * LTBUDGET / njob:
                    break
                for cl in (2, 5):
                    addc(lay, [66, tile, 1, 0, 0, 0, stg[0], 0, cl])
        res.sort()
        for _, code in res[:4]:
            if time.time() > dl:
                break
            lay, cfg = code // 100000, list(ext.lt_cfg(m, n, K, f32, code))
            if cfg[0] != 66:
                continue
            for cu in range(min(cmax, 7) + 1):
                for cl in (0, 2, 3, 4, 5, 6):
                    if time.time() > dl:
                        break
                    addc(lay, cfg[:5] + [cu] + cfg[6:8] + [cl])
            if ksup:
                for sk in (2, 3, 4):
                    addc(lay, cfg[:2] + [sk, 2] + cfg[4:])
            for sg in stages[::max(1, len(stages) // 6)]:
                if time.time() > dl:
                    break
                addc(lay, cfg[:6] + [sg] + cfg[7:])
    res.sort()
    best = (t_torch, -1)
    for _, code in res[:3]:  # re-time the leaders against torch's default
        a = _tm(lambda: run(code), flush, 15)
        t_torch = min(t_torch, _tm(torch_fn, flush, 15))
        if a < best[0]:
            best = (a, code)
    if best[1] >= 0 and best[0] > t_torch * 0.99:
        best = (t_torch, -1)
    _LTDONE.append((layer, m))
    cfg = "torch" if best[1] < 0 else "%d:%s" % (best[1] // 100000, ".".join(map(str, ext.lt_cfg(m, n, K, f32, best[1]))))
    print("lt %s m%d heur %.2f best %s %.2f (%d tried, %.1fs)" % (layer, m, t_torch * 1e3, cfg, best[0] * 1e3, len(tried), time.time() - t0), flush=True)
    return best[1] if best[1] >= 0 else None


_LTDONE, _LT0 = [], []


def _cublas(layer, m):
    return layer != "router" and (m > 1 or layer == "o") and not (layer == "unpooler" and m == 1)


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
            st["router_hl_t"] = torch.cat((hi, lo)).t().contiguous()  # [K, 2N]: one cuBLAS mm for m > 32
            st["router_hl"] = torch.cat((hi, lo)).contiguous()  # [2N, K] for cuBLASLt
        else:
            st[k] = (_pack(w), None)
            st[k + "_t"] = w.t().contiguous()  # [K, N] for the packed cuBLASLt layouts
        st[k + "_raw"] = w
    dev = weights["qkv_gate"].device
    st["ws"] = torch.zeros((256 * 8192,), dtype=torch.float32, device=dev)
    st["cnt"] = torch.zeros((8 * 512,), dtype=torch.int32, device=dev)
    st["none"] = torch.empty(0, dtype=torch.bfloat16, device=dev)
    import threading
    th = [threading.Thread(target=f) for f in (_gvext, _wbext)]  # build the three extensions concurrently
    for t in th:
        t.start()
    _ext()
    for t in th:
        t.join()
    st["lt"] = {}
    if TUNE:
        _tune(st, torch.empty(64 << 20, dtype=torch.float32, device=dev))
    st["gv"] = {}
    if GV:
        _gvprep(st, torch.empty(64 << 20, dtype=torch.float32, device=dev))
        if DIAG:
            _wvprofile(st, torch.empty(64 << 20, dtype=torch.float32, device=dev))
    return st


def kernel_fn(data, state):
    cfg = data["config"]
    layer, n = cfg["layer"], cfg["n"]
    x = data["x"]
    p = state["gv"].get((layer, _bk(x.shape[0])))
    if p is not None and p[4] == n:
        out = torch.empty((x.shape[0], n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
        _gvcall(state, p, x.contiguous(), out)
        return out
    return _v12(x, layer, n, state)


def _v12(x, layer, n, state):
    m = x.shape[0]
    if _cublas(layer, m):  # cuBLAS beats the skinny kernel here (judge v7: 0.45-0.9x); algorithm tuned in prepare()
        i = state["lt"].get((layer, m))
        if i is None:
            return x @ state[layer + "_raw"][:n].T
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        _ext().lt_mm(x.contiguous(), state[layer + "_raw"], state[layer + "_t"], out, n, i)
        return out
    if layer == "router" and m > 32:
        i = state["lt"].get(("router", m))
        if i is None:
            a = torch.mm(x, state["router_hl_t"], out_dtype=torch.float32)
        else:
            a = torch.empty((m, 2 * n), dtype=torch.float32, device=x.device)
            _ext().lt_mm(x.contiguous(), state["router_hl"], state["router_hl_t"], a, 2 * n, i)
        return a[:, :n] + a[:, n:]
    A, A2 = state[layer]
    out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
    if layer == "router" and m <= 32 and RTR:
        _ext().run3(A, A2, x.contiguous(), out, state["ws"], state["cnt"], n)
        return out
    (_ext().run2 if V6 and m <= 32 else _ext().run)(A, A2 if A2 is not None else state["none"], x.contiguous(), out, state["ws"], state["cnt"], n)
    return out
