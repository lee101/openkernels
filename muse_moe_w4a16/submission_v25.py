# okc: problem=muse_moe_w4a16
"""v22 + wgmma on sm_90a (dequantized A from registers, activations as SW128 K-major smem tiles; mma.sync elsewhere on
the same layouts). One persistent kernel: routing -> W13 (SwiGLU fused) -> W2 (+post-norm stats) -> norm + weighted sum.
CTA-cooperative streaming: a producer warp fills a CTA ring of stages (16*RPW-row tiles x 256 K of NVFP4 weights by TMA bulk
copies + the unit's activation rows by cp.async); 8 consumer warps share each stage's activations from smem, dequantize their
own rows (mma.sync A-fragment order, 11-op bit layout) and run mma.sync. Units = (W13|W2, expert, row block, K part, token pass)
from one global queue; K parts meet through fp32 L2 atomics (self-cleaning workspace)."""
from typing import TypedDict
import torch
from torch.utils.cpp_extension import load_inline


class Weights(TypedDict):
    w13: torch.Tensor
    w13_sf: torch.Tensor
    w13_gs: torch.Tensor
    w2: torch.Tensor
    w2_sf: torch.Tensor
    w2_gs: torch.Tensor
    post_norm_w: torch.Tensor


class Config(TypedDict):
    bs: int
    d_hidden: int
    d_expert: int
    n_experts: int
    local_start: int
    local_count: int
    top_k: int
    sharding: str


class Inputs(TypedDict):
    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    config: Config


CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>

#define HD 4096
#define NLOC 32
#define HSB 2304
#define NHS 16
#ifndef TP
#define TP 32
#endif
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
#define WGM 1
#else
#define WGM 0
#endif
#ifndef NCW
#define NCW 8
#endif
#define NTH ((NCW + 1) * 32)
#define TPR (NCW * RPW)
#define WB (TPR * HSB)
#define NTM (TP / 8)
#define SLOTB (WB + TP * 512)
#define XA (TP * 128)
#if WGM
#define TI(w, r) ((((w) >> 2) * 2 + (r)) * 4 + ((w) & 3))
#else
#define TI(w, r) ((w) * RPW + (r))
#endif
#define NRB1 (8192 / (16 * TPR))
#define NRB2 (4096 / (16 * TPR))
#define C_W2 32
#define C_Q 33
#define C_DONE 34
#define C_UC 64
#ifndef LAG
#define LAG 1
#endif
#ifndef CPB
#define CPB 1
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define TMA 1
#define FULLCNT 33
#else
#define TMA 0
#define FULLCNT 32
#endif

struct P {
  const uint8_t* w1; const uint8_t* w2; const float* g1; const float* g2;
  const __nv_bfloat16* x; const int* ids; const float* tw; const float* nw;
  __nv_bfloat16* inter; __nv_bfloat16* y; float* ss; __nv_bfloat16* out; int* ctr; float* ws;
  int M, lo, ks; unsigned long long* prof;
};
struct Hdr { int g, e, rb, ps, s, ntok, flags, pad; };
struct Sm {
  Hdr hdr[NSLOT];
  int cnt[NLOC], off[NLOC], act[NLOC], np[NLOC], base[NLOC];
  int segst[2 * NLOC + 1], segk[2 * NLOC];
  int nseg, na, ntot, nw2, flag;
  int wc[NTH / 32][NLOC];
};

#ifdef PROF
__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
#define PSET(k, v) do { if (lane == 0 && p.prof) p.prof[(blockIdx.x * (NTH / 32) + wid) * 16 + (k)] = (v); } while (0)
#define PADD(k, v) do { if (lane == 0 && p.prof) p.prof[(blockIdx.x * (NTH / 32) + wid) * 16 + (k)] += (v); } while (0)
#else
#define PSET(k, v)
#define PADD(k, v)
#endif

__device__ __forceinline__ void cp16(uint32_t s, const void* g) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(g) : "memory"); }
__device__ __forceinline__ uint32_t bmul(uint32_t a, uint32_t b) { uint32_t d; asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(0x80008000u)); return d; }
__device__ __forceinline__ void mma(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ int ldacq(const int* p) { int v; asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ int atom_ar(int* p, int v) { int o; asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\n" : "=r"(o) : "l"(p), "r"(v) : "memory"); return o; }
__device__ __forceinline__ void red_rel(int* p, int v) { asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" :: "l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ void redf(float* p, float v) { asm volatile("red.relaxed.gpu.global.add.f32 [%0], %1;\n" :: "l"(p), "f"(v) : "memory"); }
__device__ __forceinline__ void cbar() { asm volatile("bar.sync 1, %0;\n" :: "n"(NCW * 32) : "memory"); }
__device__ __forceinline__ uint32_t rotl(uint32_t q, int d) { return __funnelshift_l(q, q, d); }
__device__ __forceinline__ uint32_t sc2(uint32_t w, uint32_t sel) { return __byte_perm(w, 0, sel) * 16u + 0x77007700u; }
__device__ __forceinline__ void mbinit(uint32_t b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(b), "r"(n)); }
__device__ __forceinline__ void mbarrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(b) : "memory"); }
__device__ __forceinline__ void cpa_arrive(uint32_t b) { asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" :: "r"(b) : "memory"); }
__device__ __forceinline__ bool mbtest(uint32_t b, uint32_t ph) {
  uint32_t r;
  asm volatile("{\n.reg .pred p;\nmbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n" : "=r"(r) : "r"(b), "r"(ph) : "memory");
  return r;
}
__device__ __forceinline__ void mbwait(uint32_t b, uint32_t ph) {
#if TMA
  asm volatile("{\n.reg .pred p;\nW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W;\n}\n" :: "r"(b), "r"(ph) : "memory");
#else
  asm volatile("{\n.reg .pred p;\nW: mbarrier.test_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W;\n}\n" :: "r"(b), "r"(ph) : "memory");
#endif
}
#if TMA
__device__ __forceinline__ void expect_tx(uint32_t bar, uint32_t n) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(bar), "r"(n) : "memory"); }
__device__ __forceinline__ void bulk(uint32_t dst, const void* src, uint32_t n, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" :: "r"(dst), "l"(src), "r"(n), "r"(bar) : "memory");
}
#endif

__device__ __forceinline__ uint32_t sel4(const uint4& v, int c) { return c == 0 ? v.x : c == 1 ? v.y : c == 2 ? v.z : v.w; }
#ifdef FAKEDQ
#define DEQ(q, sA, sB) const uint32_t a0 = (q) ^ (sA), a1 = (q) ^ (sB), a2 = (q) + (sA), a3 = (q) + (sB);
#else
#define DEQ(q, sA, sB) \
  const uint32_t a0 = bmul((q) & 0x81C081C0u, sA); \
  const uint32_t a1 = bmul((((q) << 3) & 0x01C001C0u) | (((q) << 1) & 0x80008000u), sB); \
  const uint32_t a2 = bmul(((q) << 6) & 0x81C081C0u, sA); \
  const uint32_t a3 = bmul((((q) >> 5) & 0x01C001C0u) | (((q) << 5) & 0x80008000u), sB);
#endif

#if !WGM
template <int NT>
__device__ __forceinline__ void stage(float (&acc)[RPW][NTM][4], const uint8_t* sl, int wid, int lane) {
  const int g = lane >> 2, t = lane & 3;
  constexpr int CH = NT == 1 ? 4 : (NT == 2 ? 2 : 1);
  const uint8_t* xs = sl + WB;
  uint4 s0[RPW], s1[RPW];
#pragma unroll
  for (int r = 0; r < RPW; r++) { const uint8_t* ws = sl + TI(wid, r) * HSB; s0[r] = *(const uint4*)(ws + 2048 + g * 16); s1[r] = *(const uint4*)(ws + 2048 + (g + 8) * 16); }
#pragma unroll
  for (int c = 0; c < 4; c++) {
    uint4 w[RPW];
#pragma unroll
    for (int r = 0; r < RPW; r++) w[r] = *(const uint4*)(sl + TI(wid, r) * HSB + c * 512 + lane * 16);
#pragma unroll
    for (int j = 0; j < 4; j++) {
      uint32_t b0[NT], b1[NT];
#pragma unroll
      for (int n = 0; n < NT; n++) {
        const uint8_t* xr = xs + c * XA + (n * 8 + g) * 128 + t * 4;
        b0[n] = *(const uint32_t*)(xr + (((2 * j) ^ g) << 4)); b1[n] = *(const uint32_t*)(xr + (((2 * j + 1) ^ g) << 4));
      }
#pragma unroll
      for (int r = 0; r < RPW; r++) {
        const uint32_t sA = sc2(sel4(s0[r], c), 0x4040 + j * 0x0101), sB = sc2(sel4(s1[r], c), 0x4040 + j * 0x0101);
        const uint32_t q = sel4(w[r], j);
        DEQ(q, sA, sB)
#pragma unroll
        for (int n = 0; n < NT; n++) mma(acc[r][(n * CH + j % CH) % NTM], a0, a1, a2, a3, b0[n], b1[n]);
      }
    }
  }
}
#else
__device__ __forceinline__ uint64_t gdesc(uint32_t saddr) {  // K-major SW128: LBO 16, SBO 1024
  return (uint64_t)((saddr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)64 << 32) | (1ull << 62);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_wait1() { asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory"); }
__device__ __forceinline__ void wg_wait0() { asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_rs(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db, int sd);
template <> __device__ __forceinline__ void wg_rs<8>(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db, int sd) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %9, 0;\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0,%1,%2,%3}, {%4,%5,%6,%7}, %8, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db), "r"(sd));
}
template <> __device__ __forceinline__ void wg_rs<16>(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db, int sd) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %13, 0;\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, %12, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db), "r"(sd));
}
template <> __device__ __forceinline__ void wg_rs<32>(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db, int sd) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %21, 0;\nwgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19}, %20, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db), "r"(sd));
}
template <> __device__ __forceinline__ void wg_rs<64>(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db, int sd) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %36, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db), "r"(sd));
}
template <int N>
__device__ __forceinline__ void stage_wg(float (&acc)[RPW][NTM][4], const uint8_t* sl, uint32_t xaddr, int wid, int lane, int sd) {
  const int g = lane >> 2;
  uint4 s0[RPW], s1[RPW];
#pragma unroll
  for (int r = 0; r < RPW; r++) { const uint8_t* ws = sl + TI(wid, r) * HSB; s0[r] = *(const uint4*)(ws + 2048 + g * 16); s1[r] = *(const uint4*)(ws + 2048 + (g + 8) * 16); }
#pragma unroll
  for (int c = 0; c < 4; c++) {
    uint4 w[RPW];
#pragma unroll
    for (int r = 0; r < RPW; r++) w[r] = *(const uint4*)(sl + TI(wid, r) * HSB + c * 512 + lane * 16);
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const uint64_t desc = gdesc(xaddr + c * XA + j * 32);
      uint32_t A[RPW][4];
#pragma unroll
      for (int r = 0; r < RPW; r++) {
        const uint32_t sA = sc2(sel4(s0[r], c), 0x4040 + j * 0x0101), sB = sc2(sel4(s1[r], c), 0x4040 + j * 0x0101);
        const uint32_t q = sel4(w[r], j);
        DEQ(q, sA, sB)
        A[r][0] = a0; A[r][1] = a1; A[r][2] = a2; A[r][3] = a3;
      }
      wg_fence();
#pragma unroll
      for (int r = 0; r < RPW; r++) wg_rs<N>(&acc[r][0][0], A[r][0], A[r][1], A[r][2], A[r][3], desc, sd);
      sd = 1;
      wg_commit();
      wg_wait1();
    }
  }
  wg_wait0();
}
#endif

// unit finished: acc holds this CTA's partial (ks > 1) or full sums for rows of tiles rb*TPR + wid*RPW + r, tokens of pass ps
template <int NT>
__device__ __forceinline__ void epilogue(const P& p, Sm* sm, const uint16_t* list, float (&acc)[RPW][NTM][4], const Hdr& h, int wid, int lane, int& pend) {
  const int g = lane >> 2, t = lane & 3, e = h.e, ntok = h.ntok;
  const uint16_t* lst = list + sm->off[e] + h.ps * TP;
  const int rows = h.g ? 4096 : 8192;
  constexpr int CH = WGM ? 1 : (NT == 1 ? 4 : (NT == 2 ? 2 : 1));
  if (CH > 1) {
#pragma unroll
    for (int r = 0; r < RPW; r++)
#pragma unroll
      for (int n = 0; n < NT; n++)
#pragma unroll
        for (int i = 0; i < 4; i++) {
          float v = acc[r][n * CH][i];
#pragma unroll
          for (int k = 1; k < CH; k++) v += acc[r][n * CH + k][i];
          acc[r][n][i] = v;
        }
  }
  if (p.ks > 1) {  // partials meet in L2; this warp's arrival is checked one stage later (latency hidden)
    float* wsb = p.ws + (h.g ? (size_t)p.M * 8 * 8192 : 0);
#pragma unroll
    for (int r = 0; r < RPW; r++) {
      const int mt = h.rb * TPR + TI(wid, r);
#pragma unroll
      for (int n = 0; n < NT; n++)
#pragma unroll
        for (int i = 0; i < 4; i++) {
          const int tok = n * 8 + 2 * t + (i & 1);
          if (tok < ntok) redf(wsb + (size_t)lst[tok] * rows + mt * 16 + g + 8 * (i >> 1), acc[r][n][i]);
        }
    }
    __syncwarp();
    if (lane == 0) pend = atom_ar(p.ctr + C_UC + ((h.g * NLOC + e) * 64 + h.rb) * 8 + h.ps, 1);
    return;
  }
  if (!h.g) {
    const float gs = p.g1[p.lo + e];
#pragma unroll
    for (int r = 0; r < RPW; r++) {
      const int mt = h.rb * TPR + TI(wid, r);
#pragma unroll
      for (int n = 0; n < NT; n++)
#pragma unroll
        for (int hh = 0; hh < 2; hh++) {
          const int tok = n * 8 + 2 * t + hh;
          const float gt = acc[r][n][hh] * gs, up = acc[r][n][2 + hh] * gs;
          if (tok < ntok) p.inter[(size_t)lst[tok] * HD + mt * 8 + g] = __float2bfloat16(gt / (1.f + __expf(-gt)) * up);
        }
    }
    __syncwarp();
    if (lane == 0) red_rel(p.ctr + e, 1);
  } else {
    const float gs = p.g2[p.lo + e];
    float sa[NT], sb[NT];
#pragma unroll
    for (int n = 0; n < NT; n++) sa[n] = sb[n] = 0.f;
#pragma unroll
    for (int r = 0; r < RPW; r++) {
      const int mt = h.rb * TPR + TI(wid, r);
      const float n0 = p.nw[mt * 16 + g] * gs, n1 = p.nw[mt * 16 + g + 8] * gs;
#pragma unroll
      for (int n = 0; n < NT; n++) {
        const float v0 = acc[r][n][0] * n0, v1 = acc[r][n][1] * n0, v2 = acc[r][n][2] * n1, v3 = acc[r][n][3] * n1;
        const int t0 = n * 8 + 2 * t;
        if (t0 < ntok) { __nv_bfloat16* yr = p.y + (size_t)lst[t0] * HD + mt * 16 + g; yr[0] = __float2bfloat16(v0); yr[8] = __float2bfloat16(v2); }
        if (t0 + 1 < ntok) { __nv_bfloat16* yr = p.y + (size_t)lst[t0 + 1] * HD + mt * 16 + g; yr[0] = __float2bfloat16(v1); yr[8] = __float2bfloat16(v3); }
        sa[n] += v0 * v0 + v2 * v2; sb[n] += v1 * v1 + v3 * v3;
      }
    }
#pragma unroll
    for (int n = 0; n < NT; n++) {
#pragma unroll
      for (int o = 4; o < 32; o <<= 1) { sa[n] += __shfl_xor_sync(0xffffffffu, sa[n], o); sb[n] += __shfl_xor_sync(0xffffffffu, sb[n], o); }
      const int t0 = n * 8 + 2 * t;
      if (g == 0) {
        if (t0 < ntok) atomicAdd(p.ss + lst[t0], sa[n]);
        if (t0 + 1 < ntok) atomicAdd(p.ss + lst[t0 + 1], sb[n]);
      }
    }
    __syncwarp();
    if (lane == 0) red_rel(p.ctr + C_W2, 1);
  }
}

// last arriving warp of a K-split unit: full sums from the workspace (zeroed behind), epilogue for all of the unit's rows
__device__ __noinline__ void finish(const P& p, const Sm* sm, const uint16_t* list, const Hdr h, int lane) {
  const int e = h.e, ntok = h.ntok;
  const uint16_t* lst = list + sm->off[e] + h.ps * TP;
  if (!h.g) {
    const float gs = p.g1[p.lo + e];
    for (int tok = 0; tok < ntok; tok++) {
      const int r = lst[tok];
      for (int c = lane; c < TPR * 8; c += 32) {
        const int mt = h.rb * TPR + (c >> 3);
        float* a = p.ws + (size_t)r * 8192 + mt * 16 + (c & 7);
        const float gt = __ldcg(a) * gs, up = __ldcg(a + 8) * gs;
        __stcg(a, 0.f); __stcg(a + 8, 0.f);
        p.inter[(size_t)r * HD + mt * 8 + (c & 7)] = __float2bfloat16(gt / (1.f + __expf(-gt)) * up);
      }
    }
  } else {
    const float gs = p.g2[p.lo + e];
    float* wsb = p.ws + (size_t)p.M * 8 * 8192;
    for (int tok = 0; tok < ntok; tok++) {
      const int r = lst[tok];
      float s2 = 0.f;
      for (int i = lane; i < TPR * 16; i += 32) {
        const int row = h.rb * TPR * 16 + i;
        float* a = wsb + (size_t)r * 4096 + row;
        const float v = __ldcg(a) * gs * p.nw[row];
        __stcg(a, 0.f);
        p.y[(size_t)r * HD + row] = __float2bfloat16(v);
        s2 += v * v;
      }
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) s2 += __shfl_xor_sync(0xffffffffu, s2, o);
      if (lane == 0) atomicAdd(p.ss + r, s2);
    }
  }
  __syncwarp();
  if (lane == 0) {
    p.ctr[C_UC + ((h.g * NLOC + e) * 64 + h.rb) * 8 + h.ps] = 0;
    red_rel(p.ctr + (h.g ? C_W2 : e), NCW);
  }
}

__device__ __forceinline__ void decode(const Sm* sm, int ks, int u, int& g, int& e, int& rb, int& kp, int& ps) {
  int k = 0;
  while (sm->segst[k + 1] <= u) k++;
  const int code = sm->segk[k];
  g = code >> 6; e = sm->act[code & 63];
  const int np = sm->np[e];
  int v = u - sm->segst[k];
  kp = v % ks; v /= ks; ps = v % np; rb = v / np;
}

__global__ void __launch_bounds__(NTH, CPB) moe_kernel(P p) {
  extern __shared__ __align__(1024) uint8_t smem[];
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5, R = p.M * 8, ks = p.ks;
  Sm* sm = (Sm*)(smem + NSLOT * SLOTB + NSLOT * 16);
  uint16_t* list = (uint16_t*)(sm + 1);
  uint8_t* lid = smem;                       // prologue scratch in the ring
  uint16_t* posi = (uint16_t*)(smem + 4096);
  const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem);
  const uint32_t fullb = sbase + NSLOT * SLOTB, emptyb = fullb + NSLOT * 8;
#ifdef PROF
  PSET(0, gtime());
#endif
#if WGM
  if (sbase & 1023) __trap();
#endif
  if (tid < NSLOT) { mbinit(fullb + tid * 8, FULLCNT); mbinit(emptyb + tid * 8, NCW); }
  if (tid < NLOC) sm->base[tid] = 0;
#if TMA
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#endif
  int u = 0, nx = 0;
  if (tid == NCW * 32) { u = atomicAdd(p.ctr + C_Q, 1); nx = atomicAdd(p.ctr + C_Q, 1); }  // queue heads early: latency hides behind routing
  for (int i = tid; i < R; i += NTH) { unsigned e = p.ids[i] - p.lo; lid[i] = e < NLOC ? e : 0xff; }
  __syncthreads();
  const unsigned ltm = (1u << lane) - 1;
  for (int c0 = 0; c0 < R; c0 += NTH) {  // stable per-expert lists (route order), identical in every CTA
    const int i = c0 + tid;
    const int e = i < R ? lid[i] : 0xff;
    const unsigned m = __match_any_sync(0xffffffffu, e);
    sm->wc[wid][lane] = 0;
    __syncwarp();
    if (e < NLOC && (m & ltm) == 0) sm->wc[wid][e] = __popc(m);
    __syncthreads();
    if (tid < NLOC) { int s = sm->base[tid];
      for (int w = 0; w < NTH / 32; w++) { const int v = sm->wc[w][tid]; sm->wc[w][tid] = s; s += v; }
      sm->base[tid] = s; }
    __syncthreads();
    if (e < NLOC) posi[i] = sm->wc[wid][e] + __popc(m & ltm);
    __syncthreads();
  }
  if (wid == 0) {
    const int c = sm->base[lane];
    int v = c;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) { int n = __shfl_up_sync(0xffffffffu, v, o); if (lane >= o) v += n; }
    sm->cnt[lane] = c; sm->off[lane] = v - c; sm->np[lane] = (c + TP - 1) / TP;
    int rk = 0;
#pragma unroll
    for (int j = 0; j < 32; j++) { const int cj = __shfl_sync(0xffffffffu, c, j); rk += (cj > c) || (cj == c && j < lane); }
    if (c > 0) sm->act[rk] = lane;
    const unsigned am = __ballot_sync(0xffffffffu, c > 0);
    int w2 = c > 0 ? NRB2 * ((c + TP - 1) / TP) * NCW : 0;
#pragma unroll
    for (int o = 16; o; o >>= 1) w2 += __shfl_xor_sync(0xffffffffu, w2, o);
    __syncwarp();
    if (lane == 0) {
      const int na = __popc(am);
      int ns = 0, u = 0;
      auto push = [&](int g, int a) { sm->segst[ns] = u; sm->segk[ns] = g * 64 + a; u += (g ? NRB2 : NRB1) * ks * sm->np[sm->act[a]]; ns++; };
      for (int a = 0; a < na; a++) { push(0, a); if (a >= LAG) push(1, a - LAG); }
      for (int a = na - LAG < 0 ? 0 : na - LAG; a < na; a++) push(1, a);
      sm->segst[ns] = u; sm->nseg = ns; sm->ntot = u; sm->na = na; sm->nw2 = w2;
    }
  }
  __syncthreads();
  for (int i = tid; i < R; i += NTH) { const int e = lid[i]; if (e < NLOC) list[sm->off[e] + posi[i]] = i; }
  __syncthreads();
#ifdef PROF
  PSET(1, gtime());
#endif
  const int ntot = sm->ntot;
  if (wid == NCW) {  // producer
    u = __shfl_sync(0xffffffffu, u, 0); nx = __shfl_sync(0xffffffffu, nx, 0);
    const int nst = NHS / ks;
    unsigned depok = 0;
    int seq = 0;
#ifdef PROF
    long long wemp = 0, wdep = 0;
#endif
    for (; u < ntot;) {
      int g, e, rb, kp, ps; decode(sm, ks, u, g, e, rb, kp, ps);
      const int ntok = min(TP, sm->cnt[e] - ps * TP);
      const uint16_t* lst = list + sm->off[e] + ps * TP;
      const uint8_t* wsrc = (g ? p.w2 + (size_t)((p.lo + e) * 256 + rb * TPR) * NHS * HSB
                               : p.w1 + (size_t)((p.lo + e) * 512 + rb * TPR) * NHS * HSB);
      const uint8_t* xsrc = g ? (const uint8_t*)p.inter : (const uint8_t*)p.x;
      int my[TP / 32];
#pragma unroll
      for (int k = 0; k < TP / 32; k++) { const int i = k * 32 + lane; my[k] = 0; if (i < ntok) { const int r = lst[i]; my[k] = g ? r : r >> 3; } }
      for (int i = 0; i < nst; i++, seq++) {
        const int slot = seq % NSLOT, s = kp * nst + i;
#ifdef PROF
        long long c0 = clock64();
#endif
        if (seq >= NSLOT) mbwait(emptyb + slot * 8, ((seq / NSLOT) - 1) & 1);
#ifdef PROF
        wemp += clock64() - c0;
#endif
        if (lane == 0) { sm->hdr[slot] = Hdr{g, e, rb, ps, s, ntok, (i == 0 ? 1 : 0) | (i == nst - 1 ? 2 : 0) | ((!g && rb == 0 && kp == 0) ? 4 : 0), 0}; __threadfence_block(); }
        const uint32_t dst = sbase + slot * SLOTB;
        const uint32_t fb = fullb + slot * 8;
#if TMA
        if (lane == 0) expect_tx(fb, WB);
        __syncwarp();
        if (lane < TPR) bulk(dst + lane * HSB, wsrc + ((size_t)lane * NHS + s) * HSB, HSB, fb);
#else
        for (int q = lane; q < WB / 16; q += 32) {
          const int tile = q / (HSB / 16), o = q % (HSB / 16);
          cp16(dst + tile * HSB + o * 16, wsrc + ((size_t)tile * NHS + s) * HSB + o * 16);
        }
#endif
        if (g && i == 0 && !(depok >> e & 1)) {
#ifdef PROF
          long long c1 = clock64();
#endif
#if defined(DEPPF) && TMA
          {  // dependency not met yet: pull this unit's remaining weight stages into L2 meanwhile
            const int tgt = NRB1 * sm->np[e] * NCW;
            int ok = 0;
            if (lane == 0) ok = ldacq(p.ctr + e) >= tgt;
            if (!__shfl_sync(0xffffffffu, ok, 0)) {
              if (lane < TPR) for (int k = i + 1; k < nst; k++)
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" :: "l"(wsrc + ((size_t)lane * NHS + kp * nst + k) * HSB), "n"(HSB) : "memory");
              if (lane == 0) while (ldacq(p.ctr + e) < tgt) __nanosleep(64);
            }
          }
#else
          if (lane == 0) { const int tgt = NRB1 * sm->np[e] * NCW; while (ldacq(p.ctr + e) < tgt) __nanosleep(64); }
#endif
          __syncwarp();
          depok |= 1u << e;
#ifdef PROF
          wdep += clock64() - c1;
#endif
        }
#pragma unroll
        for (int k = 0; k < TP / 32; k++)
        for (int rw = k * 32; rw < min(ntok, k * 32 + 32); rw++) {
          const int row = __shfl_sync(0xffffffffu, my[k], rw & 31);
          cp16(dst + WB + (lane >> 3) * XA + rw * 128 + (((lane & 7) ^ (rw & 7)) << 4), xsrc + (size_t)row * (HD * 2) + s * 512 + lane * 16);
        }
        cpa_arrive(fb);
      }
      u = nx;
      if (lane == 0 && nx < ntot) nx = atomicAdd(p.ctr + C_Q, 1);
      nx = __shfl_sync(0xffffffffu, nx, 0);
    }
    const int slot = seq % NSLOT;
    if (seq >= NSLOT) mbwait(emptyb + slot * 8, ((seq / NSLOT) - 1) & 1);
    if (lane == 0) sm->hdr[slot].g = -1;
#if TMA
    if (lane == 0) mbarrive(fullb + slot * 8);
#endif
    cpa_arrive(fullb + slot * 8);
#ifdef PROF
    PSET(2, gtime()); PSET(3, wemp); PSET(4, wdep);
#endif
  } else {  // consumers
    float acc[RPW][NTM][4];
#ifdef PROF
    long long wful = 0; int nun = 0;
#endif
    int pend = 0, pv = 0;
    Hdr ph;
    for (int q = 0;; q++) {
      const int slot = q % NSLOT;
#ifdef PROF
      long long c0 = clock64();
#endif
      if (pv && !mbtest(fullb + slot * 8, (q / NSLOT) & 1)) {  // never sit on a finished K-split unit (its W2 may be what we wait for)
        pv = 0; if (__shfl_sync(0xffffffffu, pend, 0) == ks * NCW - 1) finish(p, sm, list, ph, lane);
      }
      mbwait(fullb + slot * 8, (q / NSLOT) & 1);
#ifdef PROF
      wful += clock64() - c0;
#endif
      const Hdr h = sm->hdr[slot];
      if (h.g < 0) {
        if (pv && __shfl_sync(0xffffffffu, pend, 0) == ks * NCW - 1) finish(p, sm, list, ph, lane);
        break;
      }
      if (h.flags & 1) {
#if !WGM
#pragma unroll
        for (int r = 0; r < RPW; r++)
#pragma unroll
          for (int n = 0; n < NTM; n++) acc[r][n][0] = acc[r][n][1] = acc[r][n][2] = acc[r][n][3] = 0.f;
#endif
        if ((h.flags & 4) && wid == 0) for (int i = lane; i < h.ntok; i += 32) p.ss[list[sm->off[h.e] + h.ps * TP + i]] = 0.f;
      }
      const uint8_t* sl = smem + slot * SLOTB;
#ifdef ONLYNT
      const int nt = ONLYNT;
#else
      const int nt = (h.ntok + 7) >> 3;
#endif
#ifndef NOCOMP
#if WGM
      {
        const uint32_t xa = sbase + slot * SLOTB + WB;
        const int sd = !(h.flags & 1);
        if (nt == 1) stage_wg<8>(acc, sl, xa, wid, lane, sd);
        else if (nt == 2) stage_wg<16>(acc, sl, xa, wid, lane, sd);
        else if (nt <= 4 || TP <= 32) stage_wg<32>(acc, sl, xa, wid, lane, sd);
#if TP > 32
        else stage_wg<64>(acc, sl, xa, wid, lane, sd);
#endif
      }
#else
      if (nt == 1) stage<1>(acc, sl, wid, lane);
      else if (nt == 2) stage<2>(acc, sl, wid, lane);
      else if (nt == 3) stage<3>(acc, sl, wid, lane);
      else if (nt == 4 || TP <= 32) stage<4>(acc, sl, wid, lane);
#if TP > 32
      else if (nt == 5) stage<5>(acc, sl, wid, lane);
      else if (nt == 6) stage<6>(acc, sl, wid, lane);
      else if (nt == 7) stage<7>(acc, sl, wid, lane);
      else stage<8>(acc, sl, wid, lane);
#endif
#endif
#endif
      __syncwarp();
      if (lane == 0) mbarrive(emptyb + slot * 8);
      if (pv) { pv = 0; if (__shfl_sync(0xffffffffu, pend, 0) == ks * NCW - 1) finish(p, sm, list, ph, lane); }
      if (h.flags & 2) {
        if (ks > 1) { pv = 1; ph = h; }
#ifdef PROF
        nun++;
#endif
        if (nt == 1) epilogue<1>(p, sm, list, acc, h, wid, lane, pend);
        else if (nt == 2) epilogue<2>(p, sm, list, acc, h, wid, lane, pend);
        else if (nt == 3) epilogue<3>(p, sm, list, acc, h, wid, lane, pend);
        else if (nt == 4 || TP <= 32) epilogue<4>(p, sm, list, acc, h, wid, lane, pend);
#if TP > 32
        else if (nt == 5) epilogue<5>(p, sm, list, acc, h, wid, lane, pend);
        else if (nt == 6) epilogue<6>(p, sm, list, acc, h, wid, lane, pend);
        else if (nt == 7) epilogue<7>(p, sm, list, acc, h, wid, lane, pend);
        else epilogue<8>(p, sm, list, acc, h, wid, lane, pend);
#endif
      }
    }
#ifdef PROF
    PSET(2, gtime()); PSET(3, wful); PSET(5, nun);
#endif
  }
  const int nw2 = sm->nw2;
  if (lane == 0) while (ldacq(p.ctr + C_W2) < nw2) __nanosleep(32);
  __syncwarp();
#ifdef PROF
  PSET(6, gtime());
#endif
  for (int it = blockIdx.x * (NTH / 32) + wid; it < p.M * 16; it += gridDim.x * (NTH / 32)) {
    const int t = it >> 4, col = (it & 15) * 256 + lane * 8;
    float a[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int k = 0; k < 8; k++) {
      const int r = t * 8 + k;
      unsigned e = p.ids[r] - p.lo;
      if (e < NLOC) {
        float cf = p.tw[r] * rsqrtf(__ldcg(p.ss + r) * (1.f / HD) + 1e-8f);
        uint4 v = __ldcg((const uint4*)(p.y + (size_t)r * HD + col));
        const __nv_bfloat162* hh = (const __nv_bfloat162*)&v;
#pragma unroll
        for (int j = 0; j < 4; j++) { float2 f = __bfloat1622float2(hh[j]); a[2 * j] += cf * f.x; a[2 * j + 1] += cf * f.y; }
      }
    }
    uint4 o; __nv_bfloat162* ob = (__nv_bfloat162*)&o;
#pragma unroll
    for (int j = 0; j < 4; j++) ob[j] = __floats2bfloat162_rn(a[2 * j], a[2 * j + 1]);
    *(uint4*)(p.out + (size_t)t * HD + col) = o;
  }
#ifdef PROF
  PSET(7, gtime());
#endif
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    if (atomicAdd(p.ctr + C_DONE, 1) == (int)gridDim.x - 1) {
      for (int i = 0; i <= C_DONE; i++) p.ctr[i] = 0;
      __threadfence();
    }
  }
}

static int g_grid = 0;
static size_t smem_bytes(int M) { return (size_t)NSLOT * SLOTB + NSLOT * 16 + sizeof(Sm) + (size_t)M * 8 * 2; }

void moe(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor w1, torch::Tensor w2, torch::Tensor g1,
         torch::Tensor g2, torch::Tensor nw, torch::Tensor inter, torch::Tensor y, torch::Tensor ss, torch::Tensor ctr,
         torch::Tensor ws, torch::Tensor out, int64_t lo, int64_t ks, torch::Tensor prof) {
  int M = x.size(0);
  if (!g_grid) {
    int dev; cudaGetDevice(&dev); int mx; cudaDeviceGetAttribute(&mx, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaDeviceGetAttribute(&g_grid, cudaDevAttrMultiProcessorCount, dev);
    cudaFuncSetAttribute(moe_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, mx);
    g_grid *= CPB;
  }
  if (CPB > 1) { int nb = 0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, moe_kernel, NTH, smem_bytes(M)); TORCH_CHECK(nb >= CPB, "CPB not resident"); }
  TORCH_CHECK(M * 8 <= 4096 && M * 8 * 3 <= NSLOT * SLOTB, "M too large");
  P p{w1.data_ptr<uint8_t>(), w2.data_ptr<uint8_t>(), g1.data_ptr<float>(), g2.data_ptr<float>(),
      (const __nv_bfloat16*)x.data_ptr(), ids.data_ptr<int>(), tw.data_ptr<float>(), nw.data_ptr<float>(),
      (__nv_bfloat16*)inter.data_ptr(), (__nv_bfloat16*)y.data_ptr(), ss.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr(),
      ctr.data_ptr<int>(), ws.data_ptr<float>(), M, (int)lo, (int)ks,
      prof.numel() ? (unsigned long long*)prof.data_ptr() : nullptr};
  moe_kernel<<<g_grid, NTH, smem_bytes(M), at::cuda::getCurrentCUDAStream()>>>(p);
}
"""

_mods = {}
_DBG = []
_ROT = ((0, 0), (3, 1), (6, 6), (-5, 5))
KS_MAX_M = 64  # K-split workspace covers M <= 64


def _big():
    pr = torch.cuda.get_device_properties(0)
    sm = getattr(pr, "shared_memory_per_block_optin", None)
    return sm >= 180000 if sm else (pr.major, pr.minor) in ((9, 0), (10, 0))


def _cfg():
    return (2, 4) if _big() else (1, 2)  # (16-row tiles per consumer warp, ring slots)


def _gencode():
    return ["-gencode=arch=compute_90a,code=sm_90a"] if torch.cuda.get_device_capability() == (9, 0) else []


def _ext():
    rpw, ns = _cfg()
    key = (rpw, ns, tuple(_DBG))
    if key not in _mods:
        fl = [d for d in ("-DRPW=%d" % rpw, "-DNSLOT=%d" % ns) if not any(x.startswith(d.split("=")[0] + "=") for x in _DBG)] + list(_DBG)
        name = "moe_w4a16_v25_" + "".join(fl).replace("-", "_").replace("=", "")
        _mods[key] = load_inline(name, cpp_sources="void moe(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,torch::Tensor);",
                                 cuda_sources=CUDA, functions=["moe"], extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"] + fl + _gencode(), verbose=False)
    return _mods[key]


def _place(word, code, pair, hi):
    de, ds = _ROT[pair]
    for bit, tgt, d in ((0, 6, de), (1, 7, de), (2, 8, de), (3, 15, ds)):
        word |= ((code >> bit) & 1) << ((tgt + 16 * hi - d) % 32)
    return word


_IDX = None


def _repack(w, sf, perm):
    """w uint8 [R, K/2], sf e4m3 [R, K/16] -> uint8 [R/16, K/256, 2304]: per 16-row tile and 256-K half stage, 4 x 512 B of
    mma A fragments (lane-major uint4; k16 step j of 64-K chunk c = K [64c + 16j, +16)) then block scales [16 rows][16 blocks]."""
    global _IDX
    R, K = w.shape[0], w.shape[1] * 2
    MT, NS = R // 16, K // 512
    if _IDX is None or _IDX.device != w.device:
        _IDX = torch.tensor([j * 16 + 8 * (e >> 1) + 2 * t + (e & 1) for t in range(4) for j in range(4) for e in range(4)], device=w.device)
    codes = torch.stack((w & 15, w >> 4), -1).view(R, K)[perm].to(torch.int64)
    s8 = sf.view(torch.uint8)[perm].to(torch.int64)
    neg = (s8 >> 7).repeat_interleave(16, 1)
    codes = codes ^ (neg << 3)
    zero = ((s8 & 0x7F) == 0).repeat_interleave(16, 1)
    codes = torch.where(zero, torch.zeros_like(codes), codes)
    s8 = torch.where((s8 & 0x7F) == 0, torch.full_like(s8, 0x08), s8 & 0x7F)
    if ((s8 >> 3) == 0).any():
        raise ValueError("subnormal E4M3 block scales are not supported by this fast path")
    codes = codes.view(R, K // 64, 64)[:, :, _IDX].reshape(R, K)
    C = codes.view(MT, 16, NS, 8, 4, 4, 4)
    a, b = C[:, :8], C[:, 8:]
    word = torch.zeros(a.shape[:-1], dtype=torch.int64, device=w.device)
    for pair, (src, e0) in enumerate(((a, 0), (b, 0), (a, 2), (b, 2))):
        word = _place(word, src[..., e0], pair, 0)
        word = _place(word, src[..., e0 + 1], pair, 1)
    word = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32)
    word = word.permute(0, 2, 3, 1, 4, 5).contiguous().view(torch.uint8).view(MT, NS, 2, 2048)
    sc = s8.to(torch.uint8).view(MT, 16, NS * 2, 16).permute(0, 2, 1, 3).contiguous().view(MT, NS, 2, 256)
    return torch.cat((word, sc), -1).contiguous().view(MT, NS * 2, 2304)


def prepare(weights):
    E, R1 = weights["w13"].shape[0], weights["w13"].shape[1]
    I = R1 // 2
    dev = weights["w13"].device
    perm1 = torch.stack((torch.arange(I, device=dev).view(-1, 8), I + torch.arange(I, device=dev).view(-1, 8)), 1).reshape(-1)
    perm2 = torch.arange(weights["w2"].shape[1], device=dev)
    w1 = torch.empty((E, R1 // 16, 16, 2304), dtype=torch.uint8, device=dev)
    w2 = torch.empty((E, weights["w2"].shape[1] // 16, 16, 2304), dtype=torch.uint8, device=dev)
    for e in range(E):
        w1[e] = _repack(weights["w13"][e], weights["w13_sf"][e], perm1)
        w2[e] = _repack(weights["w2"][e], weights["w2_sf"][e], perm2)
    _ext()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return {"w1": w1, "w2": w2, "g1": weights["w13_gs"].float() * 256.0, "g2": weights["w2_gs"].float() * 256.0,
            "nw": 1.0 + weights["post_norm_w"].float(), "ctr": torch.zeros(C_CTR, dtype=torch.int32, device=dev),
            "ws": torch.zeros((KS_MAX_M * 8 * 12288,), dtype=torch.float32, device=dev),
            "prof": torch.zeros((0,), dtype=torch.int64, device=dev)}


C_CTR = 64 + 2 * 32 * 64 * 8
KS = {1: 8, 4: 8, 8: 4, 16: 2, 32: 2}


def ksplit(M):
    if M > KS_MAX_M: return 1
    for m in sorted(KS):
        if M <= m: return KS[m]
    return 1


def kernel_fn(data, state):
    x = data["hidden_states"].contiguous()
    M = x.shape[0]
    ids = data["topk_ids"].to(torch.int32).contiguous()
    tw = data["topk_weights"].float().contiguous()
    inter = torch.empty((M * 8, 4096), dtype=torch.bfloat16, device=x.device)
    y = torch.empty((M * 8, 4096), dtype=torch.bfloat16, device=x.device)
    ss = torch.empty((M * 8,), dtype=torch.float32, device=x.device)
    out = torch.empty_like(x)
    _ext().moe(x, ids, tw, state["w1"], state["w2"], state["g1"], state["g2"], state["nw"], inter, y, ss, state["ctr"],
               state["ws"], out, data["config"]["local_start"], ksplit(M), state["prof"])
    return out
