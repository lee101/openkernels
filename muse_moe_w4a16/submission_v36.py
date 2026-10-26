# okc: problem=muse_moe_w4a16
"""v36: v35 + MT2 (two W13 tiles per warp share activation loads) + wait/class diagnostics. v35: v34 + last-arriver combine option (LASTA: no grid barrier). v34: v30 + fused split-K path (fs) for M <= 32: each CTA owns ~1 32-column inter granule per active expert,
streams its W13 rows and the matching W2 column slice back to back (no W13->W2 grid dependency, no tile queue), fp32 red into
a self-cleaning accumulator, one grid barrier, per-token norm + combine. v30: per-M choice (autotuned in prepare on the judge GPU) between v28 and the ring path below with K split 1-8 and
lookahead-1 queue fetch. v29: M <= MBIG runs v28 (v17 per-warp rings, rotation-free dequant) on the v22 weight layout; M > MBIG runs CUDA_L (v22
CTA-cooperative ring, activations shared through smem, all W13 tiles queued before W2). One persistent kernel: routing -> W13 (SwiGLU fused) -> W2 (+post-norm stats) -> norm + weighted sum.
Weights repacked per 16-row tile into mma.sync A-fragment order with a bit layout that dequantizes
NVFP4 -> BF16 exactly in 11 integer ops per 8 values (v28: no rotations, so ptxas moves 4 of 5 shifts to IMAD on the FMA pipe); streamed by TMA bulk copies (cp.async fallback)."""
from typing import TypedDict
import torch
from torch.utils.cpp_extension import load_inline
import time
_T0 = 0.0


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
#define STG 4608
#define SPT 8
#define T1E 512
#define T2E 256
#ifndef NTM
#define NTM 4
#endif
#ifndef SCR
#define SCR 1024
#endif
#define KSM 4
#ifndef KSF
#define KSF 0
#endif
#ifndef EARLYREL
#define EARLYREL 1
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define TMA 1
#else
#define TMA 0
#endif

struct P {
  const uint8_t* w1; const uint8_t* w2; const float* g1; const float* g2;
  const __nv_bfloat16* x; const int* ids; const float* tw; const float* nw;
  __nv_bfloat16* inter; __nv_bfloat16* y; float* ss; __nv_bfloat16* out; int* ctr; int* tcnt; float4* ws;
  int M, lo; unsigned long long* prof;
};
#ifdef PROF
__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
#define PSET(k, v) do { if (lane == 0) p.prof[gw * 16 + (k)] = (v); } while (0)
#define PADD(k, v) do { if (lane == 0) p.prof[gw * 16 + (k)] += (v); } while (0)
#else
#define PSET(k, v)
#define PADD(k, v)
#endif

__device__ __forceinline__ void cp16(uint32_t s, const void* g) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(g)); }
__device__ __forceinline__ void cpc() { asm volatile("cp.async.commit_group;\n"); }
template <int N> __device__ __forceinline__ void cpw() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }
__device__ __forceinline__ uint4 lds(uint32_t a) { uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a)); return v; }
__device__ __forceinline__ uint2 lds64(uint32_t a) { uint2 v; asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];\n" : "=r"(v.x), "=r"(v.y) : "r"(a)); return v; }
__device__ __forceinline__ uint32_t bmul(uint32_t a, uint32_t b) { uint32_t d; asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(0x80008000u)); return d; }
__device__ __forceinline__ void mma(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ int ldacq(const int* p) { int v; asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void fence_ar() { asm volatile("fence.acq_rel.gpu;\n" ::: "memory"); }
__device__ __forceinline__ void gbar(int id, int n) { asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ uint32_t rotl(uint32_t q, int d) { return __funnelshift_l(q, q, d); }
__device__ __forceinline__ uint32_t sc2(uint32_t w, uint32_t sel) { return __byte_perm(w, 0, sel) * 16u + 0x77007700u; }
#if TMA
__device__ __forceinline__ void mb_init(uint32_t b) { asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(b)); }
__device__ __forceinline__ void tma_ld(uint32_t dst, const void* src, uint32_t bar) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(bar), "n"(STG) : "memory");
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" :: "r"(dst), "l"(src), "n"(STG), "r"(bar) : "memory");
}
__device__ __forceinline__ void mb_wait(uint32_t bar, uint32_t ph) {
  asm volatile("{\n.reg .pred p;\nW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W;\n}\n" :: "r"(bar), "r"(ph) : "memory");
}
#endif

#ifndef HEAVY1
#define HEAVY1 1
#endif
#define QI (NLOC + 2)  // ctr[QI]: dynamic tile queue
#define RED (2 * 2 * 32 * 16)  // per-warp split-K partial slots: 2 tile parities x NT <= 2 float4 per lane
struct Sm { int cnt[NLOC], off[NLOC], fill[NLOC], act[NLOC], na, ks, maxc; int ids[16][4]; };

// Work = tiles from one global queue (all W13 tiles, then all W2 tiles, expert-major). A group of ks warps of one CTA
// takes a tile together, each streaming SPT/ks K-stages; partials meet in smem; the finisher rotates per tile.
template <int S>
struct Ctx {
  const P* p; Sm* sm; const uint16_t* list; uint32_t ring, bars; int lane, N1, N, gi, gj, ks;
  int q; unsigned long long wcyc;
  int pi, pp, ps;  // producer: group tile index, pass, stage
  __device__ __forceinline__ int tid_of(int i) const { return ((volatile int*)sm->ids[gi])[i & 3]; }
  __device__ __forceinline__ void item(int u, int& g2, int& e, int& mt) const {
    if (u < N1) { g2 = 0; e = sm->act[u >> 9]; mt = u & 511; }
    else { int v = u - N1; g2 = 1; e = sm->act[v >> 8]; mt = v & 255; }
  }
  __device__ __forceinline__ void issue(int slot) {
    const int u = tid_of(pi);
    if (u < N) {
      int g2, e, mt; item(u, g2, e, mt);
      const int span = SPT / ks, s = gj * span + ps;
      const uint8_t* src = g2 ? p->w2 + ((size_t)((p->lo + e) * T2E + mt) * SPT + s) * STG
                              : p->w1 + ((size_t)((p->lo + e) * T1E + mt) * SPT + s) * STG;
#if TMA
      if (lane == 0) tma_ld(ring + slot * STG, src, bars + slot * 8);
#else
      uint32_t d = ring + slot * STG + lane * 16; src += lane * 16;
#pragma unroll
      for (int k = 0; k < 9; k++) cp16(d + k * 512, src + k * 512);
#endif
      if (++ps == span) { ps = 0; if (++pp == (sm->cnt[e] + 8 * NTM - 1) / (8 * NTM)) { pp = 0; pi++; } }
    }
#if !TMA
    cpc();
#endif
  }
  __device__ __forceinline__ uint32_t acquire() {  // wait for stage q, refill the slot freed last iteration
    const int slot = q % S;
#ifdef PROF
    long long c0 = clock64();
#endif
#if TMA
    mb_wait(bars + slot * 8, (q / S) & 1);
    __syncwarp();
#else
    cpw<S - 2>();
#endif
#ifdef PROF
    wcyc += clock64() - c0;
#endif
#if !EARLYREL
    issue((q + S - 1) % S);
#endif
    q++;
    return ring + slot * STG + lane * 16;
  }
  __device__ __forceinline__ void release() {  // stage q - 1 consumed: refill its slot now, before any epilogue
#if EARLYREL
    __syncwarp();
    issue((q - 1) % S);
#endif
  }
};

template <int NT, int G2, int S>
__device__ __forceinline__ bool run(Ctx<S>& C, int e, int mt, int pass, int ti, int s0, int s1) {
  const P& p = *C.p;
  const int lane = C.lane, g = lane >> 2, t = lane & 3;
  const int cnt = C.sm->cnt[e], off = C.sm->off[e];
  constexpr int CH = NT == 1 ? 4 : (NT == 2 ? 2 : 1);
#ifndef BD1
#define BD1 7
#endif
  constexpr int D = NT == 1 ? BD1 : (NT == 2 ? 3 : 1);  // B prefetch distance in chunks; (D + 1) divides 8 so slots stay static
  const uint4* rp[NT];
#pragma unroll
  for (int n = 0; n < NT; n++) {
    int i = pass * 8 * NTM + n * 8 + g; int r = C.list[off + (i < cnt ? i : 0)];
    rp[n] = (const uint4*)((G2 ? p.inter + (size_t)r * HD : p.x + (size_t)(r >> 3) * HD) + t * 16);
  }
  uint4 B[D + 1][NT][2];
  auto ldb = [&](uint4 (&b)[NT][2], int s, int c) {
#pragma unroll
    for (int n = 0; n < NT; n++) {
      const uint4* q = rp[n] + s * 64 + c * 8;
      if (G2) { b[n][0] = __ldca(q); b[n][1] = __ldca(q + 1); } else { b[n][0] = __ldg(q); b[n][1] = __ldg(q + 1); }
    }
  };
  float acc[NT][CH][4];
#pragma unroll
  for (int n = 0; n < NT; n++)
#pragma unroll
    for (int h = 0; h < CH; h++) acc[n][h][0] = acc[n][h][1] = acc[n][h][2] = acc[n][h][3] = 0.f;
#pragma unroll
  for (int c = 0; c < D; c++) ldb(B[c], s0, c);
  for (int s = s0; s < s1; s++) {
    const uint32_t slot = C.acquire();
    const uint2 sh0 = lds64(slot - lane * 8 + 2048), sh1 = lds64(slot - lane * 8 + 4352);  // slot = stage + lane * 16
    const uint4 sc = make_uint4(sh0.x, sh0.y, sh1.x, sh1.y);
#pragma unroll
    for (int c = 0; c < 8; c++) {
      const uint4 w = lds(slot + c * 512 + (c >= 4 ? 256 : 0));
      const uint32_t sw = (c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w));
      const uint32_t sA = sc2(sw, (c & 1) ? 0x4242 : 0x4040), sB = sc2(sw, (c & 1) ? 0x4343 : 0x4141);
      if (c + D < 8) ldb(B[(c + D) % (D + 1)], s, c + D);
      else if (s + 1 < s1) ldb(B[(c + D) % (D + 1)], s + 1, c + D - 8);
      const uint4 (&bc)[NT][2] = B[c % (D + 1)];
      const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const uint32_t q = ww[j];
#ifdef FAKEDQ
        const uint32_t a0 = q ^ sA, a1 = q ^ sB, a2 = q + sA, a3 = q + sB;
#else
        const uint32_t a0 = bmul(q & 0x81C081C0u, sA);
        const uint32_t a1 = bmul(((q << 3) & 0x01C001C0u) | ((q << 1) & 0x80008000u), sB);
        const uint32_t a2 = bmul((q << 6) & 0x81C081C0u, sA);
        const uint32_t a3 = bmul(((q >> 5) & 0x01C001C0u) | ((q << 5) & 0x80008000u), sB);
#endif
#pragma unroll
        for (int n = 0; n < NT; n++) {
          const uint32_t* bb = (const uint32_t*)&bc[n][j >> 1];
#ifdef NOMMA
          acc[n][j % CH][0] += __uint_as_float(a0 ^ a1 ^ a2 ^ a3 ^ bb[(j & 1) * 2] ^ bb[(j & 1) * 2 + 1]);
#else
          mma(acc[n][j % CH], a0, a1, a2, a3, bb[(j & 1) * 2], bb[(j & 1) * 2 + 1]);
#endif
        }
      }
    }
    C.release();
  }
#ifdef PROF
  const long long ce0 = clock64();
  struct EpiT { long long c; int lane; unsigned long long* pf; int gw; __device__ ~EpiT() { if (lane == 0 && pf) pf[gw * 16 + 15] += clock64() - c; } } epit{ce0, lane, p.prof, (int)(blockIdx.x * WPC + (threadIdx.x >> 5))};
#endif
  extern __shared__ __align__(16) uint8_t smem[];
  float f[NT][4];
#pragma unroll
  for (int n = 0; n < NT; n++)
#pragma unroll
    for (int i = 0; i < 4; i++) { float v = 0.f;
#pragma unroll
      for (int h = 0; h < CH; h++) v += acc[n][h][i];
      f[n][i] = v; }
  if (NT <= 2 && C.ks > 1) {  // group split-K: partials meet in smem (double-buffered by tile parity)
    const int wid = threadIdx.x >> 5, ks = C.ks, w0 = wid - C.gj;
    float4* red = (float4*)(smem + (size_t)WPC * S * STG + WPC * SCR + WPC * S * 8) + (ti & 1) * WPC * 64;
#pragma unroll
    for (int n = 0; n < NT; n++) red[(wid * 2 + n) * 32 + lane] = make_float4(f[n][0], f[n][1], f[n][2], f[n][3]);
    gbar(1 + C.gi, ks * 32);
    if (C.gj != (ti & (ks - 1))) return false;
    for (int o = w0; o < w0 + ks; o++) {
      if (o == wid) continue;
#pragma unroll
      for (int n = 0; n < NT; n++) { float4 v = red[(o * 2 + n) * 32 + lane]; f[n][0] += v.x; f[n][1] += v.y; f[n][2] += v.z; f[n][3] += v.w; }
    }
  }
  __nv_bfloat16* scr = (__nv_bfloat16*)(smem + (size_t)(blockDim.x >> 5) * S * STG + (threadIdx.x >> 5) * SCR);
  if (!G2) {
    const float gs = p.g1[p.lo + e];
#pragma unroll
    for (int n = 0; n < NT; n++)
#pragma unroll
      for (int h = 0; h < 2; h++) {
        float gt = f[n][h] * gs, up = f[n][2 + h] * gs;
        scr[(n * 8 + 2 * t + h) * 8 + g] = __float2bfloat16(gt / (1.f + __expf(-gt)) * up);
      }
    __syncwarp();
    for (int i = lane; i < NT * 8; i += 32) {
      int ii = pass * 8 * NTM + i;
      if (ii < cnt) { int r = C.list[off + ii]; *(uint4*)(p.inter + (size_t)r * HD + mt * 8) = *(uint4*)(scr + i * 8); }
    }
  } else {
    const float gs = p.g2[p.lo + e];
    const float n0 = p.nw[mt * 16 + g] * gs, n1 = p.nw[mt * 16 + g + 8] * gs;
#pragma unroll
    for (int n = 0; n < NT; n++) {
      float v0 = f[n][0] * n0, v1 = f[n][1] * n0, v2 = f[n][2] * n1, v3 = f[n][3] * n1;
      scr[(n * 8 + 2 * t) * 16 + g] = __float2bfloat16(v0);
      scr[(n * 8 + 2 * t + 1) * 16 + g] = __float2bfloat16(v1);
      scr[(n * 8 + 2 * t) * 16 + g + 8] = __float2bfloat16(v2);
      scr[(n * 8 + 2 * t + 1) * 16 + g + 8] = __float2bfloat16(v3);
      float sa = v0 * v0 + v2 * v2, sb = v1 * v1 + v3 * v3;
#pragma unroll
      for (int o = 4; o < 32; o <<= 1) { sa += __shfl_xor_sync(0xffffffffu, sa, o); sb += __shfl_xor_sync(0xffffffffu, sb, o); }
      if (g == 0) {
        int i0 = pass * 8 * NTM + n * 8 + 2 * t;
        if (i0 < cnt) atomicAdd(p.ss + C.list[off + i0], sa);
        if (i0 + 1 < cnt) atomicAdd(p.ss + C.list[off + i0 + 1], sb);
      }
    }
    __syncwarp();
    for (int i = lane; i < NT * 16; i += 32) {
      int ii = pass * 8 * NTM + (i >> 1);
      if (ii < cnt) { int r = C.list[off + ii]; *(uint4*)(p.y + (size_t)r * HD + mt * 16 + (i & 1) * 8) = *(uint4*)(scr + i * 8); }
    }
  }
  __syncwarp();
  return true;
}

template <int G2, int S>
__device__ __forceinline__ bool dispatch(Ctx<S>& C, int nt, int e, int mt, int pass, int ti, int s0, int s1) {
  if (nt <= 1) return run<1, G2, S>(C, e, mt, pass, ti, s0, s1);
  if (nt <= 2) return run<2, G2, S>(C, e, mt, pass, ti, s0, s1);
#if NTM >= 8
  if (nt > 4) return run<8, G2, S>(C, e, mt, pass, ti, s0, s1);
#endif
  return run<4, G2, S>(C, e, mt, pass, ti, s0, s1);
}

__global__ void __launch_bounds__(WPC * 32, 1) moe_kernel(P p) {
  constexpr int S = NSTG;
  extern __shared__ __align__(16) uint8_t smem[];
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5, R = p.M * 8;
  const bool small = p.M <= 64;  // split-K groups (and their smem) only for small batches
  Sm* sm = (Sm*)(smem + WPC * S * STG + WPC * SCR + WPC * S * 8 + (small ? WPC * RED : 0));
  uint16_t* list = (uint16_t*)(sm + 1);
  uint8_t* lid = (uint8_t*)(list + R);
  const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem);
  const int gw = blockIdx.x * WPC + wid;
#ifdef PROF
  if (lane == 0) { for (int k = 0; k < 16; k++) p.prof[gw * 16 + k] = 0; unsigned sid; asm("mov.u32 %0, %smid;" : "=r"(sid)); p.prof[gw * 16 + 10] = sid; }
  PSET(0, gtime()); PSET(13, clock64());
#endif
#if TMA
  if (lane < S) mb_init(sbase + WPC * S * STG + WPC * SCR + (wid * S + lane) * 8);
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#endif
  if (tid < NLOC) sm->cnt[tid] = 0;
  __syncthreads();
  for (int i = tid; i < R; i += blockDim.x) { unsigned e = p.ids[i] - p.lo; lid[i] = e < NLOC ? e : 0xff; if (e < NLOC) atomicAdd(&sm->cnt[e], 1); }
  __syncthreads();
  if (wid == 0) {
    int c = sm->cnt[lane], v = c;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) { int n = __shfl_up_sync(0xffffffffu, v, o); if (lane >= o) v += n; }
    sm->off[lane] = v - c; sm->fill[lane] = v - c;
    unsigned m = __ballot_sync(0xffffffffu, c > 0);
#if HEAVY1
    {  // active experts heaviest first: their multi-pass tiles start early instead of forming the tail
      int rk = 0;
#pragma unroll
      for (int j = 0; j < 32; j++) { const int cj = __shfl_sync(0xffffffffu, c, j); rk += (cj > c) || (cj == c && j < lane); }
      if (c > 0) sm->act[rk] = lane;
    }
#else
    if (c > 0) sm->act[__popc(m & ((1u << lane) - 1))] = lane;
#endif
    int mx = c;
#pragma unroll
    for (int o = 16; o; o >>= 1) mx = max(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    if (lane == 0) { sm->na = __popc(m); sm->maxc = mx; const int tw = __popc(m) * (T1E + T2E) / (int)(gridDim.x * WPC / 16);  // tiles per 16 warps
      sm->ks = (small && mx <= 16) ? (KSF ? KSF : (tw < 32 ? 4 : (tw < 96 ? 2 : 1))) : 1; }
  }
  __syncthreads();
  for (int i = tid; i < R; i += blockDim.x) { unsigned e = lid[i]; if (e < NLOC) list[atomicAdd(&sm->fill[e], 1)] = i; }
  const int ks = sm->ks, na = sm->na;
  Ctx<S> C;
  C.p = &p; C.sm = sm; C.list = list; C.lane = lane; C.ks = ks; C.gi = wid / ks; C.gj = wid % ks;
  C.N1 = na * T1E; C.N = C.N1 + na * T2E;
  C.ring = sbase + wid * S * STG; C.bars = sbase + WPC * S * STG + WPC * SCR + wid * S * 8; C.q = 0; C.wcyc = 0;
  C.pi = 0; C.pp = 0; C.ps = 0;
  const bool lead = C.gj == 0 && lane == 0;
  if (lead) { sm->ids[C.gi][0] = atomicAdd(p.ctr + QI, 1); sm->ids[C.gi][1] = atomicAdd(p.ctr + QI, 1); }
  __syncthreads();
#ifdef PROF
  PSET(1, gtime());
#endif
#pragma unroll
  for (int i = 0; i < S - 1 + EARLYREL; i++) C.issue(i);
  for (int i = 0;; i++) {
    const int u = C.tid_of(i);
    if (u >= C.N) break;
    int nxt = 0;
    if (lead) nxt = atomicAdd(p.ctr + QI, 1);  // tile for i + 2; latency hidden behind this tile
    int g2, e, mt; C.item(u, g2, e, mt);
    const int cnt = sm->cnt[e], np = (cnt + 8 * NTM - 1) / (8 * NTM);
#ifdef PROF
    long long c0 = clock64();
#endif
    if (g2) {
      if (lane == 0) while (ldacq(p.ctr + e) < T1E) __nanosleep(32);
      __syncwarp();
#ifdef PROF
      PADD(4, clock64() - c0); c0 = clock64();
#endif
    } else if (mt == 0 && C.gj == 0) {
      for (int k = lane; k < cnt; k += 32) p.ss[list[sm->off[e] + k]] = 0.f;
    }
    if (lead) sm->ids[C.gi][(i + 2) & 3] = nxt;  // read 2 tiles later, after the group barrier (ks > 1) / syncwarp
    __syncwarp();
    const int span = SPT / ks, s0 = C.gj * span;
    bool fin = false;
    for (int ps = 0; ps < np; ps++) {
      int nt = min(NTM, (cnt - ps * 8 * NTM + 7) >> 3);
      fin = g2 ? dispatch<1, S>(C, nt, e, mt, ps, i, s0, s0 + span) : dispatch<0, S>(C, nt, e, mt, ps, i, s0, s0 + span);
    }
    if (fin) {  // release this tile's stores to the dependent phase (no full fence: the release add orders them)
      __syncwarp();
      if (lane == 0) asm volatile("red.release.gpu.global.add.s32 [%0], 1;\n" :: "l"(p.ctr + (g2 ? NLOC : e)) : "memory");
    }
#ifdef PROF
    PADD(g2 ? 3 : 2, clock64() - c0); PADD(8, 1);
#endif
  }
#if !TMA
  cpw<0>();
#endif
  const int T2 = sm->na * T2E;
#ifdef PROF
  PSET(5, gtime()); PSET(9, C.wcyc); PSET(11, sm->na); PSET(12, sm->ks);
#endif
  if (lane == 0) while (ldacq(p.ctr + NLOC) < T2) __nanosleep(32);
  __syncwarp();
#ifdef PROF
  PSET(6, gtime());
#endif
  for (int it = gw; it < p.M * 16; it += gridDim.x * WPC) {
    const int t = it >> 4, col = (it & 15) * 256 + lane * 8;
    float a[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int k = 0; k < 8; k++) {
      const int r = t * 8 + k;
      unsigned e = p.ids[r] - p.lo;
      if (e < NLOC) {
        float cf = p.tw[r] * rsqrtf(__ldcg(p.ss + r) * (1.f / HD) + 1e-8f);
        uint4 v = __ldcg((const uint4*)(p.y + (size_t)r * HD + col));
        const __nv_bfloat162* h = (const __nv_bfloat162*)&v;
#pragma unroll
        for (int j = 0; j < 4; j++) { float2 f = __bfloat1622float2(h[j]); a[2 * j] += cf * f.x; a[2 * j + 1] += cf * f.y; }
      }
    }
    uint4 o; __nv_bfloat162* ob = (__nv_bfloat162*)&o;
#pragma unroll
    for (int j = 0; j < 4; j++) ob[j] = __floats2bfloat162_rn(a[2 * j], a[2 * j + 1]);
    *(uint4*)(p.out + (size_t)t * HD + col) = o;
  }
#ifdef PROF
  PSET(7, gtime()); PSET(14, clock64());
#endif
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    if (atomicAdd(p.ctr + NLOC + 1, 1) == (int)gridDim.x - 1) {
      for (int i = 0; i < NLOC + 3; i++) p.ctr[i] = 0;
      __threadfence();
    }
  }
}

static int g_grid = 0;
static size_t smem_bytes(int M) { return (size_t)WPC * NSTG * STG + WPC * SCR + WPC * NSTG * 8 + (M <= 64 ? WPC * RED : 0) + sizeof(Sm) + (size_t)M * 8 * 3; }

void moe(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor w1, torch::Tensor w2, torch::Tensor g1,
         torch::Tensor g2, torch::Tensor nw, torch::Tensor inter, torch::Tensor y, torch::Tensor ss, torch::Tensor ctr,
         torch::Tensor tcnt, torch::Tensor ws, torch::Tensor out, int64_t lo, torch::Tensor prof) {
  int M = x.size(0);
  if (!g_grid) {
    int dev; cudaGetDevice(&dev); int mx; cudaDeviceGetAttribute(&mx, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaDeviceGetAttribute(&g_grid, cudaDevAttrMultiProcessorCount, dev);
    cudaFuncSetAttribute(moe_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, mx);
  }
  P p{w1.data_ptr<uint8_t>(), w2.data_ptr<uint8_t>(), g1.data_ptr<float>(), g2.data_ptr<float>(),
      (const __nv_bfloat16*)x.data_ptr(), ids.data_ptr<int>(), tw.data_ptr<float>(), nw.data_ptr<float>(),
      (__nv_bfloat16*)inter.data_ptr(), (__nv_bfloat16*)y.data_ptr(), ss.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr(),
      ctr.data_ptr<int>(), tcnt.data_ptr<int>(), (float4*)ws.data_ptr(), M, (int)lo,
      prof.numel() ? (unsigned long long*)prof.data_ptr() : nullptr};
  moe_kernel<<<g_grid, WPC * 32, smem_bytes(M), at::cuda::getCurrentCUDAStream()>>>(p);
}
"""

CUDA_L = r"""
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
#define XROW 528
#ifndef NCW
#define NCW 8
#endif
#define NTH ((NCW + 1) * 32)
#define TPR (NCW * RPW)
#define WB (TPR * HSB)
#define SLOTB (WB + TP * XROW)
#define NRB1 (8192 / (16 * TPR))
#define NRB2 (4096 / (16 * TPR))
#define C_W2 32
#define C_Q 33
#define C_DONE 34
#define C_UC 64
#ifndef LAG
#define LAG 64
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

template <int NT>
__device__ __forceinline__ void stage(float (&acc)[RPW][4][4], const uint8_t* wsm, const uint8_t* xs, int lane) {
  const int g = lane >> 2, t = lane & 3;
  uint2 sc[RPW];
#pragma unroll
  for (int r = 0; r < RPW; r++) sc[r] = *(const uint2*)(wsm + r * HSB + 2048 + lane * 8);
#pragma unroll
  for (int c = 0; c < 4; c++) {
    uint4 w[RPW];
#pragma unroll
    for (int r = 0; r < RPW; r++) w[r] = *(const uint4*)(wsm + r * HSB + c * 512 + lane * 16);
    uint4 b[NT][2];
#pragma unroll
    for (int n = 0; n < NT; n++)
#pragma unroll
      for (int h = 0; h < 2; h++) b[n][h] = *(const uint4*)(xs + (n * 8 + g) * XROW + c * 128 + t * 32 + h * 16);
    uint32_t sA[RPW], sB[RPW];
#pragma unroll
    for (int r = 0; r < RPW; r++) {
      const uint32_t sw = c < 2 ? sc[r].x : sc[r].y;
      sA[r] = sc2(sw, (c & 1) ? 0x4242 : 0x4040); sB[r] = sc2(sw, (c & 1) ? 0x4343 : 0x4141);
    }
#pragma unroll
    for (int j = 0; j < 4; j++) {
#pragma unroll
      for (int r = 0; r < RPW; r++) {
        const uint32_t q = j == 0 ? w[r].x : j == 1 ? w[r].y : j == 2 ? w[r].z : w[r].w;
#ifdef FAKEDQ
        const uint32_t a0 = q ^ sA[r], a1 = q ^ sB[r], a2 = q + sA[r], a3 = q + sB[r];
#else
        const uint32_t a0 = bmul(q & 0x81C081C0u, sA[r]);
        const uint32_t a1 = bmul(((q << 3) & 0x01C001C0u) | ((q << 1) & 0x80008000u), sB[r]);
        const uint32_t a2 = bmul((q << 6) & 0x81C081C0u, sA[r]);
        const uint32_t a3 = bmul(((q >> 5) & 0x01C001C0u) | ((q << 5) & 0x80008000u), sB[r]);
#endif
        constexpr int CH = NT == 1 ? 4 : (NT == 2 ? 2 : 1);
#pragma unroll
        for (int n = 0; n < NT; n++) {
          const uint32_t* bb = (const uint32_t*)&b[n][j >> 1];
#ifdef NOMMA
          acc[r][n * CH + j % CH][0] += __uint_as_float((a0 ^ a1 ^ a2 ^ a3 ^ bb[(j & 1) * 2] ^ bb[(j & 1) * 2 + 1]) & 0x3fffffffu);
#else
          mma(acc[r][n * CH + j % CH], a0, a1, a2, a3, bb[(j & 1) * 2], bb[(j & 1) * 2 + 1]);
#endif
        }
      }
    }
  }
}

// unit finished: acc holds this CTA's partial (ks > 1) or full sums for rows of tiles rb*TPR + wid*RPW + r, tokens of pass ps
template <int NT>
__device__ __forceinline__ void epilogue(const P& p, Sm* sm, const uint16_t* list, float (&acc)[RPW][4][4], const Hdr& h, int wid, int lane, int& pend) {
  const int g = lane >> 2, t = lane & 3, e = h.e, ntok = h.ntok;
  const uint16_t* lst = list + sm->off[e] + h.ps * TP;
  const int rows = h.g ? 4096 : 8192;
  constexpr int CH = NT == 1 ? 4 : (NT == 2 ? 2 : 1);
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
      const int mt = h.rb * TPR + wid * RPW + r;
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
      const int mt = h.rb * TPR + wid * RPW + r;
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
      const int mt = h.rb * TPR + wid * RPW + r;
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
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5, R = p.M * 8, ks = p.ks;
  Sm* sm = (Sm*)(smem + NSLOT * SLOTB + NSLOT * 16);
  uint16_t* list = (uint16_t*)(sm + 1);
  uint8_t* lid = smem;                       // prologue scratch in the ring
  uint16_t* posi = (uint16_t*)(smem + 4096);
  const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem);
  const uint32_t fullb = sbase + NSLOT * SLOTB, emptyb = fullb + NSLOT * 8;
#ifdef PROF
  PSET(0, gtime()); PSET(8, clock64());
#endif
  if (tid < NSLOT) { mbinit(fullb + tid * 8, FULLCNT); mbinit(emptyb + tid * 8, NCW); }
  if (tid < NLOC) sm->base[tid] = 0;
#if TMA
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#endif
  int u = 0, nx = 0;
#ifdef NX1
  if (tid == NCW * 32) u = atomicAdd(p.ctr + C_Q, 1);
#else
  if (tid == NCW * 32) { u = atomicAdd(p.ctr + C_Q, 1); nx = atomicAdd(p.ctr + C_Q, 1); }  // queue heads early: latency hides behind routing
#endif
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
      int myrow = -1;
      if (lane < ntok) { const int r = lst[lane]; myrow = g ? r : r >> 3; }
      for (int i = 0; i < nst; i++, seq++) {
        const int slot = seq % NSLOT, s = kp * nst + i;
#ifdef NX1
        if (i == nst - 1 && lane == 0) nx = atomicAdd(p.ctr + C_Q, 1);
#endif
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
        for (int rw = 0; rw < ntok; rw++) {
          const int row = __shfl_sync(0xffffffffu, myrow, rw);
          cp16(dst + WB + rw * XROW + lane * 16, xsrc + (size_t)row * (HD * 2) + s * 512 + lane * 16);
        }
        cpa_arrive(fb);
      }
#ifdef NX1
      u = __shfl_sync(0xffffffffu, nx, 0);
#else
      u = nx;
      if (lane == 0 && nx < ntot) nx = atomicAdd(p.ctr + C_Q, 1);
      nx = __shfl_sync(0xffffffffu, nx, 0);
#endif
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
    float acc[RPW][4][4];
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
#pragma unroll
        for (int r = 0; r < RPW; r++)
#pragma unroll
          for (int n = 0; n < 4; n++) acc[r][n][0] = acc[r][n][1] = acc[r][n][2] = acc[r][n][3] = 0.f;
        if ((h.flags & 4) && wid == 0 && lane < h.ntok) p.ss[list[sm->off[h.e] + h.ps * TP + lane]] = 0.f;
      }
      const uint8_t* sl = smem + slot * SLOTB;
      const uint8_t* wsm = sl + wid * RPW * HSB;
      const uint8_t* xs = sl + WB;
#ifdef ONLYNT
      const int nt = ONLYNT;
#else
      const int nt = (h.ntok + 7) >> 3;
#endif
#ifndef NOCOMP
      if (nt == 1) stage<1>(acc, wsm, xs, lane);
      else if (nt == 2) stage<2>(acc, wsm, xs, lane);
      else if (nt == 3) stage<3>(acc, wsm, xs, lane);
      else stage<4>(acc, wsm, xs, lane);
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
        else epilogue<4>(p, sm, list, acc, h, wid, lane, pend);
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
  PSET(7, gtime()); PSET(9, clock64());
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

_mod = None
_DBG = []
# pair p of each 32-bit word: (eem rotation, sign rotation); bf16 targets m->6, e0->7, e1->8, s->15 (+16 for the high value)
_ROT = ((0, 0), (3, 1), (6, 6), (-5, 5))


def _big():
    pr = torch.cuda.get_device_properties(0)
    sm = getattr(pr, "shared_memory_per_block_optin", None)
    return sm >= 180000 if sm else (pr.major, pr.minor) in ((9, 0), (10, 0))


_mod8 = None
LARGE = True  # M > 64: 8 warps per SM with 4-stage rings and up to 64 tokens per pass (one dequant per stage)


def _ext(large=False):
    global _mod, _mod8
    if (_mod8 if large else _mod) is None:
        big = _big()
        wpc, stg = (16, 2) if big else (8, 2)
        extra = []
        if large:
            wpc, stg, extra = 8, (4 if big else 2), ["-DNTM=8", "-DSCR=2048"]
        for d in _DBG:
            if d.startswith("-DCFG=") and not large: wpc, stg = map(int, d[6:].split("x"))
        m = load_inline("moe_w4a16_v30_%d_%d%s%s" % (wpc, stg, "_L" if large else "", "".join(_DBG).replace("-", "_").replace("=", "")),
                        cpp_sources="void moe(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,torch::Tensor);",
                        cuda_sources=CUDA, functions=["moe"],
                        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo", "-DWPC=%d" % wpc, "-DNSTG=%d" % stg] + extra + [d for d in _DBG if not d.startswith("-DCFG=")], verbose=False)
        if large: _mod8 = m
        else: _mod = m
    return _mod8 if large else _mod


def _place(word, code, pair, hi):
    de, ds = _ROT[pair]
    for bit, tgt, d in ((0, 6, de), (1, 7, de), (2, 8, de), (3, 15, ds)):
        word |= ((code >> bit) & 1) << ((tgt + 16 * hi - d) % 32)
    return word


def _repack(w, sf, perm):
    """w uint8 [R, K/2], sf e4m3 [R, K/16] -> uint8 [R/16, K/256, 2304]: per 16-row tile and 256-K half stage, 4 x 512 B of
    mma A fragments (lane-major uint4) then 32 lanes x 8 B of block scales."""
    R, K = w.shape[0], w.shape[1] * 2
    MT, NS = R // 16, K // 512
    codes = torch.stack((w & 15, w >> 4), -1).view(R, K)[perm].to(torch.int64)
    s8 = sf.view(torch.uint8)[perm].to(torch.int64)
    neg = (s8 >> 7).repeat_interleave(16, 1)
    codes = codes ^ (neg << 3)
    zero = ((s8 & 0x7F) == 0).repeat_interleave(16, 1)
    codes = torch.where(zero, torch.zeros_like(codes), codes)
    s8 = torch.where((s8 & 0x7F) == 0, torch.full_like(s8, 0x08), s8 & 0x7F)
    if ((s8 >> 3) == 0).any():
        raise ValueError("subnormal E4M3 block scales are not supported by this fast path")
    C = codes.view(MT, 16, NS, 8, 4, 4, 4)
    a, b = C[:, :8], C[:, 8:]
    word = torch.zeros(a.shape[:-1], dtype=torch.int64, device=w.device)
    for pair, (src, e0) in enumerate(((a, 0), (b, 0), (a, 2), (b, 2))):
        word = _place(word, src[..., e0], pair, 0)
        word = _place(word, src[..., e0 + 1], pair, 1)
    word = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32)
    word = word.permute(0, 2, 3, 1, 4, 5).contiguous().view(torch.uint8).view(MT, NS, 2, 4, 512)
    S = s8.to(torch.uint8).view(MT, 16, NS, 8, 4)
    sc = torch.stack((S[:, :8], S[:, 8:]), -1).permute(0, 2, 1, 4, 3, 5).contiguous().view(MT, NS, 32, 2, 8)
    sc = sc.permute(0, 1, 3, 2, 4).contiguous()
    return torch.cat((word.view(MT, NS, 2, 2048), sc.view(MT, NS, 2, 256)), -1).contiguous().view(MT, NS * 2, 2304)


CUDA_F = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>

#define HD 4096
#define NLOC 32
#define STG 4608
#define GR 128
#define RB 64
#define REC 1152
#ifdef MT2
#define MTL 2
#else
#define MTL 1
#endif
#ifndef FW1
#define FW1 30
#endif
#ifndef FW2
#define FW2 62
#endif
#ifndef FW4
#define FW4 110
#endif
#ifndef WPC
#define WPC 16
#endif
#ifndef SMAX
#define SMAX 3
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define TMA 1
#else
#define TMA 0
#endif

struct P {
  const uint8_t* w1; const uint8_t* w2; const float* g1; const float* g2;
  const __nv_bfloat16* x; const int* ids; const float* tw; const float* nw;
  float* yacc; __nv_bfloat16* out; int* ctr; int M, lo, smem; unsigned long long* prof;
};
struct Seg { int e, gr0, ng, T, NT, ioff; };
struct Sm { int cnt[NLOC], off[NLOC], fill[NLOC], act[NLOC]; int na, g, ks, S, ns, rounds, target, ioff, red, roff, ntm; Seg seg[3]; float ssr[8][WPC]; int fin[3], tq[32], ntq; float g2s[NLOC], tws[256]; };

__device__ __forceinline__ void cp16(uint32_t s, const void* g) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(g) : "memory"); }
__device__ __forceinline__ uint4 lds(uint32_t a) { uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a)); return v; }
__device__ __forceinline__ uint2 lds64(uint32_t a) { uint2 v; asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];\n" : "=r"(v.x), "=r"(v.y) : "r"(a)); return v; }
__device__ __forceinline__ uint32_t lds16(uint32_t a) { unsigned short v; asm volatile("ld.shared.u16 %0, [%1];\n" : "=h"(v) : "r"(a)); return v; }
__device__ __forceinline__ uint32_t bmul(uint32_t a, uint32_t b) { uint32_t d; asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(0x80008000u)); return d; }
__device__ __forceinline__ void mma(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ int ldacq(const int* p) { int v; asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void gbar(int id, int n) { asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ uint32_t sc2(uint32_t w, uint32_t sel) { return __byte_perm(w, 0, sel) * 16u + 0x77007700u; }
__device__ __forceinline__ void mb_init(uint32_t b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(b), "r"(n)); }
__device__ __forceinline__ void mb_wait(uint32_t bar, uint32_t ph) {
#if TMA
  asm volatile("{\n.reg .pred p;\nW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W;\n}\n" :: "r"(bar), "r"(ph) : "memory");
#else
  asm volatile("{\n.reg .pred p;\nW: mbarrier.test_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W;\n}\n" :: "r"(bar), "r"(ph) : "memory");
#endif
}
#ifdef FAKEDQ
#define BMUL(a, b) ((a) ^ (b))
#else
#define BMUL(a, b) bmul(a, b)
#endif
#ifdef NOMMA
#define MMA(c, a0, a1, a2, a3, b0, b1) (c)[0] += __uint_as_float((a0) ^ (a1) ^ (a2) ^ (a3) ^ (b0) ^ (b1))
#else
#define MMA mma
#endif
#ifdef PROF
__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
#define PST(k) do { if (threadIdx.x == 0 && p.prof) p.prof[blockIdx.x * 16 + (k)] = gtime(); } while (0)
#else
#define PST(k)
#endif

// per-warp TMA ring over a fixed byte sequence: W13 stages (lt, s) of this warp, then W2 records (rowblock, segment, 4 granules)
struct Ctx {
  const P* p; Sm* sm; uint32_t ring, bars; int lane, wid, S, q; unsigned long long wns;
  int r, s, j, sg, gc; bool ph2;
  __device__ __forceinline__ void load(int slot, const uint8_t* src, int bytes) {
    const uint32_t dst = ring + slot * STG, bar = bars + slot * 8;
#if TMA
    if (lane == 0) {
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(bar), "r"(bytes) : "memory");
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
    }
#else
    for (int k = lane; k < bytes / 16; k += 32) cp16(dst + k * 16, src + k * 16);
    asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" :: "r"(bar) : "memory");
#endif
  }
  __device__ __forceinline__ void load2(int slot, const uint8_t* sa, const uint8_t* sb) {  // two 2304 B half-stages
    const uint32_t dst = ring + slot * STG, bar = bars + slot * 8;
#if TMA
    if (lane == 0) {
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(bar), "n"(STG) : "memory");
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], 2304, [%2];\n" :: "r"(dst), "l"(sa), "r"(bar) : "memory");
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], 2304, [%2];\n" :: "r"(dst + 2304), "l"(sb), "r"(bar) : "memory");
    }
#else
    for (int k = lane; k < 144; k += 32) { cp16(dst + k * 16, sa + k * 16); cp16(dst + 2304 + k * 16, sb + k * 16); }
    asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" :: "r"(bar) : "memory");
#endif
  }
  __device__ __forceinline__ void issue(int slot) {
    const int ks = sm->ks, g = sm->g;
    if (!ph2) {
#ifdef MT2
      const int u = r * (WPC / ks) + wid / ks;
      if (u < 2 * g) {
        const int lt = 2 * u, gq = lt >> 2, si = gq < sm->seg[0].ng ? 0 : (gq < sm->seg[0].ng + sm->seg[1].ng ? 1 : 2);
        const Seg& sq = sm->seg[si];
        const int gl = gq - (si ? sm->seg[0].ng + (si > 1 ? sm->seg[1].ng : 0) : 0);
        const int mt = (sq.gr0 + gl) * 4 + (lt & 3), h = (wid % ks) * (16 / ks) + s;
        const uint8_t* src = p->w1 + (((size_t)(p->lo + sq.e) * 512 + mt) * 8 + (h >> 1)) * STG + (h & 1) * 2304;
        load2(slot, src, src + 8 * STG);
        if (++s == 16 / ks) { s = 0; r++; }
        return;
      }
      ph2 = true;
    }
    if (false) {
#endif
      const int lt = r * (WPC / ks) + wid / ks;
      if (lt < 4 * g) {
        const int gq = lt >> 2, si = gq < sm->seg[0].ng ? 0 : (gq < sm->seg[0].ng + sm->seg[1].ng ? 1 : 2);
        const Seg& sq = sm->seg[si];
        const int gl = gq - (si ? sm->seg[0].ng + (si > 1 ? sm->seg[1].ng : 0) : 0);
        const int mt = (sq.gr0 + gl) * 4 + (lt & 3), st = (wid % ks) * (8 / ks) + s;
        load(slot, p->w1 + (((size_t)(p->lo + sq.e) * 512 + mt) * 8 + st) * STG, STG);
        if (++s == 8 / ks) { s = 0; r++; }
        return;
      }
      ph2 = true;
    }
    while (j < RB / WPC) {
      if (sg < sm->ns) {
        const Seg& sq = sm->seg[sg];
        const int n = min(4, sq.ng - gc), rb = wid + j * WPC;
        load(slot, p->w2 + (((size_t)sq.e * RB + rb) * GR + sq.gr0 + gc) * REC, n * REC);
        gc += 4; if (gc >= sq.ng) { gc = 0; sg++; }
        return;
      }
      sg = 0; j++;
    }
  }
  __device__ __forceinline__ uint32_t acquire() {
    const int slot = q % S;
#ifdef PROF
    const unsigned long long w0 = gtime();
#endif
    mb_wait(bars + slot * 8, (q / S) & 1);
    __syncwarp();
#ifdef PROF
    wns += gtime() - w0;
#endif
    q++;
    return ring + slot * STG;
  }
  __device__ __forceinline__ void release() { __syncwarp(); issue((q - 1) % S); }
};

template <int NT>
__device__ __forceinline__ void run13(Ctx& C, const uint16_t* list, int off, int cnt, int s0, int s1, float (&f)[NT][4]) {
  const P& p = *C.p;
  const int lane = C.lane, g = lane >> 2, t = lane & 3;
  constexpr int CH = NT == 1 ? 4 : (NT == 2 ? 2 : 1);
#ifndef FD1
#define FD1 7
#endif
  constexpr int D = NT == 1 ? FD1 : (NT == 2 ? 3 : 1);
  const uint4* rp[NT];
#pragma unroll
  for (int n = 0; n < NT; n++) { int i = n * 8 + g; int r = list[off + (i < cnt ? i : 0)]; rp[n] = (const uint4*)(p.x + (size_t)(r >> 3) * HD + t * 16); }
  uint4 B[D + 1][NT][2];
  auto ldb = [&](uint4 (&b)[NT][2], int s, int c) {
#pragma unroll
    for (int n = 0; n < NT; n++) {
#ifdef NOXB
      b[n][0] = make_uint4(s, c, n, 1); b[n][1] = make_uint4(c, s, n, 2);
#else
      const uint4* q = rp[n] + s * 64 + c * 8; b[n][0] = __ldg(q); b[n][1] = __ldg(q + 1);
#endif
    }
  };
  float acc[NT][CH][4];
#pragma unroll
  for (int n = 0; n < NT; n++)
#pragma unroll
    for (int h = 0; h < CH; h++) acc[n][h][0] = acc[n][h][1] = acc[n][h][2] = acc[n][h][3] = 0.f;
#pragma unroll
  for (int c = 0; c < D; c++) ldb(B[c], s0, c);
  for (int s = s0; s < s1; s++) {
    const uint32_t slot = C.acquire() + lane * 16;
#ifdef NOCOMP
    { const uint4 w = lds(slot); acc[0][0][0] += __uint_as_float(w.x ^ w.y); C.release(); continue; }
#endif
    const uint2 sh0 = lds64(slot - lane * 8 + 2048), sh1 = lds64(slot - lane * 8 + 4352);
    const uint4 sc = make_uint4(sh0.x, sh0.y, sh1.x, sh1.y);
#pragma unroll
    for (int c = 0; c < 8; c++) {
      const uint4 w = lds(slot + c * 512 + (c >= 4 ? 256 : 0));
      const uint32_t sw = (c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w));
      const uint32_t sA = sc2(sw, (c & 1) ? 0x4242 : 0x4040), sB = sc2(sw, (c & 1) ? 0x4343 : 0x4141);
      if (c + D < 8) ldb(B[(c + D) % (D + 1)], s, c + D);
      else if (s + 1 < s1) ldb(B[(c + D) % (D + 1)], s + 1, c + D - 8);
      const uint4 (&bc)[NT][2] = B[c % (D + 1)];
      const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const uint32_t q = ww[j];
        const uint32_t a0 = BMUL(q & 0x81C081C0u, sA);
        const uint32_t a1 = BMUL(((q << 3) & 0x01C001C0u) | ((q << 1) & 0x80008000u), sB);
        const uint32_t a2 = BMUL((q << 6) & 0x81C081C0u, sA);
        const uint32_t a3 = BMUL(((q >> 5) & 0x01C001C0u) | ((q << 5) & 0x80008000u), sB);
#pragma unroll
        for (int n = 0; n < NT; n++) {
          const uint32_t* bb = (const uint32_t*)&bc[n][j >> 1];
          MMA(acc[n][j % CH], a0, a1, a2, a3, bb[(j & 1) * 2], bb[(j & 1) * 2 + 1]);
        }
      }
    }
    C.release();
  }
#pragma unroll
  for (int n = 0; n < NT; n++)
#pragma unroll
    for (int i = 0; i < 4; i++) { float v = 0.f;
#pragma unroll
      for (int h = 0; h < CH; h++) v += acc[n][h][i];
      f[n][i] = v; }
}

// one W13 tile part: (optional group split-K reduction) then SwiGLU into the CTA's inter buffer
template <int NT>
__device__ __forceinline__ void tile13(Ctx& C, const uint16_t* list, uint8_t* smem, int r, int lt) {
  Sm* sm = C.sm; const P& p = *C.p;
  const int lane = C.lane, wid = C.wid, g = lane >> 2, t = lane & 3, ks = sm->ks;
  const int gq = lt >> 2, si = gq < sm->seg[0].ng ? 0 : (gq < sm->seg[0].ng + sm->seg[1].ng ? 1 : 2);
  const Seg& sq = sm->seg[si];
  const int gl = gq - (si ? sm->seg[0].ng + (si > 1 ? sm->seg[1].ng : 0) : 0);
  const int e = sq.e, cnt = sq.T, off = sm->off[e];
  const int s0 = (wid % ks) * (8 / ks);
  float f[NT][4];
  run13<NT>(C, list, off, cnt, s0, s0 + 8 / ks, f);
  if (ks > 1) {
    float4* red = (float4*)(smem + sm->red);
    const int ntm = sm->ntm;
#pragma unroll
    for (int n = 0; n < NT; n++) red[((wid * 2 + (r & 1)) * ntm + n) * 32 + lane] = make_float4(f[n][0], f[n][1], f[n][2], f[n][3]);
    gbar(1 + wid / ks, ks * 32);
    if (wid % ks != (r & (ks - 1))) return;
    const int w0 = wid - wid % ks;
    for (int o = w0; o < w0 + ks; o++) {
      if (o == wid) continue;
#pragma unroll
      for (int n = 0; n < NT; n++) { float4 v = red[((o * 2 + (r & 1)) * ntm + n) * 32 + lane]; f[n][0] += v.x; f[n][1] += v.y; f[n][2] += v.z; f[n][3] += v.w; }
    }
  }
  const float gs = p.g1[p.lo + e];
  __nv_bfloat16* is = (__nv_bfloat16*)(smem + sm->ioff + sq.ioff) + (size_t)gl * NT * 8 * 32 + (lt & 3) * 8 + g;
#pragma unroll
  for (int n = 0; n < NT; n++)
#pragma unroll
    for (int h = 0; h < 2; h++) {
      const float gt = f[n][h] * gs, up = f[n][2 + h] * gs;
      is[(n * 8 + 2 * t + h) * 32] = __float2bfloat16(gt / (1.f + __expf(-gt)) * up);
    }
}

#ifdef MT2
// two adjacent W13 tiles per warp: each ring slot holds the same 256-K half-stage of both, so every activation B fragment
// is loaded once for two tiles (half the activation traffic and B registers of run13)
template <int NT>
__device__ __forceinline__ void run13p(Ctx& C, const uint16_t* list, int off, int cnt, int h0, int h1, float (&f)[2][NT][4]) {
  const P& p = *C.p;
  const int lane = C.lane, g = lane >> 2, t = lane & 3;
  constexpr int CH = NT == 1 ? 2 : 1;
  constexpr int D = NT == 1 ? 3 : 1;  // (D + 1) divides 4
  const uint4* rp[NT];
#pragma unroll
  for (int n = 0; n < NT; n++) { int i = n * 8 + g; int r = list[off + (i < cnt ? i : 0)]; rp[n] = (const uint4*)(p.x + (size_t)(r >> 3) * HD + t * 16); }
  uint4 B[D + 1][NT][2];
  auto ldb = [&](uint4 (&b)[NT][2], int h, int c) {
#pragma unroll
    for (int n = 0; n < NT; n++) {
#ifdef NOXB
      b[n][0] = make_uint4(h, c, n, 1); b[n][1] = make_uint4(c, h, n, 2);
#else
      const uint4* q = rp[n] + (h >> 1) * 64 + ((h & 1) * 4 + c) * 8; b[n][0] = __ldg(q); b[n][1] = __ldg(q + 1);
#endif
    }
  };
  float acc[2][NT][CH][4];
#pragma unroll
  for (int k = 0; k < 2; k++)
#pragma unroll
    for (int n = 0; n < NT; n++)
#pragma unroll
      for (int hh = 0; hh < CH; hh++) acc[k][n][hh][0] = acc[k][n][hh][1] = acc[k][n][hh][2] = acc[k][n][hh][3] = 0.f;
#pragma unroll
  for (int c = 0; c < D; c++) ldb(B[c], h0, c);
  for (int h = h0; h < h1; h++) {
    const uint32_t slot = C.acquire() + lane * 16;
    uint2 sh[2];
#pragma unroll
    for (int k = 0; k < 2; k++) sh[k] = lds64(slot - lane * 8 + k * 2304 + 2048);
#pragma unroll
    for (int c = 0; c < 4; c++) {
      if (c + D < 4) ldb(B[(c + D) % (D + 1)], h, c + D);
      else if (h + 1 < h1) ldb(B[(c + D) % (D + 1)], h + 1, c + D - 4);
      const uint4 (&bc)[NT][2] = B[c % (D + 1)];
#pragma unroll
      for (int k = 0; k < 2; k++) {
        const uint4 w = lds(slot + k * 2304 + c * 512);
        const uint32_t sw = c < 2 ? sh[k].x : sh[k].y;
        const uint32_t sA = sc2(sw, (c & 1) ? 0x4242 : 0x4040), sB = sc2(sw, (c & 1) ? 0x4343 : 0x4141);
        const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
        for (int j = 0; j < 4; j++) {
          const uint32_t q = ww[j];
          const uint32_t a0 = BMUL(q & 0x81C081C0u, sA);
          const uint32_t a1 = BMUL(((q << 3) & 0x01C001C0u) | ((q << 1) & 0x80008000u), sB);
          const uint32_t a2 = BMUL((q << 6) & 0x81C081C0u, sA);
          const uint32_t a3 = BMUL(((q >> 5) & 0x01C001C0u) | ((q << 5) & 0x80008000u), sB);
#pragma unroll
          for (int n = 0; n < NT; n++) {
            const uint32_t* bb = (const uint32_t*)&bc[n][j >> 1];
            MMA(acc[k][n][j % CH], a0, a1, a2, a3, bb[(j & 1) * 2], bb[(j & 1) * 2 + 1]);
          }
        }
      }
    }
    C.release();
  }
#pragma unroll
  for (int k = 0; k < 2; k++)
#pragma unroll
    for (int n = 0; n < NT; n++)
#pragma unroll
      for (int i = 0; i < 4; i++) { float v = 0.f;
#pragma unroll
        for (int hh = 0; hh < CH; hh++) v += acc[k][n][hh][i];
        f[k][n][i] = v; }
}

template <int NT>
__device__ __forceinline__ void tile13p(Ctx& C, const uint16_t* list, uint8_t* smem, int r, int u) {
  Sm* sm = C.sm; const P& p = *C.p;
  const int lane = C.lane, wid = C.wid, g = lane >> 2, t = lane & 3, ks = sm->ks;
  const int lt = 2 * u, gq = lt >> 2, si = gq < sm->seg[0].ng ? 0 : (gq < sm->seg[0].ng + sm->seg[1].ng ? 1 : 2);
  const Seg& sq = sm->seg[si];
  const int gl = gq - (si ? sm->seg[0].ng + (si > 1 ? sm->seg[1].ng : 0) : 0);
  const int e = sq.e, cnt = sq.T, off = sm->off[e];
  const int h0 = (wid % ks) * (16 / ks);
  float f[2][NT][4];
  run13p<NT>(C, list, off, cnt, h0, h0 + 16 / ks, f);
  if (ks > 1) {
    float4* red = (float4*)(smem + sm->red);
    const int ntm = sm->ntm;
#pragma unroll
    for (int k = 0; k < 2; k++)
#pragma unroll
      for (int n = 0; n < NT; n++) red[(((wid * 2 + (r & 1)) * 2 + k) * ntm + n) * 32 + lane] = make_float4(f[k][n][0], f[k][n][1], f[k][n][2], f[k][n][3]);
    gbar(1 + wid / ks, ks * 32);
    if (wid % ks != (r & (ks - 1))) return;
    const int w0 = wid - wid % ks;
    for (int o = w0; o < w0 + ks; o++) {
      if (o == wid) continue;
#pragma unroll
      for (int k = 0; k < 2; k++)
#pragma unroll
        for (int n = 0; n < NT; n++) { float4 v = red[(((o * 2 + (r & 1)) * 2 + k) * ntm + n) * 32 + lane]; f[k][n][0] += v.x; f[k][n][1] += v.y; f[k][n][2] += v.z; f[k][n][3] += v.w; }
    }
  }
  const float gs = p.g1[p.lo + e];
#pragma unroll
  for (int k = 0; k < 2; k++) {
    __nv_bfloat16* is = (__nv_bfloat16*)(smem + sm->ioff + sq.ioff) + (size_t)gl * NT * 8 * 32 + ((lt + k) & 3) * 8 + g;
#pragma unroll
    for (int n = 0; n < NT; n++)
#pragma unroll
      for (int hh = 0; hh < 2; hh++) {
        const float gt = f[k][n][hh] * gs, up = f[k][n][2 + hh] * gs;
        is[(n * 8 + 2 * t + hh) * 32] = __float2bfloat16(gt / (1.f + __expf(-gt)) * up);
      }
  }
}
#endif

// W2 column slice of one segment for one 64-row block: partial y (fp32) reduced into yacc
template <int NT>
__device__ __forceinline__ void seg2(Ctx& C, const uint16_t* list, uint8_t* smem, int sgi, int rb) {
  Sm* sm = C.sm; const P& p = *C.p;
  const int lane = C.lane, g = lane >> 2, t = lane & 3;
  const Seg& sq = sm->seg[sgi];
  const uint32_t ib = (uint32_t)__cvta_generic_to_shared(smem + sm->ioff + sq.ioff) + (g * 32 + t * 8) * 2;
  float acc[4][NT][4];
#pragma unroll
  for (int rt = 0; rt < 4; rt++)
#pragma unroll
    for (int n = 0; n < NT; n++) acc[rt][n][0] = acc[rt][n][1] = acc[rt][n][2] = acc[rt][n][3] = 0.f;
  for (int gc = 0; gc < sq.ng; gc += 4) {
    const uint32_t slot = C.acquire();
    const int n4 = min(4, sq.ng - gc);
#ifdef NOCOMP
    { const uint2 w = lds64(slot + lane * 8); acc[0][0][0] += __uint_as_float(w.x ^ w.y); C.release(); continue; }
#endif
#pragma unroll
    for (int gi = 0; gi < 4; gi++) {
      if (gi < n4) {
        uint4 bv[NT];
#pragma unroll
        for (int n = 0; n < NT; n++) bv[n] = lds(ib + ((gc + gi) * NT * 8 + n * 8) * 64);
#pragma unroll
        for (int rt = 0; rt < 4; rt++) {
          const uint32_t base = slot + gi * REC + rt * 288;
          const uint2 w = lds64(base + lane * 8);
          const uint32_t s16 = lds16(base + 256 + g * 4 + (t >> 1) * 2);
          const uint32_t sA = sc2(s16, 0x4040), sB = sc2(s16, 0x4141);
#pragma unroll
          for (int h = 0; h < 2; h++) {
            const uint32_t q = h ? w.y : w.x;
            const uint32_t a0 = BMUL(q & 0x81C081C0u, sA);
            const uint32_t a1 = BMUL(((q << 3) & 0x01C001C0u) | ((q << 1) & 0x80008000u), sB);
            const uint32_t a2 = BMUL((q << 6) & 0x81C081C0u, sA);
            const uint32_t a3 = BMUL(((q >> 5) & 0x01C001C0u) | ((q << 5) & 0x80008000u), sB);
#pragma unroll
            for (int n = 0; n < NT; n++) MMA(acc[rt][n], a0, a1, a2, a3, h ? bv[n].z : bv[n].x, h ? bv[n].w : bv[n].y);
          }
        }
      }
    }
    C.release();
  }
  const int off = sm->off[sq.e], T = sq.T;
  const bool odd = (g & 1) != 0; (void)odd;
#pragma unroll
  for (int rt = 0; rt < 4; rt++)
#pragma unroll
    for (int n = 0; n < NT; n++) {
      const float c0 = acc[rt][n][0], c1 = acc[rt][n][1], c2 = acc[rt][n][2], c3 = acc[rt][n][3];
      const int row = rb * 64 + rt * 16 + 2 * g;
#if TMA
      const float r0 = __shfl_xor_sync(0xffffffffu, odd ? c0 : c1, 4), r1 = __shfl_xor_sync(0xffffffffu, odd ? c2 : c3, 4);
      const int tok = n * 8 + 2 * t + (odd ? 1 : 0);
      if (tok < T) {
        float* a = p.yacc + (size_t)list[off + tok] * HD + (odd ? row - 2 : row);
        const float4 v = odd ? make_float4(r0, r1, c1, c3) : make_float4(c0, c2, r0, r1);
#ifndef NORED
        asm volatile("red.global.add.v4.f32 [%0], {%1,%2,%3,%4};\n" :: "l"(a), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
#else
        if (v.x == 123.f) *a = v.y;
#endif
      }
#else
      const int ta = n * 8 + 2 * t;
      if (ta < T) { float* a = p.yacc + (size_t)list[off + ta] * HD + row; atomicAdd(a, c0); atomicAdd(a + 1, c2); }
      if (ta + 1 < T) { float* a = p.yacc + (size_t)list[off + ta + 1] * HD + row; atomicAdd(a, c1); atomicAdd(a + 1, c3); }
#endif
    }
}

__global__ void __launch_bounds__(WPC * 32, 1) moe_f(P p) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5, R = p.M * 8;
  const uint32_t sb = (uint32_t)__cvta_generic_to_shared(smem);
  Sm* sm = (Sm*)(smem + WPC * SMAX * 8);
  uint16_t* list = (uint16_t*)(sm + 1);
  uint8_t* lid = (uint8_t*)(list + R);
  const int fixed = (WPC * SMAX * 8 + (int)sizeof(Sm) + R * 3 + 127) & ~127;
  PST(0);
  if (lane < SMAX) mb_init(sb + (wid * SMAX + lane) * 8, TMA ? 1 : 32);
#if TMA
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#endif
  if (tid < NLOC) sm->cnt[tid] = 0;
  __syncthreads();
  for (int i = tid; i < R; i += blockDim.x) { unsigned e = p.ids[i] - p.lo; lid[i] = e < NLOC ? e : 0xff; if (e < NLOC) atomicAdd(&sm->cnt[e], 1); if (i < 256) sm->tws[i] = p.tw[i]; }
  if (tid < NLOC) sm->g2s[tid] = p.g2[p.lo + tid];
  __syncthreads();
  if (wid == 0) {
    const int c = sm->cnt[lane]; int v = c;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) { int n = __shfl_up_sync(0xffffffffu, v, o); if (lane >= o) v += n; }
    sm->off[lane] = v - c; sm->fill[lane] = v - c;
    const unsigned m = __ballot_sync(0xffffffffu, c > 0);
    if (c > 0) sm->act[__popc(m & ((1u << lane) - 1))] = lane;
    __syncwarp();
    if (lane == 0) {
      const int na = __popc(m);
      // cost-weighted granule ranges: a granule of an expert with more tokens costs more (x re-reads, mma per n-block)
      long long Ct = 0;
      for (int a = 0; a < na; a++) { const int T = sm->cnt[sm->act[a]]; Ct += GR * (T <= 8 ? FW1 : (T <= 16 ? FW2 : FW4)); }
      const long long lo_c = (long long)blockIdx.x * Ct / gridDim.x, hi_c = (long long)(blockIdx.x + 1) * Ct / gridDim.x;
      int ns = 0, io = 0, ntm = 1, G1 = 0, G0 = 0;
      for (int k = 0; k < 3; k++) sm->seg[k] = Seg{0, 0, 0, 0, 1, 0};
      long long base = 0;
      for (int a = 0; a < na; a++) {
        const int e = sm->act[a], T = sm->cnt[e], w = T <= 8 ? FW1 : (T <= 16 ? FW2 : FW4);
        const int NT = T <= 8 ? 1 : (T <= 16 ? 2 : 4);
        int k0 = lo_c > base ? (int)((lo_c - base + w - 1) / w) : 0, k1 = hi_c > base ? (int)((hi_c - base + w - 1) / w) : 0;
        k0 = min(k0, GR); k1 = min(k1, GR);
        if (k1 > k0 && ns < 3) { sm->seg[ns++] = Seg{e, k0, k1 - k0, T, NT, io}; io += (k1 - k0) * NT * 8 * 64; ntm = max(ntm, NT); G1 += k1 - k0; }
        base += (long long)GR * w;
      }
      const int g = G1 - G0, tiles = 4 * g;
      int bk = 1; float bc = 1e30f;
      for (int ks = 1; ks <= 8; ks <<= 1) {
        if (ks > 1 && ntm > 2) break;
        const int rounds = (tiles / MTL * ks + WPC - 1) / WPC;
        const float cost = rounds * (8.f * MTL / ks + (ks > 1 ? 0.35f : 0.f));
        if (cost < bc - 1e-3f) { bc = cost; bk = ks; }
      }
#ifdef KSF
      if (KSF == 1 || ntm <= 2) bk = KSF;
#endif
      const int red = bk > 1 ? WPC * 2 * ntm * 512 * MTL : 0, roff = (fixed + io + red + 127) & ~127;
      int S = (p.smem - roff) / (WPC * STG);
      sm->roff = roff; sm->ntm = ntm;
      sm->na = na; sm->g = g; sm->ks = bk; sm->ns = ns; sm->rounds = (tiles / MTL * bk + WPC - 1) / WPC;
      sm->ioff = fixed; sm->red = fixed + io; sm->S = S < SMAX ? (S < 1 ? 1 : S) : SMAX;
      if (S < 1) { sm->g = 0; asm volatile("trap;"); }
    }
  }
  __syncthreads();
  Ctx C;
  C.p = &p; C.sm = sm; C.lane = lane; C.wid = wid; C.S = sm->S; C.q = 0; C.wns = 0;
  C.ring = sb + sm->roff + wid * C.S * STG;
  C.bars = sb + wid * SMAX * 8;
  C.r = 0; C.s = 0; C.j = 0; C.sg = 0; C.gc = 0; C.ph2 = false;
  const bool any = sm->g > 0;
  if (any)
    for (int i = 0; i < C.S; i++) C.issue(i);
  for (int i = tid; i < R; i += blockDim.x) { unsigned e = lid[i]; if (e < NLOC) list[atomicAdd(&sm->fill[e], 1)] = i; }
  __syncthreads();
  PST(1);
  if (any) {
    const int ks = sm->ks, g = sm->g;
    for (int r = 0; r < sm->rounds; r++) {
#ifdef MT2
      const int u = r * (WPC / ks) + wid / ks;
      if (u >= 2 * g) break;
      const int gq2 = u >> 1, si2 = gq2 < sm->seg[0].ng ? 0 : (gq2 < sm->seg[0].ng + sm->seg[1].ng ? 1 : 2);
      const int nt2 = sm->seg[si2].NT;
      if (nt2 == 1) tile13p<1>(C, list, smem, r, u);
      else if (nt2 == 2) tile13p<2>(C, list, smem, r, u);
      else tile13p<4>(C, list, smem, r, u);
      continue;
#endif
      const int lt = r * (WPC / ks) + wid / ks;
      if (lt >= 4 * g) break;
      const int gq = lt >> 2, si = gq < sm->seg[0].ng ? 0 : (gq < sm->seg[0].ng + sm->seg[1].ng ? 1 : 2);
      const int nt = sm->seg[si].NT;
      if (nt == 1) tile13<1>(C, list, smem, r, lt);
      else if (nt == 2) tile13<2>(C, list, smem, r, lt);
      else tile13<4>(C, list, smem, r, lt);
    }
  }
#ifdef PROF
  if (lane == 0 && p.prof) atomicAdd(p.prof + blockIdx.x * 16 + 8, C.wns);
  C.wns = 0;
  if (tid == 0 && p.prof) {
    unsigned long long cmp = 0;
    for (int k = 0; k < sm->ns; k++) cmp += (unsigned long long)sm->seg[k].ng * (sm->seg[k].NT == 1 ? 1ull : (sm->seg[k].NT == 2 ? 1000ull : 1000000ull));
    p.prof[blockIdx.x * 16 + 11] = cmp; p.prof[blockIdx.x * 16 + 12] = sm->ks * 10 + sm->S;
  }
#endif
  __syncthreads();
  PST(2);
  if (any) {
    for (int j = 0; j < RB / WPC; j++)
      for (int k = 0; k < sm->ns; k++) {
        const int nt = sm->seg[k].NT, rb = wid + j * WPC;
        if (nt == 1) seg2<1>(C, list, smem, k, rb);
        else if (nt == 2) seg2<2>(C, list, smem, k, rb);
        else seg2<4>(C, list, smem, k, rb);
      }
  }
#ifdef PROF
  if (lane == 0 && p.prof) atomicAdd(p.prof + blockIdx.x * 16 + 9, C.wns);
#endif
  __syncthreads();
  PST(3);
#ifdef LASTA
  // no grid barrier: each CTA adds its granule count of every segment to the counters of that segment's tokens; the CTA
  // that brings a token to 128 x (its local routes) normalizes and combines it. Counters are reset by their completers.
  {
    int* tokd = p.ctr + 9280;
    if (tid == 0) sm->ntq = 0;
    __syncthreads();
    for (int k = 0; k < sm->ns; k++) {
      const int e = sm->seg[k].e, T = sm->seg[k].T, off = sm->off[e], ng = sm->seg[k].ng;
      if (tid < T) {
        const int t = list[off + tid] >> 3;
        int nl = 0;
#pragma unroll
        for (int j = 0; j < 8; j++) nl += lid[t * 8 + j] < NLOC;
        int old;
        asm volatile("atom.add.acq_rel.gpu.global.s32 %0, [%1], %2;\n" : "=r"(old) : "l"(tokd + t), "r"(ng) : "memory");
        if (old + ng == GR * nl) { tokd[t] = 0; sm->tq[atomicAdd(&sm->ntq, 1)] = t; }
      }
    }
    __syncthreads();
    PST(4);
    const int ntq = sm->ntq;
    float* stg = (float*)(smem + sm->roff);
    const bool stage = 8 * HD * 4 <= sm->S * WPC * STG;
    constexpr int CPT = HD / (WPC * 32);
    for (int qi = 0; qi < ntq; qi++) {
      const int t = sm->tq[qi];
      float ssk[8];
#pragma unroll
      for (int k = 0; k < 8; k++) ssk[k] = 0.f;
#pragma unroll
      for (int c = 0; c < CPT; c += 8) {
        const int col = (c / 8 * WPC * 32 + tid) * 8;
        const float4 n0 = *(const float4*)(p.nw + col), n1 = *(const float4*)(p.nw + col + 4);
        float4 u[8][2];
#pragma unroll
        for (int k = 0; k < 8; k++) {
          if (lid[t * 8 + k] < NLOC) { const float* yr = p.yacc + (size_t)(t * 8 + k) * HD + col; u[k][0] = __ldcg((const float4*)yr); u[k][1] = __ldcg((const float4*)(yr + 4)); }
          else u[k][0] = u[k][1] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
#pragma unroll
        for (int k = 0; k < 8; k++) {
          if (stage && lid[t * 8 + k] < NLOC) { *(float4*)(stg + k * HD + col) = u[k][0]; *(float4*)(stg + k * HD + col + 4) = u[k][1]; }
          const float a0 = u[k][0].x * n0.x, a1 = u[k][0].y * n0.y, a2 = u[k][0].z * n0.z, a3 = u[k][0].w * n0.w;
          const float a4 = u[k][1].x * n1.x, a5 = u[k][1].y * n1.y, a6 = u[k][1].z * n1.z, a7 = u[k][1].w * n1.w;
          ssk[k] += a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3 + a4 * a4 + a5 * a5 + a6 * a6 + a7 * a7;
        }
      }
#pragma unroll
      for (int k = 0; k < 8; k++) {
        float ss = ssk[k];
#pragma unroll
        for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        if (lane == 0) sm->ssr[k][wid] = ss;
      }
      __syncthreads();
      PST(6);
      float coef[8];
#pragma unroll
      for (int k = 0; k < 8; k++) {
        const unsigned e = lid[t * 8 + k];
        float s = 0.f;
#pragma unroll
        for (int w = 0; w < WPC; w++) s += sm->ssr[k][w];
        const float gs = e < NLOC ? sm->g2s[e] : 0.f;
        coef[k] = e < NLOC ? sm->tws[t * 8 + k] * gs * rsqrtf(s * gs * gs * (1.f / HD) + 1e-8f) : 0.f;
      }
      PST(7);
#pragma unroll
      for (int c = 0; c < CPT; c += 8) {
        const int col = (c / 8 * WPC * 32 + tid) * 8;
        const float4 n0 = *(const float4*)(p.nw + col), n1 = *(const float4*)(p.nw + col + 4);
        float o[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int k = 0; k < 8; k++) {
          if (lid[t * 8 + k] >= NLOC) continue;
          float4 u0, u1;
          if (stage) { u0 = *(const float4*)(stg + k * HD + col); u1 = *(const float4*)(stg + k * HD + col + 4); }
          else { const float* yr = p.yacc + (size_t)(t * 8 + k) * HD + col; u0 = __ldcg((const float4*)yr); u1 = __ldcg((const float4*)(yr + 4)); }
          const float cf = coef[k];
          o[0] += cf * u0.x; o[1] += cf * u0.y; o[2] += cf * u0.z; o[3] += cf * u0.w;
          o[4] += cf * u1.x; o[5] += cf * u1.y; o[6] += cf * u1.z; o[7] += cf * u1.w;
        }
        uint4 ov; __nv_bfloat162* ob = (__nv_bfloat162*)&ov;
        ob[0] = __floats2bfloat162_rn(o[0] * n0.x, o[1] * n0.y); ob[1] = __floats2bfloat162_rn(o[2] * n0.z, o[3] * n0.w);
        ob[2] = __floats2bfloat162_rn(o[4] * n1.x, o[5] * n1.y); ob[3] = __floats2bfloat162_rn(o[6] * n1.z, o[7] * n1.w);
        *(uint4*)(p.out + (size_t)t * HD + col) = ov;
#pragma unroll
        for (int k = 0; k < 8; k++)
          if (lid[t * 8 + k] < NLOC) { float* yr = p.yacc + (size_t)(t * 8 + k) * HD + col; __stcg((float4*)yr, make_float4(0.f, 0.f, 0.f, 0.f)); __stcg((float4*)(yr + 4), make_float4(0.f, 0.f, 0.f, 0.f)); }
      }
      __syncthreads();
    }
    if ((int)blockIdx.x < p.M) {  // tokens without a local route
      int nl = 0;
#pragma unroll
      for (int j = 0; j < 8; j++) nl += lid[blockIdx.x * 8 + j] < NLOC;
      if (nl == 0)
        for (int c = tid * 8; c < HD; c += WPC * 256) *(uint4*)(p.out + (size_t)blockIdx.x * HD + c) = make_uint4(0, 0, 0, 0);
    }
    PST(5);
    return;
  }
#endif
  if (tid == 0) {
    int old;
    asm volatile("atom.add.release.gpu.global.s32 %0, [%1], 1;\n" : "=r"(old) : "l"(p.ctr) : "memory");
    sm->target = (old / (int)gridDim.x + 1) * (int)gridDim.x;
  }
  __syncthreads();
  // combine: token b's 4096 columns split over P CTAs (P = 32 at M <= 4); per-slice partial sums of squares go through a
  // flag-synchronized exchange (flags carry the call epoch: no reset), then each CTA normalizes and sums its column slice.
  int P = 32;
  while (P > 1 && P * p.M > (int)gridDim.x) P >>= 1;
  if ((int)blockIdx.x >= p.M * P) return;
  const int b = blockIdx.x / P, js = blockIdx.x % P, nthr = HD / 8 / P;
  const bool act = tid < nthr;
  const int col = (js * nthr + (act ? tid : 0)) * 8;
  int* flg = p.ctr + 64 + b * 32;
  float* ssp = (float*)(p.ctr + 64 + 1024) + b * 8 * 32;
  float gsk[8], twk[8];
#pragma unroll
  for (int k = 0; k < 8; k++) { const unsigned e = lid[b * 8 + k]; gsk[k] = e < NLOC ? p.g2[p.lo + e] : 0.f; twk[k] = p.tw[b * 8 + k]; }
  const float4 n0 = *(const float4*)(p.nw + col), n1 = *(const float4*)(p.nw + col + 4);
  if (tid == 0) { while (ldacq(p.ctr) < sm->target) __nanosleep(32); }
  __syncthreads();
  PST(4);
  const int epoch = sm->target / (int)gridDim.x;
  {
    float4 u[8][2];
#pragma unroll
    for (int k = 0; k < 8; k++) {
      if (act && gsk[k] != 0.f) { const float* yr = p.yacc + (size_t)(b * 8 + k) * HD + col; u[k][0] = __ldcg((const float4*)yr); u[k][1] = __ldcg((const float4*)(yr + 4)); }
      else u[k][0] = u[k][1] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
#pragma unroll
    for (int k = 0; k < 8; k++) {
      const float a0 = u[k][0].x * n0.x, a1 = u[k][0].y * n0.y, a2 = u[k][0].z * n0.z, a3 = u[k][0].w * n0.w;
      const float a4 = u[k][1].x * n1.x, a5 = u[k][1].y * n1.y, a6 = u[k][1].z * n1.z, a7 = u[k][1].w * n1.w;
      float ss = a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3 + a4 * a4 + a5 * a5 + a6 * a6 + a7 * a7;
#pragma unroll
      for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
      if (lane == 0) sm->ssr[k][wid] = ss;
    }
  }
  __syncthreads();
  const int nw_ = (nthr + 31) / 32;
  if (tid == 0) {  // one thread writes all partials, so its release store alone orders them
#pragma unroll
    for (int k = 0; k < 8; k++) { float s = 0.f; for (int w = 0; w < nw_; w++) s += sm->ssr[k][w]; ssp[k * 32 + js] = s; }
    asm volatile("st.release.gpu.global.b32 [%0], %1;\n" :: "l"(flg + js), "r"(epoch) : "memory");
  }
  if (wid == 0) {
    if (lane < P) { while (ldacq(flg + lane) != epoch) __nanosleep(20); }
    __syncwarp();
    if (lane < 8) {
      float s = 0.f;
      for (int j = 0; j < P; j++) s += __ldcg(ssp + lane * 32 + j);
      sm->ssr[lane][0] = s;
    }
  }
  __syncthreads();
  PST(6);
  float coef[8];
#pragma unroll
  for (int k = 0; k < 8; k++) { const float gs = gsk[k]; coef[k] = gs != 0.f ? twk[k] * gs * rsqrtf(sm->ssr[k][0] * gs * gs * (1.f / HD) + 1e-8f) : 0.f; }
  PST(7);
  if (act) {
    float4 u[8][2];
#pragma unroll
    for (int k = 0; k < 8; k++) {
      if (gsk[k] != 0.f) { const float* yr = p.yacc + (size_t)(b * 8 + k) * HD + col; u[k][0] = __ldcg((const float4*)yr); u[k][1] = __ldcg((const float4*)(yr + 4)); }
      else u[k][0] = u[k][1] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    float o[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int k = 0; k < 8; k++) {
      const float cf = coef[k];
      o[0] += cf * u[k][0].x; o[1] += cf * u[k][0].y; o[2] += cf * u[k][0].z; o[3] += cf * u[k][0].w;
      o[4] += cf * u[k][1].x; o[5] += cf * u[k][1].y; o[6] += cf * u[k][1].z; o[7] += cf * u[k][1].w;
    }
    uint4 ov; __nv_bfloat162* ob = (__nv_bfloat162*)&ov;
    ob[0] = __floats2bfloat162_rn(o[0] * n0.x, o[1] * n0.y); ob[1] = __floats2bfloat162_rn(o[2] * n0.z, o[3] * n0.w);
    ob[2] = __floats2bfloat162_rn(o[4] * n1.x, o[5] * n1.y); ob[3] = __floats2bfloat162_rn(o[6] * n1.z, o[7] * n1.w);
    *(uint4*)(p.out + (size_t)b * HD + col) = ov;
#pragma unroll
    for (int k = 0; k < 8; k++)
      if (gsk[k] != 0.f) { float* yr = p.yacc + (size_t)(b * 8 + k) * HD + col; __stcg((float4*)yr, make_float4(0.f, 0.f, 0.f, 0.f)); __stcg((float4*)(yr + 4), make_float4(0.f, 0.f, 0.f, 0.f)); }
  }
  PST(5);
}

static int g_gridF = 0, g_smemF = 0;
void moe(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor w1, torch::Tensor w2, torch::Tensor g1,
         torch::Tensor g2, torch::Tensor nw, torch::Tensor yacc, torch::Tensor ctr, torch::Tensor out, int64_t lo, torch::Tensor prof) {
  if (!g_gridF) {
    int dev; cudaGetDevice(&dev); int mx; cudaDeviceGetAttribute(&mx, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    int sms; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cudaFuncSetAttribute(moe_f, cudaFuncAttributeMaxDynamicSharedMemorySize, mx);
#ifdef SMREQ
    if (SMREQ * 1024 < mx) {  // leave the rest of the unified 256 KB to L1 (activation rows)
      mx = SMREQ * 1024;
      cudaFuncSetAttribute(moe_f, cudaFuncAttributePreferredSharedMemoryCarveout, (mx * 100 + 228 * 1024 - 1) / (228 * 1024));
    }
#endif
    g_smemF = mx; g_gridF = sms;
  }
  P p{w1.data_ptr<uint8_t>(), w2.data_ptr<uint8_t>(), g1.data_ptr<float>(), g2.data_ptr<float>(),
      (const __nv_bfloat16*)x.data_ptr(), ids.data_ptr<int>(), tw.data_ptr<float>(), nw.data_ptr<float>(),
      yacc.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr(), ctr.data_ptr<int>(), (int)x.size(0), (int)lo, g_smemF,
      prof.numel() ? (unsigned long long*)prof.data_ptr() : nullptr};
  moe_f<<<g_gridF, WPC * 32, g_smemF, at::cuda::getCurrentCUDAStream()>>>(p);
}
"""


_modF = {}
FS_MAX = 32  # fused split-K path for M <= FS_MAX (tokens per expert <= 32: one pass, NT <= 4)
FS_CFG = ((0, 3, "-DLASTA -DSMREQ=196"), (1, 3, "-DSMREQ=196"), (0, 3, "-DMT2 -DLASTA -DSMREQ=196"), (0, 3, "-DMT2 -DSMREQ=196"))  # (forced W13 K split or 0 = auto, ring depth cap, extra defines)
DIAGF = False  # prepare(): print the fused path per-CTA phase timeline (PROF build) for a few M


def _extF(ksf=0, smax=3, extra=""):
    k = (ksf, smax, extra)
    if k not in _modF:
        wpc = 16 if _big() else 8
        fl = ([] if "-DWPC=" in extra else ["-DWPC=%d" % wpc]) + ["-DSMAX=%d" % smax] + (["-DKSF=%d" % ksf] if ksf else []) + extra.replace("+", " ").split() + [d for d in _DBG if not d.startswith("-DCFG=")]
        _modF[k] = load_inline("moe_w4a16_v36F" + "".join(fl).replace("-", "_").replace("=", ""),
                               cpp_sources="void moe(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,torch::Tensor);",
                               cuda_sources=CUDA_F, functions=["moe"], extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"] + fl, verbose=False)
    return _modF[k]


def _repack2(w, sf):
    """W2 for the fused path: uint8 [64 rowblocks, 128 granules (32 K), 4 rowtiles, 288]: per 16-row tile and 32-K granule,
    32 lanes x 2 words (mma A fragments of two k16 steps; lane t holds K 8t..8t+7) then 32 B of block scales [g][blk][row g / g+8].
    Rows permuted inside each tile (mma row i < 8 -> 2i, i >= 8 -> 2(i-8)+1) so a lane's C fragment holds adjacent output rows."""
    R, K = w.shape[0], w.shape[1] * 2
    RT, NG = R // 16, K // 32
    dev = w.device
    pi = torch.tensor([2 * i for i in range(8)] + [2 * i + 1 for i in range(8)], device=dev)
    perm = (torch.arange(RT, device=dev)[:, None] * 16 + pi[None, :]).reshape(-1)
    codes = torch.stack((w & 15, w >> 4), -1).view(R, K)[perm].to(torch.int64)
    s8 = sf.view(torch.uint8)[perm].to(torch.int64)
    neg = (s8 >> 7).repeat_interleave(16, 1)
    codes = codes ^ (neg << 3)
    zero = ((s8 & 0x7F) == 0).repeat_interleave(16, 1)
    codes = torch.where(zero, torch.zeros_like(codes), codes)
    s8 = torch.where((s8 & 0x7F) == 0, torch.full_like(s8, 0x08), s8 & 0x7F)
    if ((s8 >> 3) == 0).any():
        raise ValueError("subnormal E4M3 block scales are not supported by this fast path")
    C = codes.view(RT, 16, NG, 4, 2, 4)
    a, b = C[:, :8], C[:, 8:]
    word = torch.zeros(a.shape[:-1], dtype=torch.int64, device=dev)
    for pair, (src, e0) in enumerate(((a, 0), (b, 0), (a, 2), (b, 2))):
        word = _place(word, src[..., e0], pair, 0)
        word = _place(word, src[..., e0 + 1], pair, 1)
    word = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32)
    word = word.permute(0, 2, 1, 3, 4).contiguous().view(torch.uint8).view(RT, NG, 256)
    S = s8.to(torch.uint8).view(RT, 2, 8, NG, 2).permute(0, 3, 2, 4, 1).contiguous().view(RT, NG, 32)
    rec = torch.cat((word, S), -1).view(RT // 4, 4, NG, 288).permute(0, 2, 1, 3).contiguous()
    return rec


def _fs(data, state, cfg):
    x = data["hidden_states"].contiguous()
    ids = data["topk_ids"].to(torch.int32).contiguous()
    tw = data["topk_weights"].float().contiguous()
    out = torch.empty_like(x)
    _extF(*cfg[1:]).moe(x, ids, tw, state["w1"], state["w2f"], state["g1"], state["g2"], state["nw"], state["yacc"],
                              state["ctrF"], out, data["config"]["local_start"], state["profF"])
    return out


def prepare(weights):
    global _T0
    _T0 = time.time()
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
    w2f = torch.stack([_repack2(weights["w2"][e], weights["w2_sf"][e]) for e in range(min(E, 32))])
    with ThreadPoolExecutor(8) as ex: list(ex.map(lambda f: f(), [lambda: _extL(0), lambda: _extL(1)] + [lambda k=k: _extS(k) for k in V17_KSF] + [lambda c=c: _extF(*c) for c in FS_CFG]
                                                + ([lambda c=c: _extF(c[0], c[1], (c[2] + " -DPROF").strip()) for c in FS_CFG] if DIAGF else [])))
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    st = {"w1": w1, "w2": w2, "g1": weights["w13_gs"].float() * 256.0, "g2": weights["w2_gs"].float() * 256.0,
            "nw": 1.0 + weights["post_norm_w"].float(), "ctr": torch.zeros(64, dtype=torch.int32, device=dev),
            "tcnt": torch.zeros(32 * 768, dtype=torch.int32, device=dev),
            "prof": torch.zeros((65536 * 16 if "-DPROF" in _DBG else 0,), dtype=torch.int64, device=dev),
            "ws": torch.empty((32 * 768 * 8 * 64 * 4,), dtype=torch.float32, device=dev),
            "ctrL": torch.zeros(64 + 2 * 32 * 64 * 8, dtype=torch.int32, device=dev), "w2f": w2f,
            "yacc": torch.zeros((FS_MAX * 8, 4096), dtype=torch.float32, device=dev), "ctrF": torch.zeros(9280 + 64, dtype=torch.int32, device=dev),
            "profF": torch.zeros((4096 if "-DPROF" in _DBG else 0,), dtype=torch.int64, device=dev), "wsL": torch.zeros((KSL_MAX * 8 * 12288,), dtype=torch.float32, device=dev)}
    if DIAG:
        _diag(st, weights)
    el = time.time() - _T0
    print("v36 prepare %.0fs before tuning" % el, flush=True)
    if TUNE and FORCE is None and el < 200:
        st["tune"] = _tune(st)
    if DIAGF and time.time() - _T0 < 250:
        _diagF(st)
    return st


def _diagF(state):
    dev = state["w1"].device
    fl = torch.ones(2 ** 27, device=dev); sink = torch.empty((), device=dev)
    st = dict(state, profF=torch.zeros(4096, dtype=torch.int64, device=dev))
    wpc = 16 if _big() else 8
    for M in (1, 4, 8, 16, 32):
        data = _synth(M, 900 + M, dev)
        ids = data["topk_ids"]; l = ids[ids < 32]; c = torch.bincount(l.long(), minlength=32)
        for cfg in FS_CFG:
            ext = _extF(cfg[0], cfg[1], (cfg[2] + " -DPROF").strip())
            x = data["hidden_states"]; out = torch.empty_like(x)
            for _ in range(3):
                st["profF"].zero_(); sink.copy_(fl.sum())
                ext.moe(x, ids, data["topk_weights"], st["w1"], st["w2f"], st["g1"], st["g2"], st["nw"], st["yacc"], st["ctrF"], out, 0, st["profF"])
            torch.cuda.synchronize()
            P = st["profF"].view(-1, 16).cpu()
            n = int((P[:, 0] > 0).sum()); P = P[:n].double(); t0 = P[:, 0].min()
            q = lambda k, f: float((P[:, k] - t0).quantile(f)) / 1e3
            mx = lambda k: float((P[:, k] - t0).max()) / 1e3
            dA = (P[:, 2] - P[:, 1]).clamp(min=1); dB = (P[:, 3] - P[:, 2]).clamp(min=1)
            wA = float((P[:, 8] / (wpc * dA)).median()); wB = float((P[:, 9] / (wpc * dB)).median())
            cmp = P[:, 11].long(); n1 = cmp % 1000; n2 = (cmp // 1000) % 1000; n4 = cmp // 1000000
            cls = []
            for nm, sel in (("nt1", (n1 > 0) & (n2 == 0) & (n4 == 0)), ("nt2", (n2 > 0) & (n1 == 0) & (n4 == 0)), ("nt4", (n4 > 0) & (n1 == 0) & (n2 == 0))):
                if sel.any():
                    g = (n1 + n2 + n4)[sel].double()
                    cls.append(f"{nm} n{int(sel.sum())} A/gr {float((dA[sel] / g).median()) / 1e3:.2f} B/gr {float((dB[sel] / g).median()) / 1e3:.2f}")
            print(f"FD M{M} na {int((c > 0).sum())} maxT {int(c.max())} {_nm(('fs',) + cfg)}: pro {q(1, .5):.1f}/{q(1, 1):.1f} A {q(2, .1):.1f}/{q(2, .5):.1f}/{q(2, 1):.1f} "
                  f"B {q(3, .1):.1f}/{q(3, .5):.1f}/{q(3, 1):.1f} bar {mx(4):.1f} x6 {mx(6):.1f} x7 {mx(7):.1f} end {mx(5):.1f} waitA {wA:.2f} waitB {wB:.2f} ks/S {int(P[0, 12])} | "
                  + " | ".join(cls), flush=True)


DIAG = False  # diagnostics only: print a per-warp phase timeline from prepare() (untimed)


def _diag(state, weights):
    global _mod, _DBG
    keep, _mod, _DBG = (_mod, _DBG), None, _DBG + ["-DPROF"]
    ext = _ext()
    state = dict(state, prof=torch.zeros((65536 * 16,), dtype=torch.int64, device="cuda"))
    nw = torch.cuda.get_device_properties(0).multi_processor_count * (16 if _big() else 8)
    g = torch.Generator(device="cuda").manual_seed(7)
    for bs in (1, 8, 32, 256):
        sc = torch.rand((bs, 256), generator=g, device="cuda")
        ids = sc.topk(8, -1).indices
        pr = torch.bincount((ids // 32).flatten(), minlength=8)
        ids = ((ids - 32 * pr.argmax()) % 256).int().contiguous()
        x = torch.randn((bs, 4096), generator=g, device="cuda").bfloat16()
        tw = torch.full((bs, 8), 0.125, device="cuda")
        fl = torch.empty(2**28, device="cuda")
        for _ in range(3):
            fl.add_(1)
            inter = torch.empty((bs * 8, 4096), dtype=torch.bfloat16, device="cuda"); y = torch.empty_like(inter)
            ss = torch.empty((bs * 8,), device="cuda"); out = torch.empty_like(x)
            ext.moe(x, ids, tw, state["w1"], state["w2"], state["g1"], state["g2"], state["nw"], inter, y, ss, state["ctr"],
                    state["tcnt"], state["ws"], out, 0, state["prof"])
        torch.cuda.synchronize()
        P = state["prof"].view(-1, 16)[:nw].cpu().tolist()
        t0 = min(r[0] for r in P)
        q = lambda v, f: sorted(v)[min(len(v) - 1, int(f * len(v)))]
        col = lambda k: [(r[k] - t0) / 1e3 for r in P]
        mhz = [(r[14] - r[13]) / max(1, r[7] - r[0]) * 1e3 for r in P]
        cy = lambda k: [r[k] / q(mhz, .5) for r in P]
        print(f"DIAG bs {bs} na {P[0][11]} mode {P[0][12]} span {max(col(7)):.1f} routed {max(col(1)):.1f} "
              f"done p10/50/max {q(col(5), .1):.1f}/{q(col(5), .5):.1f}/{max(col(5)):.1f} comb {max(col(6)):.1f} mhz {q(mhz, .5):.0f} "
              f"g1 {q(cy(2), .5):.1f}/{max(cy(2)):.1f} g2 {q(cy(3), .5):.1f}/{max(cy(3)):.1f} dep {q(cy(4), .5):.1f}/{max(cy(4)):.1f} "
              f"tma {q(cy(9), .5):.1f}/{max(cy(9)):.1f} segs {max(r[8] for r in P)}", flush=True)
    _mod, _DBG = keep


def _kernel_small(data, state):
    x = data["hidden_states"].contiguous()
    M = x.shape[0]
    ids = data["topk_ids"].to(torch.int32).contiguous()
    tw = data["topk_weights"].float().contiguous()
    inter = torch.empty((M * 8, 4096), dtype=torch.bfloat16, device=x.device)
    y = torch.empty((M * 8, 4096), dtype=torch.bfloat16, device=x.device)
    ss = torch.empty((M * 8,), dtype=torch.float32, device=x.device)
    out = torch.empty_like(x)
    _ext(LARGE and M > 64).moe(x, ids, tw, state["w1"], state["w2"], state["g1"], state["g2"], state["nw"], inter, y, ss, state["ctr"],
                               state["tcnt"], state["ws"], out, data["config"]["local_start"], state["prof"])
    return out


import time
from concurrent.futures import ThreadPoolExecutor
_modL = {}
TUNE_M = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)
KSL_MAX = 64  # ring K-split workspace covers M <= 64
TUNE = True
V17_KSF = (0,)  # v17 path: 0 = its own K-split rule (4/2/1 by tiles per warp group), else forced
FORCE = None  # e.g. ("ring", 2, 1) or ("v17", 0, 0): skip the tuned table


def _extL(nx1=0):
    if nx1 not in _modL:
        fl = (["-DRPW=2", "-DNSLOT=4"] if _big() else ["-DRPW=1", "-DNSLOT=2"]) + (["-DNX1"] if nx1 else []) + list(_DBG)
        _modL[nx1] = load_inline("moe_w4a16_v30L" + "".join(fl).replace("-", "_").replace("=", ""),
                                 cpp_sources="void moe(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,torch::Tensor);",
                                 cuda_sources=CUDA_L, functions=["moe"], extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"] + fl, verbose=False)
    return _modL[nx1]


def _ring(data, state, ks, nx1):
    x = data["hidden_states"].contiguous()
    M = x.shape[0]
    ids = data["topk_ids"].to(torch.int32).contiguous()
    tw = data["topk_weights"].float().contiguous()
    inter = torch.empty((M * 8, 4096), dtype=torch.bfloat16, device=x.device)
    y = torch.empty((M * 8, 4096), dtype=torch.bfloat16, device=x.device)
    ss = torch.empty((M * 8,), dtype=torch.float32, device=x.device)
    out = torch.empty_like(x)
    _extL(nx1).moe(x, ids, tw, state["w1"], state["w2"], state["g1"], state["g2"], state["nw"], inter, y, ss, state["ctrL"],
                   state["wsL"], out, data["config"]["local_start"], ks if M <= KSL_MAX else 1, state["prof"])
    return out


_modS = {}


def _extS(ksf):
    if not ksf: return _ext()
    if ksf not in _modS:
        wpc, stg = (16, 2) if _big() else (8, 2)
        _modS[ksf] = load_inline("moe_w4a16_v30S_%d_%d_k%d" % (wpc, stg, ksf),
                                 cpp_sources="void moe(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,torch::Tensor);",
                                 cuda_sources=CUDA, functions=["moe"],
                                 extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo", "-DWPC=%d" % wpc, "-DNSTG=%d" % stg, "-DKSF=%d" % ksf] + list(_DBG), verbose=False)
    return _modS[ksf]


def _small(data, state, ksf):
    x = data["hidden_states"].contiguous()
    M = x.shape[0]
    ids = data["topk_ids"].to(torch.int32).contiguous()
    tw = data["topk_weights"].float().contiguous()
    inter = torch.empty((M * 8, 4096), dtype=torch.bfloat16, device=x.device)
    y = torch.empty((M * 8, 4096), dtype=torch.bfloat16, device=x.device)
    ss = torch.empty((M * 8,), dtype=torch.float32, device=x.device)
    out = torch.empty_like(x)
    _extS(ksf).moe(x, ids, tw, state["w1"], state["w2"], state["g1"], state["g2"], state["nw"], inter, y, ss, state["ctr"],
                   state["tcnt"], state["ws"], out, data["config"]["local_start"], state["prof"])
    return out


def _run(cfg, data, state):
    if cfg[0] == "fs":
        return _fs(data, state, cfg)
    return _small(data, state, cfg[1]) if cfg[0] == "v17" else _ring(data, state, cfg[1], cfg[2])


def _default(M):
    return ("v17", 0, 0) if M <= 64 else ("ring", 1, 0)


def _cands(M):
    c = [("v17", k, 0) for k in V17_KSF] if M <= 64 else []
    if M <= FS_MAX: c += [("fs",) + f for f in FS_CFG]
    for nx in (0, 1):
        for ks in ((1, 2, 4, 8) if M <= KSL_MAX else (1,)):
            c.append(("ring", ks, nx))
    return c


def _synth(M, seed, dev):
    """busiest-GPU top-8 routing with heavy-tailed expert popularity (the judge's soft routing shape), random activations"""
    g = torch.Generator(device=dev).manual_seed(seed)
    sc = torch.rand((M, 256), generator=g, device=dev)
    z = torch.randn((256,), generator=g, device=dev)
    u = torch.rand((256,), generator=g, device=dev).clamp(min=1e-12)
    r = (sc + 0.03 * z / (-u.log()).sqrt()).topk(8, dim=-1)
    ids = r.indices
    ids = (ids - 32 * torch.bincount((ids // 32).flatten(), minlength=8).argmax()) % 256
    return {"hidden_states": torch.randn((M, 4096), generator=g, device=dev).to(torch.bfloat16), "topk_ids": ids.to(torch.int32),
            "topk_weights": (r.values / r.values.sum(-1, keepdim=True)).float(), "config": {"local_start": 0}}


def _nm(c):
    if c[0] == "fs":
        x = c[3] if len(c) > 3 else ""
        return f"f{c[1]}s{c[2]}" + ("w8" if "-DWPC=8" in x else "") + ("L" if "-DLASTA" in x else "") + ("P" if "-DMT2" in x else "") + ("m" + x.split("-DSMREQ=")[1].split()[0] if "-DSMREQ=" in x else "")
    return f"{c[0][0]}{c[1]}{'n' if c[2] else ''}"


def _tune(state):
    dev = state["w1"].device
    flush = torch.empty(2 ** 27, device=dev); sink = torch.empty((), device=dev)
    table, lines = {}, []
    for M in TUNE_M:
        res = {}
        for seed in (11, 12):
            data = _synth(M, seed + M, dev)
            for cfg in _cands(M):
                try:
                    sr = torch.cuda.Stream(); sr.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(sr):
                        for _ in range(2): _run(cfg, data, state)
                    torch.cuda.current_stream().wait_stream(sr); torch.cuda.synchronize()
                    gr = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(gr): _run(cfg, data, state)
                    ts = []
                    for _ in range(12):
                        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                        sink.copy_(flush.sum()); e0.record(); gr.replay(); e1.record(); torch.cuda.synchronize()
                        ts.append(e0.elapsed_time(e1) * 1e3)
                    res[cfg] = res.get(cfg, 0.0) + sorted(ts)[len(ts) // 2]
                    del gr
                except Exception as ex:
                    res[cfg] = float("inf"); print("tune", M, cfg, "failed", ex, flush=True)
        d = _default(M)
        best = min(res, key=res.get)
        if res[best] > 0.98 * res.get(d, float("inf")): best = d
        table[M] = best
        lines.append(f"M{M}:" + ",".join(f"{_nm(c)}={res[c] / 2:.0f}" for c in res) + f"->{_nm(best)}")
    print("v36 tune " + " | ".join(lines), flush=True)
    torch.cuda.empty_cache()
    return table


def kernel_fn(data, state):
    M = data["hidden_states"].shape[0]
    if FORCE is not None and (FORCE[0] != "fs" or (M <= FS_MAX and data["config"]["local_start"] == 0)): return _run(FORCE, data, state)
    tb = state.get("tune") or {}
    b = next((m for m in TUNE_M if m >= M), TUNE_M[-1])
    cfg = tb.get(b, _default(M))
    if cfg[0] == "fs" and (M > FS_MAX or data["config"]["local_start"] != 0): cfg = _default(M)
    return _run(cfg, data, state)
