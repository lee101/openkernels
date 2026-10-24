# okc: problem=muse_moe_w4a16
"""One persistent kernel: routing -> W13 (SwiGLU fused) -> W2 (+post-norm stats) -> norm + weighted sum.
Weights repacked per 16-row tile into mma.sync A-fragment order with a bit layout that dequantizes
NVFP4 -> BF16 exactly in 11 integer ops per 8 values; streamed by TMA bulk copies (cp.async fallback)."""
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
// event log: warps gw < 64 keep up to 96 (type, time) pairs after the 65536 x 16 counters
#define PEVC(pp, gww, lanee, evv, ty) do { if ((lanee) == 0 && (gww) < 64 && (evv) < 96) { (pp).prof[65536 * 16 + (gww) * 192 + 2 * (evv)] = (ty); (pp).prof[65536 * 16 + (gww) * 192 + 2 * (evv) + 1] = gtime(); } (evv)++; } while (0)
#define PADD(k, v) do { if (lane == 0) p.prof[gw * 16 + (k)] += (v); } while (0)
#else
#define PSET(k, v)
#define PADD(k, v)
#define PEVC(pp, gww, lanee, evv, ty)
#endif

__device__ __forceinline__ void cp16(uint32_t s, const void* g) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(g)); }
__device__ __forceinline__ void cpc() { asm volatile("cp.async.commit_group;\n"); }
template <int N> __device__ __forceinline__ void cpw() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }
__device__ __forceinline__ uint4 lds(uint32_t a) { uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a)); return v; }
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

#define QI (NLOC + 2)  // ctr[QI]: dynamic tile queue
#define RED (2 * 2 * 32 * 16)  // per-warp split-K partial slots: 2 tile parities x NT <= 2 float4 per lane
struct Sm { int cnt[NLOC], off[NLOC], fill[NLOC], act[NLOC], na, ks, maxc; int ids[16][4]; };

// Work = tiles from one global queue (all W13 tiles, then all W2 tiles, expert-major). A group of ks warps of one CTA
// takes a tile together, each streaming SPT/ks K-stages; partials meet in smem; the finisher rotates per tile.
template <int S>
struct Ctx {
  const P* p; Sm* sm; const uint16_t* list; uint32_t ring, bars; int lane, N1, N, gi, gj, ks;
  int q; unsigned long long wcyc; int ev, gw;
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
    PEVC(*p, gw, lane, ev, 1);
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
    PEVC(*p, gw, lane, ev, 2);
#if !EARLYREL
    issue((q + S - 1) % S);
#endif
    q++;
    return ring + slot * STG + lane * 16;
  }
  __device__ __forceinline__ void release() {  // stage q - 1 consumed: refill its slot now, before any epilogue
    PEVC(*p, gw, lane, ev, 3);
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
    const uint4 sc = lds(slot + 4096);
#pragma unroll
    for (int c = 0; c < 8; c++) {
      const uint4 w = lds(slot + c * 512);
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
        const uint32_t a1 = bmul((rotl(q, 3) & 0x01C001C0u) | (rotl(q, 1) & 0x80008000u), sB);
        const uint32_t a2 = bmul(rotl(q, 6) & 0x81C081C0u, sA);
        const uint32_t a3 = bmul((rotl(q, 11) & 0x01C001C0u) | (rotl(q, 5) & 0x80008000u), sB);
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
  PEVC(p, C.gw, lane, C.ev, 4);
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
    if (C.gj != (ti & (ks - 1))) { PEVC(p, C.gw, lane, C.ev, 6); return false; }
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
  PEVC(p, C.gw, lane, C.ev, 5);
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
    if (c > 0) sm->act[__popc(m & ((1u << lane) - 1))] = lane;
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
  C.ring = sbase + wid * S * STG; C.bars = sbase + WPC * S * STG + WPC * SCR + wid * S * 8; C.q = 0; C.wcyc = 0; C.ev = 0; C.gw = gw;
  C.pi = 0; C.pp = 0; C.ps = 0;
  const bool lead = C.gj == 0 && lane == 0;
  if (lead) { sm->ids[C.gi][0] = atomicAdd(p.ctr + QI, 1); sm->ids[C.gi][1] = atomicAdd(p.ctr + QI, 1); sm->ids[C.gi][2] = atomicAdd(p.ctr + QI, 1); }
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
    if (lead) nxt = atomicAdd(p.ctr + QI, 1);  // tile for i + 3: the result is stored after this tile
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
    __syncwarp();
    const int span = SPT / ks, s0 = C.gj * span;
    bool fin = false;
    for (int ps = 0; ps < np; ps++) {
      int nt = min(NTM, (cnt - ps * 8 * NTM + 7) >> 3);
      fin = g2 ? dispatch<1, S>(C, nt, e, mt, ps, i, s0, s0 + span) : dispatch<0, S>(C, nt, e, mt, ps, i, s0, s0 + span);
    }
    if (fin) {  // this tile's results are visible before its count (release)
      __syncwarp();
      if (lane == 0) asm volatile("red.release.gpu.global.add.s32 [%0], 1;\n" :: "l"(p.ctr + (g2 ? NLOC : e)) : "memory");
    }
    if (lead) sm->ids[C.gi][(i + 3) & 3] = nxt;  // read by the group's producers after the next tile's barrier
    __syncwarp();
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

_mod = None
_DBG = []
# pair p of each 32-bit word: (eem rotation, sign rotation); bf16 targets m->6, e0->7, e1->8, s->15 (+16 for the high value)
_ROT = ((0, 0), (3, 1), (6, 6), (11, 5))


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
        m = load_inline("moe_w4a16_v3_%d_%d%s%s" % (wpc, stg, "_L" if large else "", "".join(_DBG).replace("-", "_").replace("=", "")),
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
    """w uint8 [R, K/2], sf e4m3 [R, K/16] -> uint8 [R/16, K/512, 4608] in mma A-fragment order."""
    R, K = w.shape[0], w.shape[1] * 2
    MT, NS = R // 16, K // 512
    codes = torch.stack((w & 15, w >> 4), -1).view(R, K)[perm].to(torch.int64)
    s8 = sf.view(torch.uint8)[perm].to(torch.int64)
    neg = (s8 >> 7).repeat_interleave(16, 1)  # fold a negative block scale into the codes' signs (same values)
    codes = codes ^ (neg << 3)
    zero = ((s8 & 0x7F) == 0).repeat_interleave(16, 1)  # zero scale: every value is 0; keep it 0 with a normal scale
    codes = torch.where(zero, torch.zeros_like(codes), codes)
    s8 = torch.where((s8 & 0x7F) == 0, torch.full_like(s8, 0x08), s8 & 0x7F)
    if ((s8 >> 3) == 0).any():
        raise ValueError("subnormal E4M3 block scales are not supported by this fast path")
    C = codes.view(MT, 16, NS, 8, 4, 4, 4)  # mt, i(row), s, c, t, j, e
    a, b = C[:, :8], C[:, 8:]  # rows g, g+8
    word = torch.zeros(a.shape[:-1], dtype=torch.int64, device=w.device)
    for pair, (src, e0) in enumerate(((a, 0), (b, 0), (a, 2), (b, 2))):
        word = _place(word, src[..., e0], pair, 0)
        word = _place(word, src[..., e0 + 1], pair, 1)
    word = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32)
    word = word.permute(0, 2, 3, 1, 4, 5).contiguous().view(torch.uint8).view(MT, NS, 4096)
    S = s8.to(torch.uint8).view(MT, 16, NS, 8, 4)
    sc = torch.stack((S[:, :8], S[:, 8:]), -1).permute(0, 2, 1, 4, 3, 5).contiguous().view(MT, NS, 512)
    return torch.cat((word, sc), -1).contiguous()


def prepare(weights):
    E, R1 = weights["w13"].shape[0], weights["w13"].shape[1]
    I = R1 // 2
    dev = weights["w13"].device
    perm1 = torch.stack((torch.arange(I, device=dev).view(-1, 8), I + torch.arange(I, device=dev).view(-1, 8)), 1).reshape(-1)
    perm2 = torch.arange(weights["w2"].shape[1], device=dev)
    w1 = torch.empty((E, R1 // 16, 8, 4608), dtype=torch.uint8, device=dev)
    w2 = torch.empty((E, weights["w2"].shape[1] // 16, 8, 4608), dtype=torch.uint8, device=dev)
    for e in range(E):
        w1[e] = _repack(weights["w13"][e], weights["w13_sf"][e], perm1)
        w2[e] = _repack(weights["w2"][e], weights["w2_sf"][e], perm2)
    _ext(); _ext(LARGE)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    st = {"w1": w1, "w2": w2, "g1": weights["w13_gs"].float() * 256.0, "g2": weights["w2_gs"].float() * 256.0,
            "nw": 1.0 + weights["post_norm_w"].float(), "ctr": torch.zeros(64, dtype=torch.int32, device=dev),
            "tcnt": torch.zeros(32 * 768, dtype=torch.int32, device=dev),
            "prof": torch.zeros((65536 * 16 + 64 * 192 if "-DPROF" in _DBG else 0,), dtype=torch.int64, device=dev),
            "ws": torch.empty((32 * 768 * 8 * 64 * 4,), dtype=torch.float32, device=dev)}
    if DIAG:
        _diag(st, weights)
    return st


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


def kernel_fn(data, state):
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
