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
#define NTM 4
#define SCR 1024
#define KSM 4
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

#define QI (NLOC + 2)  // ctr[QI]: global tile queue
#ifndef NSLOT
#define NSLOT 36       // CTA-shared ring of weight stages
#endif
#ifndef NACC
#define NACC 16        // per-unit smem accumulators (16 rows x NTM*8 tokens fp32)
#endif
#define NCW (WPC - 1)  // consumer warps; warp 0 produces
#ifndef CHK
#define CHK 4          // consecutive stages of one tile per consumer work item
#endif
#define ACCB (NTM * 32 * 16)
struct Hdr { int u, s, k, pass; };
struct Sm { int rel[NSLOT], take[NSLOT]; int cnt[NLOC], off[NLOC], fill[NLOC], act[NLOC], na, maxc, cnext, pseq, pdone; int accn[NACC], accph[NACC]; Hdr hdr[NSLOT]; };

#if TMA
#define FULLCNT 1
#else
#define FULLCNT 32
#endif
__device__ __forceinline__ void mbinit(uint32_t b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(b), "r"(n)); }
__device__ __forceinline__ void mbarrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(b) : "memory"); }
__device__ __forceinline__ bool mbtest(uint32_t b, uint32_t ph) {
  uint32_t r;
  asm volatile("{\n.reg .pred p;\nmbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n" : "=r"(r) : "r"(b), "r"(ph) : "memory");
  return r;
}
__device__ __forceinline__ void mbwait(uint32_t b, uint32_t ph) {
#ifdef TRACE
  long long n = 0;
  while (!mbtest(b, ph)) if (++n == 50000000) printf("STUCK mbwait blk %d w %d bar %u ph %u\n", blockIdx.x, threadIdx.x >> 5, b, ph);
  return;
#endif
#if TMA
  asm volatile("{\n.reg .pred p;\nW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W;\n}\n" :: "r"(b), "r"(ph) : "memory");
#else
  asm volatile("{\n.reg .pred p;\nW: mbarrier.test_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W;\n}\n" :: "r"(b), "r"(ph) : "memory");
#endif
}

// a parity wait is only meaningful within one lap of the ring: first wait for the producer to reach the slot
__device__ __forceinline__ void slotwait(const Sm* sm, uint32_t fullb, int q) {
#ifdef TRACE
  long long n = 0;
  while (*(volatile const int*)&sm->pseq <= q) { if (++n == 50000000) printf("STUCK pseq blk %d w %d q %d pseq %d\n", blockIdx.x, threadIdx.x >> 5, q, *(volatile const int*)&sm->pseq); }
#else
  while (*(volatile const int*)&sm->pseq <= q) {}
#endif
  mbwait(fullb + (q % NSLOT) * 8, (q / NSLOT) & 1);
}

struct K {
  const P* p; Sm* sm; const uint16_t* list; uint32_t ring, fullb, emptyb; float4* acc; int lane, N1, N;
  __device__ __forceinline__ void item(int u, int& g2, int& e, int& mt) const {
    if (u < N1) { g2 = 0; e = sm->act[u >> 9]; mt = u & 511; }
    else { int v = u - N1; g2 = 1; e = sm->act[v >> 8]; mt = v & 255; }
  }
  __device__ __forceinline__ int npass(int e) const { return (sm->cnt[e] + 8 * NTM - 1) / (8 * NTM); }
};

// one 512-K stage of a 16-row tile: dequant from the ring slot, B (activations) from global, partial sums out
template <int NT, int G2>
__device__ __forceinline__ void stage(const K& C, int sq, int e, int pass, int s, float (&f)[NT][4]) {
  const P& p = *C.p;
  const int lane = C.lane, g = lane >> 2, t = lane & 3;
  const int cnt = C.sm->cnt[e], off = C.sm->off[e];
  constexpr int CH = NT == 1 ? 4 : (NT == 2 ? 2 : 1);
  constexpr int D = NT == 1 ? 3 : 1;
  const uint4* rp[NT];
#pragma unroll
  for (int n = 0; n < NT; n++) {
    int i = pass * 8 * NTM + n * 8 + g; int r = C.list[off + (i < cnt ? i : 0)];
    rp[n] = (const uint4*)((G2 ? p.inter + (size_t)r * HD : p.x + (size_t)(r >> 3) * HD) + t * 16) + s * 64;
  }
  uint4 B[D + 1][NT][2];
  auto ldb = [&](uint4 (&b)[NT][2], int c) {  // c: chunk index across the item's stages
#pragma unroll
    for (int n = 0; n < NT; n++) {
      const uint4* q = rp[n] + (c >> 3) * 64 + (c & 7) * 8;
#ifdef BCHK
      if ((c >> 3) + s >= 8 || c < 0) printf("BAD ldb c %d s %d sq %d blk %d\n", c, s, sq, blockIdx.x);
#endif
      if (G2) { b[n][0] = __ldca(q); b[n][1] = __ldca(q + 1); } else { b[n][0] = __ldg(q); b[n][1] = __ldg(q + 1); }
    }
  };
  float acc[NT][CH][4];
#pragma unroll
  for (int n = 0; n < NT; n++)
#pragma unroll
    for (int h = 0; h < CH; h++) acc[n][h][0] = acc[n][h][1] = acc[n][h][2] = acc[n][h][3] = 0.f;
#pragma unroll
  for (int c = 0; c < D; c++) ldb(B[c], c);
  for (int ss = 0; ss < CHK; ss++) {
  const int sl = (sq + ss) % NSLOT;
  slotwait(C.sm, C.fullb, sq + ss);
#ifdef TRACE
  if (lane == 0) atomicAdd(&C.sm->take[sl], 1);
#endif
  const uint32_t slot = C.ring + sl * STG + lane * 16;
  const uint4 sc = lds(slot + 4096);
#pragma unroll
  for (int c = 0; c < 8; c++) {
    const uint4 w = lds(slot + c * 512);
    const uint32_t sw = (c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w));
    const uint32_t sA = sc2(sw, (c & 1) ? 0x4242 : 0x4040), sB = sc2(sw, (c & 1) ? 0x4343 : 0x4141);
    if (c + D < 8) ldb(B[(c + D) % (D + 1)], ss * 8 + c + D);
    else if (ss + 1 < CHK) ldb(B[(c + D) % (D + 1)], ss * 8 + c + D);
    const uint4 (&bc)[NT][2] = B[c % (D + 1)];
    const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const uint32_t q = ww[j];
      const uint32_t a0 = bmul(q & 0x81C081C0u, sA);
      const uint32_t a1 = bmul((rotl(q, 3) & 0x01C001C0u) | (rotl(q, 1) & 0x80008000u), sB);
      const uint32_t a2 = bmul(rotl(q, 6) & 0x81C081C0u, sA);
      const uint32_t a3 = bmul((rotl(q, 11) & 0x01C001C0u) | (rotl(q, 5) & 0x80008000u), sB);
#pragma unroll
      for (int n = 0; n < NT; n++) {
        const uint32_t* bb = (const uint32_t*)&bc[n][j >> 1];
        mma(acc[n][j % CH], a0, a1, a2, a3, bb[(j & 1) * 2], bb[(j & 1) * 2 + 1]);
      }
    }
  }
  __syncwarp();
  if (lane == 0) mbarrive(C.emptyb + sl * 8);  // slot may be refilled
#ifdef TRACE
  if (lane == 0) atomicAdd(&C.sm->rel[sl], 1);
#endif
  }
#pragma unroll
  for (int n = 0; n < NT; n++)
#pragma unroll
    for (int i = 0; i < 4; i++) { float v = 0.f;
#pragma unroll
      for (int h = 0; h < CH; h++) v += acc[n][h][i];
      f[n][i] = v; }
}

// epilogue of a finished (tile, pass): f holds the full-K sums
template <int NT, int G2>
__device__ __forceinline__ void finish(const K& C, int e, int mt, int pass, const float (&f)[NT][4], __nv_bfloat16* scr) {
  const P& p = *C.p;
  const int lane = C.lane, g = lane >> 2, t = lane & 3;
  const int cnt = C.sm->cnt[e], off = C.sm->off[e];
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
}

// consumer: one stage-unit; the warp that lands a unit's 8th stage runs its epilogue
template <int NT, int G2>
__device__ __forceinline__ void consume(const K& C, int sq, const Hdr& h, int e, int mt) {
  float f[NT][4];
  stage<NT, G2>(C, sq, e, h.pass, h.s, f);
  float* a = (float*)(C.acc + h.k * (NTM * 32));
#pragma unroll
  for (int n = 0; n < NT; n++)
#pragma unroll
    for (int i = 0; i < 4; i++) atomicAdd(a + (n * 32 + C.lane) * 4 + i, f[n][i]);
  __threadfence_block(); __syncwarp();
  int old = 0;
  if (C.lane == 0) old = atomicAdd(&C.sm->accn[h.k], 1);
  old = __shfl_sync(0xffffffffu, old, 0);
  if (old != SPT / CHK - 1) return;
  __threadfence_block();
  volatile float* va = a;
#pragma unroll
  for (int n = 0; n < NT; n++)
#pragma unroll
    for (int i = 0; i < 4; i++) f[n][i] = va[(n * 32 + C.lane) * 4 + i];
  __syncwarp();
  finish<NT, G2>(C, e, mt, h.pass, f, (__nv_bfloat16*)a);
  fence_ar(); __syncwarp();
  if (C.lane == 0) { atomicAdd(C.p->ctr + (G2 ? NLOC : e), 1); __threadfence_block(); C.sm->accn[h.k] = 0; ((volatile int*)C.sm->accph)[h.k] += 1; }
}

__global__ void __launch_bounds__(WPC * 32, 1) moe_kernel(P p) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5, R = p.M * 8;
  const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem);
  float4* accs = (float4*)(smem + NSLOT * STG);
  Sm* sm = (Sm*)(smem + NSLOT * STG + NACC * ACCB + NSLOT * 16);
  uint16_t* list = (uint16_t*)(sm + 1);
  uint8_t* lid = (uint8_t*)(list + R);
  const uint32_t fullb = sbase + NSLOT * STG + NACC * ACCB, emptyb = fullb + NSLOT * 8;
  if (tid < NSLOT) { mbinit(fullb + tid * 8, FULLCNT); mbinit(emptyb + tid * 8, 1); }
  if (tid < NLOC) sm->cnt[tid] = 0;
  if (tid < NACC) { sm->accn[tid] = 0; sm->accph[tid] = 0; }
  if (tid == 0) { sm->cnext = 0; sm->pseq = 0; sm->pdone = 0; }
  if (tid < NSLOT) { sm->rel[tid] = 0; sm->take[tid] = 0; }
#if TMA
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#endif
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
    if (lane == 0) sm->na = __popc(m);
  }
  __syncthreads();
  for (int i = tid; i < R; i += blockDim.x) { unsigned e = lid[i]; if (e < NLOC) list[atomicAdd(&sm->fill[e], 1)] = i; }
  __syncthreads();
  K C;
  C.p = &p; C.sm = sm; C.list = list; C.ring = sbase; C.fullb = fullb; C.emptyb = emptyb; C.acc = accs; C.lane = lane;
  const int na = sm->na;
  C.N1 = na * T1E; C.N = C.N1 + na * T2E;
  if (wid == 0) {  // producer: grab tiles, zero their accumulators, stream their stages into the ring
    int seq = 0, units = 0;
    int u = 0, nx = 0;
    if (lane == 0) { u = atomicAdd(p.ctr + QI, 1); nx = atomicAdd(p.ctr + QI, 1); }
    u = __shfl_sync(0xffffffffu, u, 0); nx = __shfl_sync(0xffffffffu, nx, 0);
    for (; u < C.N; ) {
      int g2, e, mt; C.item(u, g2, e, mt);
      const int np = C.npass(e);
      if (!g2 && mt == 0) for (int i = lane; i < sm->cnt[e]; i += 32) p.ss[list[sm->off[e] + i]] = 0.f;
      for (int ps = 0; ps < np; ps++, units++) {
        const int k = units % NACC;
#ifdef TRACE
        if (lane == 0) { long long n = 0; while (((volatile int*)sm->accph)[k] != units / NACC) { if (++n == 50000000) printf("STUCK acc blk %d k %d units %d ph %d\n", blockIdx.x, k, units, sm->accph[k]); } }
#else
        if (lane == 0) while (((volatile int*)sm->accph)[k] != units / NACC) {}
#endif
        __syncwarp();
        float4* a = accs + k * (NTM * 32);
        for (int i = lane; i < NTM * 32; i += 32) a[i] = make_float4(0.f, 0.f, 0.f, 0.f);
        __syncwarp();
        for (int s = 0; s < SPT; s++, seq++) {
          const int slot = seq % NSLOT;
#ifdef TRACE
          if (seq >= NSLOT) { long long n = 0; while (!mbtest(emptyb + slot * 8, ((seq / NSLOT) - 1) & 1)) if (++n == 50000000 && lane == 0) printf("STUCKP blk %d seq %d slot %d rel %d take %d\n", blockIdx.x, seq, slot, sm->rel[slot], sm->take[slot]); }
#else
          if (seq >= NSLOT) mbwait(emptyb + slot * 8, ((seq / NSLOT) - 1) & 1);
#endif
          if (lane == 0) sm->hdr[slot] = Hdr{u, s, k, ps};

          const uint8_t* src = g2 ? p.w2 + ((size_t)((p.lo + e) * T2E + mt) * SPT + s) * STG
                                  : p.w1 + ((size_t)((p.lo + e) * T1E + mt) * SPT + s) * STG;
#if TMA
          if (lane == 0) tma_ld(sbase + slot * STG, src, fullb + slot * 8);
#else
          for (int q = 0; q < 9; q++) cp16(sbase + slot * STG + lane * 16 + q * 512, src + lane * 16 + q * 512);
          asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" :: "r"(fullb + slot * 8) : "memory");
#endif
          if (lane == 0) *(volatile int*)&sm->pseq = seq + 1;
        }
      }
      u = nx;
      if (lane == 0 && nx < C.N) nx = atomicAdd(p.ctr + QI, 1);
      nx = __shfl_sync(0xffffffffu, nx, 0);
    }
    if (lane == 0) { __threadfence_block(); *(volatile int*)&sm->pdone = 1; }
  } else {  // consumers
    int okE = -1;
    for (;;) {
      int sq = 0;
      if (lane == 0) sq = atomicAdd(&sm->cnext, CHK);
      sq = __shfl_sync(0xffffffffu, sq, 0);
      const int slot = sq % NSLOT;
      int stop = 0;
      if (lane == 0) {  // wait until the producer has reached sq, or finished short of it
        for (;;) {
          const int done = *(volatile int*)&sm->pdone;
          if (*(volatile int*)&sm->pseq > sq) break;
          if (done) { stop = 1; break; }
        }
      }
      if (__shfl_sync(0xffffffffu, stop, 0)) break;
      slotwait(sm, fullb, sq);
      volatile Hdr* vh = sm->hdr + slot;
      const Hdr h{vh->u, vh->s, vh->k, vh->pass};
      int g2, e, mt; C.item(h.u, g2, e, mt);

      if (g2 && e != okE) {
#ifdef TRACE
        if (lane == 0) { const int tgt = T1E * C.npass(e); long long n = 0; while (ldacq(p.ctr + e) < tgt) { if (++n == 2000000) printf("STUCK dep blk %d w %d e %d have %d tgt %d\n", blockIdx.x, wid, e, p.ctr[e], tgt); } }
#else
        if (lane == 0) { const int tgt = T1E * C.npass(e); while (ldacq(p.ctr + e) < tgt) __nanosleep(64); }
#endif
        __syncwarp(); okE = e;
      }
      const int nt = min(NTM, (sm->cnt[e] - h.pass * 8 * NTM + 7) >> 3);
      if (g2) { if (nt <= 1) consume<1, 1>(C, sq, h, e, mt); else if (nt <= 2) consume<2, 1>(C, sq, h, e, mt); else consume<4, 1>(C, sq, h, e, mt); }
      else { if (nt <= 1) consume<1, 0>(C, sq, h, e, mt); else if (nt <= 2) consume<2, 0>(C, sq, h, e, mt); else consume<4, 0>(C, sq, h, e, mt); }
    }
  }
  const int gw = blockIdx.x * WPC + wid;
  int T2 = 0;
  for (int a = 0; a < na; a++) T2 += T2E * C.npass(sm->act[a]);
  if (lane == 0) while (ldacq(p.ctr + NLOC) < T2) __nanosleep(32);
  __syncwarp();
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
static size_t smem_bytes(int M) { return (size_t)NSLOT * STG + NACC * ACCB + NSLOT * 16 + sizeof(Sm) + (size_t)M * 8 * 3; }

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


def _ext():
    global _mod
    if _mod is None:
        big = _big()
        wpc, stg = (16, 36) if big else (8, 8)
        for d in _DBG:
            if d.startswith("-DCFG="): wpc, stg = map(int, d[6:].split("x"))
        _mod = load_inline("moe_w4a16_v6_%d_%d%s" % (wpc, stg, "".join(_DBG).replace("-", "_").replace("=", "")),
                           cpp_sources="void moe(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,torch::Tensor);",
                           cuda_sources=CUDA, functions=["moe"],
                           extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo", "-DWPC=%d" % wpc] + ([] if any(d.startswith("-DNSLOT") for d in _DBG) else ["-DNSLOT=%d" % stg, "-DNACC=%d" % (16 if big else 8)]) + [d for d in _DBG if not d.startswith("-DCFG=")], verbose=False)
    return _mod


def _place(word, code, pair, hi):
    de, ds = _ROT[pair]
    for bit, tgt, d in ((0, 6, de), (1, 7, de), (2, 8, de), (3, 15, ds)):
        word |= ((code >> bit) & 1) << ((tgt + 16 * hi - d) % 32)
    return word


def _repack(w, sf, perm, chunk=64):
    """chunked over 16-row tiles to bound temporary memory"""
    R = w.shape[0]
    return torch.cat([_repack1(w, sf, perm[i:i + chunk * 16]) for i in range(0, R, chunk * 16)], 0)


def _repack1(w, sf, perm):
    """w uint8 [R, K/2], sf e4m3 [R, K/16] -> uint8 [R/16, K/512, 4608] in mma A-fragment order."""
    R, K = perm.numel(), w.shape[1] * 2
    MT, NS = R // 16, K // 512
    wp = w[perm]
    codes = torch.stack((wp & 15, wp >> 4), -1).view(R, K).to(torch.int64)
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
    _ext()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    st = {"w1": w1, "w2": w2, "g1": weights["w13_gs"].float() * 256.0, "g2": weights["w2_gs"].float() * 256.0,
            "nw": 1.0 + weights["post_norm_w"].float(), "ctr": torch.zeros(64, dtype=torch.int32, device=dev),
            "tcnt": torch.zeros(32 * 768, dtype=torch.int32, device=dev),
            "prof": torch.zeros((65536 * 16 if "-DPROF" in _DBG else 0,), dtype=torch.int64, device=dev),
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
    _ext().moe(x, ids, tw, state["w1"], state["w2"], state["g1"], state["g2"], state["nw"], inter, y, ss, state["ctr"],
               state["tcnt"], state["ws"], out, data["config"]["local_start"], state["prof"])
    return out
