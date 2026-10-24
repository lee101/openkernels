// extracted from muse_moe_w4a16 (CUDA); edit then `python3 tools/cu.py embed muse_moe_w4a16`
// ---- cu.py: end of standalone prelude ----
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
#ifndef LA3
#define LA3 0
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
  int M, lo; unsigned long long* prof; __nv_bfloat16* xp;
};
#ifdef PROF
__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
#define PSET(k, v) do { if (lane == 0) p.prof[gw * 16 + (k)] = (v); } while (0)
// event log: warps gw < 64 keep up to 96 (type, time) pairs after the 65536 x 16 counters
#define PEVC(pp, gww, lanee, evv, ty) do { if ((lanee) == 0 && (gww) < 64 && (evv) < 96) { (pp).prof[65536 * 16 + (gww) * 192 + 2 * (evv)] = (ty); (pp).prof[65536 * 16 + (gww) * 192 + 2 * (evv) + 1] = gtime(); } (evv)++; } while (0)
#ifdef EVONLY
#define PADD(k, v)
#else
#define PADD(k, v) do { if (lane == 0) p.prof[gw * 16 + (k)] += (v); } while (0)
#endif
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
  // tile of the group's i-th item; slots carry (i & 2047) << 20 so a reader never takes a stale tile
  __device__ __forceinline__ int tid_of(int i) const {
    int v;
    do { v = ((volatile int*)sm->ids[gi])[i & 3]; } while ((v >> 20) != (i & 2047));
    return v & 0xFFFFF;
  }
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
  PEVC(p, C.gw, lane, C.ev, 9);
  const int cnt = C.sm->cnt[e], off = C.sm->off[e];
  constexpr int CH = NT == 1 ? 4 : (NT == 2 ? 2 : 1);
#ifndef BD1
#define BD1 7
#endif
  constexpr int D = NT == 1 ? BD1 : (NT == 2 ? 3 : 1);  // B prefetch distance in chunks; (D + 1) divides 8 so slots stay static
  const uint4* rp[NT];
#pragma unroll
  for (int n = 0; n < NT; n++) {
    int i = pass * 8 * NTM + n * 8 + g;
#ifdef NOBLD
    i = 0;  // ablation: every n-tile reads one token row (L2 activation traffic ~ 1 row per chunk)
#endif
    int r = C.list[off + (i < cnt ? i : 0)];
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
  // the first 3 tiles of each group are static (no contended start-up atomics); the queue hands out the rest
  const int NG = gridDim.x * (WPC / ks), ggi = blockIdx.x * (WPC / ks) + C.gi;
  if (lead) { sm->ids[C.gi][0] = ggi; sm->ids[C.gi][1] = (NG + ggi) | (1 << 20); sm->ids[C.gi][2] = -1; sm->ids[C.gi][3] = -1; }
  __syncthreads();
#ifdef PROF
  PSET(1, gtime());
#endif
  PEVC(p, gw, lane, C.ev, 7);
#pragma unroll
  for (int i = 0; i < S - 1 + EARLYREL; i++) C.issue(i);
  PEVC(p, gw, lane, C.ev, 8);
  for (int i = 0;; i++) {
    PEVC(p, gw, lane, C.ev, 10);
    const int u = C.tid_of(i);
    if (u >= C.N) break;
    int nxt = 0;
    // tile for i + 3 (minus 3 NG), consumed only after this tile. Plain asm: atomicAdd gets warp-aggregated into an
    // atomic + shfl, and the shfl would wait for the (contended) atomic right here.
    if (lead) asm volatile("atom.global.inc.u32 %0, [%1], 2147483647;\n" : "=r"(nxt) : "l"(p.ctr + QI) : "memory");

    int g2, e, mt; C.item(u, g2, e, mt);
    const int cnt = sm->cnt[e], np = (cnt + 8 * NTM - 1) / (8 * NTM);
    PEVC(p, gw, lane, C.ev, 11);
#ifdef PROF
    long long c0 = clock64();
#endif
    if (g2) {
      if (lane < 4) {  // this warp's K range needs only the G1 tiles of its inter-column quarters (128 tiles each)
        const int sp2 = SPT / ks, q0 = (C.gj * sp2) >> 1, q1 = (C.gj * sp2 + sp2 - 1) >> 1;
        if (lane >= q0 && lane <= q1) while (ldacq(p.ctr + 64 + e * 4 + lane) < T1E / 4) __nanosleep(32);
      }
      __syncwarp();
#ifdef PROF
      PADD(4, clock64() - c0); c0 = clock64();
#endif
    } else if (mt == 0 && C.gj == 0) {
      for (int k = lane; k < cnt; k += 32) p.ss[list[sm->off[e] + k]] = 0.f;
    }
    __syncwarp();
    const int span = SPT / ks, s0 = C.gj * span;
    PEVC(p, gw, lane, C.ev, 12);
    bool fin = false;
    for (int ps = 0; ps < np; ps++) {
      int nt = min(NTM, (cnt - ps * 8 * NTM + 7) >> 3);
      fin = g2 ? dispatch<1, S>(C, nt, e, mt, ps, i, s0, s0 + span) : dispatch<0, S>(C, nt, e, mt, ps, i, s0, s0 + span);
    }
    if (fin) {  // this tile's results are visible before its count (release)
      __syncwarp();
      if (lane == 0) asm volatile("red.release.gpu.global.add.s32 [%0], 1;\n" :: "l"(p.ctr + (g2 ? NLOC : 64 + e * 4 + mt / (T1E / 4))) : "memory");
    }
    // tile for item i + 2, published after this tile: the atomic's latency hides behind it; readers check the tag
    if (lead) ((volatile int*)sm->ids[C.gi])[(i + 2) & 3] = (nxt + 2 * NG) | (((i + 2) & 2047) << 20);
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
      for (int i = 64; i < 64 + NLOC * 4; i++) p.ctr[i] = 0;
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

// ---------------------------------------------------------------- large M: CTA-wide activation tiles, wgmma (Hopper)
// Item = (expert, LW-tile row block, token pass of <= LN tokens): warp w streams tile rb * LW + w through its own LS_-deep
// TMA ring (the packed 4608 B stages above); the CTA stages the pass's activations once per 512-k stage into a
// 2-deep SW128 ring that all warps read. Each warp dequantizes its 16 rows into registers (A of wgmma m64nNk16 RS,
// one warpgroup = 4 tiles), so weights are dequantized once and activations leave L2 once per LW tiles.
// The packed weights use a K order permuted within each 64-k chunk; the activations are stored in that same order
// (x by a pre-pass into xp, inter by the G1 epilogue), so B tiles are plain 16 B copies.
#ifndef LW
#define LW 8
#endif
#ifndef LN
#define LN 48
#endif
#ifndef LS_
#define LS_ 3
#endif
#define LXA (((LN + 7) / 8) * 1024)
#define LXS (8 * LXA)
#define XPC 200
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
#define HAS_WG 1
#else
#define HAS_WG 0
#endif
struct LSm { int cnt[NLOC], off[NLOC], fill[NLOC], act[NLOC], np[NLOC], pf[NLOC + 1]; int na, n1, n; int it[4], flag[2];
  int dg2[4], de[4], dps[4], drb[4]; };  // item slot k & 3 decoded once (by its publisher)

__device__ __forceinline__ void ldec(const LSm* s, int u, int& g2, int& e, int& ps, int& rb) {
  int v = u; g2 = v >= s->n1; if (g2) v -= s->n1;
  const int bpe = (g2 ? T2E : T1E) / LW;
  int ei = 0; while (s->pf[ei + 1] * bpe <= v) ei++;
  v -= s->pf[ei] * bpe; ps = v / bpe; rb = v - ps * bpe; e = s->act[ei];
}
__device__ __forceinline__ void lpub(LSm* s, int k, int u) {
  s->it[k & 3] = u;
  if (u < s->n) { int g2, e, ps, rb; ldec(s, u, g2, e, ps, rb); s->dg2[k & 3] = g2; s->de[k & 3] = e; s->dps[k & 3] = ps; s->drb[k & 3] = rb; }
}
__device__ __forceinline__ uint64_t gdesc(uint32_t saddr) {  // K-major SW128 tile, 8-row atoms 1 KB apart
  return (uint64_t)((saddr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | (1ull << 62);
}
#define LD8(d) "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
#define LD16(d) LD8(d), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
#define LD24(d) LD16(d), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23])
template <int N>
__device__ __forceinline__ void wl(float* d, const uint32_t* a, uint64_t db) {
#if HAS_WG
  if constexpr (N == 16)
    asm volatile("wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, %12, 1, 1, 1, 0;\n"
                 : LD8(d) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db));
  else if constexpr (N == 32)
    asm volatile("wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19}, %20, 1, 1, 1, 0;\n"
                 : LD16(d) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db));
  else
    asm volatile("wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23}, {%24,%25,%26,%27}, %28, 1, 1, 1, 0;\n"
                 : LD24(d) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db));
#endif
}
__device__ __forceinline__ void lwg_fence() {
#if HAS_WG
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void lwg_commit() {
#if HAS_WG
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
#endif
}
template <int K> __device__ __forceinline__ void lwg_wait() {
#if HAS_WG
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(K) : "memory");
#endif
}
__device__ __forceinline__ void lmb_init(uint32_t b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(b), "r"(n)); }
__device__ __forceinline__ void lmb_wait(uint32_t b, uint32_t ph) {
#if TMA
  asm volatile("{\n.reg .pred p;\nW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W;\n}\n" :: "r"(b), "r"(ph) : "memory");
#else
  asm volatile("{\n.reg .pred p;\nW: mbarrier.test_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W;\n}\n" :: "r"(b), "r"(ph) : "memory");
#endif
}
__device__ __forceinline__ void fence_async_l() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
__device__ __forceinline__ uint32_t lds32(uint32_t a) { uint32_t v; asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(v) : "r"(a)); return v; }

// one 512-k stage of a 16-row tile against the staged activations (N columns)
template <int N>
__device__ __forceinline__ void lstage(float* acc, uint32_t ws, uint32_t xs, int lane) {
  ws += lane * 16;
  const uint4 sc = lds(ws + 4096);
  uint32_t A[2][4][4];
#if !HAS_WG
  const int g = lane >> 2, t = lane & 3;
#endif
#pragma unroll
  for (int c = 0; c < 8; c++) {
    const uint4 w = lds(ws + c * 512);
    const uint32_t sw = (c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w));
    const uint32_t sA = sc2(sw, (c & 1) ? 0x4242 : 0x4040), sB = sc2(sw, (c & 1) ? 0x4343 : 0x4141);
    const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const uint32_t q = ww[j];
      A[c & 1][j][0] = bmul(q & 0x81C081C0u, sA);
      A[c & 1][j][1] = bmul((rotl(q, 3) & 0x01C001C0u) | (rotl(q, 1) & 0x80008000u), sB);
      A[c & 1][j][2] = bmul(rotl(q, 6) & 0x81C081C0u, sA);
      A[c & 1][j][3] = bmul((rotl(q, 11) & 0x01C001C0u) | (rotl(q, 5) & 0x80008000u), sB);
    }
#if HAS_WG
    lwg_fence();
#pragma unroll
    for (int j = 0; j < 4; j++) wl<N>(acc, A[c & 1][j], gdesc(xs + c * LXA + 32 * j));
    lwg_commit();
    lwg_wait<1>();  // chunk c - 1 retired: its A registers are reused by chunk c + 1
#else
#pragma unroll
    for (int j = 0; j < 4; j++)
#pragma unroll
      for (int nb = 0; nb < N / 8; nb++) {
        const int n = nb * 8 + g;
        const uint32_t rb = xs + c * LXA + (n >> 3) * 1024 + (n & 7) * 128 + 4 * t;
        mma(acc + 4 * nb, A[c & 1][j][0], A[c & 1][j][1], A[c & 1][j][2], A[c & 1][j][3],
            lds32(rb + (((2 * j) ^ (n & 7)) << 4)), lds32(rb + (((2 * j + 1) ^ (n & 7)) << 4)));
      }
#endif
  }
  lwg_wait<0>();
}

// within-chunk K permutation of the packed weights: true index t*16 + jh*8 + jl*4 + hi*2 + lo sits at logical
// (2 jh + jl)*16 + hi*8 + 2t + lo. Pair pi = jl*2 + hi of 16 B piece pc (t = (pc&7)>>1, jh = pc&1) -> logical offset:
__device__ __forceinline__ int lperm(int pc, int pi) { return (2 * (pc & 1) + (pi >> 1)) * 16 + (pi & 1) * 8 + 2 * ((pc & 7) >> 1); }

struct LC {
  const P* p; LSm* sm; const uint16_t* list; uint8_t* sg; uint32_t sb, wring, xring, bars; int tid, lane, wid, pend; bool xpok;
  // G2 activations depend on G1 quarters; a prefetch never blocks on them (the CTA's own previous block may be one of
  // them, published only at its next item): tid 0 checks without waiting, and a stage that was not ready is loaded at
  // its own top instead, after a blocking wait
  __device__ __forceinline__ bool dep(int qq, bool block) const {
    const int k = (qq >> 3) & 3, s = qq & 7;
    if (sm->it[k] >= sm->n || !sm->dg2[k]) return true;
    const int e = sm->de[k];
    const int* c = p->ctr + 64 + e * 4 + (s >> 1); const int need = (T1E / 4 / LW) * sm->np[e];
    if (!block) return ldacq(c) >= need;
    while (ldacq(c) < need) __nanosleep(64);
    return true;
  }
  __device__ __forceinline__ void issue_w(int qq) {
    const int k = (qq >> 3) & 3, s = qq & 7;
    if (sm->it[k] >= sm->n) return;
    const int g2 = sm->dg2[k], e = sm->de[k], mt = sm->drb[k] * LW + wid;
    const uint8_t* src = g2 ? p->w2 + ((size_t)((p->lo + e) * T2E + mt) * SPT + s) * STG : p->w1 + ((size_t)((p->lo + e) * T1E + mt) * SPT + s) * STG;
    const uint32_t dst = wring + (qq % LS_) * STG, bar = bars + (wid * LS_ + qq % LS_) * 8;
#if TMA
    if (lane == 0) tma_ld(dst, src, bar);
#else
#pragma unroll
    for (int k = 0; k < 9; k++) cp16(dst + k * 512 + lane * 16, src + k * 512 + lane * 16);
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" :: "r"(bar) : "memory");
#endif
  }
  __device__ __forceinline__ void issue_x(int qq) {
    const int k = (qq >> 3) & 3, s = qq & 7;
    if (sm->it[k] < sm->n) {
      const int g2 = sm->dg2[k], e = sm->de[k], ps = sm->dps[k];
      const int nt = min(LN, sm->cnt[e] - ps * LN), off = sm->off[e] + ps * LN;
      if (!g2 && !xpok) { if (lane == 0) while (ldacq(p->ctr + XPC) < (int)gridDim.x) __nanosleep(64); __syncwarp(); xpok = true; }
      const uint32_t xs = xring + (qq & 1) * LXS;
      for (int i = tid; i < nt * 64; i += LW * 32) {
        const int n = i >> 6, c = (i >> 3) & 7, uu = i & 7, r = list[off + n];
        const __nv_bfloat16* src = g2 ? p->inter + (size_t)r * HD : p->xp + (size_t)(r >> 3) * HD;
        cp16(xs + c * LXA + (n >> 3) * 1024 + (n & 7) * 128 + ((uu ^ (n & 7)) << 4), src + s * 512 + c * 64 + uu * 8);
      }
    }
    cpc();
  }
};

template <int N>
__device__ __forceinline__ int litem(LC& C, int q, int g2, int e, int ps, int rb) {
  const P& p = *C.p;
  const int lane = C.lane, g = lane >> 2, t = lane & 3, wid = C.wid, mt = rb * LW + wid;
  const int cnt = C.sm->cnt[e], nt = min(LN, cnt - ps * LN), off = C.sm->off[e] + ps * LN;
  float acc[N / 2];
#pragma unroll
  for (int i = 0; i < N / 2; i++) acc[i] = 0.f;
  int rel = -1;
  for (int s = 0; s < SPT; s++, q++) {
    if (C.pend == q) {  // CTA-uniform: x(q) was not ready at prefetch time
      if (C.tid == 0) C.dep(q, true);
      __syncthreads();
      C.issue_x(q);
      C.pend = -1;
    }
    cpw<0>();
#if TMA
    fence_async_l();
#endif
    if (C.tid == 0) C.sm->flag[q & 1] = C.dep(q + 1, false);
    __syncthreads();  // x(q) visible to all warps; every warp is done with x(q - 1) and weight slot (q - 1) % LS_ (refilled with stage q + LS_ - 1)
    if (C.sm->flag[q & 1]) C.issue_x(q + 1); else { C.pend = q + 1; cpc(); }
    C.issue_w(q + LS_ - 1);
    lmb_wait(C.bars + (wid * LS_ + q % LS_) * 8, (q / LS_) & 1);
    lstage<N>(acc, C.wring + (q % LS_) * STG, C.xring + (q & 1) * LXS, lane);
  }
  // epilogue: scratch in this warp's weight slot of the last stage (refilled only at the next stage's top)
  __nv_bfloat16* scr = (__nv_bfloat16*)(C.sg + (C.wring + ((q - 1) % LS_) * STG - C.sb));
  if (!g2) {
    const float gs = p.g1[p.lo + e];
#pragma unroll
    for (int nb = 0; nb < N / 8; nb++)
#pragma unroll
      for (int h = 0; h < 2; h++) {
        const float gt = acc[4 * nb + h] * gs, up = acc[4 * nb + 2 + h] * gs;
        scr[(nb * 8 + 2 * t + h) * 8 + g] = __float2bfloat16(gt / (1.f + __expf(-gt)) * up);
      }
    __syncwarp();
    const int c = mt >> 3, pc = mt & 7;
    for (int i = lane; i < nt * 4; i += 32) {
      const int tok = i >> 2, pi = i & 3, r = C.list[off + tok];
      *(uint32_t*)(p.inter + (size_t)r * HD + c * 64 + lperm(pc, pi)) = *(const uint32_t*)(scr + tok * 8 + 2 * pi);
    }
    if (ps == 0 && rb == 0 && wid == 0) for (int k = lane; k < cnt; k += 32) p.ss[C.list[C.sm->off[e] + k]] = 0.f;
    rel = 64 + e * 4 + rb / (T1E / 4 / LW);
  } else {
    const float gs = p.g2[p.lo + e];
    const float n0 = p.nw[mt * 16 + g] * gs, n1 = p.nw[mt * 16 + g + 8] * gs;
#pragma unroll
    for (int nb = 0; nb < N / 8; nb++) {
      const float v0 = acc[4 * nb] * n0, v1 = acc[4 * nb + 1] * n0, v2 = acc[4 * nb + 2] * n1, v3 = acc[4 * nb + 3] * n1;
      scr[(nb * 8 + 2 * t) * 16 + g] = __float2bfloat16(v0);
      scr[(nb * 8 + 2 * t + 1) * 16 + g] = __float2bfloat16(v1);
      scr[(nb * 8 + 2 * t) * 16 + g + 8] = __float2bfloat16(v2);
      scr[(nb * 8 + 2 * t + 1) * 16 + g + 8] = __float2bfloat16(v3);
      float sa = v0 * v0 + v2 * v2, sb = v1 * v1 + v3 * v3;
#pragma unroll
      for (int o = 4; o < 32; o <<= 1) { sa += __shfl_xor_sync(0xffffffffu, sa, o); sb += __shfl_xor_sync(0xffffffffu, sb, o); }
      if (g == 0) {
        const int i0 = nb * 8 + 2 * t;
        if (i0 < nt) atomicAdd(p.ss + C.list[off + i0], sa);
        if (i0 + 1 < nt) atomicAdd(p.ss + C.list[off + i0 + 1], sb);
      }
    }
    __syncwarp();
    for (int i = lane; i < nt * 2; i += 32) {
      const int r = C.list[off + (i >> 1)];
      *(uint4*)(p.y + (size_t)r * HD + mt * 16 + (i & 1) * 8) = *(const uint4*)(scr + i * 8);
    }
    rel = NLOC;
  }
  __threadfence();
  return rel;
}

__global__ void __launch_bounds__(LW * 32, 1) moe_large(P p) {
  extern __shared__ __align__(1024) uint8_t smem[];
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5, R = p.M * 8;
  const uint32_t s0 = (uint32_t)__cvta_generic_to_shared(smem), sb = (s0 + 1023) & ~1023u;
  uint8_t* sg = smem + (sb - s0);
  const uint32_t wring = sb + wid * LS_ * STG, xring = sb + LW * LS_ * STG, bars = xring + 2 * LXS;
  LSm* sm = (LSm*)(sg + LW * LS_ * STG + 2 * LXS + LW * LS_ * 8);
  uint16_t* list = (uint16_t*)(sm + 1);
  uint8_t* lid = (uint8_t*)(list + R);
  if (lane < LS_) lmb_init(bars + (wid * LS_ + lane) * 8, TMA ? 1 : 32);
#if TMA
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#endif
  // activations into the packed K order (each CTA a few rows); published with ctr[XPC]
  for (int i = blockIdx.x * blockDim.x + tid; i < p.M * 512; i += gridDim.x * blockDim.x) {
    const int tok = i >> 9, pc = i & 511;
    const uint4 v = *(const uint4*)(p.x + (size_t)tok * HD + pc * 8);
    __nv_bfloat16* d = p.xp + (size_t)tok * HD + (pc >> 3) * 64;
    *(uint32_t*)(d + lperm(pc, 0)) = v.x; *(uint32_t*)(d + lperm(pc, 1)) = v.y;
    *(uint32_t*)(d + lperm(pc, 2)) = v.z; *(uint32_t*)(d + lperm(pc, 3)) = v.w;
  }
  __threadfence();
  if (tid < NLOC) sm->cnt[tid] = 0;
  __syncthreads();
  if (tid == 0) asm volatile("red.release.gpu.global.add.s32 [%0], 1;\n" :: "l"(p.ctr + XPC) : "memory");
  for (int i = tid; i < R; i += blockDim.x) { unsigned e = p.ids[i] - p.lo; lid[i] = e < NLOC ? e : 0xff; if (e < NLOC) atomicAdd(&sm->cnt[e], 1); }
  __syncthreads();
  if (wid == 0) {
    const int c = sm->cnt[lane], npl = (c + LN - 1) / LN;
    int v = c, w = npl;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int n = __shfl_up_sync(0xffffffffu, v, o), m = __shfl_up_sync(0xffffffffu, w, o);
      if (lane >= o) { v += n; w += m; }
    }
    sm->off[lane] = v - c; sm->fill[lane] = v - c; sm->np[lane] = npl;
    const unsigned msk = __ballot_sync(0xffffffffu, c > 0);
    if (c > 0) { const int ei = __popc(msk & ((1u << lane) - 1)); sm->act[ei] = lane; sm->pf[ei + 1] = w; }
    if (lane == 0) { sm->pf[0] = 0; sm->na = __popc(msk); }
    const int tp = __shfl_sync(0xffffffffu, w, 31);
    if (lane == 0) { sm->n1 = tp * (T1E / LW); sm->n = tp * ((T1E + T2E) / LW); }
  }
  __syncthreads();
  if (tid == 0) { lpub(sm, 0, blockIdx.x); lpub(sm, 1, gridDim.x + blockIdx.x); }
  if (wid == 0) {  // stable order (by route): every CTA must split an expert's tokens into the same passes
    for (int i0 = 0; i0 < R; i0 += 32) {
      const int i = i0 + lane;
      const unsigned e = i < R ? lid[i] : 0xffu;
      const unsigned peers = __match_any_sync(0xffffffffu, e);
      const int rank = __popc(peers & ((1u << lane) - 1));
      const int base = e < NLOC ? sm->fill[e] : 0;
      __syncwarp();
      if (e < NLOC) { list[base + rank] = i; if (rank == 0) sm->fill[e] = base + __popc(peers); }
      __syncwarp();
    }
  }
  __syncthreads();
  LC C{&p, sm, list, sg, sb, wring, xring, bars, tid, lane, wid, 0, false};  // pend = 0: x(0) loads at stage 0's top
  for (int i = 0; i < LS_ - 1; i++) C.issue_w(i);
  int q = 0, nxt = 0, rel = -1;
  for (int k = 0;; k++) {
    if (sm->it[k & 3] >= sm->n) break;
    if (tid == 0) asm volatile("atom.global.inc.u32 %0, [%1], 2147483647;\n" : "=r"(nxt) : "l"(p.ctr + QI) : "memory");
    const int g2 = sm->dg2[k & 3], e = sm->de[k & 3], ps = sm->dps[k & 3], rb = sm->drb[k & 3];
    const int nt = min(LN, sm->cnt[e] - ps * LN);
    if (rel >= 0) {  // the previous item's results (every warp's epilogue) are complete
      __syncthreads();
      if (tid == 0) asm volatile("red.release.gpu.global.add.s32 [%0], 1;\n" :: "l"(p.ctr + rel) : "memory");
    }
    if (nt <= 16) rel = litem<16>(C, q, g2, e, ps, rb);
    else if (nt <= 32) rel = litem<32>(C, q, g2, e, ps, rb);
    else rel = litem<LN>(C, q, g2, e, ps, rb);
    q += SPT;
    if (tid == 0) lpub(sm, k + 2, nxt + 2 * gridDim.x);
  }
  cpw<0>();
  __syncthreads();
  if (tid == 0 && rel >= 0) asm volatile("red.release.gpu.global.add.s32 [%0], 1;\n" :: "l"(p.ctr + rel) : "memory");
  const int T2 = sm->n - sm->n1;
  if (lane == 0) while (ldacq(p.ctr + NLOC) < T2) __nanosleep(32);
  __syncwarp();
  const int gw = blockIdx.x * LW + wid;
  for (int it = gw; it < p.M * 16; it += gridDim.x * LW) {
    const int t = it >> 4, col = (it & 15) * 256 + lane * 8;
    float a[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int k = 0; k < 8; k++) {
      const int r = t * 8 + k;
      const unsigned e = p.ids[r] - p.lo;
      if (e < NLOC) {
        const float cf = p.tw[r] * rsqrtf(__ldcg(p.ss + r) * (1.f / HD) + 1e-8f);
        const uint4 v = __ldcg((const uint4*)(p.y + (size_t)r * HD + col));
        const __nv_bfloat162* h = (const __nv_bfloat162*)&v;
#pragma unroll
        for (int j = 0; j < 4; j++) { const float2 f = __bfloat1622float2(h[j]); a[2 * j] += cf * f.x; a[2 * j + 1] += cf * f.y; }
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
      for (int i = 64; i < 64 + NLOC * 4; i++) p.ctr[i] = 0;
      p.ctr[XPC] = 0;
      __threadfence();
    }
  }
}

static size_t lsmem(int M) { return 1024 + (size_t)LW * LS_ * STG + 2 * LXS + LW * LS_ * 8 + sizeof(LSm) + (size_t)M * 8 * 3; }
void moel(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor w1, torch::Tensor w2, torch::Tensor g1,
          torch::Tensor g2, torch::Tensor nw, torch::Tensor inter, torch::Tensor y, torch::Tensor ss, torch::Tensor ctr,
          torch::Tensor xp, torch::Tensor out, int64_t lo) {
  const int M = x.size(0);
  static int grid = 0;
  if (!grid) {
    int dev; cudaGetDevice(&dev); int mx; cudaDeviceGetAttribute(&mx, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaDeviceGetAttribute(&grid, cudaDevAttrMultiProcessorCount, dev);
    cudaFuncSetAttribute(moe_large, cudaFuncAttributeMaxDynamicSharedMemorySize, mx);
  }
  P p{w1.data_ptr<uint8_t>(), w2.data_ptr<uint8_t>(), g1.data_ptr<float>(), g2.data_ptr<float>(),
      (const __nv_bfloat16*)x.data_ptr(), ids.data_ptr<int>(), tw.data_ptr<float>(), nw.data_ptr<float>(),
      (__nv_bfloat16*)inter.data_ptr(), (__nv_bfloat16*)y.data_ptr(), ss.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr(),
      ctr.data_ptr<int>(), nullptr, nullptr, M, (int)lo, nullptr, (__nv_bfloat16*)xp.data_ptr()};
  moe_large<<<grid, LW * 32, lsmem(M), at::cuda::getCurrentCUDAStream()>>>(p);
}
