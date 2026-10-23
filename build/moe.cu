
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>

#define HD 4096
#define NLOC 32
#define WPC 8
#define STG 4608
#define SPT 8
#define T1E 512
#define T2E 256
#define NTM 8
#define SCR 2048

struct P {
  const uint8_t* w1; const uint8_t* w2; const float* g1; const float* g2;
  const __nv_bfloat16* x; const int* ids; const float* tw; const float* nw;
  __nv_bfloat16* inter; __nv_bfloat16* y; float* ss; __nv_bfloat16* out; int* ctr;
  int M, lo;
};

__device__ __forceinline__ void cp16(uint32_t s, const void* g) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(g)); }
__device__ __forceinline__ void cpc() { asm volatile("cp.async.commit_group;\n"); }
template <int N> __device__ __forceinline__ void cpw() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }
__device__ __forceinline__ uint4 lds(uint32_t a) { uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a)); return v; }
__device__ __forceinline__ uint32_t bmul(uint32_t a, uint32_t b) { uint32_t d; asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(0x80008000u)); return d; }
__device__ __forceinline__ void mma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ int ldacq(const int* p) { int v; asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p)); return v; }
__device__ __forceinline__ uint4 ldnc(const void* p) { return __ldg((const uint4*)p); }
__device__ __forceinline__ uint4 ldcg(const void* p) { return __ldcg((const uint4*)p); }

struct Sm { int cnt[NLOC], off[NLOC], fill[NLOC], act[NLOC], na; };

struct It {
  int u, p, s;
};

struct Ctx {
  const P* p; Sm* sm; int* list; uint32_t ring; int lane, W, T1, T;
  int q;      // consumed stages
  It pr;      // producer position
  __device__ __forceinline__ void job(int u, int& g2, int& e, int& mt) const {
    if (u < T1) { g2 = 0; e = sm->act[u >> 9]; mt = u & 511; }
    else { int v = u - T1; g2 = 1; e = sm->act[v >> 8]; mt = v & 255; }
  }
  __device__ __forceinline__ void issue(uint32_t slot) {
    if (pr.u < T) {
      int g2, e, mt; job(pr.u, g2, e, mt);
      const uint8_t* src = g2 ? p->w2 + ((size_t)((p->lo + e) * T2E + mt) * SPT + pr.s) * STG
                              : p->w1 + ((size_t)((p->lo + e) * T1E + mt) * SPT + pr.s) * STG;
      src += lane * 16; uint32_t d = slot + lane * 16;
#pragma unroll
      for (int i = 0; i < 9; i++) cp16(d + i * 512, src + i * 512);
      if (++pr.s == SPT) { pr.s = 0; int np = (sm->cnt[e] + 63) >> 6; if (++pr.p == np) { pr.p = 0; pr.u += W; } }
    }
    cpc();
  }
};

template <int NT, int G2, int S>
__device__ __forceinline__ void run(Ctx& C, int e, int mt, int pass) {
  const P& p = *C.p;
  const int lane = C.lane, g = lane >> 2, t = lane & 3;
  const int cnt = C.sm->cnt[e], off = C.sm->off[e];
  const __nv_bfloat16* rp[NT];
#pragma unroll
  for (int n = 0; n < NT; n++) {
    int i = pass * 64 + n * 8 + g; int r = C.list[off + (i < cnt ? i : 0)];
    rp[n] = (G2 ? p.inter + (size_t)r * HD : p.x + (size_t)(r >> 3) * HD) + t * 16;
  }
  float acc[NT][4];
#pragma unroll
  for (int n = 0; n < NT; n++) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f;
  for (int s = 0; s < SPT; s++) {
    cpw<S - 2>();
    C.issue(C.ring + ((C.q + S - 1) % S) * STG);
    uint32_t slot = C.ring + (C.q % S) * STG + lane * 16; C.q++;
    uint4 sc = lds(slot + 4096);
#pragma unroll
    for (int c = 0; c < 8; c++) {
      uint4 w = lds(slot + c * 512);
      uint32_t sw = (c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w));
      uint32_t pr = (c & 1) ? sw >> 16 : sw & 0xffff;
      uint32_t sb = ((pr & 0x7f) << 4) | ((pr & 0x80) << 8) | ((pr & 0x7f00) << 12) | ((pr & 0x8000) << 16);
      sb = bmul(bmul(sb, 0x7B007B00u), 0x7B007B00u);  // * 2^119 * 2^119
      uint32_t sA = __byte_perm(sb, 0, 0x1010), sB = __byte_perm(sb, 0, 0x3232);
      uint32_t a[4][4];
      uint32_t ww[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
      for (int j = 0; j < 4; j++) {
        uint32_t q = ww[j];
        a[j][0] = bmul(((q >> 6) & 0x01C001C0u) | (q & 0x80008000u), sA);
        a[j][1] = bmul(((q >> 2) & 0x01C001C0u) | ((q << 4) & 0x80008000u), sB);
        a[j][2] = bmul(((q << 2) & 0x01C001C0u) | ((q << 8) & 0x80008000u), sA);
        a[j][3] = bmul(((q << 6) & 0x01C001C0u) | ((q << 12) & 0x80008000u), sB);
      }
      const int kb = s * 512 + c * 64;
#pragma unroll
      for (int n = 0; n < NT; n++) {
        uint4 b0, b1;
        if (G2) { b0 = ldcg(rp[n] + kb); b1 = ldcg(rp[n] + kb + 8); }
        else { b0 = ldnc(rp[n] + kb); b1 = ldnc(rp[n] + kb + 8); }
        mma(acc[n], a[0], b0.x, b0.y); mma(acc[n], a[1], b0.z, b0.w);
        mma(acc[n], a[2], b1.x, b1.y); mma(acc[n], a[3], b1.z, b1.w);
      }
    }
  }
  // epilogue
  extern __shared__ __align__(16) uint8_t smem[];
  __nv_bfloat16* scr = (__nv_bfloat16*)(smem + WPC * S * STG + (threadIdx.x >> 5) * SCR);
  if (!G2) {
    const float gs = p.g1[p.lo + e];
#pragma unroll
    for (int n = 0; n < NT; n++) {
#pragma unroll
      for (int h = 0; h < 2; h++) {
        float gt = acc[n][h] * gs, up = acc[n][2 + h] * gs;
        scr[(n * 8 + 2 * t + h) * 8 + g] = __float2bfloat16(gt / (1.f + __expf(-gt)) * up);
      }
    }
    __syncwarp();
    for (int i = lane; i < NT * 8; i += 32) {
      int ii = pass * 64 + i;
      if (ii < cnt) { int r = C.list[off + ii]; *(uint4*)(p.inter + (size_t)r * HD + mt * 8) = *(uint4*)(scr + i * 8); }
    }
  } else {
    const float gs = p.g2[p.lo + e];
    const float n0 = p.nw[mt * 16 + g] * gs, n1 = p.nw[mt * 16 + g + 8] * gs;
#pragma unroll
    for (int n = 0; n < NT; n++) {
      float v0 = acc[n][0] * n0, v1 = acc[n][1] * n0, v2 = acc[n][2] * n1, v3 = acc[n][3] * n1;
      scr[(n * 8 + 2 * t) * 16 + g] = __float2bfloat16(v0);
      scr[(n * 8 + 2 * t + 1) * 16 + g] = __float2bfloat16(v1);
      scr[(n * 8 + 2 * t) * 16 + g + 8] = __float2bfloat16(v2);
      scr[(n * 8 + 2 * t + 1) * 16 + g + 8] = __float2bfloat16(v3);
      float sa = v0 * v0 + v2 * v2, sb = v1 * v1 + v3 * v3;
#pragma unroll
      for (int o = 4; o < 32; o <<= 1) { sa += __shfl_xor_sync(0xffffffffu, sa, o); sb += __shfl_xor_sync(0xffffffffu, sb, o); }
      if (g == 0) {
        int i0 = pass * 64 + n * 8 + 2 * t;
        if (i0 < cnt) atomicAdd(p.ss + C.list[off + i0], sa);
        if (i0 + 1 < cnt) atomicAdd(p.ss + C.list[off + i0 + 1], sb);
      }
    }
    __syncwarp();
    for (int i = lane; i < NT * 16; i += 32) {
      int ii = pass * 64 + (i >> 1);
      if (ii < cnt) { int r = C.list[off + ii]; *(uint4*)(p.y + (size_t)r * HD + mt * 16 + (i & 1) * 8) = *(uint4*)(scr + i * 8); }
    }
  }
  __syncwarp();
}

template <int G2, int S>
__device__ __forceinline__ void dispatch(Ctx& C, int nt, int e, int mt, int pass) {
  if (nt <= 1) run<1, G2, S>(C, e, mt, pass);
  else if (nt <= 2) run<2, G2, S>(C, e, mt, pass);
  else if (nt <= 4) run<4, G2, S>(C, e, mt, pass);
  else run<8, G2, S>(C, e, mt, pass);
}

#ifndef NSTG
#define NSTG 4
#endif
__global__ void __launch_bounds__(WPC * 32, 1) moe_kernel(P p) {
  constexpr int S = NSTG;
  extern __shared__ __align__(16) uint8_t smem[];
  Sm* sm = (Sm*)(smem + WPC * S * STG + WPC * SCR);
  int* list = (int*)(sm + 1);
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5, R = p.M * 8;
  if (tid < NLOC) sm->cnt[tid] = 0;
  __syncthreads();
  for (int i = tid; i < R; i += blockDim.x) { unsigned e = p.ids[i] - p.lo; if (e < NLOC) atomicAdd(&sm->cnt[e], 1); }
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
  for (int i = tid; i < R; i += blockDim.x) { unsigned e = p.ids[i] - p.lo; if (e < NLOC) list[atomicAdd(&sm->fill[e], 1)] = i; }
  __syncthreads();

  Ctx C;
  C.p = &p; C.sm = sm; C.list = list; C.lane = lane; C.W = gridDim.x * WPC;
  C.T1 = sm->na * T1E; C.T = C.T1 + sm->na * T2E;
  C.ring = (uint32_t)__cvta_generic_to_shared(smem) + wid * S * STG; C.q = 0;
  const int gw = wid * gridDim.x + blockIdx.x;
  C.pr.u = gw; C.pr.p = 0; C.pr.s = 0;
#pragma unroll
  for (int i = 0; i < S - 1; i++) C.issue(C.ring + i * STG);
  for (int u = gw; u < C.T; u += C.W) {
    int g2, e, mt; C.job(u, g2, e, mt);
    const int cnt = sm->cnt[e], np = (cnt + 63) >> 6;
    if (g2) {
      if (lane == 0) while (ldacq(p.ctr + e) < T1E) __nanosleep(64);
      __syncwarp();
    } else if (mt == 0) {
      for (int i = lane; i < cnt; i += 32) p.ss[list[sm->off[e] + i]] = 0.f;
    }
    for (int ps = 0; ps < np; ps++) {
      int nt = min(NTM, (cnt - ps * 64 + 7) >> 3);
      if (g2) dispatch<1, S>(C, nt, e, mt, ps); else dispatch<0, S>(C, nt, e, mt, ps);
    }
    __threadfence(); __syncwarp();
    if (lane == 0) atomicAdd(p.ctr + (g2 ? NLOC : e), 1);
  }
  cpw<0>();
  // combine
  const int T2 = sm->na * T2E;
  if (lane == 0) while (ldacq(p.ctr + NLOC) < T2) __nanosleep(64);
  __syncwarp();
  for (int it = gw; it < p.M * 16; it += C.W) {
    const int t = it >> 4, col = (it & 15) * 256 + lane * 8;
    float a[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int k = 0; k < 8; k++) {
      const int r = t * 8 + k;
      unsigned e = p.ids[r] - p.lo;
      if (e < NLOC) {
        float cf = p.tw[r] * rsqrtf(__ldcg(p.ss + r) * (1.f / HD) + 1e-8f);
        uint4 v = ldcg(p.y + (size_t)r * HD + col);
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
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    if (atomicAdd(p.ctr + NLOC + 1, 1) == (int)gridDim.x - 1) {
      for (int i = 0; i < NLOC + 2; i++) p.ctr[i] = 0;
      __threadfence();
    }
  }
}

static int g_grid = 0;
static size_t smem_bytes(int M) { return (size_t)WPC * NSTG * STG + WPC * SCR + sizeof(Sm) + (size_t)M * 8 * 4; }

void moe(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor w1, torch::Tensor w2, torch::Tensor g1,
         torch::Tensor g2, torch::Tensor nw, torch::Tensor inter, torch::Tensor y, torch::Tensor ss, torch::Tensor ctr,
         torch::Tensor out, int64_t lo) {
  int M = x.size(0);
  if (!g_grid) {
    int dev; cudaGetDevice(&dev); int mx; cudaDeviceGetAttribute(&mx, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaDeviceGetAttribute(&g_grid, cudaDevAttrMultiProcessorCount, dev);
    cudaFuncSetAttribute(moe_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, mx);
  }
  P p{w1.data_ptr<uint8_t>(), w2.data_ptr<uint8_t>(), g1.data_ptr<float>(), g2.data_ptr<float>(),
      (const __nv_bfloat16*)x.data_ptr(), ids.data_ptr<int>(), tw.data_ptr<float>(), nw.data_ptr<float>(),
      (__nv_bfloat16*)inter.data_ptr(), (__nv_bfloat16*)y.data_ptr(), ss.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr(),
      ctr.data_ptr<int>(), M, (int)lo};
  auto st = at::cuda::getCurrentCUDAStream();
  moe_kernel<<<g_grid, WPC * 32, smem_bytes(M), st>>>(p);
}
