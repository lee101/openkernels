PF2_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
typedef __nv_bfloat16 bf;
__device__ __forceinline__ void unp8(uint4 u, float* v) {
  const __nv_bfloat162* p = (const __nv_bfloat162*)&u;
#pragma unroll
  for (int i = 0; i < 4; i++) { const float2 f = __bfloat1622float2(p[i]); v[2 * i] = f.x; v[2 * i + 1] = f.y; }
}
__device__ __forceinline__ uint4 pk8(const float* v) {
  uint4 u; __nv_bfloat162* p = (__nv_bfloat162*)&u;
#pragma unroll
  for (int i = 0; i < 4; i++) p[i] = __floats2bfloat162_rn(v[2 * i], v[2 * i + 1]);
  return u;
}
__device__ __forceinline__ void ld8f(const float* p, float* v) {
  const float4 x = __ldg((const float4*)p), y = __ldg((const float4*)p + 1);
  v[0] = x.x; v[1] = x.y; v[2] = x.z; v[3] = x.w; v[4] = y.x; v[5] = y.y; v[6] = y.z; v[7] = y.w;
}
__device__ __forceinline__ void ld8s(const float* p, float* v) {
  const float4 x = __ldcs((const float4*)p), y = __ldcs((const float4*)p + 1);
  v[0] = x.x; v[1] = x.y; v[2] = x.z; v[3] = x.w; v[4] = y.x; v[5] = y.y; v[6] = y.z; v[7] = y.w;
}
__device__ __forceinline__ float wsum(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
template <int NW>
__device__ __forceinline__ float bsum(float v, float* red) {
  v = wsum(v);
  __syncthreads();
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < NW; i++) s += red[i];
  return s;
}
__device__ __forceinline__ void rope_cs(const float* invf, int pos, int pair, float& sn, float& cs) {
  const float ang = (float)pos * invf[pair];
  const double a = (double)ang, k = rint(a * 0.15915494309189535);
  __sincosf((float)(a - k * 6.283185307179586), &sn, &cs);
}
__device__ __forceinline__ uint16_t f8x2(float a, float b) { return __nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3); }

// q/k RMSNorm + interleaved RoPE + FP8 (q8, k/v cache rows): one warp per token (all heads' loads in flight), lane = RoPE pair
template <int HQ, int HKV>
__global__ void __launch_bounds__(64) qkp(const bf* qkv, const int64_t* pos, const int64_t* loc, uint8_t* kc, uint8_t* vc, const float* ksc,
                                           const float* vsc, uint8_t* q8, int T, float theta, float qsc, const float* invf) {
  constexpr int NH = HQ + 2 * HKV, W = (2 * HQ + 2 * HKV) * 64;
  const int tok = blockIdx.x * 2 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (tok >= T) return;
  const bf* row = qkv + (size_t)tok * W + 2 * lane;
  float2 x[NH];
#pragma unroll
  for (int h = 0; h < NH; h++) x[h] = __bfloat1622float2(*(const __nv_bfloat162*)(row + h * 64));
  const size_t kv = (size_t)loc[tok] * HKV * 64 + 2 * lane;
  float sn = 0.f, cs = 1.f;
  if (theta > 0.f) rope_cs(invf, (int)pos[tok], lane, sn, cs);
  const float ks = *ksc, vs = *vsc;
#pragma unroll
  for (int h = 0; h < HQ + HKV; h++) {
    float x0 = x[h].x, x1 = x[h].y;
    const float r = rsqrtf(wsum(x0 * x0 + x1 * x1) / 64 + 1e-5f);
    x0 *= r; x1 *= r;
    if (h < HQ) { x0 *= qsc; x1 *= qsc; }
    if (theta > 0.f) { const float y0 = x0 * cs - x1 * sn, y1 = x0 * sn + x1 * cs; x0 = y0; x1 = y1; }
    if (h < HQ) *(uint16_t*)(q8 + ((size_t)tok * HQ + h) * 64 + 2 * lane) = f8x2(x0 / qsc, x1 / qsc);
    else *(uint16_t*)(kc + kv + (h - HQ) * 64) = f8x2(x0 / ks, x1 / ks);
  }
#pragma unroll
  for (int h = 0; h < HKV; h++) *(uint16_t*)(vc + kv + h * 64) = f8x2(x[HQ + HKV + h].x / vs, x[HQ + HKV + h].y / vs);
}
// attention output: bf16(rms(o, 1e-5) * sigmoid(gate)), one warp per token
template <int HQ, int HKV>
__global__ void __launch_bounds__(64) ong(const bf* o, const bf* qkv, bf* out, int T) {
  constexpr int W = (2 * HQ + 2 * HKV) * 64, G = (HQ + 2 * HKV) * 64;
  const int tok = blockIdx.x * 2 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (tok >= T) return;
  float2 x[HQ], g[HQ];
#pragma unroll
  for (int h = 0; h < HQ; h++) {
    x[h] = __bfloat1622float2(*(const __nv_bfloat162*)(o + ((size_t)tok * HQ + h) * 64 + 2 * lane));
    g[h] = __bfloat1622float2(*(const __nv_bfloat162*)(qkv + (size_t)tok * W + G + h * 64 + 2 * lane));
  }
#pragma unroll
  for (int h = 0; h < HQ; h++) {
    const float r = rsqrtf(wsum(x[h].x * x[h].x + x[h].y * x[h].y) / 64 + 1e-5f);
    *(__nv_bfloat162*)(out + ((size_t)tok * HQ + h) * 64 + 2 * lane) =
        __floats2bfloat162_rn(x[h].x * r * (1.f / (1.f + __expf(-g[h].x))), x[h].y * r * (1.f / (1.f + __expf(-g[h].y))));
  }
}
#define HD 8192
#define RNC (HD / 8 / NT)
#define RCOL(j) (((j) * NT + threadIdx.x) * 8)
// boundary 1 (256 threads x 32 columns per row): s1 = al*s + be*rms(a, 1e-8) (not stored: ra = that rms factor), h = bf16(rms(s1, 1e-5) * gn)
template <int NT, int MB>
__global__ void __launch_bounds__(NT, MB) resa(const bf* a, const float* s, const float* al, const float* be, const float* gn, bf* h, float* ra) {
  __shared__ float red[NT / 32];
  const size_t rb = (size_t)blockIdx.x * HD;
  uint4 ar[RNC]; float sv[RNC * 8], t0[8], t1[8];
#pragma unroll
  for (int j = 0; j < RNC; j++) { ar[j] = __ldcs((const uint4*)(a + rb + RCOL(j))); ld8s(s + rb + RCOL(j), sv + 8 * j); }
  float ss = 0.f;
#pragma unroll
  for (int j = 0; j < RNC; j++) { unp8(ar[j], t0);
#pragma unroll
    for (int i = 0; i < 8; i++) ss += t0[i] * t0[i]; }
  const float rr = rsqrtf(bsum<NT / 32>(ss, red) / HD + 1e-8f);
  float s2 = 0.f;
#pragma unroll
  for (int j = 0; j < RNC; j++) {
    float x[8]; unp8(ar[j], x); ld8f(al + RCOL(j), t0); ld8f(be + RCOL(j), t1);
#pragma unroll
    for (int i = 0; i < 8; i++) { float& v = sv[8 * j + i]; v = t0[i] * v + t1[i] * (x[i] * rr); s2 += v * v; }
  }
  const float rs = rsqrtf(bsum<NT / 32>(s2, red) / HD + 1e-5f);
#pragma unroll
  for (int j = 0; j < RNC; j++) {
    ld8f(gn + RCOL(j), t0);
#pragma unroll
    for (int i = 0; i < 8; i++) t0[i] *= sv[8 * j + i] * rs;
    *(uint4*)(h + rb + RCOL(j)) = pk8(t0);
  }
  if (threadIdx.x == 0) ra[blockIdx.x] = rr;
}
// boundary 2: s1 recomputed from (s, a, ra); s2 = al2*s1 + be2*rms(f*gf, 1e-8); hid = bf16(rms(s2, 1e-5) * gn)
template <int NT, int MB>
__global__ void __launch_bounds__(NT, MB) resb(const bf* f, const float* gf, const bf* a, const float* ra, const float* s, const float* al1, const float* be1,
                                               const float* al2, const float* be2, const float* gn, float* so, bf* hid) {
  __shared__ float red[NT / 32];
  const size_t rb = (size_t)blockIdx.x * HD;
  uint4 fr[RNC], ar[RNC]; float sv[RNC * 8], t0[8], t1[8], x[8];
#pragma unroll
  for (int j = 0; j < RNC; j++) {
    fr[j] = __ldcs((const uint4*)(f + rb + RCOL(j))); ar[j] = __ldcs((const uint4*)(a + rb + RCOL(j))); ld8s(s + rb + RCOL(j), sv + 8 * j);
  }
  const float rr = ra[blockIdx.x];
  float ss = 0.f;
#pragma unroll
  for (int j = 0; j < RNC; j++) {
    unp8(ar[j], x); ld8f(al1 + RCOL(j), t0); ld8f(be1 + RCOL(j), t1);
#pragma unroll
    for (int i = 0; i < 8; i++) sv[8 * j + i] = t0[i] * sv[8 * j + i] + t1[i] * (x[i] * rr);
    unp8(fr[j], x); ld8f(gf + RCOL(j), t0);
#pragma unroll
    for (int i = 0; i < 8; i++) { const float z = x[i] * t0[i]; ss += z * z; }
  }
  const float rf = rsqrtf(bsum<NT / 32>(ss, red) / HD + 1e-8f);
  float s2 = 0.f;
#pragma unroll
  for (int j = 0; j < RNC; j++) {
    unp8(fr[j], x); ld8f(gf + RCOL(j), t0); ld8f(al2 + RCOL(j), t1);
#pragma unroll
    for (int i = 0; i < 8; i++) x[i] *= t0[i];
    ld8f(be2 + RCOL(j), t0);
#pragma unroll
    for (int i = 0; i < 8; i++) { float& v = sv[8 * j + i]; v = t1[i] * v + t0[i] * (x[i] * rf); s2 += v * v; }
  }
  const float rs = rsqrtf(bsum<NT / 32>(s2, red) / HD + 1e-5f);
#pragma unroll
  for (int j = 0; j < RNC; j++) {
    float4* sp = (float4*)(so + rb + RCOL(j));
    __stcs(sp, make_float4(sv[8 * j], sv[8 * j + 1], sv[8 * j + 2], sv[8 * j + 3]));
    __stcs(sp + 1, make_float4(sv[8 * j + 4], sv[8 * j + 5], sv[8 * j + 6], sv[8 * j + 7]));
    ld8f(gn + RCOL(j), t0);
#pragma unroll
    for (int i = 0; i < 8; i++) t0[i] *= sv[8 * j + i] * rs;
    __stcs((uint4*)(hid + rb + RCOL(j)), pk8(t0));
  }
}
// out = bf16(rms(x, 1e-5) * g) over 4096 columns (512 threads x 8)
__global__ void __launch_bounds__(512) rmsg2(const bf* x, const float* g, bf* out) {
  __shared__ float red[16];
  const int r = blockIdx.x, c0 = threadIdx.x * 8;
  float v[8], gv[8], ss = 0.f;
  unp8(__ldg((const uint4*)(x + (size_t)r * 4096 + c0)), v);
  ld8f(g + c0, gv);
#pragma unroll
  for (int i = 0; i < 8; i++) ss += v[i] * v[i];
  const float rr = rsqrtf(bsum<16>(ss, red) / 4096 + 1e-5f);
#pragma unroll
  for (int i = 0; i < 8; i++) v[i] = v[i] * rr * gv[i];
  *(uint4*)(out + (size_t)r * 4096 + c0) = pk8(v);
}
// A[i] = bf16(rms(x[token of route i], 1e-5) * g): persistent over the grouped rows, 256 threads x 16 columns, next row prefetched
__global__ void __launch_bounds__(256) gnorm(const bf* x, const int* list, const int* offs, const float* g, bf* A) {
  __shared__ float red[8];
  const int c0 = threadIdx.x * 8, c1 = c0 + 2048, tot = offs[31];
  int i = blockIdx.x;
  uint4 n0 = make_uint4(0, 0, 0, 0), n1 = n0;
  if (i < tot) { const bf* p = x + (size_t)(list[i] >> 3) * 4096; n0 = __ldg((const uint4*)(p + c0)); n1 = __ldg((const uint4*)(p + c1)); }
  for (; i < tot; i += gridDim.x) {
    const uint4 u0 = n0, u1 = n1;
    if (i + gridDim.x < tot) { const bf* p = x + (size_t)(list[i + gridDim.x] >> 3) * 4096; n0 = __ldg((const uint4*)(p + c0)); n1 = __ldg((const uint4*)(p + c1)); }
    float v[16], gv[8], ss = 0.f;
    unp8(u0, v); unp8(u1, v + 8);
#pragma unroll
    for (int k = 0; k < 16; k++) ss += v[k] * v[k];
    const float rr = rsqrtf(bsum<8>(ss, red) / 4096 + 1e-5f);
    ld8f(g + c0, gv);
#pragma unroll
    for (int k = 0; k < 8; k++) v[k] = v[k] * rr * gv[k];
    ld8f(g + c1, gv);
#pragma unroll
    for (int k = 0; k < 8; k++) v[8 + k] = v[8 + k] * rr * gv[k];
    *(uint4*)(A + (size_t)i * 4096 + c0) = pk8(v); *(uint4*)(A + (size_t)i * 4096 + c1) = pk8(v + 8);
  }
}
__device__ __forceinline__ uint4 swg(uint4 a, uint4 b) {
  float gg[8], u[8]; unp8(a, gg); unp8(b, u);
#pragma unroll
  for (int k = 0; k < 8; k++) gg[k] = gg[k] / (1.f + __expf(-gg[k])) * u[k];
  return pk8(gg);
}
__global__ void __launch_bounds__(256) swiglu2(const bf* H, const int* offs, bf* I) {
  const long long n = (long long)offs[31] * 512, st = (long long)gridDim.x * blockDim.x;
  for (long long j = (long long)blockIdx.x * blockDim.x + threadIdx.x; j < n; j += 2 * st) {
    const long long j2 = j + st, i = j >> 9, i2 = j2 >> 9; const int c = (int)(j & 511) * 8, c2 = (int)(j2 & 511) * 8;
    const uint4 a = __ldcs((const uint4*)(H + i * 8192 + c)), b = __ldcs((const uint4*)(H + i * 8192 + 4096 + c));
    uint4 a2 = make_uint4(0, 0, 0, 0), b2 = a2;
    if (j2 < n) { a2 = __ldcs((const uint4*)(H + i2 * 8192 + c2)); b2 = __ldcs((const uint4*)(H + i2 * 8192 + 4096 + c2)); }
    *(uint4*)(I + i * 4096 + c) = swg(a, b);
    if (j2 < n) *(uint4*)(I + i2 * 4096 + c2) = swg(a2, b2);
  }
}
// MoE combine as a weighted gather: out[t] = sum over t's local slots of tw * rms(Y[pos] * gw, 1e-8), bf16; 256 threads x 16 columns,
// the next slot's row prefetched, one barrier per slot (double-buffered partials)
__global__ void __launch_bounds__(256) comb(const bf* Y, const int* loc, const int* pos, const float* gw, const float* tw, const bf* add, bf* out) {
  __shared__ float red[2][8];
  const int t = blockIdx.x, c0 = threadIdx.x * 8, c1 = c0 + 2048, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  unsigned msk = 0;
#pragma unroll
  for (int k = 0; k < 8; k++) msk |= ((unsigned)loc[t * 8 + k] < 32u ? 1u : 0u) << k;
  float acc[16];
#pragma unroll
  for (int j = 0; j < 16; j++) acc[j] = 0.f;
  if (add) { unp8(__ldcs((const uint4*)(add + (size_t)t * 4096 + c0)), acc); unp8(__ldcs((const uint4*)(add + (size_t)t * 4096 + c1)), acc + 8); }
  int k = -1; uint4 u0 = make_uint4(0, 0, 0, 0), u1 = u0;
  if (msk) {
    k = __ffs(msk) - 1; msk &= msk - 1;
    const bf* p = Y + (size_t)(pos ? pos[t * 8 + k] : t * 8 + k) * 4096; u0 = __ldcs((const uint4*)(p + c0)); u1 = __ldcs((const uint4*)(p + c1));
  }
  for (int b = 0; k >= 0; b ^= 1) {
    int kn = -1; uint4 n0 = u0, n1 = u1;
    if (msk) {
      kn = __ffs(msk) - 1; msk &= msk - 1;
      const bf* p = Y + (size_t)(pos ? pos[t * 8 + kn] : t * 8 + kn) * 4096; n0 = __ldcs((const uint4*)(p + c0)); n1 = __ldcs((const uint4*)(p + c1));
    }
    float v[16], gv[8], q = 0.f;
    unp8(u0, v); unp8(u1, v + 8);
    ld8f(gw + c0, gv);
#pragma unroll
    for (int j = 0; j < 8; j++) { v[j] *= gv[j]; q += v[j] * v[j]; }
    ld8f(gw + c1, gv);
#pragma unroll
    for (int j = 0; j < 8; j++) { v[8 + j] *= gv[j]; q += v[8 + j] * v[8 + j]; }
    q = wsum(q);
    if (lane == 0) red[b][w] = q;
    __syncthreads();
    q = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++) q += red[b][i];
    const float cf = tw[t * 8 + k] * rsqrtf(q / 4096 + 1e-8f);
#pragma unroll
    for (int j = 0; j < 16; j++) acc[j] += v[j] * cf;
    k = kn; u0 = n0; u1 = n1;
  }
  *(uint4*)(out + (size_t)t * 4096 + c0) = pk8(acc); *(uint4*)(out + (size_t)t * 4096 + c1) = pk8(acc + 8);
}
// routing on the [hi | lo] router GEMM ([T, 512] fp32): one warp per token; ids, loc, Marlin's local ids, weights, local counts
__global__ void route2(const float* lg, int ld, const float* bias, const int* lstart, int T, int* ids, int* loc, int* lid, float* tw, int* cnt) {
  const int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (r >= T) return;
  float sc[8], key[8];
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const int e = i * 32 + lane;
    const float l = lg[(size_t)r * ld + e] + lg[(size_t)r * ld + 256 + e];
    sc[i] = 1.f / (1.f + expf(-l * 0.1767766952966369f));
    key[i] = sc[i] + bias[e];
  }
  const int ls = *lstart;
  float pick = 0.f, sum = 0.f;
  int myid = 0;
#pragma unroll
  for (int k = 0; k < 8; k++) {
    float bv = key[0]; int bi = 0;
#pragma unroll
    for (int i = 1; i < 8; i++) if (key[i] > bv) { bv = key[i]; bi = i; }
    int be = bi * 32 + lane;
#pragma unroll
    for (int o = 16; o; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, bv, o); const int oe = __shfl_xor_sync(0xffffffffu, be, o);
      if (ov > bv || (ov == bv && oe < be)) { bv = ov; be = oe; }
    }
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++) if (i * 32 + lane == be) { s = sc[i]; key[i] = -INFINITY; }
    s = __shfl_sync(0xffffffffu, s, be & 31);
    sum += s;
    if (lane == k) { pick = s; myid = be; }
  }
  if (lane < 8) {
    const int l = (myid - ls) & 255;
    ids[r * 8 + lane] = myid; loc[r * 8 + lane] = l; lid[r * 8 + lane] = l < 32 ? l : -1;
    tw[r * 8 + lane] = pick / (sum + 1e-15f);
    if (cnt && l < 32) atomicAdd(cnt + l, 1);
  }
}
// hybrid MoE: experts with more than thr local routes go to dequant + grouped GEMM (locb, cntb), the rest to the fused ring kernel (locq)
__global__ void msplit_k(const int* loc, const int* cnt, int thr, int R, int* locq, int* locb, int* cntb) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 32) cntb[i] = cnt[i] > thr ? cnt[i] : 0;
  if (i < R) {
    const int l = loc[i];
    const bool lc = l < 32, hv = lc && cnt[l] > thr;
    locq[i] = hv ? 255 : l; locb[i] = (lc && !hv) ? 255 : l;
  }
}
__global__ void rscan2(const int* cnt, int* offs, int* fill) {
  const int l = threadIdx.x;
  int v = cnt[l], x = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) { const int n = __shfl_up_sync(0xffffffffu, x, o); if (l >= o) x += n; }
  offs[l] = x; fill[l] = x - v;
}
__global__ void rfill2(const int* loc, int R, int* fill, int* list, int* pos) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < R && loc[i] < 32) { const int j = atomicAdd(fill + loc[i], 1); list[j] = i; pos[i] = j; }
}
// NVFP4 -> bf16, 32 values per thread (two 16-value scale blocks), e2m1 decoded arithmetically, experts without local routes skipped
__device__ __forceinline__ float e2m1(uint32_t c) {
  const uint32_t m = c & 7, e = m >> 1, t = m & 1;
  return __uint_as_float((e ? ((e + 126) << 23) | (t << 22) : t * 0x3F000000u) | ((c & 8) << 28));
}
__global__ void __launch_bounds__(256) deq2(const uint4* w, const uint16_t* sf, const float* gs, const int* cnt, uint4* out, int nbe, int n2) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n2) return;
  const int e = (2 * i) / nbe;
  if (cnt[e] == 0) return;
  const uint4 q = __ldcs(w + i);
  const uint16_t s2 = __ldcs(sf + i);
  const float g = gs[e];
  __nv_fp8_e4m3 a, b; a.__x = s2 & 0xff; b.__x = s2 >> 8;
  const float sa = (float)a * g, sb = (float)b * g;
  const uint32_t qq[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
  for (int h = 0; h < 4; h++) {
    const float sc = h < 2 ? sa : sb;
    float v[8];
#pragma unroll
    for (int k = 0; k < 8; k++) v[k] = e2m1(qq[h] >> (4 * k)) * sc;
    __stcs(out + 4 * (size_t)i + h, pk8(v));
  }
}
// deq2 with coalesced stores: one 4-byte word (8 values) per lane per step, a warp's 32 words contiguous, so each 16-byte store
// instruction covers 512 contiguous bytes; 4 independent words per thread (stride 32 words) loaded up front
__global__ void __launch_bounds__(256) deq3(const uint32_t* w, const uint8_t* sf, const float* gs, const int* cnt, uint4* out, long long wpe, long long nw) {
  const long long base = ((long long)blockIdx.x * 8 + (threadIdx.x >> 5)) * 128 + (threadIdx.x & 31);
  if (base >= nw) return;
  const int e = (int)(base / wpe);
  if (cnt[e] == 0) return;
  uint32_t q[4]; uint32_t sb[4];
#pragma unroll
  for (int k = 0; k < 4; k++) { q[k] = __ldcs(w + base + 32 * k); sb[k] = __ldcs(sf + ((base + 32 * k) >> 1)); }
  const float g = gs[e];
#pragma unroll
  for (int k = 0; k < 4; k++) {
    __nv_fp8_e4m3 a; a.__x = (uint8_t)sb[k];
    const float sc = (float)a * g;
    float v[8];
#pragma unroll
    for (int j = 0; j < 8; j++) v[j] = e2m1(q[k] >> (4 * j)) * sc;
    __stcs(out + base + 32 * k, pk8(v));
  }
}
void dequant3(torch::Tensor w, torch::Tensor sf, torch::Tensor gs, torch::Tensor cnt, torch::Tensor out) {
  const long long nw = w.numel() / 4, wpe = nw / 32;
  TORCH_CHECK(wpe % 1024 == 0);
  deq3<<<(unsigned)((nw + 1023) / 1024), 256, 0, at::cuda::getCurrentCUDAStream()>>>((const uint32_t*)w.data_ptr(), (const uint8_t*)sf.data_ptr(), gs.data_ptr<float>(),
      cnt.data_ptr<int>(), (uint4*)out.data_ptr(), wpe, nw);
}
void dequant2(torch::Tensor w, torch::Tensor sf, torch::Tensor gs, torch::Tensor cnt, torch::Tensor out) {
  const long long per_e = w.numel() * 2 / 32;
  const int nbe = (int)(per_e / 16), n2 = (int)(w.numel() / 16);
  deq2<<<(n2 + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>((const uint4*)w.data_ptr(), (const uint16_t*)sf.data_ptr(), gs.data_ptr<float>(),
      cnt.data_ptr<int>(), (uint4*)out.data_ptr(), nbe, n2);
}
// each request's visible keys/values (sliding: its window) gathered into contiguous [B, maxk, 128 B] rows for FA3's TMA path
__global__ void __launch_bounds__(256) kvg(const uint4* kc, const uint4* vc, const int* pt, const int* seq, int maxp, int ps, int L, int win, int maxk,
                                           uint4* ko, uint4* vo, int* cslo) {
  const int b = blockIdx.y, sb = seq[b], s0 = win ? max(0, sb - win + 1) : 0, n = sb + L - s0;
  if (blockIdx.x == 0 && threadIdx.x == 0) cslo[b] = n;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n * 8; i += gridDim.x * blockDim.x) {
    const int j = i >> 3, c = i & 7, p = s0 + j;
    const size_t r = ((size_t)pt[(size_t)b * maxp + p / ps] * ps + p % ps) * 8 + c, o = ((size_t)b * maxk + j) * 8 + c;
    ko[o] = __ldcs(kc + r); vo[o] = __ldcs(vc + r);
  }
}
void kvgather(torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seq, int64_t L, int64_t win, torch::Tensor ko, torch::Tensor vo, torch::Tensor cslo) {
  TORCH_CHECK(kc.size(2) * kc.size(3) == 128);
  const int B = ko.size(0), maxk = ko.size(1);
  dim3 grid(min((maxk * 8 + 255) / 256, 1024), B);
  kvg<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>((const uint4*)kc.data_ptr(), (const uint4*)vc.data_ptr(), pt.data_ptr<int>(), seq.data_ptr<int>(),
      (int)pt.size(1), (int)kc.size(1), (int)L, (int)win, maxk, (uint4*)ko.data_ptr(), (uint4*)vo.data_ptr(), cslo.data_ptr<int>());
}
template <typename K> static int occ(K k, int nt, int smem) {
  int n = 0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, k, nt, smem); return n > 0 ? n : 1;
}
static int nsm() { static int n = 0; if (!n) cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, 0); return n; }
#define ST at::cuda::getCurrentCUDAStream()
#define BP(t) ((bf*)t.data_ptr())
void qkprep(torch::Tensor qkv, torch::Tensor pos, torch::Tensor loc, torch::Tensor kc, torch::Tensor vc, torch::Tensor ksc, torch::Tensor vsc,
            torch::Tensor q8, int64_t hq, int64_t hkv, double theta, double qsc, torch::Tensor invf) {
  TORCH_CHECK(hq == 16 && hkv == 2);
  const int T = qkv.numel() / ((2 * hq + 2 * hkv) * 64);
  qkp<16, 2><<<(T + 1) / 2, 64, 0, ST>>>(BP(qkv), pos.data_ptr<int64_t>(), loc.data_ptr<int64_t>(), (uint8_t*)kc.data_ptr(), (uint8_t*)vc.data_ptr(),
      ksc.data_ptr<float>(), vsc.data_ptr<float>(), (uint8_t*)q8.data_ptr(), T, (float)theta, (float)qsc, invf.data_ptr<float>());
}
void ongate(torch::Tensor o, torch::Tensor qkv, torch::Tensor out, int64_t hq, int64_t hkv) {
  TORCH_CHECK(hq == 16 && hkv == 2);
  const int T = out.size(0);
  ong<16, 2><<<(T + 1) / 2, 64, 0, ST>>>(BP(o), BP(qkv), BP(out), T);
}
#define RA_ARGS BP(a), s.data_ptr<float>(), al.data_ptr<float>(), be.data_ptr<float>(), gn.data_ptr<float>(), BP(h), ra.data_ptr<float>()
void res_a(torch::Tensor a, torch::Tensor s, torch::Tensor al, torch::Tensor be, torch::Tensor gn, torch::Tensor h, torch::Tensor ra) {
  const int T = a.size(0);
  resa<512, 2><<<T, 512, 0, ST>>>(RA_ARGS);
}
#define RB_ARGS BP(f), gf.data_ptr<float>(), BP(a), ra.data_ptr<float>(), s.data_ptr<float>(), al1.data_ptr<float>(), be1.data_ptr<float>(), \
      al2.data_ptr<float>(), be2.data_ptr<float>(), gn.data_ptr<float>(), so.data_ptr<float>(), BP(hid)
void res_b(torch::Tensor f, torch::Tensor gf, torch::Tensor a, torch::Tensor ra, torch::Tensor s, torch::Tensor al1, torch::Tensor be1, torch::Tensor al2,
           torch::Tensor be2, torch::Tensor gn, torch::Tensor so, torch::Tensor hid) {
  const int T = f.size(0);
  resb<256, 2><<<T, 256, 0, ST>>>(RB_ARGS);
}
void rmsgain2(torch::Tensor x, torch::Tensor g, torch::Tensor out) {
  rmsg2<<<x.size(0), 512, 0, ST>>>(BP(x), g.data_ptr<float>(), BP(out));
}
void gathnorm(torch::Tensor x, torch::Tensor list, torch::Tensor offs, torch::Tensor g, torch::Tensor A) {
  gnorm<<<min((long)A.size(0), (long)nsm() * occ(gnorm, 256, 0)), 256, 0, ST>>>(BP(x), list.data_ptr<int>(), offs.data_ptr<int>(), g.data_ptr<float>(), BP(A));
}
void act2(torch::Tensor H, torch::Tensor offs, torch::Tensor I) {
  swiglu2<<<nsm() * occ(swiglu2, 256, 0), 256, 0, ST>>>(BP(H), offs.data_ptr<int>(), BP(I));
}
void combine(torch::Tensor Y, torch::Tensor loc, torch::Tensor pos, torch::Tensor gw, torch::Tensor tw, torch::Tensor add, torch::Tensor out) {
  comb<<<out.size(0), 256, 0, ST>>>(BP(Y), loc.data_ptr<int>(), pos.numel() ? pos.data_ptr<int>() : nullptr, gw.data_ptr<float>(), tw.data_ptr<float>(),
      add.numel() ? (const bf*)add.data_ptr() : nullptr, BP(out));
}
void routing2(torch::Tensor lg, torch::Tensor bias, torch::Tensor ls, torch::Tensor ids, torch::Tensor loc, torch::Tensor lid, torch::Tensor tw, torch::Tensor cnt) {
  const int T = lg.size(0);
  int* c = cnt.numel() ? cnt.data_ptr<int>() : nullptr;
  if (c) cudaMemsetAsync(c, 0, 32 * 4, ST);
  route2<<<(T + 3) / 4, 128, 0, ST>>>(lg.data_ptr<float>(), (int)lg.stride(0), bias.data_ptr<float>(), ls.data_ptr<int>(), T, ids.data_ptr<int>(), loc.data_ptr<int>(),
      lid.data_ptr<int>(), tw.data_ptr<float>(), c);
}
void msplit(torch::Tensor loc, torch::Tensor cnt, int64_t thr, torch::Tensor locq, torch::Tensor locb, torch::Tensor cntb) {
  const int R = loc.numel();
  msplit_k<<<(R + 255) / 256, 256, 0, ST>>>(loc.data_ptr<int>(), cnt.data_ptr<int>(), (int)thr, R, locq.data_ptr<int>(), locb.data_ptr<int>(), cntb.data_ptr<int>());
}
void rlist(torch::Tensor loc, torch::Tensor cnt, torch::Tensor offs, torch::Tensor fill, torch::Tensor list, torch::Tensor pos) {
  const int R = loc.numel();
  rscan2<<<1, 32, 0, ST>>>(cnt.data_ptr<int>(), offs.data_ptr<int>(), fill.data_ptr<int>());
  rfill2<<<(R + 255) / 256, 256, 0, ST>>>(loc.data_ptr<int>(), R, fill.data_ptr<int>(), list.data_ptr<int>(), pos.data_ptr<int>());
}
"""
_pf2mod = None
import time
_PF2_FN = ["qkprep", "ongate", "res_a", "res_b", "rmsgain2", "gathnorm", "act2", "combine", "routing2", "rlist", "dequant2", "kvgather", "msplit", "dequant3"]


def _pf2():
    global _pf2mod
    if _pf2mod is None:
        import re
        decl = " ".join(re.sub(r"\s+", " ", m) + ";" for m in re.findall(r"^void \w+\([^)]*\)", PF2_CUDA, re.M) if m.split("(")[0][5:] in _PF2_FN)
        _pf2mod = load_inline("pf2_glue_v12", cpp_sources=decl, cuda_sources=PF2_CUDA, functions=_PF2_FN, extra_cuda_cflags=["-O3"], verbose=False)
    return _pf2mod


PF2_LT = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>
#include <map>
#include <tuple>
#include <vector>
// layout bit 0: row-major descriptors with x as A (else column-major, W as A); bit 1: W pre-transposed [K, N]
// handle = layout * 100000 + index into that plan's list (heuristic results, then configs added by lt_add)
struct LtPlan { cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t a, b, c; std::vector<cublasLtMatmulHeuristicResult_t> algos; };
static cublasLtHandle_t g_lt = nullptr;
static void* g_ltws = nullptr;
static const size_t LTWS = 64ull << 20;
static std::map<std::tuple<int, int, int, int>, LtPlan> g_plans;
static cublasLtHandle_t lth() { if (!g_lt) { cublasLtCreate(&g_lt); cudaMalloc(&g_ltws, LTWS); } return g_lt; }
static LtPlan& ltplan(int m, int n, int k, int lay) {
  auto key = std::make_tuple(m, n, k, lay);
  auto it = g_plans.find(key);
  if (it != g_plans.end()) return it->second;
  lth();
  const bool row = lay & 1, pk = lay & 2;
  LtPlan p;
  cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  cublasOperation_t ta = !row && !pk ? CUBLAS_OP_T : CUBLAS_OP_N, tb = row && !pk ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
  cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
  if (row) {
    cublasLtMatrixLayoutCreate(&p.a, CUDA_R_16BF, m, k, k);
    cublasLtMatrixLayoutCreate(&p.b, CUDA_R_16BF, pk ? k : n, pk ? n : k, pk ? n : k);
    cublasLtMatrixLayoutCreate(&p.c, CUDA_R_16BF, m, n, n);
    cublasLtOrder_t o = CUBLASLT_ORDER_ROW;
    for (auto d : {p.a, p.b, p.c}) cublasLtMatrixLayoutSetAttribute(d, CUBLASLT_MATRIX_LAYOUT_ORDER, &o, sizeof(o));
  } else {
    cublasLtMatrixLayoutCreate(&p.a, CUDA_R_16BF, pk ? n : k, pk ? k : n, pk ? n : k);
    cublasLtMatrixLayoutCreate(&p.b, CUDA_R_16BF, k, m, k);
    cublasLtMatrixLayoutCreate(&p.c, CUDA_R_16BF, n, m, n);
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
int64_t lt_algos(int64_t m, int64_t n, int64_t k, int64_t lay) { return (int64_t)ltplan(m, n, k, lay).algos.size(); }
std::vector<int64_t> lt_caps(int64_t id) {
  cublasLtMatmulAlgo_t a;
  std::vector<int64_t> r;
  if (cublasLtMatmulAlgoInit(lth(), CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, (int)id, &a) != CUBLAS_STATUS_SUCCESS) return r;
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
int64_t lt_add(int64_t m, int64_t n, int64_t k, int64_t lay, std::vector<int64_t> cfg) {
  if (cfg.size() != 9) return -1;
  LtPlan& p = ltplan(m, n, k, lay);
  cublasLtMatmulAlgo_t a;
  if (cublasLtMatmulAlgoInit(g_lt, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, (int)cfg[0], &a) != CUBLAS_STATUS_SUCCESS) return -1;
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
std::vector<int64_t> lt_cfg(int64_t m, int64_t n, int64_t k, int64_t code) {
  LtPlan& p = ltplan(m, n, k, code / 100000);
  std::vector<int64_t> r;
  const auto& a = p.algos.at(code % 100000).algo;
  for (int i = 0; i < 9; i++) {
    size_t w = 0;
    if (i >= 7) { uint16_t v = 0; cublasLtMatmulAlgoConfigGetAttribute(&a, (cublasLtMatmulAlgoConfigAttributes_t)i, &v, sizeof(v), &w); r.push_back(v); }
    else { uint32_t v = 0; cublasLtMatmulAlgoConfigGetAttribute(&a, (cublasLtMatmulAlgoConfigAttributes_t)i, &v, sizeof(v), &w); r.push_back(v); }
  }
  return r;
}
int64_t lt_mm(torch::Tensor x, torch::Tensor w, torch::Tensor wt, torch::Tensor out, int64_t code) {
  const int m = x.size(0), k = x.size(1), n = w.size(0), lay = (int)(code / 100000);
  LtPlan& p = ltplan(m, n, k, lay);
  const float alpha = 1.f, beta = 0.f;
  const void* wp = (lay & 2) ? wt.data_ptr() : w.data_ptr();
  const void* a = (lay & 1) ? x.data_ptr() : wp;
  const void* b = (lay & 1) ? wp : x.data_ptr();
  const auto& al = p.algos.at(code % 100000);
  return (int64_t)cublasLtMatmul(g_lt, p.op, &alpha, a, p.a, b, p.b, &beta, out.data_ptr(), p.c, out.data_ptr(), p.c,
                                 &al.algo, g_ltws, LTWS, at::cuda::getCurrentCUDAStream());
}
"""
_ltmod = []
LT_TUNE = True
ALIGN_LOCAL = True  # align only the local routes over 32 buckets: Marlin (is_ep) then has no -1 blocks to walk past
KVC_TUNE = True
FA_TUNE = True  # time FA3 num_splits x pack_gqa per case after the paged/contiguous pick (FA3's heuristic: ~94 CTAs at B1 L753 full)
FA_FX = tuple({"num_splits": n, "pack_gqa": pg} for n in (1, 2, 3, 4, 6, 8) for pg in (False, True))
KVC_CASES = ((1, 753, 80000, 0), (1, 753, 80000, 2048), (4, 753, 80000, 0), (4, 753, 80000, 2048), (1, 16384, 32768, 0), (1, 16384, 32768, 2048))
MOE_Q = True  # time the fused W4A16 prefill kernel in prepare() and use it where it wins
# fused-ring (muse_moe_w4a16 v23) builds timed in prepare(): extra -D flags over Q__cfg() (TP=128 needs RPW=1: RPW=2 spills, NSLOT=3 exceeds smem)
QV = {"a": (), "b": ("-DLAG=1",)}  # XBULK (583 vs 560 us at T753) and TP=128 builds lost on H100  # TP=128 (RPW=1, NSLOT=2) lost 1.4-1.5x on H100
HTHR = (256, 384, 512, 768, 1024)  # hybrid split: experts with more local routes than this go to dequant + grouped GEMM
# per-expert local route counts of the scored soft routing (busiest 32-expert range, ~1.8 routes/token; reference router + judge-style
# router seeds picked for that load, two draws per size), used for the prepare() timings
MOE_PROF = {753: ((400, 314, 241, 174, 105, 57, 19, 15, 9, 9, 5, 2, 1, 1, 1), (635, 184, 181, 154, 69, 68, 18, 16, 10, 6, 5, 4, 4, 3)),
            3012: ((1571, 1248, 749, 490, 488, 280, 255, 120, 61, 46, 42, 30, 25, 6, 2, 2, 1, 1, 1),
                   (2466, 1799, 397, 261, 256, 78, 46, 43, 21, 19, 14, 11, 8, 7, 1, 1, 1)),
            16384: ((10543, 8504, 3315, 2228, 1774, 1545, 852, 661, 306, 128, 109, 60, 57, 28, 24, 13, 5, 2),
                    (7853, 5763, 4218, 4026, 2729, 2553, 1105, 608, 511, 336, 281, 94, 75, 8, 6, 5, 5, 3, 2, 1))}
MOE_T = (753, 3012, 16384)
MAR_BM = {}  # T -> Marlin moe_block_size override (default: SGLang's heuristic)
# H200 picks from the pf8 judge search, (layout, CUBLASLT_ALGO_CONFIG_*) seeds tried next to the heuristics
LT_JOBS = (("o", 3012), ("o", 16384), ("pooler", 16384), ("unpooler", 16384))
LT_SEEDS = ((0, (66, 197, 1, 0, 0, 0, 35, 0, 5)), (3, (66, 29, 1, 0, 0, 1, 35, 0, 3)), (1, (66, 29, 1, 0, 0, 1, 35, 0, 3)),
            (1, (66, 201, 1, 0, 0, 2, 35, 0, 5)))
_T0 = []


def _ltext():
    if not _ltmod:
        try:
            import glob, os
            try:
                import nvidia.cublas as nc
                d = os.path.join(list(nc.__path__)[0], "lib")
            except Exception:
                d = os.path.join(os.environ.get("CUDA_HOME", "/usr/local/cuda"), "lib64")
            so = sorted(glob.glob(os.path.join(d, "libcublasLt.so*")))
            fl = ["-L" + d, "-l:" + os.path.basename(so[0]) if so else "-lcublasLt", "-Wl,-rpath," + d]
            _ltmod.append(load_inline("pf2_lt_v2", cpp_sources=PF2_LT, functions=["lt_algos", "lt_caps", "lt_add", "lt_cfg", "lt_mm"], with_cuda=True,
                                      extra_cflags=["-O2"], extra_ldflags=fl, verbose=False))
        except Exception as e:
            print("pf2 lt unavailable", e, flush=True)
            _ltmod.append(None)
    return _ltmod[0]


def _gtm(fn, fl, sink, it=9):
    """median CUDA-graph replay time of fn from a cold L2 (read-only flush)"""
    g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn(); torch.cuda.synchronize()
        with torch.cuda.graph(g):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    ts = []
    for _ in range(it):
        sink.copy_(fl.sum()); a = torch.cuda.Event(True); b = torch.cuda.Event(True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    ts.sort()
    return ts[len(ts) // 2]


def _lt_tune(st, fl, sink):
    """the shapes where cuBLASLt beat torch's default on the judge: torch vs the layout-0 heuristics and the pinned seeds"""
    ext = _ltext()
    if ext is None:
        return
    dev = st["o"].device
    t0 = time.time()
    for name, m in LT_JOBS:
        W, Wt = st[name], st[name + "_t"]
        n, K = W.shape
        x = torch.randn((m, K), device=dev).to(torch.bfloat16)
        out = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
        ref = (x @ W.T).float()
        cands = list(range(ext.lt_algos(m, n, K, 0)))
        for lay, cfg in LT_SEEDS:
            c = ext.lt_add(m, n, K, lay, list(cfg))
            if c >= 0:
                cands.append(c)
        res = []
        for c in cands:
            if ext.lt_mm(x, W, Wt, out, c) != 0:
                continue
            torch.cuda.synchronize()
            if not torch.isfinite(out).all() or float((out.float() - ref).norm() / ref.norm()) > 1e-2:
                continue
            res.append((_gtm(lambda: ext.lt_mm(x, W, Wt, out, c), fl, sink, 7), c))
        res.sort()
        tt = _gtm(lambda: torch.mm(x, W.T, out=out), fl, sink, 9)
        best = (tt, -1)
        for _, c in res[:2]:
            a = _gtm(lambda: ext.lt_mm(x, W, Wt, out, c), fl, sink, 15)
            tt = min(tt, _gtm(lambda: torch.mm(x, W.T, out=out), fl, sink, 15))
            if a < best[0]:
                best = (a, c)
        if best[1] >= 0 and best[0] > tt * 0.98:
            best = (tt, -1)
        if best[1] >= 0:
            st["lt"][(name, m)] = best[1]
        cfg = "torch" if best[1] < 0 else "%d:%s" % (best[1] // 100000, ".".join(map(str, ext.lt_cfg(m, n, K, best[1]))))
        print("pf2 lt %s m%d torch %.1fus best %s %.1fus (%d tried, %.1fs)" % (name, m, tt * 1e3, cfg, best[0] * 1e3, len(res), time.time() - t0), flush=True)
        del x, out, ref


def _dmm(x, st, name):
    i = st["lt"].get((name, x.shape[0]))
    if i is None:
        return x @ st[name].T
    out = torch.empty((x.shape[0], st[name].shape[0]), dtype=torch.bfloat16, device=x.device)
    _ltext().lt_mm(x, st[name], st[name + "_t"], out, i)
    return out


_prepare_dl, _kernel_fn_dl = prepare, kernel_fn


def _lean():
    """Hopper with FA3 and SGLang's Marlin: every prefill shape runs this file's path, so the decode-only extensions are not built"""
    return torch.cuda.get_device_capability() == (9, 0) and A__fa3() is not None and _marlin_mods() is not None


def prepare(weights):
    _T0[:] = [time.time()]
    from concurrent.futures import ThreadPoolExecutor
    lean = _lean()
    with ThreadPoolExecutor(3 + len(QV)) as ex:
        jobs = [ex.submit(_pf2)] + ([ex.submit(_ltext)] if LT_TUNE else []) + ([ex.submit(_qx, v) for v in QV if QV[v]] if MOE_Q and lean else [])
        w4j = ex.submit(_w4g) if W4 and lean else None
        qj = ex.submit(Q_prepare, {"w13": weights["w13"], "w13_sf": weights["w13_sf"], "w13_gs": weights["w13_gs"], "w2": weights["w2"],
                                   "w2_sf": weights["w2_sf"], "w2_gs": weights["w2_gs"], "post_norm_w": weights["post_expert_norm"]}) if MOE_Q else None
        if lean:
            st = {k: weights[k] for k in ("qkv_gate", "o", "router", "router_bias", "pooler", "unpooler")}
            hi = weights["router"].to(torch.bfloat16)
            st["rhi"], st["rlo"] = hi.contiguous(), (weights["router"] - hi.float()).to(torch.bfloat16).contiguous()
            st["rhl_t"] = torch.cat((st["rhi"], st["rlo"])).t().contiguous()
            st["router_bias"] = weights["router_bias"].float().contiguous()
            st["side"] = torch.cuda.Stream()
            st["a1"], st["b1"] = _coef(weights["attn_residual_gate"])
            st["a2"], st["b2"] = _coef(weights["ffn_residual_gate"])
            for k in ("pre_ffn_norm", "pre_expert_norm", "post_ffn_norm", "next_pre_attn_norm", "post_expert_norm"):
                st["g_" + k] = (1.0 + weights[k].float()).contiguous()
            for k in ("w13", "w13_sf", "w13_gs", "w2", "w2_sf", "w2_gs"):
                st["p" + k] = weights[k].contiguous()
            st["bank"] = _marlin_bank(weights, _marlin_mods())
            st["weights"] = weights
        else:
            st = _prepare_dl(weights)
        for j in jobs:
            j.result()
    dev = weights["o"].device
    st["lean"] = lean
    st["nocnt"] = torch.empty(0, dtype=torch.int32, device=dev)
    if st.get("bank") is not None:
        st["bank"]["nopos"], st["bank"]["post_wf"] = st["nocnt"], st["g_post_expert_norm"]
    for k in ("o", "pooler", "unpooler"):
        st[k + "_t"] = weights[k].t().contiguous()
    st["lt"], st["kvc"], st["mv"], st["fx"] = {}, {}, {}, {}
    st["q"] = qj.result() if qj is not None else None
    st["w4"] = None
    if W4 and lean:
        try:
            t0 = time.time()
            ext = w4j.result()
            if ext is not None:
                w13, g13 = _w4pack(weights["w13"], weights["w13_sf"], weights["w13_gs"])
                w2, g2 = _w4pack(weights["w2"], weights["w2_sf"], weights["w2_gs"])
                st["w4"] = {"ext": ext, "w13": w13, "g13": g13, "w2": w2, "g2": g2, "ctr": torch.zeros(16, dtype=torch.int32, device=dev)}
                torch.cuda.synchronize()
                print(f"pf2 w4g packed in {time.time() - t0:.1f}s, smem {ext.w4g_smem()}", flush=True)
        except Exception as e:
            print("pf2 w4g unavailable", str(e)[:300], flush=True)
    torch.cuda.synchronize()
    print(f"pf2 compiled at {time.time() - _T0[0]:.1f}s (lean {lean})", flush=True)
    try:
        import deep_gemm as _dg
        print("pf2 deep_gemm", getattr(_dg, "__version__", "?"), [n for n in dir(_dg) if "grouped" in n][:12], flush=True)
    except Exception as e:
        print("pf2 deep_gemm unavailable", str(e)[:120], flush=True)
    fl, sink = torch.empty(64 << 20, dtype=torch.float32, device=dev), torch.empty((), device=dev)
    if LT_TUNE:
        _lt_tune(st, fl, sink)
    if st["q"] is not None:
        _moe_tune(st, fl, sink)
    if KVC_TUNE and lean:
        _kvc_tune(st, fl, sink)
    del fl, sink
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"pf2 prepare done at {time.time() - _T0[0]:.1f}s", flush=True)
    return st


def _fa3_run(q8, kc, vc, pt, seq, csl, cuq, B, L, hkv, win, sm, desc, sched, kvc, fx=None):
    """FA3 on the paged cache, or (kvc) on each request's visible keys gathered into a contiguous cache (TMA path); fx: num_splits / pack_gqa"""
    ws = (win - 1, 0) if win else (-1, -1)
    if fx:
        desc, sched = {**desc, **fx}, None
    if kvc:
        maxk = (win - 1 + L) if win else pt.shape[1] * kc.shape[1]
        k2 = torch.empty((B, maxk, hkv, 64), dtype=kc.dtype, device=kc.device); v2 = torch.empty_like(k2)
        c2 = torch.empty((B,), dtype=torch.int32, device=kc.device)
        _pf2().kvgather(kc, vc, pt, seq, L, win, k2, v2, c2)
        return A__fa3()(q=q8.view(torch.float8_e4m3fn), k_cache=k2, v_cache=v2, cache_seqlens=c2, cu_seqlens_q=cuq, max_seqlen_q=L,
                        softmax_scale=sm, causal=True, window_size=ws, **desc)
    return A__fa3()(q=q8.view(torch.float8_e4m3fn), k_cache=kc, v_cache=vc, page_table=pt, cache_seqlens=csl, cu_seqlens_q=cuq, max_seqlen_q=L,
                    softmax_scale=sm, causal=True, window_size=ws, scheduler_metadata=sched, **desc)


def _kvc_tune(st, fl, sink):
    """per (batch, q_len, sliding): FA3 on the page-1 cache vs gather-to-contiguous + FA3, synthetic caches at the scored contexts"""
    dev = st["o"].device
    for B, L, ctx, win in KVC_CASES:
        try:
            n = ctx + L
            kc = torch.randn((B * n, 1, 2, 64), device=dev).to(torch.float8_e4m3fn); vc = torch.randn_like(kc, dtype=torch.float32).to(torch.float8_e4m3fn)
            pt = torch.randperm(B * n, device=dev).to(torch.int32).view(B, n)
            seq = torch.full((B,), ctx, dtype=torch.int32, device=dev); csl = seq + L
            cuq = torch.arange(0, B * L + 1, L, dtype=torch.int32, device=dev)
            q8 = torch.randn((B * L, 16, 64), device=dev).to(torch.float8_e4m3fn).view(torch.uint8)
            run = lambda k: _fa3_run(q8, kc, vc, pt, seq, csl, cuq, B, L, 2, win, 0.125, {}, None, k)
            o0, o1 = run(False), run(True)
            o0 = o0[0] if isinstance(o0, tuple) else o0; o1 = o1[0] if isinstance(o1, tuple) else o1
            err = float((o1.float() - o0.float()).norm() / o0.float().norm())
            t0, t1 = _gtm(lambda: run(False), fl, sink, 9), _gtm(lambda: run(True), fl, sink, 9)
            t0, t1 = min(t0, _gtm(lambda: run(False), fl, sink, 9)), min(t1, _gtm(lambda: run(True), fl, sink, 9))
            kv = st["kvc"][(B, L, bool(win))] = err < 1e-2 and t1 < t0 * 0.98
            print(f"pf2 kvc B{B} L{L} ctx{ctx} win{win}: paged {t0 * 1e3:.1f}us contiguous {t1 * 1e3:.1f}us err {err:.1e} -> {st['kvc'][(B, L, bool(win))]}", flush=True)
            if FA_TUNE:
                base = min(t0, t1) if kv else t0
                ref = o1 if kv else o0
                res = []
                for fx in FA_FX:
                    try:
                        o2 = _fa3_run(q8, kc, vc, pt, seq, csl, cuq, B, L, 2, win, 0.125, {}, None, kv, fx)
                        o2 = o2[0] if isinstance(o2, tuple) else o2
                        e2 = float((o2.float() - ref.float()).norm() / ref.float().norm())
                        if e2 < 1e-2:
                            f = lambda: _fa3_run(q8, kc, vc, pt, seq, csl, cuq, B, L, 2, win, 0.125, {}, None, kv, fx)
                            res.append((min(_gtm(f, fl, sink, 9), _gtm(f, fl, sink, 9)), fx))
                        del o2
                    except Exception as e:
                        print(f"pf2 fa3 {fx} failed: {str(e)[:120]}", flush=True)
                res.sort(key=lambda r: r[0])
                if res and res[0][0] < base * 0.98:
                    st["fx"][(B, L, bool(win))] = res[0][1]
                print(f"pf2 fa3 B{B} L{L} ctx{ctx} win{win} (us): default {base * 1e3:.1f} " + " ".join(f"{fx['num_splits']}{'p' if fx.get('pack_gqa') else ''} {t * 1e3:.1f}" for t, fx in res[:6])
                      + f" -> {st['fx'].get((B, L, bool(win)))}", flush=True)
            del kc, vc, pt, q8, o0, o1
        except Exception as e:
            print(f"pf2 kvc B{B} L{L} failed: {e}", flush=True)


def _attn_pf(data, cfg, qkv, B, L, T, st):
    hq, hkv = cfg["num_q_heads"], cfg["num_kv_heads"]
    dev = qkv.device
    if not (A_FA3PF and A__fa3() is not None and A__hopper() and cfg["head_dim"] == 64):
        ad = {k: data[k] for k in ("positions", "cache_loc", "k_cache", "v_cache", "k_scale", "v_scale", "page_table", "seq_lens")}
        ad.update({k: data[k] for k in ("cache_seqlens", "cu_seqlens_q", "scheduler_metadata") if k in data})
        ad["qkv"] = qkv.view(B, L, -1)
        ad["config"] = cfg
        return A_kernel_fn(ad).view(T, -1)
    g = _pf2()
    q8 = torch.empty((T, hq, 64), dtype=torch.uint8, device=dev)
    g.qkprep(qkv, data["positions"], data["cache_loc"], data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"], q8,
             hq, hkv, cfg["rope_theta"], cfg["qk_scale_factor"], A__invf(cfg["rope_theta"], dev))
    csl = data.get("cache_seqlens")
    if csl is None:
        csl = (data["seq_lens"] + L).to(torch.int32)
    cuq = data.get("cu_seqlens_q")
    if cuq is None:
        cuq = torch.arange(0, T + 1, L, dtype=torch.int32, device=dev)
    desc = {}
    if float(cfg["kv_scale"]) != 1.0:
        desc = dict(k_descale=data["k_scale"].reshape(1, 1).expand(B, hkv), v_descale=data["v_scale"].reshape(1, 1).expand(B, hkv))
    win = cfg["window"]
    o = _fa3_run(q8, data["k_cache"], data["v_cache"], data["page_table"], data["seq_lens"], csl, cuq, B, L, hkv, win,
                 cfg["sm_scale"] * cfg["qk_scale_factor"], desc, data.get("scheduler_metadata"), st["kvc"].get((B, L, bool(win)), False),
                 st.get("fx", {}).get((B, L, bool(win))))
    if isinstance(o, tuple):
        o = o[0]
    out = torch.empty((T, hq * 64), dtype=torch.bfloat16, device=dev)
    if o.dtype == torch.bfloat16 and o.is_contiguous():
        g.ongate(o, qkv, out, hq, hkv)
        return out
    x = o.float().view(T, hq, 64)
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
    gt = qkv[:, (hq + 2 * hkv) * 64:].float().view(T, hq, 64)
    return (x * torch.sigmoid(gt)).to(torch.bfloat16).view(T, hq * 64)


def _mbm(T):
    for bm in (8, 16, 32, 48, 64):
        if T * 8 / 32 / bm < 0.9:
            break
    return bm


def _marlin2(x, tw, lid, al, bm, bank, mm):
    T, K = x.shape
    N = bank["size_n"]
    sid, bexp, npad = al
    c2 = torch.empty((T * 8, N), device=x.device, dtype=x.dtype)
    c13 = torch.empty((T * 8 * max(2 * N, K),), device=x.device, dtype=x.dtype)
    c1 = c13[: T * 8 * 2 * N].view(-1, 2 * N)
    c3 = c13[: T * 8 * K].view(-1, K)
    com = dict(moe_block_size=bm, mul_topk_weights=False, is_ep=True, b_q_type=mm["fp4"], is_k_full=True, use_atomic_add=False,
               use_fp32_reduce=True, is_zp_float=False)
    ws = bank["workspace"]
    c1 = mm["gemm"](x, c1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws, sid, bexp, npad, tw, top_k=8,
                    size_m=T, size_n=2 * N, size_k=K, **com)
    mm["silu_and_mul"](c1.view(-1, 2 * N), c2, expert_ids=lid.view(-1), expert_step=1)
    y = mm["gemm"](c2, c3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws, sid, bexp, npad, tw, top_k=1,
                   size_m=T * 8, size_n=K, size_k=N, **com).view(T, 8, K)
    out = torch.empty((T, K), device=x.device, dtype=x.dtype)
    _pf2().combine(y, lid, bank["nopos"], bank["post_wf"], tw, bank["nopos"], out)
    return out


# XBULK ring builds: the producer fetches each activation row slice (512 B) with one cp.async.bulk per lane (tx-counted on the stage's
# full barrier, cross-proxy fence after the W13 -> W2 dependency wait) instead of 64 x 32-lane cp.async 16-B loops: the H100 PROF showed
# the single producer warp issue-bound (NOCOMP 344 of 558 us at T753, producer never waiting on empty slots)
_QX0 = """        if (lane == 0) expect_tx(fb, WB);
        __syncwarp();"""
_QX1 = """#pragma unroll
        for (int k = 0; k < TP / 32; k++)
          for (int rw = k * 32; rw < min(ntok, k * 32 + 32); rw++) {
            const int row = __shfl_sync(0xffffffffu, my[k], rw & 31);
            cp16(dst + WB + rw * XROW + lane * 16, xsrc + (size_t)row * (HD * 2) + i * 512 + lane * 16);
          }"""
if Q_CUDA.count(_QX0) == 1 and Q_CUDA.count(_QX1) == 1:
    Q_CUDA = Q_CUDA.replace(_QX0, """#if defined(XBULK) && TMA
        if (lane == 0) expect_tx(fb, WB + ntok * 512);
#else
        if (lane == 0) expect_tx(fb, WB);
#endif
        __syncwarp();""").replace(_QX1, """#if defined(XBULK) && TMA
        if (g && i == 0) asm volatile("fence.proxy.async.global;\\n" ::: "memory");
#pragma unroll
        for (int k = 0; k < TP / 32; k++)
          if (k * 32 + lane < ntok) bulk(dst + WB + (k * 32 + lane) * XROW, xsrc + (size_t)my[k] * (HD * 2) + i * 512, 512, fb);
#else
""" + _QX1 + """
#endif""")
else:
    print("pf2 XBULK patch did not apply", flush=True)
Q_DECL = "void moe(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,torch::Tensor);"


def _qfits(fl):
    d = {}
    for x in list(Q__cfg()) + list(fl):
        k, _, val = x[2:].partition("=")
        if val.isdigit():
            d[k] = int(val)
    need = d.get("NSLOT", 2) * (d.get("NCW", 8) * d.get("RPW", 1) * 2304 + d.get("TP", 64) * 528) + 2048
    pr = torch.cuda.get_device_properties(0)
    return need <= getattr(pr, "shared_memory_per_block_optin", 232448)


def _qx(v):
    fl = QV[v]
    if not fl:
        return Q__ext()
    if not _qfits(fl):
        return None
    key = ("pf12",) + fl
    if key not in Q__mods:
        base = [d for d in Q__cfg() if not any(x.startswith(d.split("=")[0] + "=") for x in fl)] + list(fl)
        name = "moe_w4a16_v23_" + "".join(base).replace("-", "_").replace("=", "")
        try:
            Q__mods[key] = load_inline(name, cpp_sources=Q_DECL, cuda_sources=Q_CUDA, functions=["moe"],
                                       extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"] + base, verbose=False)
        except Exception as e:
            print("pf2 ring", v, "build failed", str(e)[-300:], flush=True)
            Q__mods[key] = None
    return Q__mods[key]


def _moe_q(p, loc, tw, st, v="a"):
    """the MoE agent's prefill-scale fused W4A16 kernel (muse_moe_w4a16 v23, build v): p = normalized expert input, loc = local ids"""
    q = st["q"]
    T, dev = p.shape[0], p.device
    R = T * 8
    inter = torch.empty((R, 4096), dtype=torch.bfloat16, device=dev); y = torch.empty((R, 4096), dtype=torch.bfloat16, device=dev)
    ss = torch.empty((R,), dtype=torch.float32, device=dev)
    lst = torch.empty((R,), dtype=torch.int32, device=dev); pos = torch.empty((R,), dtype=torch.int32, device=dev)
    out = torch.empty_like(p)
    _qx(v).moe(p, loc, tw, q["w1"], q["w2"], q["g1"], q["g2"], q["nw"], inter, y, ss, q["ctr"], lst, pos, out, 0, q["prof"])
    return out


# ---------------------------------------------------------------------------------------------------------------------------------
# w4g: grouped NVFP4 x BF16 GEMM on wgmma, weights dequantized once per (256-row x <=128-token x 64-K) CTA step into SW128 smem.
#   CTA = 3 warpgroups: WG0 loads (TMA bulk of the repacked NVFP4 chunk into a staging ring, cp.async of the token rows into the
#   B ring) and dequantizes into the A ring; WG1/WG2 each run two m64 x nN x k16 wgmma blocks per k16 (A = 64 weight rows, B = tokens),
#   4-stage full/empty mbarrier ring, one commit per 64-K stage, wait_group 1. Persistent CTAs take (expert, token tile, row block)
#   tiles from an atomic queue, heaviest experts first. N per tile = the tile's tokens rounded up to 16/32/64/128 (template).
#   Epilogue: mode 1 (W13): rows interleaved in prepare (per WG: 64 gate rows then the matching 64 up rows), SwiGLU in registers,
#   writes inter [route, 4096]; mode 0 (W2): writes Y [route, 4096]. Values carry the 2^-8 of the bit-trick dequant: g = gs * 256.
W4G_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
#define HAS_WG 1
#else
#define HAS_WG 0
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define TMA 1
#else
#define TMA 0
#endif
#ifndef BNMAX
#define BNMAX 128
#endif
#ifndef NS
#define NS 4
#endif
#ifndef DQD
#define DQD 2
#endif
#define NSTG (DQD + 1)
static_assert(NS >= DQD + 2, "a slot is released only once the next stage is full: NS >= DQD + 2");
#define WCH 9216
#define ATB 32768
#define BTB (BNMAX * 128)
#define KST 64
#define NCWG 4  // consumer warpgroups, one m64 row block each (2 x m64 per WG at n128 needs > 168 registers)
struct Hdr { int tile, e, rb, row0, ntok, nsel, pad0, pad1; };
struct Sm { unsigned long long full[NS], empty[NS], wbar[NSTG]; Hdr hdr[NS]; int order[32], cum[33]; int tnext; };
struct P { const __nv_bfloat16* A; const int* offs; const uint8_t* W; const float* g; __nv_bfloat16* out; int* ctr; int mode, nrb, nout; };

__device__ __forceinline__ void mbinit(uint32_t b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(b), "r"(n)); }
__device__ __forceinline__ void mbarrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(b) : "memory"); }
__device__ __forceinline__ void cpa_arrive(uint32_t b) { asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" :: "r"(b) : "memory"); }
// watchdog: an untested-on-hardware pipeline must not hang the run. A wait that spins past WDOG polls sets g_w4err; once set, every
// wait returns at once and the CTAs drain (garbage output, rejected by prepare()'s error check; w4g_err() reports it)
#ifndef WDOG
#define WDOG (TMA ? (1u << 22) : (1u << 28))
#endif
__device__ int g_w4err;
__device__ __forceinline__ void mbwait(uint32_t b, uint32_t ph) {
  for (uint32_t i = 0;; i++) {
    uint32_t ok;
#if TMA
    asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n" : "=r"(ok) : "r"(b), "r"(ph) : "memory");
#else
    asm volatile("{\n.reg .pred p;\nmbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n" : "=r"(ok) : "r"(b), "r"(ph) : "memory");
#endif
    if (ok) return;
    if ((i & 255) == 255) {
      if (*(volatile int*)&g_w4err) return;
      if (i > WDOG) { atomicExch(&g_w4err, 1); return; }
    }
  }
}
__device__ __forceinline__ void cp16(uint32_t s, const void* g) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(g) : "memory"); }
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory"); }
__device__ __forceinline__ void fence_async() {
#if TMA
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#endif
}
#if TMA
__device__ __forceinline__ void expect_tx(uint32_t bar, uint32_t n) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(bar), "r"(n) : "memory"); }
__device__ __forceinline__ void bulk(uint32_t dst, const void* src, uint32_t n, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" :: "r"(dst), "l"(src), "r"(n), "r"(bar) : "memory");
}
#endif
__device__ __forceinline__ void wgbar() { asm volatile("bar.sync 1, 128;\n" ::: "memory"); }
__device__ __forceinline__ uint32_t swz(uint32_t a) { return a ^ (((a >> 7) & 7) << 4); }  // SW128 on the absolute smem address
__device__ __forceinline__ uint32_t bmul(uint32_t a, uint32_t b) { uint32_t d; asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(0x80008000u)); return d; }
__device__ __forceinline__ uint32_t sc2(uint32_t w, uint32_t sel) { return __byte_perm(w, 0, sel) * 16u + 0x77007700u; }
__device__ __forceinline__ void sts128(uint32_t a, uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
  asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};\n" :: "r"(a), "r"(x), "r"(y), "r"(z), "r"(w) : "memory");
}
__device__ __forceinline__ uint64_t gdesc(uint32_t saddr) {  // K-major, SW128, LBO 16 (unused), SBO 1024
  return (uint64_t)((saddr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)64 << 32) | (1ull << 62);
}
#if !HAS_WG
__device__ __forceinline__ uint32_t emu_kmaj(uint64_t desc, int r, int kb) {
  const uint32_t st = (uint32_t)(desc & 0x3FFF) << 4, sbo = (uint32_t)((desc >> 32) & 0x3FFF) << 4;
  return swz(st + (r >> 3) * sbo + (r & 7) * 128 + kb);
}
__device__ __forceinline__ float emu_ldb(uint32_t a) { unsigned short v; asm volatile("ld.shared.u16 %0, [%1];" : "=h"(v) : "r"(a)); return __uint_as_float((uint32_t)v << 16); }
template <int N>
__device__ __noinline__ void wg_emu(float* d, uint64_t da, uint64_t db, int sd) {
  const int wt = threadIdx.x & 127, w = wt >> 5, g = (wt & 31) >> 2, t = wt & 3;
  for (int i = 0; i < N / 2; i++) {
    const int hh = (i >> 1) & 1, e = i & 1, j = i >> 2, m = 16 * w + g + 8 * hh, n = 8 * j + 2 * t + e;
    float acc = 0.f;
    for (int k = 0; k < 16; k++) acc += emu_ldb(emu_kmaj(da, m, 2 * k)) * emu_ldb(emu_kmaj(db, n, 2 * k));
    d[i] = sd ? d[i] + acc : acc;
  }
}
#endif
__device__ __forceinline__ void wg_fence() {
#if HAS_WG
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void wg_commit() {
#if HAS_WG
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
#endif
}
template <int N> __device__ __forceinline__ void wg_wait() {
#if HAS_WG
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory");
#endif
}
@@WGM@@

__device__ __forceinline__ void dq_stage(uint32_t stg, uint32_t abase, int tid) {
#pragma unroll
  for (int rr = 0; rr < 2; rr++) {
    const int row = tid + 128 * rr;
    uint4 c0, c1; uint32_t sw;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(c0.x), "=r"(c0.y), "=r"(c0.z), "=r"(c0.w) : "r"(stg + row * 32));
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(c1.x), "=r"(c1.y), "=r"(c1.z), "=r"(c1.w) : "r"(stg + row * 32 + 16));
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(sw) : "r"(stg + 8192 + row * 4));
    const uint32_t qs[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
#pragma unroll
    for (int c = 0; c < 8; c++) {
      const uint32_t q = qs[c], s = sc2(sw, 0x4040u + (uint32_t)(c >> 1) * 0x101u);
      const uint32_t a0 = bmul(q & 0x81C081C0u, s);
      const uint32_t a1 = bmul(((q << 3) & 0x01C001C0u) | ((q << 1) & 0x80008000u), s);
      const uint32_t a2 = bmul((q << 6) & 0x81C081C0u, s);
      const uint32_t a3 = bmul(((q >> 5) & 0x01C001C0u) | ((q << 5) & 0x80008000u), s);
      sts128(swz(abase + row * 128 + c * 16), a0, a1, a2, a3);
    }
  }
}

template <int N>
__device__ __forceinline__ void ctile(const P& p, Sm* sm, uint32_t abase0, uint32_t bbase0, int q, const Hdr h, int wi, int tid) {
  float a0[N / 2];
  for (int ks = 0; ks < KST; ks++) {
    const int qq = q + ks, slot = qq % NS;
    if (ks) mbwait((uint32_t)__cvta_generic_to_shared(&sm->full[slot]), (qq / NS) & 1);
    const uint64_t da = gdesc(abase0 + slot * ATB + wi * 8192), db = gdesc(bbase0 + slot * BTB);
    wg_fence();
#pragma unroll
    for (int kk = 0; kk < 4; kk++) wgm<N>(a0, da + 2 * kk, db + 2 * kk, ks | kk);
    wg_commit();
    if (ks) {
      wg_wait<1>();
      if ((tid & 31) == 0) mbarrive((uint32_t)__cvta_generic_to_shared(&sm->empty[(qq - 1) % NS]));
    }
  }
  wg_wait<0>();
  if ((tid & 31) == 0) mbarrive((uint32_t)__cvta_generic_to_shared(&sm->empty[(q + KST - 1) % NS]));
  const int wt = tid & 127, w = wt >> 5, g = (wt & 31) >> 2, t4 = wt & 3;
  const float gs = p.g[h.e];
  __nv_bfloat16* ob = p.out + (size_t)(h.row0 + 2 * t4) * p.nout + h.rb * 256 + wi * 64 + 16 * w + g;
  const int nl = h.ntok - 2 * t4;
#pragma unroll
  for (int j = 0; j < N / 8; j++)
#pragma unroll
    for (int e2 = 0; e2 < 2; e2++)
      if (8 * j + e2 < nl) {
        __nv_bfloat16* o = ob + (size_t)(8 * j + e2) * p.nout;
        o[0] = __float2bfloat16(a0[4 * j + e2] * gs); o[8] = __float2bfloat16(a0[4 * j + 2 + e2] * gs);
      }
}

// NT = BNMAX: full token tiles + remainders > 32; NT = 32: remainders <= 32 (one wgmma shape per kernel: mixing shapes spills)
template <int NT>
__global__ void __launch_bounds__(128 * (NCWG + 1), 1) w4g_k(P p) {
  extern __shared__ uint8_t smraw[];
#if HAS_WG
  uint8_t* sm0 = (uint8_t*)(((uintptr_t)smraw + 1023) & ~(uintptr_t)1023);
#else
  uint8_t* sm0 = smraw;
#endif
  const uint32_t abase0 = (uint32_t)__cvta_generic_to_shared(sm0), bbase0 = abase0 + NS * ATB, sbase0 = bbase0 + NS * BTB;
  Sm* sm = (Sm*)(sm0 + NS * ATB + NS * BTB + NSTG * WCH);
  const int tid = threadIdx.x, wg = tid >> 7;
  if (tid == 0) {
    for (int i = 0; i < NS; i++) { mbinit((uint32_t)__cvta_generic_to_shared(&sm->full[i]), 128); mbinit((uint32_t)__cvta_generic_to_shared(&sm->empty[i]), 4 * NCWG); }  // one arrival per consumer warp after its wait
    for (int i = 0; i < NSTG; i++) mbinit((uint32_t)__cvta_generic_to_shared(&sm->wbar[i]), TMA ? 1 : 128);
#if TMA
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#endif
  }
  if (tid < 32) {
    const int lane = tid, end = p.offs[lane], c = end - (lane ? p.offs[lane - 1] : 0);
    int rk = 0;
    for (int j = 0; j < 32; j++) { const int cj = __shfl_sync(0xffffffffu, c, j); rk += (cj > c) || (cj == c && j < lane); }
    sm->order[rk] = lane;
    const int full = c / BNMAX, rem = c % BNMAX;
    const int nt = (NT == 32 ? (rem > 0 && rem <= 32 ? 1 : 0) : full + (rem > 32 ? 1 : 0)) * p.nrb;
    // prefix over ranks: rank r's tile count is the nt of the lane with rank r
    int ntr = 0;
    for (int j = 0; j < 32; j++) { const int rj = __shfl_sync(0xffffffffu, rk, j), nj = __shfl_sync(0xffffffffu, nt, j); if (rj == lane) ntr = nj; }
    int v = ntr;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) { const int n2 = __shfl_up_sync(0xffffffffu, v, o); if (lane >= o) v += n2; }
    sm->cum[lane + 1] = v;
    if (lane == 0) sm->cum[0] = 0;
  }
  __syncthreads();
  const int total = sm->cum[32];
  if (wg == 0) {
    int s = 0;
    for (;;) {
      if (tid == 0) sm->tnext = atomicAdd(p.ctr, 1);
      wgbar();
      const int u = sm->tnext;
      wgbar();
      if (u >= total || *(volatile int*)&g_w4err) {
        if (tid == 0 && atomicAdd(p.ctr + 1, 1) == (int)gridDim.x - 1) { p.ctr[0] = 0; p.ctr[1] = 0; __threadfence(); }
        break;
      }
      int r = 0;
      while (sm->cum[r + 1] <= u) r++;
      const int e = sm->order[r], v = u - sm->cum[r], tt = v / p.nrb, rb = v % p.nrb;
      const int st0 = e ? p.offs[e - 1] : 0, cnt = p.offs[e] - st0;
      const int t0 = NT == 32 ? cnt / BNMAX : tt;
      const int row0 = st0 + t0 * BNMAX, ntok = min(BNMAX, cnt - t0 * BNMAX), nsel = 0;
      const uint8_t* wsrc = p.W + ((size_t)(e * p.nrb + rb) * KST) * WCH;
      const __nv_bfloat16* xsrc = p.A + (size_t)row0 * 4096;
      for (int ks = 0; ks < KST; ks++, s++) {
        const int slot = s % NS, stg = s % NSTG;
        if (s >= NS) mbwait((uint32_t)__cvta_generic_to_shared(&sm->empty[slot]), ((s / NS) - 1) & 1);
        if (tid == 0) sm->hdr[slot] = Hdr{u, e, rb, row0, ntok, nsel, ks, 0};
        const uint32_t wb = (uint32_t)__cvta_generic_to_shared(&sm->wbar[stg]);
#if TMA
        if (tid == 0) { fence_async(); expect_tx(wb, WCH); bulk(sbase0 + stg * WCH, wsrc + (size_t)ks * WCH, WCH, wb); }
#else
        for (int i = tid; i < WCH / 16; i += 128) cp16(sbase0 + stg * WCH + i * 16, wsrc + (size_t)ks * WCH + i * 16);
#endif
        const uint32_t bb = bbase0 + slot * BTB;
#pragma unroll
        for (int i = 0; i < BNMAX * 8 / 128; i++) {
          const int qd = tid + 128 * i, row = qd >> 3, c = qd & 7;
          if (row < ntok) cp16(swz(bb + row * 128 + c * 16), xsrc + (size_t)row * 4096 + ks * 64 + c * 8);
        }
#if !TMA
        cpa_arrive(wb);
#endif
        cp_commit();
        if (s >= DQD) {
          const int t = s - DQD, ts = t % NS, tg = t % NSTG;
          mbwait((uint32_t)__cvta_generic_to_shared(&sm->wbar[tg]), (t / NSTG) & 1);
          dq_stage(sbase0 + tg * WCH, abase0 + ts * ATB, tid);
          cp_wait<DQD>();
          fence_async();
          mbarrive((uint32_t)__cvta_generic_to_shared(&sm->full[ts]));
          wgbar();
        }
      }
    }
    for (int t = s - DQD < 0 ? 0 : s - DQD; t < s; t++) {
      const int ts = t % NS, tg = t % NSTG;
      mbwait((uint32_t)__cvta_generic_to_shared(&sm->wbar[tg]), (t / NSTG) & 1);
      dq_stage(sbase0 + tg * WCH, abase0 + ts * ATB, tid);
      cp_wait<0>();
      fence_async();
      mbarrive((uint32_t)__cvta_generic_to_shared(&sm->full[ts]));
      wgbar();
    }
    const int slot = s % NS;
    if (s >= NS) mbwait((uint32_t)__cvta_generic_to_shared(&sm->empty[slot]), ((s / NS) - 1) & 1);
    if (tid == 0) sm->hdr[slot].tile = -1;
    mbarrive((uint32_t)__cvta_generic_to_shared(&sm->full[slot]));
  } else {
    const int wi = wg - 1;
    for (int q = 0;; q += KST) {
      const int slot = q % NS;
      mbwait((uint32_t)__cvta_generic_to_shared(&sm->full[slot]), (q / NS) & 1);
      const Hdr h = sm->hdr[slot];
      if (h.tile < 0 || *(volatile int*)&g_w4err) break;
      ctile<NT>(p, sm, abase0, bbase0, q, h, wi, tid);
    }
  }
}

static int g_nsm = 0;
static size_t smem_bytes();
static void w4g_init() {
  if (g_nsm) return;
  int dev; cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&g_nsm, cudaDevAttrMultiProcessorCount, dev);
  cudaFuncSetAttribute(w4g_k<BNMAX>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes());
  cudaFuncSetAttribute(w4g_k<32>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes());
}
int64_t w4g_err(int64_t reset) {
  int v = 0;
  cudaMemcpyFromSymbol(&v, g_w4err, sizeof(int));
  if (reset) { const int z = 0; cudaMemcpyToSymbol(g_w4err, &z, sizeof(int)); }
  return v;
}
static size_t smem_bytes() { return (size_t)NS * ATB + NS * BTB + NSTG * WCH + sizeof(Sm) + (HAS_WG_HOST ? 1024 : 0); }
void w4g(torch::Tensor A, torch::Tensor offs, torch::Tensor W, torch::Tensor g, torch::Tensor out, torch::Tensor ctr, int64_t mode, int64_t nrb) {
  w4g_init();
  P p{(const __nv_bfloat16*)A.data_ptr(), offs.data_ptr<int>(), W.data_ptr<uint8_t>(), g.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr(),
      ctr.data_ptr<int>(), (int)mode, (int)nrb, (int)out.size(1)};
  w4g_k<BNMAX><<<g_nsm, 128 * (NCWG + 1), smem_bytes(), at::cuda::getCurrentCUDAStream()>>>(p);
  p.ctr += 2;
  w4g_k<32><<<g_nsm, 128 * (NCWG + 1), smem_bytes(), at::cuda::getCurrentCUDAStream()>>>(p);
}
void w4g_part(torch::Tensor A, torch::Tensor offs, torch::Tensor W, torch::Tensor g, torch::Tensor out, torch::Tensor ctr, int64_t mode, int64_t nrb, int64_t cls) {
  w4g_init();
  P p{(const __nv_bfloat16*)A.data_ptr(), offs.data_ptr<int>(), W.data_ptr<uint8_t>(), g.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr(),
      ctr.data_ptr<int>(), (int)mode, (int)nrb, (int)out.size(1)};
  if (cls == 0) w4g_k<BNMAX><<<g_nsm, 128 * (NCWG + 1), smem_bytes(), at::cuda::getCurrentCUDAStream()>>>(p);
  else w4g_k<32><<<g_nsm, 128 * (NCWG + 1), smem_bytes(), at::cuda::getCurrentCUDAStream()>>>(p);
}
int64_t w4g_smem() { return (int64_t)smem_bytes(); }
"""


def _w4g_wgm():
    out = []
    for N in (32, 64, 128):
        n = N // 2
        regs = ",".join(f"%{i}" for i in range(n))
        cons = ", ".join(f'"+f"(d[{i}])' for i in range(n))
        out.append(f'''template <> __device__ __forceinline__ void wgm<{N}>(float* d, uint64_t da, uint64_t db, int sd) {{
#if HAS_WG
  asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, %{n + 2}, 0;\\nwgmma.mma_async.sync.aligned.m64n{N}k16.f32.bf16.bf16 {{{regs}}}, %{n}, %{n + 1}, p, 1, 1, 0, 0;\\n}}\\n"
               : {cons} : "l"(da), "l"(db), "r"(sd));
#else
  wg_emu<{N}>(d, da, db, sd);
#endif
}}''')
    return "template <int N> __device__ __forceinline__ void wgm(float* d, uint64_t da, uint64_t db, int sd);\n" + "\n".join(out)


W4G_CUDA = W4G_CUDA.replace("@@WGM@@", _w4g_wgm())
_w4gmod = []


def _w4g():
    if not _w4gmod:
        hop = torch.cuda.get_device_capability() == (9, 0)
        fl = (["-gencode=arch=compute_90a,code=sm_90a", "-DHAS_WG_HOST=1"] if hop else ["-DBNMAX=64", "-DNS=2", "-DDQD=0", "-DHAS_WG_HOST=0"])
        try:
            _w4gmod.append(load_inline("pf2_w4g_v4" + ("h" if hop else "e"), cpp_sources="void w4g(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t); void w4g_part(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t); int64_t w4g_smem(); int64_t w4g_err(int64_t);",
                                       cuda_sources=W4G_CUDA, functions=["w4g", "w4g_part", "w4g_smem", "w4g_err"], extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"] + fl, verbose=False))
        except Exception as e:
            print("pf2 w4g build failed", str(e)[-1500:], flush=True)
            _w4gmod.append(None)
    return _w4gmod[0]


def _w4pack(w, sf, gs):
    """NVFP4 [E, N, K/2] + e4m3 scales [E, N, K/16] -> [E, N/256, 64, 9216] chunks: per (row block, 64-K step) 256 rows x 8 words of 8
    codes in the ring's bit-rotated layout (one bf16x2 pair per mask) + 256 rows x 4 scale bytes; natural row order"""
    E, N, K = w.shape[0], w.shape[1], w.shape[2] * 2
    nrb, dev = N // 256, w.device
    perm = torch.arange(N, device=dev)
    out = torch.empty((E, nrb, 64, 9216), dtype=torch.uint8, device=dev)
    for e in range(E):
        codes = torch.stack((w[e] & 15, w[e] >> 4), -1).view(N, K)[perm].to(torch.int64)
        s8 = sf[e].view(torch.uint8)[perm].to(torch.int64)
        codes = codes ^ ((s8 >> 7).repeat_interleave(16, 1) << 3)
        zero = ((s8 & 0x7F) == 0)
        codes = torch.where(zero.repeat_interleave(16, 1), torch.zeros_like(codes), codes)
        s8 = torch.where(zero, torch.full_like(s8, 8), s8 & 0x7F)
        if ((s8 >> 3) == 0).any():
            raise ValueError("subnormal E4M3 block scales: w4g fast path unsupported")
        C = codes.view(nrb, 256, 64, 8, 8)
        word = torch.zeros(C.shape[:-1], dtype=torch.int64, device=dev)
        for pr in range(4):
            for hi in range(2):
                word = Q__place(word, C[..., 2 * pr + hi], pr, hi)
        word = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32)
        cb = word.permute(0, 2, 1, 3).contiguous().view(torch.uint8).view(nrb, 64, 8192)
        sb = s8.to(torch.uint8).view(nrb, 256, 64, 4).permute(0, 2, 1, 3).contiguous().view(nrb, 64, 1024)
        out[e] = torch.cat((cb, sb), -1)
    return out, (gs.float() * 256.0).contiguous()


def _w4gemm(A, offs, Wp, g, mode, ctr):
    nrb = Wp.shape[1]
    out = torch.empty((A.shape[0], nrb * 256), dtype=torch.bfloat16, device=A.device)
    _w4g().w4g(A, offs, Wp, g, out, ctr, mode, nrb)
    return out


# MoE paths (kind, thr, ring build, overlap): "mar" SGLang's Marlin, "q" fused ring for all experts, "big" dequant + grouped GEMM for all,
# "h" hybrid (experts with > thr routes on big, the rest on the ring, whose output big's combine adds); overlap: w2 dequant on the side
# stream under the first grouped GEMM
def _moe_path(st, T):
    if st["mv"]:
        return st["mv"][min(st["mv"], key=lambda m: abs(m - T))]
    return ("mar", 0, "a", 0) if MARLIN_MIN < T <= MARLIN_MAX and st.get("bank") is not None else ("big", 0, "a", 0)


def _moe_pre(path, loc, lid, cnt, T, st):
    """routing-side prep (side stream in kernel_fn): Marlin's align, or the grouped lists of the big part"""
    kind, thr = path[0], path[1]
    dev = loc.device
    if kind == "mar":
        mm = _marlin_mods()
        bm = MAR_BM.get(T, _mbm(T))
        return {"al": mm["align"](lid, bm, 32, ignore_invalid_expert=True) if ALIGN_LOCAL else mm["align"](lid, bm, 256), "bm": bm}
    if kind == "q":
        return {}
    g = _pf2()
    d = {}
    if kind in ("h", "h4"):
        d["locq"], d["locb"], d["cntb"] = torch.empty_like(loc), torch.empty_like(loc), torch.empty_like(cnt)
        g.msplit(loc, cnt, thr, d["locq"], d["locb"], d["cntb"])
    else:
        d["locb"], d["cntb"] = loc, cnt
    d["offs"], d["fill"] = torch.empty_like(cnt), torch.empty_like(cnt)
    d["lst"] = torch.empty(T * 8, dtype=torch.int32, device=dev); d["pos"] = torch.empty_like(d["lst"])
    g.rlist(d["locb"], d["cntb"], d["offs"], d["fill"], d["lst"], d["pos"])
    return d


def _moe_main(path, pr, loc, lid, tw, d, st, wait=None):
    kind, v, ovl = path[0], path[2], path[3]
    g = _pf2()
    if kind in ("mar", "q", "h", "h4"):
        p = torch.empty_like(pr)
        g.rmsgain2(pr, st["g_pre_expert_norm"], p)
    if wait is not None:
        wait()
    if kind == "mar":
        return _marlin2(p, tw, lid, d["al"], d["bm"], st["bank"], _marlin_mods())
    if kind == "q":
        return _moe_q(p, loc, tw, st, v)
    if kind == "h" and ovl == 2:
        dev = pr.device
        w13 = torch.empty((32, 8192, 4096), dtype=torch.bfloat16, device=dev); w2 = torch.empty((32, 4096, 4096), dtype=torch.bfloat16, device=dev)
        cur, side = torch.cuda.current_stream(), st["side"]
        side.wait_stream(cur)
        with torch.cuda.stream(side):
            g.dequant3(st["pw13"], st["pw13_sf"], st["pw13_gs"], d["cntb"], w13)
            g.dequant3(st["pw2"], st["pw2_sf"], st["pw2_gs"], d["cntb"], w2)
        add = _moe_q(p, d["locq"], tw, st, v)
        cur.wait_stream(side)
        return _moe_big(g, pr, d["locb"], tw, d["cntb"], d["offs"], d["lst"], d["pos"], st, add, 0, (w13, w2))
    if kind in ("w4", "h4"):
        add = _moe_q(p, d["locq"], tw, st, v) if kind == "h4" else st["nocnt"]
        return _moe_w4(g, pr, d["locb"], tw, d["offs"], d["lst"], d["pos"], st, add)
    add = _moe_q(p, d["locq"], tw, st, v) if kind == "h" else st["nocnt"]
    return _moe_big(g, pr, d["locb"], tw, d["cntb"], d["offs"], d["lst"], d["pos"], st, add, ovl)


def _moe_w4(g, pr, loc, tw, offs, lst, pos, st, add, parts=None):
    """gathered rms-normed rows -> w4g W13 (H, gate|up) -> SwiGLU -> w4g W2 -> post-norm weighted combine (+ add)"""
    T, dev = pr.shape[0], pr.device
    R = T * 8
    w = st["w4"]
    A = torch.empty((R, 4096), dtype=torch.bfloat16, device=dev)
    g.gathnorm(pr, lst, offs, st["g_pre_expert_norm"], A)
    if parts is not None:
        parts.append(("gnorm", A))
    Hh = _w4gemm(A, offs, w["w13"], w["g13"], 0, w["ctr"][0:4])
    if parts is not None:
        parts.append(("w13", Hh))
    del A
    I = torch.empty((R, 4096), dtype=torch.bfloat16, device=dev)
    g.act2(Hh, offs, I)
    del Hh
    Y = _w4gemm(I, offs, w["w2"], w["g2"], 0, w["ctr"][4:8])
    del I
    y = torch.empty((T, 4096), dtype=torch.bfloat16, device=dev)
    g.combine(Y, loc, pos, st["g_post_expert_norm"], tw, add, y)
    return y


def _w4_diag(st, fl, sink, T, prof, dev):
    """pf14diag: per-component times of the w4 path on one scored profile"""
    g = _pf2()
    pr = torch.randn((T, 4096), device=dev).to(torch.bfloat16)
    loc, lid, tw, cnt = _prof_routes(T, prof, dev, 11)
    d = _moe_pre(("w4", 0, "a", 0), loc, lid, cnt, T, st)
    w = st["w4"]
    R = T * 8
    A = torch.empty((R, 4096), dtype=torch.bfloat16, device=dev); g.gathnorm(pr, d["lst"], d["offs"], st["g_pre_expert_norm"], A)
    Hh = _w4gemm(A, d["offs"], w["w13"], w["g13"], 0, w["ctr"][0:4])
    I = torch.empty((R, 4096), dtype=torch.bfloat16, device=dev); g.act2(Hh, d["offs"], I)
    Y = _w4gemm(I, d["offs"], w["w2"], w["g2"], 0, w["ctr"][4:8])
    y = torch.empty((T, 4096), dtype=torch.bfloat16, device=dev)
    nr = int(cnt.sum())
    tm = {"rlist": lambda: _moe_pre(("w4", 0, "a", 0), loc, lid, cnt, T, st),
          "gnorm": lambda: g.gathnorm(pr, d["lst"], d["offs"], st["g_pre_expert_norm"], A),
          "w13": lambda: w["ext"].w4g(A, d["offs"], w["w13"], w["g13"], Hh, w["ctr"][0:4], 0, w["w13"].shape[1]),
          "w13big": lambda: w["ext"].w4g_part(A, d["offs"], w["w13"], w["g13"], Hh, w["ctr"][8:12], 0, w["w13"].shape[1], 0),
          "w13small": lambda: w["ext"].w4g_part(A, d["offs"], w["w13"], w["g13"], Hh, w["ctr"][8:12], 0, w["w13"].shape[1], 1),
          "act2": lambda: g.act2(Hh, d["offs"], I),
          "w2": lambda: w["ext"].w4g(I, d["offs"], w["w2"], w["g2"], Y, w["ctr"][4:8], 0, w["w2"].shape[1]),
          "comb": lambda: g.combine(Y, loc, d["pos"], st["g_post_expert_norm"], tw, st["nocnt"], y),
          "total": lambda: _moe_main(("w4", 0, "a", 0), pr, loc, lid, tw, d, st)}
    out = []
    for k, f in tm.items():
        try:
            t = min(_gtm(f, fl, sink, 9), _gtm(f, fl, sink, 9))
            extra = ""
            if k in ("w13", "w2", "w13big"):
                fl_ = nr * 4096 * (8192 if k != "w2" else 4096) * 2
                extra = f" ({fl_ / t / 1e9:.0f} TF)"
            out.append(f"{k} {t * 1e3:.1f}{extra}")
        except Exception as e:
            out.append(f"{k} failed {str(e)[:80]}")
    ne = int((cnt > 0).sum())
    print(f"pf2 w4diag T{T} routes {nr} experts {ne} (us): " + ", ".join(out), flush=True)


def _prof_routes(T, prof, dev, seed):
    """synthetic routes with a measured per-expert local count profile (scaled to T): each expert takes distinct random tokens"""
    gen = torch.Generator().manual_seed(seed)
    loc = torch.full((T, 8), 255, dtype=torch.int32)
    fill = torch.zeros(T, dtype=torch.int64)
    tref = min(MOE_PROF, key=lambda m: abs(m - T))
    for c, e in zip(prof, torch.randperm(32, generator=gen).tolist()):
        free = torch.nonzero(fill < 8).squeeze(1)
        tok = free[torch.randperm(free.numel(), generator=gen)[: min(free.numel(), max(1, round(c * T / tref)))]]
        loc[tok, fill[tok]] = e
        fill[tok] += 1
    ar = torch.arange(T)
    for k in range(8):
        m = loc[:, k] == 255
        loc[m, k] = (32 + 28 * k + ar[m] % 28).int()
    perm = torch.argsort(torch.rand((T, 8), generator=gen), dim=1)
    loc = torch.gather(loc, 1, perm).contiguous().to(dev)
    lid = torch.where(loc < 32, loc, torch.full_like(loc, -1))
    tw = torch.rand((T, 8), generator=gen).to(dev)
    cnt = torch.bincount(loc[loc < 32].long(), minlength=32).to(torch.int32)
    return loc, lid, (tw / tw.sum(1, keepdim=True)).contiguous(), cnt


W4 = True  # fused NVFP4 wgmma grouped GEMM (w4g) paths "w4" / "h4", timed in prepare() (Hopper only)
W4DIAG = False  # pf14diag: per-component w4 timings in prepare()
W4THR = (32, 96, 192)  # h4: experts with more routes than this on w4g, the rest on the ring
DEQ3 = True  # coalesced-store dequant (deq3) for the big path
MOE_DFLT = {753: ("q", 0, "a", 0), 3012: ("q", 0, "a", 0), 16384: ("big", 0, "a", 0)}  # pf11's judge picks


def _mname(k):
    return k[0] + (str(k[1]) if k[0] in ("h", "h4") else "") + (k[2] if k[0] in ("q", "h", "h4") else "") + ("", "o", "d")[k[3]]


def _moe_tune(st, fl, sink):
    """per scored size, on routes with the scored per-expert load profile: ring builds, big (with/without the w2-dequant overlap), Marlin
    (turn sizes), then hybrid splits with the best ring build; default (pf11's pick) kept unless >3% faster"""
    dev = st["o"].device
    vs = [v for v in QV if _qx(v) is not None]
    hasmar = st.get("bank") is not None and _marlin_mods() is not None
    for T in MOE_T:
        try:
            pr = torch.randn((T, 4096), device=dev).to(torch.bfloat16)
            rs = [_prof_routes(T, pf, dev, 5 + i) for i, pf in enumerate(MOE_PROF[T])]
            dflt = MOE_DFLT.get(T, ("big", 0, "a", 0))
            refs = [_moe_main(dflt, pr, loc, lid, tw, _moe_pre(dflt, loc, lid, cnt, T, st), st) for loc, lid, tw, cnt in rs]
            res = {}

            def run(path):
                if path in res:
                    return
                try:
                    tot, err = 0.0, 0.0
                    for (loc, lid, tw, cnt), ref in zip(rs, refs):
                        d = _moe_pre(path, loc, lid, cnt, T, st)
                        f = lambda: _moe_main(path, pr, loc, lid, tw, d, st)
                        out = f()
                        torch.cuda.synchronize()
                        err = max(err, float((out.float() - ref.float()).norm() / ref.float().norm()))
                        tot += min(_gtm(f, fl, sink, 7), _gtm(f, fl, sink, 7))
                        del out, d
                    res[path] = (tot / len(rs) if err < 2e-2 else 1e9, err)
                except Exception as e:
                    print(f"pf2 moe T{T} {path} failed: {str(e)[-200:]}", flush=True)
                    res[path] = (1e9, 1.0)
            run(dflt)
            if hasmar and MARLIN_MIN < T <= MARLIN_MAX:
                run(("mar", 0, "a", 0))
            for v in vs:
                run(("q", 0, v, 0))
            for o in (0, 1):
                run(("big", 0, "a", o))
            bv = min(vs, key=lambda v: res[("q", 0, v, 0)][0])
            bo = min((0, 1), key=lambda o: res[("big", 0, "a", o)][0])
            for thr in HTHR:
                run(("h", thr, bv, bo))
            bt = min(HTHR, key=lambda t: res[("h", t, bv, bo)][0])
            run(("h", bt, bv, 2))
            if st.get("w4") is not None:
                run(("w4", 0, "a", 0))
                for thr in W4THR:
                    run(("h4", thr, bv, 0))
                if W4DIAG:
                    for pf in MOE_PROF[T][:1]:
                        _w4_diag(st, fl, sink, T, pf, dev)
                if st["w4"]["ext"].w4g_err(1):
                    print(f"pf2 w4g watchdog fired at T{T}: w4 paths disabled", flush=True)
                    for k in res:
                        if k[0] in ("w4", "h4"):
                            res[k] = (1e9, 1.0)
                    st["w4"] = None
            best = min(res, key=lambda k: res[k][0])
            st["mv"][T] = best if res[best][0] < res[dflt][0] * 0.97 else dflt
            print(f"pf2 moe T{T} (us, err): " + " ".join(f"{_mname(k)} {v[0] * 1e3:.0f} {v[1]:.0e}" for k, v in sorted(res.items(), key=lambda kv: kv[1][0]))
                  + f" -> {_mname(st['mv'][T])}", flush=True)
            del pr, rs, refs
        except Exception as e:
            print(f"pf2 moe T{T} failed: {e}", flush=True)
        torch.cuda.empty_cache()


def _moe_big(g, pr, loc, tw, cnt, offs, lst, pos, st, add, ovl=0, pre=None):
    T, dev = pr.shape[0], pr.device
    R = T * 8
    dq = g.dequant3 if DEQ3 else g.dequant2
    if pre is None:
        w13 = torch.empty((32, 8192, 4096), dtype=torch.bfloat16, device=dev)
        w2 = torch.empty((32, 4096, 4096), dtype=torch.bfloat16, device=dev)
        dq(st["pw13"], st["pw13_sf"], st["pw13_gs"], cnt, w13)
    else:
        w13, w2 = pre
    cur = torch.cuda.current_stream()
    if ovl:
        side = st["side"]
        side.wait_stream(cur)
        with torch.cuda.stream(side):
            dq(st["pw2"], st["pw2_sf"], st["pw2_gs"], cnt, w2)
    A = torch.empty((R, 4096), dtype=torch.bfloat16, device=dev)
    g.gathnorm(pr, lst, offs, st["g_pre_expert_norm"], A)
    Hh = _gmm(A, w13, offs)
    del A, w13
    if ovl:
        cur.wait_stream(side)
    elif pre is None:
        dq(st["pw2"], st["pw2_sf"], st["pw2_gs"], cnt, w2)
    I = torch.empty((R, 4096), dtype=torch.bfloat16, device=dev)
    g.act2(Hh, offs, I)
    del Hh
    Y = _gmm(I, w2, offs)
    del I, w2
    y = torch.empty((T, 4096), dtype=torch.bfloat16, device=dev)
    g.combine(Y, loc, pos, st["g_post_expert_norm"], tw, add, y)
    return y


def kernel_fn(data, st):
    cfg = data["config"]
    B, L, T = cfg["batch"], cfg["q_len"], cfg["tokens"]
    if T <= 256 or cfg["num_q_heads"] != 16 or cfg["num_kv_heads"] != 2:
        if st.get("lean"):
            if "dl" not in st:
                st["dl"] = _prepare_dl(st["weights"])
            return _kernel_fn_dl(data, st["dl"])
        return _kernel_fn_dl(data, st)
    x = data["attn_input"]
    dev = x.device
    g = _pf2()
    qkv = _dmm(x, st, "qkv_gate")
    att = _attn_pf(data, cfg, qkv, B, L, T, st)
    a = _dmm(att, st, "o")
    h = torch.empty((T, 8192), dtype=torch.bfloat16, device=dev)
    ra = torch.empty((T,), dtype=torch.float32, device=dev)
    g.res_a(a, data["stream"], st["a1"], st["b1"], st["g_pre_ffn_norm"], h, ra)
    path = _moe_path(st, T)
    ids = torch.empty((T, 8), dtype=torch.int32, device=dev)
    loc, lid = torch.empty_like(ids), torch.empty_like(ids)
    tw = torch.empty((T, 8), dtype=torch.float32, device=dev)
    cnt = torch.empty(32, dtype=torch.int32, device=dev) if path[0] in ("big", "h", "w4", "h4") else st["nocnt"]
    cur, side = torch.cuda.current_stream(), st["side"]
    side.wait_stream(cur)
    pr = _dmm(h, st, "pooler")
    with torch.cuda.stream(side):
        lg = _mm32(h, st["rhl_t"])
        g.routing2(lg, st["router_bias"], data["local_start"], ids, loc, lid, tw, cnt)
        d = _moe_pre(path, loc, lid, cnt, T, st)
        if not torch.cuda.is_current_stream_capturing():
            for t_ in list(d.get("al", ())) + [v for v in d.values() if isinstance(v, torch.Tensor)] + [lg, ids, loc, lid, tw, cnt]:
                t_.record_stream(cur)
    y = _moe_main(path, pr, loc, lid, tw, d, st, wait=lambda: cur.wait_stream(side))
    f = _dmm(y, st, "unpooler")
    s2 = torch.empty((T, 8192), dtype=torch.float32, device=dev)
    hid = torch.empty_like(h)
    g.res_b(f, st["g_post_ffn_norm"], a, ra, data["stream"], st["a1"], st["b1"], st["a2"], st["b2"], st["g_next_pre_attn_norm"], s2, hid)
    return hid, s2, ids
