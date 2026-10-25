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
__device__ __forceinline__ void f8u0(uint4 x, uint4 y, float* v) {
  v[0] = __uint_as_float(x.x); v[1] = __uint_as_float(x.y); v[2] = __uint_as_float(x.z); v[3] = __uint_as_float(x.w);
  v[4] = __uint_as_float(y.x); v[5] = __uint_as_float(y.y); v[6] = __uint_as_float(y.z); v[7] = __uint_as_float(y.w);
}
// rmsg2 on an fp32 [T, ld] GEMM output (pooler columns 0-4095), rounded to bf16 first as a bf16 GEMM would
__global__ void __launch_bounds__(512) rmsg2f(const float* x, int ld, const float* g, bf* out) {
  __shared__ float red[16];
  const int r = blockIdx.x, c0 = threadIdx.x * 8;
  float v[8], gv[8], ss = 0.f;
  f8u0(__ldcs((const uint4*)(x + (size_t)r * ld + c0)), __ldcs((const uint4*)(x + (size_t)r * ld + c0 + 4)), v);
  ld8f(g + c0, gv);
#pragma unroll
  for (int i = 0; i < 8; i++) { v[i] = __bfloat162float(__float2bfloat16(v[i])); ss += v[i] * v[i]; }
  const float rr = rsqrtf(bsum<16>(ss, red) / 4096 + 1e-5f);
#pragma unroll
  for (int i = 0; i < 8; i++) v[i] = v[i] * rr * gv[i];
  *(uint4*)(out + (size_t)r * 4096 + c0) = pk8(v);
}
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
__global__ void __launch_bounds__(256) comb(const bf* Y, const int* loc, const int* pos, const float* gw, const float* tw, bf* out) {
  __shared__ float red[2][8];
  const int t = blockIdx.x, c0 = threadIdx.x * 8, c1 = c0 + 2048, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  unsigned msk = 0;
#pragma unroll
  for (int k = 0; k < 8; k++) msk |= ((unsigned)loc[t * 8 + k] < 32u ? 1u : 0u) << k;
  float acc[16];
#pragma unroll
  for (int j = 0; j < 16; j++) acc[j] = 0.f;
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
void dequant2(torch::Tensor w, torch::Tensor sf, torch::Tensor gs, torch::Tensor cnt, torch::Tensor out) {
  const long long per_e = w.numel() * 2 / 32;
  const int nbe = (int)(per_e / 16), n2 = (int)(w.numel() / 16);
  deq2<<<(n2 + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>((const uint4*)w.data_ptr(), (const uint16_t*)sf.data_ptr(), gs.data_ptr<float>(),
      cnt.data_ptr<int>(), (uint4*)out.data_ptr(), nbe, n2);
}
// ---- persistent cp.async ring variants: each thread prefetches exactly the 16-byte chunks it later reads (no barrier for the ring)
__device__ __forceinline__ void cpa16(const void* sdst, const void* g) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"((uint32_t)__cvta_generic_to_shared(sdst)), "l"(g));
}
__device__ __forceinline__ void cpcommit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cpwait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory"); }
__device__ __forceinline__ void f8u(uint4 x, uint4 y, float* v) {
  v[0] = __uint_as_float(x.x); v[1] = __uint_as_float(x.y); v[2] = __uint_as_float(x.z); v[3] = __uint_as_float(x.w);
  v[4] = __uint_as_float(y.x); v[5] = __uint_as_float(y.y); v[6] = __uint_as_float(y.z); v[7] = __uint_as_float(y.w);
}
__device__ __forceinline__ void u8f(const float* v, uint4& x, uint4& y) {
  x = make_uint4(__float_as_uint(v[0]), __float_as_uint(v[1]), __float_as_uint(v[2]), __float_as_uint(v[3]));
  y = make_uint4(__float_as_uint(v[4]), __float_as_uint(v[5]), __float_as_uint(v[6]), __float_as_uint(v[7]));
}
// boundary 1, ring of S rows (a 16 KB + s 32 KB per stage)
template <int NT, int S, int W = HD>
__global__ void __launch_bounds__(NT, 1) resar(const bf* a, const float* s, const float* al, const float* be, const float* gn, bf* h, float* ra, int T) {
  extern __shared__ uint4 sm[];
  __shared__ float red[NT / 32];
  constexpr int NC = W / 8 / NT;
  const int G = gridDim.x, tid = threadIdx.x;
  auto issue = [&](int i) {
    const int r = blockIdx.x + i * G;
    if (r < T) {
      uint4* b = sm + (i % S) * (3 * W / 8);
      const uint4* ag = (const uint4*)(a + (size_t)r * W); const uint4* sg = (const uint4*)(s + (size_t)r * W);
#pragma unroll
      for (int j = 0; j < NC; j++) { const int q = j * NT + tid; cpa16(b + q, ag + q); cpa16(b + W / 8 + 2 * q, sg + 2 * q); cpa16(b + W / 8 + 1 + 2 * q, sg + 2 * q + 1); }
    }
    cpcommit();
  };
  const int nr = blockIdx.x < T ? (T - blockIdx.x + G - 1) / G : 0;
#pragma unroll
  for (int i = 0; i < S - 1; i++) issue(i);
  for (int i = 0; i < nr; i++) {
    cpwait<S - 2>();
    uint4* b = sm + (i % S) * (3 * W / 8);
    const int r = blockIdx.x + i * G;
    float x[8], v[8], t0[8], t1[8], ss = 0.f;
#pragma unroll
    for (int j = 0; j < NC; j++) { unp8(b[j * NT + tid], x);
#pragma unroll
      for (int k = 0; k < 8; k++) ss += x[k] * x[k]; }
    const float rr = rsqrtf(bsum<NT / 32>(ss, red) / W + 1e-8f);
    float s2 = 0.f;
#pragma unroll
    for (int j = 0; j < NC; j++) {
      const int q = j * NT + tid;
      unp8(b[q], x); f8u(b[W / 8 + 2 * q], b[W / 8 + 1 + 2 * q], v); ld8f(al + 8 * q, t0); ld8f(be + 8 * q, t1);
#pragma unroll
      for (int k = 0; k < 8; k++) { v[k] = t0[k] * v[k] + t1[k] * (x[k] * rr); s2 += v[k] * v[k]; }
      u8f(v, b[W / 8 + 2 * q], b[W / 8 + 1 + 2 * q]);
    }
    const float rs = rsqrtf(bsum<NT / 32>(s2, red) / W + 1e-5f);
#pragma unroll
    for (int j = 0; j < NC; j++) {
      const int q = j * NT + tid;
      f8u(b[W / 8 + 2 * q], b[W / 8 + 1 + 2 * q], v); ld8f(gn + 8 * q, t0);
#pragma unroll
      for (int k = 0; k < 8; k++) t0[k] *= v[k] * rs;
      *(uint4*)(h + (size_t)r * W + 8 * q) = pk8(t0);
    }
    if (tid == 0) ra[r] = rr;
    issue(i + S - 1);
  }
  cpwait<0>();
}
// boundary 2, ring of S rows (f 16 KB + a 16 KB + s 32 KB per stage)
template <int NT, int S, int W = HD>
__global__ void __launch_bounds__(NT, 1) resbr(const bf* f, const float* gf, const bf* a, const float* ra, const float* s, const float* al1, const float* be1,
                                               const float* al2, const float* be2, const float* gn, float* so, bf* hid, int T) {
  extern __shared__ uint4 sm[];
  __shared__ float red[NT / 32];
  constexpr int NC = W / 8 / NT;
  const int G = gridDim.x, tid = threadIdx.x;
  auto issue = [&](int i) {
    const int r = blockIdx.x + i * G;
    if (r < T) {
      uint4* b = sm + (i % S) * (W / 2);
      const uint4* fg = (const uint4*)(f + (size_t)r * W); const uint4* ag = (const uint4*)(a + (size_t)r * W); const uint4* sg = (const uint4*)(s + (size_t)r * W);
#pragma unroll
      for (int j = 0; j < NC; j++) {
        const int q = j * NT + tid;
        cpa16(b + q, fg + q); cpa16(b + W / 8 + q, ag + q); cpa16(b + W / 4 + 2 * q, sg + 2 * q); cpa16(b + W / 4 + 1 + 2 * q, sg + 2 * q + 1);
      }
    }
    cpcommit();
  };
  const int nr = blockIdx.x < T ? (T - blockIdx.x + G - 1) / G : 0;
#pragma unroll
  for (int i = 0; i < S - 1; i++) issue(i);
  for (int i = 0; i < nr; i++) {
    cpwait<S - 2>();
    uint4* b = sm + (i % S) * (W / 2);
    const int r = blockIdx.x + i * G;
    const float rr = ra[r];
    float x[8], v[8], t0[8], t1[8], ss = 0.f;
#pragma unroll
    for (int j = 0; j < NC; j++) {
      const int q = j * NT + tid;
      unp8(b[W / 8 + q], x); f8u(b[W / 4 + 2 * q], b[W / 4 + 1 + 2 * q], v); ld8f(al1 + 8 * q, t0); ld8f(be1 + 8 * q, t1);
#pragma unroll
      for (int k = 0; k < 8; k++) v[k] = t0[k] * v[k] + t1[k] * (x[k] * rr);
      u8f(v, b[W / 4 + 2 * q], b[W / 4 + 1 + 2 * q]);
      unp8(b[q], x); ld8f(gf + 8 * q, t0);
#pragma unroll
      for (int k = 0; k < 8; k++) { const float z = x[k] * t0[k]; ss += z * z; }
    }
    const float rf = rsqrtf(bsum<NT / 32>(ss, red) / W + 1e-8f);
    float s2 = 0.f;
#pragma unroll
    for (int j = 0; j < NC; j++) {
      const int q = j * NT + tid;
      unp8(b[q], x); ld8f(gf + 8 * q, t0); f8u(b[W / 4 + 2 * q], b[W / 4 + 1 + 2 * q], v);
#pragma unroll
      for (int k = 0; k < 8; k++) x[k] *= t0[k];
      ld8f(al2 + 8 * q, t0); ld8f(be2 + 8 * q, t1);
#pragma unroll
      for (int k = 0; k < 8; k++) { v[k] = t0[k] * v[k] + t1[k] * (x[k] * rf); s2 += v[k] * v[k]; }
      u8f(v, b[W / 4 + 2 * q], b[W / 4 + 1 + 2 * q]);
    }
    const float rs = rsqrtf(bsum<NT / 32>(s2, red) / W + 1e-5f);
#pragma unroll
    for (int j = 0; j < NC; j++) {
      const int q = j * NT + tid;
      f8u(b[W / 4 + 2 * q], b[W / 4 + 1 + 2 * q], v);
      float4* sp = (float4*)(so + (size_t)r * W + 8 * q);
      __stcs(sp, make_float4(v[0], v[1], v[2], v[3])); __stcs(sp + 1, make_float4(v[4], v[5], v[6], v[7]));
      ld8f(gn + 8 * q, t0);
#pragma unroll
      for (int k = 0; k < 8; k++) t0[k] *= v[k] * rs;
      __stcs((uint4*)(hid + (size_t)r * W + 8 * q), pk8(t0));
    }
    issue(i + S - 1);
  }
  cpwait<0>();
}
// rmsg + gather, ring of S rows (8 KB each), persistent over grouped rows
template <int S>
__global__ void __launch_bounds__(256) gnormr(const bf* x, const int* list, const int* offs, const float* g, bf* A) {
  extern __shared__ uint4 sm[];
  __shared__ float red[8];
  const int G = gridDim.x, tid = threadIdx.x, tot = offs[31];
  auto issue = [&](int i) {
    const int r = blockIdx.x + i * G;
    if (r < tot) {
      const uint4* src = (const uint4*)(x + (size_t)(list[r] >> 3) * 4096);
      uint4* d = sm + (i % S) * 512;
      cpa16(d + tid, src + tid); cpa16(d + 256 + tid, src + 256 + tid);
    }
    cpcommit();
  };
  const int nr = blockIdx.x < tot ? (tot - blockIdx.x + G - 1) / G : 0;
#pragma unroll
  for (int i = 0; i < S - 1; i++) issue(i);
  for (int i = 0; i < nr; i++) {
    cpwait<S - 2>();
    const uint4* d = sm + (i % S) * 512;
    const int r = blockIdx.x + i * G;
    float v[16], gv[8], ss = 0.f;
    unp8(d[tid], v); unp8(d[256 + tid], v + 8);
#pragma unroll
    for (int k = 0; k < 16; k++) ss += v[k] * v[k];
    const float rr = rsqrtf(bsum<8>(ss, red) / 4096 + 1e-5f);
    ld8f(g + 8 * tid, gv);
#pragma unroll
    for (int k = 0; k < 8; k++) v[k] = v[k] * rr * gv[k];
    ld8f(g + 2048 + 8 * tid, gv);
#pragma unroll
    for (int k = 0; k < 8; k++) v[8 + k] = v[8 + k] * rr * gv[k];
    *(uint4*)(A + (size_t)r * 4096 + 8 * tid) = pk8(v); *(uint4*)(A + (size_t)r * 4096 + 2048 + 8 * tid) = pk8(v + 8);
    issue(i + S - 1);
  }
  cpwait<0>();
}
// combine, ring of S Y rows over the (token, local slot) items of this CTA's tokens
template <int S>
__global__ void __launch_bounds__(256) combr(const bf* Y, const int* loc, const int* pos, const float* gw, const float* tw, bf* out, int T) {
  extern __shared__ uint4 sm[];
  __shared__ float red[2][8];
  const int G = gridDim.x, tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  auto mask = [&](int t) {
    unsigned m = 0;
#pragma unroll
    for (int k = 0; k < 8; k++) m |= ((unsigned)loc[t * 8 + k] < 32u ? 1u : 0u) << k;
    return m;
  };
  int pt = blockIdx.x, ps = 0, cs = 0, b = 0;
  unsigned pm = pt < T ? mask(pt) : 0u;
  auto issue = [&]() {
    while (pt < T && !pm) { pt += G; pm = pt < T ? mask(pt) : 0u; }
    if (pt < T) {
      const int k = __ffs(pm) - 1; pm &= pm - 1;
      const uint4* src = (const uint4*)(Y + (size_t)(pos ? pos[pt * 8 + k] : pt * 8 + k) * 4096);
      uint4* d = sm + (ps % S) * 512; ps++;
      cpa16(d + tid, src + tid); cpa16(d + 256 + tid, src + 256 + tid);
    }
    cpcommit();
  };
#pragma unroll
  for (int i = 0; i < S - 1; i++) issue();
  float gv[16]; ld8f(gw + 8 * tid, gv); ld8f(gw + 2048 + 8 * tid, gv + 8);
  for (int t = blockIdx.x; t < T; t += G) {
    unsigned m = mask(t);
    float acc[16];
#pragma unroll
    for (int j = 0; j < 16; j++) acc[j] = 0.f;
    while (m) {
      const int k = __ffs(m) - 1; m &= m - 1;
      cpwait<S - 2>();
      const uint4* d = sm + (cs % S) * 512; cs++;
      float v[16], q = 0.f;
      unp8(d[tid], v); unp8(d[256 + tid], v + 8);
#pragma unroll
      for (int j = 0; j < 16; j++) { v[j] *= gv[j]; q += v[j] * v[j]; }
      q = wsum(q);
      if (lane == 0) red[b][w] = q;
      __syncthreads();
      q = 0.f;
#pragma unroll
      for (int i = 0; i < 8; i++) q += red[b][i];
      b ^= 1;
      const float cf = tw[t * 8 + k] * rsqrtf(q / 4096 + 1e-8f);
#pragma unroll
      for (int j = 0; j < 16; j++) acc[j] += v[j] * cf;
      issue();
    }
    *(uint4*)(out + (size_t)t * 4096 + 8 * tid) = pk8(acc); *(uint4*)(out + (size_t)t * 4096 + 2048 + 8 * tid) = pk8(acc + 8);
  }
  cpwait<0>();
}
template <typename K> static int occ(K k, int nt, int smem) {
  int n = 0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, k, nt, smem); return n > 0 ? n : 1;
}
template <typename K> static void bigsmem(K k, int smem) { cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); }
static int nsm();
static int maxsm() { static int n = 0; if (!n) cudaDeviceGetAttribute(&n, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0); return n; }
static bool res_ar(torch::Tensor a, torch::Tensor s, torch::Tensor al, torch::Tensor be, torch::Tensor gn, torch::Tensor h, torch::Tensor ra, int64_t v) {
  const int T = a.size(0);
  auto launch = [&](auto k, int nt, int stg) {
    const int sb = stg * 3072 * 16; if (sb > maxsm()) return false; bigsmem(k, sb);
    k<<<min(T, nsm() * occ(k, nt, sb)), nt, sb, at::cuda::getCurrentCUDAStream()>>>((const bf*)a.data_ptr(), s.data_ptr<float>(), al.data_ptr<float>(),
        be.data_ptr<float>(), gn.data_ptr<float>(), (bf*)h.data_ptr(), ra.data_ptr<float>(), T);
    return true;
  };
  return v == 0 ? launch(resar<512, 4>, 512, 4) : v == 1 ? launch(resar<1024, 4>, 1024, 4) : launch(resar<512, 2>, 512, 2);
}
static bool res_br(torch::Tensor f, torch::Tensor gf, torch::Tensor a, torch::Tensor ra, torch::Tensor s, torch::Tensor al1, torch::Tensor be1, torch::Tensor al2,
            torch::Tensor be2, torch::Tensor gn, torch::Tensor so, torch::Tensor hid, int64_t v) {
  const int T = f.size(0);
  auto launch = [&](auto k, int nt, int stg) {
    const int sb = stg * 4096 * 16; if (sb > maxsm()) return false; bigsmem(k, sb);
    k<<<min(T, nsm() * occ(k, nt, sb)), nt, sb, at::cuda::getCurrentCUDAStream()>>>((const bf*)f.data_ptr(), gf.data_ptr<float>(), (const bf*)a.data_ptr(),
        ra.data_ptr<float>(), s.data_ptr<float>(), al1.data_ptr<float>(), be1.data_ptr<float>(), al2.data_ptr<float>(), be2.data_ptr<float>(),
        gn.data_ptr<float>(), so.data_ptr<float>(), (bf*)hid.data_ptr(), T);
    return true;
  };
  return v == 0 ? launch(resbr<512, 3>, 512, 3) : v == 1 ? launch(resbr<1024, 3>, 1024, 3) : launch(resbr<512, 2>, 512, 2);
}
void ring_test(torch::Tensor a, torch::Tensor s, torch::Tensor f, torch::Tensor al1, torch::Tensor be1, torch::Tensor al2, torch::Tensor be2,
               torch::Tensor gn, torch::Tensor gf, torch::Tensor h, torch::Tensor ra, torch::Tensor so, torch::Tensor hid) {
  const int T = a.size(0);
  auto st = at::cuda::getCurrentCUDAStream();
  bigsmem(resar<256, 3, 4096>, 3 * 1536 * 16); bigsmem(resbr<256, 2, 4096>, 2 * 2048 * 16);
  resar<256, 3, 4096><<<min(T, 40), 256, 3 * 1536 * 16, st>>>((const bf*)a.data_ptr(), s.data_ptr<float>(), al1.data_ptr<float>(), be1.data_ptr<float>(),
      gn.data_ptr<float>(), (bf*)h.data_ptr(), ra.data_ptr<float>(), T);
  resbr<256, 2, 4096><<<min(T, 40), 256, 2 * 2048 * 16, st>>>((const bf*)f.data_ptr(), gf.data_ptr<float>(), (const bf*)a.data_ptr(), ra.data_ptr<float>(),
      s.data_ptr<float>(), al1.data_ptr<float>(), be1.data_ptr<float>(), al2.data_ptr<float>(), be2.data_ptr<float>(), gn.data_ptr<float>(),
      so.data_ptr<float>(), (bf*)hid.data_ptr(), T);
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
void res_a(torch::Tensor a, torch::Tensor s, torch::Tensor al, torch::Tensor be, torch::Tensor gn, torch::Tensor h, torch::Tensor ra, int64_t v) {
  const int T = a.size(0);
  if (v >= 4 && res_ar(a, s, al, be, gn, h, ra, v - 4)) return;
  if (v >= 4) v = 1;
  if (v == 0) resa<1024, 1><<<T, 1024, 0, ST>>>(RA_ARGS);
  else if (v == 1) resa<512, 2><<<T, 512, 0, ST>>>(RA_ARGS);
  else if (v == 2) resa<256, 2><<<T, 256, 0, ST>>>(RA_ARGS);
  else resa<256, 3><<<T, 256, 0, ST>>>(RA_ARGS);
}
#define RB_ARGS BP(f), gf.data_ptr<float>(), BP(a), ra.data_ptr<float>(), s.data_ptr<float>(), al1.data_ptr<float>(), be1.data_ptr<float>(), \
      al2.data_ptr<float>(), be2.data_ptr<float>(), gn.data_ptr<float>(), so.data_ptr<float>(), BP(hid)
void res_b(torch::Tensor f, torch::Tensor gf, torch::Tensor a, torch::Tensor ra, torch::Tensor s, torch::Tensor al1, torch::Tensor be1, torch::Tensor al2,
           torch::Tensor be2, torch::Tensor gn, torch::Tensor so, torch::Tensor hid, int64_t v) {
  const int T = f.size(0);
  if (v >= 4 && res_br(f, gf, a, ra, s, al1, be1, al2, be2, gn, so, hid, v - 4)) return;
  if (v >= 4) v = 2;
  if (v == 0) resb<1024, 1><<<T, 1024, 0, ST>>>(RB_ARGS);
  else if (v == 1) resb<512, 2><<<T, 512, 0, ST>>>(RB_ARGS);
  else if (v == 2) resb<256, 2><<<T, 256, 0, ST>>>(RB_ARGS);
  else resb<256, 3><<<T, 256, 0, ST>>>(RB_ARGS);
}
void rmsgain2(torch::Tensor x, torch::Tensor g, torch::Tensor out) {
  if (x.scalar_type() == torch::kFloat32) rmsg2f<<<x.size(0), 512, 0, ST>>>(x.data_ptr<float>(), (int)x.stride(0), g.data_ptr<float>(), BP(out));
  else rmsg2<<<x.size(0), 512, 0, ST>>>(BP(x), g.data_ptr<float>(), BP(out));
}
#define RS 6
void gathnorm(torch::Tensor x, torch::Tensor list, torch::Tensor offs, torch::Tensor g, torch::Tensor A, int64_t v) {
  if (v == 0) { gnorm<<<min((long)A.size(0), (long)nsm() * occ(gnorm, 256, 0)), 256, 0, ST>>>(BP(x), list.data_ptr<int>(), offs.data_ptr<int>(), g.data_ptr<float>(), BP(A)); return; }
  const int sb = RS * 8192; bigsmem(gnormr<RS>, sb);
  gnormr<RS><<<min((long)A.size(0), (long)nsm() * occ(gnormr<RS>, 256, sb)), 256, sb, ST>>>(BP(x), list.data_ptr<int>(), offs.data_ptr<int>(), g.data_ptr<float>(), BP(A));
}
void act2(torch::Tensor H, torch::Tensor offs, torch::Tensor I) {
  swiglu2<<<nsm() * occ(swiglu2, 256, 0), 256, 0, ST>>>(BP(H), offs.data_ptr<int>(), BP(I));
}
void combine(torch::Tensor Y, torch::Tensor loc, torch::Tensor pos, torch::Tensor gw, torch::Tensor tw, torch::Tensor out, int64_t v) {
  const int* pp = pos.numel() ? pos.data_ptr<int>() : nullptr;
  const int T = out.size(0);
  if (v == 0) { comb<<<T, 256, 0, ST>>>(BP(Y), loc.data_ptr<int>(), pp, gw.data_ptr<float>(), tw.data_ptr<float>(), BP(out)); return; }
  const int sb = RS * 8192; bigsmem(combr<RS>, sb);
  combr<RS><<<min(T, nsm() * occ(combr<RS>, 256, sb)), 256, sb, ST>>>(BP(Y), loc.data_ptr<int>(), pp, gw.data_ptr<float>(), tw.data_ptr<float>(), BP(out), T);
}
void routing2(torch::Tensor lg, torch::Tensor bias, torch::Tensor ls, torch::Tensor ids, torch::Tensor loc, torch::Tensor lid, torch::Tensor tw, torch::Tensor cnt) {
  const int T = lg.size(0);
  int* c = cnt.numel() ? cnt.data_ptr<int>() : nullptr;
  if (c) cudaMemsetAsync(c, 0, 32 * 4, ST);
  route2<<<(T + 3) / 4, 128, 0, ST>>>(lg.data_ptr<float>(), (int)lg.stride(0), bias.data_ptr<float>(), ls.data_ptr<int>(), T, ids.data_ptr<int>(), loc.data_ptr<int>(),
      lid.data_ptr<int>(), tw.data_ptr<float>(), c);
}
void rlist(torch::Tensor loc, torch::Tensor cnt, torch::Tensor offs, torch::Tensor fill, torch::Tensor list, torch::Tensor pos) {
  const int R = loc.numel();
  rscan2<<<1, 32, 0, ST>>>(cnt.data_ptr<int>(), offs.data_ptr<int>(), fill.data_ptr<int>());
  rfill2<<<(R + 255) / 256, 256, 0, ST>>>(loc.data_ptr<int>(), R, fill.data_ptr<int>(), list.data_ptr<int>(), pos.data_ptr<int>());
}
"""
_pf2mod = None
import time
_PF2_FN = ["qkprep", "ongate", "res_a", "res_b", "rmsgain2", "gathnorm", "act2", "combine", "routing2", "rlist", "dequant2", "ring_test"]


def _pf2():
    global _pf2mod
    if _pf2mod is None:
        import re
        decl = " ".join(re.sub(r"\s+", " ", m) + ";" for m in re.findall(r"^void \w+\([^)]*\)", PF2_CUDA, re.M) if m.split("(")[0][5:] in _PF2_FN)
        _pf2mod = load_inline("pf2_glue_v8", cpp_sources=decl, cuda_sources=PF2_CUDA, functions=_PF2_FN, extra_cuda_cflags=["-O3"], verbose=False)
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
RV_TUNE = True
MOE_TUNE = True
RF_TUNE = True
LT_M = (753, 3012, 16384)
LT_BUDGET = 70.0  # seconds for the whole dense-GEMM search, capped by the prepare deadline
PREP_DEADLINE = 255.0  # prepare() wall clock at which tuning stops starting new work
_T0 = []


def _left():
    return PREP_DEADLINE - (time.time() - _T0[0])


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
    """per prefill shape: torch's default vs cuBLASLt heuristics of all four layouts, then an algo-66 (sm90 nvjet) sweep over
    tile / cluster and a custom-option / cluster / split-K / stages refinement of the leaders (each config checked, then timed)"""
    ext = _ltext()
    if ext is None:
        return
    dev = st["o"].device
    jobs = [(name, m) for m in LT_M for name in ("qkv_gate", "o", "pooler", "unpooler")]
    t0 = time.time()
    budget = min(LT_BUDGET, max(0.0, _left()))
    caps = ext.lt_caps(66)
    for ji, (name, m) in enumerate(jobs):
        W, Wt = st[name], st[name + "_t"]
        n, K = W.shape
        dl = t0 + budget * (ji + 1) / len(jobs)
        x = torch.randn((m, K), device=dev).to(torch.bfloat16)
        out = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
        ref = (x @ W.T).float()
        def ok(c):
            if ext.lt_mm(x, W, Wt, out, c) != 0:
                return False
            torch.cuda.synchronize()
            return bool(torch.isfinite(out).all()) and float((out.float() - ref).norm() / ref.norm()) < 1e-2
        run = lambda c: ext.lt_mm(x, W, Wt, out, c)
        tried, res, seen = set(), [], set()
        def tryc(c):
            if c < 0 or c in tried or time.time() > dl:
                return
            tried.add(c)
            if ok(c):
                res.append((_gtm(lambda: run(c), fl, sink, 5), c))
        def addc(lay, cfg):
            if time.time() > dl:
                return
            c = ext.lt_add(m, n, K, lay, cfg)
            if c >= 0:
                k2 = (lay, tuple(ext.lt_cfg(m, n, K, c)))
                if k2 not in seen:
                    seen.add(k2)
                    tryc(c)
        t_torch = _gtm(lambda: torch.mm(x, W.T, out=out), fl, sink, 9)
        for lay in range(4):
            for i in range(min(ext.lt_algos(m, n, K, lay), 8)):
                seen.add((lay, tuple(ext.lt_cfg(m, n, K, lay * 100000 + i))))
                tryc(lay * 100000 + i)
        if caps:
            nt = caps[0]; tiles = caps[1:1 + nt]; ns = caps[1 + nt]; stages = caps[2 + nt:2 + nt + ns]
            cmax, ksup = caps[2 + nt + ns], caps[3 + nt + ns]
            stg = [35] if 35 in stages else stages[:1] or [0]
            res.sort()
            lays = [res[0][1] // 100000] if res else [2]
            lays += [l for l in (2, 0, 3, 1) if l not in lays]
            for lay in lays:
                for tile in tiles:
                    for cl in (2, 5):
                        addc(lay, [66, tile, 1, 0, 0, 0, stg[0], 0, cl])
            res.sort()
            for _, code in res[:3]:
                lay, cfg = code // 100000, list(ext.lt_cfg(m, n, K, code))
                if cfg[0] != 66:
                    continue
                for cu in range(min(cmax, 7) + 1):
                    for cl in (0, 2, 3, 4, 5, 6):
                        addc(lay, cfg[:5] + [cu] + cfg[6:8] + [cl])
                if ksup and m <= 3012:
                    for sk in (2, 3, 4):
                        addc(lay, cfg[:2] + [sk, 2] + cfg[4:])
                for sg in stages[::max(1, len(stages) // 6)]:
                    addc(lay, cfg[:6] + [sg] + cfg[7:])
        res.sort()
        best = (t_torch, -1)
        for _, code in res[:3]:
            a = _gtm(lambda: run(code), fl, sink, 11)
            t_torch = min(t_torch, _gtm(lambda: torch.mm(x, W.T, out=out), fl, sink, 11))
            if a < best[0]:
                best = (a, code)
        if best[1] >= 0 and best[0] > t_torch * 0.985:
            best = (t_torch, -1)
        if best[1] >= 0:
            st["lt"][(name, m)] = best[1]
        cfg = "torch" if best[1] < 0 else "%d:%s" % (best[1] // 100000, ".".join(map(str, ext.lt_cfg(m, n, K, best[1]))))
        print("pf2 lt %s m%d torch %.1fus best %s %.1fus (%d tried, %.1fs)" % (name, m, t_torch * 1e3, cfg, best[0] * 1e3, len(tried), time.time() - t0), flush=True)
        del x, out, ref


def _synth_routes(T, hot, dev, seed):
    """synthetic top-8 routes for timing: uniform over 256 experts, or with local expert 0 picked by 40% of tokens"""
    gen = torch.Generator(device=dev).manual_seed(seed)
    ids = torch.argsort(torch.rand((T, 256), generator=gen, device=dev), dim=1)[:, :8].to(torch.int32)
    if hot:
        sel = torch.rand((T,), generator=gen, device=dev) < 0.4
        ids[:, 0] = torch.where(sel & (ids != 0).all(1), torch.zeros_like(ids[:, 0]), ids[:, 0])
    loc = ids.contiguous()
    lid = torch.where(loc < 32, loc, torch.full_like(loc, -1))
    tw = torch.rand((T, 8), generator=gen, device=dev)
    tw = (tw / tw.sum(1, keepdim=True)).contiguous()
    return loc, lid, tw


def _routes_list(g, loc, T, dev):
    cnt = torch.bincount(loc[loc < 32].long(), minlength=32).to(torch.int32)
    offs, fill = torch.empty_like(cnt), torch.empty_like(cnt)
    lst = torch.empty(T * 8, dtype=torch.int32, device=dev); pos = torch.empty_like(lst)
    g.rlist(loc, cnt, offs, fill, lst, pos)
    return cnt, offs, lst, pos


def _glue_tune(st, fl, sink):
    """residual-boundary, gather-norm and combine kernel variants per prefill size, timed on this GPU"""
    g = _pf2()
    dev = st["o"].device
    for m in LT_M:
        a = torch.randn((m, 8192), device=dev).to(torch.bfloat16); f = torch.randn_like(a); s = torch.randn((m, 8192), device=dev)
        h = torch.empty_like(a); ra = torch.empty((m,), device=dev); so = torch.empty_like(s)
        ta = [_gtm(lambda: g.res_a(a, s, st["a1"], st["b1"], st["g_pre_ffn_norm"], h, ra, v), fl, sink) for v in range(7)]
        tb = [_gtm(lambda: g.res_b(f, st["g_post_ffn_norm"], a, ra, s, st["a1"], st["b1"], st["a2"], st["b2"], st["g_next_pre_attn_norm"], so, h, v), fl, sink)
              for v in range(7)]
        del a, f, s, h, so
        loc, lid, tw = _synth_routes(m, True, dev, 7)
        cnt, offs, lst, pos = _routes_list(g, loc, m, dev)
        pr = torch.randn((m, 4096), device=dev).to(torch.bfloat16)
        A = torch.empty((m * 8, 4096), dtype=torch.bfloat16, device=dev)
        Y = torch.randn((m * 8, 4096), device=dev).to(torch.bfloat16); y = torch.empty_like(pr)
        tg = [_gtm(lambda: g.gathnorm(pr, lst, offs, st["g_pre_expert_norm"], A, v), fl, sink) for v in range(2)]
        tc = [_gtm(lambda: g.combine(Y, loc, pos, st["g_post_expert_norm"], tw, y, v), fl, sink) for v in range(2)]
        tm_ = [_gtm(lambda: g.combine(Y, lid, st["nocnt"], st["g_post_expert_norm"], tw, y, v), fl, sink) for v in range(2)]
        am = lambda t: min(range(len(t)), key=t.__getitem__)
        st["rv"][m] = (am(ta), am(tb), am(tg), am(tc), am(tm_))
        fm = lambda t: " ".join(f"{x * 1e3:.1f}" for x in t)
        print(f"pf2 glue m{m}: resid a {fm(ta)} b {fm(tb)} gnorm {fm(tg)} comb {fm(tc)} comb-slot {fm(tm_)} -> {st['rv'][m]}", flush=True)
        del pr, A, Y, y


def _rv(st, T):
    return st["rv"][min(st["rv"], key=lambda m: abs(m - T))] if st["rv"] else (0, 0, 1, 1, 1)


def _moe_tune(st, fl, sink):
    """per turn size: SGLang's Marlin (each moe_block_size) vs our dequant + grouped GEMM, on synthetic uniform and hot routings"""
    mm = _marlin_mods()
    if mm is None or st.get("bank") is None:
        return
    g = _pf2()
    dev = st["o"].device
    for T in (753, 3012):
        if _left() < 25:
            break
        pr = torch.randn((T, 4096), device=dev).to(torch.bfloat16)
        cand = {("mar", bm): 0.0 for bm in (16, 32, 48, 64)}
        cand[("big", 0)] = 0.0
        for hot in (False, True):
            loc, lid, tw = _synth_routes(T, hot, dev, 11 + hot)
            cnt, offs, lst, pos = _routes_list(g, loc, T, dev)
            for c in cand:
                if c[0] == "mar":
                    al = mm["align"](lid, c[1], 256)
                    torch.cuda.synchronize()
                    fn = lambda: _moe_mar(g, pr, tw, lid, al, c[1], st, mm, _rv(st, T))
                else:
                    fn = lambda: _moe_big(g, pr, loc, tw, cnt, offs, lst, pos, st, _rv(st, T))
                cand[c] += _gtm(fn, fl, sink, 7)
        best = min(cand, key=cand.get)
        dflt = ("mar", _mbm(T))
        if cand[best] > cand[dflt] * 0.97:
            best = dflt
        st["mv"][T] = best
        print(f"pf2 moe T{T}: " + " ".join(f"{k[0]}{k[1] or ''} {v * 500:.0f}" for k, v in cand.items()) + f" -> {best}", flush=True)
        del pr


def _rf_tune(st, fl, sink):
    """Marlin-path sizes: pooler GEMM with the router hi/lo GEMM on a side stream, or one fp32-output GEMM on [pooler; hi; lo]"""
    g = _pf2()
    dev = st["o"].device
    side = st["side"]
    for T in (753, 3012):
        h = torch.randn((T, 8192), device=dev).to(torch.bfloat16)
        ids = torch.empty((T, 8), dtype=torch.int32, device=dev); loc, lid = torch.empty_like(ids), torch.empty_like(ids)
        tw = torch.empty((T, 8), device=dev); p = torch.empty((T, 4096), dtype=torch.bfloat16, device=dev)
        ls = torch.zeros(1, dtype=torch.int32, device=dev)
        def seg_a():
            cur = torch.cuda.current_stream()
            side.wait_stream(cur)
            pr = _dmm(h, st, "pooler")
            with torch.cuda.stream(side):
                g.routing2(_mm32(h, st["rhl_t"]), st["router_bias"], ls, ids, loc, lid, tw, st["nocnt"])
            g.rmsgain2(pr, st["g_pre_expert_norm"], p)
            cur.wait_stream(side)
        def seg_b():
            cur = torch.cuda.current_stream()
            big = _mm32(h, st["prt_t"])
            side.wait_stream(cur)
            with torch.cuda.stream(side):
                g.routing2(big[:, 4096:], st["router_bias"], ls, ids, loc, lid, tw, st["nocnt"])
            g.rmsgain2(big, st["g_pre_expert_norm"], p)
            cur.wait_stream(side)
        ta, tb = _gtm(seg_a, fl, sink, 11), _gtm(seg_b, fl, sink, 11)
        ta, tb = min(ta, _gtm(seg_a, fl, sink, 11)), min(tb, _gtm(seg_b, fl, sink, 11))
        st["rf"][T] = tb < ta * 0.98
        print(f"pf2 router T{T}: side-stream {ta * 1e3:.1f}us fused {tb * 1e3:.1f}us -> {'fused' if st['rf'][T] else 'side'}", flush=True)
        del h


def _dmm(x, st, name):
    i = st["lt"].get((name, x.shape[0]))
    if i is None:
        return x @ st[name].T
    out = torch.empty((x.shape[0], st[name].shape[0]), dtype=torch.bfloat16, device=x.device)
    _ltext().lt_mm(x, st[name], st[name + "_t"], out, i)
    return out


_prepare_dl, _kernel_fn_dl = prepare, kernel_fn


def prepare(weights):
    _T0[:] = [time.time()]
    st = _prepare_dl(weights)
    dev = weights["o"].device
    st["nocnt"] = torch.empty(0, dtype=torch.int32, device=dev)
    if st.get("bank") is not None:
        st["bank"]["nopos"], st["bank"]["post_wf"] = st["nocnt"], st["g_post_expert_norm"]
    for k in ("qkv_gate", "o", "pooler", "unpooler"):
        st[k + "_t"] = weights[k].t().contiguous()
    st["lt"], st["rv"], st["mv"], st["rf"] = {}, {}, {}, {}
    st["prt_t"] = torch.cat((weights["pooler"], st["rhi"], st["rlo"])).t().contiguous()
    _pf2()
    if LT_TUNE:
        _ltext()
    print(f"pf2 compiled at {time.time() - _T0[0]:.1f}s", flush=True)
    fl, sink = torch.empty(64 << 20, dtype=torch.float32, device=dev), torch.empty((), device=dev)
    if RV_TUNE and _left() > 30:
        _glue_tune(st, fl, sink)
    if RF_TUNE and st.get("bank") is not None and _left() > 30:
        _rf_tune(st, fl, sink)
    if MOE_TUNE and _left() > 30:
        _moe_tune(st, fl, sink)
    if LT_TUNE and _left() > 20:
        _lt_tune(st, fl, sink)
    del fl, sink
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"pf2 prepare done at {time.time() - _T0[0]:.1f}s", flush=True)
    return st


def _attn_pf(data, cfg, qkv, B, L, T):
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
    o = A__fa3()(q=q8.view(torch.float8_e4m3fn), k_cache=data["k_cache"], v_cache=data["v_cache"], page_table=data["page_table"],
                 cache_seqlens=csl, cu_seqlens_q=cuq, max_seqlen_q=L, softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"], causal=True,
                 window_size=(win - 1, 0) if win else (-1, -1), scheduler_metadata=data.get("scheduler_metadata"), **desc)
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


def _marlin2(x, tw, lid, al, bm, bank, mm, cv):
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
    _pf2().combine(y, lid, bank["nopos"], bank["post_wf"], tw, out, cv)
    return out


def _moe_mar(g, pr, tw, lid, al, bm, st, mm, rv, p=None):
    if p is None:
        p = torch.empty_like(pr)
        g.rmsgain2(pr, st["g_pre_expert_norm"], p)
    return _marlin2(p, tw, lid, al, bm, st["bank"], mm, rv[4])


def _moe_big(g, pr, loc, tw, cnt, offs, lst, pos, st, rv):
    T, dev = pr.shape[0], pr.device
    R = T * 8
    w13 = torch.empty((32, 8192, 4096), dtype=torch.bfloat16, device=dev)
    g.dequant2(st["pw13"], st["pw13_sf"], st["pw13_gs"], cnt, w13)
    A = torch.empty((R, 4096), dtype=torch.bfloat16, device=dev)
    g.gathnorm(pr, lst, offs, st["g_pre_expert_norm"], A, rv[2])
    Hh = _gmm(A, w13, offs)
    del A, w13
    w2 = torch.empty((32, 4096, 4096), dtype=torch.bfloat16, device=dev)
    g.dequant2(st["pw2"], st["pw2_sf"], st["pw2_gs"], cnt, w2)
    I = torch.empty((R, 4096), dtype=torch.bfloat16, device=dev)
    g.act2(Hh, offs, I)
    del Hh
    Y = _gmm(I, w2, offs)
    del I, w2
    y = torch.empty((T, 4096), dtype=torch.bfloat16, device=dev)
    g.combine(Y, loc, pos, st["g_post_expert_norm"], tw, y, rv[3])
    return y


def kernel_fn(data, st):
    cfg = data["config"]
    B, L, T = cfg["batch"], cfg["q_len"], cfg["tokens"]
    if T <= 256 or cfg["num_q_heads"] != 16 or cfg["num_kv_heads"] != 2:
        return _kernel_fn_dl(data, st)
    x = data["attn_input"]
    dev = x.device
    g = _pf2()
    qkv = _dmm(x, st, "qkv_gate")
    att = _attn_pf(data, cfg, qkv, B, L, T)
    a = _dmm(att, st, "o")
    h = torch.empty((T, 8192), dtype=torch.bfloat16, device=dev)
    ra = torch.empty((T,), dtype=torch.float32, device=dev)
    rv = _rv(st, T)
    g.res_a(a, data["stream"], st["a1"], st["b1"], st["g_pre_ffn_norm"], h, ra, rv[0])
    mv = st["mv"].get(T)
    if mv is None:
        mv = ("mar", _mbm(T)) if MARLIN_MIN < T <= MARLIN_MAX else ("big", 0)
    if st.get("bank") is None:
        mv = ("big", 0)
    mar = mv[0] == "mar"
    ids = torch.empty((T, 8), dtype=torch.int32, device=dev)
    loc, lid = torch.empty_like(ids), torch.empty_like(ids)
    tw = torch.empty((T, 8), dtype=torch.float32, device=dev)
    if not mar:
        cnt = torch.empty(32, dtype=torch.int32, device=dev); offs = torch.empty_like(cnt); fill = torch.empty_like(cnt)
        lst = torch.empty(T * 8, dtype=torch.int32, device=dev); pos = torch.empty(T * 8, dtype=torch.int32, device=dev)
    cur, side = torch.cuda.current_stream(), st["side"]
    side.wait_stream(cur)
    if not (mar and st["rf"].get(T, False)):
        pr = _dmm(h, st, "pooler")
    rf = mar and st["rf"].get(T, False)
    if rf:
        pr = None
        big = _mm32(h, st["prt_t"])
        side.wait_stream(cur)
    with torch.cuda.stream(side):
        lg = big[:, 4096:] if rf else _mm32(h, st["rhl_t"])
        g.routing2(lg, st["router_bias"], data["local_start"], ids, loc, lid, tw, st["nocnt"] if mar else cnt)
        if mar:
            mm = _marlin_mods()
            al = mm["align"](lid, mv[1], 256)
            if not torch.cuda.is_current_stream_capturing():
                for t_ in al:
                    t_.record_stream(cur)
        else:
            g.rlist(loc, cnt, offs, fill, lst, pos)
    if mar:
        p = torch.empty((T, 4096), dtype=torch.bfloat16, device=dev)
        g.rmsgain2(big if rf else pr, st["g_pre_expert_norm"], p)
        cur.wait_stream(side)
        y = _moe_mar(g, pr, tw, lid, al, mv[1], st, mm, rv, p)
    else:
        cur.wait_stream(side)
        y = _moe_big(g, pr, loc, tw, cnt, offs, lst, pos, st, rv)
    f = _dmm(y, st, "unpooler")
    s2 = torch.empty((T, 8192), dtype=torch.float32, device=dev)
    hid = torch.empty_like(h)
    g.res_b(f, st["g_post_ffn_norm"], a, ra, data["stream"], st["a1"], st["b1"], st["a2"], st["b2"], st["g_next_pre_attn_norm"], s2, hid, rv[1])
    return hid, s2, ids
