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
__global__ void route2(const float* lg, const float* bias, const int* lstart, int T, int* ids, int* loc, int* lid, float* tw, int* cnt) {
  const int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (r >= T) return;
  float sc[8], key[8];
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const int e = i * 32 + lane;
    const float l = lg[(size_t)r * 512 + e] + lg[(size_t)r * 512 + 256 + e];
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
  if (v == 0) resb<1024, 1><<<T, 1024, 0, ST>>>(RB_ARGS);
  else if (v == 1) resb<512, 2><<<T, 512, 0, ST>>>(RB_ARGS);
  else if (v == 2) resb<256, 2><<<T, 256, 0, ST>>>(RB_ARGS);
  else resb<256, 3><<<T, 256, 0, ST>>>(RB_ARGS);
}
void rmsgain2(torch::Tensor x, torch::Tensor g, torch::Tensor out) { rmsg2<<<x.size(0), 512, 0, ST>>>(BP(x), g.data_ptr<float>(), BP(out)); }
void gathnorm(torch::Tensor x, torch::Tensor list, torch::Tensor offs, torch::Tensor g, torch::Tensor A) {
  gnorm<<<min((long)A.size(0), (long)nsm() * 8), 256, 0, ST>>>(BP(x), list.data_ptr<int>(), offs.data_ptr<int>(), g.data_ptr<float>(), BP(A));
}
void act2(torch::Tensor H, torch::Tensor offs, torch::Tensor I) {
  swiglu2<<<nsm() * 8, 256, 0, ST>>>(BP(H), offs.data_ptr<int>(), BP(I));
}
void combine(torch::Tensor Y, torch::Tensor loc, torch::Tensor pos, torch::Tensor gw, torch::Tensor tw, torch::Tensor out) {
  comb<<<out.size(0), 256, 0, ST>>>(BP(Y), loc.data_ptr<int>(), pos.numel() ? pos.data_ptr<int>() : nullptr, gw.data_ptr<float>(), tw.data_ptr<float>(), BP(out));
}
void routing2(torch::Tensor lg, torch::Tensor bias, torch::Tensor ls, torch::Tensor ids, torch::Tensor loc, torch::Tensor lid, torch::Tensor tw, torch::Tensor cnt) {
  const int T = lg.size(0);
  int* c = cnt.numel() ? cnt.data_ptr<int>() : nullptr;
  if (c) cudaMemsetAsync(c, 0, 32 * 4, ST);
  route2<<<(T + 3) / 4, 128, 0, ST>>>(lg.data_ptr<float>(), bias.data_ptr<float>(), ls.data_ptr<int>(), T, ids.data_ptr<int>(), loc.data_ptr<int>(),
      lid.data_ptr<int>(), tw.data_ptr<float>(), c);
}
void rlist(torch::Tensor loc, torch::Tensor cnt, torch::Tensor offs, torch::Tensor fill, torch::Tensor list, torch::Tensor pos) {
  const int R = loc.numel();
  rscan2<<<1, 32, 0, ST>>>(cnt.data_ptr<int>(), offs.data_ptr<int>(), fill.data_ptr<int>());
  rfill2<<<(R + 255) / 256, 256, 0, ST>>>(loc.data_ptr<int>(), R, fill.data_ptr<int>(), list.data_ptr<int>(), pos.data_ptr<int>());
}
"""
_pf2mod = None
_PF2_FN = ["qkprep", "ongate", "res_a", "res_b", "rmsgain2", "gathnorm", "act2", "combine", "routing2", "rlist", "dequant2"]


def _pf2():
    global _pf2mod
    if _pf2mod is None:
        import re
        decl = " ".join(re.sub(r"\s+", " ", m) + ";" for m in re.findall(r"^void \w+\([^)]*\)", PF2_CUDA, re.M) if m.split("(")[0][5:] in _PF2_FN)
        _pf2mod = load_inline("pf2_glue_v4", cpp_sources=decl, cuda_sources=PF2_CUDA, functions=_PF2_FN, extra_cuda_cflags=["-O3"], verbose=False)
    return _pf2mod


PF2_LT = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>
struct LtPlan { cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t a, b, c; int n; cublasLtMatmulHeuristicResult_t algos[24]; };
static cublasLtHandle_t g_lt = nullptr;
static void* g_ltws = nullptr;
static const size_t LTWS = 64ull << 20;
static LtPlan g_pl[64];
static int g_np = 0, g_key[64][3];
static LtPlan& ltplan(int m, int n, int k) {
  for (int i = 0; i < g_np; i++) if (g_key[i][0] == m && g_key[i][1] == n && g_key[i][2] == k) return g_pl[i];
  TORCH_CHECK(g_np < 64);
  if (!g_lt) { cublasLtCreate(&g_lt); cudaMalloc(&g_ltws, LTWS); }
  LtPlan& p = g_pl[g_np]; g_key[g_np][0] = m; g_key[g_np][1] = n; g_key[g_np][2] = k; g_np++;
  cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
  cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
  cublasLtMatrixLayoutCreate(&p.a, CUDA_R_16BF, k, n, k);
  cublasLtMatrixLayoutCreate(&p.b, CUDA_R_16BF, k, m, k);
  cublasLtMatrixLayoutCreate(&p.c, CUDA_R_16BF, n, m, n);
  cublasLtMatmulPreference_t pref;
  cublasLtMatmulPreferenceCreate(&pref);
  size_t ws = LTWS;
  cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
  int cnt = 0;
  cublasLtMatmulAlgoGetHeuristic(g_lt, p.op, p.a, p.b, p.c, p.c, pref, 24, p.algos, &cnt);
  p.n = cnt;
  cublasLtMatmulPreferenceDestroy(pref);
  return p;
}
int64_t lt_algos(int64_t m, int64_t n, int64_t k) { return ltplan(m, n, k).n; }
int64_t lt_mm(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t idx) {
  const int m = x.size(0), k = x.size(1), n = w.size(0);
  LtPlan& p = ltplan(m, n, k);
  const float alpha = 1.f, beta = 0.f;
  return (int64_t)cublasLtMatmul(g_lt, p.op, &alpha, w.data_ptr(), p.a, x.data_ptr(), p.b, &beta, out.data_ptr(), p.c, out.data_ptr(), p.c,
                                 &p.algos[idx].algo, g_ltws, LTWS, at::cuda::getCurrentCUDAStream());
}
"""
_ltmod = []
LT_TUNE = True
RV_TUNE = True
LT_M = (753, 3012, 16384)


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
            _ltmod.append(load_inline("pf2_lt_v1", cpp_sources="int64_t lt_algos(int64_t,int64_t,int64_t); int64_t lt_mm(torch::Tensor,torch::Tensor,torch::Tensor,int64_t);",
                                      cuda_sources=PF2_LT, functions=["lt_algos", "lt_mm"], extra_cuda_cflags=["-O3"], extra_ldflags=fl, verbose=False))
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
    """per prefill shape: torch's cuBLAS default vs each cuBLASLt heuristic algorithm"""
    ext = _ltext()
    if ext is None:
        return
    dev = st["o"].device
    for m in LT_M:
        for name in ("qkv_gate", "o", "pooler", "unpooler"):
            W = st[name]
            x = torch.randn((m, W.shape[1]), device=dev).to(torch.bfloat16)
            out = torch.empty((m, W.shape[0]), dtype=torch.bfloat16, device=dev)
            ref = (x @ W.T).float()
            t0 = _gtm(lambda: torch.mm(x, W.T, out=out), fl, sink)
            best = (t0 * 0.97, -1)
            for i in range(ext.lt_algos(m, W.shape[0], W.shape[1])):
                if ext.lt_mm(x, W, out, i) != 0:
                    continue
                torch.cuda.synchronize()
                if not torch.isfinite(out).all() or ((out.float() - ref).norm() / ref.norm()).item() > 1e-2:
                    continue
                best = min(best, (_gtm(lambda: ext.lt_mm(x, W, out, i), fl, sink), i))
            if best[1] >= 0:
                st["lt"][(name, m)] = best[1]
            print(f"pf2 lt {name} m{m}: torch {t0 * 1e3:.1f}us lt {best[0] * 1e3:.1f}us algo {best[1]}", flush=True)
            del x, out, ref


def _rv_tune(st, fl, sink):
    """residual-boundary kernel shape (threads per row) per prefill size, timed on this GPU"""
    g = _pf2()
    dev = st["o"].device
    for m in LT_M:
        a = torch.randn((m, 8192), device=dev).to(torch.bfloat16); f = torch.randn_like(a); s = torch.randn((m, 8192), device=dev)
        h = torch.empty_like(a); ra = torch.empty((m,), device=dev); so = torch.empty_like(s)
        ta = [_gtm(lambda: g.res_a(a, s, st["a1"], st["b1"], st["g_pre_ffn_norm"], h, ra, v), fl, sink) for v in range(4)]
        tb = [_gtm(lambda: g.res_b(f, st["g_post_ffn_norm"], a, ra, s, st["a1"], st["b1"], st["a2"], st["b2"], st["g_next_pre_attn_norm"], so, h, v), fl, sink)
              for v in range(4)]
        st["rv"][m] = (min(range(4), key=ta.__getitem__), min(range(4), key=tb.__getitem__))
        print(f"pf2 resid m{m}: a " + " ".join(f"{t * 1e3:.1f}" for t in ta) + " b " + " ".join(f"{t * 1e3:.1f}" for t in tb) + f" -> {st['rv'][m]}", flush=True)
        del a, f, s, h, so


def _rv(st, T):
    return st["rv"][min(st["rv"], key=lambda m: abs(m - T))] if st["rv"] else (0, 0)


def _dmm(x, st, name):
    i = st["lt"].get((name, x.shape[0]))
    if i is None:
        return x @ st[name].T
    out = torch.empty((x.shape[0], st[name].shape[0]), dtype=torch.bfloat16, device=x.device)
    _ltext().lt_mm(x, st[name], out, i)
    return out


_prepare_dl, _kernel_fn_dl = prepare, kernel_fn


def prepare(weights):
    st = _prepare_dl(weights)
    st["nocnt"] = torch.empty(0, dtype=torch.int32, device=weights["o"].device)
    if st.get("bank") is not None:
        st["bank"]["nopos"], st["bank"]["post_wf"] = st["nocnt"], st["g_post_expert_norm"]
    st["lt"], st["rv"] = {}, {}
    _pf2()
    fl, sink = torch.empty(64 << 20, dtype=torch.float32, device=weights["o"].device), torch.empty((), device=weights["o"].device)
    if RV_TUNE:
        _rv_tune(st, fl, sink)
    if LT_TUNE:
        _lt_tune(st, fl, sink)
    del fl, sink
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
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


def _marlin2(x, tw, lid, al, bank, mm):
    T, K = x.shape
    N = bank["size_n"]
    bm = _mbm(T)
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
    _pf2().combine(y, lid, bank["nopos"], bank["post_wf"], tw, out)
    return out


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
    mar = MARLIN_MIN < T <= MARLIN_MAX and st.get("bank") is not None
    ids = torch.empty((T, 8), dtype=torch.int32, device=dev)
    loc, lid = torch.empty_like(ids), torch.empty_like(ids)
    tw = torch.empty((T, 8), dtype=torch.float32, device=dev)
    R = T * 8
    if not mar:
        cnt = torch.empty(32, dtype=torch.int32, device=dev); offs = torch.empty_like(cnt); fill = torch.empty_like(cnt)
        lst = torch.empty(R, dtype=torch.int32, device=dev); pos = torch.empty(R, dtype=torch.int32, device=dev)
    cur, side = torch.cuda.current_stream(), st["side"]
    side.wait_stream(cur)
    pr = _dmm(h, st, "pooler")
    with torch.cuda.stream(side):
        lg = _mm32(h, st["rhl_t"])
        g.routing2(lg, st["router_bias"], data["local_start"], ids, loc, lid, tw, st["nocnt"] if mar else cnt)
        if mar:
            mm = _marlin_mods()
            al = mm["align"](lid, _mbm(T), 256)
            if not torch.cuda.is_current_stream_capturing():
                for t_ in al:
                    t_.record_stream(cur)
        else:
            g.rlist(loc, cnt, offs, fill, lst, pos)
    if mar:
        p = torch.empty_like(pr)
        g.rmsgain2(pr, st["g_pre_expert_norm"], p)
        cur.wait_stream(side)
        y = _marlin2(p, tw, lid, al, st["bank"], mm)
    else:
        cur.wait_stream(side)
        w13 = torch.empty((32, 8192, 4096), dtype=torch.bfloat16, device=dev)
        g.dequant2(st["pw13"], st["pw13_sf"], st["pw13_gs"], cnt, w13)
        A = torch.empty((R, 4096), dtype=torch.bfloat16, device=dev)
        g.gathnorm(pr, lst, offs, st["g_pre_expert_norm"], A)
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
        g.combine(Y, loc, pos, st["g_post_expert_norm"], tw, y)
    f = _dmm(y, st, "unpooler")
    s2 = torch.empty((T, 8192), dtype=torch.float32, device=dev)
    hid = torch.empty_like(h)
    g.res_b(f, st["g_post_ffn_norm"], a, ra, data["stream"], st["a1"], st["b1"], st["a2"], st["b2"], st["g_next_pre_attn_norm"], s2, hid, rv[1])
    return hid, s2, ids
