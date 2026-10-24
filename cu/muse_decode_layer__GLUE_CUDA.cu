// extracted from muse_decode_layer (GLUE_CUDA); edit then `python3 tools/cu.py embed muse_decode_layer`
// ---- cu.py: end of standalone prelude ----
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#define H 8192
// one CTA (1024 threads, 8 values each) per token row
__device__ __forceinline__ float bsum(float v, float* red) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  __syncthreads();
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < 32; i++) s += red[i];
  return s;
}
// s' = alpha * s + beta * rms(a * ga, 1e-8);  out = bf16(rms(s', 1e-5) * gn). a has `na` columns (rest zero).
// every load is issued before the first reduction
__device__ __forceinline__ void ld8(const float* p, float* v) { const float4 x = __ldg((const float4*)p), y = __ldg((const float4*)p + 1); v[0] = x.x; v[1] = x.y; v[2] = x.z; v[3] = x.w; v[4] = y.x; v[5] = y.y; v[6] = y.z; v[7] = y.w; }
__global__ void __launch_bounds__(1024) resid(const __nv_bfloat16* a, int na, const float* ga, const float* s, const float* al, const float* be,
                                              const float* gn, float* so, __nv_bfloat16* out) {
  __shared__ float red[32];
  const int r = blockIdx.x, c0 = threadIdx.x * 8;
  float x[8], sv[8], av[8], bv[8], gv[8], gav[8];
  const bool in = c0 < na;
  uint4 araw = make_uint4(0, 0, 0, 0);
  if (in) araw = __ldg((const uint4*)(a + (size_t)r * na + c0));
  ld8(s + (size_t)r * H + c0, sv); ld8(al + c0, av); ld8(be + c0, bv); ld8(gn + c0, gv);
  if (ga && in) ld8(ga + c0, gav);
  const __nv_bfloat16* ab = (const __nv_bfloat16*)&araw;
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < 8; i++) { x[i] = in ? __bfloat162float(ab[i]) * (ga ? gav[i] : 1.f) : 0.f; ss += x[i] * x[i]; }
  const float ra = rsqrtf(bsum(ss, red) / H + 1e-8f);
  float v[8], s2 = 0.f;
#pragma unroll
  for (int i = 0; i < 8; i++) { v[i] = av[i] * sv[i] + bv[i] * (x[i] * ra); s2 += v[i] * v[i]; }
  const float rs = rsqrtf(bsum(s2, red) / H + 1e-5f);
  __nv_bfloat16 ob[8];
#pragma unroll
  for (int i = 0; i < 8; i++) ob[i] = __float2bfloat16(v[i] * rs * gv[i]);
  float4* sp = (float4*)(so + (size_t)r * H + c0);
  sp[0] = make_float4(v[0], v[1], v[2], v[3]); sp[1] = make_float4(v[4], v[5], v[6], v[7]);
  *(uint4*)(out + (size_t)r * H + c0) = *(uint4*)ob;
}
// out = bf16(rms(x) * g) over n = 4096 columns, 512 threads
__global__ void __launch_bounds__(512) rmsg(const __nv_bfloat16* x, const float* g, __nv_bfloat16* out) {
  __shared__ float red[32];
  if (threadIdx.x < 32) red[threadIdx.x] = 0.f;
  const int r = blockIdx.x, c0 = threadIdx.x * 8;
  float v[8], ss = 0.f;
#pragma unroll
  for (int i = 0; i < 8; i++) { v[i] = __bfloat162float(x[(size_t)r * 4096 + c0 + i]); ss += v[i] * v[i]; }
#pragma unroll
  for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  __syncthreads();
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
  __syncthreads();
  float t = 0.f;
#pragma unroll
  for (int i = 0; i < 16; i++) t += red[i];
  const float rr = rsqrtf(t / 4096 + 1e-5f);
#pragma unroll
  for (int i = 0; i < 8; i++) out[(size_t)r * 4096 + c0 + i] = __float2bfloat16(v[i] * rr * g[c0 + i]);
}
// routing: one warp per token. sigmoid(0.1767767 * logit); top 8 of score + bias; renormalized picked scores
__global__ void route(const float* lg, const float* bias, const int* lstart, int T, int* ids, int* loc, float* tw) {
  const int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (r >= T) return;
  float sc[8], key[8];
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const int e = i * 32 + lane;
    sc[i] = 1.f / (1.f + expf(-lg[(size_t)r * 256 + e] * 0.1767766952966369f));
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
    ids[r * 8 + lane] = myid;
    loc[r * 8 + lane] = (myid - ls) & 255;
    tw[r * 8 + lane] = pick / (sum + 1e-15f);
  }
}
void routing(torch::Tensor lg, torch::Tensor bias, torch::Tensor ls, torch::Tensor ids, torch::Tensor loc, torch::Tensor tw) {
  const int T = lg.size(0);
  route<<<(T + 3) / 4, 128, 0, at::cuda::getCurrentCUDAStream()>>>(lg.data_ptr<float>(), bias.data_ptr<float>(), ls.data_ptr<int>(), T,
      ids.data_ptr<int>(), loc.data_ptr<int>(), tw.data_ptr<float>());
}
// prefetch byte ranges into L2 (fire and forget): grid-stride over 32 KB chunks
__global__ void l2pf(const char* p0, long long n0, const char* p1, long long n1, const char* p2, long long n2) {
  const long long C = 32768;
  const long long c0 = (n0 + C - 1) / C, c1 = (n1 + C - 1) / C, c2 = (n2 + C - 1) / C;
  for (long long i = blockIdx.x * blockDim.x + threadIdx.x; i < c0 + c1 + c2; i += (long long)gridDim.x * blockDim.x) {
    const char* p; long long off, n;
    if (i < c0) { p = p0; off = i * C; n = n0; } else if (i < c0 + c1) { p = p1; off = (i - c0) * C; n = n1; } else { p = p2; off = (i - c0 - c1) * C; n = n2; }
    const unsigned sz = (unsigned)min(C, n - off);
#if __CUDA_ARCH__ >= 900
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" :: "l"(p + off), "r"(sz & ~15u) : "memory");
#else
    for (unsigned k = 0; k < sz; k += 128) asm volatile("prefetch.global.L2 [%0];\n" :: "l"(p + off + k));
#endif
  }
}
__global__ void spin(long long ns) { long long t0; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0)); long long t; do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); } while (t - t0 < ns); }
void prefetch(torch::Tensor a, torch::Tensor b, torch::Tensor c, int64_t nb, int64_t grid) {
  l2pf<<<grid, 32, 0, at::cuda::getCurrentCUDAStream()>>>((const char*)a.data_ptr(), a.numel() * a.element_size(), (const char*)b.data_ptr(),
      b.numel() * b.element_size(), (const char*)c.data_ptr(), min((long long)nb, (long long)(c.numel() * c.element_size())));
}
void spinfor(int64_t ns) { spin<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(ns); }
void resid1(torch::Tensor a, torch::Tensor s, torch::Tensor al, torch::Tensor be, torch::Tensor gn, torch::Tensor so, torch::Tensor out) {
  resid<<<a.size(0), 1024, 0, at::cuda::getCurrentCUDAStream()>>>((const __nv_bfloat16*)a.data_ptr(), a.size(1), nullptr,
      s.data_ptr<float>(), al.data_ptr<float>(), be.data_ptr<float>(), gn.data_ptr<float>(), so.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr());
}
void resid2(torch::Tensor a, torch::Tensor ga, torch::Tensor s, torch::Tensor al, torch::Tensor be, torch::Tensor gn, torch::Tensor so, torch::Tensor out) {
  resid<<<a.size(0), 1024, 0, at::cuda::getCurrentCUDAStream()>>>((const __nv_bfloat16*)a.data_ptr(), a.size(1), ga.data_ptr<float>(),
      s.data_ptr<float>(), al.data_ptr<float>(), be.data_ptr<float>(), gn.data_ptr<float>(), so.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr());
}
void rmsgain(torch::Tensor x, torch::Tensor g, torch::Tensor out) {
  rmsg<<<x.size(0), 512, 0, at::cuda::getCurrentCUDAStream()>>>((const __nv_bfloat16*)x.data_ptr(), g.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr());
}
