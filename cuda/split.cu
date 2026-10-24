// Split-count scaling for paged KV reads + merge. No attention math: this
// isolates parallelism and merge overhead, which is what A3 is about.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

#define CK(x) do { cudaError_t e=(x); if(e){fprintf(stderr,"%d %s\n",__LINE__,cudaGetErrorString(e));exit(1);} } while(0)

__global__ void split_read(const uint4* kv, int span, int ns, float* part) {
  const int b = blockIdx.x / ns, sp = blockIdx.x % ns;
  const int k0 = (int)((long long)span * sp / ns), k1 = (int)((long long)span * (sp + 1) / ns);
  float s = 0.f;
  for (int k = k0 + threadIdx.x; k < k1; k += blockDim.x) {
    const uint4* p = kv + ((size_t)b * span + k) * 8;
    uint4 a = p[0], c = p[1], d = p[2], e = p[3], f = p[4], g = p[5], h = p[6], i = p[7];
    s += __uint_as_float(a.x ^ c.y ^ d.z ^ e.w ^ f.x ^ g.y ^ h.z ^ i.w);
  }
  __shared__ float sh[32];
  s += __shfl_down_sync(0xffffffffu, s, 16);
  s += __shfl_down_sync(0xffffffffu, s, 8);
  s += __shfl_down_sync(0xffffffffu, s, 4);
  s += __shfl_down_sync(0xffffffffu, s, 2);
  s += __shfl_down_sync(0xffffffffu, s, 1);
  if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = s;
  __syncthreads();
  if (threadIdx.x == 0) { float t = 0; for (int i = 0; i < 8; i++) t += sh[i]; part[blockIdx.x] = t; }
}
__global__ void merge(const float* part, int n, float* out) {
  float s = 0.f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) s += part[i];
  s += __shfl_down_sync(0xffffffffu, s, 16);
  s += __shfl_down_sync(0xffffffffu, s, 8);
  s += __shfl_down_sync(0xffffffffu, s, 4);
  s += __shfl_down_sync(0xffffffffu, s, 2);
  s += __shfl_down_sync(0xffffffffu, s, 1);
  if (threadIdx.x == 0) out[0] = s;
}
float run(const uint4* kv, int B, int span, int ns, float* part, float* out) {
  auto launch = [&] {
    split_read<<<B * ns, 256>>>(kv, span, ns, part);
    merge<<<1, 256>>>(part, B * ns, out);
  };
  launch(); CK(cudaDeviceSynchronize());
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  float best = 1e30f;
  for (int r = 0; r < 6; r++) {
    CK(cudaEventRecord(a)); launch(); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms, a, b)); best = std::min(best, ms);
  }
  const double bytes = (double)B * span * 128;
  printf("split B=%2d span=%7d ns=%3d ctas=%4d  %7.3f ms  %7.1f GB/s\n", B, span, ns, B * ns, best, bytes / (best * 1e-3) / 1e9);
  return best;
}
int main() {
  const int span = 1 << 17;  // 128K tokens
  const int B = 32;
  uint4* d; float* part; float* out;
  CK(cudaMalloc(&d, (size_t)B * span * 128)); CK(cudaMemset(d, 1, (size_t)B * span * 128));
  CK(cudaMalloc(&part, (size_t)B * 512 * 4)); CK(cudaMalloc(&out, 4));
  printf("B=32, span=80K, total KV %.1f MB\n", (double)B * 80000 * 128 / 1e6);
  for (int ns : {1, 2, 4, 8, 16}) run(d, B, 80000, ns, part, out);
  printf("B=1, span=80K (10 MB)\n");
  for (int ns : {8, 16, 32, 64, 132}) run(d, 1, 80000, ns, part, out);
  printf("B=1, span=2K (0.25 MB)\n");
  for (int ns : {2, 4, 8, 16, 32}) run(d, 1, 2048, ns, part, out);
  printf("B=1, span=1M (128 MB)\n");
  for (int ns : {8, 16, 32, 64, 132}) run(d, 1, 1 << 20, ns, part, out);
  cudaFree(d); cudaFree(part); cudaFree(out);
  return 0;
}
