// Paged-cache gather microbenchmark: row size vs access runs.
// Build: nvcc -O3 -arch=sm_86 gather.cu -o gather
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>

#define CK(x) do { cudaError_t e=(x); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} } while(0)

template <int R>
__global__ void gather(const unsigned char* base, const int* idx, int per, float* sink) {
  extern __shared__ __align__(16) unsigned char sm[];
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  constexpr int LPR = R / 16, RPI = 32 / LPR;
  const unsigned ring = (unsigned)__cvta_generic_to_shared(sm) + wid * 4 * 2048;
  const int* ix = idx + (size_t)(blockIdx.x * 8 + wid) * per;
  float acc = 0.f;
  auto issue = [&](int slot, int blk) {
    if (blk < per * R / 2048) {
      for (int i = 0; i < 4; i++) {
        const int rr = blk * (2048 / R) + i * RPI + lane / LPR;
        const int row = ix[rr % per];
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
                     "r"(ring + slot * 2048 + i * 512 + lane * 16),
                     "l"(base + (size_t)row * R + (lane % LPR) * 16));
      }
    }
    asm volatile("cp.async.commit_group;\n");
  };
  const int nb = per * R / 2048;
  for (int i = 0; i < 3; i++) issue(i, i);
  for (int b = 0; b < nb; b++) {
    asm volatile("cp.async.wait_group 2;\n");
    issue((b + 3) & 3, b + 3);
    unsigned v;
    asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(v) : "r"(ring + (b & 3) * 2048 + lane * 16));
    acc += __uint_as_float(v);
  }
  if (acc == 1234.f) sink[0] = acc;
}

template <int R>
float run(const char* variant, const std::vector<int>& idx, int per, int warps, int reps) {
  const int rows = 1 << 20;
  unsigned char* d = nullptr;
  int* di = nullptr;
  float* sink = nullptr;
  CK(cudaMalloc(&d, (size_t)rows * R));
  CK(cudaMemset(d, 1, (size_t)rows * R));
  CK(cudaMalloc(&di, idx.size() * sizeof(int)));
  CK(cudaMemcpy(di, idx.data(), idx.size() * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&sink, 4));
  const int blocks = warps / 8;
  const int smem = 8 * 4 * 2048;  // 64 KiB
  CK(cudaFuncSetAttribute(gather<R>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  auto launch = [&] { gather<R><<<blocks, 256, smem>>>(d, di, per, sink); };
  launch(); CK(cudaDeviceSynchronize());
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  float best = 1e30f;
  for (int r = 0; r < reps; r++) {
    CK(cudaEventRecord(a)); launch(); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms, a, b)); best = std::min(best, ms);
  }
  const double bytes = (double)warps * per * R;
  const double gbs = bytes / (best * 1e-3) / 1e9;
  printf("gather R=%3d %-7s warps=%4d  %7.2f GB/s  (%.2f ms)\n", R, variant, warps, gbs, best);
  cudaFree(d); cudaFree(di); cudaFree(sink);
  return (float)gbs;
}

template <int R>
void one(const char* variant, int warps, int reps) {
  const int per = 65536 / R;  // 64 KiB per warp
  std::mt19937 rng(7);
  std::vector<int> idx((size_t)warps * per);
  const int rows = 1 << 20;
  if (!strcmp(variant, "random")) {
    for (auto& x : idx) x = rng() % rows;
  } else if (!strcmp(variant, "runs4") || !strcmp(variant, "runs16")) {
    int run = !strcmp(variant, "runs4") ? 4 : 16;
    for (size_t i = 0; i < idx.size(); i += run) {
      int base = (rng() % (rows / run)) * run;
      for (int j = 0; j < run && i + j < idx.size(); j++) idx[i + j] = base + j;
    }
  } else {
    for (size_t i = 0; i < idx.size(); i++) idx[i] = (int)(i % rows);
  }
  run<R>(variant, idx, per, warps, reps);
}

int main() {
  const int warps = 84 * 8;  // 8 blocks/SM x 8 warps; one CTA/SM for 64 KiB
  for (int r : {64, 128, 256}) {
    if (r == 64) { one<64>("random", warps, 8); one<64>("runs4", warps, 8); one<64>("runs16", warps, 8); one<64>("seq", warps, 8); }
    if (r == 128) { one<128>("random", warps, 8); one<128>("runs4", warps, 8); one<128>("runs16", warps, 8); one<128>("seq", warps, 8); }
    if (r == 256) { one<256>("random", warps, 8); one<256>("runs4", warps, 8); one<256>("runs16", warps, 8); one<256>("seq", warps, 8); }
  }
  return 0;
}
