// Does holding one MoE weight tile across token passes beat re-streaming it?
// 3090 Ti: 84 CTAs x 36 KB = 3 MB, fits 6 MB L2, so re-streamed passes are L2 hits.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

#define CK(x) do { cudaError_t e=(x); if(e){fprintf(stderr,"%d %s\n",__LINE__,cudaGetErrorString(e));exit(1);} } while(0)
constexpr int TILE = 8 * 4608, SLICE = 4608, WARPS = 8, PASSES = 4;

__device__ __forceinline__ unsigned ldsu(unsigned a) { unsigned v; asm volatile("ld.shared.u32 %0,[%1];\n":"=r"(v):"r"(a)); return v; }

template <bool RESIDENT>
__global__ void reuse(const unsigned char* w, int ntiles, float* sink) {
  extern __shared__ __align__(16) unsigned char sm[];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const unsigned base = (unsigned)__cvta_generic_to_shared(sm);
  const unsigned ring = base + warp * 2 * SLICE;
  const unsigned tile = base + warp * SLICE;
  float acc = 0.f;
  for (int t = blockIdx.x; t < ntiles; t += gridDim.x) {
    const unsigned char* src = w + (size_t)t * TILE + warp * SLICE;
    auto issue = [&](unsigned dst) {
      for (int i = lane; i < SLICE / 16; i += 32) {
        asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n" ::
                     "r"(dst + i * 16), "l"(src + i * 16));
      }
      asm volatile("cp.async.commit_group;\n");
    };
    if constexpr (RESIDENT) {
      issue(tile);
      asm volatile("cp.async.wait_group 0;\n");
      __syncwarp();
      for (int p = 0; p < PASSES; p++)
        for (int i = lane; i < SLICE / 16; i += 32) acc += __uint_as_float(ldsu(tile + i * 16));
    } else {
      for (int p = 0; p < PASSES; p++) {
        const unsigned dst = ring + (p & 1) * SLICE;
        issue(dst);
        asm volatile("cp.async.wait_group 0;\n");
        __syncwarp();
        for (int i = lane; i < SLICE / 16; i += 32) acc += __uint_as_float(ldsu(dst + i * 16));
      }
    }
  }
  if (acc == 1234.f) sink[0] = acc;
}

float run(bool resident, const unsigned char* d, int ntiles, float* sink) {
  const int smem = resident ? WARPS * SLICE : WARPS * 2 * SLICE;
  auto launch = [&] {
    if (resident) reuse<true><<<84, 256, smem>>>(d, ntiles, sink);
    else reuse<false><<<84, 256, smem>>>(d, ntiles, sink);
  };
  CK(cudaFuncSetAttribute(reuse<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, WARPS * 2 * SLICE));
  CK(cudaFuncSetAttribute(reuse<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, WARPS * 2 * SLICE));
  launch(); CK(cudaDeviceSynchronize());
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  float best = 1e30f;
  for (int r = 0; r < 8; r++) {
    CK(cudaEventRecord(a)); launch(); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms, a, b)); best = std::min(best, ms);
  }
  printf("reuse %-8s  %7.2f ms  issued %.1f MB  useful %.1f MB\n", resident ? "resident" : "stream",
         best, (double)ntiles * TILE * (resident ? 1 : PASSES) / 1e6, (double)ntiles * TILE / 1e6);
  return best;
}

int main() {
  const int ntiles = 1024;
  unsigned char* d; float* sink;
  CK(cudaMalloc(&d, (size_t)ntiles * TILE)); CK(cudaMemset(d, 1, (size_t)ntiles * TILE));
  CK(cudaMalloc(&sink, 4));
  float a = run(false, d, ntiles, sink);
  float b = run(true, d, ntiles, sink);
  printf("speedup resident/stream %.2fx\n", a / b);
  cudaFree(d); cudaFree(sink);
  return 0;
}
