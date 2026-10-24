// microbenchmarks with known H200 numbers, used to validate Accel-Sim's SM90_H200 config on sm_86 traces
// build: nvcc -O3 -arch=sm_86 ubench.cu -o ubench ; run: ./ubench stream|gather64|gather128|gather256|mma [MB] [ctas] [reps]
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#define CK(x) do { cudaError_t e = (x); if (e) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

__global__ void __launch_bounds__(256) stream(const uint4* __restrict__ p, size_t n, float* sink) {
  uint32_t acc = 0;
  for (size_t i = blockIdx.x * 256 + threadIdx.x; i < n; i += (size_t)gridDim.x * 256 * 4) {
    uint4 v[4];
#pragma unroll
    for (int u = 0; u < 4; u++) v[u] = i + u * (size_t)gridDim.x * 256 < n ? __ldcs(p + i + u * (size_t)gridDim.x * 256) : make_uint4(0, 0, 0, 0);
#pragma unroll
    for (int u = 0; u < 4; u++) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
  }
  if (acc == 0x12345678u) sink[0] = acc;
}

// each warp gathers `per` random R-byte rows into a 4-slot cp.async ring of 2 KiB stages
template <int R>
__global__ void __launch_bounds__(256) gather(const unsigned char* base, const int* idx, int per, float* sink) {
  extern __shared__ __align__(16) unsigned char sm[];
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  constexpr int LPR = R / 16, RPI = 32 / LPR;
  const unsigned ring = (unsigned)__cvta_generic_to_shared(sm) + wid * 4 * 2048;
  const int* ix = idx + (size_t)(blockIdx.x * 8 + wid) * per;
  float acc = 0.f;
  const int nb = per * R / 2048;
  auto issue = [&](int slot, int blk) {
    if (blk < nb) {
      for (int i = 0; i < 4; i++) {
        const int row = ix[(blk * (2048 / R) + i * RPI + lane / LPR) % per];
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(ring + slot * 2048 + i * 512 + lane * 16), "l"(base + (size_t)row * R + (lane % LPR) * 16));
      }
    }
    asm volatile("cp.async.commit_group;\n");
  };
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

// mma.sync m16n8k16 bf16 throughput: 4 independent accumulators per warp
__global__ void __launch_bounds__(256) mma(int iters, float* sink) {
  float c[4][4] = {};
  uint32_t a0 = threadIdx.x, a1 = a0 * 3, a2 = a0 * 5, a3 = a0 * 7, b0 = a0 * 11, b1 = a0 * 13;
  for (int i = 0; i < iters; i++) {
#pragma unroll
    for (int j = 0; j < 4; j++)
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                   : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  }
  float s = 0;
  for (int j = 0; j < 4; j++) s += c[j][0] + c[j][1] + c[j][2] + c[j][3];
  if (s == 1234.f) sink[0] = s;
}

int main(int argc, char** argv) {
  const char* k = argc > 1 ? argv[1] : "stream";
  const double mb = argc > 2 ? atof(argv[2]) : 128;
  const int ctas = argc > 3 ? atoi(argv[3]) : 528;
  const int reps = argc > 4 ? atoi(argv[4]) : 1;  // reps=1: single launch (for tracing)
  float* sink; CK(cudaMalloc(&sink, 4));
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  double bytes = 0, flops = 0;
  auto timeit = [&](auto launch) {
    float best = 1e30f;
    for (int r = 0; r < reps; r++) {
      CK(cudaEventRecord(a)); launch(); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
      float ms; CK(cudaEventElapsedTime(&ms, a, b)); if (ms < best) best = ms;
    }
    return best;
  };
  float ms = 0;
  if (!strcmp(k, "stream")) {
    size_t n = (size_t)(mb * 1e6) / 16; uint4* p; CK(cudaMalloc(&p, n * 16)); CK(cudaMemset(p, 1, n * 16));
    bytes = n * 16.0;
    ms = timeit([&] { stream<<<ctas, 256>>>(p, n, sink); });
  } else if (!strncmp(k, "gather", 6)) {
    const int R = atoi(k + 6), warps = ctas * 8, per = (int)(mb * 1e6 / warps / R) / (2048 / R) * (2048 / R);
    const int rows = 1 << 22;
    unsigned char* d; CK(cudaMalloc(&d, (size_t)rows * R)); CK(cudaMemset(d, 1, (size_t)rows * R));
    std::mt19937 rng(7); std::vector<int> idx((size_t)warps * per); for (auto& x : idx) x = rng() % rows;
    int* di; CK(cudaMalloc(&di, idx.size() * 4)); CK(cudaMemcpy(di, idx.data(), idx.size() * 4, cudaMemcpyHostToDevice));
    const int smem = 8 * 4 * 2048;
    auto kern = R == 64 ? gather<64> : (R == 128 ? gather<128> : gather<256>);
    CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    bytes = (double)warps * per * R;
    ms = timeit([&] { kern<<<ctas, 256, smem>>>(d, di, per, sink); });
  } else if (!strcmp(k, "mma")) {
    const int iters = (int)mb;  // reuse arg: iterations
    flops = (double)ctas * 8 * iters * 4 * 4096;
    ms = timeit([&] { mma<<<ctas, 256>>>(iters, sink); });
  }
  CK(cudaDeviceSynchronize());
  printf("%s ctas=%d bytes=%.0f flops=%.0f ms=%.4f GB/s=%.1f TF=%.1f\n", k, ctas, bytes, flops, ms, bytes / ms / 1e6, flops / ms / 1e9);
  return 0;
}
