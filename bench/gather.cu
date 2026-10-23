// random-row gather throughput: each warp streams `per` random rows of R bytes via cp.async into an 8-deep smem ring
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>
template <int R>
__global__ void gath(const uint8_t* base, const int* idx, int per, float* out) {
  extern __shared__ __align__(16) uint8_t sm[];
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, gw = blockIdx.x * 8 + wid;
  constexpr int LPR = R / 16, RPI = 32 / LPR;  // lanes per row, rows per instruction
  const uint32_t ring = (uint32_t)__cvta_generic_to_shared(sm) + wid * 8 * 2048;
  const int* ix = idx + (size_t)gw * per;
  float acc = 0.f;
  auto issue = [&](int slot, int blk) {
    for (int i = 0; i < 2048 / 512; i++) {
      const int rr = blk * (2048 / R) + i * RPI + lane / LPR;
      const int row = ix[rr % per];
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(ring + slot * 2048 + i * 512 + lane * 16), "l"(base + (size_t)row * R + (lane % LPR) * 16));
    }
    asm volatile("cp.async.commit_group;\n");
  };
  const int nb = per * R / 2048;
  for (int i = 0; i < 7; i++) issue(i, i);
  for (int b = 0; b < nb; b++) {
    asm volatile("cp.async.wait_group 6;\n");
    issue((b + 7) % 8, b + 7);
    uint32_t v; asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(ring + (b % 8) * 2048 + lane * 16));
    acc += __uint_as_float(v);
  }
  if (acc == 1234.f) out[0] = acc;
}
double run(torch::Tensor base, torch::Tensor idx, int64_t R, int64_t per, int64_t reps) {
  int dev; cudaGetDevice(&dev); int sms; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  const int warps = idx.numel() / per, blocks = warps / 8;
  auto st = at::cuda::getCurrentCUDAStream();
  float* o; cudaMalloc(&o, 4);
  auto k = R == 64 ? gath<64> : (R == 128 ? gath<128> : gath<256>);
  cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, 8 * 8 * 2048);
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  k<<<blocks, 256, 8 * 8 * 2048, st>>>(base.data_ptr<uint8_t>(), idx.data_ptr<int>(), per, o);
  cudaEventRecord(a, st);
  for (int r = 0; r < reps; r++) k<<<blocks, 256, 8 * 8 * 2048, st>>>(base.data_ptr<uint8_t>(), idx.data_ptr<int>(), per, o);
  cudaEventRecord(b, st); cudaEventSynchronize(b);
  float ms; cudaEventElapsedTime(&ms, a, b); cudaFree(o);
  return (double)warps * per * R * reps / (ms * 1e-3) / 1e9;
}
