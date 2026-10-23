// TMA bulk streaming microbenchmark: each warp streams its own range in STG-byte stages through an S-deep smem ring.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>
__device__ __forceinline__ void mb_init(uint32_t b) { asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(b)); }
__device__ __forceinline__ void tma_ld(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(bar), "r"(bytes) : "memory");
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void mb_wait(uint32_t bar, uint32_t ph) {
  asm volatile("{\n.reg .pred p;\nW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W;\n}\n" :: "r"(bar), "r"(ph) : "memory");
}
// mode 0: warp-contiguous ranges; mode 1: stage-interleaved (stage i of warp w at (i*W + w))
__global__ void stream(const uint8_t* src, size_t nstage, int stg, int S, int mode, int spin, float* out) {
  extern __shared__ __align__(128) uint8_t sm[];
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, wpc = blockDim.x >> 5;
  const int W = gridDim.x * wpc, gw = blockIdx.x * wpc + wid;
  const uint32_t base = (uint32_t)__cvta_generic_to_shared(sm) + wid * S * stg, bars = (uint32_t)__cvta_generic_to_shared(sm) + wpc * S * stg + wid * S * 8;
  if (lane < S) mb_init(bars + lane * 8);
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  __syncwarp();
  const size_t per = nstage / W, b0 = mode ? 0 : gw * per;
  auto addr = [&](size_t i) { return src + (mode ? (i * W + gw) : (b0 + i)) * (size_t)stg; };
  float acc = 0.f;
  for (int i = 0; i < S - 1 && i < (int)per; i++) if (lane == 0) tma_ld(base + i * stg, addr(i), stg, bars + i * 8);
  for (size_t q = 0; q < per; q++) {
    const int slot = q % S;
    mb_wait(bars + slot * 8, (q / S) & 1);
    __syncwarp();
    if (q + S - 1 < per && lane == 0) tma_ld(base + ((q + S - 1) % S) * stg, addr(q + S - 1), stg, bars + ((q + S - 1) % S) * 8);
    const uint32_t s = base + slot * stg;
    for (int o = lane * 16; o < stg; o += 512) { uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(s + o)); acc += __uint_as_float(v.x ^ v.y ^ v.z ^ v.w); }
    for (int k = 0; k < spin; k++) acc = acc * 1.0001f + 0.5f;  // fake per-stage compute
  }
  if (acc == 12345.f) out[0] = acc;
}
double run(torch::Tensor buf, int64_t wpc, int64_t S, int64_t stg, int64_t mode, int64_t spin, int64_t reps) {
  if (reps == 0) {  // single capturable launch
    int dev; cudaGetDevice(&dev); int sms; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    size_t smem = (size_t)wpc * S * stg + wpc * S * 8;
    cudaFuncSetAttribute(stream, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    static float* out = nullptr; if (!out) cudaMalloc(&out, 4);
    stream<<<sms, wpc * 32, smem, at::cuda::getCurrentCUDAStream()>>>(buf.data_ptr<uint8_t>(), buf.numel() / stg, stg, S, mode, spin, out);
    return 0;
  }
  int dev; cudaGetDevice(&dev); int sms; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  size_t smem = (size_t)wpc * S * stg + wpc * S * 8;
  cudaFuncSetAttribute(stream, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  size_t nstage = buf.numel() / stg;
  auto st = at::cuda::getCurrentCUDAStream();
  float* out; cudaMalloc(&out, 4);
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  stream<<<sms, wpc * 32, smem, st>>>(buf.data_ptr<uint8_t>(), nstage, stg, S, mode, spin, out);
  cudaEventRecord(a, st);
  for (int r = 0; r < reps; r++) stream<<<sms, wpc * 32, smem, st>>>(buf.data_ptr<uint8_t>(), nstage, stg, S, mode, spin, out);
  cudaEventRecord(b, st); cudaEventSynchronize(b);
  if (cudaGetLastError() != cudaSuccess) { cudaFree(out); return -1.0; }
  float ms; cudaEventElapsedTime(&ms, a, b); cudaFree(out);
  size_t W = (size_t)sms * wpc; size_t bytes = (nstage / W) * W * stg;
  return bytes * (double)reps / (ms * 1e-3) / 1e9;  // GB/s
}
