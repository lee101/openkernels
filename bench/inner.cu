// Incremental inner-loop microbenchmark for the MoE W4A16 decode kernel on a TMA ring.
// V0 LDS only | V1 +scales | V2 +dequant | V3 +mma (B const) | V4 +B ldg 1 row | V5 +B ldg 8 rows | V6 +B 8 rows ldca
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>
#define STG 4608
__device__ __forceinline__ void mb_init(uint32_t b) { asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(b)); }
__device__ __forceinline__ void tma_ld(uint32_t dst, const void* src, uint32_t bar) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(bar), "n"(STG) : "memory");
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" :: "r"(dst), "l"(src), "n"(STG), "r"(bar) : "memory");
}
__device__ __forceinline__ void mb_wait(uint32_t bar, uint32_t ph) {
  asm volatile("{\n.reg .pred p;\nW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W;\n}\n" :: "r"(bar), "r"(ph) : "memory");
}
__device__ __forceinline__ uint4 lds(uint32_t a) { uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a)); return v; }
__device__ __forceinline__ uint32_t bmul(uint32_t a, uint32_t b) { uint32_t d; asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(0x80008000u)); return d; }
__device__ __forceinline__ void mma(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t rotl(uint32_t q, int d) { return __funnelshift_l(q, q, d); }
__device__ __forceinline__ uint32_t sc2(uint32_t w, uint32_t sel) { return __byte_perm(w, 0, sel) * 16u + 0x77007700u; }

template <int V, int WPC, int S>
__global__ void __launch_bounds__(WPC * 32, 1) inner(const uint8_t* src, size_t nstage, const uint4* xb, float* out) {
  extern __shared__ __align__(128) uint8_t sm[];
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int W = gridDim.x * WPC, gw = blockIdx.x * WPC + wid;
  const uint32_t base = (uint32_t)__cvta_generic_to_shared(sm) + wid * S * STG, bars = (uint32_t)__cvta_generic_to_shared(sm) + WPC * S * STG + wid * S * 8;
  if (lane < S) mb_init(bars + lane * 8);
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  __syncwarp();
  const size_t per = nstage / W, b0 = gw * per;
  const uint4* rp = xb + (V >= 5 ? g : 0) * 512 + t * 2;  // 8 distinct 8KB rows (V5+) or one row
  float acc[4][4] = {};
  uint32_t xs = 0;
  for (int i = 0; i < S - 1 && i < (int)per; i++) if (lane == 0) tma_ld(base + i * STG, src + (b0 + i) * STG, bars + i * 8);
  for (size_t q = 0; q < per; q++) {
    const int slot = q % S;
    mb_wait(bars + slot * 8, (q / S) & 1);
    __syncwarp();
    if (q + S - 1 < per && lane == 0) tma_ld(base + ((q + S - 1) % S) * STG, src + (b0 + q + S - 1) * STG, bars + ((q + S - 1) % S) * 8);
    const uint32_t s = base + slot * STG + lane * 16;
    const uint4 sc = lds(s + 4096);
    const int ks = (int)(q & 7) * 64;
#pragma unroll
    for (int c = 0; c < 8; c++) {
      const uint4 w = lds(s + c * 512);
      uint32_t sA = 0x3f803f80u, sB = 0x3f803f80u;
      if (V >= 1) { const uint32_t sw = (c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w)); sA = sc2(sw, (c & 1) ? 0x4242 : 0x4040); sB = sc2(sw, (c & 1) ? 0x4343 : 0x4141); }
      uint4 bb0 = make_uint4(0x3f803f80u, 0, 0x3f803f80u, 0), bb1 = bb0;
      if (V >= 6) { bb0 = __ldca(rp + ks + c * 8); bb1 = __ldca(rp + ks + c * 8 + 1); }
      else if (V >= 4) { bb0 = __ldg(rp + ks + c * 8); bb1 = __ldg(rp + ks + c * 8 + 1); }
      const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
      const uint32_t* b = (const uint32_t*)&bb0; const uint32_t* bh = (const uint32_t*)&bb1;
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const uint32_t q4 = ww[j];
        uint32_t a0, a1, a2, a3;
        if (V >= 2) {
          a0 = bmul(q4 & 0x81C081C0u, sA);
          a1 = bmul((rotl(q4, 3) & 0x01C001C0u) | (rotl(q4, 1) & 0x80008000u), sB);
          a2 = bmul(rotl(q4, 6) & 0x81C081C0u, sA);
          a3 = bmul((rotl(q4, 11) & 0x01C001C0u) | (rotl(q4, 5) & 0x80008000u), sB);
        } else { a0 = q4 ^ sA; a1 = q4 ^ sB; a2 = q4; a3 = q4 + 1; }
        if (V >= 3) mma(acc[j], a0, a1, a2, a3, (j < 2 ? b : bh)[(j & 1) * 2], (j < 2 ? b : bh)[(j & 1) * 2 + 1]);
        else xs ^= a0 ^ a1 ^ a2 ^ a3 ^ b[j & 1] ^ bh[j & 1];
      }
    }
  }
  float r = __uint_as_float(xs);
#pragma unroll
  for (int j = 0; j < 4; j++) r += acc[j][0] + acc[j][1] + acc[j][2] + acc[j][3];
  if (r == 1234.5f) out[0] = r;
}

template <int V, int WPC, int S>
static double go(torch::Tensor buf, torch::Tensor xb, int reps) {
  int dev; cudaGetDevice(&dev); int sms; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  size_t smem = (size_t)WPC * S * STG + WPC * S * 8;
  if (cudaFuncSetAttribute(inner<V, WPC, S>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) != cudaSuccess) return -1;
  size_t nstage = buf.numel() / STG;
  auto st = at::cuda::getCurrentCUDAStream();
  float* out; cudaMalloc(&out, 4);
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  inner<V, WPC, S><<<sms, WPC * 32, smem, st>>>(buf.data_ptr<uint8_t>(), nstage, (const uint4*)xb.data_ptr(), out);
  cudaEventRecord(a, st);
  for (int r = 0; r < reps; r++) inner<V, WPC, S><<<sms, WPC * 32, smem, st>>>(buf.data_ptr<uint8_t>(), nstage, (const uint4*)xb.data_ptr(), out);
  cudaEventRecord(b, st); cudaEventSynchronize(b);
  bool bad = cudaGetLastError() != cudaSuccess;
  float ms; cudaEventElapsedTime(&ms, a, b); cudaFree(out);
  if (bad) return -1;
  size_t W = (size_t)sms * WPC;
  return (nstage / W) * W * (double)STG * reps / (ms * 1e-3) / 1e9;
}

double run(torch::Tensor buf, torch::Tensor xb, int64_t v, int64_t cfg, int64_t reps) {
#define CASE(VV) \
  if (v == VV) { if (cfg == 0) return go<VV, 16, 2>(buf, xb, reps); if (cfg == 1) return go<VV, 8, 4>(buf, xb, reps); if (cfg == 2) return go<VV, 12, 3>(buf, xb, reps); if (cfg == 3) return go<VV, 8, 2>(buf, xb, reps); }
  CASE(0) CASE(1) CASE(2) CASE(3) CASE(4) CASE(5) CASE(6)
  return -2;
}
