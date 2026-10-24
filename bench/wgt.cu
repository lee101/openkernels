// wgmma layout unit test: S = Q K^T (A, B K-major SW128 from smem), O = f16(S) V (A from registers, B = V MN-major SW128)
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <stdint.h>
__device__ __forceinline__ uint64_t desc(uint32_t saddr, uint32_t lbo, uint32_t sbo, uint32_t sw) {
  return (uint64_t)((saddr & 0x3FFFF) >> 4) | ((uint64_t)(lbo >> 4) << 16) | ((uint64_t)(sbo >> 4) << 32) | ((uint64_t)sw << 62);
}
#define WG_S(d) "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), \
  "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
#define WG_D "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}"
__device__ __forceinline__ void wg_ss(float* d, uint64_t da, uint64_t db) {
  asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 " WG_D ", %32, %33, 1, 1, 1, 0, 0;\n" : WG_S(d) : "l"(da), "l"(db));
}
template <int TB>
__device__ __forceinline__ void wg_rs(float* d, const uint32_t* a, uint64_t db) {
  asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 " WG_D ", {%32,%33,%34,%35}, %36, 1, 1, 1, %37;\n"
               : WG_S(d) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "n"(TB));
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_wait0() { asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory"); }
__device__ __forceinline__ uint32_t pk(float a, float b) { __half2 h = __floats2half2_rn(a, b); return *(uint32_t*)&h; }

// q, k, v: [64][64] f16 row-major. smem rows of 128 B with 16 B chunk c at c ^ (row & 7)
__global__ void wgt(const __half* q, const __half* k, const __half* v, float* S, float* O, int vl, int vs, int vtr) {
  __shared__ __align__(1024) uint8_t sm[3 * 8192];
  const int tid = threadIdx.x;
  for (int i = tid; i < 3 * 512; i += 128) {
    const int m = i / 512, r = (i % 512) / 8, c = i % 8;
    const __half* src = (m == 0 ? q : m == 1 ? k : v) + r * 64 + c * 8;
    *(uint4*)(sm + m * 8192 + r * 128 + ((c ^ (r & 7)) << 4)) = *(const uint4*)src;
  }
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  __syncthreads();
  const uint32_t base = (uint32_t)__cvta_generic_to_shared(sm);
  float s[32];
  for (int i = 0; i < 32; i++) s[i] = 0.f;
  wg_fence();
  for (int kk = 0; kk < 4; kk++) wg_ss(s, desc(base + kk * 32, 16, 1024, 1), desc(base + 8192 + kk * 32, 16, 1024, 1));
  wg_commit(); wg_wait0();
  const int w = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  for (int j = 0; j < 8; j++)
    for (int e = 0; e < 4; e++) {
      const int row = 16 * w + g + (e >> 1) * 8, col = 8 * j + 2 * t + (e & 1);
      S[row * 64 + col] = s[4 * j + e];
    }
  float o[32];
  for (int i = 0; i < 32; i++) o[i] = 0.f;
  wg_fence();
  for (int kc = 0; kc < 4; kc++) {
    const uint32_t a[4] = {pk(s[8 * kc], s[8 * kc + 1]), pk(s[8 * kc + 2], s[8 * kc + 3]), pk(s[8 * kc + 4], s[8 * kc + 5]), pk(s[8 * kc + 6], s[8 * kc + 7])};
    const uint64_t db = desc(base + 16384 + kc * (vtr ? 2048 : 32), vl, vs, 1);
    if (vtr) wg_rs<1>(o, a, db); else wg_rs<0>(o, a, db);
  }
  wg_commit(); wg_wait0();
  for (int j = 0; j < 8; j++)
    for (int e = 0; e < 4; e++) {
      const int row = 16 * w + g + (e >> 1) * 8, col = 8 * j + 2 * t + (e & 1);
      O[row * 64 + col] = o[4 * j + e];
    }
}
void run(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor S, torch::Tensor O, int64_t vl, int64_t vs, int64_t vtr) {
  wgt<<<1, 128, 0, at::cuda::getCurrentCUDAStream()>>>((const __half*)q.data_ptr(), (const __half*)k.data_ptr(), (const __half*)v.data_ptr(),
      S.data_ptr<float>(), O.data_ptr<float>(), vl, vs, vtr);
}
