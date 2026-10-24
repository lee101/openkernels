// FP8->F16 conversion vs HMMA cost. On sm86 the conversion is emulated, but the
// ratio still shows why converting per-consumer (instead of once) can bind.
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>

#define CK(x) do { cudaError_t e=(x); if(e){fprintf(stderr,"%d %s\n",__LINE__,cudaGetErrorString(e));exit(1);} } while(0)

__device__ __forceinline__ unsigned cvt2(unsigned x) {
  __half2_raw h = __nv_cvt_fp8x2_to_halfraw2((unsigned short)x, __NV_E4M3);
  return *reinterpret_cast<unsigned*>(&h);
}
__device__ __forceinline__ void mma(float* c, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

template <int CP>
__global__ void k(const unsigned* in, float* out, int iters, int mode) {
  __shared__ unsigned xs[64];
  for (int i = threadIdx.x; i < 64; i += blockDim.x) xs[i] = in[i];
  __syncthreads();
  unsigned a0 = 0, a1 = 0, a2 = 0, a3 = 0, b0 = 0, b1 = 0;
  float c[4][4] = {};
  for (int it = 0; it < iters; it++) {
    if (mode == 0) {  // MMA only
      unsigned q = xs[(threadIdx.x + it) & 63];
      mma(c[0], q, q + 1, q + 2, q + 3, q + 4, q + 5);
      mma(c[1], q, q + 1, q + 2, q + 3, q + 4, q + 5);
      mma(c[2], q, q + 1, q + 2, q + 3, q + 4, q + 5);
      mma(c[3], q, q + 1, q + 2, q + 3, q + 4, q + 5);
    } else if (mode == 1) {  // conversion only
#pragma unroll
      for (int cp = 0; cp < CP; cp++) {
#pragma unroll
        for (int i = 0; i < 8; i++) {
          unsigned v = xs[(i + cp + it) & 63];
          a0 ^= cvt2(v); a1 ^= cvt2(v >> 16);
        }
      }
      if (a0 == 0x12345678u) c[0][0] += 1.f;
    } else {  // conversion + MMA
#pragma unroll
      for (int cp = 0; cp < CP; cp++) {
        unsigned q = xs[(threadIdx.x + cp + it) & 63];
        a0 ^= cvt2(q); a1 ^= cvt2(q >> 16);
        a2 ^= cvt2(q + 1); b0 ^= cvt2(q + 2); b1 ^= cvt2(q + 3);
      }
      mma(c[0], a0, a1, a2, a3, b0, b1);
      mma(c[1], a0, a1, a2, a3, b0, b1);
      mma(c[2], a0, a1, a2, a3, b0, b1);
      mma(c[3], a0, a1, a2, a3, b0, b1);
    }
  }
  float s = 0;
  for (int i = 0; i < 4; i++) s += c[i][0] + c[i][1] + c[i][2] + c[i][3];
  if (s == 1234.5f) out[0] = s;
}

template <int CP>
void run(const char* label, int mode, int iters, int blocks, const unsigned* d, float* o) {
  auto launch = [&] { k<CP><<<blocks, 256>>>(d, o, iters, mode); };
  launch(); CK(cudaDeviceSynchronize());
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  float best = 1e30f;
  for (int r = 0; r < 5; r++) {
    CK(cudaEventRecord(a)); launch(); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms, a, b)); best = std::min(best, ms);
  }
  const double warps = (double)blocks * 8, ops = warps * iters;
  printf("cvt %-10s cp=%d  %7.1f us  %8.2f Mops/s  (per warp-iter)\n", label, CP, best * 1e3, ops / (best * 1e-3) / 1e6);
}

int main() {
  unsigned h[64]; for (int i = 0; i < 64; i++) h[i] = 0x3c3c3c3cu + i * 0x01010101u;
  unsigned* d; float* o; CK(cudaMalloc(&d, sizeof h)); CK(cudaMalloc(&o, 4));
  CK(cudaMemcpy(d, h, sizeof h, cudaMemcpyHostToDevice));
  const int it = 4096, blocks = 84 * 4;
  run<0>("mma", 0, it, blocks, d, o);
  run<1>("conv", 1, it, blocks, d, o);
  run<4>("conv", 4, it, blocks, d, o);
  run<1>("conv+mma", 2, it, blocks, d, o);
  run<2>("conv+mma", 2, it, blocks, d, o);
  run<4>("conv+mma", 2, it, blocks, d, o);
  cudaFree(d); cudaFree(o);
  return 0;
}
