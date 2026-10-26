"""random 128 B row gathers on Hopper: cp.async 16 B (whole rows / half rows from two warps, +-L2::128B hint) vs
cp.async.bulk 128 B per row from one warp (mbarrier complete_tx); ring depth sweeps. run(): prints GB lines."""
import torch
from torch.utils.cpp_extension import load_inline

SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>
// CTA = 256 threads streams `per` random rows of K and of V (two 128 B rows per key) through an RS-slot ring of 64 keys
// mode 0: cp.async, 8 lanes x 16 B per row (whole rows); 1: two warpgroups each copy a 64 B half; 2: 1 + L2::128B;
// 3: cp.async.bulk 128 B per row by warp 0 (2 rows per lane per K/V) on one mbarrier per slot
template <int MODE, int RS>
__global__ void __launch_bounds__(256, 2) gk(const uint8_t* kc, const uint8_t* vc, const int* idx, int per, float* out) {
  extern __shared__ __align__(128) uint8_t sm[];
  __shared__ __align__(8) uint64_t bar[8];
  const int tid = threadIdx.x;
  const uint32_t ring = (uint32_t)__cvta_generic_to_shared(sm);
  const int* ixg = idx + (size_t)blockIdx.x * per;
  int* ix = (int*)(sm + RS * 16384);  // this CTA's row indices, staged once (the attention kernel prefetches them 7 steps ahead)
  for (int i = tid; i < per; i += 256) ix[i] = ixg[i];
  const int nb = per / 64;
  if (MODE == 3 && tid == 0) for (int s = 0; s < RS; s++) asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"((uint32_t)__cvta_generic_to_shared(bar + s)));
  __syncthreads();
  auto issue = [&](int slot, int st) {
    if (st >= nb) { if (MODE != 3) asm volatile("cp.async.commit_group;\n"); return; }
    if (MODE == 3) {
      if (tid < 32) {
        const uint32_t mb = (uint32_t)__cvta_generic_to_shared(bar + slot);
        if (tid == 0) asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"(mb), "r"(16384) : "memory");
        __syncwarp();
#pragma unroll
        for (int j = 0; j < 2; j++) {
          const int key = j * 32 + tid, row = ix[st * 64 + key];
          asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], 128, [%2];" :: "r"(ring + slot * 16384 + key * 128), "l"(kc + (size_t)row * 128), "r"(mb) : "memory");
          asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], 128, [%2];" :: "r"(ring + slot * 16384 + 8192 + key * 128), "l"(vc + (size_t)row * 128), "r"(mb) : "memory");
        }
      }
      return;
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
      int key, part, c8;
      if (MODE == 0) { key = i * 16 + (tid >> 4); part = (tid >> 3) & 1; c8 = tid & 7; }
      else { const int wt = tid & 127, h = tid >> 7; key = i * 16 + (wt >> 3); part = (wt >> 2) & 1; c8 = 4 * h + (wt & 3); }
      const int row = ix[st * 64 + key];
      const uint32_t dd = ring + slot * 16384 + part * 8192 + key * 128 + c8 * 16;
      const uint8_t* src = (part ? vc : kc) + (size_t)row * 128 + c8 * 16;
      if (MODE == 2) asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" :: "r"(dd), "l"(src));
      else asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dd), "l"(src));
    }
    asm volatile("cp.async.commit_group;\n");
  };
  for (int s = 0; s < RS - 1; s++) issue(s, s);
  float acc = 0.f;
  for (int b = 0; b < nb; b++) {
    if (MODE == 3) {
      const uint32_t mb = (uint32_t)__cvta_generic_to_shared(bar + b % RS);
      const uint32_t ph = (b / RS) & 1;
      asm volatile("{\n.reg .pred p;\nW: mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n@!p bra W;\n}\n" :: "r"(mb), "r"(ph) : "memory");
    } else asm volatile("cp.async.wait_group %0;\n" :: "n"(RS - 2));
    __syncthreads();
    issue((b + RS - 1) % RS, b + RS - 1);
    uint32_t v; asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(ring + (b % RS) * 16384 + tid * 64));
    acc += __uint_as_float(v);
  }
  if (acc == 1234.5f) out[0] = acc;
}
double run(torch::Tensor kc, torch::Tensor vc, torch::Tensor idx, int64_t mode, int64_t rs, int64_t ctas, int64_t reps, torch::Tensor fl) {
  const int per = idx.numel() / ctas;
  auto st = at::cuda::getCurrentCUDAStream();
  float* o; cudaMalloc(&o, 4);
  void (*k)(const uint8_t*, const uint8_t*, const int*, int, float*) = nullptr;
#define KS(M, R) if (mode == M && rs == R) k = gk<M, R>;
  KS(0, 4) KS(1, 4) KS(2, 4) KS(3, 4) KS(0, 3) KS(1, 3) KS(2, 3) KS(3, 3) KS(3, 8) KS(0, 8) KS(0, 2) KS(1, 2)
  if (!k) return -1;
  const int smb = rs * 16384 + per * 4;
  cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smb);
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  float best = 1e30f;
  for (int r = 0; r < reps; r++) {
    fl.mul_(1.0001f);
    cudaEventRecord(a, st);
    k<<<ctas, 256, smb, st>>>(kc.data_ptr<uint8_t>(), vc.data_ptr<uint8_t>(), idx.data_ptr<int>(), per, o);
    cudaEventRecord(b, st); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b); best = ms < best ? ms : best;
  }
  cudaFree(o);
  if (cudaGetLastError() != cudaSuccess) return -2;
  return (double)idx.numel() * 256 / (best * 1e-3) / 1e9;
}
"""
_m = []


def run():
    if not _m:
        _m.append(load_inline("gbench_v49", cpp_sources="double run(torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t,torch::Tensor);",
                              cuda_sources=SRC, functions=["run"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False))
    m = _m[0]
    rows = 1376016  # B16 pool
    kc = torch.randint(0, 255, (rows * 128,), dtype=torch.uint8, device="cuda"); vc = kc.clone()
    fl = torch.ones(64 << 20, device="cuda")
    for ctas, per, cfgs in ((264, 4864, ((0, 2), (1, 2), (0, 3), (1, 3), (0, 4), (1, 4), (2, 4), (3, 4))),
                            (132, 9728, ((0, 4), (2, 4), (3, 4), (0, 8), (3, 8))), (396, 3264, ((0, 2), (1, 2), (0, 3), (1, 3), (2, 3)))):
        idx = torch.randint(0, rows, (ctas * per,), dtype=torch.int32, device="cuda")  # ~1.28M keys: the v16f volume
        for mode, rs in cfgs:
            try:
                print("GB ctas %d mode %d rs %d %.0f GB/s" % (ctas, mode, rs, m.run(kc, vc, idx, mode, rs, ctas, 5, fl)), flush=True)
            except Exception as e:
                print("GB ctas %d mode %d rs %d ERR %s" % (ctas, mode, rs, str(e)[:60]), flush=True)
