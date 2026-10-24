"""wgmma bf16 GEMM for m 33..256 (bf16_proj verify shapes): out[m, n] = x[m, k] @ W[n, k]^T.
CTA = 128 weight rows (2 consumer warpgroups x 64) x all tokens (wgmma N = TN), producer warp: W via 1D bulk copies of
prepare-time SW128 tiles, x via a TMA tensor map (rows >= m zero-filled). Split-K over a thread-block cluster; partials
meet in distributed shared memory. python bench/wgg.py [check|time]"""
import os, sys, torch
from torch.utils.cpp_extension import load_inline


def wg_fn(n):
    regs = n // 2
    outs = ", ".join(f'"+f"(d[{i}])' for i in range(regs))
    lst = ",".join(f"%{i}" for i in range(regs))
    return f'''__device__ __forceinline__ void wg{n}(float* d, uint64_t da, uint64_t db, int sd) {{
  asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, %{regs + 2}, 0;\\n"
               "wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{lst}}}, %{regs}, %{regs + 1}, p, 1, 1, 0, 0;\\n}}\\n"
               : {outs} : "l"(da), "l"(db), "r"(sd));
}}
'''


CUDA = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#define NST 4
#ifndef NORED
#define NORED 0
#endif
__device__ __forceinline__ uint64_t gdesc(uint32_t s) {  // K-major SW128: 8-row atoms 1 KB apart
  return (uint64_t)((s & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)64 << 32) | (1ull << 62);
}
__device__ __forceinline__ void mbinit(uint32_t b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(b), "r"(n)); }
__device__ __forceinline__ void mbwait(uint32_t b, uint32_t ph) {
  asm volatile("{\n.reg .pred p;\nW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W;\n}\n" :: "r"(b), "r"(ph) : "memory");
}
__device__ __forceinline__ void mbarrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(b) : "memory"); }
WGFNS
template <int TN>
__global__ void __launch_bounds__(288, 1) wgg(const __grid_constant__ CUtensorMap xm, const uint8_t* __restrict__ wp, __nv_bfloat16* out,
                                              int M, int N, int KB, int KS) {
  constexpr int STAGE = 16384 + TN * 128, LD = TN + 4;
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  const uint32_t sb = ((uint32_t)__cvta_generic_to_shared(smem_raw) + 1023) & ~1023u;
  uint8_t* smem = smem_raw + (sb - (uint32_t)__cvta_generic_to_shared(smem_raw));
  const uint32_t full = sb + NST * STAGE, empty = full + NST * 8;
  const int tid = threadIdx.x, wg = tid >> 7, lane = tid & 31;
  cg::cluster_group cl = cg::this_cluster();
  const int kr = (int)cl.block_rank(), rb = blockIdx.x / KS;
  const int k0 = KB * kr / KS, k1 = KB * (kr + 1) / KS, nk = k1 - k0;
  if (tid == 0) {
    for (int s = 0; s < NST; s++) { mbinit(full + s * 8, 1); mbinit(empty + s * 8, 2); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  float d[TN / 2];
  if (wg == 2) {  // producer warp
    if (tid == 256) {
      for (int i = 0; i < nk; i++) {
        const int s = i % NST;
        if (i >= NST) mbwait(empty + s * 8, ((i / NST) - 1) & 1);
        const uint32_t dst = sb + s * STAGE, bar = full + s * 8;
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(bar), "r"(STAGE) : "memory");
        const uint8_t* src = wp + ((size_t)rb * KB + k0 + i) * 16384;
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], 16384, [%2];\n" :: "r"(dst), "l"(src), "r"(bar) : "memory");
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
                     :: "r"(dst + 16384), "l"(&xm), "r"((k0 + i) * 64), "r"(0), "r"(bar) : "memory");
      }
    }
  } else {
    for (int i = 0; i < nk; i++) {
      const int s = i % NST;
      mbwait(full + s * 8, (i / NST) & 1);
      const uint32_t wa = sb + s * STAGE + wg * 8192, xa = sb + s * STAGE + 16384;
      asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
      for (int j = 0; j < 4; j++) WGN(d, gdesc(wa) + 2 * j, gdesc(xa) + 2 * j, (i | j) != 0);
      asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
      asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
      if (i > 0 && (tid & 127) == 0) mbarrive(empty + ((i - 1) % NST) * 8);
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    if (nk == 0) {
#pragma unroll
      for (int i = 0; i < TN / 2; i++) d[i] = 0.f;
    }
  }
  __syncthreads();  // every load consumed: the ring becomes the partial buffer [TN tokens][LDR rows]
  constexpr int LDR = 132;
  float* red = (float*)smem;
  if (wg < 2) {
    const int w = (tid >> 5) & 3, g = lane >> 2, t = lane & 3, r0 = wg * 64 + w * 16 + g;
#pragma unroll
    for (int i = 0; i < TN / 8; i++) {
      const int tk = 8 * i + 2 * t;
      red[tk * LDR + r0] = d[4 * i]; red[(tk + 1) * LDR + r0] = d[4 * i + 1];
      red[tk * LDR + r0 + 8] = d[4 * i + 2]; red[(tk + 1) * LDR + r0 + 8] = d[4 * i + 3];
    }
  }
  cl.sync();
  // rank kr reduces tokens [ta, tb) over all 128 rows: float4 remote loads, 256 B contiguous output rows
  const int mrows = M < TN ? M : TN, ta = mrows * kr / KS, tb = mrows * (kr + 1) / KS;
  for (int e = tid; e < (NORED ? 0 : (tb - ta) * 32); e += blockDim.x) {
    const int tk = ta + (e >> 5), c4 = (e & 31) * 4;
    float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll 8
    for (int q = 0; q < KS; q++) {
      const float4 v = *(const float4*)(cl.map_shared_rank(red, q) + tk * LDR + c4);
      a.x += v.x; a.y += v.y; a.z += v.z; a.w += v.w;
    }
    __nv_bfloat162 o2[2] = {__floats2bfloat162_rn(a.x, a.y), __floats2bfloat162_rn(a.z, a.w)};
    *(uint2*)(out + (size_t)tk * N + rb * 128 + c4) = *(uint2*)o2;
  }
  cl.sync();
}

static PFN_cuTensorMapEncodeTiled_v12000 g_enc = nullptr;
void run(torch::Tensor x, torch::Tensor wp, torch::Tensor out, int64_t N, int64_t KS) {
  const int M = x.size(0), K = x.size(1), KB = K / 64, RB = N / 128;
  if (!g_enc) {
    cudaDriverEntryPointQueryResult q;
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", (void**)&g_enc, cudaEnableDefault, &q);
  }
  const int TN = M <= 128 ? 128 : 256;
  CUtensorMap xm;
  cuuint64_t dims[2] = {(cuuint64_t)K, (cuuint64_t)M}, strides[1] = {(cuuint64_t)K * 2};
  cuuint32_t box[2] = {64, (cuuint32_t)TN}, es[2] = {1, 1};
  g_enc(&xm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, x.data_ptr(), dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  const int smem = 1024 + NST * (16384 + TN * 128) + NST * 16;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(RB * KS); cfg.blockDim = dim3(288); cfg.dynamicSmemBytes = smem; cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = (unsigned)KS; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  cfg.attrs = at; cfg.numAttrs = 1;
  if (TN == 128) {
    cudaFuncSetAttribute(wgg<128>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (KS > 8) cudaFuncSetAttribute(wgg<128>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    cudaLaunchKernelEx(&cfg, wgg<128>, xm, (const uint8_t*)wp.data_ptr(), (__nv_bfloat16*)out.data_ptr(), M, (int)N, KB, (int)KS);
  } else {
    cudaFuncSetAttribute(wgg<256>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (KS > 8) cudaFuncSetAttribute(wgg<256>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    cudaLaunchKernelEx(&cfg, wgg<256>, xm, (const uint8_t*)wp.data_ptr(), (__nv_bfloat16*)out.data_ptr(), M, (int)N, KB, (int)KS);
  }
}
'''
SRC = CUDA.replace("WGFNS", wg_fn(128) + wg_fn(256)).replace(
    "WGN(d,", "if constexpr (TN == 128) wg128(d,").replace(
    ", (i | j) != 0);", ", (i | j) != 0); else wg256(d, gdesc(wa) + 2 * j, gdesc(xa) + 2 * j, (i | j) != 0);")


def pack(w):
    """bf16 [N, K] -> [N/128, K/64, 128 rows, 128 B] with the SW128 chunk swizzle (16 B chunk c of row r at c ^ (r & 7))"""
    N, K = w.shape
    t = w.view(N // 128, 128, K // 64, 8, 8).permute(0, 2, 1, 3, 4)  # rb, kb, row, chunk, 8
    r = torch.arange(128, device=w.device).view(1, 1, 128, 1, 1)
    c = torch.arange(8, device=w.device).view(1, 1, 1, 8, 1)
    idx = (c ^ (r & 7)).expand(t.shape[0], t.shape[1], 128, 8, 8)
    return torch.gather(t, 3, idx).contiguous()


def ks_for(N, K):
    rb = N // 128
    ks = 1
    while ks * 2 <= min(8, 132 // rb, K // 64):
        ks *= 2  # power of two: clusters pack into the GPCs
    return ks


if __name__ == "__main__":
    ext = load_inline("wgg_bench" + "".join(sys.argv[1:]).replace("-", "_").replace("=", ""), cpp_sources="void run(torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t);", cuda_sources=SRC,
                      functions=["run"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"] + sys.argv[1:], extra_ldflags=["-lcuda"], verbose=False)
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    def tm(fn, it=20):
        g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            fn(); torch.cuda.synchronize()
            with torch.cuda.graph(g): fn()
        ts = []
        for _ in range(it):
            flush.zero_(); a = torch.cuda.Event(True); b = torch.cuda.Event(True)
            a.record(); g.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1e3)
        ts.sort(); return ts[len(ts) // 2]
    e0 = tm(lambda: None)
    for name, (N, K) in [x for x in (("qkv", (2304, 8192)), ("o", (8192, 1024)), ("pooler", (4096, 8192)), ("unpooler", (8192, 4096))) if x[0] in os.environ.get("SH", "qkv,o,pooler,unpooler")]:
        W = (torch.randn(N, K, device="cuda") / K ** .5).bfloat16(); wp = pack(W)
        for M in [int(v) for v in os.environ.get("MS", "128,256,200,33").split(",")]:
            x = torch.randn(M, K, device="cuda").bfloat16(); out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
            KS = int(os.environ.get("KS", ks_for(N, K)))
            ext.run(x, wp, out, N, KS); torch.cuda.synchronize()
            ref = x.float() @ W.float().T
            err = ((out.float() - ref).norm() / ref.norm()).item()
            t_us = tm(lambda: ext.run(x, wp, out, N, KS)) - e0; t_cb = tm(lambda: x @ W.T) - e0
            print(f"{name:8s} M={M:3d} KS={KS} err {err:.1e}  wgg {t_us:6.1f}  cublas {t_cb:6.1f}", flush=True)
