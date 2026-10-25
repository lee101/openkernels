"""wgmma GEMM on plain BF16 weights (sm_90a): out[m, n] = x[m, k] @ W[n, k]^T, for the router (fp32 hi + lo pair, fp32 out)
and qkv / o at m 32..256. CTA = 64 * NWG tokens (NWG consumer warpgroups, x tile = wgmma A via TMA, rows >= m zero-filled)
x BN weight rows (wgmma B via TMA, SW128); producer warp streams x (L2 evict_last: x stays resident across row blocks) and
W (evict_first) into a QS-deep ring; split-K over a z-cluster (up to 16, non-portable above 8) reduced in DSMEM.
python bench/wgb.py   (env SH=router,qkv_gate,o MS=8,16,32,128,256 WGB=BN.QS.S.HL.NWG,...)"""
import os, sys, torch
from torch.utils.cpp_extension import load_inline


def wg_fn(n):
    regs = n // 2
    outs = ", ".join(f'"+f"(d[{i}])' for i in range(regs))
    lst = ",".join(f"%{i}" for i in range(regs))
    return f'''__device__ __forceinline__ void wb{n}(float* d, uint64_t da, uint64_t db, int sd) {{
  asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, %{regs + 2}, 0;\\n"
               "wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{lst}}}, %{regs}, %{regs + 1}, p, 1, 1, 0, 0;\\n}}\\n"
               : {outs} : "l"(da), "l"(db), "r"(sd));
}}
'''


WBCU = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <stdint.h>

struct BP { void* out; int m, N, K, KC, NH; };

__device__ __forceinline__ uint64_t bdesc(uint32_t s) {
  return (uint64_t)((s & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)64 << 32) | (1ull << 62);
}
__device__ __forceinline__ void bwait(uint32_t bar, uint32_t ph) {
  asm volatile("{\n.reg .pred P;\nWT: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n@!P bra WT;\n}\n" :: "r"(bar), "r"(ph) : "memory");
}
__device__ __forceinline__ void barr(uint32_t bar) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(bar) : "memory"); }
__device__ __forceinline__ void btma(uint32_t dst, const CUtensorMap* m, int c0, int c1, uint32_t bar, uint64_t pol) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3}], [%4], %5;\n"
               :: "r"(dst), "l"(m), "r"(c0), "r"(c1), "r"(bar), "l"(pol) : "memory");
}
WBFNS

template <int BN, int QS, int S, int HL, int NWG>
__global__ void __launch_bounds__(NWG * 128 + 32, 1) wgb(const __grid_constant__ CUtensorMap xm, const __grid_constant__ CUtensorMap wm, BP p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  constexpr int NA = HL ? 2 : 1, WT = BN * 128, XT = NWG * 8192, SB = XT + NA * WT, SBA = (SB + 1023) & ~1023, LD = BN + 4, TM = 64 * NWG;
  constexpr int NT = NWG * 128 + 32;
  using O = typename std::conditional<HL, float, __nv_bfloat16>::type;
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw), sb = (sraw + 1023) & ~1023u;
  float* red = (float*)(smem_raw + (sb - sraw));
  const uint32_t full = sb + QS * SBA, empty = full + QS * 8;
  const int tid = threadIdx.x, wg = tid >> 7, w = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int rb = blockIdx.x, tb = blockIdx.y, ks = blockIdx.z;
  const int c0 = p.KC * ks / S, nk = p.KC * (ks + 1) / S - c0;
  if (tid == 0) {
    for (int s = 0; s < QS; s++) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(full + s * 8));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(empty + s * 8), "n"(NWG));
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  float d[BN / 2];
  if (w == NWG * 4) {
    if (lane == 0) {
      uint64_t pw, px;
      asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pw));
      asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(px));
      for (int c = 0; c < nk; c++) {
        const int s = c % QS;
        if (c >= QS) bwait(empty + s * 8, ((c / QS) - 1) & 1);
        const uint32_t bar = full + s * 8, dst = sb + s * SBA;
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(bar), "n"(SB) : "memory");
        btma(dst, &xm, (c0 + c) * 64, tb * TM, bar, px);
        btma(dst + XT, &wm, (c0 + c) * 64, rb * BN, bar, pw);
        if (HL) btma(dst + XT + WT, &wm, (c0 + c) * 64, p.NH + rb * BN, bar, pw);
      }
    }
  } else {
    for (int c = 0; c < nk; c++) {
      const int s = c % QS;
      bwait(full + s * 8, (c / QS) & 1);
      asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
      const uint32_t xa = sb + s * SBA + wg * 8192, wa = sb + s * SBA + XT;
#pragma unroll
      for (int j = 0; j < 4; j++)
#pragma unroll
        for (int h = 0; h < NA; h++) WBN(d, bdesc(xa) + 2 * j, bdesc(wa + h * WT) + 2 * j, (c | j | h) != 0);
      asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
      asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
      if (c > 0 && (tid & 127) == 0) barr(empty + ((c - 1) % QS) * 8);
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    if (nk == 0) for (int i = 0; i < BN / 2; i++) d[i] = 0.f;
  }
  __syncthreads();
  const int tk0 = wg * 64 + (w & 3) * 16 + g;
  O* out = (O*)p.out;
  const int mt = min(TM, p.m - tb * TM);
  if (S == 1) {
    if (w < NWG * 4) {
#pragma unroll
      for (int i = 0; i < BN / 8; i++) {
        const int col = rb * BN + 8 * i + 2 * t;
#pragma unroll
        for (int hh = 0; hh < 2; hh++) {
          const int tk = tk0 + 8 * hh;
          if (tk >= mt) continue;
          O* o = out + (size_t)(tb * TM + tk) * p.N + col;
          if (HL) *(float2*)o = make_float2(d[4 * i + 2 * hh], d[4 * i + 2 * hh + 1]);
          else *(__nv_bfloat162*)o = __floats2bfloat162_rn(d[4 * i + 2 * hh], d[4 * i + 2 * hh + 1]);
        }
      }
    }
    return;
  }
  if (w < NWG * 4) {
#pragma unroll
    for (int i = 0; i < BN / 8; i++) {
      *(float2*)(red + tk0 * LD + 8 * i + 2 * t) = make_float2(d[4 * i], d[4 * i + 1]);
      *(float2*)(red + (tk0 + 8) * LD + 8 * i + 2 * t) = make_float2(d[4 * i + 2], d[4 * i + 3]);
    }
  }
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
  const uint32_t rbase = (uint32_t)__cvta_generic_to_shared(red);
  constexpr int C4 = BN / 4;
  const int tpr = (mt + S - 1) / S, ta = ks * tpr, tz = min(mt, ta + tpr);
  for (int e2 = tid; e2 < (tz - ta) * C4; e2 += NT) {
    const int tk = ta + e2 / C4, c4 = (e2 % C4) * 4;
    float4 v[S];
#pragma unroll
    for (int q = 0; q < S; q++) {
      uint32_t ra; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(ra) : "r"(rbase + (tk * LD + c4) * 4), "r"(q));
      asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];\n" : "=f"(v[q].x), "=f"(v[q].y), "=f"(v[q].z), "=f"(v[q].w) : "r"(ra) : "memory");
    }
#pragma unroll
    for (int q = 1; q < S; q++) { v[0].x += v[q].x; v[0].y += v[q].y; v[0].z += v[q].z; v[0].w += v[q].w; }
    O* o = out + (size_t)(tb * TM + tk) * p.N + rb * BN + c4;
    if (HL) *(float4*)o = v[0];
    else { __nv_bfloat162 o2[2] = {__floats2bfloat162_rn(v[0].x, v[0].y), __floats2bfloat162_rn(v[0].z, v[0].w)}; *(uint2*)o = *(uint2*)o2; }
  }
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
#endif
}

static PFN_cuTensorMapEncodeTiled_v12000 g_benc = nullptr;
static bool bmap(CUtensorMap* m, const void* p, int K, int rows, int box) {
  cuuint64_t dims[2] = {(cuuint64_t)K, (cuuint64_t)rows}, strides[1] = {(cuuint64_t)K * 2};
  cuuint32_t bx[2] = {64, (cuuint32_t)box}, es[2] = {1, 1};
  return g_benc(m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(p), dims, strides, bx, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}
template <int BN, int QS, int S, int HL, int NWG>
static int64_t wbl(BP p, const void* x, const void* W, int wrows, cudaStream_t st) {
  constexpr int NA = HL ? 2 : 1, TM = 64 * NWG;
  if (p.N % BN || S > 16) return -1;
  if (!g_benc) {
    cudaDriverEntryPointQueryResult q;
    if (cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", (void**)&g_benc, cudaEnableDefault, &q) != cudaSuccess || !g_benc) return -3;
  }
  CUtensorMap xm, wm;
  if (!bmap(&xm, x, p.K, p.m, TM) || !bmap(&wm, W, p.K, wrows, BN)) return -4;
  const size_t SBA = ((size_t)NWG * 8192 + NA * BN * 128 + 1023) / 1024 * 1024, ring = QS * SBA + QS * 16, part = (size_t)TM * (BN + 4) * 4;
  const size_t sm = 1024 + (ring > part ? ring : part);
  auto k = wgb<BN, QS, S, HL, NWG>;
  static bool once = false;
  if (!once) {
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    if (S > 8) cudaFuncSetAttribute(k, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    once = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(p.N / BN, (p.m + TM - 1) / TM, S); cfg.blockDim = dim3(NWG * 128 + 32); cfg.dynamicSmemBytes = sm; cfg.stream = st;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = 1; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = S;
  cfg.attrs = at; cfg.numAttrs = S > 1 ? 1 : 0;
  const cudaError_t r = cudaLaunchKernelEx(&cfg, k, xm, wm, p);
  if (r != cudaSuccess) cudaGetLastError();
  return (int64_t)r;
}

// W: [N, K] (or [2N, K] hi | lo for HL); out: [m, N]
int64_t wbrun(int64_t inst, torch::Tensor W, torch::Tensor x, torch::Tensor out) {
  BP p;
  p.out = out.data_ptr(); p.m = x.size(0); p.K = x.size(1); p.N = out.size(1); p.KC = p.K / 64; p.NH = p.N;
  auto st = at::cuda::getCurrentCUDAStream();
  const void* xp = x.data_ptr(); const void* wp = W.data_ptr(); const int wr = W.size(0);
  switch (inst) {
@@CASES@@
  }
  return -2;
}
'''

WBI = [tuple(map(int, c.split("."))) for c in os.environ.get("WGB", "64.6.16.1.1,64.6.8.1.1,32.8.16.1.1,64.6.16.1.2,64.6.8.1.2,128.5.8.0.1,128.5.4.0.1,64.6.8.0.1,128.4.8.0.2,128.4.4.0.2").split(",")]


def wb_src(insts):
    ns = sorted({i[0] for i in insts})
    src = WBCU.replace("WBFNS", "".join(wg_fn(n) for n in ns))
    call = "".join(f"if constexpr (BN == {n}) wb{n}(d, bdesc(xa) + 2 * j, bdesc(wa + h * WT) + 2 * j, (c | j | h) != 0); " for n in ns)
    src = src.replace("WBN(d, bdesc(xa) + 2 * j, bdesc(wa + h * WT) + 2 * j, (c | j | h) != 0);", "{ " + call + "}")
    cases = "\n".join("    case %d: return wbl<%s>(p, xp, wp, wr, st);" % (k, ",".join(map(str, c))) for k, c in enumerate(insts))
    return src.replace("@@CASES@@", cases)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../muse_bf16_proj")
    ext = load_inline("wgb_bench_%x" % (hash(tuple(WBI) + tuple(sys.argv[1:])) & 0xffffffff), cpp_sources="int64_t wbrun(int64_t, torch::Tensor, torch::Tensor, torch::Tensor);",
                      cuda_sources=wb_src(WBI), functions=["wbrun"], extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_90a,code=sm_90a"] + sys.argv[1:], verbose=False)
    flush = torch.ones(64 << 20, dtype=torch.float32, device="cuda"); fo = torch.empty((), device="cuda")
    def tm(fn, it=30):
        g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn(); torch.cuda.synchronize()
            with torch.cuda.graph(g): fn()
        ts = []
        for _ in range(it):
            torch.sum(flush, 0, out=fo); a = torch.cuda.Event(True); b = torch.cuda.Event(True)
            a.record(); g.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1e3)
        ts.sort(); return ts[len(ts) // 2]
    z = torch.zeros(1, device="cuda")
    off = tm(lambda: z.zero_())
    import reference as ref
    ws = ref.generate_weights(ref.WEIGHTS_SEED)
    for name in os.environ.get("SH", "router,qkv_gate,o").split(","):
        w = ws[name]
        hl = name == "router"
        Wd = torch.cat((w.to(torch.bfloat16), (w - w.to(torch.bfloat16).float()).to(torch.bfloat16))) if hl else w
        exact = Wd[:256].float() + Wd[256:].float() if hl else w
        N, K = w.shape
        for M in [int(v) for v in os.environ.get("MS", "8,16,32,128,256").split(",")]:
            x = torch.randn(M, K, device="cuda").bfloat16()
            x[M - 1].zero_(); x[M - 1, 777] = 16.0
            refo = x.double() @ w.double().T
            if hl:
                hlt = Wd.t().contiguous()
                tcb = tm(lambda: (lambda a: a[:, :N].float() + a[:, N:].float())(x @ hlt)) - off  # bf16-out stand-in for v12's N=512 GEMM + add
            else:
                tcb = tm(lambda: x @ w.T) - off
            for k, inst in enumerate(WBI):
                BN, QS, S, HL, NWG = inst
                if HL != hl or N % BN:
                    continue
                out = torch.empty(M, N, device="cuda", dtype=torch.float32 if hl else torch.bfloat16)
                rc = ext.wbrun(k, Wd, x, out)
                torch.cuda.synchronize()
                if rc:
                    print(name, M, inst, "rc", rc); continue
                err = ((out.double() - refo).norm() / refo.norm()).item()
                pr = (out[M - 1].double() - exact[:, 777].double() * 16).norm().item() / (exact[:, 777].double() * 16).norm().item()
                t_us = tm(lambda: ext.wbrun(k, Wd, x, out)) - off
                print(f"{name:8s} M={M:3d} {inst} err {err:.1e} probe {pr:.1e}  wgb {t_us:6.2f}  ref {tcb:6.2f}", flush=True)
