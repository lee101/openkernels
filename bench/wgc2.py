"""wgmma GEMM on 12-bit-packed BF16 weights for 64 < m <= 256 (sm_90a): out[m, n] = x[m, k] @ W[n, k]^T.
CTA = 128 tokens (2 warpgroups x m64, x tiles via TMA, SW128) x BN weight rows (wgmma N); every warp decodes its c12 units
(register-direct loads, natural-k A-fragment pack) into a 3-deep SW128 smem ring that both warpgroups read as wgmma B;
split-K over a z-cluster reduced in DSMEM. HL: router BF16 hi + lo pair (two B tiles, one fp32 accumulator).
python bench/wgc.py [check|time]"""
import os, sys, time, torch
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


WGCU = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <stdint.h>
#ifndef WDIAG
#define WDIAG 0
#endif

struct WP { const uint4* W; const uint4* W2; void* out; const uint32_t* esc; int m, N, K, KC; uint32_t T0, U0; };

__device__ __forceinline__ uint64_t wdesc(uint32_t s) {  // K-major SW128, 8-row atoms 1 KB apart
  return (uint64_t)((s & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)64 << 32) | (1ull << 62);
}
__device__ __forceinline__ uint4 wnc(const void* p) {
  uint4 v; asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}
__device__ __forceinline__ uint32_t wprmt(uint32_t a, uint32_t b, uint32_t s) { uint32_t d; asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(s)); return d; }
__device__ __forceinline__ void wdec(uint32_t L0, uint32_t L1, uint32_t C, uint32_t Tb, uint32_t* a) {
  const uint32_t hl = (C & 0x87878787u) + Tb, hh = (__funnelshift_r(C, C, 4) & 0x87878787u) + Tb;
  a[0] = wprmt(L0, hl, 0x5140); a[1] = wprmt(L0, hl, 0x7362); a[2] = wprmt(L1, hh, 0x5140); a[3] = wprmt(L1, hh, 0x7362);
}
__device__ __forceinline__ void sts32(uint32_t a, uint32_t v) { asm volatile("st.shared.b32 [%0], %1;\n" :: "r"(a), "r"(v) : "memory"); }
WGFNS

template <int BN, int QS, int S, int HL>
__global__ void __launch_bounds__(256, 1) wgc(const __grid_constant__ CUtensorMap xm, WP p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  constexpr int NA = HL ? 2 : 1, TPB = BN / 16, NU = TPB * NA, U = (NU + 7) / 8, WT = BN * 128, LD = BN + 4;
  constexpr int WR = NU * 1536, XB = 16384, SB = XB + WR;
  using O = typename std::conditional<HL, float, __nv_bfloat16>::type;
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw), sb = (sraw + 1023) & ~1023u;
  float* red = (float*)(smem_raw + (sb - sraw));
  const uint32_t ws0 = sb, st0 = sb + 3 * NA * WT, bars = st0 + QS * ((SB + 1023) & ~1023);
  constexpr int SBA = (SB + 1023) & ~1023;
  const int tid = threadIdx.x, wg = tid >> 7, w = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int rb = blockIdx.x, tb = blockIdx.y, ks = blockIdx.z;
  const int c0 = p.KC * ks / S, nk = p.KC * (ks + 1) / S - c0;
  if (tid == 0) {
    for (int s = 0; s < QS; s++) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(bars + s * 8));
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  const uint8_t* wsrc = (const uint8_t*)p.W + ((size_t)rb * p.KC + c0) * WR;
  auto issue = [&](int c) {
    if (c < nk) {
      const uint32_t bar = bars + (c % QS) * 8, dst = st0 + (c % QS) * SBA;
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(bar), "r"(SB) : "memory");
      asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
                   :: "r"(dst), "l"(&xm), "r"((c0 + c) * 64), "r"(tb * 128), "r"(bar) : "memory");
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                   :: "r"(dst + XB), "l"(wsrc + (size_t)c * WR), "n"(WR), "r"(bar) : "memory");
    }
  };
  if (tid == 0) for (int c = 0; c < QS - 2; c++) issue(c);
  uint32_t tbl[U], srow[U];
#pragma unroll
  for (int i = 0; i < U; i++) {
    const int uu = w + 8 * i, h = uu / TPB, tl = uu % TPB;
    tbl[i] = h ? p.U0 : p.T0;
    srow[i] = h * WT + (tl * 16 + g) * 128;
  }
  const uint32_t e = p.esc[((size_t)(rb * S + ks) * 8 + w) * 32 + lane];
  uint32_t nx = __shfl_sync(~0u, e, 0);
  int ep = 0;
  float d[BN / 2];
#pragma unroll 1
  for (int c = 0; c < nk; c++) {
    const uint32_t stg = st0 + (c % QS) * SBA;
    {
      const uint32_t bar = bars + (c % QS) * 8, ph = (c / QS) & 1;
      asm volatile("{\n.reg .pred P;\nWT: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n@!P bra WT;\n}\n" :: "r"(bar), "r"(ph) : "memory");
    }
#pragma unroll
    for (int i = 0; i < U; i++) {
      const int uu = w + 8 * i;
      if (NU % 8 == 0 || uu < NU) {
#if !(WDIAG & 1)
        uint32_t A[16];
        const uint32_t ra = stg + XB + uu * 1536 + lane * 16;
        uint4 q0, q1, q2;
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(q0.x), "=r"(q0.y), "=r"(q0.z), "=r"(q0.w) : "r"(ra));
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(q1.x), "=r"(q1.y), "=r"(q1.z), "=r"(q1.w) : "r"(ra + 512));
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(q2.x), "=r"(q2.y), "=r"(q2.z), "=r"(q2.w) : "r"(ra + 1024));
        wdec(q0.x, q0.y, q2.x, tbl[i], A); wdec(q0.z, q0.w, q2.y, tbl[i], A + 4);
        wdec(q1.x, q1.y, q2.z, tbl[i], A + 8); wdec(q1.z, q1.w, q2.w, tbl[i], A + 12);
        while ((nx >> 20) == (uint32_t)(c * U + i)) {
          if (lane == (int)((nx >> 15) & 31)) {
            const uint32_t rr = (nx >> 11) & 15, xr = (nx & 0x7fu) << (8 + ((nx >> 10) & 1) * 16);
#pragma unroll
            for (int k = 0; k < 16; k++) if (k == (int)rr) A[k] ^= xr;
          }
          nx = ++ep < 32 ? __shfl_sync(~0u, e, ep) : ~0u;
        }
        const uint32_t base = ws0 + (c % 3) * NA * WT + srow[i];
#pragma unroll
        for (int j = 0; j < 4; j++) {
          const uint32_t c0a = (((2 * j) ^ g) << 4) + 4 * t, c1a = (((2 * j + 1) ^ g) << 4) + 4 * t;
          sts32(base + c0a, A[4 * j]); sts32(base + 1024 + c0a, A[4 * j + 1]);
          sts32(base + c1a, A[4 * j + 2]); sts32(base + 1024 + c1a, A[4 * j + 3]);
        }
#endif
      }
    }
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    __syncthreads();
    if (tid == 0) issue(c + QS - 2);
#if WDIAG & 2
    if (c == 0) for (int i = 0; i < BN / 2; i++) d[i] = 0.f;
    continue;
#endif
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
    const uint32_t xa = stg + wg * 8192, wa = ws0 + (c % 3) * NA * WT;
#pragma unroll
    for (int j = 0; j < 4; j++)
#pragma unroll
      for (int h = 0; h < NA; h++) WGN(d, wdesc(xa) + 2 * j, wdesc(wa + h * WT) + 2 * j, (c | j | h) != 0);
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
  const int tk0 = wg * 64 + (w & 3) * 16 + g;
  O* out = (O*)p.out;
  if (S == 1) {
#pragma unroll
    for (int i = 0; i < BN / 8; i++) {
      const int col = rb * BN + 8 * i + 2 * t;
      O* o = out + (size_t)(tb * 128 + tk0) * p.N + col;
      if (HL) { *(float2*)o = make_float2(d[4 * i], d[4 * i + 1]); *(float2*)(o + 8 * (size_t)p.N) = make_float2(d[4 * i + 2], d[4 * i + 3]); }
      else { *(__nv_bfloat162*)o = __floats2bfloat162_rn(d[4 * i], d[4 * i + 1]); *(__nv_bfloat162*)(o + 8 * (size_t)p.N) = __floats2bfloat162_rn(d[4 * i + 2], d[4 * i + 3]); }
    }
    return;
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < BN / 8; i++) {
    *(float2*)(red + tk0 * LD + 8 * i + 2 * t) = make_float2(d[4 * i], d[4 * i + 1]);
    *(float2*)(red + (tk0 + 8) * LD + 8 * i + 2 * t) = make_float2(d[4 * i + 2], d[4 * i + 3]);
  }
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
  const uint32_t rbase = (uint32_t)__cvta_generic_to_shared(red);
  constexpr int TPR = 128 / S, C4 = BN / 4;
  for (int e2 = tid; e2 < TPR * C4; e2 += 256) {
    const int tk = ks * TPR + e2 / C4, c4 = (e2 % C4) * 4;
    float4 v[S];
#pragma unroll
    for (int q = 0; q < S; q++) {
      uint32_t ra; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(ra) : "r"(rbase + (tk * LD + c4) * 4), "r"(q));
      asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];\n" : "=f"(v[q].x), "=f"(v[q].y), "=f"(v[q].z), "=f"(v[q].w) : "r"(ra) : "memory");
    }
#pragma unroll
    for (int q = 1; q < S; q++) { v[0].x += v[q].x; v[0].y += v[q].y; v[0].z += v[q].z; v[0].w += v[q].w; }
    O* o = out + (size_t)(tb * 128 + tk) * p.N + rb * BN + c4;
    if (HL) *(float4*)o = v[0];
    else { __nv_bfloat162 o2[2] = {__floats2bfloat162_rn(v[0].x, v[0].y), __floats2bfloat162_rn(v[0].z, v[0].w)}; *(uint2*)o = *(uint2*)o2; }
  }
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
#endif
}

static PFN_cuTensorMapEncodeTiled_v12000 g_enc = nullptr;
template <int BN, int QS, int S, int HL>
static int64_t wgl(const WP& p, const void* x, cudaStream_t st) {
  constexpr int NA = HL ? 2 : 1;
  if (p.m % 128 || p.N % BN) return -1;
  if (!g_enc) {
    cudaDriverEntryPointQueryResult q;
    if (cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", (void**)&g_enc, cudaEnableDefault, &q) != cudaSuccess || !g_enc) return -3;
  }
  CUtensorMap xm;
  cuuint64_t dims[2] = {(cuuint64_t)p.K, (cuuint64_t)p.m}, strides[1] = {(cuuint64_t)p.K * 2};
  cuuint32_t box[2] = {64, 128}, es[2] = {1, 1};
  if (g_enc(&xm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(x), dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS) return -4;
  const size_t SBA = ((size_t)16384 + BN / 16 * NA * 1536 + 1023) / 1024 * 1024;
  const size_t ring = (size_t)3 * NA * BN * 128 + QS * SBA + QS * 8, part = (size_t)128 * (BN + 4) * 4;
  const size_t sm = 1024 + (ring > part ? ring : part);
  auto k = wgc<BN, QS, S, HL>;
  static bool once = false;
  if (!once) { cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm); once = true; }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(p.N / BN, p.m / 128, S); cfg.blockDim = dim3(256); cfg.dynamicSmemBytes = sm; cfg.stream = st;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = 1; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = S;
  cfg.attrs = at; cfg.numAttrs = S > 1 ? 1 : 0;
  const cudaError_t r = cudaLaunchKernelEx(&cfg, k, xm, p);
  if (r != cudaSuccess) cudaGetLastError();
  return (int64_t)r;
}

int64_t wgrun(int64_t inst, torch::Tensor W, torch::Tensor W2, torch::Tensor x, torch::Tensor out, torch::Tensor esc, int64_t T, int64_t U) {
  WP p;
  p.W = (const uint4*)W.data_ptr(); p.W2 = (const uint4*)W2.data_ptr(); p.out = out.data_ptr(); p.esc = (const uint32_t*)esc.data_ptr();
  p.m = x.size(0); p.K = x.size(1); p.N = out.size(1); p.KC = p.K / 64; p.T0 = (uint32_t)T; p.U0 = (uint32_t)U;
  auto st = at::cuda::getCurrentCUDAStream();
  switch (inst) {
@@CASES@@
  }
  return -2;
}
'''

WGI = [tuple(map(int, c.split("."))) for c in os.environ.get("WGI", "128.5.4.0,128.5.2.0,128.5.1.0,128.5.8.0,64.5.4.0,64.5.8.1,32.5.8.1").split(",")]


def wg_src(insts):
    ns = sorted({i[0] for i in insts})
    src = WGCU.replace("WGFNS", "".join(wg_fn(n) for n in ns))
    src = src.replace("WGN(d,", " ".join(f"if constexpr (BN == {n}) wg{n}(d," for n in ns[:1]) if False else "WGCALL(d,")
    call = "".join(f"if constexpr (BN == {n}) wg{n}(DARGS); " for n in ns)
    src = src.replace("WGCALL(d, wdesc(xa) + 2 * j, wdesc(wa + h * WT) + 2 * j, (c | j | h) != 0);",
                      "{ " + call.replace("DARGS", "d, wdesc(xa) + 2 * j, wdesc(wa + h * WT) + 2 * j, (c | j | h) != 0") + "}")
    cases = "\n".join("    case %d: return wgl<%s>(p, x.data_ptr(), st);" % (k, ",".join(map(str, c))) for k, c in enumerate(insts))
    return src.replace("@@CASES@@", cases)


def pack_nat(w):
    """bf16 [N, K] -> natural-k A fragments [N/16, K/64, 4 steps, 32 lanes, 8]: lane (g, t) step j holds rows g, g + 8 at
    k 16 j + 2 t (+1) and 16 j + 8 + 2 t (+1), in mma register order"""
    N, K = w.shape
    W = w.view(N // 16, 2, 8, K // 64, 4, 2, 4, 2)  # T, half(row+8), g, kc, j, h(k+8), t, e
    fr = torch.stack((W[:, 0, :, :, :, 0, :, 0], W[:, 0, :, :, :, 0, :, 1], W[:, 1, :, :, :, 0, :, 0], W[:, 1, :, :, :, 0, :, 1],
                      W[:, 0, :, :, :, 1, :, 0], W[:, 0, :, :, :, 1, :, 1], W[:, 1, :, :, :, 1, :, 0], W[:, 1, :, :, :, 1, :, 1]), -1)
    return fr.permute(0, 2, 3, 1, 4, 5).contiguous()  # T, kc, j, g, t, 8


def wg_pack(datas, BN):
    """c12 units [T, KC, 3, 32, 16] per matrix -> [RB, KC, NA, BN/16, 1536]: one contiguous bulk copy per (row block, chunk)"""
    T, KC = datas[0].shape[:2]
    tpb = BN // 16
    return torch.stack([d.view(T // tpb, tpb, KC, 1536).permute(0, 2, 1, 3) for d in datas], 2).contiguous()


def wg_esc(esc, T, BN, S, KC, NA):
    """per-warp escape lists for one wgc geometry; esc [E, 7] (tile, chunk, step, lane, value, xor, matrix)"""
    tpb = BN // 16
    U = (tpb * NA + 7) // 8
    rbn = T // tpb
    out = torch.full((rbn * S * 8 * 32,), -1, dtype=torch.int64, device=esc.device)
    if len(esc):
        t, c, j, l, v, x, h = esc.t()
        c0 = torch.tensor([KC * k // S for k in range(S)], device=esc.device)
        kk = torch.searchsorted(c0, c, right=True) - 1
        uu = h * tpb + t % tpb
        gw = ((t // tpb) * S + kk) * 8 + uu % 8
        seq = (c - c0[kk]) * U + uu // 8
        o = torch.argsort(gw * 4096 + seq)
        gw, ent = gw[o], ((seq << 20) | (l << 15) | ((j * 4 + v // 2) << 11) | ((v % 2) << 10) | x)[o]
        rank = torch.arange(len(gw), device=gw.device) - torch.searchsorted(gw, gw)
        if int(rank.max()) >= 32:
            return None
        out[gw * 32 + rank] = ent
    return torch.where(out >= 2 ** 31, out - 2 ** 32, out).int()


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../muse_bf16_proj")
    import submission_v16 as sv
    ext = load_inline("wgc_bench_%x" % (hash(tuple(WGI) + tuple(sys.argv[2:])) & 0xffffffff), cpp_sources="int64_t wgrun(int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t);",
                      cuda_sources=wg_src(WGI), functions=["wgrun"], extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_90a,code=sm_90a"] + sys.argv[2:], verbose=False)
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
    shapes = [s for s in os.environ.get("SH", "qkv_gate,o,pooler,unpooler,router").split(",")]
    for name in shapes:
        w = ws[name]
        hl = name == "router"
        mats = [w.to(torch.bfloat16), (w - w.to(torch.bfloat16).float()).to(torch.bfloat16)] if hl else [w]
        cs = [sv._c12(pack_nat(mm), 1) for mm in mats]
        if any(c is None for c in cs):
            print(name, "c12 pack failed"); continue
        esc = torch.cat([torch.cat((c[2], torch.full_like(c[2][:, :1], h)), 1) for h, c in enumerate(cs)])
        N, K = w.shape
        exact = (mats[0].float() + mats[1].float()) if hl else w
        for M in [int(v) for v in os.environ.get("MS", "128,256").split(",")]:
            x = torch.randn(M, K, device="cuda").bfloat16()
            x[3].zero_(); x[3, 777] = 16.0
            dt = torch.float32 if hl else torch.bfloat16
            refo = x.double() @ w.double().T
            tcb = tm(lambda: (x.float() @ w.T) if hl else x @ w.T) - off
            for k, inst in enumerate(WGI):
                BN, QS, S, HL = inst
                if HL != hl or N % BN:
                    continue
                wpk = wg_pack([c[0] for c in cs], BN)
                e = wg_esc(esc, N // 16, BN, S, K // 64, 2 if hl else 1)
                if e is None:
                    print(name, inst, "escape overflow"); continue
                out = torch.empty(M, N, device="cuda", dtype=dt)
                rc = ext.wgrun(k, wpk, wpk, x, out, e, cs[0][1], cs[-1][1])
                torch.cuda.synchronize()
                if rc:
                    print(name, M, inst, "launch rc", rc); continue
                err = ((out.double() - refo).norm() / refo.norm()).item()
                pr = (out[3].double() - exact[:, 777].double() * 16).norm().item() / (exact[:, 777].double() * 16).norm().item()
                t_us = tm(lambda: ext.wgrun(k, wpk, wpk, x, out, e, cs[0][1], cs[-1][1])) - off
                print(f"{name:8s} M={M:3d} {inst} err {err:.1e} probe {pr:.1e}  wgc {t_us:6.2f}  cublas {tcb:6.2f}", flush=True)
