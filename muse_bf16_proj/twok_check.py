# okc: problem=muse_bf16_proj
"""Stream-K skinny GEMM: out[m, n] = x @ W[:n].T for m <= 32 (and the router at any m).
W is repacked into mma.sync A-fragment order (2 KB contiguous per warp per 64-wide K chunk); every warp streams an
equal contiguous range of (tile, chunk) units; tiles split across warps meet through fp32 atomics and the last
arriver writes the output. The FP32 router runs as a BF16 hi + lo pair with FP32 accumulation."""
from typing import TypedDict
import torch
from torch.utils.cpp_extension import load_inline


class Weights(TypedDict):
    qkv_gate: torch.Tensor
    o: torch.Tensor
    pooler: torch.Tensor
    unpooler: torch.Tensor
    router: torch.Tensor


class Config(TypedDict):
    m: int
    n: int
    k: int
    layer: str


class Inputs(TypedDict):
    x: torch.Tensor
    config: Config


CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>
#ifndef WPC
#define WPC 16
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define TMA 1
#else
#define TMA 0
#endif

__device__ __forceinline__ void mma(float* c, const uint4& a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint4 lds(uint32_t a) { uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a)); return v; }
__device__ __forceinline__ void mbinit(uint32_t b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(b), "r"(n)); }
__device__ __forceinline__ void mbwait(uint32_t b, uint32_t ph) {
#if TMA
  asm volatile("{\n.reg .pred p;\nW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W;\n}\n" :: "r"(b), "r"(ph) : "memory");
#else
  asm volatile("{\n.reg .pred p;\nW: mbarrier.test_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W;\n}\n" :: "r"(b), "r"(ph) : "memory");
#endif
}
__device__ __forceinline__ void cp16(uint32_t s, const void* g) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(g)); }

struct Q {
  const uint8_t* A; const uint8_t* A2; const __nv_bfloat16* x; void* out; float* ws; int* cnt;
  int M, N, K, MT, KC, TB, f32;
};

__device__ __forceinline__ void put(const Q& q, int tok, int row, float v) {
  if (q.f32) ((float*)q.out)[(size_t)tok * q.N + row] = v; else ((__nv_bfloat16*)q.out)[(size_t)tok * q.N + row] = __float2bfloat16(v);
}

// Each warp streams an equal contiguous range of (tile, 64-wide K chunk) units through its own RS-deep TMA ring
// (2 KB A fragments per unit, +2 KB for the router's lo half); tiles split across warps meet in fp32 atomics.
template <int NT, bool HL, int RS>
__global__ void __launch_bounds__(WPC * 32, 1) skinny(Q q) {
  constexpr int SB = HL ? 4096 : 2048;
  constexpr int CH = NT == 1 ? 4 : (NT == 2 ? 2 : 1);
  extern __shared__ __align__(128) uint8_t smem[];
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int W = gridDim.x * WPC, gw = blockIdx.x * WPC + wid;
  const uint32_t ring = (uint32_t)__cvta_generic_to_shared(smem) + wid * RS * SB;
  const uint32_t bars = (uint32_t)__cvta_generic_to_shared(smem) + WPC * RS * SB + wid * RS * 8;
#if TMA
  if (lane < RS) mbinit(bars + lane * 8, 1);
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#else
  if (lane < RS) mbinit(bars + lane * 8, 32);
#endif
  __syncwarp();
  const long long U = (long long)q.TB * q.MT * q.KC;
  const long long L0 = U * gw / W, Le = U * (gw + 1) / W;
  if (L0 >= Le) return;
  long long pu = L0;
  auto issue = [&](int slot) {
    if (pu < Le) {
      const long long tile = pu / q.KC; const int c = (int)(pu - tile * q.KC), mt = (int)(tile % q.MT);
      const size_t off = ((size_t)mt * q.KC + c) * 2048;
      const uint32_t d = ring + slot * SB, bar = bars + slot * 8;
#if TMA
      if (lane == 0) {
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(bar), "n"(SB) : "memory");
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], 2048, [%2];\n" :: "r"(d), "l"(q.A + off), "r"(bar) : "memory");
        if (HL) asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], 2048, [%2];\n" :: "r"(d + 2048), "l"(q.A2 + off), "r"(bar) : "memory");
      }
#else
#pragma unroll
      for (int k = 0; k < 4; k++) cp16(d + k * 512 + lane * 16, q.A + off + k * 512 + lane * 16);
      if (HL) {
#pragma unroll
        for (int k = 0; k < 4; k++) cp16(d + 2048 + k * 512 + lane * 16, q.A2 + off + k * 512 + lane * 16);
      }
      asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" :: "r"(bar) : "memory");
#endif
    } else {
#if !TMA
      asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(bars + slot * 8) : "memory");
#endif
    }
    pu++;
  };
#pragma unroll
  for (int i = 0; i < RS - 1; i++) issue(i);
  uint4 b0[NT][2], b1[NT][2];
  auto ldb = [&](long long u, uint4 (&b)[NT][2]) {
    if (u >= Le) return;
    const long long tile = u / q.KC; const int c = (int)(u - tile * q.KC), tb = (int)(tile / q.MT);
#pragma unroll
    for (int n = 0; n < NT; n++) {
      const int tok = min(tb * 32 + n * 8 + g, q.M - 1);
      const uint4* xr = (const uint4*)(q.x + (size_t)tok * q.K) + t * 2 + c * 8;
      b[n][0] = __ldg(xr); b[n][1] = __ldg(xr + 1);
    }
  };
  ldb(L0, b0);
  float acc[NT][CH][4];
  int qn = 0;
  long long u = L0;
  while (u < Le) {
    const long long tile = u / q.KC; const int c0 = (int)(u - tile * q.KC);
    const int c1 = (int)min((long long)q.KC, c0 + (Le - u));
#pragma unroll
    for (int n = 0; n < NT; n++)
#pragma unroll
      for (int h = 0; h < CH; h++) acc[n][h][0] = acc[n][h][1] = acc[n][h][2] = acc[n][h][3] = 0.f;
    for (int c = c0; c < c1; c++, u++, qn++) {
      const int slot = qn % RS;
      mbwait(bars + slot * 8, (qn / RS) & 1);
      __syncwarp();
      issue((qn + RS - 1) % RS);
      ldb(u + 1, b1);
      const uint32_t sa = ring + slot * SB + lane * 16;
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const uint4 a = lds(sa + j * 512);
        uint4 h; if (HL) h = lds(sa + 2048 + j * 512);
#pragma unroll
        for (int n = 0; n < NT; n++) {
          const uint32_t* bb = (const uint32_t*)&b0[n][j >> 1];
          mma(acc[n][j % CH], a, bb[(j & 1) * 2], bb[(j & 1) * 2 + 1]);
          if (HL) mma(acc[n][(j + 1) % CH], h, bb[(j & 1) * 2], bb[(j & 1) * 2 + 1]);
        }
      }
#pragma unroll
      for (int n = 0; n < NT; n++) { b0[n][0] = b1[n][0]; b0[n][1] = b1[n][1]; }
    }
    float f[NT][4];
#pragma unroll
    for (int n = 0; n < NT; n++)
#pragma unroll
      for (int i = 0; i < 4; i++) { float v = 0.f;
#pragma unroll
        for (int h = 0; h < CH; h++) v += acc[n][h][i];
        f[n][i] = v; }
    const int tb = (int)(tile / q.MT), mt = (int)(tile - (long long)tb * q.MT), row0 = mt * 16 + g, mrem = min(32, q.M - tb * 32);
#ifdef TWOK
    if (false) {
#else
    if (c0 == 0 && c1 == q.KC) {
#endif
#pragma unroll
      for (int n = 0; n < NT; n++) {
        const int tk = n * 8 + 2 * t;
        if (tk < mrem) { put(q, tb * 32 + tk, row0, f[n][0]); put(q, tb * 32 + tk, row0 + 8, f[n][2]); }
        if (tk + 1 < mrem) { put(q, tb * 32 + tk + 1, row0, f[n][1]); put(q, tb * 32 + tk + 1, row0 + 8, f[n][3]); }
      }
      continue;
    }
    float* w = q.ws + (size_t)tb * 32 * q.N;
#pragma unroll
    for (int n = 0; n < NT; n++) {
      const int tk = n * 8 + 2 * t;
      if (tk < mrem) { atomicAdd(w + (size_t)tk * q.N + row0, f[n][0]); atomicAdd(w + (size_t)tk * q.N + row0 + 8, f[n][2]); }
      if (tk + 1 < mrem) { atomicAdd(w + (size_t)(tk + 1) * q.N + row0, f[n][1]); atomicAdd(w + (size_t)(tk + 1) * q.N + row0 + 8, f[n][3]); }
    }
#ifdef TWOK
    continue;  // finalize kernel converts ws
#endif
    __threadfence(); __syncwarp();
    int old = 0;
    if (lane == 0) old = atomicAdd(q.cnt + tile, c1 - c0);
    old = __shfl_sync(0xffffffffu, old, 0);
    if (old + c1 - c0 != q.KC) continue;
    __threadfence();
    if (lane == 0) q.cnt[tile] = 0;
    float v[16];
#pragma unroll
    for (int k = 0; k < 16; k++) { const int i = lane + 32 * k, tk = i >> 4; v[k] = tk < mrem ? __ldcg(w + (size_t)tk * q.N + mt * 16 + (i & 15)) : 0.f; }
#pragma unroll
    for (int k = 0; k < 16; k++) {
      const int i = lane + 32 * k, tk = i >> 4, r = mt * 16 + (i & 15);
      if (tk < mrem) { w[(size_t)tk * q.N + r] = 0.f; put(q, tb * 32 + tk, r, v[k]); }
    }
  }
}

// split tiles: ws (fp32, sum of partials) -> out; ws cleared for the next call
__global__ void fin(Q q, int nflag) {
  const int MN = q.M * q.N;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < MN; i += gridDim.x * blockDim.x) {
    const int tok = i / q.N, row = i - tok * q.N;
    float* p = q.ws + i;
    const float v = *p;
    if (v != 0.f || nflag) { *p = 0.f; put(q, tok, row, v); }
  }
}

static int g_grid = 0;
template <int NT, bool HL> static void go(const Q& q, cudaStream_t st) {
  constexpr int RS = HL ? 3 : 6;
  const size_t sm = (size_t)WPC * RS * (HL ? 4096 : 2048) + WPC * RS * 8;
  static bool once = false;
  if (!once) { cudaFuncSetAttribute(skinny<NT, HL, RS>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm); once = true; }
  skinny<NT, HL, RS><<<g_grid, WPC * 32, sm, st>>>(q);
}
void run(torch::Tensor A, torch::Tensor A2, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int64_t N) {
  if (!g_grid) { int dev; cudaGetDevice(&dev); cudaDeviceGetAttribute(&g_grid, cudaDevAttrMultiProcessorCount, dev); }
  Q q;
  q.A = (const uint8_t*)A.data_ptr(); q.A2 = A2.numel() ? (const uint8_t*)A2.data_ptr() : nullptr;
  q.x = (const __nv_bfloat16*)x.data_ptr(); q.out = out.data_ptr(); q.ws = ws.data_ptr<float>(); q.cnt = cnt.data_ptr<int>();
  q.M = x.size(0); q.K = x.size(1); q.N = (int)N; q.MT = (int)((N + 15) / 16); q.KC = q.K / 64; q.TB = (q.M + 31) / 32;
  q.f32 = out.scalar_type() == torch::kFloat32;
  auto st = at::cuda::getCurrentCUDAStream();
  const int nt = q.M >= 32 ? 4 : (q.M + 7) / 8;
  if (q.A2) { if (nt <= 1) go<1, true>(q, st); else if (nt <= 2) go<2, true>(q, st); else go<4, true>(q, st); }
  else { if (nt <= 1) go<1, false>(q, st); else if (nt <= 2) go<2, false>(q, st); else go<4, false>(q, st); }
#ifdef TWOK
  fin<<<g_grid, 256, 0, st>>>(q, 1);
#endif
}
"""

_mod = None
_DBG = ["-DTWOK"]


def _big():
    pr = torch.cuda.get_device_properties(0)
    sm = getattr(pr, "shared_memory_per_block_optin", None)
    return sm >= 180000 if sm else (pr.major, pr.minor) in ((9, 0), (10, 0))


def _ext():
    global _mod
    if _mod is None:
        wpc = 16 if _big() else 8
        _mod = load_inline("skinny_v4tk_%d" % wpc + "".join(_DBG).replace("-", "_").replace("=", ""),
                           cpp_sources="void run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t);",
                           cuda_sources=CUDA, functions=["run"], extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo", "-DWPC=%d" % wpc] + _DBG,
                           verbose=False)
    return _mod


def _pack(w):
    """bf16 [N, K] -> A fragments [N/16, K/64, 32 lanes, 4 steps, 8] (K permuted so each lane's 16 k are contiguous)."""
    N, K = w.shape
    W = w.view(N // 16, 16, K // 64, 4, 4, 4)  # mt, i, ch, t, j, e
    lo, hi = W[:, :8], W[:, 8:]
    fr = torch.stack((lo[..., 0], lo[..., 1], hi[..., 0], hi[..., 1], lo[..., 2], lo[..., 3], hi[..., 2], hi[..., 3]), -1)
    return fr.permute(0, 2, 4, 1, 3, 5).contiguous()  # mt, ch, j, g, t, 8: per step j the 32 lanes are contiguous


def prepare(weights):
    st = {}
    for k, w in weights.items():
        if k == "router":
            hi = w.to(torch.bfloat16)
            lo = (w - hi.float()).to(torch.bfloat16)
            st[k] = (_pack(hi), _pack(lo))
        else:
            st[k] = (_pack(w), None)
        st[k + "_raw"] = w
    dev = weights["qkv_gate"].device
    st["ws"] = torch.zeros((256 * 8192,), dtype=torch.float32, device=dev)
    st["cnt"] = torch.zeros((8 * 512,), dtype=torch.int32, device=dev)
    st["none"] = torch.empty(0, dtype=torch.bfloat16, device=dev)
    _ext()
    return st


def kernel_fn(data, state):
    cfg = data["config"]
    layer, n = cfg["layer"], cfg["n"]
    x = data["x"]
    m = x.shape[0]
    if layer != "router" and m > 32:
        return x @ state[layer + "_raw"][:n].T
    A, A2 = state[layer]
    out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
    _ext().run(A, A2 if A2 is not None else state["none"], x.contiguous(), out, state["ws"], state["cnt"], n)
    return out
