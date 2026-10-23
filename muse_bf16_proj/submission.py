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
#define WPC 16

__device__ __forceinline__ void mma(float* c, const uint4& a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint4 ldnc(const uint4* p) { uint4 v; asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v; }

struct Q {
  const uint4* A; const uint4* A2; const __nv_bfloat16* x; void* out; float* ws; int* cnt;
  int M, N, K, MT, KC, TB, f32;
};

// a CTA's warps stream chunks w, w+16, ... of one (tile, [c0, c1)) segment; partial sums land in smem
template <int NT, bool HL>
__device__ __forceinline__ void part(const Q& q, int tile, int c0, int c1, int lane, int wid, float4* red) {
  const int tb = tile / q.MT, mt = tile - tb * q.MT, g = lane >> 2, t = lane & 3;
  constexpr int CH = NT == 1 ? 4 : (NT == 2 ? 2 : 1);
  const uint4* xr[NT];
#pragma unroll
  for (int n = 0; n < NT; n++) {
    int tok = min(tb * 32 + n * 8 + g, q.M - 1);
    xr[n] = (const uint4*)(q.x + (size_t)tok * q.K) + t * 2;
  }
  const uint4* ap = q.A + ((size_t)mt * q.KC) * 128 + lane * 4;
  const uint4* ap2 = HL ? q.A2 + ((size_t)mt * q.KC) * 128 + lane * 4 : nullptr;
  float acc[NT][CH][4];
#pragma unroll
  for (int n = 0; n < NT; n++)
#pragma unroll
    for (int h = 0; h < CH; h++) acc[n][h][0] = acc[n][h][1] = acc[n][h][2] = acc[n][h][3] = 0.f;
  uint4 a[2][4], a2[2][4], b[2][NT][2];
  auto load = [&](int s, int c) {
#pragma unroll
    for (int j = 0; j < 4; j++) { a[s][j] = ldnc(ap + (size_t)c * 128 + j); if (HL) a2[s][j] = ldnc(ap2 + (size_t)c * 128 + j); }
#pragma unroll
    for (int n = 0; n < NT; n++) { b[s][n][0] = __ldg(xr[n] + c * 8); b[s][n][1] = __ldg(xr[n] + c * 8 + 1); }
  };
  auto comp = [&](int s) {
#pragma unroll
    for (int j = 0; j < 4; j++)
#pragma unroll
      for (int n = 0; n < NT; n++) {
        const uint32_t* bb = (const uint32_t*)&b[s][n][j >> 1];
        mma(acc[n][j % CH], a[s][j], bb[(j & 1) * 2], bb[(j & 1) * 2 + 1]);
        if (HL) mma(acc[n][(j + 1) % CH], a2[s][j], bb[(j & 1) * 2], bb[(j & 1) * 2 + 1]);
      }
  };
  int c = c0 + wid;
  if (c < c1) load(0, c);
  for (; c + WPC < c1; c += 2 * WPC) {
    load(1, c + WPC);
    comp(0);
    if (c + 2 * WPC < c1) load(0, c + 2 * WPC);
    comp(1);
  }
  if (c < c1) comp(0);
#pragma unroll
  for (int n = 0; n < NT; n++) {
    float4 v;
    v.x = v.y = v.z = v.w = 0.f;
#pragma unroll
    for (int h = 0; h < CH; h++) { v.x += acc[n][h][0]; v.y += acc[n][h][1]; v.z += acc[n][h][2]; v.w += acc[n][h][3]; }
    red[(wid * 4 + n) * 32 + lane] = v;
  }
}

__device__ __forceinline__ void put(const Q& q, int tok, int row, float v) {
  if (q.f32) ((float*)q.out)[(size_t)tok * q.N + row] = v; else ((__nv_bfloat16*)q.out)[(size_t)tok * q.N + row] = __float2bfloat16(v);
}

template <bool HL>
__global__ void __launch_bounds__(WPC * 32, 1) skinny(Q q) {
  __shared__ float4 red[WPC * 4 * 32];
  __shared__ int last;
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
  const long long U = (long long)q.TB * q.MT * q.KC;
  const long long a = U * blockIdx.x / gridDim.x, b = U * (blockIdx.x + 1) / gridDim.x;
  if (a >= b) return;
  // segments: boundary pieces first (their cross-CTA reduction then hides behind the whole tiles)
  long long sa[3], sb[3]; int ns = 0;
  const long long t0 = a / q.KC, t1 = (b - 1) / q.KC;
  const long long fa = (a % q.KC) ? t0 + 1 : t0, fb = (b % q.KC) ? t1 : t1 + 1;  // full tiles [fa, fb)
  if (t0 == t1 && (a % q.KC || b % q.KC)) { sa[ns] = a; sb[ns++] = b; }
  else {
    if (a % q.KC) { sa[ns] = a; sb[ns++] = (t0 + 1) * q.KC; }
    if (b % q.KC) { sa[ns] = t1 * q.KC; sb[ns++] = b; }
  }
  const int nfull = (t0 == t1 && (a % q.KC || b % q.KC)) ? 0 : (int)(fb - fa);
  for (int i = 0; i < ns + nfull; i++) {
    long long s0, s1;
    if (i < ns) { s0 = sa[i]; s1 = sb[i]; } else { s0 = (fa + i - ns) * q.KC; s1 = s0 + q.KC; }
    const int tile = (int)(s0 / q.KC), c0 = (int)(s0 - (long long)tile * q.KC), c1 = (int)(s1 - (long long)tile * q.KC);
    const int tb = tile / q.MT, mt = tile - tb * q.MT, mrem = min(32, q.M - tb * 32), nt = mrem >= 32 ? 4 : (mrem + 7) >> 3;
    if (nt <= 1) part<1, HL>(q, tile, c0, c1, lane, wid, red);
    else if (nt <= 2) part<2, HL>(q, tile, c0, c1, lane, wid, red);
    else part<4, HL>(q, tile, c0, c1, lane, wid, red);
    __syncthreads();
    const bool full = c0 == 0 && c1 == q.KC;
    float* w = q.ws + (size_t)tb * 32 * q.N;
    // thread -> (n, lane', i): sum the 16 warp partials
    float v = 0.f; int tok = -1, row = 0;
    if (tid < nt * 128) {
      const int n = tid >> 7, ln = (tid >> 2) & 31, e = tid & 3;
      const float* rf = (const float*)red;
#pragma unroll
      for (int ww = 0; ww < WPC; ww++) v += rf[((ww * 4 + n) * 32 + ln) * 4 + e];
      const int tk = n * 8 + 2 * (ln & 3) + (e & 1);
      if (tk < mrem) { tok = tk; row = mt * 16 + (ln >> 2) + 8 * (e >> 1); }
    }
    if (full) {
      if (tok >= 0) put(q, tb * 32 + tok, row, v);
    } else {
      if (tok >= 0) atomicAdd(w + (size_t)tok * q.N + row, v);
      __threadfence();
      __syncthreads();
      if (tid == 0) last = atomicAdd(q.cnt + tile, c1 - c0) + c1 - c0 == q.KC;
      __syncthreads();
      if (last) {
        __threadfence();
        if (tok >= 0) { float* p = w + (size_t)tok * q.N + row; const float r = __ldcg(p); *p = 0.f; put(q, tb * 32 + tok, row, r); }
        if (tid == 0) q.cnt[tile] = 0;
      }
    }
    __syncthreads();
  }
}

static int g_grid = 0;
void run(torch::Tensor A, torch::Tensor A2, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int64_t N) {
  if (!g_grid) { int dev; cudaGetDevice(&dev); cudaDeviceGetAttribute(&g_grid, cudaDevAttrMultiProcessorCount, dev); }
  Q q;
  q.A = (const uint4*)A.data_ptr(); q.A2 = A2.numel() ? (const uint4*)A2.data_ptr() : nullptr;
  q.x = (const __nv_bfloat16*)x.data_ptr(); q.out = out.data_ptr(); q.ws = ws.data_ptr<float>(); q.cnt = cnt.data_ptr<int>();
  q.M = x.size(0); q.K = x.size(1); q.N = (int)N; q.MT = (int)((N + 15) / 16); q.KC = q.K / 64; q.TB = (q.M + 31) / 32;
  q.f32 = out.scalar_type() == torch::kFloat32;
  auto st = at::cuda::getCurrentCUDAStream();
  if (q.A2) skinny<true><<<g_grid, WPC * 32, 0, st>>>(q); else skinny<false><<<g_grid, WPC * 32, 0, st>>>(q);
}
"""

_mod = None
_DBG = []


def _ext():
    global _mod
    if _mod is None:
        _mod = load_inline("skinny_v2" + "".join(_DBG).replace("-", "_").replace("=", ""),
                           cpp_sources="void run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t);",
                           cuda_sources=CUDA, functions=["run"], extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"] + _DBG,
                           verbose=False)
    return _mod


def _pack(w):
    """bf16 [N, K] -> A fragments [N/16, K/64, 32 lanes, 4 steps, 8] (K permuted so each lane's 16 k are contiguous)."""
    N, K = w.shape
    W = w.view(N // 16, 16, K // 64, 4, 4, 4)  # mt, i, ch, t, j, e
    lo, hi = W[:, :8], W[:, 8:]
    fr = torch.stack((lo[..., 0], lo[..., 1], hi[..., 0], hi[..., 1], lo[..., 2], lo[..., 3], hi[..., 2], hi[..., 3]), -1)
    return fr.permute(0, 2, 1, 3, 4, 5).contiguous()  # mt, ch, g, t, j, 8


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
