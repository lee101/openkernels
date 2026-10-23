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

template <int NT, bool HL>
__device__ __forceinline__ void seg(const Q& q, int tile, int c0, int c1, int lane) {
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
  load(0, c0);
  int c = c0;
  for (; c + 1 < c1; c += 2) {
    load(1, c + 1);
    comp(0);
    if (c + 2 < c1) load(0, c + 2);
    comp(1);
  }
  if (c < c1) comp(0);
  float f[NT][4];
#pragma unroll
  for (int n = 0; n < NT; n++)
#pragma unroll
    for (int i = 0; i < 4; i++) { float v = 0.f;
#pragma unroll
      for (int h = 0; h < CH; h++) v += acc[n][h][i];
      f[n][i] = v; }
  const int row0 = mt * 16 + g;
  auto put = [&](int tok, int row, float v) {
    if (tok >= q.M || row >= q.N) return;
    if (q.f32) ((float*)q.out)[(size_t)tok * q.N + row] = v; else ((__nv_bfloat16*)q.out)[(size_t)tok * q.N + row] = __float2bfloat16(v);
  };
  if (c0 == 0 && c1 == q.KC) {
#pragma unroll
    for (int n = 0; n < NT; n++) {
      const int tk = tb * 32 + n * 8 + 2 * t;
      put(tk, row0, f[n][0]); put(tk + 1, row0, f[n][1]); put(tk, row0 + 8, f[n][2]); put(tk + 1, row0 + 8, f[n][3]);
    }
    return;
  }
  // partial: accumulate into ws[tok][row] (tok within this token block), last arriver writes out
  float* w = q.ws + (size_t)tb * 32 * q.N;
  const int mrem = min(32, q.M - tb * 32);
#pragma unroll
  for (int n = 0; n < NT; n++) {
    const int tk = n * 8 + 2 * t;
    if (tk < mrem) { atomicAdd(w + (size_t)tk * q.N + row0, f[n][0]); atomicAdd(w + (size_t)tk * q.N + row0 + 8, f[n][2]); }
    if (tk + 1 < mrem) { atomicAdd(w + (size_t)(tk + 1) * q.N + row0, f[n][1]); atomicAdd(w + (size_t)(tk + 1) * q.N + row0 + 8, f[n][3]); }
  }
  __threadfence(); __syncwarp();
  int old = 0;
  if (lane == 0) old = atomicAdd(q.cnt + tile, c1 - c0);
  old = __shfl_sync(0xffffffffu, old, 0);
  if (old + c1 - c0 != q.KC) return;
  __threadfence();
  if (lane == 0) q.cnt[tile] = 0;
  float v[16];  // this tile: mrem tokens x 16 rows; all loads in flight before any store
#pragma unroll
  for (int k = 0; k < 16; k++) { const int i = lane + 32 * k, tk = i >> 4; v[k] = tk < mrem ? __ldcg(w + (size_t)tk * q.N + mt * 16 + (i & 15)) : 0.f; }
#pragma unroll
  for (int k = 0; k < 16; k++) {
    const int i = lane + 32 * k, tk = i >> 4, r = mt * 16 + (i & 15);
    if (tk < mrem) { w[(size_t)tk * q.N + r] = 0.f; put(tb * 32 + tk, r, v[k]); }
  }
}

template <bool HL>
__global__ void __launch_bounds__(WPC * 32, 1) skinny(Q q) {
  const int lane = threadIdx.x & 31, W = gridDim.x * WPC, gw = blockIdx.x * WPC + (threadIdx.x >> 5);
  const long long U = (long long)q.TB * q.MT * q.KC;
  long long L = U * gw / W;
  const long long Le = U * (gw + 1) / W;
  while (L < Le) {
    const int tile = (int)(L / q.KC), c0 = (int)(L - (long long)tile * q.KC);
    const int c1 = (int)min((long long)q.KC, c0 + (Le - L));
    const int tb = tile / q.MT, rem = q.M - tb * 32, nt = rem >= 32 ? 4 : (rem + 7) >> 3;
    if (nt <= 1) seg<1, HL>(q, tile, c0, c1, lane);
    else if (nt <= 2) seg<2, HL>(q, tile, c0, c1, lane);
    else seg<4, HL>(q, tile, c0, c1, lane);
    L += c1 - c0;
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
        _mod = load_inline("skinny_v1" + "".join(_DBG).replace("-", "_").replace("=", ""),
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
    _fixed(st, weights)
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


_FC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
__global__ void k_empty() {}
__global__ void k_one(float* p) { if (threadIdx.x == 0) p[blockIdx.x] = 1.f; }
__global__ void k_atom(float* p, int* c) { if (threadIdx.x == 0) { atomicAdd(p + blockIdx.x, 1.f); __threadfence(); atomicAdd(c, 1); } }
void empty(int64_t g) { k_empty<<<g, 32, 0, at::cuda::getCurrentCUDAStream()>>>(); }
void one(torch::Tensor p, int64_t g) { k_one<<<g, 32, 0, at::cuda::getCurrentCUDAStream()>>>(p.data_ptr<float>()); }
void atom(torch::Tensor p, torch::Tensor c, int64_t g) { k_atom<<<g, 32, 0, at::cuda::getCurrentCUDAStream()>>>(p.data_ptr<float>(), c.data_ptr<int>()); }
"""


def _gt(fn, reps=50):
    import statistics
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn(); fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    fl = torch.ones(2 ** 27, device="cuda"); sink = torch.empty((), device="cuda")
    ts = []
    for _ in range(reps):
        sink.copy_(fl.sum()); a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1e3)
    return statistics.median(ts)


def _fixed(st, weights):
    m = load_inline("fixedcost", cpp_sources="void empty(int64_t); void one(torch::Tensor,int64_t); void atom(torch::Tensor,torch::Tensor,int64_t);",
                    cuda_sources=_FC, functions=["empty", "one", "atom"], verbose=False)
    p = torch.zeros(4096, device="cuda"); c = torch.zeros(1, dtype=torch.int32, device="cuda")
    print("FIX empty-graph", round(_gt(lambda: None), 2), flush=True)
    for g in (1, 132, 1056):
        print("FIX empty-kernel g", g, round(_gt(lambda: m.empty(g)), 2), " one", round(_gt(lambda: m.one(p, g)), 2), " atom", round(_gt(lambda: m.atom(p, c, g)), 2), flush=True)
    for name, M in (("o", 1), ("unpooler", 1), ("router", 8), ("qkv_gate", 1), ("o", 32)):
        W = weights[name]; n = 1024 if (name == "unpooler" and M <= 8) else W.shape[0]
        x = torch.randn((M, W.shape[1]), device="cuda").bfloat16()
        ref = (lambda: x.float() @ W.T) if name == "router" else (lambda: x @ W[:n].T)
        d = {"x": x, "config": {"m": M, "n": n, "k": W.shape[1], "layer": name}}
        print("FIX", name, M, "torch", round(_gt(ref), 2), "ours", round(_gt(lambda: kernel_fn(d, st)), 2), flush=True)
