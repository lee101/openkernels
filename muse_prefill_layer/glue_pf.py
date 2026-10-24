PF_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
// NVFP4 -> bf16 (x = e2m1 * e4m3 block scale * fp32 tensor scale), experts without local routes skipped
__global__ void deq(const uint8_t* w, const uint8_t* sf, const float* gs, const int* cnt, __nv_bfloat16* out, long long per_e) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;  // one 16-value block
  const long long e = i * 16 / per_e;
  if (i * 16 >= per_e * 32 || cnt[e] == 0) return;
  const uint2 q = *(const uint2*)(w + i * 8);
  __nv_fp8_e4m3 s8; s8.__x = sf[i];
  const float sc = (float)s8 * gs[e];
  const float grid[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  const uint32_t qq[2] = {q.x, q.y};
  __nv_bfloat16 o[16];
#pragma unroll
  for (int k = 0; k < 16; k++) {
    const uint32_t c = (qq[k >> 3] >> (4 * (k & 7))) & 15;
    const float m = grid[c & 7] * sc;
    o[k] = __float2bfloat16(c & 8 ? -m : m);
  }
  uint4* op = (uint4*)(out + i * 16);
  op[0] = *(uint4*)o; op[1] = *(uint4*)(o + 8);
}
// local routes grouped by expert: cnt / offsets / route list (token * 8 + slot)
__global__ void rcount(const int* loc, int R, int* cnt) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < R && loc[i] < 32) atomicAdd(cnt + loc[i], 1);
}
__global__ void rscan(const int* cnt, int* offs, int* fill) {
  const int l = threadIdx.x;
  int v = cnt[l], x = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) { const int n = __shfl_up_sync(0xffffffffu, x, o); if (l >= o) x += n; }
  offs[l] = x; fill[l] = x - v;
}
__global__ void rfill(const int* loc, int R, int* fill, int* list) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < R && loc[i] < 32) list[atomicAdd(fill + loc[i], 1)] = i;
}
// A[i] = x[token of route i] for i < total
__global__ void gath(const __nv_bfloat16* x, const int* list, const int* offs, __nv_bfloat16* A) {
  const int i = blockIdx.x, tot = offs[31];
  if (i >= tot) return;
  const uint4* src = (const uint4*)(x + (size_t)(list[i] >> 3) * 4096);
  uint4* dst = (uint4*)(A + (size_t)i * 4096);
  for (int c = threadIdx.x; c < 512; c += blockDim.x) dst[c] = src[c];
}
__global__ void swiglu(const __nv_bfloat16* H, const int* offs, __nv_bfloat16* I) {
  const int i = blockIdx.x;
  if (i >= offs[31]) return;
  for (int c = threadIdx.x; c < 4096; c += blockDim.x) {
    const float g = __bfloat162float(H[(size_t)i * 8192 + c]), u = __bfloat162float(H[(size_t)i * 8192 + 4096 + c]);
    I[(size_t)i * 4096 + c] = __float2bfloat16(g / (1.f + __expf(-g)) * u);
  }
}
// y * (1 + w) -> bf16 -> RMSNorm (1e-8) -> x routing weight, accumulated per token in fp32
__global__ void post(const __nv_bfloat16* Y, const int* list, const int* offs, const float* gw, const float* tw, float* acc) {
  const int i = blockIdx.x;
  if (i >= offs[31]) return;
  __shared__ float red[8];
  const int r = list[i], tok = r >> 3, c0 = threadIdx.x * 16;  // 256 threads x 16
  float v[16], ss = 0.f;
#pragma unroll
  for (int k = 0; k < 16; k++) { v[k] = __bfloat162float(__float2bfloat16(__bfloat162float(Y[(size_t)i * 4096 + c0 + k]) * gw[c0 + k])); ss += v[k] * v[k]; }
#pragma unroll
  for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
  __syncthreads();
  float t = 0.f;
#pragma unroll
  for (int k = 0; k < 8; k++) t += red[k];
  const float cf = tw[r] * rsqrtf(t / 4096 + 1e-8f);
#pragma unroll
  for (int k = 0; k < 16; k++) atomicAdd(acc + (size_t)tok * 4096 + c0 + k, v[k] * cf);
}
__global__ void tobf(const float* a, __nv_bfloat16* o, long long n) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) o[i] = __float2bfloat16(a[i]);
}
void dequant(torch::Tensor w, torch::Tensor sf, torch::Tensor gs, torch::Tensor cnt, torch::Tensor out) {
  const long long per_e = w.numel() * 2 / 32, nb = per_e * 32 / 16;
  deq<<<(nb + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(w.data_ptr<uint8_t>(), (const uint8_t*)sf.data_ptr(), gs.data_ptr<float>(), cnt.data_ptr<int>(), (__nv_bfloat16*)out.data_ptr(), per_e);
}
void routes(torch::Tensor loc, torch::Tensor cnt, torch::Tensor offs, torch::Tensor fill, torch::Tensor list) {
  auto st = at::cuda::getCurrentCUDAStream();
  const int R = loc.numel();
  cudaMemsetAsync(cnt.data_ptr<int>(), 0, 32 * 4, st);
  rcount<<<(R + 255) / 256, 256, 0, st>>>(loc.data_ptr<int>(), R, cnt.data_ptr<int>());
  rscan<<<1, 32, 0, st>>>(cnt.data_ptr<int>(), offs.data_ptr<int>(), fill.data_ptr<int>());
  rfill<<<(R + 255) / 256, 256, 0, st>>>(loc.data_ptr<int>(), R, fill.data_ptr<int>(), list.data_ptr<int>());
}
void gather(torch::Tensor x, torch::Tensor list, torch::Tensor offs, torch::Tensor A) {
  gath<<<A.size(0), 128, 0, at::cuda::getCurrentCUDAStream()>>>((const __nv_bfloat16*)x.data_ptr(), list.data_ptr<int>(), offs.data_ptr<int>(), (__nv_bfloat16*)A.data_ptr());
}
void act(torch::Tensor H, torch::Tensor offs, torch::Tensor I) {
  swiglu<<<H.size(0), 256, 0, at::cuda::getCurrentCUDAStream()>>>((const __nv_bfloat16*)H.data_ptr(), offs.data_ptr<int>(), (__nv_bfloat16*)I.data_ptr());
}
void finish(torch::Tensor Y, torch::Tensor list, torch::Tensor offs, torch::Tensor gw, torch::Tensor tw, torch::Tensor acc, torch::Tensor out) {
  auto st = at::cuda::getCurrentCUDAStream();
  cudaMemsetAsync(acc.data_ptr<float>(), 0, acc.numel() * 4, st);
  post<<<Y.size(0), 256, 0, st>>>((const __nv_bfloat16*)Y.data_ptr(), list.data_ptr<int>(), offs.data_ptr<int>(), gw.data_ptr<float>(), tw.data_ptr<float>(), acc.data_ptr<float>());
  tobf<<<(acc.numel() + 255) / 256, 256, 0, st>>>(acc.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr(), acc.numel());
}
"""
_pfmod = None


def _pfext():
    global _pfmod
    if _pfmod is None:
        _pfmod = load_inline("pf_moe_v1", cpp_sources="void dequant(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor); void routes(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor); void gather(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor); void act(torch::Tensor,torch::Tensor,torch::Tensor); void finish(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor);",
                             cuda_sources=PF_CUDA, functions=["dequant", "routes", "gather", "act", "finish"], extra_cuda_cflags=["-O3"], verbose=False)
    return _pfmod


def _gmm(A, B, offs):
    """grouped A[rows of group e] @ B[e]; CUTLASS grouped GEMM on Hopper, a per-group loop elsewhere (host sync: tests only)"""
    if torch.cuda.get_device_capability() == (9, 0):
        return torch._grouped_mm(A, B, offs=offs)
    out = torch.zeros((A.shape[0], B.shape[2]), dtype=torch.bfloat16, device=A.device)
    o = [0] + offs.tolist()
    for e in range(B.shape[0]):
        if o[e + 1] > o[e]: out[o[e]:o[e + 1]] = A[o[e]:o[e + 1]] @ B[e]
    return out


def _moe_large(p, loc, tw, st):
    """prefill-sized MoE: dequantize the routed experts to bf16, two grouped GEMMs, fused post-norm combine"""
    T = p.shape[0]
    dev = p.device
    g = _pfext()
    R = T * 8
    cnt = torch.empty(32, dtype=torch.int32, device=dev); offs = torch.empty_like(cnt); fill = torch.empty_like(cnt)
    lst = torch.empty(R, dtype=torch.int32, device=dev)
    g.routes(loc.view(-1), cnt, offs, fill, lst)
    w13 = torch.empty((32, 8192, 4096), dtype=torch.bfloat16, device=dev)
    w2 = torch.empty((32, 4096, 4096), dtype=torch.bfloat16, device=dev)
    g.dequant(st["pw13"], st["pw13_sf"], st["pw13_gs"], cnt, w13)
    g.dequant(st["pw2"], st["pw2_sf"], st["pw2_gs"], cnt, w2)
    A = torch.empty((R, 4096), dtype=torch.bfloat16, device=dev)
    g.gather(p, lst, offs, A)
    H = _gmm(A, w13.transpose(1, 2), offs)
    I = torch.empty((R, 4096), dtype=torch.bfloat16, device=dev)
    g.act(H, offs, I)
    Y = _gmm(I, w2.transpose(1, 2), offs)
    acc = torch.empty((T, 4096), dtype=torch.float32, device=dev)
    out = torch.empty((T, 4096), dtype=torch.bfloat16, device=dev)
    g.finish(Y, lst, offs, st["g_post_expert_norm"], tw.view(-1), acc, out)
    return out
