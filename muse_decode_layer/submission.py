# okc: problem=muse_decode_layer
"""Decoder layer: cuBLAS projections + fused split-KV FP8 attention + fused residual norms + NVFP4 W4A16 MoE."""
from typing import TypedDict
import torch
from torch.utils.cpp_extension import load_inline

"""Gated attention decode over a paged FP8 cache in 3 launches:
prep (QK RMSNorm, RoPE, FP8 cache write, FP16 query) -> split-KV flash decoding on tensor cores straight from the
scattered FP8 cache rows -> split combine + per-head RMSNorm x sigmoid(gate)."""
from typing import TypedDict
import torch
from torch.utils.cpp_extension import load_inline


class Config(TypedDict):
    batch: int
    q_len: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    window: int
    rope_theta: float
    qk_scale_factor: float
    sm_scale: float
    kv_scale: float


class Inputs(TypedDict):
    qkv: torch.Tensor
    positions: torch.Tensor
    cache_loc: torch.Tensor
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    scheduler_metadata: torch.Tensor | None
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    config: Config


A_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <cooperative_groups.h>
#define D 64
#ifndef CLUS
#define CLUS 1
#endif
#define LOG2E 1.4426950408889634f
#ifndef PFF
#define PFF 0
#endif

// ---------------------------------------------------------------- prep: one CTA (64 x Hq+2Hkv threads) per new token
__device__ __forceinline__ void rope_cs(const float* invf, int pos, int pair, float& sn, float& cs);
__global__ void prep(const __nv_bfloat16* qkv, const int64_t* pos, const int64_t* loc, uint8_t* kc, uint8_t* vc,
                     const float* ksc, const float* vsc, __half* qo, uint8_t* q8, int hq, int hkv, float theta, float qsc, const float* invf) {
  const int tok = blockIdx.x, hd = blockIdx.y, d = threadIdx.x;  // one 64-thread block per (token, head)
  const int W = (2 * hq + 2 * hkv) * D;
  const __nv_bfloat16* row = qkv + (size_t)tok * W;
  __shared__ float red[2];
  float x;
  const bool isq = hd < hq, isv = hd >= hq + hkv;  // heads: q [0, hq), k [hq, hq+hkv), v [hq+hkv, hq+2hkv)
  if (isv) {
    const int h = hd - hq - hkv;
    x = __bfloat162float(row[(hq + hkv) * D + h * D + d]);
    __nv_fp8_e4m3 f(x / *vsc);
    vc[(size_t)loc[tok] * hkv * D + h * D + d] = f.__x;
#ifdef PTRACE
    if (d == 0 && tok < 3) printf("V tok %d h %d loc %lld idx %llu\n", tok, h, (long long)loc[tok], (unsigned long long)((size_t)loc[tok] * hkv * D + h * D));
#endif
    return;
  }
  x = __bfloat162float(row[hd * D + d]);
  float ss = x * x;
#pragma unroll
  for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  if ((d & 31) == 0) red[d >> 5] = ss;
  __syncthreads();
  ss = red[0] + red[1];
  x = x * rsqrtf(ss / D + 1e-5f);
  if (isq) x *= qsc;
  if (theta > 0.f) {  // interleaved RoPE: pair (2i, 2i+1), angle pos * theta^(-2i/64), fp32 as the reference
    float sn, cs; rope_cs(invf, (int)pos[tok], d >> 1, sn, cs);
    const float other = __shfl_xor_sync(0xffffffffu, x, 1);
    x = (d & 1) ? other * sn + x * cs : x * cs - other * sn;
  }
  if (isq) {
    if (q8) { __nv_fp8_e4m3 f(x / qsc); q8[((size_t)tok * hq + hd) * D + d] = f.__x; }  // E4M3 rounds the normalized query
    else qo[((size_t)tok * hq + hd) * D + d] = __float2half_rn(x);
  }
  else {
    const int h = hd - hq;
    __nv_fp8_e4m3 f(x / *ksc);
    kc[(size_t)loc[tok] * hkv * D + h * D + d] = f.__x;
  }
}

// ---------------------------------------------------------------- attention: one warp per (request, kv head, split)
#ifndef MMAVOL
#define MMAVOL 0
#endif
#if MMAVOL
#define MMA_ASM asm volatile
#else
#define MMA_ASM asm
#endif
__device__ __forceinline__ void mma16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  MMA_ASM("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t e2h(uint32_t two) {  // 2 x e4m3 (low 16 bits) -> f16x2
  __half2_raw r = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(two & 0xffff), __NV_E4M3);
  return *(uint32_t*)&r;
}
__device__ __forceinline__ float ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ uint32_t pk(float a, float b) { __half2 h = __floats2half2_rn(a, b); return *(uint32_t*)&h; }
__device__ __forceinline__ uint4 ldg16(const void* p) { return __ldg((const uint4*)p); }
__device__ __forceinline__ uint2 ldg8(const void* p) { return __ldg((const uint2*)p); }

struct A {
  const __half* q; const uint8_t* q8; const uint8_t* kc; const uint8_t* vc; const int* pt; const int* seq; float* wo; float* wml;
  int B, L, hq, hkv, ps, maxp, window, nsplit, split, rows;
  float qk;  // k_scale * sm_scale * log2(e): S in log2 units
};

// rows: L * group (group = hq/hkv = 8); MT m-tiles of 16
#ifndef RB
#define RB 8
#endif
template <int MT>
__global__ void __launch_bounds__(256) attn(A a) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int item = blockIdx.x * 8 + (threadIdx.x >> 5);
  if (item >= a.B * a.hkv * a.nsplit) return;
  const int h = item % a.hkv, rest = item / a.hkv, sp = rest % a.nsplit, b = rest / a.nsplit;
  const int grp = a.hq / a.hkv;
  const int seq = a.seq[b], nk = seq + a.L;  // keys 0 .. seq+L-1 (new tokens already in the cache)
  int k0 = a.window ? max(0, seq - a.window + 1) : 0;
  const int span = nk - k0;
  const int ks = k0 + (int)((long long)span * sp / a.nsplit), ke = k0 + (int)((long long)span * (sp + 1) / a.nsplit);
  const size_t it = ((size_t)b * a.hkv + h) * a.nsplit + sp;
  float* wo = a.wo + it * a.rows * D;
  float* wml = a.wml + it * a.rows * 2;
  // Q A-fragments: row r = l * grp + head (head within this kv group); lane needs q[r][16t .. 16t+15]
  uint32_t qa[MT][4][4];
#pragma unroll
  for (int m = 0; m < MT; m++)
#pragma unroll
    for (int hh = 0; hh < 2; hh++) {
      const int r = m * 16 + g + hh * 8, l = r / grp, hd = h * grp + r % grp;
      uint4 v0 = make_uint4(0, 0, 0, 0), v1 = v0;
      if (r < a.rows) {
        const __half* qp = a.q + (((size_t)b * a.L + l) * a.hq + hd) * D + t * 16;
        v0 = *(const uint4*)qp; v1 = *(const uint4*)(qp + 8);
      }
      const uint32_t w[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma unroll
      for (int j = 0; j < 4; j++) { qa[m][j][hh] = w[2 * j]; qa[m][j][2 + hh] = w[2 * j + 1]; }
    }
  float o[MT][8][4], mx[MT][2], ls[MT][2];
#pragma unroll
  for (int m = 0; m < MT; m++) {
    mx[m][0] = mx[m][1] = -INFINITY; ls[m][0] = ls[m][1] = 0.f;
#pragma unroll
    for (int n = 0; n < 8; n++) o[m][n][0] = o[m][n][1] = o[m][n][2] = o[m][n][3] = 0.f;
  }
  const int* ptb = a.pt + (size_t)b * a.maxp;
  auto rowof = [&](int kp) -> int { return ptb[kp / a.ps] * a.ps + kp % a.ps; };
  // per-warp cp.async ring of RB key blocks (16 keys: K 2 x 16B + V 4 x 8B per lane, read back by the same lane)
  extern __shared__ __align__(16) uint8_t smem[];
  const uint32_t ring = (uint32_t)__cvta_generic_to_shared(smem) + (threadIdx.x >> 5) * RB * 2048;
  auto rowsfor = [&](int kb) -> int { const int kp = kb + (lane & 15); return (kb < ke && kp < ke) ? rowof(kp) : -1; };
  auto cpa = [&](uint32_t dst, const void* src, int n, bool ok) {
    if (n == 16) asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(ok ? 16 : 0));
    else asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" :: "r"(dst), "l"(src), "r"(ok ? 8 : 0));
  };
  auto issue = [&](int slot, int rl) {
    const uint32_t d0 = ring + slot * 2048;
    const int rK0 = __shfl_sync(0xffffffffu, rl, g), rK1 = __shfl_sync(0xffffffffu, rl, g + 8);
    const int rV[4] = {__shfl_sync(0xffffffffu, rl, 2 * t), __shfl_sync(0xffffffffu, rl, 2 * t + 1),
                       __shfl_sync(0xffffffffu, rl, 2 * t + 8), __shfl_sync(0xffffffffu, rl, 2 * t + 9)};
    cpa(d0 + lane * 16, a.kc + (size_t)max(rK0, 0) * a.hkv * D + h * D + t * 16, 16, rK0 >= 0);
    cpa(d0 + 512 + lane * 16, a.kc + (size_t)max(rK1, 0) * a.hkv * D + h * D + t * 16, 16, rK1 >= 0);
#pragma unroll
    for (int i = 0; i < 4; i++) cpa(d0 + 1024 + i * 256 + lane * 8, a.vc + (size_t)max(rV[i], 0) * a.hkv * D + h * D + g * 8, 8, rV[i] >= 0);
    asm volatile("cp.async.commit_group;\n");
  };
  const int nb = (ke - ks + 15) >> 4;
  int rnext = rowsfor(ks + (RB - 1) * 16);
#pragma unroll
  for (int i = 0; i < RB - 1; i++) issue(i, rowsfor(ks + i * 16));
  for (int bi = 0, kb = ks; bi < nb; bi++, kb += 16) {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(RB - 2));
    const int r2 = rowsfor(kb + RB * 16);
    issue((bi + RB - 1) % RB, rnext);
    rnext = r2;
    const uint32_t sl = ring + (bi % RB) * 2048;
    uint4 kw0, kw1; uint2 vw[4];
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(kw0.x), "=r"(kw0.y), "=r"(kw0.z), "=r"(kw0.w) : "r"(sl + lane * 16));
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(kw1.x), "=r"(kw1.y), "=r"(kw1.z), "=r"(kw1.w) : "r"(sl + 512 + lane * 16));
#pragma unroll
    for (int i = 0; i < 4; i++) asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];\n" : "=r"(vw[i].x), "=r"(vw[i].y) : "r"(sl + 1024 + i * 256 + lane * 8));
    // S = Q K^T (log2 units)
    const uint32_t k0w[4] = {kw0.x, kw0.y, kw0.z, kw0.w}, k1w[4] = {kw1.x, kw1.y, kw1.z, kw1.w};
    float s[MT][2][4];
#pragma unroll
    for (int m = 0; m < MT; m++)
#pragma unroll
      for (int n = 0; n < 2; n++) {
        s[m][n][0] = s[m][n][1] = s[m][n][2] = s[m][n][3] = 0.f;
#pragma unroll
        for (int j = 0; j < 4; j++) {
          const uint32_t w = n ? k1w[j] : k0w[j];
          mma16(s[m][n], qa[m][j][0], qa[m][j][1], qa[m][j][2], qa[m][j][3], e2h(w), e2h(w >> 16));
        }
      }
    // mask: key kp = kb + n*8 + 2t + e; row r -> token l: visible if k0(l) <= kp <= seq + l and kp < ke
#pragma unroll
    for (int m = 0; m < MT; m++)
#pragma unroll
      for (int hh = 0; hh < 2; hh++) {
        const int r = m * 16 + g + hh * 8, l = min(r / grp, a.L - 1), p = seq + l;
        const int lo = a.window ? max(0, p - a.window + 1) : 0;
#pragma unroll
        for (int n = 0; n < 2; n++)
#pragma unroll
          for (int e = 0; e < 2; e++) {
            const int kp = kb + n * 8 + 2 * t + e;
            float& v = s[m][n][hh * 2 + e];
            v = (kp < ke && kp <= p && kp >= lo) ? v * a.qk : -INFINITY;
          }
      }
    // online softmax
    uint32_t pa[MT][4];
#pragma unroll
    for (int m = 0; m < MT; m++)
#pragma unroll
      for (int hh = 0; hh < 2; hh++) {
        float bm = fmaxf(fmaxf(s[m][0][hh * 2], s[m][0][hh * 2 + 1]), fmaxf(s[m][1][hh * 2], s[m][1][hh * 2 + 1]));
        bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, 1));
        bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, 2));
        const float nm = fmaxf(mx[m][hh], bm);
        const float sc = nm == -INFINITY ? 1.f : ex2(mx[m][hh] - nm);
        mx[m][hh] = nm;
        float p0 = nm == -INFINITY ? 0.f : ex2(s[m][0][hh * 2] - nm), p1 = nm == -INFINITY ? 0.f : ex2(s[m][0][hh * 2 + 1] - nm);
        float p2 = nm == -INFINITY ? 0.f : ex2(s[m][1][hh * 2] - nm), p3 = nm == -INFINITY ? 0.f : ex2(s[m][1][hh * 2 + 1] - nm);
        ls[m][hh] = ls[m][hh] * sc + (p0 + p1 + p2 + p3);
#pragma unroll
        for (int n = 0; n < 8; n++) { o[m][n][hh * 2] *= sc; o[m][n][hh * 2 + 1] *= sc; }
        pa[m][hh] = pk(p0, p1);      // a0 / a1: keys 2t, 2t+1
        pa[m][2 + hh] = pk(p2, p3);  // a2 / a3: keys 8 + 2t, 8 + 2t + 1
      }
    // O += P V; output col (n, g') <-> head dim 8 g' + n, so this lane's V bytes are [8g, 8g+8) of 4 key rows
#pragma unroll
    for (int n = 0; n < 8; n += 2) {
      const int wi = n >> 2, sh = (n & 3) * 8;
      const uint32_t x0 = wi ? vw[0].y : vw[0].x, x1 = wi ? vw[1].y : vw[1].x, x2 = wi ? vw[2].y : vw[2].x, x3 = wi ? vw[3].y : vw[3].x;
      // pair bytes (key 2t, key 2t+1) for dims n, n+1 and (key 2t+8, 2t+9)
      const uint32_t lo01 = __byte_perm(x0 >> sh, x1 >> sh, 0x5140), lo23 = __byte_perm(x2 >> sh, x3 >> sh, 0x5140);
      const uint32_t b0n = e2h(lo01), b1n = e2h(lo23), b0m = e2h(lo01 >> 16), b1m = e2h(lo23 >> 16);
#pragma unroll
      for (int m = 0; m < MT; m++) {
        mma16(o[m][n], pa[m][0], pa[m][1], pa[m][2], pa[m][3], b0n, b1n);
        mma16(o[m][n + 1], pa[m][0], pa[m][1], pa[m][2], pa[m][3], b0m, b1m);
      }
    }
  }
  // partial results: O[row][d] (d = 8 * (2t + e) + n), m, l per row
#pragma unroll
  for (int m = 0; m < MT; m++)
#pragma unroll
    for (int hh = 0; hh < 2; hh++) {
      float l = ls[m][hh];
      l += __shfl_xor_sync(0xffffffffu, l, 1);
      l += __shfl_xor_sync(0xffffffffu, l, 2);
      const int r = m * 16 + g + hh * 8;
      if (r >= a.rows) continue;
#pragma unroll
      for (int n = 0; n < 8; n++) {
        wo[r * D + 8 * (2 * t) + n] = o[m][n][hh * 2];
        wo[r * D + 8 * (2 * t + 1) + n] = o[m][n][hh * 2 + 1];
      }
      if (t == 0) { wml[r * 2] = mx[m][hh]; wml[r * 2 + 1] = l; }
    }
}


// ---------------------------------------------------------------- decode (L = 1): one warp per (request, split), both kv heads.
// Full 128-byte K and V rows are gathered by cp.async into a swizzled per-warp ring (16B chunk c of row r at c ^ (r & 7)).
#ifndef RB2
#define RB2 6
#endif
__global__ void __launch_bounds__(256) attn2(A a) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int item = blockIdx.x * 8 + (threadIdx.x >> 5);
  if (item >= a.B * a.nsplit) return;
  const int sp = item % a.nsplit, b = item / a.nsplit;
  const int seq = a.seq[b], nk = seq + 1;
  const int k0 = a.window ? max(0, seq - a.window + 1) : 0, span = nk - k0;
  const int ks = k0 + (int)((long long)span * sp / a.nsplit), ke = k0 + (int)((long long)span * (sp + 1) / a.nsplit);
  uint32_t qa[2][4][4];
#pragma unroll
  for (int h = 0; h < 2; h++) {
    const __half* qp = a.q + ((size_t)b * a.hq + h * 8 + g) * D + t * 16;
    const uint4 v0 = *(const uint4*)qp, v1 = *(const uint4*)(qp + 8);
    const uint32_t w[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma unroll
    for (int j = 0; j < 4; j++) { qa[h][j][0] = w[2 * j]; qa[h][j][2] = w[2 * j + 1]; qa[h][j][1] = 0; qa[h][j][3] = 0; }
  }
  float o[2][8][2], mx[2], ls[2];
#pragma unroll
  for (int h = 0; h < 2; h++) { mx[h] = -INFINITY; ls[h] = 0.f;
#pragma unroll
    for (int n = 0; n < 8; n++) o[h][n][0] = o[h][n][1] = 0.f; }
  const int* ptb = a.pt + (size_t)b * a.maxp;
  extern __shared__ __align__(16) uint8_t smem[];
  const uint32_t ring = (uint32_t)__cvta_generic_to_shared(smem) + (threadIdx.x >> 5) * RB2 * 4096;
  auto rowsfor = [&](int kb) -> int { const int kp = kb + (lane & 15); return (kb < ke && kp < ke) ? ptb[kp / a.ps] * a.ps + kp % a.ps : -1; };
  const int c = lane & 7;
  auto issue = [&](int slot, int rl) {
    const uint32_t d0 = ring + slot * 4096;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const int r = i * 4 + (lane >> 3), row = __shfl_sync(0xffffffffu, rl, r);
      const uint32_t dd = d0 + r * 128 + ((c ^ (r & 7)) << 4);
      const size_t off = (size_t)max(row, 0) * 128 + c * 16;
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dd), "l"(a.kc + off), "r"(row >= 0 ? 16 : 0));
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dd + 2048), "l"(a.vc + off), "r"(row >= 0 ? 16 : 0));
    }
    asm volatile("cp.async.commit_group;\n");
  };
  const int nb = (ke - ks + 15) >> 4;
  int rnext = rowsfor(ks + (RB2 - 1) * 16);
#pragma unroll
  for (int i = 0; i < RB2 - 1; i++) issue(i, rowsfor(ks + i * 16));
  for (int bi = 0, kb = ks; bi < nb; bi++, kb += 16) {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(RB2 - 2));
    const int r2 = rowsfor(kb + RB2 * 16);
    issue((bi + RB2 - 1) % RB2, rnext);
    rnext = r2;
    const uint32_t sl = ring + (bi % RB2) * 4096;
#pragma unroll
    for (int h = 0; h < 2; h++) {
      uint4 kw[2];
      const uint32_t ck = ((h * 4 + t) ^ (g & 7)) << 4;
#pragma unroll
      for (int n = 0; n < 2; n++)
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(kw[n].x), "=r"(kw[n].y), "=r"(kw[n].z), "=r"(kw[n].w) : "r"(sl + (g + 8 * n) * 128 + ck));
      uint2 vw[4];
      const int rv[4] = {2 * t, 2 * t + 1, 2 * t + 8, 2 * t + 9};
#pragma unroll
      for (int i = 0; i < 4; i++)
        asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];\n" : "=r"(vw[i].x), "=r"(vw[i].y)
                     : "r"(sl + 2048 + rv[i] * 128 + (((h * 4 + (g >> 1)) ^ (rv[i] & 7)) << 4) + (g & 1) * 8));
      float s[2][4];
#pragma unroll
      for (int n = 0; n < 2; n++) {
        s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f;
        const uint32_t kk[4] = {kw[n].x, kw[n].y, kw[n].z, kw[n].w};
#pragma unroll
        for (int j = 0; j < 4; j++) mma16(s[n], qa[h][j][0], qa[h][j][1], qa[h][j][2], qa[h][j][3], e2h(kk[j]), e2h(kk[j] >> 16));
      }
      // row g only (rows g+8 are padding); keys kb + n*8 + 2t + e valid if < ke (window lower edge is ks's k0)
      float v[4];
#pragma unroll
      for (int n = 0; n < 2; n++)
#pragma unroll
        for (int e = 0; e < 2; e++) { const int kp = kb + n * 8 + 2 * t + e; v[n * 2 + e] = kp < ke ? s[n][e] * a.qk : -INFINITY; }
      float bm = fmaxf(fmaxf(v[0], v[1]), fmaxf(v[2], v[3]));
      bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, 1));
      bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, 2));
      const float nm = fmaxf(mx[h], bm), sc = ex2(mx[h] - nm);  // nm finite: every block has a valid key
      mx[h] = nm;
      const float p0 = ex2(v[0] - nm), p1 = ex2(v[1] - nm), p2 = ex2(v[2] - nm), p3 = ex2(v[3] - nm);
      ls[h] = ls[h] * sc + (p0 + p1 + p2 + p3);
      const uint32_t pa0 = pk(p0, p1), pa2 = pk(p2, p3);
#pragma unroll
      for (int n = 0; n < 8; n += 2) {
        const int wi = n >> 2, sh = (n & 3) * 8;
        const uint32_t x0 = wi ? vw[0].y : vw[0].x, x1 = wi ? vw[1].y : vw[1].x, x2 = wi ? vw[2].y : vw[2].x, x3 = wi ? vw[3].y : vw[3].x;
        const uint32_t lo01 = __byte_perm(x0 >> sh, x1 >> sh, 0x5140), lo23 = __byte_perm(x2 >> sh, x3 >> sh, 0x5140);
        float c0[4] = {o[h][n][0] * sc, o[h][n][1] * sc, 0.f, 0.f}, c1[4] = {o[h][n + 1][0] * sc, o[h][n + 1][1] * sc, 0.f, 0.f};
        mma16(c0, pa0, 0u, pa2, 0u, e2h(lo01), e2h(lo23));
        mma16(c1, pa0, 0u, pa2, 0u, e2h(lo01 >> 16), e2h(lo23 >> 16));
        o[h][n][0] = c0[0]; o[h][n][1] = c0[1]; o[h][n + 1][0] = c1[0]; o[h][n + 1][1] = c1[1];
      }
    }
  }
#pragma unroll
  for (int h = 0; h < 2; h++) {
    float l = ls[h];
    l += __shfl_xor_sync(0xffffffffu, l, 1);
    l += __shfl_xor_sync(0xffffffffu, l, 2);
    const size_t it = ((size_t)b * a.hkv + h) * a.nsplit + sp;
    float* wo = a.wo + it * a.rows * D + g * D;
#pragma unroll
    for (int n = 0; n < 8; n++) { wo[8 * (2 * t) + n] = o[h][n][0]; wo[8 * (2 * t + 1) + n] = o[h][n][1]; }
    if (t == 0) { a.wml[(it * a.rows + g) * 2] = nb ? mx[h] : -INFINITY; a.wml[(it * a.rows + g) * 2 + 1] = l; }
  }
}


// ---------------------------------------------------------------- verify (tp8, L = 8): one CTA (8 warps) per (request, split).
// The CTA gathers each 16-key block's full K/V rows once into a shared swizzled ring; warp w computes kv head w >> 2 for
// tokens 2(w & 3), 2(w & 3) + 1 (16 rows: 8 query heads x 2 tokens).
#ifndef RB3
#define RB3 20
#endif
#ifndef SB
#define SB 4
#endif
__global__ void __launch_bounds__(256, 2) attn3(A a) {
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, t = lane & 3;
  const int sp = blockIdx.x % a.nsplit, b = blockIdx.x / a.nsplit;
  const int h = w >> 2, m = w & 3;  // kv head, token pair
  const int seq = a.seq[b], nk = seq + a.L;
  const int k0 = a.window ? max(0, seq - a.window + 1) : 0, span = nk - k0;
  const int ks = k0 + (int)((long long)span * sp / a.nsplit), ke = k0 + (int)((long long)span * (sp + 1) / a.nsplit);
  uint32_t qa[4][4];
#pragma unroll
  for (int hh = 0; hh < 2; hh++) {
    const int l = 2 * m + hh;
    const __half* qp = a.q + (((size_t)b * a.L + l) * a.hq + h * 8 + g) * D + t * 16;
    const uint4 v0 = *(const uint4*)qp, v1 = *(const uint4*)(qp + 8);
    const uint32_t ww[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma unroll
    for (int j = 0; j < 4; j++) { qa[j][hh] = ww[2 * j]; qa[j][2 + hh] = ww[2 * j + 1]; }
  }
  float o[8][4], mx[2] = {-INFINITY, -INFINITY}, ls[2] = {0.f, 0.f};
#pragma unroll
  for (int n = 0; n < 8; n++) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
  const int* ptb = a.pt + (size_t)b * a.maxp;
  extern __shared__ __align__(16) uint8_t smem[];
  const uint32_t ring = (uint32_t)__cvta_generic_to_shared(smem);
  // thread tid copies 16B: rows r = tid >> 3 (0..31: K rows 0..15, V rows 16..31), chunk c = tid & 7
  const int r = tid >> 3, c = tid & 7, kr = r & 15;
  auto rowof = [&](int kb) -> int { const int kp = kb + kr; return (kb < ke && kp < ke) ? ptb[kp / a.ps] * a.ps + kp % a.ps : -1; };
  auto issue = [&](int slot, int row) {
    const uint32_t dd = ring + slot * 4096 + (r >= 16 ? 2048 : 0) + kr * 128 + ((c ^ (kr & 7)) << 4);
    const uint8_t* src = (r >= 16 ? a.vc : a.kc) + (size_t)max(row, 0) * 128 + c * 16;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dd), "l"(src), "r"(row >= 0 ? 16 : 0));
    asm volatile("cp.async.commit_group;\n");
  };
  // steps of SB blocks per barrier; at a step's start blocks < bi are consumed and nxt = bi + RB3 - SB are issued
  const int nb = (ke - ks + 15) >> 4;
  int nxt = 0, rn[SB];
#pragma unroll
  for (int i = 0; i < RB3 - SB; i++) { issue(i, rowof(ks + i * 16)); nxt++; }
#pragma unroll
  for (int j = 0; j < SB; j++) rn[j] = rowof(ks + (nxt + j) * 16);
  for (int bs = 0; bs < nb; bs += SB) {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(RB3 - 2 * SB));
    __syncthreads();  // blocks bs .. bs+SB-1 visible; slots of blocks bs-SB .. bs-1 free
#pragma unroll
    for (int j = 0; j < SB; j++) { issue((nxt + j) % RB3, rn[j]); }
    nxt += SB;
#pragma unroll
    for (int j = 0; j < SB; j++) rn[j] = rowof(ks + (nxt + j) * 16);
  for (int bi = bs, kb = ks + bs * 16; bi < min(nb, bs + SB); bi++, kb += 16) {
    const uint32_t sl = ring + (bi % RB3) * 4096;
    uint4 kw[2];
    const uint32_t ck = ((h * 4 + t) ^ (g & 7)) << 4;
#pragma unroll
    for (int n = 0; n < 2; n++)
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(kw[n].x), "=r"(kw[n].y), "=r"(kw[n].z), "=r"(kw[n].w) : "r"(sl + (g + 8 * n) * 128 + ck));
    uint2 vw[4];
    const int rv[4] = {2 * t, 2 * t + 1, 2 * t + 8, 2 * t + 9};
#pragma unroll
    for (int i = 0; i < 4; i++)
      asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];\n" : "=r"(vw[i].x), "=r"(vw[i].y)
                   : "r"(sl + 2048 + rv[i] * 128 + (((h * 4 + (g >> 1)) ^ (rv[i] & 7)) << 4) + (g & 1) * 8));
    float s[2][4];
#pragma unroll
    for (int n = 0; n < 2; n++) {
      s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f;
      const uint32_t kk[4] = {kw[n].x, kw[n].y, kw[n].z, kw[n].w};
#pragma unroll
      for (int j = 0; j < 4; j++) mma16(s[n], qa[j][0], qa[j][1], qa[j][2], qa[j][3], e2h(kk[j]), e2h(kk[j] >> 16));
    }
    uint32_t pa[4];
#pragma unroll
    for (int hh = 0; hh < 2; hh++) {
      const int p = seq + 2 * m + hh, lo = a.window ? max(0, p - a.window + 1) : 0;
      float v[4];
#pragma unroll
      for (int n = 0; n < 2; n++)
#pragma unroll
        for (int e = 0; e < 2; e++) {
          const int kp = kb + n * 8 + 2 * t + e;
          v[n * 2 + e] = (kp < ke && kp <= p && kp >= lo) ? s[n][hh * 2 + e] * a.qk : -INFINITY;
        }
      float bm = fmaxf(fmaxf(v[0], v[1]), fmaxf(v[2], v[3]));
      bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, 1));
      bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, 2));
      const float nm = fmaxf(mx[hh], bm);
      const float sc = nm == -INFINITY ? 1.f : ex2(mx[hh] - nm);
      mx[hh] = nm;
      float pp[4];
#pragma unroll
      for (int i = 0; i < 4; i++) pp[i] = nm == -INFINITY ? 0.f : ex2(v[i] - nm);
      ls[hh] = ls[hh] * sc + (pp[0] + pp[1] + pp[2] + pp[3]);
#pragma unroll
      for (int n = 0; n < 8; n++) { o[n][hh * 2] *= sc; o[n][hh * 2 + 1] *= sc; }
      pa[hh] = pk(pp[0], pp[1]); pa[2 + hh] = pk(pp[2], pp[3]);
    }
#pragma unroll
    for (int n = 0; n < 8; n += 2) {
      const int wi = n >> 2, sh = (n & 3) * 8;
      const uint32_t x0 = wi ? vw[0].y : vw[0].x, x1 = wi ? vw[1].y : vw[1].x, x2 = wi ? vw[2].y : vw[2].x, x3 = wi ? vw[3].y : vw[3].x;
      const uint32_t lo01 = __byte_perm(x0 >> sh, x1 >> sh, 0x5140), lo23 = __byte_perm(x2 >> sh, x3 >> sh, 0x5140);
      mma16(o[n], pa[0], pa[1], pa[2], pa[3], e2h(lo01), e2h(lo23));
      mma16(o[n + 1], pa[0], pa[1], pa[2], pa[3], e2h(lo01 >> 16), e2h(lo23 >> 16));
    }
  }
  }
  const size_t it = ((size_t)b * a.hkv + h) * a.nsplit + sp;
#pragma unroll
  for (int hh = 0; hh < 2; hh++) {
    float l = ls[hh];
    l += __shfl_xor_sync(0xffffffffu, l, 1);
    l += __shfl_xor_sync(0xffffffffu, l, 2);
    const int row = (2 * m + hh) * 8 + g;  // r = l * grp + head
    float* wo = a.wo + (it * a.rows + row) * D;
#pragma unroll
    for (int n = 0; n < 8; n++) { wo[8 * (2 * t) + n] = o[n][hh * 2]; wo[8 * (2 * t + 1) + n] = o[n][hh * 2 + 1]; }
    if (t == 0) { a.wml[(it * a.rows + row) * 2] = mx[hh]; a.wml[(it * a.rows + row) * 2 + 1] = l; }
  }
}

// ---------------------------------------------------------------- verify on FP8 tensor cores: E4M3 query, cache and P
// CTA = 4 warps (kv head h = w >> 1, token quad m = w & 1: two m-tiles of 2 tokens x 8 heads), steps of 32 keys.
// QK^T: d permuted so a lane's 16 B of a K row are its two k32 B fragments; PV: keys permuted to match the S C-fragment,
// V transposed in registers (byte_perm), output columns d = 8 * col + n-tile (same order as attn3).
__device__ __forceinline__ void mma8(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
#if __CUDA_ARCH__ >= 890
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
  mma16(c, e2h(a0), e2h(a1), e2h(a0 >> 16), e2h(a1 >> 16), e2h(b0), e2h(b0 >> 16));
  mma16(c, e2h(a2), e2h(a3), e2h(a2 >> 16), e2h(a3 >> 16), e2h(b1), e2h(b1 >> 16));
#endif
}
__device__ __forceinline__ uint32_t f2e(float lo, float hi) {
  return (uint32_t)__nv_cvt_float2_to_fp8x2(make_float2(lo, hi), __NV_SATFINITE, __NV_E4M3);
}
#ifndef RB8
#define RB8 6
#endif
__global__ void __launch_bounds__(128, 3) attn8(A a) {
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, t = lane & 3;
  const int sp = blockIdx.x % a.nsplit, b = blockIdx.x / a.nsplit;
  const int h = w >> 1, m = w & 1;
  const int seq = a.seq[b], nk = seq + a.L;
  const int k0 = a.window ? max(0, seq - a.window + 1) : 0, span = nk - k0;
  const int ks = k0 + (int)((long long)span * sp / a.nsplit), ke = k0 + (int)((long long)span * (sp + 1) / a.nsplit);
  uint32_t qa[2][2][4];  // [m-tile][k-step][frag]
#pragma unroll
  for (int mt = 0; mt < 2; mt++)
#pragma unroll
    for (int hh = 0; hh < 2; hh++) {
      const int l = 4 * m + 2 * mt + hh;
      const uint4 v = *(const uint4*)(a.q8 + (((size_t)b * a.L + l) * a.hq + h * 8 + g) * D + t * 16);
      qa[mt][0][hh] = v.x; qa[mt][0][2 + hh] = v.y; qa[mt][1][hh] = v.z; qa[mt][1][2 + hh] = v.w;
    }
  float o[2][8][4], mx[2][2], ls[2][2];
#pragma unroll
  for (int mt = 0; mt < 2; mt++) {
    mx[mt][0] = mx[mt][1] = -INFINITY; ls[mt][0] = ls[mt][1] = 0.f;
#pragma unroll
    for (int n = 0; n < 8; n++) o[mt][n][0] = o[mt][n][1] = o[mt][n][2] = o[mt][n][3] = 0.f;
  }
  const int* ptb = a.pt + (size_t)b * a.maxp;
  extern __shared__ __align__(16) uint8_t smem[];
  const uint32_t ring = (uint32_t)__cvta_generic_to_shared(smem);
  const int lr = tid >> 4, part = (tid >> 3) & 1, c = tid & 7;  // copy: key i*8 + lr, K or V, 16 B chunk c
  auto rowof = [&](int kb, int i) -> int { const int kp = kb + i * 8 + lr; return kp < ke ? ptb[kp / a.ps] * a.ps + kp % a.ps : -1; };
  auto issue = [&](int slot, const int* rw) {
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const int key = i * 8 + lr;
      const uint32_t dd = ring + slot * 8192 + part * 4096 + key * 128 + ((c ^ (key & 7)) << 4);
      const uint8_t* src = (part ? a.vc : a.kc) + (size_t)max(rw[i], 0) * 128 + c * 16;
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dd), "l"(src), "r"(rw[i] >= 0 ? 16 : 0));
    }
    asm volatile("cp.async.commit_group;\n");
  };
  const int nb = (ke - ks + 31) >> 5;
  int rn[4];
  for (int j = 0; j < RB8 - 1; j++) {
#pragma unroll
    for (int i = 0; i < 4; i++) rn[i] = rowof(ks + j * 32, i);
    issue(j, rn);
  }
#pragma unroll
  for (int i = 0; i < 4; i++) rn[i] = rowof(ks + (RB8 - 1) * 32, i);
  const int pmin = seq + 4 * m, lomax = a.window ? max(0, seq + 4 * m + 3 - a.window + 1) : 0;
  for (int bi = 0, kb = ks; bi < nb; bi++, kb += 32) {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(RB8 - 2));
    __syncthreads();
    issue((bi + RB8 - 1) % RB8, rn);
#pragma unroll
    for (int i = 0; i < 4; i++) rn[i] = rowof(kb + RB8 * 32, i);
    const uint32_t sl = ring + (bi % RB8) * 8192;
    float s[2][4][4];
#pragma unroll
    for (int mt = 0; mt < 2; mt++)
#pragma unroll
      for (int n = 0; n < 4; n++) s[mt][n][0] = s[mt][n][1] = s[mt][n][2] = s[mt][n][3] = 0.f;
#pragma unroll
    for (int n = 0; n < 4; n++) {
      uint4 kw;
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(kw.x), "=r"(kw.y), "=r"(kw.z), "=r"(kw.w)
                   : "r"(sl + (n * 8 + g) * 128 + (((h * 4 + t) ^ g) << 4)));
#pragma unroll
      for (int mt = 0; mt < 2; mt++) {
        mma8(s[mt][n], qa[mt][0][0], qa[mt][0][1], qa[mt][0][2], qa[mt][0][3], kw.x, kw.y);
        mma8(s[mt][n], qa[mt][1][0], qa[mt][1][1], qa[mt][1][2], qa[mt][1][3], kw.z, kw.w);
      }
    }
    // V: lane rows keys {2t, 2t+1, 8+2t, 9+2t} (+16), 8 B at d 8g..8g+7 of head h
    uint32_t vb[2][8];
#pragma unroll
    for (int st = 0; st < 2; st++) {
      uint2 rv[4];
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const int key = st * 16 + (j >> 1) * 8 + 2 * t + (j & 1);
        asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];\n" : "=r"(rv[j].x), "=r"(rv[j].y)
                     : "r"(sl + 4096 + key * 128 + (((h * 4 + (g >> 1)) ^ (key & 7)) << 4) + (g & 1) * 8));
      }
#pragma unroll
      for (int hf = 0; hf < 2; hf++) {
        const uint32_t r0 = hf ? rv[0].y : rv[0].x, r1 = hf ? rv[1].y : rv[1].x, r2 = hf ? rv[2].y : rv[2].x, r3 = hf ? rv[3].y : rv[3].x;
        const uint32_t t0 = __byte_perm(r0, r1, 0x5140), t1 = __byte_perm(r2, r3, 0x5140);
        const uint32_t t2 = __byte_perm(r0, r1, 0x7362), t3 = __byte_perm(r2, r3, 0x7362);
        vb[st][hf * 4 + 0] = __byte_perm(t0, t1, 0x5410); vb[st][hf * 4 + 1] = __byte_perm(t0, t1, 0x7632);
        vb[st][hf * 4 + 2] = __byte_perm(t2, t3, 0x5410); vb[st][hf * 4 + 3] = __byte_perm(t2, t3, 0x7632);
      }
    }
    const bool full = kb + 32 <= ke && kb + 32 <= pmin + 1 && kb >= lomax;
#pragma unroll
    for (int mt = 0; mt < 2; mt++) {
      uint32_t pa[4];
      bool grow = false;
      float scl[2];
#pragma unroll
      for (int hh = 0; hh < 2; hh++) {
        float v[8];
#pragma unroll
        for (int n = 0; n < 4; n++)
#pragma unroll
          for (int e = 0; e < 2; e++) v[n * 2 + e] = s[mt][n][hh * 2 + e] * a.qk;
        if (!full) {
          const int p = seq + 4 * m + 2 * mt + hh, lo = a.window ? max(0, p - a.window + 1) : 0;
#pragma unroll
          for (int n = 0; n < 4; n++)
#pragma unroll
            for (int e = 0; e < 2; e++) { const int kp = kb + n * 8 + 2 * t + e; if (!(kp < ke && kp <= p && kp >= lo)) v[n * 2 + e] = -INFINITY; }
        }
        float bm = fmaxf(fmaxf(fmaxf(v[0], v[1]), fmaxf(v[2], v[3])), fmaxf(fmaxf(v[4], v[5]), fmaxf(v[6], v[7])));
        bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, 1));
        bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, 2));
        const float nm = fmaxf(mx[mt][hh], bm);
        grow |= nm > mx[mt][hh];
        scl[hh] = nm == -INFINITY ? 1.f : ex2(mx[mt][hh] - nm);
        mx[mt][hh] = nm;
        const float off = nm == -INFINITY ? 0.f : 8.f - nm;  // P scaled by 2^8 into E4M3's normal range
        float pp[8];
#pragma unroll
        for (int i = 0; i < 8; i++) pp[i] = ex2(v[i] + off);
        ls[mt][hh] = ls[mt][hh] * scl[hh] + (((pp[0] + pp[1]) + (pp[2] + pp[3])) + ((pp[4] + pp[5]) + (pp[6] + pp[7])));
        pa[hh] = f2e(pp[0], pp[1]) | (f2e(pp[2], pp[3]) << 16);
        pa[2 + hh] = f2e(pp[4], pp[5]) | (f2e(pp[6], pp[7]) << 16);
      }
      if (__any_sync(0xffffffffu, grow)) {
#pragma unroll
        for (int n = 0; n < 8; n++) { o[mt][n][0] *= scl[0]; o[mt][n][1] *= scl[0]; o[mt][n][2] *= scl[1]; o[mt][n][3] *= scl[1]; }
      }
#pragma unroll
      for (int n = 0; n < 8; n++) mma8(o[mt][n], pa[0], pa[1], pa[2], pa[3], vb[0][n], vb[1][n]);
    }
  }
  const size_t it = ((size_t)b * a.hkv + h) * a.nsplit + sp;
#pragma unroll
  for (int mt = 0; mt < 2; mt++)
#pragma unroll
    for (int hh = 0; hh < 2; hh++) {
      float l = ls[mt][hh];
      l += __shfl_xor_sync(0xffffffffu, l, 1);
      l += __shfl_xor_sync(0xffffffffu, l, 2);
      const int row = (4 * m + 2 * mt + hh) * 8 + g;
      float* wo = a.wo + (it * a.rows + row) * D;
#pragma unroll
      for (int n = 0; n < 8; n++) { wo[8 * (2 * t) + n] = o[mt][n][hh * 2]; wo[8 * (2 * t + 1) + n] = o[mt][n][hh * 2 + 1]; }
      if (t == 0) { a.wml[(it * a.rows + row) * 2] = mx[mt][hh]; a.wml[(it * a.rows + row) * 2 + 1] = l; }
    }
}

// e4m3 byte at bits 15..8 (and 31..24) -> f16 bits of value x 2^-8 (exact: e4m3 subnormals land on f16 subnormals)
__device__ __forceinline__ uint32_t e2h_alu(uint32_t t) { return ((t & 0x7F007F00u) >> 1) | (t & 0x80008000u); }
#ifndef KALU
#define KALU 0
#endif
#ifndef VALU
#define VALU 0
#endif
#ifndef MINB
#define MINB 3
#endif
#ifndef NOGROW
#define NOGROW 1
#endif
template <bool PS1>
__global__ void __launch_bounds__(128, MINB) attn9(A a) {
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, t = lane & 3;
  const int nqb = (a.L + 7) >> 3, sp = blockIdx.x % a.nsplit, bq = blockIdx.x / a.nsplit, qb = bq % nqb, b = bq / nqb;
  const int t0 = qb * 8, lt = min(8, a.L - t0);  // this block's tokens t0 .. t0 + lt - 1
  const int h = w >> 1, m = w & 1;
  const int seq = a.seq[b], nk = seq + t0 + lt;
  const int k0 = a.window ? max(0, seq + t0 - a.window + 1) : 0, span = nk - k0;
  const int ks = k0 + (int)((long long)span * sp / a.nsplit), ke = k0 + (int)((long long)span * (sp + 1) / a.nsplit);
  uint32_t qa[2][4][4];  // [m-tile][k-step][frag]: f16, lane (g, t) holds d 16t .. 16t+15
#pragma unroll
  for (int mt = 0; mt < 2; mt++)
#pragma unroll
    for (int hh = 0; hh < 2; hh++) {
      const int l = 4 * m + 2 * mt + hh;
      const __half* qp = a.q + (((size_t)b * a.L + t0 + min(l, lt - 1)) * a.hq + h * 8 + g) * D + t * 16;
      const uint4 v0 = *(const uint4*)qp, v1 = *(const uint4*)(qp + 8);
      const uint32_t ww[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma unroll
      for (int j = 0; j < 4; j++) { qa[mt][j][hh] = ww[2 * j]; qa[mt][j][2 + hh] = ww[2 * j + 1]; }
    }
  float o[2][8][4], mx[2][2], ls[2][2];
#pragma unroll
  for (int mt = 0; mt < 2; mt++) {
    mx[mt][0] = mx[mt][1] = -INFINITY; ls[mt][0] = ls[mt][1] = 0.f;
#pragma unroll
    for (int n = 0; n < 8; n++) o[mt][n][0] = o[mt][n][1] = o[mt][n][2] = o[mt][n][3] = 0.f;
  }
  const int* ptb = a.pt + (size_t)b * a.maxp;
  extern __shared__ __align__(16) uint8_t smem[];
  const uint32_t ring = (uint32_t)__cvta_generic_to_shared(smem);
  const int lr = tid >> 4, part = (tid >> 3) & 1, c = tid & 7;  // copy: key i*8 + lr, K or V, 16 B chunk c
  auto rowof = [&](int kb, int i) -> int {  // branch-free: clamped load, then select
    const int kp = kb + i * 8 + lr, kc = min(kp, ke - 1);
    const int rr = PS1 ? __ldg(ptb + kc) : __ldg(ptb + kc / a.ps) * a.ps + kc % a.ps;
    return kp < ke ? rr : -1;
  };
  auto issue = [&](int slot, const int* rw) {
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const int key = i * 8 + lr;
      const uint32_t dd = ring + slot * 8192 + part * 4096 + key * 128 + ((c ^ (key & 7)) << 4);
      const uint8_t* src = (part ? a.vc : a.kc) + (size_t)max(rw[i], 0) * 128 + c * 16;
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dd), "l"(src), "r"(rw[i] >= 0 ? 16 : 0));
    }
    asm volatile("cp.async.commit_group;\n");
  };
  const int nb = (ke - ks + 31) >> 5;
  int rn[4];
  for (int j = 0; j < RB8 - 1; j++) {
#pragma unroll
    for (int i = 0; i < 4; i++) rn[i] = rowof(ks + j * 32, i);
    issue(j, rn);
  }
#pragma unroll
  for (int i = 0; i < 4; i++) rn[i] = rowof(ks + (RB8 - 1) * 32, i);
  const int pmin = lt == 8 ? seq + t0 + 4 * m : -1, lomax = a.window ? max(0, seq + t0 + 4 * m + 3 - a.window + 1) : 0;
  for (int bi = 0, kb = ks; bi < nb; bi++, kb += 32) {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(RB8 - 2));
    __syncthreads();
    issue((bi + RB8 - 1) % RB8, rn);
#pragma unroll
    for (int i = 0; i < 4; i++) rn[i] = rowof(kb + RB8 * 32, i);
    const uint32_t sl = ring + (bi % RB8) * 8192;
    float s[2][4][4];
#pragma unroll
    for (int mt = 0; mt < 2; mt++)
#pragma unroll
      for (int n = 0; n < 4; n++) s[mt][n][0] = s[mt][n][1] = s[mt][n][2] = s[mt][n][3] = 0.f;
#pragma unroll
    for (int n = 0; n < 4; n++) {
      uint4 kw;
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(kw.x), "=r"(kw.y), "=r"(kw.z), "=r"(kw.w)
                   : "r"(sl + (n * 8 + g) * 128 + (((h * 4 + t) ^ g) << 4)));
      const uint32_t kk[4] = {kw.x, kw.y, kw.z, kw.w};
#pragma unroll
      for (int j = 0; j < 4; j++) {
#if KALU
        const uint32_t b0 = e2h_alu(__byte_perm(kk[j], 0, 0x1404)), b1 = e2h_alu(__byte_perm(kk[j], 0, 0x3424));
#else
        const uint32_t b0 = e2h(kk[j]), b1 = e2h(kk[j] >> 16);
#endif
#pragma unroll
        for (int mt = 0; mt < 2; mt++) mma16(s[mt][n], qa[mt][j][0], qa[mt][j][1], qa[mt][j][2], qa[mt][j][3], b0, b1);
      }
    }
    uint2 vw[2][4];  // [16-key block][row 2t, 2t+1, 2t+8, 2t+9]: 8 B at d 8g .. 8g+7 of head h
#pragma unroll
    for (int st = 0; st < 2; st++)
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const int key = st * 16 + (j >> 1) * 8 + 2 * t + (j & 1);
        asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];\n" : "=r"(vw[st][j].x), "=r"(vw[st][j].y)
                     : "r"(sl + 4096 + key * 128 + (((h * 4 + (g >> 1)) ^ (key & 7)) << 4) + (g & 1) * 8));
      }
    const bool full = kb + 32 <= ke && kb + 32 <= pmin + 1 && kb >= lomax;
#pragma unroll
    for (int mt = 0; mt < 2; mt++) {
      uint32_t pa[2][4];  // [16-key block]
      bool grow = false;
      float scl[2];
#pragma unroll
      for (int hh = 0; hh < 2; hh++) {
        float v[8];
#pragma unroll
        for (int n = 0; n < 4; n++)
#pragma unroll
          for (int e = 0; e < 2; e++) v[n * 2 + e] = s[mt][n][hh * 2 + e];
        if (!full) {
          const int p = 4 * m + 2 * mt + hh < lt ? seq + t0 + 4 * m + 2 * mt + hh : -1, lo = a.window ? max(0, p - a.window + 1) : 0;
#pragma unroll
          for (int n = 0; n < 4; n++)
#pragma unroll
            for (int e = 0; e < 2; e++) { const int kp = kb + n * 8 + 2 * t + e; if (!(kp < ke && kp <= p && kp >= lo)) v[n * 2 + e] = -INFINITY; }
        }
        float bm = fmaxf(fmaxf(fmaxf(v[0], v[1]), fmaxf(v[2], v[3])), fmaxf(fmaxf(v[4], v[5]), fmaxf(v[6], v[7])));
        bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, 1));
        bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, 2));
        const float nm = fmaxf(mx[mt][hh], bm * a.qk);  // log2 units (qk > 0)
        grow |= nm > mx[mt][hh];
        const float nmf = nm == -INFINITY ? 0.f : nm;   // all masked so far: weights 0, no NaN
        scl[hh] = ex2(mx[mt][hh] - nmf);
        mx[mt][hh] = nm;
        const float off = (VALU ? 8.f : 0.f) - nmf;  // V as f16 x 2^-8: P x 2^8
        float pp[8];
#pragma unroll
        for (int i = 0; i < 8; i++) pp[i] = ex2(fmaf(v[i], a.qk, off));
        ls[mt][hh] = ls[mt][hh] * scl[hh] + (((pp[0] + pp[1]) + (pp[2] + pp[3])) + ((pp[4] + pp[5]) + (pp[6] + pp[7]))) * (VALU ? 0.00390625f : 1.f);
        pa[0][hh] = pk(pp[0], pp[1]); pa[0][2 + hh] = pk(pp[2], pp[3]);
        pa[1][hh] = pk(pp[4], pp[5]); pa[1][2 + hh] = pk(pp[6], pp[7]);
      }
      if (NOGROW || __any_sync(0xffffffffu, grow)) {
#pragma unroll
        for (int n = 0; n < 8; n++) { o[mt][n][0] *= scl[0]; o[mt][n][1] *= scl[0]; o[mt][n][2] *= scl[1]; o[mt][n][3] *= scl[1]; }
      }
#pragma unroll
      for (int st = 0; st < 2; st++)
#pragma unroll
        for (int n = 0; n < 8; n += 2) {
          const int wi = n >> 2, sh = (n & 3) * 8;
          const uint32_t x0 = wi ? vw[st][0].y : vw[st][0].x, x1 = wi ? vw[st][1].y : vw[st][1].x;
          const uint32_t x2 = wi ? vw[st][2].y : vw[st][2].x, x3 = wi ? vw[st][3].y : vw[st][3].x;
#if VALU
          // bytes n, n+1 of rows (2t, 2t+1) -> f16x2 with the byte at bits 15..8 / 31..24
          const uint32_t b00 = e2h_alu(__byte_perm(x0 >> sh, x1 >> sh, 0x4000)), b01 = e2h_alu(__byte_perm(x0 >> sh, x1 >> sh, 0x5010));
          const uint32_t b10 = e2h_alu(__byte_perm(x2 >> sh, x3 >> sh, 0x4000)), b11 = e2h_alu(__byte_perm(x2 >> sh, x3 >> sh, 0x5010));
#else
          const uint32_t lo01 = __byte_perm(x0 >> sh, x1 >> sh, 0x5140), lo23 = __byte_perm(x2 >> sh, x3 >> sh, 0x5140);
          const uint32_t b00 = e2h(lo01), b10 = e2h(lo23), b01 = e2h(lo01 >> 16), b11 = e2h(lo23 >> 16);
#endif
          mma16(o[mt][n], pa[st][0], pa[st][1], pa[st][2], pa[st][3], b00, b10);
          mma16(o[mt][n + 1], pa[st][0], pa[st][1], pa[st][2], pa[st][3], b01, b11);
        }
    }
  }
  const size_t it = ((size_t)b * a.hkv + h) * a.nsplit + sp;
#pragma unroll
  for (int mt = 0; mt < 2; mt++)
#pragma unroll
    for (int hh = 0; hh < 2; hh++) {
      float l = ls[mt][hh];
      l += __shfl_xor_sync(0xffffffffu, l, 1);
      l += __shfl_xor_sync(0xffffffffu, l, 2);
      if (4 * m + 2 * mt + hh >= lt) continue;
      const int row = (t0 + 4 * m + 2 * mt + hh) * 8 + g;
      float* wo = a.wo + (it * a.rows + row) * D;
#pragma unroll
      for (int n = 0; n < 8; n++) { wo[8 * (2 * t) + n] = o[mt][n][hh * 2]; wo[8 * (2 * t + 1) + n] = o[mt][n][hh * 2 + 1]; }
      if (t == 0) { a.wml[(it * a.rows + row) * 2] = mx[mt][hh]; a.wml[(it * a.rows + row) * 2 + 1] = l; }
    }
}

// ---------------------------------------------------------------- verify on Hopper warpgroup MMA (sm_90a)
// CTA = 2 warpgroups (one per kv head) x (request, split). Per 64-key step: the raw FP8 K/V rows land in a shared
// cp.async ring; each warpgroup converts its head's halves once to f16 tiles (128 B swizzle), then
// S = Q K^T (4 x m64n64k16, both operands in smem) and O += P V (P from registers, V MN-major in smem).
#ifndef WGV
#define WGV 1
#endif
#ifndef WVL
#define WVL 16
#endif
#ifndef WVS
#define WVS 1024
#endif
#ifndef RSW
#define RSW 4
#endif
#ifndef PFD
#define PFD 0
#endif
#ifndef VDB
#define VDB 1
#endif
#ifndef AW_FOP
#define AW_FOP 0
#endif
#ifndef AW_Z
#define AW_Z 1
#endif
#ifndef AW_LZ
#define AW_LZ 1
#endif
#ifndef LZT
#define LZT 8.f
#endif
#ifndef AW_H2
#define AW_H2 0
#endif
#ifndef FUSEV
#define FUSEV 1
#endif
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
#define HAS_WG 1
#else
#define HAS_WG 0
#endif
__device__ __forceinline__ uint64_t gdesc(uint32_t saddr, uint32_t lbo, uint32_t sbo) {
  return (uint64_t)((saddr & 0x3FFFF) >> 4) | ((uint64_t)(lbo >> 4) << 16) | ((uint64_t)(sbo >> 4) << 32) | (1ull << 62);
}
#define WG_S(d) "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), \
  "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
#define WG_D "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}"
template <int ACC>
__device__ __forceinline__ void wg_ss(float* d, uint64_t da, uint64_t db) {
#if HAS_WG
  asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 " WG_D ", %32, %33, %34, 1, 1, 0, 0;\n" : WG_S(d) : "l"(da), "l"(db), "n"(ACC));
#endif
}
// scale-d as a runtime predicate: the first PV of a split overwrites o, so o is never zero-filled by plain stores
// (a zero-fill reaching the loop header made ptxas serialize every wgmma: C7515)
__device__ __forceinline__ void wg_rsz(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db, int sd) {
#if HAS_WG
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\nwgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 " WG_D ", {%32,%33,%34,%35}, %36, p, 1, 1, 1;\n}\n"
               : WG_S(d) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db), "r"(sd));
#endif
}
__device__ __forceinline__ void wg_rs(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db) {
#if HAS_WG
  asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 " WG_D ", {%32,%33,%34,%35}, %36, 1, 1, 1, 1;\n"
               : WG_S(d) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));
#endif
}
__device__ __forceinline__ uint32_t lds32a(uint32_t a) { uint32_t v; asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(v) : "r"(a)); return v; }
__device__ __forceinline__ void wg_fence() {
#if HAS_WG
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void wg_commit_wait() {
#if HAS_WG
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void fence_async() {
#if __CUDA_ARCH__ >= 900
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#endif
}

#define WG_SO(d) "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7]), "=f"(d[8]), "=f"(d[9]), "=f"(d[10]), "=f"(d[11]), "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15]), \
  "=f"(d[16]), "=f"(d[17]), "=f"(d[18]), "=f"(d[19]), "=f"(d[20]), "=f"(d[21]), "=f"(d[22]), "=f"(d[23]), "=f"(d[24]), "=f"(d[25]), "=f"(d[26]), "=f"(d[27]), "=f"(d[28]), "=f"(d[29]), "=f"(d[30]), "=f"(d[31])
template <int ACC>
__device__ __forceinline__ void wg_ss8(float* d, uint64_t da, uint64_t db) {
#if HAS_WG
  if constexpr (ACC == 0)  // scale-d 0: write-only accumulators, so nothing defines them inside the pipeline stage
    asm volatile("wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 " WG_D ", %32, %33, 0, 1, 1;\n" : WG_SO(d) : "l"(da), "l"(db));
  else
    asm volatile("wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 " WG_D ", %32, %33, %34, 1, 1;\n" : WG_S(d) : "l"(da), "l"(db), "n"(ACC));
#endif
}
__device__ __forceinline__ void fop(float* d, int n) {  // pin accumulators: no reordering across the wgmma window
  for (int i = 0; i < n; i++) asm volatile("" : "+f"(d[i]) :: "memory");
}
__device__ __forceinline__ void wg_commit() {
#if HAS_WG
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void wg_wait0() {
#if HAS_WG
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
#endif
}

struct WF {  // fused verify: query/new-key prologue and the split merge + output gate in the same launch
  const __nv_bfloat16* qkv; const int64_t* loc; const float* invf; int* cnt; __nv_bfloat16* out;
  float theta, kinv, vinv, vscale;
};
template <bool PS1, bool FUSE, bool MRG>
__global__ void __launch_bounds__(256, 2) attnw(A a, WF f) {
  const int tid = threadIdx.x, wt = tid & 127, h = tid >> 7, w = wt >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  // block (request b, 8-token query block qb, key split sp): tokens t0 .. t0 + lt - 1
  const int nqb = (a.L + 7) >> 3, sp = blockIdx.x % a.nsplit, bq = blockIdx.x / a.nsplit, qb = bq % nqb, b = bq / nqb;
  const int t0 = qb * 8, lt = min(8, a.L - t0);
  const int seq = a.seq[b], nk = seq + t0 + lt;
  const int k0 = a.window ? max(0, seq + t0 - a.window + 1) : 0, span = nk - k0;
  const int ks = k0 + (int)((long long)span * sp / a.nsplit), ke = k0 + (int)((long long)span * (sp + 1) / a.nsplit);
  extern __shared__ __align__(1024) uint8_t smem[];
  const uint32_t sb = ((uint32_t)__cvta_generic_to_shared(smem) + 1023) & ~1023u;
  // Q (E4M3) as one 64-row x 128 B tile: warpgroup h's 64 B at byte 64 h, like the head halves of the raw K rows
  const uint32_t ring = sb, qs = sb + RSW * 16384, vf0 = qs + 8192 + h * (VDB ? 16384 : 8192);
  const uint32_t ptr = qs + 8192 + 2 * (VDB ? 16384 : 8192);  // page-table ring: 4 segments x 4 steps x 64 entries
  if constexpr (!FUSE) {
#pragma unroll
    for (int i = 0; i < 2; i++) {
      const int idx = i * 128 + wt, r = idx >> 2, c = 4 * h + (idx & 3);
      const uint4 v = (r >> 3) < lt ? *(const uint4*)(a.q8 + (((size_t)b * a.L + t0 + (r >> 3)) * a.hq + h * 8 + (r & 7)) * D + (idx & 3) * 16) : make_uint4(0, 0, 0, 0);
      asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" :: "r"(qs + r * 128 + ((c ^ (r & 7)) << 4)), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
    }
  } else {  // row r = wt >> 1 (token r >> 3, q head 8 h + (r & 7)), half wt & 1: RMSNorm, RoPE at seq + token, E4M3
    const int r = wt >> 1, hf = wt & 1, l = r >> 3;
    const __nv_bfloat16* src = f.qkv + ((size_t)b * a.L + l) * (2 * 16 + 4) * D + (h * 8 + (r & 7)) * D + hf * 32;
    float x[32], ss = 0.f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const uint4 u = *(const uint4*)(src + 8 * i);
      const __nv_bfloat162* bb = (const __nv_bfloat162*)&u;
#pragma unroll
      for (int j = 0; j < 4; j++) { const float2 v = __bfloat1622float2(bb[j]); x[8 * i + 2 * j] = v.x; x[8 * i + 2 * j + 1] = v.y; }
    }
#pragma unroll
    for (int i = 0; i < 32; i++) ss += x[i] * x[i];
    ss += __shfl_xor_sync(0xffffffffu, ss, 1);
    const float rn = rsqrtf(ss / D + 1e-5f);
#pragma unroll
    for (int i = 0; i < 32; i++) x[i] *= rn;
    if (f.theta > 0.f) {
#pragma unroll
      for (int i = 0; i < 32; i += 2) {
        float sn, cs2; rope_cs(f.invf, seq + l, (hf * 32 + i) >> 1, sn, cs2);
        const float e = x[i], o = x[i + 1];
        x[i] = e * cs2 - o * sn; x[i + 1] = e * sn + o * cs2;
      }
    }
    uint32_t q4[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
      uint32_t wv = 0;
#pragma unroll
      for (int j = 0; j < 4; j++) { __nv_fp8_e4m3 fq(x[4 * i + j]); wv |= (uint32_t)fq.__x << (8 * j); }
      q4[i] = wv;
    }
#pragma unroll
    for (int u = 0; u < 2; u++) {
      const int c = 4 * h + 2 * hf + u;
      asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" :: "r"(qs + r * 128 + ((c ^ (r & 7)) << 4)), "r"(q4[4 * u]), "r"(q4[4 * u + 1]), "r"(q4[4 * u + 2]), "r"(q4[4 * u + 3]));
    }
    if (ke == nk) {  // last split: the block's own keys and values, written to the cache before the ring reads them
      const int rr = tid >> 4, kh = rr & 1, lt = rr >> 1, d0 = (tid & 15) * 4;
      const __nv_bfloat16* krow = f.qkv + ((size_t)b * a.L + lt) * (2 * 16 + 4) * D;
      float k[4], v[4], s2 = 0.f;
#pragma unroll
      for (int i = 0; i < 4; i++) { k[i] = __bfloat162float(krow[(16 + kh) * D + d0 + i]); v[i] = __bfloat162float(krow[(18 + kh) * D + d0 + i]); s2 += k[i] * k[i]; }
#pragma unroll
      for (int o = 1; o < 16; o <<= 1) s2 += __shfl_xor_sync(0xffffffffu, s2, o);
      const float kr = rsqrtf(s2 / D + 1e-5f);
#pragma unroll
      for (int i = 0; i < 4; i++) k[i] *= kr;
      if (f.theta > 0.f) {
#pragma unroll
        for (int i = 0; i < 4; i += 2) {
          float sn, cs2; rope_cs(f.invf, seq + lt, (d0 + i) >> 1, sn, cs2);
          const float e = k[i], o = k[i + 1];
          k[i] = e * cs2 - o * sn; k[i + 1] = e * sn + o * cs2;
        }
      }
      uint32_t kb4 = 0, vb4 = 0;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        __nv_fp8_e4m3 fk(k[i] * f.kinv), fv(v[i] * f.vinv);
        kb4 |= (uint32_t)fk.__x << (8 * i); vb4 |= (uint32_t)fv.__x << (8 * i);
      }
      const size_t lc = (size_t)f.loc[b * a.L + lt] * 128 + kh * 64 + d0;
      *(uint32_t*)((uint8_t*)a.kc + lc) = kb4;
      *(uint32_t*)((uint8_t*)a.vc + lc) = vb4;
      __threadfence();
    }
    __syncthreads();
  }
  const int* ptb = a.pt + (size_t)b * a.maxp;
  // each warpgroup streams only its head's halves (K and V, 4 x 16 B per row): no CTA-wide barrier in the loop
  const int lr = wt >> 3, part = (wt >> 2) & 1, c8 = 4 * h + (wt & 3);  // copy: key i * 16 + lr, K or V, 16 B chunk
  auto rowof = [&](int kb, int i) -> int {
    const int kp = kb + i * 16 + lr, kc = min(kp, ke - 1);
    const int rr = PS1 ? __ldg(ptb + kc) : __ldg(ptb + kc / a.ps) * a.ps + kc % a.ps;
    return kp < ke ? rr : -1;
  };
  auto issue = [&](int slot, int kbi, int r0, int r1, int r2, int r3) {
    const int rw[4] = {r0, r1, r2, r3};
#ifdef AW_NOLOAD
    if (kbi >= ks + (RSW - 1) * 64 || kbi >= ke) { asm volatile("cp.async.commit_group;\n"); return; }  // ablation: only the prologue steps load
#endif
    if (kbi >= ke) { asm volatile("cp.async.commit_group;\n"); return; }  // step past the range
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const int key = i * 16 + lr;
      const uint32_t dd = ring + slot * 16384 + part * 8192 + key * 128 + ((c8 ^ (key & 7)) << 4);
      const uint8_t* src = (part ? a.vc : a.kc) + (size_t)max(rw[i], 0) * 128 + c8 * 16;
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dd), "l"(src), "r"(rw[i] >= 0 ? 16 : 0));
    }
    asm volatile("cp.async.commit_group;\n");
  };
  // page size 1: each warpgroup streams the entries of step x into slot x & 7 of its own ring (issued 7 steps ahead,
  // read 4 ahead), so lookups are shared-memory reads
  const uint32_t ptrh = ptr + h * 2048;
  auto ptseg = [&](int x) {
    const int idx = ks + x * 64 + wt;
    if (wt < 64) asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" :: "r"(ptrh + (x & 7) * 256 + wt * 4), "l"(ptb + (idx < ke ? idx : 0)), "r"(idx < ke ? 4 : 0));
  };
  auto rowr = [&](int kb, int i) -> int {
    if (!PS1) return rowof(kb, i);
    const int kp = kb + i * 16 + lr;
    if (kp >= ke) return -1;
    int v; asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(v) : "r"(ptrh + ((kp - ks) & 511) * 4));
    return v;
  };
  if (PS1) for (int x = RSW; x < RSW + 3; x++) ptseg(x);  // joins the first KV group (read from step 0)
  const int nb = (ke - ks + 63) >> 6;
  for (int j = 0; j < RSW - 1; j++) issue(j, ks + j * 64, rowof(ks + j * 64, 0), rowof(ks + j * 64, 1), rowof(ks + j * 64, 2), rowof(ks + j * 64, 3));
  int rq[4];  // rows of the next step to issue (RSW - 1 ahead), looked up a step early
#pragma unroll
  for (int i2 = 0; i2 < 4; i2++) rq[i2] = rowof(ks + (RSW - 1) * 64, i2);
  float o[32], mx[2] = {-INFINITY, -INFINITY}, ls[2] = {0.f, 0.f};
#pragma unroll
  for (int i = 0; i < 32; i++) if (!AW_Z) o[i] = 0.f;
  const int lw0 = 2 * w;  // tokens of this warp's rows: g -> lw0, g + 8 -> lw0 + 1
  const int lomax = a.window ? max(0, seq + t0 + lt - 1 - a.window + 1) : 0;
  // L2 prefetch PFD steps beyond the ring: thread tid < 128 -> key tid & 63 of K (tid < 64) or V; row looked up a step early
  auto pfrow = [&](int kb) -> int { const int kp = kb + (tid & 63), kc = min(kp, ke - 1); return kp < ke ? (PS1 ? __ldg(ptb + kc) : __ldg(ptb + kc / a.ps) * a.ps + kc % a.ps) : -1; };
  int pr = PFD > 0 && tid < 128 ? pfrow(ks + (RSW - 1 + PFD) * 64) : -1;
  for (int bi = 0, kb = ks; bi < nb; bi++, kb += 64) {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(RSW - 2));
    asm volatile("bar.sync %0, 128;\n" :: "r"(1 + h) : "memory");  // this warpgroup: step bi landed, slot of bi - 1 free
    if (PS1) ptseg(bi + RSW + 3);  // joins this step's group (complete RSW - 1 steps later, read at step bi + 3)
    issue((bi + RSW - 1) % RSW, kb + (RSW - 1) * 64, rq[0], rq[1], rq[2], rq[3]);
#pragma unroll
    for (int i2 = 0; i2 < 4; i2++) rq[i2] = rowr(kb + RSW * 64, i2);
    if (PFD > 0 && tid < 128) {
#if __CUDA_ARCH__ >= 900
      if (pr >= 0) asm volatile("cp.async.bulk.prefetch.L2.global [%0], 128;\n" :: "l"((tid < 64 ? a.kc : a.vc) + (size_t)pr * 128) : "memory");
#endif
      pr = pfrow(kb + (RSW + PFD) * 64);
    }
    // convert this head's V half to f16 (double-buffered: the previous step's PV may still be reading the other one)
    const uint32_t raw = ring + (bi % RSW) * 16384, vf = vf0 + (VDB ? (bi & 1) * 8192 : 0);
    if (!VDB) wg_wait0();  // single V buffer: the previous PV must be done before it is overwritten
    {
      const int key = wt >> 1;
#pragma unroll
      for (int u = 0; u < 2; u++) {
        const int c = 4 * h + 2 * (wt & 1) + u;
        uint4 x;
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(x.x), "=r"(x.y), "=r"(x.z), "=r"(x.w) : "r"(raw + 8192 + key * 128 + ((c ^ (key & 7)) << 4)));
        const int j0 = 2 * (c - 4 * h);
        const uint32_t dst = vf + key * 128;
        asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" :: "r"(dst + ((j0 ^ (key & 7)) << 4)), "r"(e2h(x.x)), "r"(e2h(x.x >> 16)), "r"(e2h(x.y)), "r"(e2h(x.y >> 16)));
        asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" :: "r"(dst + (((j0 + 1) ^ (key & 7)) << 4)), "r"(e2h(x.z)), "r"(e2h(x.z >> 16)), "r"(e2h(x.w)), "r"(e2h(x.w >> 16)));
      }
    }
    fence_async();
    asm volatile("bar.sync %0, 128;\n" :: "r"(1 + h) : "memory");
    float s[32];
    wg_fence();
#ifdef AW_NOMMA
#pragma unroll
    for (int i = 0; i < 32; i++) s[i] = __int_as_float(lds32a(raw + 4 * (i + wt)) & 0x3f7fffff);
#else
    wg_ss8<0>(s, gdesc(qs + 64 * h, 16, 1024), gdesc(raw + 64 * h, 16, 1024));
    wg_ss8<1>(s, gdesc(qs + 64 * h + 32, 16, 1024), gdesc(raw + 64 * h + 32, 16, 1024));
#endif
    wg_commit(); wg_wait0();  // also retires the previous step's PV
#if AW_FOP
    fop(o, 32); fop(s, 32);
#endif
    const bool full = kb + 64 <= ke && kb + 64 <= seq + t0 + 1 && kb >= lomax && lt == 8;
    float scl[2];
    bool upd[2] = {true, true};
    uint32_t pa[16];
#pragma unroll
    for (int hh = 0; hh < 2; hh++) {
      if (!full) {
        const int p = lw0 + hh < lt ? seq + t0 + lw0 + hh : -1, lo = a.window ? max(0, p - a.window + 1) : 0;
#pragma unroll
        for (int j = 0; j < 8; j++)
#pragma unroll
          for (int e = 0; e < 2; e++) { const int kp = kb + 8 * j + 2 * t + e; if (!(kp < ke && kp <= p && kp >= lo)) s[4 * j + 2 * hh + e] = -INFINITY; }
      }
      float bm = -INFINITY;
#pragma unroll
      for (int j = 0; j < 8; j++) bm = fmaxf(bm, fmaxf(s[4 * j + 2 * hh], s[4 * j + 2 * hh + 1]));
      bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, 1));
      bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, 2));
#if AW_LZ
      // lazy rescale: keep the running max unless a score exceeds it by more than 2^LZT (P <= 2^LZT in f16 is fine)
      const bool up = bm * a.qk > mx[hh] + LZT;
      const float nm = up ? fmaxf(mx[hh], bm * a.qk) : mx[hh];
      const float nmf = nm == -INFINITY ? 0.f : nm;
      upd[hh] = up;
      scl[hh] = up ? ex2(mx[hh] - nmf) : 1.f;
#else
      const float nm = fmaxf(mx[hh], bm * a.qk);
      const float nmf = nm == -INFINITY ? 0.f : nm;
      scl[hh] = ex2(mx[hh] - nmf);
#endif
      mx[hh] = nm;
#if AW_H2
      // packed f16 exp: one MUFU op per score pair; the f16 pair is P's A fragment as is
      __half2 hs = __float2half2_rn(0.f);
#pragma unroll
      for (int j = 0; j < 8; j++) {
        const __half2 ar = __floats2half2_rn(fmaf(s[4 * j + 2 * hh], a.qk, -nmf), fmaf(s[4 * j + 2 * hh + 1], a.qk, -nmf));
        uint32_t pv; asm("ex2.approx.f16x2 %0, %1;" : "=r"(pv) : "r"(*(const uint32_t*)&ar));
        pa[2 * j + hh] = pv;
        hs = __hadd2(hs, *(const __half2*)&pv);
      }
      const float2 sf = __half22float2(hs);
      ls[hh] = ls[hh] * scl[hh] + (sf.x + sf.y);
#else
      float sum = 0.f;
#pragma unroll
      for (int j = 0; j < 8; j++)
#pragma unroll
        for (int e = 0; e < 2; e++) { const float pv = ex2(fmaf(s[4 * j + 2 * hh + e], a.qk, -nmf)); s[4 * j + 2 * hh + e] = pv; sum += pv; }
      ls[hh] = ls[hh] * scl[hh] + sum;
#endif
    }
    const bool anyup = AW_LZ ? __any_sync(0xffffffffu, upd[0] | upd[1]) : true;
#pragma unroll
    for (int j = 0; j < 8; j++) {
      if (AW_LZ && !anyup) break;  // warp-uniform: no row of this warp moved its max
      o[4 * j] *= scl[0]; o[4 * j + 1] *= scl[0]; o[4 * j + 2] *= scl[1]; o[4 * j + 3] *= scl[1];
    }
#if !AW_H2
#pragma unroll
    for (int i = 0; i < 16; i++) pa[i] = pk(s[2 * i], s[2 * i + 1]);
#endif
#if AW_FOP
    fop(o, 32);
#endif
    wg_fence();
#pragma unroll
    for (int kc = 0; kc < 4; kc++) {
#ifdef AW_NOMMA
      o[kc] += __uint_as_float(pa[4 * kc] ^ pa[4 * kc + 3]);
#else
      if (AW_Z && kc == 0) wg_rsz(o, pa[0], pa[1], pa[2], pa[3], gdesc(vf, WVL, WVS), bi > 0);
      else wg_rs(o, pa[4 * kc], pa[4 * kc + 1], pa[4 * kc + 2], pa[4 * kc + 3], gdesc(vf + kc * 2048, WVL, WVS));
#endif
    }
    wg_commit();
#if AW_FOP
    fop(o, 32);
#endif
  }
  wg_wait0();
#if AW_FOP
  fop(o, 32);
#endif
  if (AW_Z && nb == 0) {
#pragma unroll
    for (int i = 0; i < 32; i++) o[i] = 0.f;
  }
  const size_t it = ((size_t)bq * a.hkv + h) * a.nsplit + sp;
#pragma unroll
  for (int hh = 0; hh < 2; hh++) {
    float l = ls[hh];
    l += __shfl_xor_sync(0xffffffffu, l, 1);
    l += __shfl_xor_sync(0xffffffffu, l, 2);
    const int row = 16 * w + g + 8 * hh;
    float* wo = a.wo + (it * 64 + row) * D;  // 64 rows per (block, head, split)
#pragma unroll
    for (int j = 0; j < 8; j++) *(float2*)(wo + 8 * j + 2 * t) = make_float2(o[4 * j + 2 * hh], o[4 * j + 2 * hh + 1]);
    if (t == 0) { a.wml[(it * 64 + row) * 2] = mx[hh]; a.wml[(it * 64 + row) * 2 + 1] = l; }
  }
  if constexpr (MRG) {
    // the last min(8, nsplit) arrivers of a request merge its 128 rows (rank, rank + nm, ...); everyone else has arrived
    __threadfence();
    __syncthreads();
    __shared__ int rank_s;
    if (tid == 0) rank_s = atomicAdd(f.cnt + bq, 1);
    __syncthreads();
    const int nmg = min(8, a.nsplit), rank = rank_s - (a.nsplit - nmg);
    if (rank < 0) return;
    if (tid == 0) { volatile int* c = f.cnt + bq; while (*c < a.nsplit) __nanosleep(64); }
    __syncthreads();
    __threadfence();
    asm volatile("cp.async.wait_all;\n" ::: "memory");
    // 16 threads per row (4 dims each), 16 rows at a time; one online-softmax pass over the splits
    const int d4 = tid & 15;
    for (int rr = rank + nmg * (tid >> 4); rr < 128; rr += nmg * 16) {
      const int hh = rr >> 6, r = rr & 63, base = (bq * a.hkv + hh) * a.nsplit;
      float Mr = -INFINITY, Lr = 0.f;
      float4 a4 = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll 4
      for (int k2 = 0; k2 < a.nsplit; k2++) {
        const float2 ml = __ldcg((const float2*)(a.wml + ((size_t)(base + k2) * 64 + r) * 2));
        const float4 v = __ldcg((const float4*)(a.wo + ((size_t)(base + k2) * 64 + r) * D) + d4);
        if (ml.x == -INFINITY) continue;
        const float nm2 = fmaxf(Mr, ml.x), c0 = ex2(Mr - nm2), c1 = ex2(ml.x - nm2);
        Mr = nm2; Lr = Lr * c0 + ml.y * c1;
        a4.x = a4.x * c0 + v.x * c1; a4.y = a4.y * c0 + v.y * c1; a4.z = a4.z * c0 + v.z * c1; a4.w = a4.w * c0 + v.w * c1;
      }
      const float inv = f.vscale / Lr;
      a4.x *= inv; a4.y *= inv; a4.z *= inv; a4.w *= inv;
      float ss = a4.x * a4.x + a4.y * a4.y + a4.z * a4.z + a4.w * a4.w;
#pragma unroll
      for (int x = 1; x < 16; x <<= 1) ss += __shfl_xor_sync(0xffffu << (lane & 16), ss, x);
      const float rn = rsqrtf(ss / D + 1e-5f);
      const int l = t0 + (r >> 3), hd = hh * 8 + (r & 7);
      if ((r >> 3) >= lt) continue;
      const __nv_bfloat16* gt = f.qkv + ((size_t)b * a.L + l) * (2 * 16 + 4) * D + 20 * D + hd * D + d4 * 4;
      const float vals[4] = {a4.x, a4.y, a4.z, a4.w};
      __nv_bfloat16 ob[4];
#pragma unroll
      for (int i2 = 0; i2 < 4; i2++) ob[i2] = __float2bfloat16(vals[i2] * rn / (1.f + expf(-__bfloat162float(gt[i2]))));
      *(uint2*)(f.out + ((size_t)b * a.L + l) * 16 * D + hd * D + d4 * 4) = *(uint2*)ob;
    }
    const int nbq = a.B * nqb;
    if (tid == 0 && atomicAdd(f.cnt + nbq + bq, 1) == nmg - 1) { f.cnt[bq] = 0; f.cnt[nbq + bq] = 0; }
  }
}

// ---------------------------------------------------------------- combine v2: one CTA (256 threads) per (token, q head)
__global__ void __launch_bounds__(1024) comb2(A a, const __nv_bfloat16* qkv, __nv_bfloat16* out, float vscale) {
  const int tid = threadIdx.x, wi = blockIdx.x;
  const int hd = wi % a.hq, tl = wi / a.hq, l = tl % a.L, b = tl / a.L;
  const int grp = a.hq / a.hkv, h = hd / grp, r = l * grp + hd % grp;
  const size_t base = ((size_t)b * a.hkv + h) * a.nsplit;
  __shared__ float red[1024];
  __shared__ float4 vec[64][16];
  float M = -INFINITY;
  for (int s = tid; s < a.nsplit; s += 1024) M = fmaxf(M, a.wml[((base + s) * a.rows + r) * 2]);
  red[tid] = M; __syncthreads();
  for (int x = 512; x; x >>= 1) { if (tid < x) red[tid] = fmaxf(red[tid], red[tid + x]); __syncthreads(); }
  M = red[0]; __syncthreads();
  const int sg = tid >> 4, dv = tid & 15;  // 64 split groups x 16 float4 of the 64 dims
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f); float Ls = 0.f;
#pragma unroll 4
  for (int s = sg; s < a.nsplit; s += 64) {
    const float* ml = a.wml + ((base + s) * a.rows + r) * 2;
    const float m = ml[0];
    if (m == -INFINITY) continue;
    const float w = ex2(m - M);
    if (dv == 0) Ls += w * ml[1];
    const float4 v = ((const float4*)(a.wo + ((base + s) * a.rows + r) * D))[dv];
    acc.x += w * v.x; acc.y += w * v.y; acc.z += w * v.z; acc.w += w * v.w;
  }
  vec[sg][dv] = acc;
  red[tid] = Ls;
  __syncthreads();
  if (tid < 16) {
    float4 v = vec[0][tid];
    for (int k = 1; k < 64; k++) { const float4 u = vec[k][tid]; v.x += u.x; v.y += u.y; v.z += u.z; v.w += u.w; }
    vec[0][tid] = v;
  }
  if (tid == 0) { float L2 = 0.f; for (int k = 0; k < 64; k++) L2 += red[k * 16]; red[0] = L2; }
  __syncthreads();
  if (tid < 32) {
    const float* vf = (const float*)vec[0];
    const float inv = vscale / red[0];
    const float o0 = vf[tid] * inv, o1 = vf[tid + 32] * inv;
    float ss = o0 * o0 + o1 * o1;
#pragma unroll
    for (int x = 16; x; x >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, x);
    const float rn = rsqrtf(ss / D + 1e-5f);
    const int W = (2 * a.hq + 2 * a.hkv) * D;
    const __nv_bfloat16* gt = qkv + (size_t)tl * W + (a.hq + 2 * a.hkv) * D + hd * D;
    const float g0 = __bfloat162float(gt[tid]), g1 = __bfloat162float(gt[tid + 32]);
    __nv_bfloat16* op = out + (size_t)tl * a.hq * D + hd * D;
    op[tid] = __float2bfloat16(o0 * rn / (1.f + expf(-g0)));
    op[tid + 32] = __float2bfloat16(o1 * rn / (1.f + expf(-g1)));
  }
}

// ---------------------------------------------------------------- combine + RMSNorm x sigmoid(gate): one warp per (token, q head)
__global__ void comb(A a, const __nv_bfloat16* qkv, __nv_bfloat16* out, float vscale) {
  const int lane = threadIdx.x & 31, wi = blockIdx.x * 8 + (threadIdx.x >> 5);
  if (wi >= a.B * a.L * a.hq) return;
  const int hd = wi % a.hq, tl = wi / a.hq, l = tl % a.L, b = tl / a.L;
  const int grp = a.hq / a.hkv, h = hd / grp, r = l * grp + hd % grp;
  const int base = (b * a.hkv + h) * a.nsplit;
  float M = -INFINITY;
  for (int s = lane; s < a.nsplit; s += 32) M = fmaxf(M, a.wml[((size_t)(base + s) * a.rows + r) * 2]);
#pragma unroll
  for (int x = 16; x; x >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, x));
  float acc[D], Ls = 0.f;
#pragma unroll
  for (int i = 0; i < D; i++) acc[i] = 0.f;
  for (int s = lane; s < a.nsplit; s += 32) {
    const float* ml = a.wml + ((size_t)(base + s) * a.rows + r) * 2;
    const float m = ml[0];
    if (m == -INFINITY) continue;
    const float w = exp2f(m - M);
    Ls += w * ml[1];
    const float4* op = (const float4*)(a.wo + ((size_t)(base + s) * a.rows + r) * D);
#pragma unroll
    for (int i = 0; i < D / 4; i++) { const float4 v = op[i]; acc[4 * i] += w * v.x; acc[4 * i + 1] += w * v.y; acc[4 * i + 2] += w * v.z; acc[4 * i + 3] += w * v.w; }
  }
#pragma unroll
  for (int x = 16; x; x >>= 1) {
    Ls += __shfl_xor_sync(0xffffffffu, Ls, x);
#pragma unroll
    for (int i = 0; i < D; i++) acc[i] += __shfl_xor_sync(0xffffffffu, acc[i], x);
  }
  float o0 = 0.f, o1 = 0.f;
#pragma unroll
  for (int i = 0; i < 32; i++) { if (i == lane) { o0 = acc[i]; o1 = acc[i + 32]; } }
  const float inv = vscale / Ls;
  o0 *= inv; o1 *= inv;
  float ss = o0 * o0 + o1 * o1;
#pragma unroll
  for (int x = 16; x; x >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, x);
  const float rn = rsqrtf(ss / D + 1e-5f);
  const int W = (2 * a.hq + 2 * a.hkv) * D;
  const __nv_bfloat16* gt = qkv + (size_t)tl * W + (a.hq + 2 * a.hkv) * D + hd * D;
  const float g0 = __bfloat162float(gt[lane]), g1 = __bfloat162float(gt[lane + 32]);
  __nv_bfloat16* op = out + (size_t)tl * a.hq * D + hd * D;
  op[lane] = __float2bfloat16(o0 * rn / (1.f + expf(-g0)));
  op[lane + 32] = __float2bfloat16(o1 * rn / (1.f + expf(-g1)));
}


// ================================================================ fused decode (L = 1, 16 q / 2 kv heads): one launch.
// CTA (request, split), 8 warps; warp w streams sub-range w of the CTA's keys through its own swizzled cp.async ring
// (full 128-byte K and V rows, both kv heads). Each CTA derives the step's query (RMSNorm, RoPE) and the new token's
// FP8 key/value itself (split 0 also stores them); the last CTA of a request merges the splits and applies the gate.
struct F {
  const __nv_bfloat16* qkv; const int64_t* loc; uint8_t* kc; uint8_t* vc; const int* pt; const int* seq;
  float* wo; float* wml; int* cnt; __nv_bfloat16* out;
  int B, ps, maxp, window, NS; float theta, qsc, qk, kinv, vinv, vscale; unsigned long long* prof; const float* invf;
};
#ifdef PROF
__device__ __forceinline__ unsigned long long gtm() { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
#define PT(k) do { if ((threadIdx.x & 31) == 0) p.prof[(blockIdx.x * 8 + (threadIdx.x >> 5)) * 8 + (k)] = gtm(); } while (0)
#else
#define PT(k)
#endif
#ifndef RBF
#define RBF 5
#endif
// RoPE angle pos * inv_freq[pair] in fp32 (as the reference), cos/sin after an exact double-precision reduction
__device__ __forceinline__ void rope_cs2(float invf, int pos, float& sn, float& cs) {  // invf: this pair's frequency
  const float ang = (float)pos * invf;
  const double a = (double)ang, k = rint(a * 0.15915494309189535);
  __sincosf((float)(a - k * 6.283185307179586), &sn, &cs);
}
__device__ __forceinline__ void rope_cs(const float* invf, int pos, int pair, float& sn, float& cs) {
  const float ang = (float)pos * invf[pair];
  const double a = (double)ang, k = rint(a * 0.15915494309189535);
  __sincosf((float)(a - k * 6.283185307179586), &sn, &cs);
}

template <bool CL>
__global__ void __launch_bounds__(256, 1) fdec(F p) {
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, t = lane & 3;
#ifdef FD_EMPTY
  if (tid == 0 && p.NS < 0) p.out[0] = __float2bfloat16(0.f);
  return;
#endif
  PT(0);
  const int b = blockIdx.x / p.NS, s = blockIdx.x % p.NS;
  // raw query / new key / value loads first: they need nothing, and everything below waits on seq and the page table
  const __nv_bfloat16* row0 = p.qkv + (size_t)b * (2 * 16 + 4) * D;
  uint4 qraw[2][2];
#pragma unroll
  for (int h = 0; h < 2; h++) { const uint4* qp = (const uint4*)(row0 + (h * 8 + g) * D + t * 16); qraw[h][0] = qp[0]; qraw[h][1] = qp[1]; }
  float ivq[8], ivk[2];  // RoPE inverse frequencies this lane needs, fetched before the KV loads queue up
#pragma unroll
  for (int j = 0; j < 8; j++) ivq[j] = __ldg(p.invf + t * 8 + j);
  ivk[0] = __ldg(p.invf + (lane & 15) * 2); ivk[1] = __ldg(p.invf + (lane & 15) * 2 + 1);
  uint2 kraw = make_uint2(0, 0), vraw = make_uint2(0, 0);
  if (w == 0) {
    const int h = lane >> 4, d0 = (lane & 15) * 4;
    kraw = *(const uint2*)(row0 + 16 * D + h * D + d0); vraw = *(const uint2*)(row0 + 18 * D + h * D + d0);
  }
  const int seq = p.seq[b];
  const int k0 = p.window ? max(0, seq - p.window + 1) : 0, span = seq + 1 - k0;
  const int cs = k0 + (int)((long long)span * s / p.NS), ce = k0 + (int)((long long)span * (s + 1) / p.NS);
  const int ws = cs + (int)((long long)(ce - cs) * w / 8), we = cs + (int)((long long)(ce - cs) * (w + 1) / 8);
  const __nv_bfloat16* row = p.qkv + (size_t)b * (2 * 16 + 4) * D;
  extern __shared__ __align__(16) uint8_t smem[];
  const uint32_t sb = (uint32_t)__cvta_generic_to_shared(smem);
  const uint32_t tab = sb + 8 * RBF * 4096;  // new token's K row (128 B) then V row (128 B)
  const int* ptb = p.pt + (size_t)b * p.maxp;
  const uint32_t ring = sb + w * RBF * 4096;
  // cache row of key ws..: -1 past the range, -2 for the new token (taken from the table)
  auto rowsfor = [&](int kb) -> int {
    const int kp = kb + (lane & 15);
    return (kb < we && kp < we) ? (kp >= seq ? -2 : ptb[kp / p.ps] * p.ps + kp % p.ps) : -1;
  };
  // page size 1: the warp's page-table entries stream into a 4-segment smem ring (8 blocks per segment), so the
  // in-loop lookups are shared-memory reads instead of a chain of dependent global loads
  const bool pt1 = p.ps == 1;
  const uint32_t ptr = tab + 256 + w * 2048;
  auto ptseg = [&](int sg) {  // entries ws + 128 sg .. + 127 -> ring slot sg & 3 (4 B copies: rows need not be aligned)
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int e = j * 32 + lane, idx = ws + sg * 128 + e;
      asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" :: "r"(ptr + (sg & 3) * 512 + e * 4), "l"(ptb + (idx < we ? idx : 0)), "r"(idx < we ? 4 : 0));
    }
  };
  auto rowss = [&](int kb) -> int {  // rowsfor through the ring (valid for blocks within the 2 loaded segments ahead)
    if (!pt1) return rowsfor(kb);
    const int kp = kb + (lane & 15);
    if (!(kb < we && kp < we)) return -1;
    if (kp >= seq) return -2;
    int v; asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(v) : "r"(ptr + ((kp - ws) & 511) * 4));
    return v;
  };
  const int c = lane & 7;
  auto issue = [&](int slot, int rl, bool tabok) {  // the new token's row comes from the table once it is written
    const uint32_t d0 = ring + slot * 4096;
    if (__all_sync(0xffffffffu, rl == -1)) { asm volatile("cp.async.commit_group;\n"); return; }  // past the range: nothing to load
#ifdef FD_NOLOAD
    asm volatile("cp.async.commit_group;\n"); return;
#endif
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const int r = i * 4 + (lane >> 3), rw = __shfl_sync(0xffffffffu, rl, r);
      const uint32_t dd = d0 + r * 128 + ((c ^ (r & 7)) << 4);
      if (rw == -2) {
        if (!tabok) continue;
        uint4 kv, vv;
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(kv.x), "=r"(kv.y), "=r"(kv.z), "=r"(kv.w) : "r"(tab + c * 16));
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(vv.x), "=r"(vv.y), "=r"(vv.z), "=r"(vv.w) : "r"(tab + 128 + c * 16));
        asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" :: "r"(dd), "r"(kv.x), "r"(kv.y), "r"(kv.z), "r"(kv.w));
        asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" :: "r"(dd + 2048), "r"(vv.x), "r"(vv.y), "r"(vv.z), "r"(vv.w));
      } else {
        const size_t off = (size_t)max(rw, 0) * 128 + c * 16;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dd), "l"(p.kc + off), "r"(rw >= 0 ? 16 : 0));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dd + 2048), "l"(p.vc + off), "r"(rw >= 0 ? 16 : 0));
      }
    }
    asm volatile("cp.async.commit_group;\n");
  };
  const int nb = max(0, (we - ws + 15) >> 4);
  int rpro[RBF - 1];
#pragma unroll
  for (int i = 0; i < RBF - 1; i++) rpro[i] = rowsfor(ws + i * 16);  // independent loads in flight together
  int rnext = rowsfor(ws + (RBF - 1) * 16), rn1 = rowsfor(ws + RBF * 16), rn2 = rowsfor(ws + (RBF + 1) * 16);  // 3 blocks ahead
  if (pt1) { ptseg(0); ptseg(1); }  // joins the first KV group
#pragma unroll
  for (int i = 0; i < RBF - 1; i++) issue(i, rpro[i], false);  // KV loads start before the query prologue
  if (w == 0) {  // new key: RMSNorm + RoPE, value as is; both quantized to E4M3
    const int h = lane >> 4, d0 = (lane & 15) * 4;
    float k[4], v[4], ss = 0.f;
#pragma unroll
    const __nv_bfloat16* kr = (const __nv_bfloat16*)&kraw; const __nv_bfloat16* vr = (const __nv_bfloat16*)&vraw;
    for (int i = 0; i < 4; i++) { k[i] = __bfloat162float(kr[i]); v[i] = __bfloat162float(vr[i]); ss += k[i] * k[i]; }
#pragma unroll
    for (int o = 1; o < 16; o <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    const float rn = rsqrtf(ss / D + 1e-5f);
#pragma unroll
    for (int i = 0; i < 4; i++) k[i] *= rn;
    if (p.theta > 0.f) {
#pragma unroll
      for (int i = 0; i < 4; i += 2) {
        float sn, cs2; rope_cs2(ivk[i >> 1], seq, sn, cs2);
        const float e = k[i], o = k[i + 1];
        k[i] = e * cs2 - o * sn; k[i + 1] = e * sn + o * cs2;
      }
    }
    uint32_t kb4 = 0, vb4 = 0;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      __nv_fp8_e4m3 fk(k[i] * p.kinv), fv(v[i] * p.vinv);
      kb4 |= (uint32_t)fk.__x << (8 * i); vb4 |= (uint32_t)fv.__x << (8 * i);
    }
    asm volatile("st.shared.u32 [%0], %1;\n" :: "r"(tab + h * D + d0), "r"(kb4));
    asm volatile("st.shared.u32 [%0], %1;\n" :: "r"(tab + 128 + h * D + d0), "r"(vb4));
    if (s == 0) {
      *(uint32_t*)(p.kc + (size_t)p.loc[b] * 128 + h * D + d0) = kb4;
      *(uint32_t*)(p.vc + (size_t)p.loc[b] * 128 + h * D + d0) = vb4;
    }
  }
  // query fragments: lane (g, t) holds q[head h*8+g][16t .. 16t+16) as f16, normalized, scaled, rotated
  uint32_t qa[2][4][4];
#pragma unroll
  for (int h = 0; h < 2; h++) {
    const __nv_bfloat162* bb = (const __nv_bfloat162*)&qraw[h][0];
    const __nv_bfloat162* bc = (const __nv_bfloat162*)&qraw[h][1];
    float x[16], ss = 0.f;
#pragma unroll
    for (int i = 0; i < 4; i++) { float2 f = __bfloat1622float2(bb[i]); x[2 * i] = f.x; x[2 * i + 1] = f.y; f = __bfloat1622float2(bc[i]); x[8 + 2 * i] = f.x; x[9 + 2 * i] = f.y; }
#pragma unroll
    for (int i = 0; i < 16; i++) ss += x[i] * x[i];
    ss += __shfl_xor_sync(0xffffffffu, ss, 1);
    ss += __shfl_xor_sync(0xffffffffu, ss, 2);
    const float rn = rsqrtf(ss / D + 1e-5f) * p.qsc;
#pragma unroll
    for (int i = 0; i < 16; i++) x[i] *= rn;
    if (p.theta > 0.f) {
#pragma unroll
      for (int i = 0; i < 16; i += 2) {
        float sn, cs2; rope_cs2(ivq[i >> 1], seq, sn, cs2);
        const float e = x[i], o = x[i + 1];
        x[i] = e * cs2 - o * sn; x[i + 1] = e * sn + o * cs2;
      }
    }
#pragma unroll
    for (int j = 0; j < 4; j++) { qa[h][j][0] = pk(x[4 * j], x[4 * j + 1]); qa[h][j][2] = pk(x[4 * j + 2], x[4 * j + 3]); qa[h][j][1] = 0; qa[h][j][3] = 0; }
  }
  PT(1);
  __syncthreads();  // table ready
  if (__any_sync(0xffffffffu, (rpro[RBF - 2] == -2) | (rpro[0] == -2) | (RBF > 2 && rpro[RBF > 2 ? 1 : 0] == -2) | (RBF > 3 && rpro[RBF > 3 ? 2 : 0] == -2))) {
#pragma unroll
    for (int i = 0; i < RBF - 1; i++) {
      const uint32_t d0 = ring + i * 4096;
#pragma unroll
      for (int q = 0; q < 4; q++) {
        const int r = q * 4 + (lane >> 3), rw = __shfl_sync(0xffffffffu, rpro[i], r);
        if (rw != -2) continue;
        const uint32_t dd = d0 + r * 128 + ((c ^ (r & 7)) << 4);
        uint4 kv, vv;
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(kv.x), "=r"(kv.y), "=r"(kv.z), "=r"(kv.w) : "r"(tab + c * 16));
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(vv.x), "=r"(vv.y), "=r"(vv.z), "=r"(vv.w) : "r"(tab + 128 + c * 16));
        asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" :: "r"(dd), "r"(kv.x), "r"(kv.y), "r"(kv.z), "r"(kv.w));
        asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" :: "r"(dd + 2048), "r"(vv.x), "r"(vv.y), "r"(vv.z), "r"(vv.w));
      }
    }
  }
  PT(2);
  float o[2][8][2], mx[2] = {-INFINITY, -INFINITY}, ls[2] = {0.f, 0.f};
#pragma unroll
  for (int h = 0; h < 2; h++)
#pragma unroll
    for (int n = 0; n < 8; n++) o[h][n][0] = o[h][n][1] = 0.f;
  int rpf = PFF > 0 ? rowsfor(ws + (RBF - 1 + PFF) * 16) : -1;  // L2 prefetch PFF blocks past the ring: lane < 16 K, else V
  for (int bi = 0, kb = ws; bi < nb; bi++, kb += 16) {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(RBF - 2));
    __syncwarp();
    if (pt1 && (bi & 7) == 0) ptseg((bi >> 3) + 2);  // segment k + 2 while in segment k: first read 9 blocks later
    issue((bi + RBF - 1) % RBF, rnext, true);
    rnext = rn1; rn1 = rn2; rn2 = rowss(kb + (RBF + 2) * 16);
    if (PFF > 0) {
#if __CUDA_ARCH__ >= 900
      if (rpf >= 0) asm volatile("cp.async.bulk.prefetch.L2.global [%0], 128;\n" :: "l"((lane < 16 ? p.kc : p.vc) + (size_t)rpf * 128) : "memory");
#endif
      rpf = rowsfor(kb + (RBF + PFF) * 16);
    }
    const uint32_t sl = ring + (bi % RBF) * 4096;
#ifdef FD_NOCOMP
    if (sl == 0xffffffffu)
#endif
#pragma unroll
    for (int h = 0; h < 2; h++) {
      uint4 kw[2];
      const uint32_t ck = ((h * 4 + t) ^ (g & 7)) << 4;
#pragma unroll
      for (int n = 0; n < 2; n++)
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(kw[n].x), "=r"(kw[n].y), "=r"(kw[n].z), "=r"(kw[n].w) : "r"(sl + (g + 8 * n) * 128 + ck));
      uint2 vw[4];
      const int rv[4] = {2 * t, 2 * t + 1, 2 * t + 8, 2 * t + 9};
#pragma unroll
      for (int i = 0; i < 4; i++)
        asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];\n" : "=r"(vw[i].x), "=r"(vw[i].y)
                     : "r"(sl + 2048 + rv[i] * 128 + (((h * 4 + (g >> 1)) ^ (rv[i] & 7)) << 4) + (g & 1) * 8));
      float sc4[2][4];
#pragma unroll
      for (int n = 0; n < 2; n++) {
        sc4[n][0] = sc4[n][1] = sc4[n][2] = sc4[n][3] = 0.f;
        const uint32_t kk[4] = {kw[n].x, kw[n].y, kw[n].z, kw[n].w};
#pragma unroll
        for (int j = 0; j < 4; j++) mma16(sc4[n], qa[h][j][0], qa[h][j][1], qa[h][j][2], qa[h][j][3], e2h(kk[j]), e2h(kk[j] >> 16));
      }
      float v[4];
#pragma unroll
      for (int n = 0; n < 2; n++)
#pragma unroll
        for (int e = 0; e < 2; e++) { const int kp = kb + n * 8 + 2 * t + e; v[n * 2 + e] = kp < we ? sc4[n][e] * p.qk : -INFINITY; }
      float bm = fmaxf(fmaxf(v[0], v[1]), fmaxf(v[2], v[3]));
      bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, 1));
      bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, 2));
      const float nm = fmaxf(mx[h], bm), scl = ex2(mx[h] - nm);
      mx[h] = nm;
      const float p0 = ex2(v[0] - nm), p1 = ex2(v[1] - nm), p2 = ex2(v[2] - nm), p3 = ex2(v[3] - nm);
      ls[h] = ls[h] * scl + (p0 + p1 + p2 + p3);
      const uint32_t pa0 = pk(p0, p1), pa2 = pk(p2, p3);
#pragma unroll
      for (int n = 0; n < 8; n += 2) {
        const int wi = n >> 2, sh = (n & 3) * 8;
        const uint32_t x0 = wi ? vw[0].y : vw[0].x, x1 = wi ? vw[1].y : vw[1].x, x2 = wi ? vw[2].y : vw[2].x, x3 = wi ? vw[3].y : vw[3].x;
        const uint32_t lo01 = __byte_perm(x0 >> sh, x1 >> sh, 0x5140), lo23 = __byte_perm(x2 >> sh, x3 >> sh, 0x5140);
        float c0[4] = {o[h][n][0] * scl, o[h][n][1] * scl, 0.f, 0.f}, c1[4] = {o[h][n + 1][0] * scl, o[h][n + 1][1] * scl, 0.f, 0.f};
        mma16(c0, pa0, 0u, pa2, 0u, e2h(lo01), e2h(lo23));
        mma16(c1, pa0, 0u, pa2, 0u, e2h(lo01 >> 16), e2h(lo23 >> 16));
        o[h][n][0] = c0[0]; o[h][n][1] = c0[1]; o[h][n + 1][0] = c1[0]; o[h][n + 1][1] = c1[1];
      }
    }
  }
  PT(3);
  asm volatile("cp.async.wait_group 0;\n");
#ifdef FDGAP
  PT(4);
#endif
  __syncthreads();  // rings free: reuse for the warps' partials
#ifdef FDGAP
  PT(5);
#endif
  // [8 warps][16 rows][72] partials: float4 units, row stride 18 units, unit u stored at u ^ ((u >> 3) & 1) so the
  // 8 lanes of a 128-bit store phase hit distinct banks (scalar stores here were ~16-way conflicted: ~2 us)
  constexpr int MS = 72;
  float* mo = (float*)smem;
  float* mm = mo + 8 * 16 * MS;             // [8][16] max
  float* ml = mm + 8 * 16;                  // [8][16] sum
#pragma unroll
  for (int h = 0; h < 2; h++) {
    float l = ls[h];
    l += __shfl_xor_sync(0xffffffffu, l, 1);
    l += __shfl_xor_sync(0xffffffffu, l, 2);
    const int r = h * 8 + g;
    float* dst = mo + (w * 16 + r) * MS;
#pragma unroll
    for (int e = 0; e < 2; e++)
#pragma unroll
      for (int k = 0; k < 2; k++) {
        const int u = 4 * t + 2 * e + k;
        *(float4*)(dst + 4 * (u ^ ((u >> 3) & 1))) = make_float4(o[h][4 * k][e], o[h][4 * k + 1][e], o[h][4 * k + 2][e], o[h][4 * k + 3][e]);
      }
    if (t == 0) { mm[w * 16 + r] = nb ? mx[h] : -INFINITY; ml[w * 16 + r] = l; }
  }
  __syncthreads();
#ifdef FDGAP
  PT(6);
#endif
  // CTA merge: thread -> (row r = tid / 16, dims 4 * (tid % 16) .. +4)
  const int r = tid >> 4, dv = tid & 15;
  float M = -INFINITY;
#pragma unroll
  for (int k = 0; k < 8; k++) M = fmaxf(M, mm[k * 16 + r]);
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f); float L = 0.f;
#pragma unroll
  for (int k = 0; k < 8; k++) {
    const float m = mm[k * 16 + r];
    if (m == -INFINITY) continue;
    const float wt = ex2(m - M);
    L += wt * ml[k * 16 + r];
    const float4 v = ((const float4*)(mo + (k * 16 + r) * MS))[dv ^ ((dv >> 3) & 1)];
    acc.x += wt * v.x; acc.y += wt * v.y; acc.z += wt * v.z; acc.w += wt * v.w;
  }
  auto finish = [&](float4 a4, float Lr) {  // row r = q head, 16 threads (one half-warp) hold its 64 dims
    const float inv = p.vscale / Lr;
    a4.x *= inv; a4.y *= inv; a4.z *= inv; a4.w *= inv;
    float ss = a4.x * a4.x + a4.y * a4.y + a4.z * a4.z + a4.w * a4.w;
#pragma unroll
    for (int x = 1; x < 16; x <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, x);
    const float rn = rsqrtf(ss / D + 1e-5f);
    const __nv_bfloat16* gt = row + 20 * D + r * D + dv * 4;
    const float vals[4] = {a4.x, a4.y, a4.z, a4.w};
    __nv_bfloat16 ob[4];
#pragma unroll
    for (int i = 0; i < 4; i++) ob[i] = __float2bfloat16(vals[i] * rn / (1.f + expf(-__bfloat162float(gt[i]))));
    *(uint2*)(p.out + (size_t)b * 16 * D + r * D + dv * 4) = *(uint2*)ob;
  };
#ifdef FDGAP
  PT(7);
  if (s == 0) finish(acc, L);
  return;
#endif
  PT(4);
#ifdef FD_NOMERGE
  if (s == 0) finish(acc, L);
  return;
#endif
  if (p.NS == 1) { finish(acc, L); PT(5); return; }
#if __CUDA_ARCH__ >= 900
  if constexpr (CL) {  // the request's NS splits form a thread-block cluster: partials meet in distributed shared memory
    namespace cg = cooperative_groups;
    cg::cluster_group cl = cg::this_cluster();
    asm volatile("cp.async.wait_all;\n" ::: "memory");
    __syncthreads();
    float* xs = (float*)smem;  // this CTA's partial: [16 rows][64] O, then [16] (m, l)
    ((float4*)(xs + r * 64))[dv] = acc;
    if (dv == 0) { xs[1024 + 2 * r] = M; xs[1024 + 2 * r + 1] = L; }
    cl.sync();
    PT(5);
    const int NS = p.NS, rp = 16 / NS, d4 = tid & 15, kg = (tid >> 4) & (NS - 1), rl = tid / (16 * NS);
    const int hr = s + rl * NS;  // rows s, s + NS, ...
    const float* rx = cl.map_shared_rank(xs, kg);
    const float2 ml = *(const float2*)(rx + 1024 + 2 * hr);
    const float4 ov = ((const float4*)(rx + hr * 64))[d4];
    float* xm = xs + 2048;            // [16][16] max
    float* xl = xm + 256;             // [16][16] weighted sums
    float* xo = xl + 256;             // [16][16][64]
    const int slot = rl * NS + kg;
    if (d4 == 0) xm[slot] = ml.x;
    __syncthreads();
    float Mr = -INFINITY;
    for (int k = 0; k < NS; k++) Mr = fmaxf(Mr, xm[rl * NS + k]);
    const float wt = ml.x == -INFINITY ? 0.f : ex2(ml.x - Mr);
    ((float4*)(xo + slot * 64))[d4] = make_float4(wt * ov.x, wt * ov.y, wt * ov.z, wt * ov.w);
    if (d4 == 0) xl[slot] = wt * ml.y;
    __syncthreads();
    if (kg == 0 && rl < rp) {
      float4 a4 = make_float4(0.f, 0.f, 0.f, 0.f);
      float Lr = 0.f;
      for (int k = 0; k < NS; k++) {
        const float4 v = ((const float4*)(xo + (rl * NS + k) * 64))[d4];
        a4.x += v.x; a4.y += v.y; a4.z += v.z; a4.w += v.w;
        Lr += xl[rl * NS + k];
      }
      const float inv = p.vscale / Lr;
      a4.x *= inv; a4.y *= inv; a4.z *= inv; a4.w *= inv;
      float ss = a4.x * a4.x + a4.y * a4.y + a4.z * a4.z + a4.w * a4.w;
#pragma unroll
      for (int x = 1; x < 16; x <<= 1) ss += __shfl_xor_sync(0xffffu << (lane & 16), ss, x);
      const float rn = rsqrtf(ss / D + 1e-5f);
      const __nv_bfloat16* gt = row + 20 * D + hr * D + d4 * 4;
      const float vals[4] = {a4.x, a4.y, a4.z, a4.w};
      __nv_bfloat16 ob[4];
#pragma unroll
      for (int i = 0; i < 4; i++) ob[i] = __float2bfloat16(vals[i] * rn / (1.f + expf(-__bfloat162float(gt[i]))));
      *(uint2*)(p.out + (size_t)b * 16 * D + hr * D + d4 * 4) = *(uint2*)ob;
    }
    PT(6);
    cl.sync();  // remote reads of this CTA's partial are done before it exits
    return;
  }
#endif
  float* gwo = p.wo + ((size_t)(b * p.NS + s) * 16 + r) * D;
  ((float4*)gwo)[dv] = acc;
  if (dv == 0) { p.wml[((size_t)(b * p.NS + s) * 16 + r) * 2] = M; p.wml[((size_t)(b * p.NS + s) * 16 + r) * 2 + 1] = L; }
  __syncthreads();
  // the last nmg arrivers (a power of two <= min(16, NS)) of a request merge its 16 rows together, all rows at once:
  // thread = (row rl of rp = 16 / nmg, split group kg of nmg, dims d4 * 4 .. + 3); everyone else has arrived
  __shared__ int rank_s;
  if (tid == 0) { int v; asm volatile("atom.add.acq_rel.gpu.global.s32 %0, [%1], 1;\n" : "=r"(v) : "l"(p.cnt + b) : "memory"); rank_s = v; }
  __syncthreads();
  PT(5);
  const int nmg = p.NS >= 16 ? 16 : (p.NS >= 8 ? 8 : (p.NS >= 4 ? 4 : (p.NS >= 2 ? 2 : 1))), rank = rank_s - (p.NS - nmg);
  if (rank < 0) return;
  if (tid == 0) { int v; do { asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(p.cnt + b) : "memory"); if (v < p.NS) __nanosleep(32); } while (v < p.NS); }
  __syncthreads();
  PT(7);
  __threadfence();
  asm volatile("cp.async.wait_all;\n" ::: "memory");
  float* xs = (float*)smem;  // [16 (rl, kg)][64] weighted O, [16] sums, [16] max
  const int d4 = tid & 15, kg = (tid >> 4) & (nmg - 1), rl = tid / (16 * nmg), slot = rl * nmg + kg;
  const int hr = rank + rl * nmg;
  const float* mlp = p.wml + ((size_t)b * p.NS * 16 + hr) * 2;
  float Mg = -INFINITY;
  for (int k = kg; k < p.NS; k += nmg) Mg = fmaxf(Mg, __ldcg(mlp + (size_t)k * 32));
  if (d4 == 0) xs[1024 + 16 + slot] = Mg;
  __syncthreads();
  float Mr = -INFINITY;
  for (int k = 0; k < nmg; k++) Mr = fmaxf(Mr, xs[1024 + 16 + rl * nmg + k]);
  float4 og = make_float4(0.f, 0.f, 0.f, 0.f);
  float lg = 0.f;
  for (int k = kg; k < p.NS; k += nmg) {
    const float2 mlv = __ldcg((const float2*)(mlp + (size_t)k * 32));
    const float4 v = __ldcg((const float4*)(p.wo + (((size_t)b * p.NS + k) * 16 + hr) * D) + d4);
    if (mlv.x == -INFINITY) continue;
    const float wt = ex2(mlv.x - Mr);
    lg += wt * mlv.y;
    og.x += wt * v.x; og.y += wt * v.y; og.z += wt * v.z; og.w += wt * v.w;
  }
  ((float4*)(xs + slot * 64))[d4] = og;
  if (d4 == 0) xs[1024 + slot] = lg;
  __syncthreads();
  if (kg == 0) {
    float4 a4 = make_float4(0.f, 0.f, 0.f, 0.f);
    float Lr = 0.f;
    for (int k = 0; k < nmg; k++) {
      const float4 v = ((const float4*)(xs + (rl * nmg + k) * 64))[d4];
      a4.x += v.x; a4.y += v.y; a4.z += v.z; a4.w += v.w;
      Lr += xs[1024 + rl * nmg + k];
    }
    const float inv = p.vscale / Lr;
    a4.x *= inv; a4.y *= inv; a4.z *= inv; a4.w *= inv;
    float ss = a4.x * a4.x + a4.y * a4.y + a4.z * a4.z + a4.w * a4.w;
#pragma unroll
    for (int x = 1; x < 16; x <<= 1) ss += __shfl_xor_sync(0xffffu << (lane & 16), ss, x);
    const float rn = rsqrtf(ss / D + 1e-5f);
    const __nv_bfloat16* gt = row + 20 * D + hr * D + d4 * 4;
    const float vals[4] = {a4.x, a4.y, a4.z, a4.w};
    __nv_bfloat16 ob[4];
#pragma unroll
    for (int i = 0; i < 4; i++) ob[i] = __float2bfloat16(vals[i] * rn / (1.f + expf(-__bfloat162float(gt[i]))));
    *(uint2*)(p.out + (size_t)b * 16 * D + hr * D + d4 * 4) = *(uint2*)ob;
  }
  PT(6);
  if (tid == 0 && atomicAdd(p.cnt + p.B + b, 1) == nmg - 1) { p.cnt[b] = 0; p.cnt[p.B + b] = 0; }
}

void prep_only(torch::Tensor qkv, torch::Tensor pos, torch::Tensor loc, torch::Tensor kc, torch::Tensor vc, torch::Tensor ksc,
               torch::Tensor vsc, torch::Tensor q8, int64_t hq, int64_t hkv, double theta, double qk_factor, torch::Tensor invf) {
  const int toks = qkv.size(0) * qkv.size(1);
  prep<<<dim3(toks, hq + 2 * hkv), D, 0, at::cuda::getCurrentCUDAStream()>>>((const __nv_bfloat16*)qkv.data_ptr(), pos.data_ptr<int64_t>(),
      loc.data_ptr<int64_t>(), (uint8_t*)kc.data_ptr(), (uint8_t*)vc.data_ptr(), ksc.data_ptr<float>(), vsc.data_ptr<float>(), nullptr,
      (uint8_t*)q8.data_ptr(), (int)hq, (int)hkv, (float)theta, (float)qk_factor, invf.data_ptr<float>());
}
void run_fdec(torch::Tensor qkv, torch::Tensor loc, torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seq,
              torch::Tensor wo, torch::Tensor wml, torch::Tensor cnt, torch::Tensor out, int64_t ps, int64_t window, double theta,
              double qk_factor, double sm_scale, double kv_scale, int64_t NS, torch::Tensor prof, torch::Tensor invf) {
  F p;
  p.qkv = (const __nv_bfloat16*)qkv.data_ptr(); p.loc = loc.data_ptr<int64_t>(); p.kc = (uint8_t*)kc.data_ptr(); p.vc = (uint8_t*)vc.data_ptr();
  p.pt = pt.data_ptr<int>(); p.seq = seq.data_ptr<int>(); p.wo = wo.data_ptr<float>(); p.wml = wml.data_ptr<float>(); p.cnt = cnt.data_ptr<int>();
  p.out = (__nv_bfloat16*)out.data_ptr(); p.B = seq.size(0); p.ps = ps; p.maxp = pt.size(1); p.window = window; p.NS = NS;
  p.theta = theta; p.qsc = qk_factor; p.qk = (float)(sm_scale * 1.4426950408889634); p.kinv = 1.f / kv_scale; p.vinv = 1.f / kv_scale; p.vscale = kv_scale;
  p.prof = prof.numel() ? (unsigned long long*)prof.data_ptr() : nullptr; p.invf = invf.data_ptr<float>();
  const int sm = max(8 * RBF * 4096 + 256 + 8 * 2048, (512 + 256 * D) * 4);  // ring + table + page-table rings, or the merge scratch
  static bool once = false;
  if (!once) {
    cudaFuncSetAttribute(fdec<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
    cudaFuncSetAttribute(fdec<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
    cudaFuncSetAttribute(fdec<true>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    once = true;
  }
  static int major = 0; if (!major) { int dev; cudaGetDevice(&dev); cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev); }
  if (CLUS && major >= 9 && NS > 1 && NS <= 16 && (NS & (NS - 1)) == 0 && p.B * NS <= 64) {  // clusters fragment GPCs: small grids only
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(p.B * NS); cfg.blockDim = dim3(256); cfg.dynamicSmemBytes = sm; cfg.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = NS; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
    cfg.attrs = at; cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, fdec<true>, p);
  } else fdec<false><<<p.B * NS, 256, sm, at::cuda::getCurrentCUDAStream()>>>(p);
}

void run(torch::Tensor qkv, torch::Tensor pos, torch::Tensor loc, torch::Tensor kc, torch::Tensor vc, torch::Tensor ksc,
         torch::Tensor vsc, torch::Tensor pt, torch::Tensor seq, torch::Tensor qbuf, torch::Tensor wo, torch::Tensor wml,
         torch::Tensor out, int64_t hq, int64_t hkv, int64_t ps, int64_t window, double theta, double qk_factor,
         double sm_scale, double kv_scale, int64_t nsplit, torch::Tensor invf, torch::Tensor cnt) {
  auto st = at::cuda::getCurrentCUDAStream();
  const int B = seq.size(0), toks = qkv.size(0) * qkv.size(1), L = toks / B;
  const bool V8 = L == 8 && hkv == 2 && hq == 16;
  const bool PF = L > 8 && hkv == 2 && hq == 16;  // prefill blocks: prep, then 8-token query blocks on wgmma
  static int major = 0; if (!major) { int dev; cudaGetDevice(&dev); cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev); }
  const bool Q8 = (V8 || PF) && major == 9 && WGV;
  const bool FU = Q8 && V8 && nsplit <= 32 && FUSEV;  // one launch: prologue, attention, merge + gate
  if (!FU) prep<<<dim3(toks, hq + 2 * hkv), D, 0, st>>>((const __nv_bfloat16*)qkv.data_ptr(), pos.data_ptr<int64_t>(), loc.data_ptr<int64_t>(),
      (uint8_t*)kc.data_ptr(), (uint8_t*)vc.data_ptr(), ksc.data_ptr<float>(), vsc.data_ptr<float>(), (__half*)qbuf.data_ptr(),
      Q8 ? (uint8_t*)qbuf.data_ptr() : nullptr, (int)hq, (int)hkv, (float)theta, (float)qk_factor, invf.data_ptr<float>());
#ifdef ONLYPREP
  return;
#endif
  A a;
  a.q = (const __half*)qbuf.data_ptr(); a.q8 = (const uint8_t*)qbuf.data_ptr(); a.kc = (const uint8_t*)kc.data_ptr(); a.vc = (const uint8_t*)vc.data_ptr();
  a.pt = pt.data_ptr<int>(); a.seq = seq.data_ptr<int>(); a.wo = wo.data_ptr<float>(); a.wml = wml.data_ptr<float>();
  a.B = B; a.L = L; a.hq = hq; a.hkv = hkv; a.ps = ps; a.maxp = pt.size(1); a.window = window; a.nsplit = nsplit;
  a.rows = L * (hq / hkv);
  a.qk = (float)(sm_scale * 1.4426950408889634) * 1.0f;  // k_scale applied below via ksc on host side (scalar)
  const int items = B * hkv * nsplit, blocks = (items + 7) / 8;
  const int mt = (a.rows + 15) / 16;
  const int sm = 8 * RB * 2048;
  static bool once = false;
  if (!once) { cudaFuncSetAttribute(attn<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm); cudaFuncSetAttribute(attn<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
               cudaFuncSetAttribute(attn<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm); once = true; }
  if (L == 1 && hkv == 2 && hq == 16) {
    const int sm2 = 8 * RB2 * 4096;
    static bool o2 = false;
    if (!o2) { cudaFuncSetAttribute(attn2, cudaFuncAttributeMaxDynamicSharedMemorySize, sm2); o2 = true; }
    attn2<<<(B * nsplit + 7) / 8, 256, sm2, st>>>(a);
  } else if (V8 || PF) {
    if (KALU) a.qk *= 256.f;
    const int sm8 = RB8 * 8192;
    static bool o8 = false;
    if (!o8) { cudaFuncSetAttribute(attn9<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm8); cudaFuncSetAttribute(attn9<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm8); o8 = true; }
    if (Q8) {
      a.qk = (float)(sm_scale * 1.4426950408889634 * qk_factor);
      const int smw = RSW * 16384 + 8192 + (VDB ? 32768 : 16384) + 4096 + 1024;
      static bool ow = false;
      if (!ow) {
        cudaFuncSetAttribute(attnw<true, false, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smw); cudaFuncSetAttribute(attnw<false, false, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smw);
        cudaFuncSetAttribute(attnw<true, true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smw); cudaFuncSetAttribute(attnw<false, true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smw);
        cudaFuncSetAttribute(attnw<true, false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smw); cudaFuncSetAttribute(attnw<false, false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smw);
        ow = true;
      }
      WF f{(const __nv_bfloat16*)qkv.data_ptr(), loc.data_ptr<int64_t>(), invf.data_ptr<float>(), cnt.data_ptr<int>(), (__nv_bfloat16*)out.data_ptr(),
           (float)theta, (float)(1.0 / kv_scale), (float)(1.0 / kv_scale), (float)kv_scale};
      if (FU) { if (ps == 1) attnw<true, true, true><<<B * nsplit, 256, smw, st>>>(a, f); else attnw<false, true, true><<<B * nsplit, 256, smw, st>>>(a, f); return; }
      if (PF) {
        const int g = B * ((L + 7) / 8) * nsplit;
        if (ps == 1) attnw<true, false, true><<<g, 256, smw, st>>>(a, f); else attnw<false, false, true><<<g, 256, smw, st>>>(a, f);
        return;
      }
      if (ps == 1) attnw<true, false, false><<<B * nsplit, 256, smw, st>>>(a, f); else attnw<false, false, false><<<B * nsplit, 256, smw, st>>>(a, f);
    } else if (ps == 1) attn9<true><<<B * ((L + 7) / 8) * nsplit, 128, sm8, st>>>(a); else attn9<false><<<B * ((L + 7) / 8) * nsplit, 128, sm8, st>>>(a);
  } else if (mt <= 1) attn<1><<<blocks, 256, sm, st>>>(a);
  else if (mt <= 2) attn<2><<<blocks, 256, sm, st>>>(a);
  else attn<4><<<blocks, 256, sm, st>>>(a);
  if (nsplit > 32) comb2<<<B * L * hq, 1024, 0, st>>>(a, (const __nv_bfloat16*)qkv.data_ptr(), (__nv_bfloat16*)out.data_ptr(), (float)kv_scale);
  else comb<<<(B * L * hq + 7) / 8, 256, 0, st>>>(a, (const __nv_bfloat16*)qkv.data_ptr(), (__nv_bfloat16*)out.data_ptr(), (float)kv_scale);
}
"""

A__mod = None
A__DBG = []
A__NOPROF = None
A__INV = {}


def A__invf(theta, dev):
    """RoPE inverse frequencies exactly as the reference builds them (fp32), cached per theta"""
    k = (float(theta), str(dev))
    if k not in A__INV:
        t = theta if theta else 1.0
        A__INV[k] = (1.0 / (t ** (torch.arange(0, 64, 2, device=dev).float() / 64))).contiguous()
    return A__INV[k]
A__BUF = {}
A__NSDIV = 128
A_OVR = {}  # (kind, B, sliding) -> split count override (tuning)


def A__big():
    pr = torch.cuda.get_device_properties(0)
    sm = getattr(pr, "shared_memory_per_block_optin", None)
    return sm >= 180000 if sm else (pr.major, pr.minor) in ((9, 0), (10, 0))


def A__hopper():
    return torch.cuda.get_device_capability() == (9, 0)


def A__gencode():
    """sm_90a (wgmma) on Hopper; elsewhere torch's own arch flags"""
    return ["-gencode=arch=compute_90a,code=sm_90a"] if torch.cuda.get_device_capability() == (9, 0) else []


def A__ext():
    global A__mod
    if A__mod is None:
        rb, rb2, rb3, rbf = (8, 6, 20, 5) if A__big() else (4, 3, 8, 2)
        A__mod = load_inline("attn_v36_%d" % rb + "".join(A__DBG).replace("-", "_").replace("=", ""),
                           cpp_sources="void prep_only(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,double,double,torch::Tensor); void run_fdec(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,double,double,double,double,int64_t,torch::Tensor,torch::Tensor); void run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t,double,double,double,double,int64_t,torch::Tensor,torch::Tensor);",
                           cuda_sources=A_CUDA, functions=["run", "run_fdec", "prep_only"], extra_cuda_cflags=["-O3", "-lineinfo", "-DRB=%d" % rb, "-DRB2=%d" % rb2, "-DRB3=%d" % rb3, "-DRBF=%d" % rbf] + A__DBG + A__gencode(), verbose=False)
    return A__mod


A_DIAG = False


def A__diag(dev):
    """timeline of the fused decode kernel on synthetic batch-1 inputs (sliding and full), printed once"""
    global A__mod, A__DBG, A__NOPROF
    keep = (A__mod, A__DBG)
    A__mod, A__DBG = None, A__DBG + ["-DPROF"]
    for window, theta, ctx in ((2048, 500000.0, 80000), (0, 0.0, 80000), (2048, 500000.0, 1000)):
        for B in (1, 32):
            mp = ctx + 8
            pt = torch.randperm(B * mp, device=dev, dtype=torch.int64).to(torch.int32).view(B, mp)
            kc = torch.randn((B * mp, 1, 2, 64), device=dev).to(torch.float8_e4m3fn); vc = kc.clone()
            seq = torch.full((B,), ctx, dtype=torch.int32, device=dev)
            loc = pt[:, ctx].long()
            qkv = torch.randn((B, 1, 2304), device=dev).bfloat16()
            span = min(mp, window + 1) if window else mp
            NS = max(1, min(132 // B, (span + A__NSDIV - 1) // A__NSDIV))
            if NS <= 17 and B * min(NS, 16) <= 64: NS = 1 << (min(NS, 16).bit_length() - 1)
            wo = torch.empty((B * NS * 16 * 64,), device=dev); wml = torch.empty((B * NS * 32,), device=dev)
            cnt = torch.zeros((2 * B,), dtype=torch.int32, device=dev); out = torch.empty((B, 1, 1024), dtype=torch.bfloat16, device=dev)
            prof = torch.zeros((B * NS * 8 * 8,), dtype=torch.int64, device=dev)
            fl = torch.ones(2 ** 26, device=dev)
            for it in range(3):
                fl.sum()
                A__ext().run_fdec(qkv, loc, kc, vc, pt, seq, wo, wml, cnt, out, 1, window, theta, 5.47, 0.125, 1.0, NS, prof, A__invf(theta, dev))
            torch.cuda.synchronize()
            P = prof.view(-1, 8).cpu().tolist()
            t0 = min(r[0] for r in P)
            col = lambda k: [(r[k] - t0) / 1e3 for r in P if r[k]]
            mx = lambda k: max(col(k)) if col(k) else -1
            print("FDIAG win %d B %d NS %d: start max %.1f prolog %.1f sync %.1f loop-end p50 %.1f max %.1f cta-merge %.1f ranked %.1f all-in %.1f last-end %.1f" % (
                window, B, NS, mx(0), mx(1), mx(2), sorted(col(3))[len(col(3)) // 2], mx(3), mx(4), mx(5), mx(7), mx(6)), flush=True)
    A__mod, A__DBG = keep


A_FA3PF = True
A__FA3 = []


def A__fa3():
    if not A__FA3:
        try:
            from sgl_kernel.flash_attn import flash_attn_with_kvcache
            A__FA3.append(flash_attn_with_kvcache)
        except Exception:
            A__FA3.append(None)
    return A__FA3[0]


def A__prefill_fa3(data, cfg, B, L, hq, hkv, qkv, dev):
    q8 = torch.empty((B * L, hq, 64), dtype=torch.uint8, device=dev)
    A__ext().prep_only(qkv, data["positions"], data["cache_loc"], data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"], q8,
                     hq, hkv, cfg["rope_theta"], cfg["qk_scale_factor"], A__invf(cfg["rope_theta"], dev))
    seq = data["seq_lens"].to(torch.int32)
    csl = data.get("cache_seqlens")
    if csl is None:
        csl = (seq + L).to(torch.int32)
    cuq = data.get("cu_seqlens_q")
    if cuq is None:
        cuq = torch.arange(0, B * L + 1, L, dtype=torch.int32, device=dev)
    desc = {}
    if float(cfg["kv_scale"]) != 1.0:
        desc = dict(k_descale=data["k_scale"].reshape(1, 1).expand(B, hkv), v_descale=data["v_scale"].reshape(1, 1).expand(B, hkv))
    win = cfg["window"]
    o = A__fa3()(q=q8.view(torch.float8_e4m3fn), k_cache=data["k_cache"], v_cache=data["v_cache"], page_table=data["page_table"],
               cache_seqlens=csl, cu_seqlens_q=cuq, max_seqlen_q=L, softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"], causal=True,
               window_size=(win - 1, 0) if win else (-1, -1), scheduler_metadata=data.get("scheduler_metadata"), **desc)
    if isinstance(o, tuple):
        o = o[0]
    x = o.float().view(B * L, hq, 64)
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
    g = qkv.reshape(B * L, -1)[:, (hq + 2 * hkv) * 64:].float().view(B * L, hq, 64)
    return (x * torch.sigmoid(g)).to(torch.bfloat16).view(B, L, hq * 64)


def A_kernel_fn(data):
    global A_DIAG
    if A_DIAG:
        A_DIAG = False
        A__diag(data["qkv"].device)
    cfg = data["config"]
    B, L, hq, hkv = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"]
    qkv = data["qkv"]
    dev = qkv.device
    if L == 1 and hq == 16 and hkv == 2 and cfg["head_dim"] == 64:  # fused single-launch decode
        maxkeys = data["page_table"].shape[1] * cfg["page_size"]
        span = min(maxkeys, cfg["window"] + 1) if cfg["window"] else maxkeys
        NS = max(1, min(132 // B, (span + A__NSDIV - 1) // A__NSDIV))  # keys per CTA >= _NSDIV, one wave
        if A__hopper() and NS <= 17 and B * min(NS, 16) <= 64:  # one cluster per request: split merge in distributed smem
            NS = 1 << (min(NS, 16).bit_length() - 1)
        NS = A_OVR.get(("d", B, bool(cfg["window"])), NS)
        key = ("f", B, NS)
        if key not in A__BUF:
            A__BUF[key] = (torch.empty((B * NS * 16 * 64,), dtype=torch.float32, device=dev),
                         torch.empty((B * NS * 16 * 2,), dtype=torch.float32, device=dev),
                         torch.zeros((2 * B,), dtype=torch.int32, device=dev))
        wo, wml, cnt = A__BUF[key]
        out = torch.empty((B, 1, hq * 64), dtype=torch.bfloat16, device=dev)
        ks = float(cfg["kv_scale"])
        A__ext().run_fdec(qkv, data["cache_loc"], data["k_cache"], data["v_cache"], data["page_table"], data["seq_lens"], wo, wml, cnt,
                        out, cfg["page_size"], cfg["window"], cfg["rope_theta"], cfg["qk_scale_factor"], cfg["sm_scale"] * ks, ks, NS,
                        torch.empty(0, device=dev), A__invf(cfg["rope_theta"], dev))
        return out
    if L > 8 and A_FA3PF and A__fa3() is not None and A__hopper():  # prefill: our prologue (norm, RoPE, cache store), FA3, our norm + gate
        return A__prefill_fa3(data, cfg, B, L, hq, hkv, qkv, dev)
    ps = cfg["page_size"]
    maxkeys = data["page_table"].shape[1] * ps
    span = min(maxkeys, cfg["window"] + L) if cfg["window"] else maxkeys
    # ~2K warps in flight, splits of at least 64 keys (shape-only: lengths are data)
    if L == 1 and hkv == 2 and hq == 16:    # decode: a warp per (request, split), >= 64 keys each
        nsplit = max(1, min((span + 63) // 64, (132 * 8 + B - 1) // B))
    elif L > 8 and hkv == 2 and hq == 16:  # prefill: 8-token query blocks, splits to fill ~2 CTAs per SM
        nqb = (L + 7) // 8
        nsplit = max(1, min(8, (264 + B * nqb - 1) // (B * nqb), (span + 511) // 512))
    elif L == 8 and hkv == 2 and hq == 16:  # verify: a CTA per (request, split), ~2 CTAs per SM
        nsplit = max(1, min((span + 127) // 128, (132 * 3) // B)) if not A__hopper() else max(1, min((span + 255) // 256, (132 * 2) // B))
    else:
        nsplit = max(1, min((span + 127) // 128, (132 * 8 + B * hkv - 1) // (B * hkv)))
    nsplit = A_OVR.get(("v" if L == 8 else "o", B, bool(cfg["window"])), nsplit)
    rows = L * (hq // hkv)
    nqb = (L + 7) // 8
    key = (B, L, hq, hkv, nsplit)
    if key not in A__BUF:
        A__BUF[key] = (torch.empty((B * L * hq * 64,), dtype=torch.float16, device=dev),
                     torch.empty((B * nqb * hkv * nsplit * 64 * 64,), dtype=torch.float32, device=dev),
                     torch.empty((B * nqb * hkv * nsplit * 64 * 2,), dtype=torch.float32, device=dev),
                     torch.zeros((2 * B * nqb,), dtype=torch.int32, device=dev))
    qbuf, wo, wml, cnt = A__BUF[key]
    out = torch.empty((B, L, hq * 64), dtype=torch.bfloat16, device=dev)
    ks = float(cfg["kv_scale"])
    # k_scale enters the logits: fold it into the query scale (qk_scale_factor already in q)
    A__ext().run(qkv, data["positions"], data["cache_loc"], data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"],
               data["page_table"], data["seq_lens"], qbuf, wo, wml, out, hq, hkv, ps, cfg["window"], cfg["rope_theta"],
               cfg["qk_scale_factor"], cfg["sm_scale"] * ks, ks, nsplit, A__invf(cfg["rope_theta"], dev), cnt)
    return out

"""One persistent kernel: routing -> W13 (SwiGLU fused) -> W2 (+post-norm stats) -> norm + weighted sum.
Weights repacked per 16-row tile into mma.sync A-fragment order with a bit layout that dequantizes
NVFP4 -> BF16 exactly in 11 integer ops per 8 values; streamed by TMA bulk copies (cp.async fallback)."""
from typing import TypedDict
import torch
from torch.utils.cpp_extension import load_inline


class Weights(TypedDict):
    w13: torch.Tensor
    w13_sf: torch.Tensor
    w13_gs: torch.Tensor
    w2: torch.Tensor
    w2_sf: torch.Tensor
    w2_gs: torch.Tensor
    post_norm_w: torch.Tensor


class Config(TypedDict):
    bs: int
    d_hidden: int
    d_expert: int
    n_experts: int
    local_start: int
    local_count: int
    top_k: int
    sharding: str


class Inputs(TypedDict):
    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    config: Config


M_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>

#define HD 4096
#define NLOC 32
#define STG 4608
#define SPT 8
#define T1E 512
#define T2E 256
#ifndef NTM
#define NTM 4
#endif
#ifndef SCR
#define SCR 1024
#endif
#define KSM 4
#ifndef KSF
#define KSF 0
#endif
#ifndef EARLYREL
#define EARLYREL 1
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define TMA 1
#else
#define TMA 0
#endif

struct P {
  const uint8_t* w1; const uint8_t* w2; const float* g1; const float* g2;
  const __nv_bfloat16* x; const int* ids; const float* tw; const float* nw;
  __nv_bfloat16* inter; __nv_bfloat16* y; float* ss; __nv_bfloat16* out; int* ctr; int* tcnt; float4* ws;
  int M, lo; unsigned long long* prof;
};
#ifdef PROF
__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
#define PSET(k, v) do { if (lane == 0) p.prof[gw * 16 + (k)] = (v); } while (0)
#define PADD(k, v) do { if (lane == 0) p.prof[gw * 16 + (k)] += (v); } while (0)
#else
#define PSET(k, v)
#define PADD(k, v)
#endif

__device__ __forceinline__ void cp16(uint32_t s, const void* g) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(g)); }
__device__ __forceinline__ void cpc() { asm volatile("cp.async.commit_group;\n"); }
template <int N> __device__ __forceinline__ void cpw() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }
__device__ __forceinline__ uint4 lds(uint32_t a) { uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a)); return v; }
__device__ __forceinline__ uint32_t bmul(uint32_t a, uint32_t b) { uint32_t d; asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(0x80008000u)); return d; }
__device__ __forceinline__ void mma(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ int ldacq(const int* p) { int v; asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void fence_ar() { asm volatile("fence.acq_rel.gpu;\n" ::: "memory"); }
__device__ __forceinline__ void gbar(int id, int n) { asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ uint32_t rotl(uint32_t q, int d) { return __funnelshift_l(q, q, d); }
__device__ __forceinline__ uint32_t sc2(uint32_t w, uint32_t sel) { return __byte_perm(w, 0, sel) * 16u + 0x77007700u; }
#if TMA
__device__ __forceinline__ void mb_init(uint32_t b) { asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(b)); }
__device__ __forceinline__ void tma_ld(uint32_t dst, const void* src, uint32_t bar) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(bar), "n"(STG) : "memory");
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" :: "r"(dst), "l"(src), "n"(STG), "r"(bar) : "memory");
}
__device__ __forceinline__ void mb_wait(uint32_t bar, uint32_t ph) {
  asm volatile("{\n.reg .pred p;\nW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra W;\n}\n" :: "r"(bar), "r"(ph) : "memory");
}
#endif

#ifndef HEAVY1
#define HEAVY1 1
#endif
#define QI (NLOC + 2)  // ctr[QI]: dynamic tile queue
#define RED (2 * 2 * 32 * 16)  // per-warp split-K partial slots: 2 tile parities x NT <= 2 float4 per lane
struct Sm { int cnt[NLOC], off[NLOC], fill[NLOC], act[NLOC], na, ks, maxc; int ids[16][4]; };

// Work = tiles from one global queue (all W13 tiles, then all W2 tiles, expert-major). A group of ks warps of one CTA
// takes a tile together, each streaming SPT/ks K-stages; partials meet in smem; the finisher rotates per tile.
template <int S>
struct Ctx {
  const P* p; Sm* sm; const uint16_t* list; uint32_t ring, bars; int lane, N1, N, gi, gj, ks;
  int q; unsigned long long wcyc;
  int pi, pp, ps;  // producer: group tile index, pass, stage
  __device__ __forceinline__ int tid_of(int i) const { return ((volatile int*)sm->ids[gi])[i & 3]; }
  __device__ __forceinline__ void item(int u, int& g2, int& e, int& mt) const {
    if (u < N1) { g2 = 0; e = sm->act[u >> 9]; mt = u & 511; }
    else { int v = u - N1; g2 = 1; e = sm->act[v >> 8]; mt = v & 255; }
  }
  __device__ __forceinline__ void issue(int slot) {
    const int u = tid_of(pi);
    if (u < N) {
      int g2, e, mt; item(u, g2, e, mt);
      const int span = SPT / ks, s = gj * span + ps;
      const uint8_t* src = g2 ? p->w2 + ((size_t)((p->lo + e) * T2E + mt) * SPT + s) * STG
                              : p->w1 + ((size_t)((p->lo + e) * T1E + mt) * SPT + s) * STG;
#if TMA
      if (lane == 0) tma_ld(ring + slot * STG, src, bars + slot * 8);
#else
      uint32_t d = ring + slot * STG + lane * 16; src += lane * 16;
#pragma unroll
      for (int k = 0; k < 9; k++) cp16(d + k * 512, src + k * 512);
#endif
      if (++ps == span) { ps = 0; if (++pp == (sm->cnt[e] + 8 * NTM - 1) / (8 * NTM)) { pp = 0; pi++; } }
    }
#if !TMA
    cpc();
#endif
  }
  __device__ __forceinline__ uint32_t acquire() {  // wait for stage q, refill the slot freed last iteration
    const int slot = q % S;
#ifdef PROF
    long long c0 = clock64();
#endif
#if TMA
    mb_wait(bars + slot * 8, (q / S) & 1);
    __syncwarp();
#else
    cpw<S - 2>();
#endif
#ifdef PROF
    wcyc += clock64() - c0;
#endif
#if !EARLYREL
    issue((q + S - 1) % S);
#endif
    q++;
    return ring + slot * STG + lane * 16;
  }
  __device__ __forceinline__ void release() {  // stage q - 1 consumed: refill its slot now, before any epilogue
#if EARLYREL
    __syncwarp();
    issue((q - 1) % S);
#endif
  }
};

template <int NT, int G2, int S>
__device__ __forceinline__ bool run(Ctx<S>& C, int e, int mt, int pass, int ti, int s0, int s1) {
  const P& p = *C.p;
  const int lane = C.lane, g = lane >> 2, t = lane & 3;
  const int cnt = C.sm->cnt[e], off = C.sm->off[e];
  constexpr int CH = NT == 1 ? 4 : (NT == 2 ? 2 : 1);
#ifndef BD1
#define BD1 7
#endif
  constexpr int D = NT == 1 ? BD1 : (NT == 2 ? 3 : 1);  // B prefetch distance in chunks; (D + 1) divides 8 so slots stay static
  const uint4* rp[NT];
#pragma unroll
  for (int n = 0; n < NT; n++) {
    int i = pass * 8 * NTM + n * 8 + g; int r = C.list[off + (i < cnt ? i : 0)];
    rp[n] = (const uint4*)((G2 ? p.inter + (size_t)r * HD : p.x + (size_t)(r >> 3) * HD) + t * 16);
  }
  uint4 B[D + 1][NT][2];
  auto ldb = [&](uint4 (&b)[NT][2], int s, int c) {
#pragma unroll
    for (int n = 0; n < NT; n++) {
      const uint4* q = rp[n] + s * 64 + c * 8;
      if (G2) { b[n][0] = __ldca(q); b[n][1] = __ldca(q + 1); } else { b[n][0] = __ldg(q); b[n][1] = __ldg(q + 1); }
    }
  };
  float acc[NT][CH][4];
#pragma unroll
  for (int n = 0; n < NT; n++)
#pragma unroll
    for (int h = 0; h < CH; h++) acc[n][h][0] = acc[n][h][1] = acc[n][h][2] = acc[n][h][3] = 0.f;
#pragma unroll
  for (int c = 0; c < D; c++) ldb(B[c], s0, c);
  for (int s = s0; s < s1; s++) {
    const uint32_t slot = C.acquire();
    const uint4 sc = lds(slot + 4096);
#pragma unroll
    for (int c = 0; c < 8; c++) {
      const uint4 w = lds(slot + c * 512);
      const uint32_t sw = (c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w));
      const uint32_t sA = sc2(sw, (c & 1) ? 0x4242 : 0x4040), sB = sc2(sw, (c & 1) ? 0x4343 : 0x4141);
      if (c + D < 8) ldb(B[(c + D) % (D + 1)], s, c + D);
      else if (s + 1 < s1) ldb(B[(c + D) % (D + 1)], s + 1, c + D - 8);
      const uint4 (&bc)[NT][2] = B[c % (D + 1)];
      const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const uint32_t q = ww[j];
#ifdef FAKEDQ
        const uint32_t a0 = q ^ sA, a1 = q ^ sB, a2 = q + sA, a3 = q + sB;
#else
        const uint32_t a0 = bmul(q & 0x81C081C0u, sA);
        const uint32_t a1 = bmul((rotl(q, 3) & 0x01C001C0u) | (rotl(q, 1) & 0x80008000u), sB);
        const uint32_t a2 = bmul(rotl(q, 6) & 0x81C081C0u, sA);
        const uint32_t a3 = bmul((rotl(q, 11) & 0x01C001C0u) | (rotl(q, 5) & 0x80008000u), sB);
#endif
#pragma unroll
        for (int n = 0; n < NT; n++) {
          const uint32_t* bb = (const uint32_t*)&bc[n][j >> 1];
#ifdef NOMMA
          acc[n][j % CH][0] += __uint_as_float(a0 ^ a1 ^ a2 ^ a3 ^ bb[(j & 1) * 2] ^ bb[(j & 1) * 2 + 1]);
#else
          mma(acc[n][j % CH], a0, a1, a2, a3, bb[(j & 1) * 2], bb[(j & 1) * 2 + 1]);
#endif
        }
      }
    }
    C.release();
  }
#ifdef PROF
  const long long ce0 = clock64();
  struct EpiT { long long c; int lane; unsigned long long* pf; int gw; __device__ ~EpiT() { if (lane == 0 && pf) pf[gw * 16 + 15] += clock64() - c; } } epit{ce0, lane, p.prof, (int)(blockIdx.x * WPC + (threadIdx.x >> 5))};
#endif
  extern __shared__ __align__(16) uint8_t smem[];
  float f[NT][4];
#pragma unroll
  for (int n = 0; n < NT; n++)
#pragma unroll
    for (int i = 0; i < 4; i++) { float v = 0.f;
#pragma unroll
      for (int h = 0; h < CH; h++) v += acc[n][h][i];
      f[n][i] = v; }
  if (NT <= 2 && C.ks > 1) {  // group split-K: partials meet in smem (double-buffered by tile parity)
    const int wid = threadIdx.x >> 5, ks = C.ks, w0 = wid - C.gj;
    float4* red = (float4*)(smem + (size_t)WPC * S * STG + WPC * SCR + WPC * S * 8) + (ti & 1) * WPC * 64;
#pragma unroll
    for (int n = 0; n < NT; n++) red[(wid * 2 + n) * 32 + lane] = make_float4(f[n][0], f[n][1], f[n][2], f[n][3]);
    gbar(1 + C.gi, ks * 32);
    if (C.gj != (ti & (ks - 1))) return false;
    for (int o = w0; o < w0 + ks; o++) {
      if (o == wid) continue;
#pragma unroll
      for (int n = 0; n < NT; n++) { float4 v = red[(o * 2 + n) * 32 + lane]; f[n][0] += v.x; f[n][1] += v.y; f[n][2] += v.z; f[n][3] += v.w; }
    }
  }
  __nv_bfloat16* scr = (__nv_bfloat16*)(smem + (size_t)(blockDim.x >> 5) * S * STG + (threadIdx.x >> 5) * SCR);
  if (!G2) {
    const float gs = p.g1[p.lo + e];
#pragma unroll
    for (int n = 0; n < NT; n++)
#pragma unroll
      for (int h = 0; h < 2; h++) {
        float gt = f[n][h] * gs, up = f[n][2 + h] * gs;
        scr[(n * 8 + 2 * t + h) * 8 + g] = __float2bfloat16(gt / (1.f + __expf(-gt)) * up);
      }
    __syncwarp();
    for (int i = lane; i < NT * 8; i += 32) {
      int ii = pass * 8 * NTM + i;
      if (ii < cnt) { int r = C.list[off + ii]; *(uint4*)(p.inter + (size_t)r * HD + mt * 8) = *(uint4*)(scr + i * 8); }
    }
  } else {
    const float gs = p.g2[p.lo + e];
    const float n0 = p.nw[mt * 16 + g] * gs, n1 = p.nw[mt * 16 + g + 8] * gs;
#pragma unroll
    for (int n = 0; n < NT; n++) {
      float v0 = f[n][0] * n0, v1 = f[n][1] * n0, v2 = f[n][2] * n1, v3 = f[n][3] * n1;
      scr[(n * 8 + 2 * t) * 16 + g] = __float2bfloat16(v0);
      scr[(n * 8 + 2 * t + 1) * 16 + g] = __float2bfloat16(v1);
      scr[(n * 8 + 2 * t) * 16 + g + 8] = __float2bfloat16(v2);
      scr[(n * 8 + 2 * t + 1) * 16 + g + 8] = __float2bfloat16(v3);
      float sa = v0 * v0 + v2 * v2, sb = v1 * v1 + v3 * v3;
#pragma unroll
      for (int o = 4; o < 32; o <<= 1) { sa += __shfl_xor_sync(0xffffffffu, sa, o); sb += __shfl_xor_sync(0xffffffffu, sb, o); }
      if (g == 0) {
        int i0 = pass * 8 * NTM + n * 8 + 2 * t;
        if (i0 < cnt) atomicAdd(p.ss + C.list[off + i0], sa);
        if (i0 + 1 < cnt) atomicAdd(p.ss + C.list[off + i0 + 1], sb);
      }
    }
    __syncwarp();
    for (int i = lane; i < NT * 16; i += 32) {
      int ii = pass * 8 * NTM + (i >> 1);
      if (ii < cnt) { int r = C.list[off + ii]; *(uint4*)(p.y + (size_t)r * HD + mt * 16 + (i & 1) * 8) = *(uint4*)(scr + i * 8); }
    }
  }
  __syncwarp();
  return true;
}

template <int G2, int S>
__device__ __forceinline__ bool dispatch(Ctx<S>& C, int nt, int e, int mt, int pass, int ti, int s0, int s1) {
  if (nt <= 1) return run<1, G2, S>(C, e, mt, pass, ti, s0, s1);
  if (nt <= 2) return run<2, G2, S>(C, e, mt, pass, ti, s0, s1);
#if NTM >= 8
  if (nt > 4) return run<8, G2, S>(C, e, mt, pass, ti, s0, s1);
#endif
  return run<4, G2, S>(C, e, mt, pass, ti, s0, s1);
}

__global__ void __launch_bounds__(WPC * 32, 1) moe_kernel(P p) {
  constexpr int S = NSTG;
  extern __shared__ __align__(16) uint8_t smem[];
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5, R = p.M * 8;
  const bool small = p.M <= 64;  // split-K groups (and their smem) only for small batches
  Sm* sm = (Sm*)(smem + WPC * S * STG + WPC * SCR + WPC * S * 8 + (small ? WPC * RED : 0));
  uint16_t* list = (uint16_t*)(sm + 1);
  uint8_t* lid = (uint8_t*)(list + R);
  const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem);
  const int gw = blockIdx.x * WPC + wid;
#ifdef PROF
  if (lane == 0) { for (int k = 0; k < 16; k++) p.prof[gw * 16 + k] = 0; unsigned sid; asm("mov.u32 %0, %smid;" : "=r"(sid)); p.prof[gw * 16 + 10] = sid; }
  PSET(0, gtime()); PSET(13, clock64());
#endif
#if TMA
  if (lane < S) mb_init(sbase + WPC * S * STG + WPC * SCR + (wid * S + lane) * 8);
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#endif
  if (tid < NLOC) sm->cnt[tid] = 0;
  __syncthreads();
  for (int i = tid; i < R; i += blockDim.x) { unsigned e = p.ids[i] - p.lo; lid[i] = e < NLOC ? e : 0xff; if (e < NLOC) atomicAdd(&sm->cnt[e], 1); }
  __syncthreads();
  if (wid == 0) {
    int c = sm->cnt[lane], v = c;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) { int n = __shfl_up_sync(0xffffffffu, v, o); if (lane >= o) v += n; }
    sm->off[lane] = v - c; sm->fill[lane] = v - c;
    unsigned m = __ballot_sync(0xffffffffu, c > 0);
#if HEAVY1
    {  // active experts heaviest first: their multi-pass tiles start early instead of forming the tail
      int rk = 0;
#pragma unroll
      for (int j = 0; j < 32; j++) { const int cj = __shfl_sync(0xffffffffu, c, j); rk += (cj > c) || (cj == c && j < lane); }
      if (c > 0) sm->act[rk] = lane;
    }
#else
    if (c > 0) sm->act[__popc(m & ((1u << lane) - 1))] = lane;
#endif
    int mx = c;
#pragma unroll
    for (int o = 16; o; o >>= 1) mx = max(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    if (lane == 0) { sm->na = __popc(m); sm->maxc = mx; const int tw = __popc(m) * (T1E + T2E) / (int)(gridDim.x * WPC / 16);  // tiles per 16 warps
      sm->ks = (small && mx <= 16) ? (KSF ? KSF : (tw < 32 ? 4 : (tw < 96 ? 2 : 1))) : 1; }
  }
  __syncthreads();
  for (int i = tid; i < R; i += blockDim.x) { unsigned e = lid[i]; if (e < NLOC) list[atomicAdd(&sm->fill[e], 1)] = i; }
  const int ks = sm->ks, na = sm->na;
  Ctx<S> C;
  C.p = &p; C.sm = sm; C.list = list; C.lane = lane; C.ks = ks; C.gi = wid / ks; C.gj = wid % ks;
  C.N1 = na * T1E; C.N = C.N1 + na * T2E;
  C.ring = sbase + wid * S * STG; C.bars = sbase + WPC * S * STG + WPC * SCR + wid * S * 8; C.q = 0; C.wcyc = 0;
  C.pi = 0; C.pp = 0; C.ps = 0;
  const bool lead = C.gj == 0 && lane == 0;
  if (lead) { sm->ids[C.gi][0] = atomicAdd(p.ctr + QI, 1); sm->ids[C.gi][1] = atomicAdd(p.ctr + QI, 1); }
  __syncthreads();
#ifdef PROF
  PSET(1, gtime());
#endif
#pragma unroll
  for (int i = 0; i < S - 1 + EARLYREL; i++) C.issue(i);
  for (int i = 0;; i++) {
    const int u = C.tid_of(i);
    if (u >= C.N) break;
    int nxt = 0;
    if (lead) nxt = atomicAdd(p.ctr + QI, 1);  // tile for i + 2; latency hidden behind this tile
    int g2, e, mt; C.item(u, g2, e, mt);
    const int cnt = sm->cnt[e], np = (cnt + 8 * NTM - 1) / (8 * NTM);
#ifdef PROF
    long long c0 = clock64();
#endif
    if (g2) {
      if (lane == 0) while (ldacq(p.ctr + e) < T1E) __nanosleep(32);
      __syncwarp();
#ifdef PROF
      PADD(4, clock64() - c0); c0 = clock64();
#endif
    } else if (mt == 0 && C.gj == 0) {
      for (int k = lane; k < cnt; k += 32) p.ss[list[sm->off[e] + k]] = 0.f;
    }
    if (lead) sm->ids[C.gi][(i + 2) & 3] = nxt;  // read 2 tiles later, after the group barrier (ks > 1) / syncwarp
    __syncwarp();
    const int span = SPT / ks, s0 = C.gj * span;
    bool fin = false;
    for (int ps = 0; ps < np; ps++) {
      int nt = min(NTM, (cnt - ps * 8 * NTM + 7) >> 3);
      fin = g2 ? dispatch<1, S>(C, nt, e, mt, ps, i, s0, s0 + span) : dispatch<0, S>(C, nt, e, mt, ps, i, s0, s0 + span);
    }
    if (fin) {
      fence_ar(); __syncwarp();
      if (lane == 0) atomicAdd(p.ctr + (g2 ? NLOC : e), 1);
    }
#ifdef PROF
    PADD(g2 ? 3 : 2, clock64() - c0); PADD(8, 1);
#endif
  }
#if !TMA
  cpw<0>();
#endif
  const int T2 = sm->na * T2E;
#ifdef PROF
  PSET(5, gtime()); PSET(9, C.wcyc); PSET(11, sm->na); PSET(12, sm->ks);
#endif
  if (lane == 0) while (ldacq(p.ctr + NLOC) < T2) __nanosleep(32);
  __syncwarp();
#ifdef PROF
  PSET(6, gtime());
#endif
  for (int it = gw; it < p.M * 16; it += gridDim.x * WPC) {
    const int t = it >> 4, col = (it & 15) * 256 + lane * 8;
    float a[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int k = 0; k < 8; k++) {
      const int r = t * 8 + k;
      unsigned e = p.ids[r] - p.lo;
      if (e < NLOC) {
        float cf = p.tw[r] * rsqrtf(__ldcg(p.ss + r) * (1.f / HD) + 1e-8f);
        uint4 v = __ldcg((const uint4*)(p.y + (size_t)r * HD + col));
        const __nv_bfloat162* h = (const __nv_bfloat162*)&v;
#pragma unroll
        for (int j = 0; j < 4; j++) { float2 f = __bfloat1622float2(h[j]); a[2 * j] += cf * f.x; a[2 * j + 1] += cf * f.y; }
      }
    }
    uint4 o; __nv_bfloat162* ob = (__nv_bfloat162*)&o;
#pragma unroll
    for (int j = 0; j < 4; j++) ob[j] = __floats2bfloat162_rn(a[2 * j], a[2 * j + 1]);
    *(uint4*)(p.out + (size_t)t * HD + col) = o;
  }
#ifdef PROF
  PSET(7, gtime()); PSET(14, clock64());
#endif
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    if (atomicAdd(p.ctr + NLOC + 1, 1) == (int)gridDim.x - 1) {
      for (int i = 0; i < NLOC + 3; i++) p.ctr[i] = 0;
      __threadfence();
    }
  }
}

static int g_grid = 0;
static size_t smem_bytes(int M) { return (size_t)WPC * NSTG * STG + WPC * SCR + WPC * NSTG * 8 + (M <= 64 ? WPC * RED : 0) + sizeof(Sm) + (size_t)M * 8 * 3; }

void moe(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor w1, torch::Tensor w2, torch::Tensor g1,
         torch::Tensor g2, torch::Tensor nw, torch::Tensor inter, torch::Tensor y, torch::Tensor ss, torch::Tensor ctr,
         torch::Tensor tcnt, torch::Tensor ws, torch::Tensor out, int64_t lo, torch::Tensor prof) {
  int M = x.size(0);
  if (!g_grid) {
    int dev; cudaGetDevice(&dev); int mx; cudaDeviceGetAttribute(&mx, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaDeviceGetAttribute(&g_grid, cudaDevAttrMultiProcessorCount, dev);
    cudaFuncSetAttribute(moe_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, mx);
  }
  P p{w1.data_ptr<uint8_t>(), w2.data_ptr<uint8_t>(), g1.data_ptr<float>(), g2.data_ptr<float>(),
      (const __nv_bfloat16*)x.data_ptr(), ids.data_ptr<int>(), tw.data_ptr<float>(), nw.data_ptr<float>(),
      (__nv_bfloat16*)inter.data_ptr(), (__nv_bfloat16*)y.data_ptr(), ss.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr(),
      ctr.data_ptr<int>(), tcnt.data_ptr<int>(), (float4*)ws.data_ptr(), M, (int)lo,
      prof.numel() ? (unsigned long long*)prof.data_ptr() : nullptr};
  moe_kernel<<<g_grid, WPC * 32, smem_bytes(M), at::cuda::getCurrentCUDAStream()>>>(p);
}
"""

M__mod = None
M__DBG = []
# pair p of each 32-bit word: (eem rotation, sign rotation); bf16 targets m->6, e0->7, e1->8, s->15 (+16 for the high value)
M__ROT = ((0, 0), (3, 1), (6, 6), (11, 5))


def M__big():
    pr = torch.cuda.get_device_properties(0)
    sm = getattr(pr, "shared_memory_per_block_optin", None)
    return sm >= 180000 if sm else (pr.major, pr.minor) in ((9, 0), (10, 0))


M__mod8 = None
M_LARGE = True  # M > 64: 8 warps per SM with 4-stage rings and up to 64 tokens per pass (one dequant per stage)


def M__ext(large=False):
    global M__mod, M__mod8
    if (M__mod8 if large else M__mod) is None:
        big = M__big()
        wpc, stg = (16, 2) if big else (8, 2)
        extra = []
        if large:
            wpc, stg, extra = 8, (4 if big else 2), ["-DNTM=8", "-DSCR=2048"]
        for d in M__DBG:
            if d.startswith("-DCFG=") and not large: wpc, stg = map(int, d[6:].split("x"))
        m = load_inline("moe_w4a16_v14_%d_%d%s%s" % (wpc, stg, "_L" if large else "", "".join(M__DBG).replace("-", "_").replace("=", "")),
                        cpp_sources="void moe(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,torch::Tensor);",
                        cuda_sources=M_CUDA, functions=["moe"],
                        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo", "-DWPC=%d" % wpc, "-DNSTG=%d" % stg] + extra + [d for d in M__DBG if not d.startswith("-DCFG=")], verbose=False)
        if large: M__mod8 = m
        else: M__mod = m
    return M__mod8 if large else M__mod


def M__place(word, code, pair, hi):
    de, ds = M__ROT[pair]
    for bit, tgt, d in ((0, 6, de), (1, 7, de), (2, 8, de), (3, 15, ds)):
        word |= ((code >> bit) & 1) << ((tgt + 16 * hi - d) % 32)
    return word


def M__repack(w, sf, perm):
    """w uint8 [R, K/2], sf e4m3 [R, K/16] -> uint8 [R/16, K/512, 4608] in mma A-fragment order."""
    R, K = w.shape[0], w.shape[1] * 2
    MT, NS = R // 16, K // 512
    codes = torch.stack((w & 15, w >> 4), -1).view(R, K)[perm].to(torch.int64)
    s8 = sf.view(torch.uint8)[perm].to(torch.int64)
    neg = (s8 >> 7).repeat_interleave(16, 1)  # fold a negative block scale into the codes' signs (same values)
    codes = codes ^ (neg << 3)
    zero = ((s8 & 0x7F) == 0).repeat_interleave(16, 1)  # zero scale: every value is 0; keep it 0 with a normal scale
    codes = torch.where(zero, torch.zeros_like(codes), codes)
    s8 = torch.where((s8 & 0x7F) == 0, torch.full_like(s8, 0x08), s8 & 0x7F)
    if ((s8 >> 3) == 0).any():
        raise ValueError("subnormal E4M3 block scales are not supported by this fast path")
    C = codes.view(MT, 16, NS, 8, 4, 4, 4)  # mt, i(row), s, c, t, j, e
    a, b = C[:, :8], C[:, 8:]  # rows g, g+8
    word = torch.zeros(a.shape[:-1], dtype=torch.int64, device=w.device)
    for pair, (src, e0) in enumerate(((a, 0), (b, 0), (a, 2), (b, 2))):
        word = M__place(word, src[..., e0], pair, 0)
        word = M__place(word, src[..., e0 + 1], pair, 1)
    word = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32)
    word = word.permute(0, 2, 3, 1, 4, 5).contiguous().view(torch.uint8).view(MT, NS, 4096)
    S = s8.to(torch.uint8).view(MT, 16, NS, 8, 4)
    sc = torch.stack((S[:, :8], S[:, 8:]), -1).permute(0, 2, 1, 4, 3, 5).contiguous().view(MT, NS, 512)
    return torch.cat((word, sc), -1).contiguous()


def M_prepare(weights):
    E, R1 = weights["w13"].shape[0], weights["w13"].shape[1]
    I = R1 // 2
    dev = weights["w13"].device
    perm1 = torch.stack((torch.arange(I, device=dev).view(-1, 8), I + torch.arange(I, device=dev).view(-1, 8)), 1).reshape(-1)
    perm2 = torch.arange(weights["w2"].shape[1], device=dev)
    w1 = torch.empty((E, R1 // 16, 8, 4608), dtype=torch.uint8, device=dev)
    w2 = torch.empty((E, weights["w2"].shape[1] // 16, 8, 4608), dtype=torch.uint8, device=dev)
    for e in range(E):
        w1[e] = M__repack(weights["w13"][e], weights["w13_sf"][e], perm1)
        w2[e] = M__repack(weights["w2"][e], weights["w2_sf"][e], perm2)
    M__ext(); M__ext(M_LARGE)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    st = {"w1": w1, "w2": w2, "g1": weights["w13_gs"].float() * 256.0, "g2": weights["w2_gs"].float() * 256.0,
            "nw": 1.0 + weights["post_norm_w"].float(), "ctr": torch.zeros(64, dtype=torch.int32, device=dev),
            "tcnt": torch.zeros(32 * 768, dtype=torch.int32, device=dev),
            "prof": torch.zeros((65536 * 16 if "-DPROF" in M__DBG else 0,), dtype=torch.int64, device=dev),
            "ws": torch.empty((32 * 768 * 8 * 64 * 4,), dtype=torch.float32, device=dev)}
    if M_DIAG:
        M__diag(st, weights)
    return st


M_DIAG = False  # diagnostics only: print a per-warp phase timeline from prepare() (untimed)


def M__diag(state, weights):
    global M__mod, M__DBG
    keep, M__mod, M__DBG = (M__mod, M__DBG), None, M__DBG + ["-DPROF"]
    ext = M__ext()
    state = dict(state, prof=torch.zeros((65536 * 16,), dtype=torch.int64, device="cuda"))
    nw = torch.cuda.get_device_properties(0).multi_processor_count * (16 if M__big() else 8)
    g = torch.Generator(device="cuda").manual_seed(7)
    for bs in (1, 8, 32, 256):
        sc = torch.rand((bs, 256), generator=g, device="cuda")
        ids = sc.topk(8, -1).indices
        pr = torch.bincount((ids // 32).flatten(), minlength=8)
        ids = ((ids - 32 * pr.argmax()) % 256).int().contiguous()
        x = torch.randn((bs, 4096), generator=g, device="cuda").bfloat16()
        tw = torch.full((bs, 8), 0.125, device="cuda")
        fl = torch.empty(2**28, device="cuda")
        for _ in range(3):
            fl.add_(1)
            inter = torch.empty((bs * 8, 4096), dtype=torch.bfloat16, device="cuda"); y = torch.empty_like(inter)
            ss = torch.empty((bs * 8,), device="cuda"); out = torch.empty_like(x)
            ext.moe(x, ids, tw, state["w1"], state["w2"], state["g1"], state["g2"], state["nw"], inter, y, ss, state["ctr"],
                    state["tcnt"], state["ws"], out, 0, state["prof"])
        torch.cuda.synchronize()
        P = state["prof"].view(-1, 16)[:nw].cpu().tolist()
        t0 = min(r[0] for r in P)
        q = lambda v, f: sorted(v)[min(len(v) - 1, int(f * len(v)))]
        col = lambda k: [(r[k] - t0) / 1e3 for r in P]
        mhz = [(r[14] - r[13]) / max(1, r[7] - r[0]) * 1e3 for r in P]
        cy = lambda k: [r[k] / q(mhz, .5) for r in P]
        print(f"DIAG bs {bs} na {P[0][11]} mode {P[0][12]} span {max(col(7)):.1f} routed {max(col(1)):.1f} "
              f"done p10/50/max {q(col(5), .1):.1f}/{q(col(5), .5):.1f}/{max(col(5)):.1f} comb {max(col(6)):.1f} mhz {q(mhz, .5):.0f} "
              f"g1 {q(cy(2), .5):.1f}/{max(cy(2)):.1f} g2 {q(cy(3), .5):.1f}/{max(cy(3)):.1f} dep {q(cy(4), .5):.1f}/{max(cy(4)):.1f} "
              f"tma {q(cy(9), .5):.1f}/{max(cy(9)):.1f} segs {max(r[8] for r in P)}", flush=True)
    M__mod, M__DBG = keep


def M_kernel_fn(data, state):
    x = data["hidden_states"].contiguous()
    M = x.shape[0]
    ids = data["topk_ids"].to(torch.int32).contiguous()
    tw = data["topk_weights"].float().contiguous()
    inter = torch.empty((M * 8, 4096), dtype=torch.bfloat16, device=x.device)
    y = torch.empty((M * 8, 4096), dtype=torch.bfloat16, device=x.device)
    ss = torch.empty((M * 8,), dtype=torch.float32, device=x.device)
    out = torch.empty_like(x)
    M__ext(M_LARGE and M > 64).moe(x, ids, tw, state["w1"], state["w2"], state["g1"], state["g2"], state["nw"], inter, y, ss, state["ctr"],
                               state["tcnt"], state["ws"], out, data["config"]["local_start"], state["prof"])
    return out

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


P_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>
#ifndef WPC
#define WPC 16
#endif
#ifndef V7
#define V7 0
#endif
#ifndef RSN
#define RSN 2
#endif
#ifndef CUN
#define CUN 2
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

// Each warp streams an equal contiguous range of (tile, unit) work through its own RS-deep TMA ring; a unit is CU
// consecutive 64-wide K chunks (2 KB of A fragments each; the router's lo half rides along). Partial tiles are
// summed with fp32 red.add into ws; a PDL-launched finalize kernel converts ws to the output and clears it.
template <int NT, bool HL, int RS, int CU>
__global__ void __launch_bounds__(WPC * 32, 1) skinny(Q q) {
  constexpr int SB = (HL ? 2 : 1) * CU * 2048;
  constexpr int CH = NT == 1 ? 4 : (NT == 2 ? 2 : 1);
  extern __shared__ __align__(128) uint8_t smem[];
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int W = gridDim.x * WPC, gw = blockIdx.x * WPC + wid;
  const uint32_t ring = (uint32_t)__cvta_generic_to_shared(smem) + wid * RS * SB;
  const uint32_t bars = (uint32_t)__cvta_generic_to_shared(smem) + WPC * RS * SB + wid * RS * 8;
#if TMA
  if (lane < RS) mbinit(bars + lane * 8, 1);
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");  // let the finalize kernel get scheduled early
#else
  if (lane < RS) mbinit(bars + lane * 8, 32);
#endif
  __syncwarp();
  const int KU = q.KC / CU;
  const long long U = (long long)q.TB * q.MT * KU;
  const long long L0 = U * gw / W, Le = U * (gw + 1) / W;
  if (L0 >= Le) return;
  long long pu = L0;
  auto issue = [&](int slot) {
    if (pu < Le) {
      const long long tile = pu / KU; const int cu = (int)(pu - tile * KU), mt = (int)(tile % q.MT);
      const size_t off = ((size_t)mt * q.KC + cu * CU) * 2048;
      const uint32_t d = ring + slot * SB, bar = bars + slot * 8;
#if TMA
      if (lane == 0) {
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(bar), "n"(SB) : "memory");
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" :: "r"(d), "l"(q.A + off), "n"(CU * 2048), "r"(bar) : "memory");
        if (HL) asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" :: "r"(d + CU * 2048), "l"(q.A2 + off), "n"(CU * 2048), "r"(bar) : "memory");
      }
#else
#pragma unroll
      for (int k = 0; k < 4 * CU; k++) cp16(d + k * 512 + lane * 16, q.A + off + k * 512 + lane * 16);
      if (HL) {
#pragma unroll
        for (int k = 0; k < 4 * CU; k++) cp16(d + CU * 2048 + k * 512 + lane * 16, q.A2 + off + k * 512 + lane * 16);
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
  auto ldb = [&](long long cg, uint4 (&b)[NT][2]) {  // cg: global chunk index tile * KC + c
    const long long tile = cg / q.KC; const int c = (int)(cg - tile * q.KC), tb = (int)(tile / q.MT);
#pragma unroll
    for (int n = 0; n < NT; n++) {
      const int tok = min(tb * 32 + n * 8 + g, q.M - 1);
      const uint4* xr = (const uint4*)(q.x + (size_t)tok * q.K) + t * 2 + c * 8;
      b[n][0] = __ldg(xr); b[n][1] = __ldg(xr + 1);
    }
  };
  const long long CE = Le * CU;  // chunk range end
  ldb(L0 * CU, b0);
  float acc[NT][CH][4];
  int qn = 0;
  long long u = L0;
  while (u < Le) {
    const long long tile = u / KU; const int u0 = (int)(u - tile * KU);
    const int u1 = (int)min((long long)KU, u0 + (Le - u));
#pragma unroll
    for (int n = 0; n < NT; n++)
#pragma unroll
      for (int h = 0; h < CH; h++) acc[n][h][0] = acc[n][h][1] = acc[n][h][2] = acc[n][h][3] = 0.f;
    for (int uu = u0; uu < u1; uu++, u++, qn++) {
      const int slot = qn % RS;
      mbwait(bars + slot * 8, (qn / RS) & 1);
      __syncwarp();
      issue((qn + RS - 1) % RS);
#pragma unroll
      for (int i = 0; i < CU; i++) {
        const long long cg = u * CU + i;
        if (cg + 1 < CE) ldb(cg + 1, b1);
        const uint32_t sa = ring + slot * SB + i * 2048 + lane * 16;
#pragma unroll
        for (int j = 0; j < 4; j++) {
          const uint4 a = lds(sa + j * 512);
          uint4 h; if (HL) h = lds(sa + CU * 2048 + j * 512);
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
    }
    const int tb = (int)(tile / q.MT), mt = (int)(tile - (long long)tb * q.MT), row0 = mt * 16 + g, mrem = min(32, q.M - tb * 32);
    float* w = q.ws + (size_t)tb * 32 * q.N;
#pragma unroll
    for (int n = 0; n < NT; n++) {
      float f[4];
#pragma unroll
      for (int i2 = 0; i2 < 4; i2++) { f[i2] = 0.f;
#pragma unroll
        for (int h = 0; h < CH; h++) f[i2] += acc[n][h][i2]; }
      const int tk = n * 8 + 2 * t;
      if (tk < mrem) { atomicAdd(w + (size_t)tk * q.N + row0, f[0]); atomicAdd(w + (size_t)tk * q.N + row0 + 8, f[2]); }
      if (tk + 1 < mrem) { atomicAdd(w + (size_t)(tk + 1) * q.N + row0, f[1]); atomicAdd(w + (size_t)(tk + 1) * q.N + row0 + 8, f[3]); }
    }
#if V7
    {  // the last warp to add into this tile writes it out and clears ws (no finalize kernel)
      auto wof = [&](long long uu) { return (int)(((uu + 1) * W - 1) / U); };  // warp streaming unit uu
      const long long t0u = tile * KU, t1u = t0u + KU - 1;
      const int nc = wof(t1u) - wof(t0u) + 1;
      __syncwarp();
      int last = 0;
      if (lane == 0) { int v; asm volatile("atom.add.acq_rel.gpu.global.s32 %0, [%1], 1;\n" : "=r"(v) : "l"(q.cnt + tile) : "memory"); last = v == nc - 1; }
      last = __shfl_sync(0xffffffffu, last, 0);
      if (last) {
        for (int e = lane; e < 16 * mrem; e += 32) {
          const int tok = e >> 4, row = mt * 16 + (e & 15);
          if (row < q.N) { float* wp = w + (size_t)tok * q.N + row; put(q, tb * 32 + tok, row, __ldcg(wp)); *wp = 0.f; }
        }
        if (lane == 0) q.cnt[tile] = 0;
      }
    }
#endif
  }
}

// ws (fp32 sums, [M][N]) -> out; ws cleared for the next call. Launched as a programmatic dependent.
__global__ void __launch_bounds__(256) fin(Q q) {
#if TMA
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
  const int n4 = q.M * q.N / 4;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) {
    float4* p = (float4*)q.ws + i;
    const float4 v = __ldcg(p);
    *p = make_float4(0.f, 0.f, 0.f, 0.f);
    if (q.f32) ((float4*)q.out)[i] = v;
    else { __nv_bfloat162 o[2] = {__floats2bfloat162_rn(v.x, v.y), __floats2bfloat162_rn(v.z, v.w)}; ((uint2*)q.out)[i] = *(uint2*)o; }
  }
}

static int g_grid = 0;
template <int NT, bool HL> static void go(const Q& q, cudaStream_t st) {
  constexpr int RS = HL ? 3 : RSN, CU = HL ? 1 : CUN;
  const size_t sm = (size_t)WPC * RS * (HL ? 2 : 1) * CU * 2048 + WPC * RS * 8;
  static bool once = false;
  if (!once) { cudaFuncSetAttribute(skinny<NT, HL, RS, CU>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm); once = true; }
  skinny<NT, HL, RS, CU><<<g_grid, WPC * 32, sm, st>>>(q);
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
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(std::min(g_grid * 4, (q.M * q.N / 4 + 255) / 256)); cfg.blockDim = dim3(256); cfg.stream = st;
  cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[0].val.programmaticStreamSerializationAllowed = 1;
  static int major = 0; if (!major) { int dev; cudaGetDevice(&dev); cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev); }
  cfg.attrs = at; cfg.numAttrs = major >= 9 ? 1 : 0;
  if (!V7) cudaLaunchKernelEx(&cfg, fin, q);
}

// ---------------------------------------------------------------- v6: deep per-CTA TMA ring (stream over 16 KB stages)
// A stage is 8 consecutive 2 KB units (one 16-row tile, 8 K chunks); CTA c streams stages [ST c / G, ST (c + 1) / G)
// through an NS-deep ring filled by a producer warp, so most of a CTA's bytes are in flight at once. Consumer warp w
// takes unit w of every stage; at a tile's end the 8 partials meet in smem, and a tile shared by CTAs is summed in
// ws with fp32 atomics, the last arriver writing the output.
#ifndef NSG
#define NSG 12
#endif
__device__ __forceinline__ void mbarrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(b) : "memory"); }
template <int NT, bool HL>
__global__ void __launch_bounds__(288, 1) gemv(Q q) {
  constexpr int SB = (HL ? 2 : 1) * 16384, NS = HL ? NSG / 2 : NSG, CH = NT == 1 ? 4 : (NT == 2 ? 2 : 1);
  extern __shared__ __align__(128) uint8_t smem[];
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const uint32_t base = (uint32_t)__cvta_generic_to_shared(smem);
  const uint32_t full = base + NS * SB, empty = full + NS * 8;
  float* red = (float*)(smem + NS * SB + NS * 16);
  if (threadIdx.x < NS) { mbinit(full + threadIdx.x * 8, TMA ? 1 : 32); mbinit(empty + threadIdx.x * 8, 8); }
#if TMA
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#endif
  __syncthreads();
  const int SPT = q.KC / 8;  // stages per tile
  const int ST = q.MT * SPT, G = gridDim.x, c = blockIdx.x;
  const int s0 = (int)((long long)ST * c / G), s1 = (int)((long long)ST * (c + 1) / G);
  if (wid == 8) {  // producer
#if !TMA
    for (int i = s0; i < s1; i++) {
      const int k = i - s0, slot = k % NS;
      if (k >= NS) mbwait(empty + slot * 8, ((k / NS) - 1) & 1);
      for (int o = lane * 16; o < 16384; o += 512) {
        cp16(base + slot * SB + o, q.A + (size_t)i * 16384 + o);
        if (HL) cp16(base + slot * SB + 16384 + o, q.A2 + (size_t)i * 16384 + o);
      }
      asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" :: "r"(full + slot * 8) : "memory");
    }
    asm volatile("cp.async.wait_all;\n" ::: "memory");
#endif
    if (lane == 0) {
      for (int i = s0; i < s1 && TMA; i++) {
        const int k = i - s0, slot = k % NS;
        if (k >= NS) mbwait(empty + slot * 8, ((k / NS) - 1) & 1);
#if TMA
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(full + slot * 8), "n"(SB) : "memory");
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" :: "r"(base + slot * SB), "l"(q.A + (size_t)i * 16384), "n"(16384), "r"(full + slot * 8) : "memory");
        if (HL) asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" :: "r"(base + slot * SB + 16384), "l"(q.A2 + (size_t)i * 16384), "n"(16384), "r"(full + slot * 8) : "memory");
#endif
      }
    }
    return;
  }
  const int tb = 0;
  float acc[NT][CH][4];
#pragma unroll
  for (int n = 0; n < NT; n++)
#pragma unroll
    for (int h = 0; h < CH; h++) acc[n][h][0] = acc[n][h][1] = acc[n][h][2] = acc[n][h][3] = 0.f;
  uint4 b0[NT][2];
  auto ldb = [&](int i, uint4 (&b)[NT][2]) {
    const int ch = (i % SPT) * 8 + wid;
#pragma unroll
    for (int n = 0; n < NT; n++) {
      const int tok = min(tb * 32 + n * 8 + g, q.M - 1);
      const uint4* xr = (const uint4*)(q.x + (size_t)tok * q.K) + t * 2 + ch * 8;
      b[n][0] = __ldg(xr); b[n][1] = __ldg(xr + 1);
    }
  };
  if (s0 < s1) ldb(s0, b0);
  for (int i = s0; i < s1; i++) {
    const int k = i - s0, slot = k % NS;
    uint4 b1[NT][2];
    if (i + 1 < s1) ldb(i + 1, b1);
    mbwait(full + slot * 8, (k / NS) & 1);
    const uint32_t sa = base + slot * SB + wid * 2048 + lane * 16;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const uint4 a = lds(sa + j * 512);
      uint4 h; if (HL) h = lds(sa + 16384 + j * 512);
#pragma unroll
      for (int n = 0; n < NT; n++) {
        const uint32_t* bb = (const uint32_t*)&b0[n][j >> 1];
        mma(acc[n][j % CH], a, bb[(j & 1) * 2], bb[(j & 1) * 2 + 1]);
        if (HL) mma(acc[n][(j + 1) % CH], h, bb[(j & 1) * 2], bb[(j & 1) * 2 + 1]);
      }
    }
    __syncwarp();
    if (lane == 0) mbarrive(empty + slot * 8);
#pragma unroll
    for (int n = 0; n < NT; n++) { b0[n][0] = b1[n][0]; b0[n][1] = b1[n][1]; }
    const int tile = i / SPT;
    if (i + 1 == s1 || (i + 1) % SPT == 0) {  // tile ends here (for this CTA): reduce the 8 warps' partials
#pragma unroll
      for (int n = 0; n < NT; n++) {
        float f[4];
#pragma unroll
        for (int i2 = 0; i2 < 4; i2++) { f[i2] = 0.f;
#pragma unroll
          for (int h = 0; h < CH; h++) f[i2] += acc[n][h][i2]; }
        ((float4*)red)[(n * 8 + wid) * 32 + lane] = make_float4(f[0], f[1], f[2], f[3]);
#pragma unroll
        for (int h = 0; h < CH; h++) acc[n][h][0] = acc[n][h][1] = acc[n][h][2] = acc[n][h][3] = 0.f;
      }
      asm volatile("bar.sync 1, 256;\n" ::: "memory");
      const int t0s = tile * SPT, t1s = t0s + SPT;
      const bool own = t0s >= s0 && t1s <= s1;
      const int row0 = tile * 16 + g;
      for (int n = wid; n < NT; n += 8) {
        float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int w2 = 0; w2 < 8; w2++) { const float4 v = ((float4*)red)[(n * 8 + w2) * 32 + lane]; sum.x += v.x; sum.y += v.y; sum.z += v.z; sum.w += v.w; }
        const int tk = n * 8 + 2 * t;
        const float fv[4] = {sum.x, sum.y, sum.z, sum.w};
#pragma unroll
        for (int e = 0; e < 4; e++) {
          const int tok = tk + (e & 1), row = row0 + (e >> 1) * 8;
          if (tok >= q.M || row >= q.N) continue;
          if (own) put(q, tok, row, fv[e]);
          else atomicAdd(q.ws + (size_t)tok * q.N + row, fv[e]);
        }
      }
      if (!own) {
        __threadfence();
        asm volatile("bar.sync 1, 256;\n" ::: "memory");
        __shared__ int lastf;
        if (threadIdx.x == 0) {
          auto cta = [&](int st2) { return (int)(((long long)(st2 + 1) * G - 1) / ST); };  // CTA streaming stage st2
          const int nc = cta(t1s - 1) - cta(t0s) + 1;
          lastf = atomicAdd(q.cnt + tile, 1) == nc - 1;
        }
        asm volatile("bar.sync 1, 256;\n" ::: "memory");
        if (lastf) {
          __threadfence();
          for (int e = threadIdx.x; e < 16 * q.M; e += 256) {
            const int tok = e >> 4, row = tile * 16 + (e & 15);
            if (row < q.N) { float* wp = q.ws + (size_t)tok * q.N + row; put(q, tok, row, __ldcg(wp)); *wp = 0.f; }
          }
          if (threadIdx.x == 0) q.cnt[tile] = 0;
        }
      }
      asm volatile("bar.sync 1, 256;\n" ::: "memory");
    }
  }
}
template <int NT, bool HL> static void go2(const Q& q, cudaStream_t st) {
  constexpr int NS = HL ? NSG / 2 : NSG;
  const size_t sm = (size_t)NS * (HL ? 32768 : 16384) + NS * 16 + 8 * 32 * 16 * NT;
  static bool once = false;
  if (!once) { cudaFuncSetAttribute(gemv<NT, HL>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm); once = true; }
  gemv<NT, HL><<<g_grid, 288, sm, st>>>(q);
}
void run2(torch::Tensor A, torch::Tensor A2, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int64_t N) {
  if (!g_grid) { int dev; cudaGetDevice(&dev); cudaDeviceGetAttribute(&g_grid, cudaDevAttrMultiProcessorCount, dev); }
  Q q;
  q.A = (const uint8_t*)A.data_ptr(); q.A2 = A2.numel() ? (const uint8_t*)A2.data_ptr() : nullptr;
  q.x = (const __nv_bfloat16*)x.data_ptr(); q.out = out.data_ptr(); q.ws = ws.data_ptr<float>(); q.cnt = cnt.data_ptr<int>();
  q.M = x.size(0); q.K = x.size(1); q.N = (int)N; q.MT = (int)((N + 15) / 16); q.KC = q.K / 64; q.TB = 1;
  q.f32 = out.scalar_type() == torch::kFloat32;
  auto st = at::cuda::getCurrentCUDAStream();
  const int nt = (q.M + 7) / 8;
  if (q.A2) { if (nt <= 1) go2<1, true>(q, st); else if (nt <= 2) go2<2, true>(q, st); else go2<4, true>(q, st); }
  else { if (nt <= 1) go2<1, false>(q, st); else if (nt <= 2) go2<2, false>(q, st); else go2<4, false>(q, st); }
}
// ---------------------------------------------------------------- router (N 256, K 8192, m <= 32)
// CTA (ks, mt) owns row tile mt and 16 consecutive 64-wide K chunks, one per warp. Every warp loads its hi + lo A
// fragments and its x chunk straight into registers in one round trip (64 KB per CTA in flight); the 16 warp partials
// meet in smem and the last of the KS CTAs on a tile sums their partials (no atomics on the output, no second kernel).
#define RPART (1 << 20)
#define RCNT 2048
__device__ __forceinline__ uint4 ldnc(const void* p) {
  uint4 v; asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}
template <int NT>
__global__ void __launch_bounds__(512, 1) rtr(Q q) {
  __shared__ float red[16 * 512];
  __shared__ int last;
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, t = lane & 3;
  const int ks = blockIdx.x, mt = blockIdx.y, KS = gridDim.x, c = ks * 16 + w;
  const uint8_t* ah = q.A + ((size_t)mt * q.KC + c) * 2048 + lane * 16;
  const uint8_t* al = q.A2 + ((size_t)mt * q.KC + c) * 2048 + lane * 16;
  uint4 a[4], h[4], b[NT][2];
#pragma unroll
  for (int j = 0; j < 4; j++) { a[j] = ldnc(ah + j * 512); h[j] = ldnc(al + j * 512); }
#pragma unroll
  for (int n = 0; n < NT; n++) {
    const uint4* xr = (const uint4*)(q.x + (size_t)min(n * 8 + g, q.M - 1) * q.K + c * 64) + t * 2;
    b[n][0] = __ldg(xr); b[n][1] = __ldg(xr + 1);
  }
  float acc[NT][4];
#pragma unroll
  for (int n = 0; n < NT; n++) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f;
#pragma unroll
  for (int j = 0; j < 4; j++)
#pragma unroll
    for (int n = 0; n < NT; n++) {
      const uint32_t* bb = (const uint32_t*)&b[n][j >> 1];
      mma(acc[n], a[j], bb[(j & 1) * 2], bb[(j & 1) * 2 + 1]);
      mma(acc[n], h[j], bb[(j & 1) * 2], bb[(j & 1) * 2 + 1]);
    }
#pragma unroll
  for (int n = 0; n < NT; n++) {
    float* r = red + w * 512 + g * 32 + n * 8 + 2 * t;
    *(float2*)r = make_float2(acc[n][0], acc[n][1]);
    *(float2*)(r + 8 * 32) = make_float2(acc[n][2], acc[n][3]);
  }
  __syncthreads();
  const int row = tid >> 5, tok = tid & 31;  // this thread's output (row of the tile, token)
  float s = 0.f;
  if (tok < NT * 8) {
#pragma unroll
    for (int ww = 0; ww < 16; ww++) s += red[ww * 512 + tid];
  }
  float* P = q.ws + RPART + (size_t)mt * KS * 512;
  P[ks * 512 + tid] = s;
  __threadfence();
  __syncthreads();
  if (tid == 0) last = atomicAdd(q.cnt + RCNT + mt, 1) == KS - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();
  float tot = 0.f;
  for (int k2 = 0; k2 < KS; k2++) tot += __ldcg(P + k2 * 512 + tid);
  if (tok < q.M) ((float*)q.out)[(size_t)tok * q.N + mt * 16 + row] = tot;
  if (tid == 0) q.cnt[RCNT + mt] = 0;
}
void run3(torch::Tensor A, torch::Tensor A2, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int64_t N) {
  Q q;
  q.A = (const uint8_t*)A.data_ptr(); q.A2 = (const uint8_t*)A2.data_ptr();
  q.x = (const __nv_bfloat16*)x.data_ptr(); q.out = out.data_ptr(); q.ws = ws.data_ptr<float>(); q.cnt = cnt.data_ptr<int>();
  q.M = x.size(0); q.K = x.size(1); q.N = (int)N; q.MT = (int)(N / 16); q.KC = q.K / 64; q.TB = 1; q.f32 = 1;
  const dim3 grid(q.KC / 16, q.MT);
  auto st = at::cuda::getCurrentCUDAStream();
  const int nt = (q.M + 7) / 8;
  if (nt <= 1) rtr<1><<<grid, 512, 0, st>>>(q); else if (nt <= 2) rtr<2><<<grid, 512, 0, st>>>(q); else rtr<4><<<grid, 512, 0, st>>>(q);
}
"""

P__mod = None
P__DBG = []
P_RTR = True
P_V6 = False  # deep per-CTA ring GEMV: slower than v5 on H200


def P__big():
    pr = torch.cuda.get_device_properties(0)
    sm = getattr(pr, "shared_memory_per_block_optin", None)
    return sm >= 180000 if sm else (pr.major, pr.minor) in ((9, 0), (10, 0))


def P__ext():
    global P__mod
    if P__mod is None:
        wpc = next((int(d[6:]) for d in P__DBG if d.startswith("-DWPC=")), 16 if P__big() else 8)
        P__mod = load_inline("skinny_v11_%d" % wpc + "".join(P__DBG).replace("-", "_").replace("=", ""),
                           cpp_sources="void run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t); void run2(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t); void run3(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t);",
                           cuda_sources=P_CUDA, functions=["run", "run2", "run3"], extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo", "-DWPC=%d" % wpc] + ([] if P__big() else ["-DNSG=4"]) + [d for d in P__DBG if not d.startswith("-DWPC=")],
                           verbose=False)
    return P__mod


def P__pack(w):
    """bf16 [N, K] -> A fragments [N/16, K/64, 32 lanes, 4 steps, 8] (K permuted so each lane's 16 k are contiguous)."""
    N, K = w.shape
    W = w.view(N // 16, 16, K // 64, 4, 4, 4)  # mt, i, ch, t, j, e
    lo, hi = W[:, :8], W[:, 8:]
    fr = torch.stack((lo[..., 0], lo[..., 1], hi[..., 0], hi[..., 1], lo[..., 2], lo[..., 3], hi[..., 2], hi[..., 3]), -1)
    return fr.permute(0, 2, 4, 1, 3, 5).contiguous()  # mt, ch, j, g, t, 8: per step j the 32 lanes are contiguous


def P_prepare(weights):
    st = {}
    for k, w in weights.items():
        if k == "router":
            hi = w.to(torch.bfloat16)
            lo = (w - hi.float()).to(torch.bfloat16)
            st[k] = (P__pack(hi), P__pack(lo))
            st["router_hl_t"] = torch.cat((hi, lo)).t().contiguous()  # [K, 2N]: one cuBLAS mm for m > 32
        else:
            st[k] = (P__pack(w), None)
        st[k + "_raw"] = w
    dev = weights["qkv_gate"].device
    st["ws"] = torch.zeros((256 * 8192,), dtype=torch.float32, device=dev)
    st["cnt"] = torch.zeros((8 * 512,), dtype=torch.int32, device=dev)
    st["none"] = torch.empty(0, dtype=torch.bfloat16, device=dev)
    P__ext()
    return st


def P_kernel_fn(data, state):
    cfg = data["config"]
    layer, n = cfg["layer"], cfg["n"]
    x = data["x"]
    m = x.shape[0]
    if layer != "router" and (m > 1 or layer == "o") and not (layer == "unpooler" and m == 1):
        return x @ state[layer + "_raw"][:n].T  # cuBLAS beats the skinny kernel here (judge v7: 0.45-0.9x)
    if layer == "router" and m > 32:
        a = torch.mm(x, state["router_hl_t"], out_dtype=torch.float32)
        return a[:, :n] + a[:, n:]
    A, A2 = state[layer]
    out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
    if layer == "router" and m <= 32 and P_RTR:
        P__ext().run3(A, A2, x.contiguous(), out, state["ws"], state["cnt"], n)
        return out
    (P__ext().run2 if P_V6 and m <= 32 else P__ext().run)(A, A2 if A2 is not None else state["none"], x.contiguous(), out, state["ws"], state["cnt"], n)
    return out

class Weights(TypedDict):
    qkv_gate: torch.Tensor            # bfloat16 [2304, 8192]: q 16x64, k 2x64, v 2x64, output gate 16x64
    o: torch.Tensor                   # bfloat16 [8192, 1024]
    attn_residual_gate: torch.Tensor  # bfloat16 [8192], the post-attention residual gate's weight
    pre_ffn_norm: torch.Tensor        # bfloat16 [8192] (applied as 1 + w)
    router: torch.Tensor              # float32 [256, 8192]
    router_bias: torch.Tensor         # float32 [256], added to the scores only to choose experts
    pooler: torch.Tensor              # bfloat16 [4096, 8192]
    pre_expert_norm: torch.Tensor     # bfloat16 [4096] (applied as 1 + w)
    w13: torch.Tensor                 # uint8 [32, 8192, 2048]: this GPU's experts, gate then up rows, packed E2M1
    w13_sf: torch.Tensor              # float8_e4m3fn [32, 8192, 256]
    w13_gs: torch.Tensor              # float32 [32]
    w2: torch.Tensor                  # uint8 [32, 4096, 2048]
    w2_sf: torch.Tensor               # float8_e4m3fn [32, 4096, 256]
    w2_gs: torch.Tensor               # float32 [32]
    post_expert_norm: torch.Tensor    # bfloat16 [4096] (applied as 1 + w, before normalizing)
    unpooler: torch.Tensor            # bfloat16 [8192, 4096]; up to 8 tokens only rows 0-1023 (this GPU's slice)
    post_ffn_norm: torch.Tensor       # bfloat16 [8192] (applied as 1 + w, before normalizing)
    ffn_residual_gate: torch.Tensor   # bfloat16 [8192]
    next_pre_attn_norm: torch.Tensor  # bfloat16 [8192], the next layer's input norm (applied as 1 + w)


class Config(TypedDict):
    batch: int            # B sequences
    q_len: int            # L: 1 for decode, 8 for a verify block (causal inside it); 200-16384 on muse_prefill_layer
    tokens: int           # T = B * L, request-major
    hidden: int           # 8192
    num_q_heads: int      # 16 per GPU under TP8
    num_kv_heads: int     # 2
    head_dim: int         # 64
    page_size: int        # 1 in the scored cases (as SGLang serves Muse on H200), 64 in some tests
    window: int           # 2048 on sliding-window layers (a token and the 2,047 before it); 0 = all
    rope_theta: float     # 500000 on sliding-window layers (interleaved RoPE); 0 = no RoPE
    qk_scale_factor: float
    sm_scale: float       # 1/8
    kv_scale: float       # the cache scales' value: 1 in the scored cases (SGLang leaves them unset)
    n_experts: int        # 256
    top_k: int            # 8
    local_experts: int    # 32: this GPU computes experts local_start .. local_start + 31


class Inputs(TypedDict):
    stream: torch.Tensor      # float32 [T, 8192], the residual stream
    attn_input: torch.Tensor  # bfloat16 [T, 8192], the normalized stream (the previous layer's boundary wrote it)
    k_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, 2, 64], written in place
    v_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, 2, 64], written in place
    positions: torch.Tensor   # int64 [T], each new token's position (seq_lens[b] + i)
    cache_loc: torch.Tensor   # int64 [T], each new token's flat cache row (page * page_size + offset)
    cache_seqlens: torch.Tensor  # int32 [B], seq_lens + L
    cu_seqlens_q: torch.Tensor   # int32 [B+1], 0, L, 2L, ...
    scheduler_metadata: torch.Tensor | None  # int32, FA3's tile schedule (get_scheduler_metadata) on
                                             # full-attention decode steps, else None
    k_scale: torch.Tensor     # float32 [], key = e4m3 * k_scale: 1 in the scored cases, 1/64 in some tests
    v_scale: torch.Tensor     # float32 []
    page_table: torch.Tensor  # int32 [B, max_pages], physical page of each logical page (-1 unused)
    seq_lens: torch.Tensor    # int32 [B], cached tokens per request; the block starts at this position
    local_start: torch.Tensor # int32 [1], this GPU's first expert (a multiple of 32; its weights are w13[0], ...)
    config: Config



GLUE_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#define H 8192
// one CTA (1024 threads, 8 values each) per token row
__device__ __forceinline__ float bsum(float v, float* red) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  __syncthreads();
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < 32; i++) s += red[i];
  return s;
}
// s' = alpha * s + beta * rms(a * ga, 1e-8);  out = bf16(rms(s', 1e-5) * gn). a has `na` columns (rest zero).
// every load is issued before the first reduction
__device__ __forceinline__ void ld8(const float* p, float* v) { const float4 x = __ldg((const float4*)p), y = __ldg((const float4*)p + 1); v[0] = x.x; v[1] = x.y; v[2] = x.z; v[3] = x.w; v[4] = y.x; v[5] = y.y; v[6] = y.z; v[7] = y.w; }
__global__ void __launch_bounds__(1024) resid(const __nv_bfloat16* a, int na, const float* ga, const float* s, const float* al, const float* be,
                                              const float* gn, float* so, __nv_bfloat16* out) {
  __shared__ float red[32];
  const int r = blockIdx.x, c0 = threadIdx.x * 8;
  float x[8], sv[8], av[8], bv[8], gv[8], gav[8];
  const bool in = c0 < na;
  uint4 araw = make_uint4(0, 0, 0, 0);
  if (in) araw = __ldg((const uint4*)(a + (size_t)r * na + c0));
  ld8(s + (size_t)r * H + c0, sv); ld8(al + c0, av); ld8(be + c0, bv); ld8(gn + c0, gv);
  if (ga && in) ld8(ga + c0, gav);
  const __nv_bfloat16* ab = (const __nv_bfloat16*)&araw;
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < 8; i++) { x[i] = in ? __bfloat162float(ab[i]) * (ga ? gav[i] : 1.f) : 0.f; ss += x[i] * x[i]; }
  const float ra = rsqrtf(bsum(ss, red) / H + 1e-8f);
  float v[8], s2 = 0.f;
#pragma unroll
  for (int i = 0; i < 8; i++) { v[i] = av[i] * sv[i] + bv[i] * (x[i] * ra); s2 += v[i] * v[i]; }
  const float rs = rsqrtf(bsum(s2, red) / H + 1e-5f);
  __nv_bfloat16 ob[8];
#pragma unroll
  for (int i = 0; i < 8; i++) ob[i] = __float2bfloat16(v[i] * rs * gv[i]);
  float4* sp = (float4*)(so + (size_t)r * H + c0);
  sp[0] = make_float4(v[0], v[1], v[2], v[3]); sp[1] = make_float4(v[4], v[5], v[6], v[7]);
  *(uint4*)(out + (size_t)r * H + c0) = *(uint4*)ob;
}
// out = bf16(rms(x) * g) over n = 4096 columns, 512 threads
__global__ void __launch_bounds__(512) rmsg(const __nv_bfloat16* x, const float* g, __nv_bfloat16* out) {
  __shared__ float red[32];
  if (threadIdx.x < 32) red[threadIdx.x] = 0.f;
  const int r = blockIdx.x, c0 = threadIdx.x * 8;
  float v[8], ss = 0.f;
#pragma unroll
  for (int i = 0; i < 8; i++) { v[i] = __bfloat162float(x[(size_t)r * 4096 + c0 + i]); ss += v[i] * v[i]; }
#pragma unroll
  for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  __syncthreads();
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
  __syncthreads();
  float t = 0.f;
#pragma unroll
  for (int i = 0; i < 16; i++) t += red[i];
  const float rr = rsqrtf(t / 4096 + 1e-5f);
#pragma unroll
  for (int i = 0; i < 8; i++) out[(size_t)r * 4096 + c0 + i] = __float2bfloat16(v[i] * rr * g[c0 + i]);
}
// routing: one warp per token. sigmoid(0.1767767 * logit); top 8 of score + bias; renormalized picked scores
__global__ void route(const float* lg, const float* bias, const int* lstart, int T, int* ids, int* loc, float* tw) {
  const int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (r >= T) return;
  float sc[8], key[8];
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const int e = i * 32 + lane;
    sc[i] = 1.f / (1.f + expf(-lg[(size_t)r * 256 + e] * 0.1767766952966369f));
    key[i] = sc[i] + bias[e];
  }
  const int ls = *lstart;
  float pick = 0.f, sum = 0.f;
  int myid = 0;
#pragma unroll
  for (int k = 0; k < 8; k++) {
    float bv = key[0]; int bi = 0;
#pragma unroll
    for (int i = 1; i < 8; i++) if (key[i] > bv) { bv = key[i]; bi = i; }
    int be = bi * 32 + lane;
#pragma unroll
    for (int o = 16; o; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, bv, o); const int oe = __shfl_xor_sync(0xffffffffu, be, o);
      if (ov > bv || (ov == bv && oe < be)) { bv = ov; be = oe; }
    }
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++) if (i * 32 + lane == be) { s = sc[i]; key[i] = -INFINITY; }
    s = __shfl_sync(0xffffffffu, s, be & 31);
    sum += s;
    if (lane == k) { pick = s; myid = be; }
  }
  if (lane < 8) {
    ids[r * 8 + lane] = myid;
    loc[r * 8 + lane] = (myid - ls) & 255;
    tw[r * 8 + lane] = pick / (sum + 1e-15f);
  }
}
void routing(torch::Tensor lg, torch::Tensor bias, torch::Tensor ls, torch::Tensor ids, torch::Tensor loc, torch::Tensor tw) {
  const int T = lg.size(0);
  route<<<(T + 3) / 4, 128, 0, at::cuda::getCurrentCUDAStream()>>>(lg.data_ptr<float>(), bias.data_ptr<float>(), ls.data_ptr<int>(), T,
      ids.data_ptr<int>(), loc.data_ptr<int>(), tw.data_ptr<float>());
}
// prefetch byte ranges into L2 (fire and forget): grid-stride over 32 KB chunks
__global__ void l2pf(const char* p0, long long n0, const char* p1, long long n1, const char* p2, long long n2) {
  const long long C = 32768;
  const long long c0 = (n0 + C - 1) / C, c1 = (n1 + C - 1) / C, c2 = (n2 + C - 1) / C;
  for (long long i = blockIdx.x * blockDim.x + threadIdx.x; i < c0 + c1 + c2; i += (long long)gridDim.x * blockDim.x) {
    const char* p; long long off, n;
    if (i < c0) { p = p0; off = i * C; n = n0; } else if (i < c0 + c1) { p = p1; off = (i - c0) * C; n = n1; } else { p = p2; off = (i - c0 - c1) * C; n = n2; }
    const unsigned sz = (unsigned)min(C, n - off);
#if __CUDA_ARCH__ >= 900
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" :: "l"(p + off), "r"(sz & ~15u) : "memory");
#else
    for (unsigned k = 0; k < sz; k += 128) asm volatile("prefetch.global.L2 [%0];\n" :: "l"(p + off + k));
#endif
  }
}
__global__ void spin(long long ns) { long long t0; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0)); long long t; do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); } while (t - t0 < ns); }
void prefetch(torch::Tensor a, torch::Tensor b, torch::Tensor c, int64_t nb, int64_t grid) {
  l2pf<<<grid, 32, 0, at::cuda::getCurrentCUDAStream()>>>((const char*)a.data_ptr(), a.numel() * a.element_size(), (const char*)b.data_ptr(),
      b.numel() * b.element_size(), (const char*)c.data_ptr(), min((long long)nb, (long long)(c.numel() * c.element_size())));
}
void spinfor(int64_t ns) { spin<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(ns); }
void resid1(torch::Tensor a, torch::Tensor s, torch::Tensor al, torch::Tensor be, torch::Tensor gn, torch::Tensor so, torch::Tensor out) {
  resid<<<a.size(0), 1024, 0, at::cuda::getCurrentCUDAStream()>>>((const __nv_bfloat16*)a.data_ptr(), a.size(1), nullptr,
      s.data_ptr<float>(), al.data_ptr<float>(), be.data_ptr<float>(), gn.data_ptr<float>(), so.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr());
}
void resid2(torch::Tensor a, torch::Tensor ga, torch::Tensor s, torch::Tensor al, torch::Tensor be, torch::Tensor gn, torch::Tensor so, torch::Tensor out) {
  resid<<<a.size(0), 1024, 0, at::cuda::getCurrentCUDAStream()>>>((const __nv_bfloat16*)a.data_ptr(), a.size(1), ga.data_ptr<float>(),
      s.data_ptr<float>(), al.data_ptr<float>(), be.data_ptr<float>(), gn.data_ptr<float>(), so.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr());
}
void rmsgain(torch::Tensor x, torch::Tensor g, torch::Tensor out) {
  rmsg<<<x.size(0), 512, 0, at::cuda::getCurrentCUDAStream()>>>((const __nv_bfloat16*)x.data_ptr(), g.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr());
}
"""
_gmod = None
MOE_SMALL_MAX = 2048 if torch.cuda.get_device_capability() == (9, 0) else 256  # above this many tokens: dequant + grouped GEMMs


def _gext():
    global _gmod
    if _gmod is None:
        _gmod = load_inline("dl_glue_v4", cpp_sources="void resid1(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor); void resid2(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor); void rmsgain(torch::Tensor,torch::Tensor,torch::Tensor); void prefetch(torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t); void spinfor(int64_t); void routing(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor);",
                            cuda_sources=GLUE_CUDA, functions=["resid1", "resid2", "rmsgain", "prefetch", "spinfor", "routing"], extra_cuda_cflags=["-O3"], verbose=False)
    return _gmod


def _coef(gate):
    w = gate.float() / 0.3
    beta = torch.sigmoid(w)
    return torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1.0 + beta), min=1e-3)).contiguous(), beta.contiguous()


MARLIN_MIN = 512  # prefill-sized batches (decode layers stop at 256): SGLang's Marlin W4A16 MoE, called exactly as the baseline calls it
_MAR = []


def _marlin_mods():
    if not _MAR:
        try:
            from sglang.kernels.ops.activation.activation import silu_and_mul
            from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
            from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import fused_post_expert_norm_weighted_sum
            from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
            from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
            from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin
            from sgl_kernel.scalar_type import scalar_types
            _MAR.append(dict(silu_and_mul=silu_and_mul, gemm=moe_wna16_marlin_gemm, post=fused_post_expert_norm_weighted_sum,
                             align=moe_align_block_size, ws=marlin_make_workspace, prep=prepare_moe_nvfp4_layer_for_marlin,
                             fp4=scalar_types.float4_e2m1f))
        except Exception:
            _MAR.append(None)
    return _MAR[0]


def _marlin_bank(weights, mm):
    import types
    dev = weights["w13"].device
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16), moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2, params_dtype=torch.bfloat16,
        w13_weight=weights["w13"].contiguous(), w2_weight=weights["w2"].contiguous(),
        w13_weight_scale=weights["w13_sf"].contiguous(), w2_weight_scale=weights["w2_sf"].contiguous(),
        w13_weight_scale_2=weights["w13_gs"].contiguous(), w2_weight_scale_2=weights["w2_gs"].contiguous())
    mm["prep"](layer)
    em = torch.full((256,), -1, dtype=torch.int32, device=dev)
    em[:32] = torch.arange(32, dtype=torch.int32, device=dev)
    return {"w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
            "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
            "expert_map": em, "size_n": layer.intermediate_size_per_partition, "workspace": mm["ws"](dev, max_blocks_per_sm=4),
            "post_w": (1.0 + weights["post_expert_norm"].float()).to(torch.bfloat16)}


def _marlin_experts(x, tw, loc, bank, mm):
    T, K = x.shape
    N = bank["size_n"]
    lid = bank["expert_map"][loc.long()].contiguous()
    tw = tw.to(torch.float32).contiguous()
    for bm in (8, 16, 32, 48, 64):
        if T * 8 / 32 / bm < 0.9:
            break
    sid, bexp, npad = mm["align"](lid, bm, 256)
    c2 = torch.empty((T * 8, N), device=x.device, dtype=x.dtype)
    c13 = torch.empty((T * 8 * max(2 * N, K),), device=x.device, dtype=x.dtype)
    c1 = c13[: T * 8 * 2 * N].view(-1, 2 * N)
    c3 = c13[: T * 8 * K].view(-1, K)
    com = dict(moe_block_size=bm, mul_topk_weights=False, is_ep=True, b_q_type=mm["fp4"], is_k_full=True, use_atomic_add=False,
               use_fp32_reduce=True, is_zp_float=False)
    ws = bank["workspace"]
    c1 = mm["gemm"](x, c1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws, sid, bexp, npad, tw, top_k=8,
                    size_m=T, size_n=2 * N, size_k=K, **com)
    mm["silu_and_mul"](c1.view(-1, 2 * N), c2, expert_ids=lid.view(-1), expert_step=1)
    y = mm["gemm"](c2, c3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws, sid, bexp, npad, tw, top_k=1,
                   size_m=T * 8, size_n=K, size_k=N, **com).view(T, 8, K)
    return mm["post"](y, tw, norm_weight=bank["post_w"], eps=1e-8, weight_first=True, slot_expert_ids=lid)


def prepare(weights):
    st = {k: weights[k] for k in ("qkv_gate", "o", "router", "router_bias", "pooler", "unpooler")}
    hi = weights["router"].to(torch.bfloat16)
    st["rA"], st["rA2"] = P__pack(hi), P__pack((weights["router"] - hi.float()).to(torch.bfloat16))
    st["rhi"], st["rlo"] = hi.contiguous(), (weights["router"] - hi.float()).to(torch.bfloat16).contiguous()
    st["pws"] = torch.zeros((256 * 8192,), dtype=torch.float32, device=hi.device)
    st["pk"] = {k: P__pack(weights[k]) for k in ("qkv_gate", "o", "pooler", "unpooler")}  # mma A-fragment order for the GEMV
    st["gws"] = torch.zeros((32 * 8192,), dtype=torch.float32, device=hi.device)
    st["gcnt"] = torch.zeros((8 * 512,), dtype=torch.int32, device=hi.device)
    st["none"] = torch.empty(0, dtype=torch.bfloat16, device=hi.device)
    st["pcnt"] = torch.zeros((8 * 512,), dtype=torch.int32, device=hi.device)
    st["router_bias"] = weights["router_bias"].float().contiguous()
    st["side"] = torch.cuda.Stream()
    st["unpool8"] = weights["unpooler"][:1024].contiguous()
    st["a1"], st["b1"] = _coef(weights["attn_residual_gate"])
    st["a2"], st["b2"] = _coef(weights["ffn_residual_gate"])
    for k in ("pre_ffn_norm", "pre_expert_norm", "post_ffn_norm", "next_pre_attn_norm"):
        st["g_" + k] = (1.0 + weights[k].float()).contiguous()
    for k in ("w13", "w13_sf", "w13_gs", "w2", "w2_sf", "w2_gs"):  # raw NVFP4 for the prefill-sized MoE path
        st["p" + k] = weights[k].contiguous() if k.endswith("gs") else weights[k]
    st["g_post_expert_norm"] = (1.0 + weights["post_expert_norm"].float()).contiguous()
    st["moe"] = M_prepare({"w13": weights["w13"], "w13_sf": weights["w13_sf"], "w13_gs": weights["w13_gs"], "w2": weights["w2"],
                           "w2_sf": weights["w2_sf"], "w2_gs": weights["w2_gs"], "post_norm_w": weights["post_expert_norm"]})
    mm = _marlin_mods()
    st["bank"] = _marlin_bank(weights, mm) if (mm is not None and MARLIN_MIN > 0) else None
    _gext(); A__ext(); P__ext(); M__ext(M_LARGE)
    torch.cuda.synchronize()
    print("env torch", torch.__version__, "grouped_mm", hasattr(torch, "_grouped_mm"), "cap", torch.cuda.get_device_capability(), flush=True)
    if PFDIAG:
        _pfdiag(st)
    return st


PFDIAG = False


def _pfdiag(st):
    g = _gext()
    fl = torch.empty(2**28, device="cuda")
    side = torch.cuda.Stream()
    x = torch.randn((1, 1024), device="cuda").bfloat16()
    h = torch.randn((1, 8192), device="cuda").bfloat16()
    def run(mode, nb, grid, spin_us):
        ts = []
        for _ in range(7):
            fl.add_(1)
            e0, e1, e2 = torch.cuda.Event(True), torch.cuda.Event(True), torch.cuda.Event(True)
            cur = torch.cuda.current_stream()
            if mode:
                side.wait_stream(cur)
                with torch.cuda.stream(side):
                    g.prefetch(st["o"], st["router"], st["pooler"], nb, grid)
            g.spinfor(int(spin_us * 1000))
            e0.record()
            a = x @ st["o"].T
            e1.record()
            p = h @ st["pooler"].T
            r = h.float() @ st["router"].T
            e2.record()
            cur.wait_stream(side)
            torch.cuda.synchronize()
            ts.append((e0.elapsed_time(e1) * 1e3, e1.elapsed_time(e2) * 1e3))
        ts.sort()
        return ts[3]
    for mode, nb, grid, sp in ((0, 0, 1, 15), (1, 0, 32, 15), (1, 16 << 20, 32, 15), (1, 24 << 20, 64, 15), (1, 0, 132, 15), (1, 16 << 20, 132, 20)):
        o, rest = run(mode, nb, grid, sp)
        print(f"PFDIAG pf {mode} pooler {nb >> 20}MB grid {grid} spin {sp}us: o {o:.1f}us pooler+router {rest:.1f}us", flush=True)


def _mm32(a, b):
    """bf16 GEMM with FP32 output (torch >= 2.8); exact FP32 GEMM on older torch"""
    try:
        return torch.mm(a, b, out_dtype=torch.float32)
    except TypeError:
        return a.float() @ b.float()


def _proj(x, st, name, n):
    """x @ W[:n].T: the deep-ring GEMV up to 32 tokens, cuBLAS above"""
    T = x.shape[0]
    if T > 32 or not GEMV:
        return x @ (st["unpool8"] if name == "unpooler" and n == 1024 else st[name]).T
    out = torch.empty((T, n), dtype=torch.bfloat16, device=x.device)
    P__ext().run2(st["pk"][name], st["none"], x.contiguous(), out, st["gws"], st["gcnt"], n)
    return out


GEMV = False


def kernel_fn(data, st):
    cfg = data["config"]
    B, L, T = cfg["batch"], cfg["q_len"], cfg["tokens"]
    x = data["attn_input"]
    dev = x.device
    qkv = _proj(x, st, "qkv_gate", 2304).view(B, L, -1)
    ad = {k: data[k] for k in ("positions", "cache_loc", "k_cache", "v_cache", "k_scale", "v_scale", "page_table", "seq_lens")}
    ad.update({k: data[k] for k in ("cache_seqlens", "cu_seqlens_q", "scheduler_metadata") if k in data})
    ad["qkv"] = qkv
    ad["config"] = {k: cfg[k] for k in ("batch", "q_len", "num_q_heads", "num_kv_heads", "head_dim", "page_size", "window",
                                         "rope_theta", "qk_scale_factor", "sm_scale", "kv_scale")}
    att = A_kernel_fn(ad).view(T, -1)
    a = _proj(att, st, "o", 8192)
    g = _gext()
    s1 = torch.empty((T, 8192), dtype=torch.float32, device=dev)
    h = torch.empty((T, 8192), dtype=torch.bfloat16, device=dev)
    g.resid1(a, data["stream"], st["a1"], st["b1"], st["g_pre_ffn_norm"], s1, h)
    big = T > MOE_SMALL_MAX
    lg = torch.empty((T, 256), dtype=torch.float32, device=dev)
    ids = torch.empty((T, 8), dtype=torch.int32, device=dev)
    loc = torch.empty_like(ids)
    tw = torch.empty((T, 8), dtype=torch.float32, device=dev)
    cur, side = torch.cuda.current_stream(), st["side"]
    side.wait_stream(cur)
    with torch.cuda.stream(side):  # router + routing overlap the pooler, as in SGLang
        if T > 256:  # FP32 router as a BF16 hi + lo pair with FP32 accumulation
            lg.copy_(_mm32(h, st["rhi"].T)); lg += _mm32(h, st["rlo"].T)
        else:
            P__ext().run(st["rA"], st["rA2"], h, lg, st["pws"], st["pcnt"], 256)
        g.routing(lg, st["router_bias"], data["local_start"], ids, loc, tw)
    pr = _proj(h, st, "pooler", 4096)
    p = torch.empty_like(pr)
    g.rmsgain(pr, st["g_pre_expert_norm"], p)
    cur.wait_stream(side)
    if T > MARLIN_MIN and st.get("bank") is not None:
        y = _marlin_experts(p, tw, loc, st["bank"], _marlin_mods())
        f = _proj(y, st, "unpooler", 8192)
        s2 = torch.empty_like(s1)
        hid = torch.empty_like(h)
        g.resid2(f, st["g_post_ffn_norm"], s1, st["a2"], st["b2"], st["g_next_pre_attn_norm"], s2, hid)
        return hid, s2, ids
    if big:
        y = _moe_large(p, loc, tw, st)
        f = y @ st["unpooler"].T
        s2 = torch.empty_like(s1)
        hid = torch.empty_like(h)
        g.resid2(f, st["g_post_ffn_norm"], s1, st["a2"], st["b2"], st["g_next_pre_attn_norm"], s2, hid)
        return hid, s2, ids
    ms = st["moe"]
    inter = torch.empty((T * 8, 4096), dtype=torch.bfloat16, device=dev)
    y2 = torch.empty((T * 8, 4096), dtype=torch.bfloat16, device=dev)
    ss = torch.empty((T * 8,), dtype=torch.float32, device=dev)
    y = torch.empty_like(p)
    M__ext(M_LARGE and T > 64).moe(p, loc, tw, ms["w1"], ms["w2"], ms["g1"], ms["g2"], ms["nw"], inter, y2, ss, ms["ctr"], ms["tcnt"], ms["ws"], y, 0, ms["prof"])
    f = _proj(y, st, "unpooler", 1024 if T <= 8 else 8192)
    s2 = torch.empty_like(s1)
    hid = torch.empty_like(h)
    g.resid2(f, st["g_post_ffn_norm"], s1, st["a2"], st["b2"], st["g_next_pre_attn_norm"], s2, hid)
    return hid, s2, ids

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

// grouped NT GEMM (fallback when torch has no _grouped_mm): C[rows of group e] = A[rows] @ B[e]^T, bf16 in/out, fp32 acc.
// tiles: 128 rows x 128 cols, K step 32, 3-stage cp.async ring, 8 warps (2 x 4) of 64 x 32, ldmatrix + mma.sync.
__device__ __forceinline__ void gcp(uint32_t s, const void* g, bool ok) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(s), "l"(g), "r"(ok ? 16 : 0)); }
__device__ __forceinline__ void ldm4(uint32_t a, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void gmma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__global__ void __launch_bounds__(256) ggemm(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C, const int* offs, int N, int K) {
  extern __shared__ __align__(128) uint8_t sm[];  // 3 x 16 KB
  __shared__ int ti[2];
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, wm = w >> 2, wn = w & 3;
  if (tid == 0) {  // which (group, row tile) this CTA owns: tiles enumerated group by group
    int t = blockIdx.x, e = 0, lo = 0;
    for (; e < 32; e++) { const int hi = offs[e], nt = (hi - lo + 127) >> 7; if (t < nt) break; t -= nt; lo = hi; }
    ti[0] = e; ti[1] = lo + t * 128;
  }
  __syncthreads();
  const int e = ti[0];
  if (e >= 32) return;
  const int r0 = ti[1], rend = offs[e], n0 = blockIdx.y * 128;
  const __nv_bfloat16* Bg = B + (size_t)e * N * K;
  const uint32_t sb = (uint32_t)__cvta_generic_to_shared(sm);
  auto load = [&](int st, int k0) {  // A tile 128 x 32 then B tile 128 x 32: 64 B rows, chunk c at c ^ (r >> 1 & 3)
#pragma unroll
    for (int i = 0; i < 2; i++) {
      const int idx = i * 256 + tid, r = idx >> 2, c = idx & 3;
      const uint32_t off = r * 64 + ((c ^ ((r >> 1) & 3)) << 4);
      gcp(sb + st * 16384 + off, A + (size_t)min(r0 + r, rend - 1) * K + k0 + c * 8, r0 + r < rend);
      gcp(sb + st * 16384 + 8192 + off, Bg + (size_t)(n0 + r) * K + k0 + c * 8, true);
    }
    asm volatile("cp.async.commit_group;\n");
  };
  float acc[4][4][4] = {};
  const int nk = K / 32;
  load(0, 0); load(1, 32);
  for (int kt = 0; kt < nk; kt++) {
    if (kt + 2 < nk) load((kt + 2) % 3, (kt + 2) * 32); else asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_group 2;\n" ::: "memory");
    __syncthreads();
    const uint32_t sa = sb + (kt % 3) * 16384, sbb = sa + 8192;
#pragma unroll
    for (int kk = 0; kk < 2; kk++) {
      uint32_t af[4][4], bf[2][4];
#pragma unroll
      for (int mi = 0; mi < 4; mi++) {
        const int r = wm * 64 + mi * 16 + (lane & 15), c = kk * 2 + (lane >> 4);
        ldm4(sa + r * 64 + ((c ^ ((r >> 1) & 3)) << 4), af[mi][0], af[mi][1], af[mi][2], af[mi][3]);
      }
#pragma unroll
      for (int nj = 0; nj < 2; nj++) {
        const int r = wn * 32 + nj * 16 + (lane & 7) + ((lane >> 4) << 3), c = kk * 2 + ((lane >> 3) & 1);
        ldm4(sbb + r * 64 + ((c ^ ((r >> 1) & 3)) << 4), bf[nj][0], bf[nj][1], bf[nj][2], bf[nj][3]);
      }
#pragma unroll
      for (int mi = 0; mi < 4; mi++)
#pragma unroll
        for (int ni = 0; ni < 4; ni++) gmma(acc[mi][ni], af[mi], bf[ni >> 1][(ni & 1) * 2], bf[ni >> 1][(ni & 1) * 2 + 1]);
    }
    __syncthreads();
  }
  const int g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int mi = 0; mi < 4; mi++)
#pragma unroll
    for (int hh = 0; hh < 2; hh++) {
      const int r = r0 + wm * 64 + mi * 16 + g + hh * 8;
      if (r >= rend) continue;
#pragma unroll
      for (int ni = 0; ni < 4; ni++) {
        const int c = n0 + wn * 32 + ni * 8 + 2 * t;
        *(__nv_bfloat162*)(C + (size_t)r * N + c) = __floats2bfloat162_rn(acc[mi][ni][2 * hh], acc[mi][ni][2 * hh + 1]);
      }
    }
}
void gmm(torch::Tensor A, torch::Tensor B, torch::Tensor offs, torch::Tensor C) {
  const int N = B.size(1), K = B.size(2), R = A.size(0);
  dim3 grid((R + 127) / 128 + 32, N / 128);
  static bool once = false;
  if (!once) { cudaFuncSetAttribute(ggemm, cudaFuncAttributeMaxDynamicSharedMemorySize, 3 * 16384); once = true; }
  ggemm<<<grid, 256, 3 * 16384, at::cuda::getCurrentCUDAStream()>>>((const __nv_bfloat16*)A.data_ptr(), (const __nv_bfloat16*)B.data_ptr(), (__nv_bfloat16*)C.data_ptr(), offs.data_ptr<int>(), N, K);
}
"""
_pfmod = None


def _pfext():
    global _pfmod
    if _pfmod is None:
        _pfmod = load_inline("pf_moe_v2", cpp_sources="void dequant(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor); void routes(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor); void gather(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor); void act(torch::Tensor,torch::Tensor,torch::Tensor); void finish(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor); void gmm(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor);",
                             cuda_sources=PF_CUDA, functions=["dequant", "routes", "gather", "act", "finish", "gmm"], extra_cuda_cflags=["-O3"], verbose=False)
    return _pfmod


def _gmm(A, W, offs):
    """grouped A[rows of group e] @ W[e].T (W [E, N, K]): torch's CUTLASS grouped GEMM when present, else ours"""
    if hasattr(torch, "_grouped_mm") and torch.cuda.get_device_capability() == (9, 0):
        return torch._grouped_mm(A, W.transpose(1, 2), offs=offs)
    out = torch.empty((A.shape[0], W.shape[1]), dtype=torch.bfloat16, device=A.device)
    _pfext().gmm(A, W, offs, out)
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
    H = _gmm(A, w13, offs)
    I = torch.empty((R, 4096), dtype=torch.bfloat16, device=dev)
    g.act(H, offs, I)
    Y = _gmm(I, w2, offs)
    acc = torch.empty((T, 4096), dtype=torch.float32, device=dev)
    out = torch.empty((T, 4096), dtype=torch.bfloat16, device=dev)
    g.finish(Y, lst, offs, st["g_post_expert_norm"], tw.view(-1), acc, out)
    return out
