# okc: problem=muse_attention_decode
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


CUDA = r"""
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
#ifndef MCH
#define MCH 8
#endif
#if FD_L2H
#define FDH ".L2::128B"
#else
#define FDH ""
#endif
#if AW_L2H
#define AWH ".L2::128B"
#else
#define AWH ""
#endif
#ifndef PFF
#define PFF 0
#endif

// ---------------------------------------------------------------- prep: one CTA (64 x Hq+2Hkv threads) per new token
__device__ __forceinline__ void rope_cs(const float* invf, int pos, int pair, float& sn, float& cs);
__device__ __forceinline__ void rope_cs2(float invf, int pos, float& sn, float& cs);
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
#ifndef AW_MS
#define AW_MS 1
#endif
#ifndef AW_LZ
#define AW_LZ 1
#endif
#ifndef LZT
#define LZT 8.f
#endif
#define LZB 0x5C00u
#define LZP 256.f
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
#ifndef WGEMU
#define WGEMU 0
#endif
#if !HAS_WG && WGEMU
__device__ __forceinline__ float emu_e4m3(uint32_t b) {
  const int e = (b >> 3) & 15, m = b & 7;
  const float v = e ? ldexpf(1.f + m * 0.125f, e - 7) : ldexpf(m * 0.125f, -6);
  return (b & 0x80) ? -v : v;
}
__device__ __forceinline__ uint32_t emu_sw(uint32_t lin, uint64_t desc) {
  return (desc >> 62) == 1 ? (lin ^ (((lin >> 7) & 7) << 4)) : lin;
}
__device__ __forceinline__ uint32_t emu_kmaj(uint64_t desc, int r, int kb) {
  const uint32_t st = (uint32_t)(desc & 0x3FFF) << 4, lbo = (uint32_t)((desc >> 16) & 0x3FFF) << 4, sbo = (uint32_t)((desc >> 32) & 0x3FFF) << 4;
  if ((desc >> 62) == 1) return emu_sw(st + (r >> 3) * sbo + (r & 7) * 128 + kb, desc);
  return st + (r >> 3) * sbo + (kb >> 4) * lbo + (r & 7) * 16 + (kb & 15);
}
__device__ __forceinline__ uint32_t emu_mnmaj(uint64_t desc, int k, int n2) {
  const uint32_t st = (uint32_t)(desc & 0x3FFF) << 4, sbo = (uint32_t)((desc >> 32) & 0x3FFF) << 4;
  return emu_sw(st + (k >> 3) * sbo + (k & 7) * 128 + n2, desc);
}
__device__ __forceinline__ uint32_t emu_ld8(uint32_t a) { uint32_t v; asm volatile("ld.shared.u8 %0, [%1];" : "=r"(v) : "r"(a)); return v; }
__device__ __forceinline__ float emu_ldh(uint32_t a) { unsigned short v; asm volatile("ld.shared.u16 %0, [%1];" : "=h"(v) : "r"(a)); __half_raw r; r.x = v; return __half2float(__half(r)); }
__device__ __forceinline__ void emu_bar() { asm volatile("bar.sync %0, 128;" :: "r"(5 + (threadIdx.x >> 7)) : "memory"); }
template <int N>
__device__ __noinline__ void emu_rs(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db, bool sd, bool mn) {
  __shared__ __half sa[2][64 * 16];
  const int wt = threadIdx.x & 127, w = wt >> 5, g = (wt & 31) >> 2, t = wt & 3, h = threadIdx.x >> 7;
  emu_bar();
  uint32_t* s32 = (uint32_t*)sa[h];
  s32[((16 * w + g) * 16 + 2 * t) >> 1] = a0; s32[((16 * w + g + 8) * 16 + 2 * t) >> 1] = a1;
  s32[((16 * w + g) * 16 + 2 * t + 8) >> 1] = a2; s32[((16 * w + g + 8) * 16 + 2 * t + 8) >> 1] = a3;
  emu_bar();
  for (int i = 0; i < N / 2; i++) {
    const int hh = (i >> 1) & 1, e = i & 1, j = i >> 2, m = 16 * w + g + 8 * hh, n = 8 * j + 2 * t + e;
    float acc = 0.f;
    for (int k = 0; k < 16; k++) {
      const float b = mn ? emu_ldh(emu_mnmaj(db, k, 2 * n)) : emu_ldh(emu_kmaj(db, n, 2 * k));
      acc += __half2float(sa[h][m * 16 + k]) * b;
    }
    d[i] = sd ? d[i] + acc : acc;
  }
}
__device__ __noinline__ void emu_ss8(float* d, uint64_t da, uint64_t db, bool sd) {
  const int wt = threadIdx.x & 127, w = wt >> 5, g = (wt & 31) >> 2, t = wt & 3;
  for (int i = 0; i < 32; i++) {
    const int hh = (i >> 1) & 1, e = i & 1, j = i >> 2, m = 16 * w + g + 8 * hh, n = 8 * j + 2 * t + e;
    float acc = 0.f;
    for (int k = 0; k < 32; k++) acc += emu_e4m3(emu_ld8(emu_kmaj(da, m, k))) * emu_e4m3(emu_ld8(emu_kmaj(db, n, k)));
    d[i] = sd ? d[i] + acc : acc;
  }
}
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
#elif WGEMU
  emu_rs<64>(d, a0, a1, a2, a3, db, sd != 0, true);
#endif
}
// row sums of P on the tensor core: P (registers, as for PV) x an all-ones B tile, m64n8k16 (every column = the sum)
__device__ __forceinline__ void wg_rs8(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db, int sd) {
#if HAS_WG
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %9, 0;\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0,%1,%2,%3}, {%4,%5,%6,%7}, %8, p, 1, 1, 0;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db), "r"(sd));
#elif WGEMU
  emu_rs<8>(d, a0, a1, a2, a3, db, sd != 0, false);
#endif
}
__device__ __forceinline__ void wg_rs(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db) {
#if HAS_WG
  asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 " WG_D ", {%32,%33,%34,%35}, %36, 1, 1, 1, 1;\n"
               : WG_S(d) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));
#elif WGEMU
  emu_rs<64>(d, a0, a1, a2, a3, db, true, true);
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
#elif WGEMU
  emu_ss8(d, da, db, ACC != 0);
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
  float theta, kinv, vinv, vscale; int nmg; unsigned long long* prof; int fl;  // fl & 1: chunked one-pass merge
};
// one-pass online merge of splits k0, k0 + st, ... < n: 8 splits' (m, l) and O loads issued together with clamped indices
// (a guarded load per iteration cannot be hoisted above its guard, which serializes one L2 round trip per split)
template <int CH = 8, class FM, class FV>
__device__ __forceinline__ void mrg8(int k0, int st, int n, FM mlof, FV vof, float& M, float& Lr, float4& a4) {
  for (int kb = k0; kb < n; kb += CH * st) {
    float2 ml[CH]; float4 v[CH];
#pragma unroll
    for (int j = 0; j < CH; j++) { const int k = min(kb + j * st, n - 1); ml[j] = mlof(k); v[j] = vof(k); }
#pragma unroll
    for (int j = 0; j < CH; j++) {
      if (kb + j * st < n) {
        if (ml[j].x > M) { const float c = ex2(M - ml[j].x); Lr *= c; a4.x *= c; a4.y *= c; a4.z *= c; a4.w *= c; M = ml[j].x; }
        const float w = ml[j].x == -INFINITY ? 0.f : ex2(ml[j].x - M);
        Lr = fmaf(w, ml[j].y, Lr);
        a4.x = fmaf(w, v[j].x, a4.x); a4.y = fmaf(w, v[j].y, a4.y); a4.z = fmaf(w, v[j].z, a4.z); a4.w = fmaf(w, v[j].w, a4.w);
      }
    }
  }
}
#ifdef PROF
__device__ __forceinline__ unsigned long long gtm2() { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
#define PS(k) do { if (f.prof && threadIdx.x == 0) f.prof[blockIdx.x * 8 + (k)] = gtm2(); } while (0)
#else
#define PS(k)
#endif
// v39 verify kernel (var < 0) with optional grafts
// X: 4 softmax fast path (P against the running max, packed-f16 overflow check), 8 prologue overlap (page table + KV issued
// before the query prologue, new keys written into the ring); X = 0 is v39
template <bool PS1, bool FUSE, bool MRG, int X>
__global__ void __launch_bounds__(256, 2) attnw39(A a, WF f) {
  static_assert(!(X & 16), "deep ring removed: page-table ring lookahead too short for RS >= 5");
  constexpr int RS = RSW;
  constexpr int VB = VDB;
  constexpr bool FS = X & 4, PO = (X & 8) && FUSE, FR = X & 32, L2H = X & 64, BAL = (X & 256) && FUSE && MRG;  // BAL: full layers, splits balanced over the batch's total keys
  PS(0);
  const int tid = threadIdx.x, wt = tid & 127, h = tid >> 7, w = wt >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  // block (request b, 8-token query block qb, key split sp): tokens t0 .. t0 + lt - 1
  int sp = blockIdx.x % a.nsplit, bq = blockIdx.x / a.nsplit, nsp = a.nsplit;
  if constexpr (BAL) {  // one query block per request (L = 8), B <= 32: every warp derives the same split map from all lengths
    // chunk c: the smallest with sum_b ceil(keys_b / c) <= grid (and <= a.nsplit splits per request); CTA -> (request, split)
    const int kb_ = lane < a.B ? a.seq[lane] + a.L : 0, G = gridDim.x;
    const int K = __reduce_add_sync(0xffffffffu, kb_), mxk = __reduce_max_sync(0xffffffffu, kb_);
    int lo = max((K + G - 1) / G, (mxk + a.nsplit - 1) / a.nsplit), hi = max(lo, mxk);
    while (lo < hi) {
      const int c = (lo + hi) >> 1;
      if ((int)__reduce_add_sync(0xffffffffu, lane < a.B ? (kb_ + c - 1) / c : 0) <= G) hi = c; else lo = c + 1;
    }
    const int nl = lane < a.B ? (kb_ + lo - 1) / lo : 0;
    int pre = nl;
#pragma unroll
    for (int o2 = 1; o2 < 32; o2 <<= 1) { const int v2 = __shfl_up_sync(0xffffffffu, pre, o2); if (lane >= o2) pre += v2; }
    const int bb = __popc(__ballot_sync(0xffffffffu, lane < a.B && pre <= (int)blockIdx.x));
    if (bb >= a.B) return;  // spare CTA (uniform over the block)
    nsp = __shfl_sync(0xffffffffu, nl, bb);
    sp = (int)blockIdx.x - (__shfl_sync(0xffffffffu, pre, bb) - nsp);
    bq = bb;
  }
  const int nqb = (a.L + 7) >> 3, qb = bq % nqb, b = bq / nqb;
  const int t0 = qb * 8, lt = min(8, a.L - t0);
  const int qr_ = wt >> 1, qhf = wt & 1, ql = qr_ >> 3;
  const int nrr = tid >> 4, nkh = nrr & 1, nlt = nrr >> 1, nd0 = (tid & 15) * 4;
  uint4 qraw[4];
  uint2 kraw = make_uint2(0, 0), vraw = make_uint2(0, 0);
  if constexpr (PO) {
    const __nv_bfloat16* src = f.qkv + ((size_t)b * a.L + ql) * (2 * 16 + 4) * D + (h * 8 + (qr_ & 7)) * D + qhf * 32;
#pragma unroll
    for (int i = 0; i < 4; i++) qraw[i] = *(const uint4*)(src + 8 * i);
    const __nv_bfloat16* krow = f.qkv + ((size_t)b * a.L + nlt) * (2 * 16 + 4) * D;
    kraw = *(const uint2*)(krow + (16 + nkh) * D + nd0); vraw = *(const uint2*)(krow + (18 + nkh) * D + nd0);
  }
  const int seq = a.seq[b], nk = seq + t0 + lt;
  const int k0 = a.window ? max(0, seq + t0 - a.window + 1) : 0, span = nk - k0;
  const int ks = k0 + (int)((long long)span * sp / nsp), ke = k0 + (int)((long long)span * (sp + 1) / nsp);
  extern __shared__ __align__(1024) uint8_t smem[];
  const uint32_t sb = ((uint32_t)__cvta_generic_to_shared(smem) + 1023) & ~1023u;
  // Q (E4M3) as one 64-row x 128 B tile: warpgroup h's 64 B at byte 64 h, like the head halves of the raw K rows
  const uint32_t ring = sb, qs = sb + RS * 16384, vf0 = qs + 8192 + h * (VB ? 16384 : 8192);
  const uint32_t ptr = qs + 8192 + 2 * (VB ? 16384 : 8192);  // page-table ring: 4 segments x 4 steps x 64 entries
  const uint32_t ones = ptr + 4096;  // 1 KB of f16 1.0: B of the row-sum wgmma (any layout: every element is 1)
#if AW_MS
  for (int x = wt; x < 256; x += 128) asm volatile("st.shared.u32 [%0], %1;\n" :: "r"(ones + x * 4), "r"(0x3C003C00u));
  const uint64_t dOnes = (uint64_t)((ones & 0x3FFFF) >> 4) | ((uint64_t)(128 >> 4) << 16) | ((uint64_t)(256 >> 4) << 32);
  float lacc[4];
#endif
  if constexpr (PO) {
  } else if constexpr (!FUSE) {
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
  // copy: key i * 16 + lr, K or V, 16 B chunk; FR: whole 128 B rows per 8 lanes (both heads: CTA-wide barrier per step)
  const int lr = FR ? tid >> 4 : wt >> 3, part = FR ? (tid >> 3) & 1 : (wt >> 2) & 1, c8 = FR ? tid & 7 : 4 * h + (wt & 3);
  auto rowof = [&](int kb, int i) -> int {
    const int kp = kb + i * 16 + lr, kc = min(kp, ke - 1);
    const int rr = PS1 ? __ldg(ptb + kc) : __ldg(ptb + kc / a.ps) * a.ps + kc % a.ps;
    return kp < ke ? rr : -1;
  };
  auto issue = [&](int slot, int kbi, int r0, int r1, int r2, int r3) {
    const int rw[4] = {r0, r1, r2, r3};
#ifdef AW_NOLOAD
    if (kbi >= ks + (RS - 1) * 64 || kbi >= ke) { asm volatile("cp.async.commit_group;\n"); return; }  // ablation: only the prologue steps load
#endif
    if (kbi >= ke) { asm volatile("cp.async.commit_group;\n"); return; }  // step past the range
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const int key = i * 16 + lr;
      const uint32_t dd = ring + slot * 16384 + part * 8192 + key * 128 + ((c8 ^ (key & 7)) << 4);
      const uint8_t* src = (part ? a.vc : a.kc) + (size_t)max(rw[i], 0) * 128 + c8 * 16;
      if (PO && rw[i] == -2) continue;
      if (L2H) asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;\n" :: "r"(dd), "l"(src), "r"(rw[i] >= 0 ? 16 : 0));
      else asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dd), "l"(src), "r"(rw[i] >= 0 ? 16 : 0));
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
  if (PS1) for (int x = RS; x < RS + 3; x++) ptseg(x);  // joins the first KV group (read from step 0)
  const int nb = (ke - ks + 63) >> 6;
  int rq[4];  // rows of the next step to issue (RS - 1 ahead), looked up a step early
  if constexpr (PO) {
    int rp[RS - 1][4];
#pragma unroll
    for (int j = 0; j < RS - 1; j++)
#pragma unroll
      for (int i = 0; i < 4; i++) rp[j][i] = rowof(ks + j * 64, i);
#pragma unroll
    for (int i2 = 0; i2 < 4; i2++) rq[i2] = rowof(ks + (RS - 1) * 64, i2);
    {  // row qr_ (token ql, q head 8 h + (qr_ & 7)), half qhf: RMSNorm, RoPE at seq + token, E4M3 (while the KV rows load)
      float x[32], ss = 0.f;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const __nv_bfloat162* bb = (const __nv_bfloat162*)&qraw[i];
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
          float sn, cs2; rope_cs(f.invf, seq + ql, (qhf * 32 + i) >> 1, sn, cs2);
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
        const int c = 4 * h + 2 * qhf + u;
        asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" :: "r"(qs + qr_ * 128 + ((c ^ (qr_ & 7)) << 4)), "r"(q4[4 * u]), "r"(q4[4 * u + 1]), "r"(q4[4 * u + 2]), "r"(q4[4 * u + 3]));
      }
    }
#pragma unroll
    for (int j = 0; j < RS - 1; j++) {  // the block's own keys are not in the cache yet: written below, not copied
#pragma unroll
      for (int i = 0; i < 4; i++) { const int kp = ks + j * 64 + i * 16 + lr; if (kp >= seq && kp < ke) rp[j][i] = -2; }
      issue(j, ks + j * 64, rp[j][0], rp[j][1], rp[j][2], rp[j][3]);
    }
    if (ke > seq) {  // a split holding the block's own keys: to the cache and straight into the prefetched ring slots
      float kk4[4], vv4[4], s2 = 0.f;
      const __nv_bfloat16* kr = (const __nv_bfloat16*)&kraw; const __nv_bfloat16* vr = (const __nv_bfloat16*)&vraw;
#pragma unroll
      for (int i = 0; i < 4; i++) { kk4[i] = __bfloat162float(kr[i]); vv4[i] = __bfloat162float(vr[i]); s2 += kk4[i] * kk4[i]; }
#pragma unroll
      for (int o = 1; o < 16; o <<= 1) s2 += __shfl_xor_sync(0xffffffffu, s2, o);
      const float kr2 = rsqrtf(s2 / D + 1e-5f);
#pragma unroll
      for (int i = 0; i < 4; i++) kk4[i] *= kr2;
      if (f.theta > 0.f) {
#pragma unroll
        for (int i = 0; i < 4; i += 2) {
          float sn, cs2; rope_cs(f.invf, seq + nlt, (nd0 + i) >> 1, sn, cs2);
          const float e = kk4[i], o = kk4[i + 1];
          kk4[i] = e * cs2 - o * sn; kk4[i + 1] = e * sn + o * cs2;
        }
      }
      uint32_t kb4 = 0, vb4 = 0;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        __nv_fp8_e4m3 fk(kk4[i] * f.kinv), fv(vv4[i] * f.vinv);
        kb4 |= (uint32_t)fk.__x << (8 * i); vb4 |= (uint32_t)fv.__x << (8 * i);
      }
      if (nlt < lt) {
        const size_t lc = (size_t)f.loc[b * a.L + nlt] * 128 + nkh * 64 + nd0;
        *(uint32_t*)((uint8_t*)a.kc + lc) = kb4;
        *(uint32_t*)((uint8_t*)a.vc + lc) = vb4;
        const int kq = seq + nlt - ks, j = kq >> 6, kk = kq & 63;
        if (kq >= 0 && j < RS - 1 && seq + nlt < ke) {
          const uint32_t dd = ring + j * 16384 + kk * 128 + ((((nkh * 64 + nd0) >> 4) ^ (kk & 7)) << 4) + (nd0 & 15);
          asm volatile("st.shared.u32 [%0], %1;\n" :: "r"(dd), "r"(kb4));
          asm volatile("st.shared.u32 [%0], %1;\n" :: "r"(dd + 8192), "r"(vb4));
        }
      }
      __threadfence();
    }
    __syncthreads();
  } else {
    for (int j = 0; j < RS - 1; j++) issue(j, ks + j * 64, rowof(ks + j * 64, 0), rowof(ks + j * 64, 1), rowof(ks + j * 64, 2), rowof(ks + j * 64, 3));
#pragma unroll
    for (int i2 = 0; i2 < 4; i2++) rq[i2] = rowof(ks + (RS - 1) * 64, i2);
  }
  PS(2);
  float o[32], mx[2] = {-INFINITY, -INFINITY}, ls[2] = {0.f, 0.f};
#pragma unroll
  for (int i = 0; i < 32; i++) if (!AW_Z) o[i] = 0.f;
  const int lw0 = 2 * w;  // tokens of this warp's rows: g -> lw0, g + 8 -> lw0 + 1
  const int lomax = a.window ? max(0, seq + t0 + lt - 1 - a.window + 1) : 0;
  // L2 prefetch PFD steps beyond the ring: thread tid < 128 -> key tid & 63 of K (tid < 64) or V; row looked up a step early
  auto pfrow = [&](int kb) -> int { const int kp = kb + (tid & 63), kc = min(kp, ke - 1); return kp < ke ? (PS1 ? __ldg(ptb + kc) : __ldg(ptb + kc / a.ps) * a.ps + kc % a.ps) : -1; };
  int pr = PFD > 0 && tid < 128 ? pfrow(ks + (RS - 1 + PFD) * 64) : -1;
  for (int bi = 0, kb = ks; bi < nb; bi++, kb += 64) {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(RS - 2));
    if (FR) asm volatile("bar.sync 3, 256;\n" ::: "memory");  // both warpgroups: step bi landed, slot of bi - 1 free
    else asm volatile("bar.sync %0, 128;\n" :: "r"(1 + h) : "memory");  // this warpgroup: step bi landed, slot of bi - 1 free
    if (PS1) ptseg(bi + RS + 3);  // joins this step's group (complete RS - 1 steps later, read at step bi + 3)
    issue((bi + RS - 1) % RS, kb + (RS - 1) * 64, rq[0], rq[1], rq[2], rq[3]);
#pragma unroll
    for (int i2 = 0; i2 < 4; i2++) rq[i2] = rowr(kb + RS * 64, i2);
    if (PFD > 0 && tid < 128) {
#if __CUDA_ARCH__ >= 900
      if (pr >= 0) asm volatile("cp.async.bulk.prefetch.L2.global [%0], 128;\n" :: "l"((tid < 64 ? a.kc : a.vc) + (size_t)pr * 128) : "memory");
#endif
      pr = pfrow(kb + (RS + PFD) * 64);
    }
    // convert this head's V half to f16 (double-buffered: the previous step's PV may still be reading the other one)
    const uint32_t raw = ring + (bi % RS) * 16384, vf = vf0 + (VB ? (bi & 1) * 8192 : 0);
    if (!VB) wg_wait0();  // single V buffer: the previous PV must be done before it is overwritten
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
    bool fast = false;
    if constexpr (FS) fast = __all_sync(0xffffffffu, mx[0] != -INFINITY && mx[1] != -INFINITY);
    if (fast) {
      // P against the running max (no row max, no shuffles); P <= 2^LZT checked on the packed f16 values
      if (!full) {
#pragma unroll
        for (int hh = 0; hh < 2; hh++) {
          const int p = lw0 + hh < lt ? seq + t0 + lw0 + hh : -1, lo = a.window ? max(0, p - a.window + 1) : 0;
#pragma unroll
          for (int j = 0; j < 8; j++)
#pragma unroll
            for (int e = 0; e < 2; e++) { const int kp = kb + 8 * j + 2 * t + e; if (!(kp < ke && kp <= p && kp >= lo)) s[4 * j + 2 * hh + e] = -INFINITY; }
        }
      }
#pragma unroll
      for (int j = 0; j < 8; j++)
#pragma unroll
        for (int hh = 0; hh < 2; hh++)
#pragma unroll
          for (int e = 0; e < 2; e++) s[4 * j + 2 * hh + e] = ex2(fmaf(s[4 * j + 2 * hh + e], a.qk, -mx[hh]));
#pragma unroll
      for (int i = 0; i < 16; i++) pa[i] = pk(s[2 * i], s[2 * i + 1]);
      __half2 hm = *(__half2*)&pa[0];
#pragma unroll
      for (int i = 1; i < 16; i++) hm = __hmax2(hm, *(__half2*)&pa[i]);
      const uint32_t hb = *(uint32_t*)&hm;
      if (__any_sync(0xffffffffu, (hb & 0xffffu) > LZB || (hb >> 16) > LZB)) {  // rare: rescale rows by their P max
        float sc0 = 1.f, sc1 = 1.f;
#pragma unroll
        for (int hh = 0; hh < 2; hh++) {
          float pm = 0.f;
#pragma unroll
          for (int j = 0; j < 8; j++) pm = fmaxf(pm, fmaxf(s[4 * j + 2 * hh], s[4 * j + 2 * hh + 1]));
          pm = fmaxf(pm, __shfl_xor_sync(0xffffffffu, pm, 1));
          pm = fmaxf(pm, __shfl_xor_sync(0xffffffffu, pm, 2));
          if (pm > LZP) {
            const float inv = 1.f / pm;
            mx[hh] += __log2f(pm);
            if (hh) sc1 = inv; else sc0 = inv;
#pragma unroll
            for (int j = 0; j < 8; j++) { s[4 * j + 2 * hh] *= inv; s[4 * j + 2 * hh + 1] *= inv; }
          }
        }
#pragma unroll
        for (int i = 0; i < 16; i++) pa[i] = pk(s[2 * i], s[2 * i + 1]);
#pragma unroll
        for (int j = 0; j < 8; j++) { o[4 * j] *= sc0; o[4 * j + 1] *= sc0; o[4 * j + 2] *= sc1; o[4 * j + 3] *= sc1; }
        lacc[0] *= sc0; lacc[1] *= sc0; lacc[2] *= sc1; lacc[3] *= sc1;
      }
    } else {
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
#if AW_MS
#pragma unroll
      for (int j = 0; j < 8; j++)
#pragma unroll
        for (int e = 0; e < 2; e++) s[4 * j + 2 * hh + e] = ex2(fmaf(s[4 * j + 2 * hh + e], a.qk, -nmf));
#else
      float sum = 0.f;
#pragma unroll
      for (int j = 0; j < 8; j++)
#pragma unroll
        for (int e = 0; e < 2; e++) { const float pv = ex2(fmaf(s[4 * j + 2 * hh + e], a.qk, -nmf)); s[4 * j + 2 * hh + e] = pv; sum += pv; }
      ls[hh] = ls[hh] * scl[hh] + sum;
#endif
#endif
    }
    const bool anyup = AW_LZ ? __any_sync(0xffffffffu, upd[0] | upd[1]) : true;
#pragma unroll
    for (int j = 0; j < 8; j++) {
      if (AW_LZ && !anyup) break;  // warp-uniform: no row of this warp moved its max
      o[4 * j] *= scl[0]; o[4 * j + 1] *= scl[0]; o[4 * j + 2] *= scl[1]; o[4 * j + 3] *= scl[1];
    }
#if AW_MS
    if (!AW_LZ || anyup) { lacc[0] *= scl[0]; lacc[1] *= scl[0]; lacc[2] *= scl[1]; lacc[3] *= scl[1]; }
#endif
#if !AW_H2
#pragma unroll
    for (int i = 0; i < 16; i++) pa[i] = pk(s[2 * i], s[2 * i + 1]);
#endif
    }
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
#if AW_MS
      wg_rs8(lacc, pa[4 * kc], pa[4 * kc + 1], pa[4 * kc + 2], pa[4 * kc + 3], dOnes, kc > 0 || bi > 0);
#endif
#endif
    }
    wg_commit();
#if AW_FOP
    fop(o, 32);
#endif
  }
  wg_wait0();
  PS(3);
#if AW_FOP
  fop(o, 32);
#endif
  if (AW_Z && nb == 0) {
#pragma unroll
    for (int i = 0; i < 32; i++) o[i] = 0.f;
  }
#if AW_MS
  if (nb == 0) { lacc[0] = lacc[1] = lacc[2] = lacc[3] = 0.f; }
#endif
  const size_t it = ((size_t)bq * a.hkv + h) * a.nsplit + sp;
#pragma unroll
  for (int hh = 0; hh < 2; hh++) {
#if AW_MS
    const float l = lacc[2 * hh];  // full row sum (every column of the ones product)
#else
    float l = ls[hh];
    l += __shfl_xor_sync(0xffffffffu, l, 1);
    l += __shfl_xor_sync(0xffffffffu, l, 2);
#endif
    const int row = 16 * w + g + 8 * hh;
    float* wo = a.wo + (it * 64 + row) * D;  // 64 rows per (block, head, split)
#pragma unroll
    for (int j = 0; j < 8; j++) *(float2*)(wo + 8 * j + 2 * t) = make_float2(o[4 * j + 2 * hh], o[4 * j + 2 * hh + 1]);
    if (t == 0) { a.wml[(it * 64 + row) * 2] = mx[hh]; a.wml[(it * 64 + row) * 2 + 1] = l; }
  }
  if constexpr (MRG) {
    // the last min(8, nsplit) arrivers of a request merge its 128 rows (rank, rank + nm, ...); everyone else has arrived
    // arrival: CTA barrier, then one acq_rel add (releases every thread's partial stores; no fence.sc)
    __syncthreads();
    PS(4);
    __shared__ int rank_s;
    if (tid == 0) { int v; asm volatile("atom.add.acq_rel.gpu.global.s32 %0, [%1], 1;\n" : "=r"(v) : "l"(f.cnt + bq) : "memory"); rank_s = v; }
    __syncthreads();
    PS(5);
    const int nmg = min(8, nsp), rank = rank_s - (nsp - nmg);
    if (rank < 0) return;
    const int rr0 = rank + nmg * (tid >> 4);  // first merge row of this thread: its gate values load during the spin
    uint2 gpre = make_uint2(0, 0);
    if (rr0 < 128 && ((rr0 & 63) >> 3) < lt)
      gpre = __ldg((const uint2*)(f.qkv + ((size_t)b * a.L + t0 + ((rr0 & 63) >> 3)) * (2 * 16 + 4) * D + 20 * D + ((rr0 >> 6) * 8 + (rr0 & 7)) * D + (tid & 15) * 4));
    if (tid == 0) {
      int v;
      do { asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(f.cnt + bq) : "memory"); if (v < nsp) __nanosleep(32); } while (v < nsp);
    }
    __syncthreads();
    PS(6);
    asm volatile("cp.async.wait_all;\n" ::: "memory");
    // 16 threads per row (4 dims each), 16 rows at a time; one online-softmax pass over the splits
    const int d4 = tid & 15;
    for (int rr = rank + nmg * (tid >> 4); rr < 128; rr += nmg * 16) {
      const int hh = rr >> 6, r = rr & 63, base = (bq * a.hkv + hh) * a.nsplit;
      // two passes with independent loads: the splits' max first, then the weighted sum (no rescale chain)
      float Mr = -INFINITY, Lr = 0.f;
      float4 a4 = make_float4(0.f, 0.f, 0.f, 0.f);
      if (f.fl & 1) {
        mrg8<MCH>(0, 1, nsp, [&](int k2) { return __ldcg((const float2*)(a.wml + ((size_t)(base + k2) * 64 + r) * 2)); },
             [&](int k2) { return __ldcg((const float4*)(a.wo + ((size_t)(base + k2) * 64 + r) * D) + d4); }, Mr, Lr, a4);
      } else {
#pragma unroll 8
      for (int k2 = 0; k2 < nsp; k2++) Mr = fmaxf(Mr, __ldcg(a.wml + ((size_t)(base + k2) * 64 + r) * 2));
#pragma unroll 8
      for (int k2 = 0; k2 < nsp; k2++) {
        const float2 ml = __ldcg((const float2*)(a.wml + ((size_t)(base + k2) * 64 + r) * 2));
        const float4 v = __ldcg((const float4*)(a.wo + ((size_t)(base + k2) * 64 + r) * D) + d4);
        const float c1 = ml.x == -INFINITY ? 0.f : ex2(ml.x - Mr);
        Lr = fmaf(ml.y, c1, Lr);
        a4.x = fmaf(v.x, c1, a4.x); a4.y = fmaf(v.y, c1, a4.y); a4.z = fmaf(v.z, c1, a4.z); a4.w = fmaf(v.w, c1, a4.w);
      }
      }
      const float inv = f.vscale / Lr;
      a4.x *= inv; a4.y *= inv; a4.z *= inv; a4.w *= inv;
      float ss = a4.x * a4.x + a4.y * a4.y + a4.z * a4.z + a4.w * a4.w;
#pragma unroll
      for (int x = 1; x < 16; x <<= 1) ss += __shfl_xor_sync(0xffffu << (lane & 16), ss, x);
      const float rn = rsqrtf(ss / D + 1e-5f);
      const int l = t0 + (r >> 3), hd = hh * 8 + (r & 7);
      if ((r >> 3) >= lt) continue;
      const uint2 gv = rr == rr0 ? gpre : __ldg((const uint2*)(f.qkv + ((size_t)b * a.L + l) * (2 * 16 + 4) * D + 20 * D + hd * D + d4 * 4));
      const __nv_bfloat16* gt = (const __nv_bfloat16*)&gv;
      const float vals[4] = {a4.x, a4.y, a4.z, a4.w};
      __nv_bfloat16 ob[4];
#pragma unroll
      for (int i2 = 0; i2 < 4; i2++) ob[i2] = __float2bfloat16(vals[i2] * rn / (1.f + expf(-__bfloat162float(gt[i2]))));
      *(uint2*)(f.out + ((size_t)b * a.L + l) * 16 * D + hd * D + d4 * 4) = *(uint2*)ob;
    }
    const int nbq = a.B * nqb;
    PS(7);
    if (tid == 0 && atomicAdd(f.cnt + nbq + bq, 1) == nmg - 1) { f.cnt[bq] = 0; f.cnt[nbq + bq] = 0; }
  }
}



// ---------------------------------------------------------------- attnp: warp-specialized verify (L = 8, page 1, fused)
// CTA = NL lanes (splits of one request) x 2 consumer warpgroups (one per kv head) + 1 producer warpgroup. The producer
// (64 threads per lane, setmaxnreg 32) streams the page table into a smem ring and the K/V rows (8 lanes x 16 B per 128 B
// row) into the lane's RS-slot ring, signalling a full mbarrier per slot (cp.async.mbarrier.arrive.noinc); consumers
// (setmaxnreg 112: no copy address math in their loop) convert their head's V half, QK on wgmma, release the slot on an
// empty mbarrier, softmax, PV. X: 4 fast softmax, 64 L2::128B hint. Block keys: written into the ring by the consumers.
#ifndef PRS
#define PRS RSW
#endif
__device__ __forceinline__ void mb_wait(uint32_t mb, uint32_t ph) {
  uint32_t ok;
  do {
#if __CUDA_ARCH__ >= 900
    asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n" : "=r"(ok) : "r"(mb), "r"(ph) : "memory");
#else
    asm volatile("{\n.reg .pred p;\nmbarrier.test_wait.parity.shared.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n" : "=r"(ok) : "r"(mb), "r"(ph) : "memory");
#endif
  } while (!ok);
}
// 2^x on the FMA/ALU pipes (FA4-style MUFU offload): Cody-Waite split at the nearest integer, degree-3 minimax on [-0.5, 0.5]
// (max rel err 1.4e-4, below P's f16 rounding), integer part added to the exponent field; x clamped at -125 (masked scores)
__device__ __forceinline__ float ex2p(float x) {
  x = fmaxf(x, -125.f);
  const float t = x + 12582912.f, fr = x - (t - 12582912.f);
  const float q = fmaf(fmaf(fmaf(0.0550292665f, fr, 0.242256982f), fr, 0.693253055f), fr, 0.999951339f);
  return __int_as_float(__float_as_int(q) + (__float_as_int(t) << 23));
}
// e4m3 x 2 (low 16 bits of w; hi: bits 16..31) -> f16 x 2 on the integer pipes, scaled by 2^-8 (exponent bias 7 vs 15):
// h = (b & 0x80) << 8 | (b & 0x7F) << 7 per byte (subnormals map to subnormals; the 2^8 goes into the output scale)
__device__ __forceinline__ uint32_t e2h_alu(uint32_t w, bool hi) {
  uint32_t t;
  asm("prmt.b32 %0, %1, 0, %2;" : "=r"(t) : "r"(w), "r"(hi ? 0x3424u : 0x1404u));
  return (t & 0x80008000u) | ((t & 0x7F007F00u) >> 1);
}
#ifndef AP_NOCVT
#define AP_NOCVT 0
#endif
#ifndef AP_NOEXP
#define AP_NOEXP 0
#endif
template <int NL, int X>
__global__ void __launch_bounds__(128 * (2 * NL + 1), 1) attnp(A a, WF f) {
  constexpr int RS = PRS;
  constexpr bool FS = X & 4, L2H = X & 64, MRG = true, BAL = X & 256, PCV = X & 128, ACV = X & 1024, PEX = X & 2048,
                 SSI = (X & 8192) && !(X & 256), PFP = X & 16384, EQ = X & 32768,
                 NOSR = X & 65536, CFM = X & 131072, EINC = X & 262144;  // EINC: consumer setmaxnreg.inc right after the barrier (v58)  // NOSR: no setmaxnreg (launch-bound registers); CFM: closed-form split map  // EQ: query / own-key rows loaded before the CTA barrier  // PFP (sliding): L2-prefetch the likely window's page table first  // SSI (full layers): splits over the page table's capacity, so the producer's
                                                  // page-table loads do not wait for seq; lanes past the context idle  // PEX: 1/4 of the fast-path exps on the FMA pipe  // ACV: V e4m3 -> f16 on the ALU
  auto cv = [&](uint32_t w, bool hi) -> uint32_t { return AP_NOCVT ? w : (ACV ? e2h_alu(w, hi) : e2h(hi ? w >> 16 : w)); };
  const float vsc_ = f.vscale * (ACV ? 256.f : 1.f);
  // PCV: the producer also converts V to f16 (and places the block's own keys), so the consumers' step is wait, QK, softmax, PV
  constexpr uint32_t LS = RS * 16384 + 8192 + 32768 + 2048 + (PCV ? 2048 : 0);  // per lane: ring, Q, V f16 (2 heads x 2), page table, own rows
#ifdef PROF
  const long long ck_e = clock64(); const unsigned long long gt_e = gtm2();
#endif
  PS(0);
  const int wg = threadIdx.x >> 7;
  const bool prod = wg == 2 * NL;
  const int LN = prod ? ((threadIdx.x & 127) >> 6) : (wg >> 1);
  int bq, sp, nsp = a.nsplit;
  bool spare = false;
  if constexpr (BAL) {  // B <= 32: lanes (grid x NL) split the batch's total keys evenly; every warp derives the same map
    const int lane_ = threadIdx.x & 31, G = gridDim.x * NL, gl = blockIdx.x * NL + LN;
    int kb_ = 0;
    if (lane_ < a.B) { const int sq = a.seq[lane_]; kb_ = sq + 8 - (a.window ? max(0, sq - a.window + 1) : 0); }
    const int K = max(1, (int)__reduce_add_sync(0xffffffffu, kb_)), mxk = __reduce_max_sync(0xffffffffu, kb_);
    int nl;
    if constexpr (CFM) {  // closed form: 1 + floor(keys (G - B) / K) splits each (sum <= G), leftover lanes to the first requests
      nl = lane_ < a.B ? min(a.nsplit, 1 + (int)(((unsigned long long)kb_ * (unsigned)max(0, G - a.B)) / (unsigned)K)) : 0;
      const int left = G - (int)__reduce_add_sync(0xffffffffu, nl);
      if (lane_ < a.B && lane_ < left && nl < a.nsplit) nl += 1;
    } else {
      // smallest chunk c with sum ceil(keys / c) <= G: it lies in [K / G, K / (G - B)] (sum ceil(k / c) <= K / c + B, so the upper
      // end is feasible): ~9 halvings instead of ~17, the same map
      int lo = max((K + G - 1) / G, (mxk + a.nsplit - 1) / a.nsplit);
      int hi = G > a.B ? max(lo, (K + G - a.B - 1) / (G - a.B)) : max(lo, mxk);
      while (lo < hi) {
        const int c = (lo + hi) >> 1;
        if ((int)__reduce_add_sync(0xffffffffu, lane_ < a.B ? (kb_ + c - 1) / c : 0) <= G) hi = c; else lo = c + 1;
      }
      nl = lane_ < a.B ? (kb_ + lo - 1) / lo : 0;
    }
    int pre = nl;
#pragma unroll
    for (int o2 = 1; o2 < 32; o2 <<= 1) { const int v2 = __shfl_up_sync(0xffffffffu, pre, o2); if (lane_ >= o2) pre += v2; }
    const int bb = __popc(__ballot_sync(0xffffffffu, lane_ < a.B && pre <= gl));
    spare = bb >= a.B;
    const int bc = spare ? 0 : bb;
    nsp = max(1, __shfl_sync(0xffffffffu, nl, bc));
    sp = spare ? 0 : gl - (__shfl_sync(0xffffffffu, pre, bc) - nsp);
    bq = bc;
  } else {
    const int nsc = a.nsplit / NL;
    bq = blockIdx.x / nsc; sp = (blockIdx.x % nsc) * NL + LN;
  }
  const int b = bq, t0 = 0, lt = 8, nqb = 1;
  if (PFP && prod && LN < NL && !spare && a.window) {
    // the window [seq - 2047, seq + 7] lies in the last 2056 + ~16 % of the page table's capacity for the scored spreads
    // (lengths within +-7.5 % of a shared center); a pure hint, issued before seq lands: correctness never depends on it
    const int cap = a.maxp, lo = max(0, cap - 2056 - (cap >> 3) - (cap >> 5)), n = cap - lo, per = (n + nsp - 1) / nsp;
    const int s0 = lo + sp * per, s1 = min(cap, s0 + per);
    for (int e = s0 + (threadIdx.x & 63) * 32; e < s1; e += 64 * 32)
      asm volatile("prefetch.global.L2 [%0];\n" :: "l"(a.pt + (size_t)b * a.maxp + e));
  }
  extern __shared__ __align__(1024) uint8_t smem[];
  const uint32_t sb = ((uint32_t)__cvta_generic_to_shared(smem) + 1023) & ~1023u;
  const uint32_t ring = sb + LN * LS, qs = ring + RS * 16384, vf0 = qs + 8192, ptr = vf0 + 32768;
  const uint32_t ones = sb + NL * LS, mbar = ones + 1024;  // full[NL][RS], empty[NL][RS], vfree[NL][2]
  const uint32_t newkv = ptr + 2048;  // PCV: the block's own K / V rows (8 tokens x 256 B), written by the consumers
  auto vfreeb = [&](int s) { return mbar + (2 * NL * RS + LN * 2 + s) * 8; };
  auto fullb = [&](int s) { return mbar + (LN * RS + s) * 8; };
  auto emptyb = [&](int s) { return mbar + (NL * RS + LN * RS + s) * 8; };
  if (threadIdx.x == 0) {
    for (int i = 0; i < NL * RS; i++) {
      asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(mbar + i * 8), "r"(64));
      asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(mbar + (NL * RS + i) * 8), "r"(8));
    }
    for (int i = 0; i < 2 * NL; i++) asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(mbar + (2 * NL * RS + i) * 8), "r"(8));
  }
  if (!prod && threadIdx.x < 256) asm volatile("st.shared.u32 [%0], %1;\n" :: "r"(ones + threadIdx.x * 4), "r"(0x3C003C00u));
  const int seq = a.seq[b];  // issued here; nothing before the CTA barrier below consumes it
  uint4 qpre[4];
  float4 ivp[4];  // EQ: the RoPE inverse frequencies of this thread's query pairs and own-key pairs, fetched with the rows
  float2 ivk = make_float2(0.f, 0.f);
  uint2 kpre = make_uint2(0, 0), vpre = make_uint2(0, 0);
  int64_t lpre = 0;
  if (EQ && !prod) {
    const int tid_ = threadIdx.x - 256 * LN, wt_ = tid_ & 127, h_ = tid_ >> 7, qr_ = wt_ >> 1;
    const __nv_bfloat16* src = f.qkv + ((size_t)b * a.L + (qr_ >> 3)) * (2 * 16 + 4) * D + (h_ * 8 + (qr_ & 7)) * D + (wt_ & 1) * 32;
#pragma unroll
    for (int i = 0; i < 4; i++) qpre[i] = *(const uint4*)(src + 8 * i);
    const __nv_bfloat16* krow = f.qkv + ((size_t)b * a.L + (wt_ >> 4)) * (2 * 16 + 4) * D;
    kpre = *(const uint2*)(krow + (16 + h_) * D + (wt_ & 15) * 4); vpre = *(const uint2*)(krow + (18 + h_) * D + (wt_ & 15) * 4);
    lpre = f.loc[b * a.L + (wt_ >> 4)];
    if (a.window) {
#pragma unroll
      for (int i = 0; i < 4; i++) ivp[i] = __ldg((const float4*)(f.invf + (wt_ & 1) * 16) + i);
      ivk = __ldg((const float2*)(f.invf + (wt_ & 15) * 2));
    }
  }
#ifdef PROF
  const long long ck_b0 = clock64(); const unsigned long long gt_b0 = gtm2();
  unsigned long long st4[4] = {0, 0, 0, 0};  // prologue sub-stamps (producer: start, seq landed, page table landed, first KV issued;
                                            // consumer: start, query tile written, own keys done, first step's data)
#endif
  __syncthreads();
#ifdef PROF
  const long long ck_b1 = clock64(); const unsigned long long gt_b1 = gtm2();
  // entry -> before the barrier -> after it -> after setmaxnreg (role 0 consumer tid 0, role 1 producer thread 0 of the lane)
  auto prolog_rec = [&](int role, unsigned long long gts) {
    unsigned long long* q = f.prof + 65536 + 4096 + ((blockIdx.x * NL + LN) * 2 + role) * 8;
    const long long cks = clock64();
    q[0] = gt_e; q[1] = gt_b0; q[2] = gt_b1; q[3] = gts; q[4] = ck_b0 - ck_e; q[5] = ck_b1 - ck_b0; q[6] = cks - ck_b1; q[7] = 1;
  };
#endif
  const int nk = seq + t0 + lt;
  const int k0 = SSI ? 0 : (a.window ? max(0, seq + t0 - a.window + 1) : 0), span = SSI ? a.maxp * a.ps : nk - k0;
  const int ks = k0 + (int)((long long)span * sp / nsp), ke0 = k0 + (int)((long long)span * (sp + 1) / nsp);
  const int ke = SSI ? min(ke0, nk) : ke0;
  const int nb = max(0, (ke - ks + 63) >> 6);
  if (prod) {
#if HAS_WG
    if (!NOSR) asm volatile("setmaxnreg.dec.sync.aligned.u32 32;\n" ::: "memory");
#endif
    const int pt = threadIdx.x & 63;
    if ((NL == 1 && (threadIdx.x & 127) >= 64) || spare) return;
#ifdef PROF
    unsigned long long twe = 0, tp0 = gtm2();
#endif
#ifdef PROF
    st4[0] = gtm2();
    if (f.prof && pt == 0) prolog_rec(1, st4[0]);
    { int sq_; asm volatile("mov.b32 %0, %1;" : "=r"(sq_) : "r"(seq)); st4[1] = gtm2() + (sq_ & 0x80000000 ? 1 : 0); }
#endif
    const int* ptb = a.pt + (size_t)b * a.maxp;
    auto ptload = [&](int x) {  // bound: the split's end over the capacity (SSI: no wait for seq; past-seq entries unused)
      const int idx = ks + x * 64 + pt;
      asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" :: "r"(ptr + ((x & 7) * 64 + pt) * 4), "l"(ptb + (idx < ke0 ? idx : 0)), "r"(idx < ke0 ? 4 : 0));
    };
#pragma unroll
    for (int x = 0; x < 7; x++) ptload(x);
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    asm volatile("bar.sync %0, 64;\n" :: "r"(1 + LN) : "memory");
#ifdef PROF
    st4[2] = gtm2();
#endif
    const int c8 = pt & 7, kr = pt >> 3;
    if constexpr (PCV) {
      auto load_step = [&](int y) {  // copies of step y into its slot (an empty group past the range keeps the count)
        const int slot = y % RS;
        if (y < nb) {
#ifdef PROF
          const unsigned long long tw0 = gtm2();
#endif
          if (y >= RS) mb_wait(emptyb(slot), ((y / RS) - 1) & 1);
#ifdef PROF
          twe += gtm2() - tw0;
#endif
          const int kb = ks + y * 64;
          uint32_t rw[8];
#pragma unroll
          for (int i = 0; i < 8; i++) rw[i] = lds32a(ptr + ((y & 7) * 64 + i * 8 + kr) * 4);
          asm volatile("bar.sync %0, 64;\n" :: "r"(1 + LN) : "memory");
          ptload(y + 7);
#pragma unroll
          for (int i = 0; i < 8; i++) {
            const int key = i * 8 + kr, kp = kb + key;
            if (kp >= seq && kp < ke) continue;
            const uint32_t n = kp < ke ? 16 : 0;
            const uint32_t dd = ring + slot * 16384 + key * 128 + ((c8 ^ (key & 7)) << 4);
            const size_t off = (size_t)(kp < ke ? rw[i] : 0) * 128 + c8 * 16;
            if (L2H) {
              asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;\n" :: "r"(dd), "l"(a.kc + off), "r"(n));
              asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;\n" :: "r"(dd + 8192), "l"(a.vc + off), "r"(n));
            } else {
              asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dd), "l"(a.kc + off), "r"(n));
              asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dd + 8192), "l"(a.vc + off), "r"(n));
            }
          }
        }
        asm volatile("cp.async.commit_group;\n");
      };
#pragma unroll 1
      for (int y = 0; y < RS - 1; y++) load_step(y);
      asm volatile("bar.sync %0, 320;\n" :: "r"(3 + LN) : "memory");  // the consumers' own-key rows are in newkv
      for (int x = 0; x < nb; x++) {
        asm volatile("cp.async.wait_group %0;\n" :: "n"(RS - 2) : "memory");  // step x landed (this thread's copies)
        asm volatile("bar.sync %0, 64;\n" :: "r"(1 + LN) : "memory");  // ... and every producer thread's
        const int slot = x % RS, kb = ks + x * 64;
        const uint32_t raw = ring + slot * 16384;
        if (kb <= seq + 7 && kb + 64 > seq) {  // the block's own rows (K and V, both heads) into the slot
#pragma unroll
          for (int u = 0; u < 2; u++) {
            const int idx = pt * 2 + u, tk = idx >> 4, part = (idx >> 3) & 1, c = idx & 7, kq = seq + tk - kb;
            if (kq >= 0 && kq < 64 && seq + tk < ke) {
              uint4 v;
              asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(newkv + tk * 256 + part * 128 + c * 16));
              asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" :: "r"(raw + part * 8192 + kq * 128 + ((c ^ (kq & 7)) << 4)), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
            }
          }
          asm volatile("bar.sync %0, 64;\n" :: "r"(1 + LN) : "memory");
        }
        if (x >= 2) mb_wait(vfreeb(x & 1), ((x - 2) >> 1) & 1);  // PV of step x - 2 retired
        {
          const int key = pt;
          const uint32_t vb = vf0 + (x & 1) * 8192 + key * 128;
#pragma unroll
          for (int c = 0; c < 8; c++) {
            uint4 v;
            asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(raw + 8192 + key * 128 + ((c ^ (key & 7)) << 4)));
            const int j0 = 2 * (c & 3);
            const uint32_t dst = vb + (c >> 2) * 16384;
            asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" :: "r"(dst + ((j0 ^ (key & 7)) << 4)), "r"(cv(v.x, false)), "r"(cv(v.x, true)), "r"(cv(v.y, false)), "r"(cv(v.y, true)));
            asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" :: "r"(dst + (((j0 + 1) ^ (key & 7)) << 4)), "r"(cv(v.z, false)), "r"(cv(v.z, true)), "r"(cv(v.w, false)), "r"(cv(v.w, true)));
          }
        }
        fence_async();  // this thread's K rows (cp.async), own rows and f16 V before the consumers' wgmma
        { uint64_t st_; asm volatile("mbarrier.arrive.shared.b64 %0, [%1];\n" : "=l"(st_) : "r"(fullb(slot)) : "memory"); }
        load_step(x + RS - 1);
      }
    } else
    for (int x = 0; x < nb; x++) {
      const int slot = x % RS;
#ifdef PROF
      const unsigned long long tw0 = gtm2();
#endif
      if (x >= RS) mb_wait(emptyb(slot), ((x / RS) - 1) & 1);
#ifdef PROF
      twe += gtm2() - tw0;
#endif
      const int kb = ks + x * 64;
      uint32_t rw[8];
#pragma unroll
      for (int i = 0; i < 8; i++) rw[i] = lds32a(ptr + ((x & 7) * 64 + i * 8 + kr) * 4);
      asm volatile("bar.sync %0, 64;\n" :: "r"(1 + LN) : "memory");  // every entry of step x read before ptload(x + 7) reuses the slot
      ptload(x + 7);
#pragma unroll
      for (int i = 0; i < 8; i++) {
        const int key = i * 8 + kr, kp = kb + key;
        if (kp >= seq && kp < ke) continue;  // the block's own keys: written by the consumers
        const uint32_t n = kp < ke ? 16 : 0;
        const uint32_t dd = ring + slot * 16384 + key * 128 + ((c8 ^ (key & 7)) << 4);
        const size_t off = (size_t)(kp < ke ? rw[i] : 0) * 128 + c8 * 16;
        if (L2H) {
          asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;\n" :: "r"(dd), "l"(a.kc + off), "r"(n));
          asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;\n" :: "r"(dd + 8192), "l"(a.vc + off), "r"(n));
        } else {
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dd), "l"(a.kc + off), "r"(n));
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dd + 8192), "l"(a.vc + off), "r"(n));
        }
      }
      asm volatile("cp.async.commit_group;\n");
      asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" :: "r"(fullb(slot)) : "memory");
#ifdef PROF
      if (x == 0) st4[3] = gtm2();
#endif
      asm volatile("cp.async.wait_group 6;\n" ::: "memory");  // page-table entries of step x + 1 (issued at step x - 6) landed
      asm volatile("bar.sync %0, 64;\n" :: "r"(1 + LN) : "memory");
    }
    asm volatile("cp.async.wait_all;\n" ::: "memory");
#ifdef PROF
    if (f.prof && pt == 0) {
      unsigned long long* q = f.prof + 65536 + (blockIdx.x * NL + LN) * 4; q[1] = twe; q[3] = gtm2() - tp0;
      unsigned long long* q2 = f.prof + 65536 + 2048 + (blockIdx.x * NL + LN) * 8;
      for (int k = 0; k < 4; k++) q2[k] = st4[k];
    }
#endif
    return;
  }
#if HAS_WG
  if (EINC && !NOSR) asm volatile("setmaxnreg.inc.sync.aligned.u32 112;\n" ::: "memory");
#endif
  if (spare) return;
  const int tid = threadIdx.x - 256 * LN, wt = tid & 127, h = tid >> 7, w = wt >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int wbar = 9 + wg;
#ifdef PROF
  st4[0] = gtm2();
  if (f.prof && tid == 0) prolog_rec(0, st4[0]);
#endif
  // query rows: row qr_ (token qr_ >> 3, q head 8 h + (qr_ & 7)), half wt & 1: RMSNorm, RoPE at seq + token, E4M3
  {
    const int qr_ = wt >> 1, qhf = wt & 1, ql = qr_ >> 3;
    const __nv_bfloat16* src = f.qkv + ((size_t)b * a.L + ql) * (2 * 16 + 4) * D + (h * 8 + (qr_ & 7)) * D + qhf * 32;
    float x[32], ss = 0.f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const uint4 u = EQ ? qpre[i] : *(const uint4*)(src + 8 * i);
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
        float sn, cs2;
        if (EQ) { const float4 v4 = ivp[i >> 3]; const int k = (i >> 1) & 3; rope_cs2(k == 0 ? v4.x : k == 1 ? v4.y : k == 2 ? v4.z : v4.w, seq + ql, sn, cs2); }
        else rope_cs(f.invf, seq + ql, (qhf * 32 + i) >> 1, sn, cs2);
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
      const int c = 4 * h + 2 * qhf + u;
      asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" :: "r"(qs + qr_ * 128 + ((c ^ (qr_ & 7)) << 4)), "r"(q4[4 * u]), "r"(q4[4 * u + 1]), "r"(q4[4 * u + 2]), "r"(q4[4 * u + 3]));
    }
  }
  // the block's own key / value of head h: token nt, dims nd0 .. nd0 + 3 (E4M3); cache store by the split holding them
  const int nt = wt >> 4, nd0 = (wt & 15) * 4;
  uint32_t kb4 = 0, vb4 = 0;
  const __nv_bfloat16* krow = f.qkv + ((size_t)b * a.L + nt) * (2 * 16 + 4) * D;
#ifdef PROF
  st4[1] = gtm2();
#endif
  const uint2 kraw = EQ ? kpre : *(const uint2*)(krow + (16 + h) * D + nd0), vraw = EQ ? vpre : *(const uint2*)(krow + (18 + h) * D + nd0);
  const int64_t lcr = EQ ? lpre : f.loc[b * a.L + nt];
  if (ke > seq) {
    float kk4[4], vv4[4], s2 = 0.f;
    const __nv_bfloat16* kr = (const __nv_bfloat16*)&kraw; const __nv_bfloat16* vr = (const __nv_bfloat16*)&vraw;
#pragma unroll
    for (int i = 0; i < 4; i++) { kk4[i] = __bfloat162float(kr[i]); vv4[i] = __bfloat162float(vr[i]); s2 += kk4[i] * kk4[i]; }
#pragma unroll
    for (int o2 = 1; o2 < 16; o2 <<= 1) s2 += __shfl_xor_sync(0xffffffffu, s2, o2);
    const float kr2 = rsqrtf(s2 / D + 1e-5f);
#pragma unroll
    for (int i = 0; i < 4; i++) kk4[i] *= kr2;
    if (f.theta > 0.f) {
#pragma unroll
      for (int i = 0; i < 4; i += 2) {
        float sn, cs2;
        if (EQ) rope_cs2(i ? ivk.y : ivk.x, seq + nt, sn, cs2);
        else rope_cs(f.invf, seq + nt, (nd0 + i) >> 1, sn, cs2);
        const float e = kk4[i], o = kk4[i + 1];
        kk4[i] = e * cs2 - o * sn; kk4[i + 1] = e * sn + o * cs2;
      }
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
      __nv_fp8_e4m3 fk(kk4[i] * f.kinv), fv(vv4[i] * f.vinv);
      kb4 |= (uint32_t)fk.__x << (8 * i); vb4 |= (uint32_t)fv.__x << (8 * i);
    }
    if (seq + nt < ke && seq + nt >= ks) {
      const size_t lc = (size_t)lcr * 128 + h * 64 + nd0;
      *(uint32_t*)((uint8_t*)a.kc + lc) = kb4;
      *(uint32_t*)((uint8_t*)a.vc + lc) = vb4;
    }
    if (PCV) {
      asm volatile("st.shared.u32 [%0], %1;\n" :: "r"(newkv + nt * 256 + h * 64 + nd0), "r"(kb4));
      asm volatile("st.shared.u32 [%0], %1;\n" :: "r"(newkv + nt * 256 + 128 + h * 64 + nd0), "r"(vb4));
    }
  }
  if (PCV) asm volatile("bar.sync %0, 320;\n" :: "r"(3 + LN) : "memory");
#ifdef PROF
  st4[2] = gtm2();
#endif
  fence_async();
  asm volatile("bar.sync %0, 128;\n" :: "r"(wbar) : "memory");
  PS(2);
#if HAS_WG
  if (!EINC && !NOSR) asm volatile("setmaxnreg.inc.sync.aligned.u32 112;\n" ::: "memory");  // late: after the query prologue
#endif
#if AW_MS
  const uint64_t dOnes = (uint64_t)((ones & 0x3FFFF) >> 4) | ((uint64_t)(128 >> 4) << 16) | ((uint64_t)(256 >> 4) << 32);
  float lacc[4];
#endif
  float o[32], mx[2] = {-INFINITY, -INFINITY}, ls[2] = {0.f, 0.f};
#pragma unroll
  for (int i = 0; i < 32; i++) if (!AW_Z) o[i] = 0.f;
  const int lw0 = 2 * w;
  const int lomax = a.window ? max(0, seq + t0 + lt - 1 - a.window + 1) : 0;
#ifdef PROF
  unsigned long long twf = 0, tc0 = gtm2();
#endif
  for (int bi = 0, kb = ks; bi < nb; bi++, kb += 64) {
    const int slot = bi % RS;
    const uint32_t raw = ring + slot * 16384, vf = vf0 + h * 16384 + (bi & 1) * 8192;
#ifdef PROF
    const unsigned long long tw0 = gtm2();
#endif
    mb_wait(fullb(slot), (bi / RS) & 1);
#ifdef PROF
    twf += gtm2() - tw0;
    if (bi == 0) st4[3] = gtm2();
#endif
    if (!PCV && kb <= seq + 7 && kb + 64 > seq) {  // the block's own keys into this head's half of the slot
      const int kq = seq + nt - kb;
      if (kq >= 0 && kq < 64 && seq + nt < ke) {
        const uint32_t dd = raw + kq * 128 + ((((h * 64 + nd0) >> 4) ^ (kq & 7)) << 4) + (nd0 & 15);
        asm volatile("st.shared.u32 [%0], %1;\n" :: "r"(dd), "r"(kb4));
        asm volatile("st.shared.u32 [%0], %1;\n" :: "r"(dd + 8192), "r"(vb4));
      }
      asm volatile("bar.sync %0, 128;\n" :: "r"(wbar) : "memory");
    }
    if (!PCV) {
      const int key = wt >> 1;
#pragma unroll
      for (int u = 0; u < 2; u++) {
        const int c = 4 * h + 2 * (wt & 1) + u;
        uint4 x;
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(x.x), "=r"(x.y), "=r"(x.z), "=r"(x.w) : "r"(raw + 8192 + key * 128 + ((c ^ (key & 7)) << 4)));
        const int j0 = 2 * (c - 4 * h);
        const uint32_t dst = vf + key * 128;
        asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" :: "r"(dst + ((j0 ^ (key & 7)) << 4)), "r"(cv(x.x, false)), "r"(cv(x.x, true)), "r"(cv(x.y, false)), "r"(cv(x.y, true)));
        asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" :: "r"(dst + (((j0 + 1) ^ (key & 7)) << 4)), "r"(cv(x.z, false)), "r"(cv(x.z, true)), "r"(cv(x.w, false)), "r"(cv(x.w, true)));
      }
    }
    if (!PCV) {
      fence_async();
      asm volatile("bar.sync %0, 128;\n" :: "r"(wbar) : "memory");
    }
    float s[32];
    wg_fence();
    wg_ss8<0>(s, gdesc(qs + 64 * h, 16, 1024), gdesc(raw + 64 * h, 16, 1024));
    wg_ss8<1>(s, gdesc(qs + 64 * h + 32, 16, 1024), gdesc(raw + 64 * h + 32, 16, 1024));
    wg_commit(); wg_wait0();  // also retires the previous step's PV
    __syncwarp();
    if (lane == 0) {
      uint64_t st_; asm volatile("mbarrier.arrive.shared.b64 %0, [%1];\n" : "=l"(st_) : "r"(emptyb(slot)) : "memory");
      if (PCV && bi > 0) asm volatile("mbarrier.arrive.shared.b64 %0, [%1];\n" : "=l"(st_) : "r"(vfreeb((bi - 1) & 1)) : "memory");  // PV of bi - 1 retired
    }
#if AW_FOP
    fop(o, 32); fop(s, 32);
#endif
    const bool full = kb + 64 <= ke && kb + 64 <= seq + t0 + 1 && kb >= lomax && lt == 8;
    float scl[2];
    bool upd[2] = {true, true};
    uint32_t pa[16];
    bool fast = false;
    if constexpr (FS) fast = __all_sync(0xffffffffu, mx[0] != -INFINITY && mx[1] != -INFINITY);
    if (fast) {
      // P against the running max (no row max, no shuffles); P <= 2^LZT checked on the packed f16 values
      if (!full) {
#pragma unroll
        for (int hh = 0; hh < 2; hh++) {
          const int p = lw0 + hh < lt ? seq + t0 + lw0 + hh : -1, lo = a.window ? max(0, p - a.window + 1) : 0;
#pragma unroll
          for (int j = 0; j < 8; j++)
#pragma unroll
            for (int e = 0; e < 2; e++) { const int kp = kb + 8 * j + 2 * t + e; if (!(kp < ke && kp <= p && kp >= lo)) s[4 * j + 2 * hh + e] = -INFINITY; }
        }
      }
#pragma unroll
      for (int j = 0; j < 8; j++)
#pragma unroll
        for (int hh = 0; hh < 2; hh++)
#pragma unroll
          for (int e = 0; e < 2; e++) s[4 * j + 2 * hh + e] = AP_NOEXP ? fmaf(s[4 * j + 2 * hh + e], a.qk, -mx[hh]) : ((PEX && (j & 3) == 0) ? ex2p(fmaf(s[4 * j + 2 * hh + e], a.qk, -mx[hh])) : ex2(fmaf(s[4 * j + 2 * hh + e], a.qk, -mx[hh])));
#pragma unroll
      for (int i = 0; i < 16; i++) pa[i] = pk(s[2 * i], s[2 * i + 1]);
      __half2 hm = *(__half2*)&pa[0];
#pragma unroll
      for (int i = 1; i < 16; i++) hm = __hmax2(hm, *(__half2*)&pa[i]);
      const uint32_t hb = *(uint32_t*)&hm;
      if (__any_sync(0xffffffffu, (hb & 0xffffu) > LZB || (hb >> 16) > LZB)) {  // rare: rescale rows by their P max
        float sc0 = 1.f, sc1 = 1.f;
#pragma unroll
        for (int hh = 0; hh < 2; hh++) {
          float pm = 0.f;
#pragma unroll
          for (int j = 0; j < 8; j++) pm = fmaxf(pm, fmaxf(s[4 * j + 2 * hh], s[4 * j + 2 * hh + 1]));
          pm = fmaxf(pm, __shfl_xor_sync(0xffffffffu, pm, 1));
          pm = fmaxf(pm, __shfl_xor_sync(0xffffffffu, pm, 2));
          if (pm > LZP) {
            const float inv = 1.f / pm;
            mx[hh] += __log2f(pm);
            if (hh) sc1 = inv; else sc0 = inv;
#pragma unroll
            for (int j = 0; j < 8; j++) { s[4 * j + 2 * hh] *= inv; s[4 * j + 2 * hh + 1] *= inv; }
          }
        }
#pragma unroll
        for (int i = 0; i < 16; i++) pa[i] = pk(s[2 * i], s[2 * i + 1]);
#pragma unroll
        for (int j = 0; j < 8; j++) { o[4 * j] *= sc0; o[4 * j + 1] *= sc0; o[4 * j + 2] *= sc1; o[4 * j + 3] *= sc1; }
        lacc[0] *= sc0; lacc[1] *= sc0; lacc[2] *= sc1; lacc[3] *= sc1;
      }
    } else {
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
#if AW_MS
#pragma unroll
      for (int j = 0; j < 8; j++)
#pragma unroll
        for (int e = 0; e < 2; e++) s[4 * j + 2 * hh + e] = AP_NOEXP ? fmaf(s[4 * j + 2 * hh + e], a.qk, -nmf) : ex2(fmaf(s[4 * j + 2 * hh + e], a.qk, -nmf));
#else
      float sum = 0.f;
#pragma unroll
      for (int j = 0; j < 8; j++)
#pragma unroll
        for (int e = 0; e < 2; e++) { const float pv = AP_NOEXP ? fmaf(s[4 * j + 2 * hh + e], a.qk, -nmf) : ex2(fmaf(s[4 * j + 2 * hh + e], a.qk, -nmf)); s[4 * j + 2 * hh + e] = pv; sum += pv; }
      ls[hh] = ls[hh] * scl[hh] + sum;
#endif
#endif
    }
    const bool anyup = AW_LZ ? __any_sync(0xffffffffu, upd[0] | upd[1]) : true;
#pragma unroll
    for (int j = 0; j < 8; j++) {
      if (AW_LZ && !anyup) break;  // warp-uniform: no row of this warp moved its max
      o[4 * j] *= scl[0]; o[4 * j + 1] *= scl[0]; o[4 * j + 2] *= scl[1]; o[4 * j + 3] *= scl[1];
    }
#if AW_MS
    if (!AW_LZ || anyup) { lacc[0] *= scl[0]; lacc[1] *= scl[0]; lacc[2] *= scl[1]; lacc[3] *= scl[1]; }
#endif
#if !AW_H2
#pragma unroll
    for (int i = 0; i < 16; i++) pa[i] = pk(s[2 * i], s[2 * i + 1]);
#endif
    }
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
#if AW_MS
      wg_rs8(lacc, pa[4 * kc], pa[4 * kc + 1], pa[4 * kc + 2], pa[4 * kc + 3], dOnes, kc > 0 || bi > 0);
#endif
#endif
    }
    wg_commit();
#if AW_FOP
    fop(o, 32);
#endif
  }
  wg_wait0();
  PS(3);
#ifdef PROF
  if (f.prof && tid == 0) {
    unsigned long long* q = f.prof + 65536 + (blockIdx.x * NL + LN) * 4; q[0] = twf; q[2] = gtm2() - tc0;
    unsigned long long* q2 = f.prof + 65536 + 2048 + (blockIdx.x * NL + LN) * 8 + 4;
    for (int k = 0; k < 4; k++) q2[k] = st4[k];
  }
#endif
#if AW_FOP
  fop(o, 32);
#endif
  if (AW_Z && nb == 0) {
#pragma unroll
    for (int i = 0; i < 32; i++) o[i] = 0.f;
  }
#if AW_MS
  if (nb == 0) { lacc[0] = lacc[1] = lacc[2] = lacc[3] = 0.f; }
#endif
  const size_t it = ((size_t)bq * a.hkv + h) * a.nsplit + sp;
#pragma unroll
  for (int hh = 0; hh < 2; hh++) {
#if AW_MS
    const float l = lacc[2 * hh];  // full row sum (every column of the ones product)
#else
    float l = ls[hh];
    l += __shfl_xor_sync(0xffffffffu, l, 1);
    l += __shfl_xor_sync(0xffffffffu, l, 2);
#endif
    const int row = 16 * w + g + 8 * hh;
    float* wo = a.wo + (it * 64 + row) * D;  // 64 rows per (block, head, split)
#pragma unroll
    for (int j = 0; j < 8; j++) *(float2*)(wo + 8 * j + 2 * t) = make_float2(o[4 * j + 2 * hh], o[4 * j + 2 * hh + 1]);
    if (t == 0) { a.wml[(it * 64 + row) * 2] = mx[hh]; a.wml[(it * 64 + row) * 2 + 1] = l; }
  }
  if constexpr (MRG) {
    // the last min(8, nsplit) arrivers of a request merge its 128 rows (rank, rank + nm, ...); everyone else has arrived
    // arrival: CTA barrier, then one acq_rel add (releases every thread's partial stores; no fence.sc)
    asm volatile("bar.sync %0, 256;\n" :: "r"(13 + LN) : "memory");
    PS(4);
    __shared__ int rank_s2[2];
    if (tid == 0) { int v; asm volatile("atom.add.acq_rel.gpu.global.s32 %0, [%1], 1;\n" : "=r"(v) : "l"(f.cnt + bq) : "memory"); rank_s2[LN] = v; }
    asm volatile("bar.sync %0, 256;\n" :: "r"(13 + LN) : "memory");
    PS(5);
    // the last nmg (largest power of two <= min(128, splits)) arrivers merge: merger rank takes rows rank, rank + nmg, ...;
    // within a merger, G split groups x 16 threads (4 dims each) per row, combined in shared memory
    const int nmg = 1 << (31 - __clz(min(f.nmg, nsp))), rank = rank_s2[LN] - (nsp - nmg);  // f.nmg: host cap (8, or 128 when every lane is co-resident and wanted)
    if (rank < 0) return;
    const int R = 128 / nmg, G = R >= 16 ? 1 : 16 / R, RP = 16 / G;  // rows per merger, split groups, rows per pass
    const int d4 = tid & 15, sl = tid >> 4, kg = sl % G, rl = sl / G;
    const int rr0 = rank + nmg * rl;  // first-pass row: its gate values load during the spin
    uint2 gpre = make_uint2(0, 0);
    if (kg == 0 && rr0 < 128 && ((rr0 & 63) >> 3) < lt)
      gpre = __ldg((const uint2*)(f.qkv + ((size_t)b * a.L + t0 + ((rr0 & 63) >> 3)) * (2 * 16 + 4) * D + 20 * D + ((rr0 >> 6) * 8 + (rr0 & 7)) * D + d4 * 4));
    if (tid == 0) {
      int v;
      do { asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(f.cnt + bq) : "memory"); if (v < nsp) __nanosleep(32); } while (v < nsp);
    }
    asm volatile("bar.sync %0, 256;\n" :: "r"(13 + LN) : "memory");
    PS(6);
    float* xo = (float*)(smem + (ring - (uint32_t)__cvta_generic_to_shared(smem)));  // [16 slots][64] partial O, then [32] (m, l)
    float* xm = xo + 16 * 64;
    for (int p0 = 0; p0 < R; p0 += RP) {
      const int rr = rank + nmg * (p0 + rl), hh = rr >> 6, r = rr & 63, base = (bq * a.hkv + hh) * a.nsplit;
      float Mr = -INFINITY, Lr = 0.f;
      float4 a4 = make_float4(0.f, 0.f, 0.f, 0.f);
      mrg8<MCH>(kg, G, nsp, [&](int k2) { return __ldcg((const float2*)(a.wml + ((size_t)(base + k2) * 64 + r) * 2)); },
           [&](int k2) { return __ldcg((const float4*)(a.wo + ((size_t)(base + k2) * 64 + r) * D) + d4); }, Mr, Lr, a4);
      if (G > 1) {
        ((float4*)(xo + sl * 64))[d4] = a4;
        if (d4 == 0) { xm[sl] = Mr; xm[16 + sl] = Lr; }
        asm volatile("bar.sync %0, 256;\n" :: "r"(13 + LN) : "memory");
        if (kg == 0) {
          float M2 = -INFINITY;
          for (int q = 0; q < G; q++) M2 = fmaxf(M2, xm[sl + q]);
          a4 = make_float4(0.f, 0.f, 0.f, 0.f); Lr = 0.f;
          for (int q = 0; q < G; q++) {
            const float mq = xm[sl + q], c1 = mq == -INFINITY ? 0.f : ex2(mq - M2);
            const float4 v = ((const float4*)(xo + (sl + q) * 64))[d4];
            Lr = fmaf(xm[16 + sl + q], c1, Lr);
            a4.x = fmaf(v.x, c1, a4.x); a4.y = fmaf(v.y, c1, a4.y); a4.z = fmaf(v.z, c1, a4.z); a4.w = fmaf(v.w, c1, a4.w);
          }
        }
        asm volatile("bar.sync %0, 256;\n" :: "r"(13 + LN) : "memory");  // slots reused by the next pass
      }
      if (kg != 0) continue;
      const float inv = vsc_ / Lr;
      a4.x *= inv; a4.y *= inv; a4.z *= inv; a4.w *= inv;
      float ss = a4.x * a4.x + a4.y * a4.y + a4.z * a4.z + a4.w * a4.w;
#pragma unroll
      for (int x = 1; x < 16; x <<= 1) ss += __shfl_xor_sync(0xffffu << (lane & 16), ss, x);
      const float rn = rsqrtf(ss / D + 1e-5f);
      const int l = t0 + (r >> 3), hd = hh * 8 + (r & 7);
      if ((r >> 3) >= lt) continue;
      const uint2 gv = p0 == 0 ? gpre : __ldg((const uint2*)(f.qkv + ((size_t)b * a.L + l) * (2 * 16 + 4) * D + 20 * D + hd * D + d4 * 4));
      const __nv_bfloat16* gt = (const __nv_bfloat16*)&gv;
      const float vals[4] = {a4.x, a4.y, a4.z, a4.w};
      __nv_bfloat16 ob[4];
#pragma unroll
      for (int i2 = 0; i2 < 4; i2++) ob[i2] = __float2bfloat16(vals[i2] * rn / (1.f + expf(-__bfloat162float(gt[i2]))));
      *(uint2*)(f.out + ((size_t)b * a.L + l) * 16 * D + hd * D + d4 * 4) = *(uint2*)ob;
    }
    const int nbq = a.B * nqb;
    PS(7);
    if (tid == 0 && atomicAdd(f.cnt + nbq + bq, 1) == nmg - 1) { f.cnt[bq] = 0; f.cnt[nbq + bq] = 0; }
  }
}
__global__ void gtres_k(unsigned long long* o) {  // globaltimer step sizes (ns) and the clock64 cycles between steps
  unsigned long long p, t; long long c = clock64(), c2;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(p));
  for (int n = 0; n < 16;) {
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
    if (t != p) { c2 = clock64(); o[2 * n] = t - p; o[2 * n + 1] = (unsigned long long)(c2 - c); p = t; c = c2; n++; }
  }
}
void gt_res(torch::Tensor o) { gtres_k<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>((unsigned long long*)o.data_ptr()); }
template <int NL, int X>
static void apf(int g, int smw, cudaStream_t st, const A& a, const WF& f) {
  static bool o = false;
  if (!o) { cudaFuncSetAttribute(attnp<NL, X>, cudaFuncAttributeMaxDynamicSharedMemorySize, smw); o = true; }
  attnp<NL, X><<<g, 128 * (2 * NL + 1), smw, st>>>(a, f);
}
static void ap_x(int nl, int x, int g, int smw, cudaStream_t st, const A& a, const WF& f) {
  if (nl == 1) { if (x & 32768) apf<1, 32836>(g, smw, st, a, f); else if (x & 8192) apf<1, 8260>(g, smw, st, a, f); else if (x & 2048) apf<1, 2372>(g, smw, st, a, f); else if (x & 1024) apf<1, 1348>(g, smw, st, a, f); else if (x & 128) apf<1, 452>(g, smw, st, a, f); else if (x & 256) apf<1, 324>(g, smw, st, a, f); else if (x & 64) apf<1, 68>(g, smw, st, a, f); else apf<1, 4>(g, smw, st, a, f); return; }
  switch (x) {
    case 8260: apf<2, 8260>(g, smw, st, a, f); break;
    case 32836: apf<2, 32836>(g, smw, st, a, f); break;
    case 98372: apf<2, 98372>(g, smw, st, a, f); break;
    case 65604: apf<2, 65604>(g, smw, st, a, f); break;
    case 131396: apf<2, 131396>(g, smw, st, a, f); break;
    case 294980: apf<2, 294980>(g, smw, st, a, f); break;
    case 262212: apf<2, 262212>(g, smw, st, a, f); break;
    case 16452: apf<2, 16452>(g, smw, st, a, f); break;
    case 8196: apf<2, 8196>(g, smw, st, a, f); break;
    case 1348: apf<2, 1348>(g, smw, st, a, f); break;
    case 2372: apf<2, 2372>(g, smw, st, a, f); break;
    case 2116: apf<2, 2116>(g, smw, st, a, f); break;
    case 1092: apf<2, 1092>(g, smw, st, a, f); break;
    case 196: apf<2, 196>(g, smw, st, a, f); break;
    case 452: apf<2, 452>(g, smw, st, a, f); break;
    case 324: apf<2, 324>(g, smw, st, a, f); break;
    case 260: apf<2, 260>(g, smw, st, a, f); break;
    case 0: apf<2, 0>(g, smw, st, a, f); break;
    case 64: apf<2, 64>(g, smw, st, a, f); break;
    case 68: apf<2, 68>(g, smw, st, a, f); break;
    default: apf<2, 4>(g, smw, st, a, f);
  }
}


// ---------------------------------------------------------------- attnd: decode (L = 1, page 1) on wgmma with keys as M
// One consumer warpgroup per split lane handles both kv heads of a request: S^T = K Q'^T (m64n16k32 e4m3 x 4, A = the raw
// 64-key x 128 B K rows of the ring, B = Q' = 16 q-head rows x 128 B with each head's other 64 B zero), P^T relative to a
// warpgroup-uniform running max per q head (exact max on the first step and when some P > 2^8), P^T (16 x 64 f16, K-major
// SW128) to smem, O^T_h += V_h^T P^T (m64n16k16 f16, A = the f16 V tile MN-major). The producer warpgroup is attnp's.
__device__ __forceinline__ void wg_n16_qk(float* d, uint64_t da, uint64_t db, int sd) {
#if HAS_WG
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %10, 0;\nwgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7}, %8, %9, p, 1, 1;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]) : "l"(da), "l"(db), "r"(sd));
#elif WGEMU
  const int wt = threadIdx.x & 127, w = wt >> 5, g = (wt & 31) >> 2, t = wt & 3;
  for (int i = 0; i < 8; i++) {
    const int hh = (i >> 1) & 1, e = i & 1, j = i >> 2, m = 16 * w + g + 8 * hh, n = 8 * j + 2 * t + e;
    float acc = 0.f;
    for (int k = 0; k < 32; k++) acc += emu_e4m3(emu_ld8(emu_kmaj(da, m, k))) * emu_e4m3(emu_ld8(emu_kmaj(db, n, k)));
    d[i] = sd ? d[i] + acc : acc;
  }
#endif
}
__device__ __forceinline__ void wg_n16_pv(float* d, uint64_t da, uint64_t db, int sd) {  // A MN-major (trans), B K-major
#if HAS_WG
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %10, 0;\nwgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {%0,%1,%2,%3,%4,%5,%6,%7}, %8, %9, p, 1, 1, 1, 0;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]) : "l"(da), "l"(db), "r"(sd));
#elif WGEMU
  const int wt = threadIdx.x & 127, w = wt >> 5, g = (wt & 31) >> 2, t = wt & 3;
  for (int i = 0; i < 8; i++) {
    const int hh = (i >> 1) & 1, e = i & 1, j = i >> 2, m = 16 * w + g + 8 * hh, n = 8 * j + 2 * t + e;
    float acc = 0.f;
    for (int k = 0; k < 16; k++) acc += emu_ldh(emu_mnmaj(da, k, 2 * m)) * emu_ldh(emu_kmaj(db, n, 2 * k));
    d[i] = sd ? d[i] + acc : acc;
  }
#endif
}
#ifndef DRS
#define DRS RSW
#endif
template <int NL, int X>
__global__ void __launch_bounds__(128 * (NL + 1), 1) attnd(A a, WF f) {
  constexpr int RS = DRS;
  constexpr bool L2H = X & 64;
  constexpr uint32_t LS = RS * 16384 + 32768 + 4096 + 2048 + 2048;  // ring, V f16 (2 bufs x 2 heads), P^T (2 bufs), Q', page table
  PS(0);
  const int wg = threadIdx.x >> 7;
  const bool prod = wg == NL;
  const int LN = prod ? ((threadIdx.x & 127) >> 6) : wg;
  const int nsc = a.nsplit / NL, b = blockIdx.x / nsc, sp = (blockIdx.x % nsc) * NL + LN, nsp = a.nsplit;
  extern __shared__ __align__(1024) uint8_t smem[];
  const uint32_t sb = ((uint32_t)__cvta_generic_to_shared(smem) + 1023) & ~1023u;
  const uint32_t ring = sb + LN * LS, vf0 = ring + RS * 16384, pt0 = vf0 + 32768, qs = pt0 + 4096, ptr = qs + 2048;
  const uint32_t mbar = sb + NL * LS, red = mbar + 256;  // full[NL][RS], empty[NL][RS]; red: [NL][4 warps][16] floats + flags
  auto fullb = [&](int s) { return mbar + (LN * RS + s) * 8; };
  auto emptyb = [&](int s) { return mbar + (NL * RS + LN * RS + s) * 8; };
  if (threadIdx.x == 0) {
    for (int i = 0; i < NL * RS; i++) {
      asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(mbar + i * 8), "r"(64));
      asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(mbar + (NL * RS + i) * 8), "r"(4));
    }
  }
  const int seq = a.seq[b], nk = seq + 1;
  const int k0 = a.window ? max(0, seq - a.window + 1) : 0, span = nk - k0;
  const int ks = k0 + (int)((long long)span * sp / nsp), ke = k0 + (int)((long long)span * (sp + 1) / nsp);
  const int nb = (ke - ks + 63) >> 6;
  __syncthreads();
  if (prod) {
    const int pt = threadIdx.x & 63;
    if (NL == 1 && (threadIdx.x & 127) >= 64) return;
    const int* ptb = a.pt + (size_t)b * a.maxp;
    auto ptload = [&](int x) {
      const int idx = ks + x * 64 + pt;
      asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" :: "r"(ptr + ((x & 7) * 64 + pt) * 4), "l"(ptb + (idx < ke ? idx : 0)), "r"(idx < ke ? 4 : 0));
    };
#pragma unroll
    for (int x = 0; x < 7; x++) ptload(x);
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    asm volatile("bar.sync %0, 64;\n" :: "r"(1 + LN) : "memory");
    const int c8 = pt & 7, kr = pt >> 3;
    for (int x = 0; x < nb; x++) {
      const int slot = x % RS;
      if (x >= RS) mb_wait(emptyb(slot), ((x / RS) - 1) & 1);
      const int kb = ks + x * 64;
      uint32_t rw[8];
#pragma unroll
      for (int i = 0; i < 8; i++) rw[i] = lds32a(ptr + ((x & 7) * 64 + i * 8 + kr) * 4);
      asm volatile("bar.sync %0, 64;\n" :: "r"(1 + LN) : "memory");
      ptload(x + 7);
#pragma unroll
      for (int i = 0; i < 8; i++) {
        const int key = i * 8 + kr, kp = kb + key;
        if (kp >= seq && kp < ke) continue;  // the new token: written by the consumers
        const uint32_t n = kp < ke ? 16 : 0;
        const uint32_t dd = ring + slot * 16384 + key * 128 + ((c8 ^ (key & 7)) << 4);
        const size_t off = (size_t)(kp < ke ? rw[i] : 0) * 128 + c8 * 16;
        if (L2H) {
          asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;\n" :: "r"(dd), "l"(a.kc + off), "r"(n));
          asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;\n" :: "r"(dd + 8192), "l"(a.vc + off), "r"(n));
        } else {
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dd), "l"(a.kc + off), "r"(n));
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dd + 8192), "l"(a.vc + off), "r"(n));
        }
      }
      asm volatile("cp.async.commit_group;\n");
      asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" :: "r"(fullb(slot)) : "memory");
      asm volatile("cp.async.wait_group 6;\n" ::: "memory");
      asm volatile("bar.sync %0, 64;\n" :: "r"(1 + LN) : "memory");
    }
    asm volatile("cp.async.wait_all;\n" ::: "memory");
    return;
  }
  const int tid = threadIdx.x - 128 * LN, w = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int wbar = 9 + LN;
  const __nv_bfloat16* row0 = f.qkv + (size_t)b * (2 * 16 + 4) * D;
  {  // Q': row r = q head (kv head r >> 3), 8 threads x 8 dims; E4M3 of the normalized (and rotated) query
    const int r = tid >> 3, d0 = (tid & 7) * 8, hr = r >> 3;
    const uint4 u = *(const uint4*)(row0 + r * D + d0);
    const __nv_bfloat162* bb = (const __nv_bfloat162*)&u;
    float x[8], ss = 0.f;
#pragma unroll
    for (int j = 0; j < 4; j++) { const float2 v = __bfloat1622float2(bb[j]); x[2 * j] = v.x; x[2 * j + 1] = v.y; }
#pragma unroll
    for (int i = 0; i < 8; i++) ss += x[i] * x[i];
#pragma unroll
    for (int o2 = 1; o2 < 8; o2 <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o2);
    const float rn = rsqrtf(ss / D + 1e-5f);
#pragma unroll
    for (int i = 0; i < 8; i++) x[i] *= rn;
    if (f.theta > 0.f) {
#pragma unroll
      for (int i = 0; i < 8; i += 2) {
        float sn, cs2; rope_cs(f.invf, seq, (d0 + i) >> 1, sn, cs2);
        const float e = x[i], o = x[i + 1];
        x[i] = e * cs2 - o * sn; x[i + 1] = e * sn + o * cs2;
      }
    }
    uint32_t q2[2] = {0, 0};
#pragma unroll
    for (int i = 0; i < 8; i++) { __nv_fp8_e4m3 fq(x[i]); q2[i >> 2] |= (uint32_t)fq.__x << (8 * (i & 3)); }
    const int bo = hr * 64 + d0, bz = (1 - hr) * 64 + d0;
    asm volatile("st.shared.v2.u32 [%0], {%1,%2};\n" :: "r"(qs + r * 128 + (((bo >> 4) ^ (r & 7)) << 4) + (bo & 15)), "r"(q2[0]), "r"(q2[1]));
    asm volatile("st.shared.v2.u32 [%0], {%1,%2};\n" :: "r"(qs + r * 128 + (((bz >> 4) ^ (r & 7)) << 4) + (bz & 15)), "r"(0u), "r"(0u));
  }
  // the new token's key (RMSNorm, RoPE) and value, E4M3: warp 0, lane = (head lane >> 4, dims 4 (lane & 15) ..)
  uint32_t kb4 = 0, vb4 = 0;
  const int nh = lane >> 4, nd0 = (lane & 15) * 4;
  if (w == 0 && ke > seq) {
    const uint2 kraw = *(const uint2*)(row0 + (16 + nh) * D + nd0), vraw = *(const uint2*)(row0 + (18 + nh) * D + nd0);
    const __nv_bfloat16* kr = (const __nv_bfloat16*)&kraw; const __nv_bfloat16* vr = (const __nv_bfloat16*)&vraw;
    float k[4], v[4], s2 = 0.f;
#pragma unroll
    for (int i = 0; i < 4; i++) { k[i] = __bfloat162float(kr[i]); v[i] = __bfloat162float(vr[i]); s2 += k[i] * k[i]; }
#pragma unroll
    for (int o2 = 1; o2 < 16; o2 <<= 1) s2 += __shfl_xor_sync(0xffffffffu, s2, o2);
    const float rk = rsqrtf(s2 / D + 1e-5f);
#pragma unroll
    for (int i = 0; i < 4; i++) k[i] *= rk;
    if (f.theta > 0.f) {
#pragma unroll
      for (int i = 0; i < 4; i += 2) {
        float sn, cs2; rope_cs(f.invf, seq, (nd0 + i) >> 1, sn, cs2);
        const float e = k[i], o = k[i + 1];
        k[i] = e * cs2 - o * sn; k[i + 1] = e * sn + o * cs2;
      }
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
      __nv_fp8_e4m3 fk(k[i] * f.kinv), fv(v[i] * f.vinv);
      kb4 |= (uint32_t)fk.__x << (8 * i); vb4 |= (uint32_t)fv.__x << (8 * i);
    }
    const size_t lc = (size_t)f.loc[b] * 128 + nh * 64 + nd0;
    *(uint32_t*)((uint8_t*)a.kc + lc) = kb4;
    *(uint32_t*)((uint8_t*)a.vc + lc) = vb4;
  }
  fence_async();
  asm volatile("bar.sync %0, 128;\n" :: "r"(wbar) : "memory");
  PS(2);
  const uint32_t rl = red + LN * 1024;  // [4 warps][16] maxima, then 4 overflow flags at +256
  float oT[2][8], mrun[4], lsum[4];  // rows n(j, e) = 8 j + 2 t + e
#pragma unroll
  for (int i = 0; i < 4; i++) { mrun[i] = -INFINITY; lsum[i] = 0.f; }
  // exact running max of this step's scores: thread max over its 2 keys, across g (lanes ^ 4, 8, 16), across the 4 warps
  auto stepmax = [&](const float* v, float* nm) {
#pragma unroll
    for (int i = 0; i < 4; i++) {
      float m = fmaxf(v[(i >> 1) * 4 + (i & 1)], v[(i >> 1) * 4 + 2 + (i & 1)]);
      m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 4));
      m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 8));
      m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 16));
      if (g == 0) asm volatile("st.shared.f32 [%0], %1;\n" :: "r"(rl + (w * 16 + 8 * (i >> 1) + 2 * t + (i & 1)) * 4), "f"(m));
    }
    asm volatile("bar.sync %0, 128;\n" :: "r"(wbar) : "memory");
#pragma unroll
    for (int i = 0; i < 4; i++) {
      float m = -INFINITY;
#pragma unroll
      for (int w2 = 0; w2 < 4; w2++) { float x; asm volatile("ld.shared.f32 %0, [%1];\n" : "=f"(x) : "r"(rl + (w2 * 16 + 8 * (i >> 1) + 2 * t + (i & 1)) * 4)); m = fmaxf(m, x); }
      nm[i] = fmaxf(mrun[i], m);
    }
    asm volatile("bar.sync %0, 128;\n" :: "r"(wbar) : "memory");  // the max slots are reused next time
  };
  int bi = 0;
  for (int kb = ks; bi < nb; bi++, kb += 64) {
    const int slot = bi % RS;
    const uint32_t raw = ring + slot * 16384, vf = vf0 + (bi & 1) * 16384, pT = pt0 + (bi & 1) * 2048;
    mb_wait(fullb(slot), (bi / RS) & 1);
    if (kb <= seq && kb + 64 > seq && seq < ke) {  // the new token's row into the slot (both heads, K and V)
      if (w == 0) {
        const int kq = seq - kb;
        const uint32_t dd = raw + kq * 128 + ((((nh * 64 + nd0) >> 4) ^ (kq & 7)) << 4) + (nd0 & 15);
        asm volatile("st.shared.u32 [%0], %1;\n" :: "r"(dd), "r"(kb4));
        asm volatile("st.shared.u32 [%0], %1;\n" :: "r"(dd + 8192), "r"(vb4));
      }
      asm volatile("bar.sync %0, 128;\n" :: "r"(wbar) : "memory");
    }
    {  // V (both heads) -> f16: key tid >> 1, head tid & 1, its 4 chunks
      const int key = tid >> 1, hv = tid & 1;
#pragma unroll
      for (int u = 0; u < 4; u++) {
        const int c = 4 * hv + u;
        uint4 x;
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(x.x), "=r"(x.y), "=r"(x.z), "=r"(x.w) : "r"(raw + 8192 + key * 128 + ((c ^ (key & 7)) << 4)));
        const int j0 = 2 * u;
        const uint32_t dst = vf + hv * 8192 + key * 128;
        asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" :: "r"(dst + ((j0 ^ (key & 7)) << 4)), "r"(e2h(x.x)), "r"(e2h(x.x >> 16)), "r"(e2h(x.y)), "r"(e2h(x.y >> 16)));
        asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" :: "r"(dst + (((j0 + 1) ^ (key & 7)) << 4)), "r"(e2h(x.z)), "r"(e2h(x.z >> 16)), "r"(e2h(x.w)), "r"(e2h(x.w >> 16)));
      }
    }
    fence_async();
    asm volatile("bar.sync %0, 128;\n" :: "r"(wbar) : "memory");
    float sT[8];
    wg_fence();
#pragma unroll
    for (int k = 0; k < 4; k++) wg_n16_qk(sT, gdesc(raw + 32 * k, 16, 1024), gdesc(qs + 32 * k, 16, 1024), k);
    wg_commit(); wg_wait0();  // also retires the previous step's PV
    __syncwarp();
    if (lane == 0) { uint64_t st_; asm volatile("mbarrier.arrive.shared.b64 %0, [%1];\n" : "=l"(st_) : "r"(emptyb(slot)) : "memory"); }
    // scores (log2 units); keys past the split -> -inf
#pragma unroll
    for (int i = 0; i < 8; i++) { const int kp = kb + 16 * w + g + 8 * ((i >> 1) & 1); sT[i] = kp < ke ? sT[i] * a.qk : -INFINITY; }
    // sT[i]: key 16 w + g + 8 hh, row n = 8 j + 2 t + e, i = 4 j + 2 hh + e; row slot q = 2 j + e
    auto pslot = [&](int i) { return 2 * (i >> 2) + (i & 1); };
    float pv[8];
    bool slow = bi == 0;
    if (!slow) {
#pragma unroll
      for (int i = 0; i < 8; i++) pv[i] = ex2(sT[i] - mrun[pslot(i)]);
      float pm = 0.f;
#pragma unroll
      for (int i = 0; i < 8; i++) pm = fmaxf(pm, pv[i]);
      const unsigned ov = __any_sync(0xffffffffu, pm > LZP);
      if (lane == 0) asm volatile("st.shared.u32 [%0], %1;\n" :: "r"(rl + 256 + w * 4), "r"(ov));
      asm volatile("bar.sync %0, 128;\n" :: "r"(wbar) : "memory");
      uint32_t any = 0;
#pragma unroll
      for (int w2 = 0; w2 < 4; w2++) any |= lds32a(rl + 256 + w2 * 4);
      slow = any != 0;
      asm volatile("bar.sync %0, 128;\n" :: "r"(wbar) : "memory");  // flags reused next step
    }
    if (slow) {  // uniform over the warpgroup: new running max, rescale O and the sums
      float v4[8], nm[4];
#pragma unroll
      for (int i = 0; i < 8; i++) v4[i] = sT[i];
      // stepmax wants v indexed as [ (i >> 1) * 4 + (i & 1) ] and [ + 2 ] for slot i: slot q = 2 j + e -> sT[4 j + e], sT[4 j + 2 + e]
      stepmax(v4, nm);
#pragma unroll
      for (int q = 0; q < 4; q++) {
        const float nmf = nm[q] == -INFINITY ? 0.f : nm[q];
        const float c = mrun[q] == -INFINITY ? 0.f : ex2(mrun[q] - nmf);
        lsum[q] *= c;
        if (bi > 0) {
#pragma unroll
          for (int h = 0; h < 2; h++) { oT[h][4 * (q >> 1) + (q & 1)] *= c; oT[h][4 * (q >> 1) + 2 + (q & 1)] *= c; }
        }
        mrun[q] = nm[q];
      }
#pragma unroll
      for (int i = 0; i < 8; i++) { const float m = mrun[pslot(i)]; pv[i] = m == -INFINITY ? 0.f : ex2(sT[i] - m); }
    }
#pragma unroll
    for (int i = 0; i < 8; i++) lsum[pslot(i)] += pv[i];
    // P^T (16 rows n x 64 keys, f16, K-major SW128) for the PV B operand
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const int key = 16 * w + g + 8 * ((i >> 1) & 1), n = 8 * (i >> 2) + 2 * t + (i & 1), kb2 = key * 2;
      const __half hv = __float2half_rn(pv[i]);
      asm volatile("st.shared.u16 [%0], %1;\n" :: "r"(pT + n * 128 + (((kb2 >> 4) ^ (n & 7)) << 4) + (kb2 & 15)), "h"(*(const unsigned short*)&hv));
    }
    fence_async();
    asm volatile("bar.sync %0, 128;\n" :: "r"(wbar) : "memory");
    wg_fence();
#pragma unroll
    for (int h = 0; h < 2; h++)
#pragma unroll
      for (int kc = 0; kc < 4; kc++) wg_n16_pv(oT[h], gdesc(vf + h * 8192 + kc * 2048, WVL, WVS), gdesc(pT + kc * 32, 16, 1024), bi > 0 || kc > 0);
    wg_commit();
  }
  wg_wait0();
  PS(3);
  if (bi == 0) {
#pragma unroll
    for (int h = 0; h < 2; h++)
#pragma unroll
      for (int i = 0; i < 8; i++) oT[h][i] = 0.f;
  }
  // row sums: across g, then across the 4 warps (smem)
#pragma unroll
  for (int q = 0; q < 4; q++) {
    float l = lsum[q];
    l += __shfl_xor_sync(0xffffffffu, l, 4);
    l += __shfl_xor_sync(0xffffffffu, l, 8);
    l += __shfl_xor_sync(0xffffffffu, l, 16);
    if (g == 0) asm volatile("st.shared.f32 [%0], %1;\n" :: "r"(rl + (w * 16 + 8 * (q >> 1) + 2 * t + (q & 1)) * 4), "f"(l));
  }
  asm volatile("bar.sync %0, 128;\n" :: "r"(wbar) : "memory");
  // partials: O[row n][dim] (fp32), only head h's rows (n >> 3 == h, i.e. j == h) of oT[h]; (m, l) per row
  const size_t it = (size_t)b * a.nsplit + sp;
#pragma unroll
  for (int h = 0; h < 2; h++)
#pragma unroll
    for (int hh = 0; hh < 2; hh++)
#pragma unroll
      for (int e = 0; e < 2; e++) {
        const int n = 8 * h + 2 * t + e, dm = 16 * w + g + 8 * hh;
        a.wo[(it * 16 + n) * D + dm] = oT[h][4 * h + 2 * hh + e];
      }
  if (tid < 16) {
    float l = 0.f;
#pragma unroll
    for (int w2 = 0; w2 < 4; w2++) { float x; asm volatile("ld.shared.f32 %0, [%1];\n" : "=f"(x) : "r"(rl + (w2 * 16 + tid) * 4)); l += x; }
    // running max of row tid: held by threads with 2 t + e == tid & 7, j == tid >> 3 (uniform): recompute from smem-free path
    a.wml[(it * 16 + tid) * 2 + 1] = l;
  }
  if (g == 0) {  // lanes t: rows 8 j + 2 t + e (slot q = 2 j + e)
#pragma unroll
    for (int q = 0; q < 4; q++) if (w == 0) a.wml[(it * 16 + 8 * (q >> 1) + 2 * t + (q & 1)) * 2] = (bi == 0) ? -INFINITY : mrun[q];
  }
  // merge: the last nmg arrivers of the request combine its 16 rows (16 threads x 4 dims per row)
  asm volatile("bar.sync %0, 128;\n" :: "r"(wbar) : "memory");
  __shared__ int rank_d[2];
  if (tid == 0) { int v; asm volatile("atom.add.acq_rel.gpu.global.s32 %0, [%1], 1;\n" : "=r"(v) : "l"(f.cnt + b) : "memory"); rank_d[LN] = v; }
  asm volatile("bar.sync %0, 128;\n" :: "r"(wbar) : "memory");
  const int nmg = min(2, nsp), rank = rank_d[LN] - (nsp - nmg);
  if (rank < 0) return;
  const int rr0 = rank + nmg * (tid >> 4);
  uint2 gpre = make_uint2(0, 0);
  if (rr0 < 16) gpre = __ldg((const uint2*)(row0 + 20 * D + rr0 * D + (tid & 15) * 4));
  if (tid == 0) {
    int v;
    do { asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(f.cnt + b) : "memory"); if (v < nsp) __nanosleep(32); } while (v < nsp);
  }
  asm volatile("bar.sync %0, 128;\n" :: "r"(wbar) : "memory");
  const int d4 = tid & 15;
  for (int rr = rr0; rr < 16; rr += nmg * 8) {
    const size_t base = (size_t)b * a.nsplit;
    float Mr = -INFINITY, Lr = 0.f;
    float4 a4 = make_float4(0.f, 0.f, 0.f, 0.f);
    mrg8<8>(0, 1, nsp, [&](int k2) { return __ldcg((const float2*)(a.wml + ((base + k2) * 16 + rr) * 2)); },
            [&](int k2) { return __ldcg((const float4*)(a.wo + ((base + k2) * 16 + rr) * D) + d4); }, Mr, Lr, a4);
    const float inv = f.vscale / Lr;
    a4.x *= inv; a4.y *= inv; a4.z *= inv; a4.w *= inv;
    float ss = a4.x * a4.x + a4.y * a4.y + a4.z * a4.z + a4.w * a4.w;
#pragma unroll
    for (int x = 1; x < 16; x <<= 1) ss += __shfl_xor_sync(0xffffu << (lane & 16), ss, x);
    const float rn = rsqrtf(ss / D + 1e-5f);
    const uint2 gv = rr == rr0 ? gpre : __ldg((const uint2*)(row0 + 20 * D + rr * D + d4 * 4));
    const __nv_bfloat16* gt = (const __nv_bfloat16*)&gv;
    const float vals[4] = {a4.x, a4.y, a4.z, a4.w};
    __nv_bfloat16 ob[4];
#pragma unroll
    for (int i2 = 0; i2 < 4; i2++) ob[i2] = __float2bfloat16(vals[i2] * rn / (1.f + expf(-__bfloat162float(gt[i2]))));
    *(uint2*)(f.out + (size_t)b * 16 * D + rr * D + d4 * 4) = *(uint2*)ob;
  }
  PS(7);
  if (tid == 0 && atomicAdd(f.cnt + a.B + b, 1) == nmg - 1) { f.cnt[b] = 0; f.cnt[a.B + b] = 0; }
}
template <int NL, int X>
static void adf(int g, int smw, cudaStream_t st, const A& a, const WF& f) {
  static bool o = false;
  if (!o) { cudaFuncSetAttribute(attnd<NL, X>, cudaFuncAttributeMaxDynamicSharedMemorySize, smw); o = true; }
  attnd<NL, X><<<g, 128 * (NL + 1), smw, st>>>(a, f);
}

// V bits: 1 copy = 2 keys x 32 B sector per thread (else 4 keys x 16 B), 2 QK issued first with loads / V convert
// overlapping it (else loads, convert, QK), 4 softmax fast path against the running max (else exact lazy), 8 prologue:
// page table + KV issued before the query prologue, new keys written into the ring (else query prologue first).
// Timing ablations only (wrong results): 16 no in-loop loads, 32 no wgmma, 64 no V convert, 128 no exp, 256 no in-loop barriers.
// 512 (full layers): splits over the page table's capacity, independent of seq (page-table loads do not wait for it)
template <bool PS1, bool FUSE, bool MRG, int V>
__global__ void __launch_bounds__(256, 2) attnw(A a, WF f) {
  constexpr bool CP = V & 1, OV = V & 2, FS = V & 4, PO = (V & 8) && FUSE;
  constexpr int NR = CP ? 2 : 4;
  constexpr bool SS = V & 512;
  constexpr bool XL = V & 16, XM = V & 32, XV = V & 64, XE = V & 128, XB = V & 256;
  const int tid = threadIdx.x, wt = tid & 127, h = tid >> 7, w = wt >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int nqb = (a.L + 7) >> 3, sp = blockIdx.x % a.nsplit, bq = blockIdx.x / a.nsplit, qb = bq % nqb, b = bq / nqb;
  const int t0 = qb * 8, lt = min(8, a.L - t0);
  const int qr_ = wt >> 1, qhf = wt & 1, ql = qr_ >> 3;
  const int nrr = tid >> 4, nkh = nrr & 1, nlt = nrr >> 1, nd0 = (tid & 15) * 4;
  uint4 qraw[4];
  uint2 kraw = make_uint2(0, 0), vraw = make_uint2(0, 0);
  auto ldq = [&]() {
    const __nv_bfloat16* src = f.qkv + ((size_t)b * a.L + ql) * (2 * 16 + 4) * D + (h * 8 + (qr_ & 7)) * D + qhf * 32;
#pragma unroll
    for (int i = 0; i < 4; i++) qraw[i] = *(const uint4*)(src + 8 * i);
  };
  auto ldkv = [&]() {
    const __nv_bfloat16* krow = f.qkv + ((size_t)b * a.L + nlt) * (2 * 16 + 4) * D;
    kraw = *(const uint2*)(krow + (16 + nkh) * D + nd0); vraw = *(const uint2*)(krow + (18 + nkh) * D + nd0);
  };
  PS(0);
  if constexpr (PO) { ldq(); ldkv(); }
  const int seq = a.seq[b], nk = seq + t0 + lt;
  const int maxk = a.maxp * a.ps, stt = SS && !a.window;
  const int k0 = a.window ? max(0, seq + t0 - a.window + 1) : 0, span = stt ? maxk : nk - k0;
  const int ks = k0 + (int)((long long)span * sp / a.nsplit), ke0 = k0 + (int)((long long)span * (sp + 1) / a.nsplit);
  const int ke = stt ? min(ke0, nk) : ke0;
  extern __shared__ __align__(1024) uint8_t smem[];
  const uint32_t sb = ((uint32_t)__cvta_generic_to_shared(smem) + 1023) & ~1023u;
  const uint32_t ring = sb, qs = sb + RSW * 16384, vf0 = qs + 8192 + h * 16384;
  const uint32_t ptr = qs + 8192 + 2 * 16384;  // page-table rings: 8 steps x 64 entries per warpgroup
  const uint32_t ones = ptr + 4096;  // 1 KB of f16 1.0: B of the row-sum wgmma (any layout: every element is 1)
  for (int x = wt; x < 256; x += 128) asm volatile("st.shared.u32 [%0], %1;\n" :: "r"(ones + x * 4), "r"(0x3C003C00u));
  const uint64_t dOnes = (uint64_t)(ones >> 4) | ((uint64_t)(128 >> 4) << 16) | ((uint64_t)(256 >> 4) << 32);
  float lacc[4];
  const int* ptb = a.pt + (size_t)b * a.maxp;
  // copy geometry: row slot i of a step is key koff(i); CP: keys ck, ck + 32, 32 B sectors; else keys 16 i + lr, 16 B chunk c8
  const int ck = wt >> 2, cpart = (wt >> 1) & 1, csub = wt & 1;
  const int lr = wt >> 3, part = (wt >> 2) & 1, c8 = 4 * h + (wt & 3);
  const uint32_t cdst = CP ? ring + cpart * 8192 + ck * 128 + (((4 * h + csub) ^ (ck & 7)) << 4) : 0u;
  const uint8_t* cbase = CP ? (cpart ? a.vc : a.kc) + (4 * h + csub) * 16 : (part ? a.vc : a.kc) + c8 * 16;
  auto koff = [&](int i) -> int { return CP ? ck + 32 * i : 16 * i + lr; };
  auto rowg = [&](int kp) -> int {
    const int kc = min(kp, stt ? maxk - 1 : ke - 1);
    const int rr = PS1 ? __ldg(ptb + kc) : __ldg(ptb + kc / a.ps) * a.ps + kc % a.ps;
    return kp < ke ? rr : -1;
  };
  auto issue = [&](int slot, int kbi, const int* rw) {  // rw: cache rows; -1 zero-fill; -2 nothing (written directly)
    if (XL && kbi >= ks + (RSW - 1) * 64) { asm volatile("cp.async.commit_group;\n"); return; }
    if (kbi < ke) {
#pragma unroll
      for (int i = 0; i < NR; i++) {
        if (rw[i] == -2) continue;
        const uint8_t* s0 = cbase + (size_t)max(rw[i], 0) * 128;
        const int n = rw[i] >= 0 ? 16 : 0;
        if constexpr (CP) {
          const uint32_t d = cdst + slot * 16384 + i * 4096;
          asm volatile("cp.async.cg.shared.global" AWH " [%0], [%1], 16, %2;\n" :: "r"(d), "l"(s0), "r"(n));
          asm volatile("cp.async.cg.shared.global" AWH " [%0], [%1], 16, %2;\n" :: "r"(d ^ 32), "l"(s0 + 32), "r"(n));
        } else {
          const int key = 16 * i + lr;
          const uint32_t d = ring + slot * 16384 + part * 8192 + key * 128 + ((c8 ^ (key & 7)) << 4);
          asm volatile("cp.async.cg.shared.global" AWH " [%0], [%1], 16, %2;\n" :: "r"(d), "l"(s0), "r"(n));
        }
      }
    }
    asm volatile("cp.async.commit_group;\n");
  };
  const uint32_t ptrh = ptr + h * 2048;
  auto ptseg = [&](int x) {
    const int idx = ks + x * 64 + wt;
    if (wt < 64) asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" :: "r"(ptrh + (x & 7) * 256 + wt * 4), "l"(ptb + (idx < ke ? idx : 0)), "r"(idx < ke ? 4 : 0));
  };
  auto rowr = [&](int kp) -> int {
    if (!PS1) return rowg(kp);
    if (kp >= ke) return -1;
    int v; asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(v) : "r"(ptrh + ((kp - ks) & 511) * 4));
    return v;
  };
  int rq[NR];
  auto qprolog = [&]() {  // row qr_ (token ql, q head 8 h + (qr_ & 7)), half qhf: RMSNorm, RoPE at seq + token, E4M3
    float x[32], ss = 0.f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const __nv_bfloat162* bb = (const __nv_bfloat162*)&qraw[i];
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
        float sn, cs2; rope_cs(f.invf, seq + ql, (qhf * 32 + i) >> 1, sn, cs2);
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
      const int c = 4 * h + 2 * qhf + u;
      asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" :: "r"(qs + qr_ * 128 + ((c ^ (qr_ & 7)) << 4)), "r"(q4[4 * u]), "r"(q4[4 * u + 1]), "r"(q4[4 * u + 2]), "r"(q4[4 * u + 3]));
    }
  };
  auto kvprolog = [&](bool ring_too) {  // the block's own keys and values (RMSNorm/RoPE, E4M3) to the cache [and the ring]
    float k[4], v[4], s2 = 0.f;
    const __nv_bfloat16* kr = (const __nv_bfloat16*)&kraw; const __nv_bfloat16* vr = (const __nv_bfloat16*)&vraw;
#pragma unroll
    for (int i = 0; i < 4; i++) { k[i] = __bfloat162float(kr[i]); v[i] = __bfloat162float(vr[i]); s2 += k[i] * k[i]; }
#pragma unroll
    for (int o = 1; o < 16; o <<= 1) s2 += __shfl_xor_sync(0xffffffffu, s2, o);
    const float kr2 = rsqrtf(s2 / D + 1e-5f);
#pragma unroll
    for (int i = 0; i < 4; i++) k[i] *= kr2;
    if (f.theta > 0.f) {
#pragma unroll
      for (int i = 0; i < 4; i += 2) {
        float sn, cs2; rope_cs(f.invf, seq + nlt, (nd0 + i) >> 1, sn, cs2);
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
    if (nlt < lt) {
      const size_t lc = (size_t)f.loc[b * a.L + nlt] * 128 + nkh * 64 + nd0;
      *(uint32_t*)((uint8_t*)a.kc + lc) = kb4;
      *(uint32_t*)((uint8_t*)a.vc + lc) = vb4;
      const int kq = seq + nlt - ks, j = kq >> 6, kk = kq & 63;
      if (ring_too && kq >= 0 && j < RSW - 1 && seq + nlt < ke) {
        const uint32_t dd = ring + j * 16384 + kk * 128 + ((((nkh * 64 + nd0) >> 4) ^ (kk & 7)) << 4) + (nd0 & 15);
        asm volatile("st.shared.u32 [%0], %1;\n" :: "r"(dd), "r"(kb4));
        asm volatile("st.shared.u32 [%0], %1;\n" :: "r"(dd + 8192), "r"(vb4));
      }
    }
    __threadfence();
  };
  if constexpr (PO) {
    if (PS1) for (int x = RSW; x < RSW + 3; x++) ptseg(x);
    int rp[RSW - 1][NR];
#pragma unroll
    for (int j = 0; j < RSW - 1; j++)
#pragma unroll
      for (int i = 0; i < NR; i++) rp[j][i] = rowg(ks + j * 64 + koff(i));
#pragma unroll
    for (int i = 0; i < NR; i++) rq[i] = rowg(ks + (RSW - 1) * 64 + koff(i));
    qprolog();
#pragma unroll
    for (int j = 0; j < RSW - 1; j++) {  // the block's own keys are not in the cache yet: written below, not copied
#pragma unroll
      for (int i = 0; i < NR; i++) { const int kp = ks + j * 64 + koff(i); if (kp >= seq && kp < ke) rp[j][i] = -2; }
      issue(j, ks + j * 64, rp[j]);
    }
    PS(1);
    if (ke > seq) kvprolog(true);
    __syncthreads();
    PS(2);
  } else {
    if constexpr (!FUSE) {
#pragma unroll
      for (int i = 0; i < 2; i++) {
        const int idx = i * 128 + wt, r = idx >> 2, c = 4 * h + (idx & 3);
        const uint4 v = (r >> 3) < lt ? *(const uint4*)(a.q8 + (((size_t)b * a.L + t0 + (r >> 3)) * a.hq + h * 8 + (r & 7)) * D + (idx & 3) * 16) : make_uint4(0, 0, 0, 0);
        asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" :: "r"(qs + r * 128 + ((c ^ (r & 7)) << 4)), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
      }
    } else {
      ldq(); qprolog();
      if (ke == nk) { ldkv(); kvprolog(false); }
      __syncthreads();
      PS(1);
    }
    if (PS1) for (int x = RSW; x < RSW + 3; x++) ptseg(x);
#pragma unroll
    for (int j = 0; j < RSW - 1; j++) {
      int rp[NR];
#pragma unroll
      for (int i = 0; i < NR; i++) rp[i] = rowg(ks + j * 64 + koff(i));
      issue(j, ks + j * 64, rp);
    }
#pragma unroll
    for (int i = 0; i < NR; i++) rq[i] = rowg(ks + (RSW - 1) * 64 + koff(i));
    PS(2);
  }
  const int nb = max(0, (ke - ks + 63) >> 6);
  float o[32], mx[2] = {-INFINITY, -INFINITY};
  const int lw0 = 2 * w;  // tokens of this warp's rows: g -> lw0, g + 8 -> lw0 + 1
  const int lomax = a.window ? max(0, seq + t0 + lt - 1 - a.window + 1) : 0;
  const uint64_t dQ = gdesc(qs + 64 * h, 16, 1024), dK = gdesc(ring + 64 * h, 16, 1024), dV = gdesc(vf0, WVL, WVS);
  const int vkey = wt >> 1;
  auto loads = [&](int bi, int kb) {
    if (PS1) ptseg(bi + RSW + 3);
    issue((bi + RSW - 1) % RSW, kb + (RSW - 1) * 64, rq);
#pragma unroll
    for (int i = 0; i < NR; i++) rq[i] = rowr(kb + RSW * 64 + koff(i));
  };
  auto vconv = [&](int bi) {  // this head's V half of step bi to f16 (double-buffered: the previous PV may still read the other)
    const uint32_t raw = ring + (bi % RSW) * 16384 + 8192 + vkey * 128, vf = vf0 + (bi & 1) * 8192 + vkey * 128;
#pragma unroll
    for (int u = 0; u < 2; u++) {
      const int c = 4 * h + 2 * (wt & 1) + u, j0 = 2 * (c - 4 * h);
      uint4 x;
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(x.x), "=r"(x.y), "=r"(x.z), "=r"(x.w) : "r"(raw + ((c ^ (vkey & 7)) << 4)));
      asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" :: "r"(vf + ((j0 ^ (vkey & 7)) << 4)), "r"(e2h(x.x)), "r"(e2h(x.x >> 16)), "r"(e2h(x.y)), "r"(e2h(x.y >> 16)));
      asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" :: "r"(vf + (((j0 + 1) ^ (vkey & 7)) << 4)), "r"(e2h(x.z)), "r"(e2h(x.z >> 16)), "r"(e2h(x.w)), "r"(e2h(x.w >> 16)));
    }
  };
  for (int bi = 0, kb = ks; bi < nb; bi++, kb += 64) {
    const int slot = bi % RSW;
    float s[32];
    asm volatile("cp.async.wait_group %0;\n" :: "n"(RSW - 2));
    if constexpr (OV) {
      fence_async();
      if (!XB) asm volatile("bar.sync %0, 128;\n" :: "r"(1 + h) : "memory");  // step bi landed (all threads), slot of bi - 1 free
      wg_fence();
      if (XM) { for (int i = 0; i < 32; i++) s[i] = __int_as_float(lds32a(ring + slot * 16384 + 4 * (i + wt)) & 0x3f7fffff); }
      else { wg_ss8<0>(s, dQ, dK + slot * 1024); wg_ss8<1>(s, dQ + 2, dK + slot * 1024 + 2); }
      wg_commit();
      loads(bi, kb);
      if (!XV) vconv(bi);
      fence_async();
      wg_wait0();  // S ready; the previous step's PV retired
    } else {
      if (!XB) asm volatile("bar.sync %0, 128;\n" :: "r"(1 + h) : "memory");
      loads(bi, kb);
      if (!XV) vconv(bi);
      fence_async();
      if (!XB) asm volatile("bar.sync %0, 128;\n" :: "r"(1 + h) : "memory");
      wg_fence();
      if (XM) { for (int i = 0; i < 32; i++) s[i] = __int_as_float(lds32a(ring + slot * 16384 + 4 * (i + wt)) & 0x3f7fffff); }
      else { wg_ss8<0>(s, dQ, dK + slot * 1024); wg_ss8<1>(s, dQ + 2, dK + slot * 1024 + 2); }
      wg_commit(); wg_wait0();
    }
    const bool full = kb + 64 <= ke && kb + 64 <= seq + t0 + 1 && kb >= lomax && lt == 8;
    if (!full) {
#pragma unroll
      for (int hh = 0; hh < 2; hh++) {
        const int p = lw0 + hh < lt ? seq + t0 + lw0 + hh : -1, lo = a.window ? max(0, p - a.window + 1) : 0;
#pragma unroll
        for (int j = 0; j < 8; j++)
#pragma unroll
          for (int e = 0; e < 2; e++) { const int kp = kb + 8 * j + 2 * t + e; if (!(kp < ke && kp <= p && kp >= lo)) s[4 * j + 2 * hh + e] = -INFINITY; }
      }
    }
    uint32_t pa[16];
    float scl[2] = {1.f, 1.f};
    bool resc = false, exact = true;
    if constexpr (FS) exact = !__all_sync(0xffffffffu, mx[0] != -INFINITY && mx[1] != -INFINITY);
    if (!exact) {
      // fast path: P against the running max, no row max; P <= 2^LZT is checked on the packed f16 values
#pragma unroll
      for (int j = 0; j < 8; j++)
#pragma unroll
        for (int hh = 0; hh < 2; hh++)
#pragma unroll
          for (int e = 0; e < 2; e++) s[4 * j + 2 * hh + e] = XE ? __saturatef(fmaf(s[4 * j + 2 * hh + e], a.qk, -mx[hh])) : ex2(fmaf(s[4 * j + 2 * hh + e], a.qk, -mx[hh]));
#pragma unroll
      for (int i = 0; i < 16; i++) { pa[i] = pk(s[2 * i], s[2 * i + 1]); asm volatile("" : "+r"(pa[i])); }
      __half2 hm = *(__half2*)&pa[0];
#pragma unroll
      for (int i = 1; i < 16; i++) hm = __hmax2(hm, *(__half2*)&pa[i]);
      const uint32_t hb = *(uint32_t*)&hm;
      const bool over = (hb & 0xffffu) > LZB || (hb >> 16) > LZB;
      if (__any_sync(0xffffffffu, over)) {  // rare: a row's max moved up by more than 2^LZT: rescale that row by its P max
#pragma unroll
        for (int hh = 0; hh < 2; hh++) {
          float pm = 0.f;
#pragma unroll
          for (int j = 0; j < 8; j++) pm = fmaxf(pm, fmaxf(s[4 * j + 2 * hh], s[4 * j + 2 * hh + 1]));
          pm = fmaxf(pm, __shfl_xor_sync(0xffffffffu, pm, 1));
          pm = fmaxf(pm, __shfl_xor_sync(0xffffffffu, pm, 2));
          if (pm > LZP) {
            const float inv = 1.f / pm;
            mx[hh] += __log2f(pm);
            scl[hh] = inv;
#pragma unroll
            for (int j = 0; j < 8; j++) { s[4 * j + 2 * hh] *= inv; s[4 * j + 2 * hh + 1] *= inv; }
          }
        }
#pragma unroll
        for (int i = 0; i < 16; i++) pa[i] = pk(s[2 * i], s[2 * i + 1]);
        resc = true;
      }
    } else {
      // exact lazy rescale: keep the running max unless a score exceeds it by more than 2^LZT
      bool upd[2];
#pragma unroll
      for (int hh = 0; hh < 2; hh++) {
        float bm = -INFINITY;
#pragma unroll
        for (int j = 0; j < 8; j++) bm = fmaxf(bm, fmaxf(s[4 * j + 2 * hh], s[4 * j + 2 * hh + 1]));
        bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, 1));
        bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, 2));
        const bool up = bm * a.qk > mx[hh] + LZT;
        const float nm = up ? fmaxf(mx[hh], bm * a.qk) : mx[hh];
        const float nmf = nm == -INFINITY ? 0.f : nm;
        upd[hh] = up;
        scl[hh] = up ? ex2(mx[hh] - nmf) : 1.f;
        mx[hh] = nm;
#pragma unroll
        for (int j = 0; j < 8; j++)
#pragma unroll
          for (int e = 0; e < 2; e++) s[4 * j + 2 * hh + e] = XE ? __saturatef(fmaf(s[4 * j + 2 * hh + e], a.qk, -nmf)) : ex2(fmaf(s[4 * j + 2 * hh + e], a.qk, -nmf));
      }
      resc = __any_sync(0xffffffffu, upd[0] | upd[1]);
#pragma unroll
      for (int i = 0; i < 16; i++) pa[i] = pk(s[2 * i], s[2 * i + 1]);
    }
    if (resc) {
#pragma unroll
      for (int j = 0; j < 8; j++) { o[4 * j] *= scl[0]; o[4 * j + 1] *= scl[0]; o[4 * j + 2] *= scl[1]; o[4 * j + 3] *= scl[1]; }
      lacc[0] *= scl[0]; lacc[1] *= scl[0]; lacc[2] *= scl[1]; lacc[3] *= scl[1];
    }
    if constexpr (OV && !XB) asm volatile("bar.sync %0, 128;\n" :: "r"(1 + h) : "memory");  // every warp's f16 V of this step is in place
    wg_fence();
#pragma unroll
    for (int kc = 0; kc < 4; kc++) {
      if (XM) { o[kc] += __uint_as_float(pa[4 * kc] ^ pa[4 * kc + 3]); lacc[kc] += __uint_as_float(pa[kc]); continue; }
      const uint64_t dv = dV + (bi & 1) * 512 + kc * 128;
      if (kc == 0) wg_rsz(o, pa[0], pa[1], pa[2], pa[3], dv, bi > 0);
      else wg_rs(o, pa[4 * kc], pa[4 * kc + 1], pa[4 * kc + 2], pa[4 * kc + 3], dv);
      wg_rs8(lacc, pa[4 * kc], pa[4 * kc + 1], pa[4 * kc + 2], pa[4 * kc + 3], dOnes, kc > 0 || bi > 0);
    }
    wg_commit();
  }
  wg_wait0();
  PS(3);
  if (nb == 0) {
#pragma unroll
    for (int i = 0; i < 32; i++) o[i] = 0.f;
    lacc[0] = lacc[1] = lacc[2] = lacc[3] = 0.f;
  }
  const size_t it = ((size_t)bq * a.hkv + h) * a.nsplit + sp;
#pragma unroll
  for (int hh = 0; hh < 2; hh++) {
    const float l = lacc[2 * hh];  // full row sum (every column of the ones product)
    const int row = 16 * w + g + 8 * hh;
    float* wo = a.wo + (it * 64 + row) * D;  // 64 rows per (block, head, split)
#pragma unroll
    for (int j = 0; j < 8; j++) *(float2*)(wo + 8 * j + 2 * t) = make_float2(o[4 * j + 2 * hh], o[4 * j + 2 * hh + 1]);
    if (t == 0) { a.wml[(it * 64 + row) * 2] = mx[hh]; a.wml[(it * 64 + row) * 2 + 1] = l; }
  }
  if constexpr (MRG) {
    // the last nmg arrivers (a power of two <= min(128, nsplit), host-chosen) of a request merge its 128 rows: merger rank takes
    // rows rank, rank + nmg, ...; within a merger, G split groups x 16 dims-of-4 threads per row, combined in shared memory
    PS(4);
    __syncthreads();
    __shared__ int rank_s;
    if (tid == 0) { int v; asm volatile("atom.add.acq_rel.gpu.global.s32 %0, [%1], 1;\n" : "=r"(v) : "l"(f.cnt + bq) : "memory"); rank_s = v; }
    __syncthreads();
    const int nmg = f.nmg, rank = rank_s - (a.nsplit - nmg);
    PS(5);
    if (rank < 0) return;
    const int R = 128 / nmg, G = R >= 16 ? 1 : 16 / R, RP = 16 / G;  // rows per merger, split groups, rows per pass
    const int d4 = tid & 15, sl = tid >> 4, kg = sl % G, rl = sl / G;
    float gv0[4];  // the first pass's gate values, loaded before the spin (overlaps the wait for the last split)
    {
      const int rr = rank + nmg * rl, r = rr & 63, l = t0 + (r >> 3), hd = (rr >> 6) * 8 + (r & 7);
      if (kg == 0 && (r >> 3) < lt) {
        const uint2 u = *(const uint2*)(f.qkv + ((size_t)b * a.L + l) * (2 * 16 + 4) * D + 20 * D + hd * D + d4 * 4);
        const __nv_bfloat16* bb = (const __nv_bfloat16*)&u;
#pragma unroll
        for (int i = 0; i < 4; i++) gv0[i] = __bfloat162float(bb[i]);
      }
    }
    if (tid == 0) {
      int v;
      do { asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(f.cnt + bq) : "memory"); if (v < a.nsplit) __nanosleep(32); } while (v < a.nsplit);
    }
    __syncthreads();
    PS(6);
    asm volatile("cp.async.wait_all;\n" ::: "memory");
    float* xo = (float*)smem + ((sb - (uint32_t)__cvta_generic_to_shared(smem)) >> 2);  // [16 slots][64] partial O
    float* xm = xo + 16 * 64;  // [16] max, [16] sum
    for (int p0 = 0; p0 < R; p0 += RP) {
      const int rr = rank + nmg * (p0 + rl), hh = rr >> 6, r = rr & 63, base = (bq * a.hkv + hh) * a.nsplit;
      float Mr = -INFINITY, Lr = 0.f;
      float4 a4 = make_float4(0.f, 0.f, 0.f, 0.f);
      if (f.fl & 1) {
        mrg8(kg, G, a.nsplit, [&](int k2) { return __ldcg((const float2*)(a.wml + ((size_t)(base + k2) * 64 + r) * 2)); },
             [&](int k2) { return __ldcg((const float4*)(a.wo + ((size_t)(base + k2) * 64 + r) * D) + d4); }, Mr, Lr, a4);
      } else {
#pragma unroll 8
      for (int k2 = kg; k2 < a.nsplit; k2 += G) Mr = fmaxf(Mr, __ldcg(a.wml + ((size_t)(base + k2) * 64 + r) * 2));
#pragma unroll 8
      for (int k2 = kg; k2 < a.nsplit; k2 += G) {
        const float2 ml = __ldcg((const float2*)(a.wml + ((size_t)(base + k2) * 64 + r) * 2));
        const float4 v = __ldcg((const float4*)(a.wo + ((size_t)(base + k2) * 64 + r) * D) + d4);
        const float c1 = ml.x == -INFINITY ? 0.f : ex2(ml.x - Mr);
        Lr = fmaf(ml.y, c1, Lr);
        a4.x = fmaf(v.x, c1, a4.x); a4.y = fmaf(v.y, c1, a4.y); a4.z = fmaf(v.z, c1, a4.z); a4.w = fmaf(v.w, c1, a4.w);
      }
      }
      if (G > 1) {
        ((float4*)(xo + sl * 64))[d4] = a4;
        if (d4 == 0) { xm[sl] = Mr; xm[16 + sl] = Lr; }
        __syncthreads();
        if (kg == 0) {
          float M2 = -INFINITY;
          for (int q = 0; q < G; q++) M2 = fmaxf(M2, xm[sl + q]);
          a4 = make_float4(0.f, 0.f, 0.f, 0.f); Lr = 0.f;
          for (int q = 0; q < G; q++) {
            const float mq = xm[sl + q], c1 = mq == -INFINITY ? 0.f : ex2(mq - M2);
            const float4 v = ((const float4*)(xo + (sl + q) * 64))[d4];
            Lr = fmaf(xm[16 + sl + q], c1, Lr);
            a4.x = fmaf(v.x, c1, a4.x); a4.y = fmaf(v.y, c1, a4.y); a4.z = fmaf(v.z, c1, a4.z); a4.w = fmaf(v.w, c1, a4.w);
          }
        }
      }
      if (kg == 0) {
        const float inv = f.vscale / Lr;
        a4.x *= inv; a4.y *= inv; a4.z *= inv; a4.w *= inv;
        float ss = a4.x * a4.x + a4.y * a4.y + a4.z * a4.z + a4.w * a4.w;
#pragma unroll
        for (int x = 1; x < 16; x <<= 1) ss += __shfl_xor_sync(0xffffu << (lane & 16), ss, x);
        const float rn = rsqrtf(ss / D + 1e-5f);
        const int l = t0 + (r >> 3), hd = hh * 8 + (r & 7);
        if ((r >> 3) < lt) {
          const __nv_bfloat16* gt = f.qkv + ((size_t)b * a.L + l) * (2 * 16 + 4) * D + 20 * D + hd * D + d4 * 4;
          const float vals[4] = {a4.x, a4.y, a4.z, a4.w};
          __nv_bfloat16 ob[4];
#pragma unroll
          for (int i2 = 0; i2 < 4; i2++) ob[i2] = __float2bfloat16(vals[i2] * rn / (1.f + expf(-(p0 == 0 ? gv0[i2] : __bfloat162float(gt[i2])))));
          *(uint2*)(f.out + ((size_t)b * a.L + l) * 16 * D + hd * D + d4 * 4) = *(uint2*)ob;
        }
      }
      if (G > 1) __syncthreads();
    }
    PS(7);
    const int nbq = a.B * nqb;
    if (tid == 0 && atomicAdd(f.cnt + nbq + bq, 1) == nmg - 1) { f.cnt[bq] = 0; f.cnt[nbq + bq] = 0; }
  }
}

// ---------------------------------------------------------------- warp-specialized verify (L = 8): 2 consumer warpgroups (one per
// KV head: QK wgmma, softmax, PV wgmma only) + 1 producer warp (page-table lookups, fdec-style gather of whole 128 B K/V rows,
// f16 conversion of V for both heads). Raw ring of RX 16 KB slots (64 keys), double-buffered f16 V; mbarriers: full[slot] (producer:
// step landed and V converted), emptyR[slot] / emptyV[buf] (one arrive per consumer warp after the wgmma reading it retired).
#define RX RSW
__device__ __forceinline__ void mb_init(uint32_t a, int n) { asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(a), "r"(n) : "memory"); }
__device__ __forceinline__ void mb_arrive(uint32_t a) { asm volatile("{\n.reg .b64 st;\nmbarrier.arrive.shared.b64 st, [%0];\n}\n" :: "r"(a) : "memory"); }
__device__ __forceinline__ void mb_wait(uint32_t a, int par) {
  uint32_t ok = 0;
  while (!ok) {
#if __CUDA_ARCH__ >= 900
    asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n" : "=r"(ok) : "r"(a), "r"(par) : "memory");
#else
    asm volatile("{\n.reg .pred p;\nmbarrier.test_wait.parity.shared.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n" : "=r"(ok) : "r"(a), "r"(par) : "memory");
#endif
  }
}
// CV: the producer converts V (else each consumer warpgroup converts its head's half, then a warpgroup barrier before PV)
template <bool PS1, bool FS, bool CV>
__global__ void __launch_bounds__(288, 2) attnx(A a, WF f) {
  const int tid = threadIdx.x, wt = tid & 127, h = tid >> 7, w = wt >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const bool prod = tid >= 256;
  const int nqb = (a.L + 7) >> 3, sp = blockIdx.x % a.nsplit, bq = blockIdx.x / a.nsplit, qb = bq % nqb, b = bq / nqb;
  const int t0 = qb * 8, lt = min(8, a.L - t0);
  const int qr_ = wt >> 1, qhf = wt & 1, ql = qr_ >> 3;
  const int nrr = (tid >> 4) & 15, nkh = nrr & 1, nlt = nrr >> 1, nd0 = (tid & 15) * 4;
  uint4 qraw[4];
  uint2 kraw = make_uint2(0, 0), vraw = make_uint2(0, 0);
  PS(0);
  if (!prod) {
    const __nv_bfloat16* src = f.qkv + ((size_t)b * a.L + ql) * (2 * 16 + 4) * D + (h * 8 + (qr_ & 7)) * D + qhf * 32;
#pragma unroll
    for (int i = 0; i < 4; i++) qraw[i] = *(const uint4*)(src + 8 * i);
    const __nv_bfloat16* krow = f.qkv + ((size_t)b * a.L + nlt) * (2 * 16 + 4) * D;
    kraw = *(const uint2*)(krow + (16 + nkh) * D + nd0); vraw = *(const uint2*)(krow + (18 + nkh) * D + nd0);
  }
  const int seq = a.seq[b], nk = seq + t0 + lt;
  const int maxk = a.maxp * a.ps, stt = (f.fl & 2) && !a.window;
  const int k0 = a.window ? max(0, seq + t0 - a.window + 1) : 0, span = stt ? maxk : nk - k0;
  const int ks = k0 + (int)((long long)span * sp / a.nsplit), ke0 = k0 + (int)((long long)span * (sp + 1) / a.nsplit);
  const int ke = stt ? min(ke0, nk) : ke0;
  const int nb = max(0, (ke - ks + 63) >> 6);
  extern __shared__ __align__(1024) uint8_t smem[];
  const uint32_t sb = ((uint32_t)__cvta_generic_to_shared(smem) + 1023) & ~1023u;
  const uint32_t ring = sb, qs = sb + RX * 16384, vfb = qs + 8192;  // f16 V: head hh, buffer u at vfb + hh * 16384 + u * 8192
  const uint32_t ones = vfb + 32768, mbf = ones + 1024, mbr = mbf + 8 * RX, mbv = mbr + 8 * RX;
  if (tid == 0) {
    for (int i = 0; i < RX; i++) { mb_init(mbf + 8 * i, 1); mb_init(mbr + 8 * i, 8); }
    mb_init(mbv, 8); mb_init(mbv + 8, 8);
  }
  if (!prod) for (int x = wt; x < 256; x += 128) asm volatile("st.shared.u32 [%0], %1;\n" :: "r"(ones + x * 4), "r"(0x3C003C00u));
  const int* ptb = a.pt + (size_t)b * a.maxp;
  auto rowg = [&](int kp) -> int {
    const int kc = min(kp, stt ? maxk - 1 : ke - 1);
    const int rr = PS1 ? __ldg(ptb + kc) : __ldg(ptb + kc / a.ps) * a.ps + kc % a.ps;
    return kp < ke ? rr : -1;
  };
  // ---- producer: lane owns the rows of keys lane and lane + 32 of a step; copy i moves key 4 i + (lane >> 3), 16 B chunk lane & 7
  // of the K and V rows (4 whole 128 B rows per instruction, as fdec)
  const int c8 = lane & 7;
  auto pissue = [&](int slot, int kbi, int r0, int r1) {
    if (kbi < ke) {
#pragma unroll
      for (int i = 0; i < 16; i++) {
        const int key = 4 * i + (lane >> 3);
        const int rw = __shfl_sync(0xffffffffu, i < 8 ? r0 : r1, key & 31);
        if (rw == -2) continue;
        const uint32_t dd = ring + slot * 16384 + key * 128 + ((c8 ^ (key & 7)) << 4);
        const size_t off = (size_t)max(rw, 0) * 128 + c8 * 16;
        const int n = rw >= 0 ? 16 : 0;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dd), "l"(a.kc + off), "r"(n));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dd + 8192), "l"(a.vc + off), "r"(n));
      }
    }
    asm volatile("cp.async.commit_group;\n");
  };
  auto pconv = [&](int slot, int u) {  // V of keys lane, lane + 32 (both heads) -> f16 tiles, MN-major 128 B swizzle per head
#pragma unroll
    for (int kk = 0; kk < 2; kk++) {
      const int key = lane + 32 * kk;
#pragma unroll
      for (int c = 0; c < 8; c++) {
        uint4 x;
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(x.x), "=r"(x.y), "=r"(x.z), "=r"(x.w) : "r"(ring + slot * 16384 + 8192 + key * 128 + ((c ^ (key & 7)) << 4)));
        const int hh = c >> 2, j0 = 2 * (c & 3);
        const uint32_t dst = vfb + hh * 16384 + u * 8192 + key * 128;
        asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" :: "r"(dst + ((j0 ^ (key & 7)) << 4)), "r"(e2h(x.x)), "r"(e2h(x.x >> 16)), "r"(e2h(x.y)), "r"(e2h(x.y >> 16)));
        asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" :: "r"(dst + (((j0 + 1) ^ (key & 7)) << 4)), "r"(e2h(x.z)), "r"(e2h(x.z >> 16)), "r"(e2h(x.w)), "r"(e2h(x.w >> 16)));
      }
    }
  };
  int pr[3][2];
  if (prod) {
#pragma unroll
    for (int j = 0; j < RX - 1; j++) {  // the block's own keys are not in the cache yet: the consumers write them into the ring
      int r0 = rowg(ks + j * 64 + lane), r1 = rowg(ks + j * 64 + lane + 32);
      if (ks + j * 64 + lane >= seq && ks + j * 64 + lane < ke) r0 = -2;
      if (ks + j * 64 + lane + 32 >= seq && ks + j * 64 + lane + 32 < ke) r1 = -2;
      pissue(j, ks + j * 64, r0, r1);
    }
#pragma unroll
    for (int j = 0; j < 3; j++) { pr[j][0] = rowg(ks + (RX - 1 + j) * 64 + lane); pr[j][1] = rowg(ks + (RX - 1 + j) * 64 + lane + 32); }
    PS(1);
  } else {
    // ---- consumers' prologue: query rows (RMSNorm, RoPE, E4M3) and the block's own keys/values (cache + prefetched ring slots)
    float x[32], ss = 0.f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const __nv_bfloat162* bb = (const __nv_bfloat162*)&qraw[i];
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
        float sn, cs2; rope_cs(f.invf, seq + ql, (qhf * 32 + i) >> 1, sn, cs2);
        const float e = x[i], o = x[i + 1];
        x[i] = e * cs2 - o * sn; x[i + 1] = e * sn + o * cs2;
      }
    }
#pragma unroll
    for (int u = 0; u < 2; u++) {
      uint32_t q4[4];
#pragma unroll
      for (int i = 0; i < 4; i++) {
        uint32_t wv = 0;
#pragma unroll
        for (int j = 0; j < 4; j++) { __nv_fp8_e4m3 fq(x[16 * u + 4 * i + j]); wv |= (uint32_t)fq.__x << (8 * j); }
        q4[i] = wv;
      }
      const int c = 4 * h + 2 * qhf + u;
      asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" :: "r"(qs + qr_ * 128 + ((c ^ (qr_ & 7)) << 4)), "r"(q4[0]), "r"(q4[1]), "r"(q4[2]), "r"(q4[3]));
    }
    if (ke > seq) {
      float k[4], v[4], s2 = 0.f;
      const __nv_bfloat16* kr = (const __nv_bfloat16*)&kraw; const __nv_bfloat16* vr = (const __nv_bfloat16*)&vraw;
#pragma unroll
      for (int i = 0; i < 4; i++) { k[i] = __bfloat162float(kr[i]); v[i] = __bfloat162float(vr[i]); s2 += k[i] * k[i]; }
#pragma unroll
      for (int o = 1; o < 16; o <<= 1) s2 += __shfl_xor_sync(0xffffffffu, s2, o);
      const float kr2 = rsqrtf(s2 / D + 1e-5f);
#pragma unroll
      for (int i = 0; i < 4; i++) k[i] *= kr2;
      if (f.theta > 0.f) {
#pragma unroll
        for (int i = 0; i < 4; i += 2) {
          float sn, cs2; rope_cs(f.invf, seq + nlt, (nd0 + i) >> 1, sn, cs2);
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
      if (nlt < lt) {
        const size_t lc = (size_t)f.loc[b * a.L + nlt] * 128 + nkh * 64 + nd0;
        *(uint32_t*)((uint8_t*)a.kc + lc) = kb4;
        *(uint32_t*)((uint8_t*)a.vc + lc) = vb4;
        const int kq = seq + nlt - ks, j = kq >> 6, kk = kq & 63;
        if (kq >= 0 && j < RX - 1 && seq + nlt < ke) {
          const uint32_t dd = ring + j * 16384 + kk * 128 + ((((nkh * 64 + nd0) >> 4) ^ (kk & 7)) << 4) + (nd0 & 15);
          asm volatile("st.shared.u32 [%0], %1;\n" :: "r"(dd), "r"(kb4));
          asm volatile("st.shared.u32 [%0], %1;\n" :: "r"(dd + 8192), "r"(vb4));
        }
      }
      __threadfence();
    }
  }
  fence_async();  // generic-proxy shared stores (query tile, new keys, ones) before the consumers' wgmma reads them
  __syncthreads();  // query tile, new keys in the ring, ones tile, barriers initialised
  PS(2);
  float o[32], mx[2] = {-INFINITY, -INFINITY};
  float lacc[4];
  if (prod) {
    for (int i = 0; i < nb; i++) {
      if (i >= 1) mb_wait(mbr + 8 * ((i - 1) % RX), ((i - 1) / RX) & 1);  // raw slot of step i - 1: both heads' QK retired
      pissue((i + RX - 1) % RX, ks + (i + RX - 1) * 64, pr[0][0], pr[0][1]);
      pr[0][0] = pr[1][0]; pr[0][1] = pr[1][1]; pr[1][0] = pr[2][0]; pr[1][1] = pr[2][1];
      pr[2][0] = rowg(ks + (i + RX + 2) * 64 + lane); pr[2][1] = rowg(ks + (i + RX + 2) * 64 + lane + 32);
      asm volatile("cp.async.wait_group %0;\n" :: "n"(RX - 1));  // step i landed (this lane's copies)
      __syncwarp();
      if (CV) {
        if (i >= 2) mb_wait(mbv + 8 * (i & 1), ((i - 2) >> 1) & 1);  // f16 buffer of step i - 2: both heads' PV retired
        pconv(i % RX, i & 1);
      }
      fence_async();
      __syncwarp();
      if (lane == 0) mb_arrive(mbf + 8 * (i % RX));
    }
    asm volatile("cp.async.wait_all;\n" ::: "memory");
  } else {
    const int lw0 = 2 * w;
    const int lomax = a.window ? max(0, seq + t0 + lt - 1 - a.window + 1) : 0;
    const uint64_t dQ = gdesc(qs + 64 * h, 16, 1024), dK = gdesc(ring + 64 * h, 16, 1024), dV = gdesc(vfb + h * 16384, WVL, WVS);
    const uint64_t dOnes = (uint64_t)(ones >> 4) | ((uint64_t)(128 >> 4) << 16) | ((uint64_t)(256 >> 4) << 32);
    for (int bi = 0, kb = ks; bi < nb; bi++, kb += 64) {
      const int slot = bi % RX;
      float s[32];
      mb_wait(mbf + 8 * slot, (bi / RX) & 1);
      wg_fence();
      wg_ss8<0>(s, dQ, dK + slot * 1024);
      wg_ss8<1>(s, dQ + 2, dK + slot * 1024 + 2);
      wg_commit();
      if (!CV) {  // this head's V half to f16 (buffer bi & 1: its last reader PV(bi - 2) retired before the previous barrier)
        const int vkey = wt >> 1;
        const uint32_t raw = ring + slot * 16384 + 8192 + vkey * 128, vf = vfb + h * 16384 + (bi & 1) * 8192 + vkey * 128;
#pragma unroll
        for (int u = 0; u < 2; u++) {
          const int c = 4 * h + 2 * (wt & 1) + u, j0 = 2 * (c - 4 * h);
          uint4 x;
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(x.x), "=r"(x.y), "=r"(x.z), "=r"(x.w) : "r"(raw + ((c ^ (vkey & 7)) << 4)));
          asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" :: "r"(vf + ((j0 ^ (vkey & 7)) << 4)), "r"(e2h(x.x)), "r"(e2h(x.x >> 16)), "r"(e2h(x.y)), "r"(e2h(x.y >> 16)));
          asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" :: "r"(vf + (((j0 + 1) ^ (vkey & 7)) << 4)), "r"(e2h(x.z)), "r"(e2h(x.z >> 16)), "r"(e2h(x.w)), "r"(e2h(x.w >> 16)));
        }
        fence_async();
      }
      wg_wait0();  // S ready; the previous step's PV retired
      if (lane == 0) { mb_arrive(mbr + 8 * slot); if (CV && bi >= 1) mb_arrive(mbv + 8 * ((bi - 1) & 1)); }
      const bool full = kb + 64 <= ke && kb + 64 <= seq + t0 + 1 && kb >= lomax && lt == 8;
      if (!full) {
#pragma unroll
        for (int hh = 0; hh < 2; hh++) {
          const int p = lw0 + hh < lt ? seq + t0 + lw0 + hh : -1, lo = a.window ? max(0, p - a.window + 1) : 0;
#pragma unroll
          for (int j = 0; j < 8; j++)
#pragma unroll
            for (int e = 0; e < 2; e++) { const int kp = kb + 8 * j + 2 * t + e; if (!(kp < ke && kp <= p && kp >= lo)) s[4 * j + 2 * hh + e] = -INFINITY; }
        }
      }
      uint32_t pa[16];
      float scl[2] = {1.f, 1.f};
      bool resc = false, exact = true;
      if constexpr (FS) exact = !__all_sync(0xffffffffu, mx[0] != -INFINITY && mx[1] != -INFINITY);
      if (!exact) {
#pragma unroll
        for (int j = 0; j < 8; j++)
#pragma unroll
          for (int hh = 0; hh < 2; hh++)
#pragma unroll
            for (int e = 0; e < 2; e++) s[4 * j + 2 * hh + e] = ex2(fmaf(s[4 * j + 2 * hh + e], a.qk, -mx[hh]));
#pragma unroll
        for (int i = 0; i < 16; i++) pa[i] = pk(s[2 * i], s[2 * i + 1]);
        __half2 hm = *(__half2*)&pa[0];
#pragma unroll
        for (int i = 1; i < 16; i++) hm = __hmax2(hm, *(__half2*)&pa[i]);
        const uint32_t hb = *(uint32_t*)&hm;
        if (__any_sync(0xffffffffu, (hb & 0xffffu) > LZB || (hb >> 16) > LZB)) {
#pragma unroll
          for (int hh = 0; hh < 2; hh++) {
            float pm = 0.f;
#pragma unroll
            for (int j = 0; j < 8; j++) pm = fmaxf(pm, fmaxf(s[4 * j + 2 * hh], s[4 * j + 2 * hh + 1]));
            pm = fmaxf(pm, __shfl_xor_sync(0xffffffffu, pm, 1));
            pm = fmaxf(pm, __shfl_xor_sync(0xffffffffu, pm, 2));
            if (pm > LZP) {
              const float inv = 1.f / pm;
              mx[hh] += __log2f(pm);
              scl[hh] = inv;
#pragma unroll
              for (int j = 0; j < 8; j++) { s[4 * j + 2 * hh] *= inv; s[4 * j + 2 * hh + 1] *= inv; }
            }
          }
#pragma unroll
          for (int i = 0; i < 16; i++) pa[i] = pk(s[2 * i], s[2 * i + 1]);
          resc = true;
        }
      } else {
        bool upd[2];
#pragma unroll
        for (int hh = 0; hh < 2; hh++) {
          float bm = -INFINITY;
#pragma unroll
          for (int j = 0; j < 8; j++) bm = fmaxf(bm, fmaxf(s[4 * j + 2 * hh], s[4 * j + 2 * hh + 1]));
          bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, 1));
          bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, 2));
          const bool up = bm * a.qk > mx[hh] + LZT;
          const float nm = up ? fmaxf(mx[hh], bm * a.qk) : mx[hh];
          const float nmf = nm == -INFINITY ? 0.f : nm;
          upd[hh] = up;
          scl[hh] = up ? ex2(mx[hh] - nmf) : 1.f;
          mx[hh] = nm;
#pragma unroll
          for (int j = 0; j < 8; j++)
#pragma unroll
            for (int e = 0; e < 2; e++) s[4 * j + 2 * hh + e] = ex2(fmaf(s[4 * j + 2 * hh + e], a.qk, -nmf));
        }
        resc = __any_sync(0xffffffffu, upd[0] | upd[1]);
#pragma unroll
        for (int i = 0; i < 16; i++) pa[i] = pk(s[2 * i], s[2 * i + 1]);
      }
      if (resc) {
#pragma unroll
        for (int j = 0; j < 8; j++) { o[4 * j] *= scl[0]; o[4 * j + 1] *= scl[0]; o[4 * j + 2] *= scl[1]; o[4 * j + 3] *= scl[1]; }
        lacc[0] *= scl[0]; lacc[1] *= scl[0]; lacc[2] *= scl[1]; lacc[3] *= scl[1];
      }
      if (!CV) asm volatile("bar.sync %0, 128;\n" :: "r"(1 + h) : "memory");  // the warpgroup's f16 V of this step is in place
      wg_fence();
#pragma unroll
      for (int kc = 0; kc < 4; kc++) {
        const uint64_t dv = dV + (bi & 1) * 512 + kc * 128;
        if (kc == 0) wg_rsz(o, pa[0], pa[1], pa[2], pa[3], dv, bi > 0);
        else wg_rs(o, pa[4 * kc], pa[4 * kc + 1], pa[4 * kc + 2], pa[4 * kc + 3], dv);
        wg_rs8(lacc, pa[4 * kc], pa[4 * kc + 1], pa[4 * kc + 2], pa[4 * kc + 3], dOnes, kc > 0 || bi > 0);
      }
      wg_commit();
    }
    wg_wait0();
    PS(3);
    if (nb == 0) {
#pragma unroll
      for (int i = 0; i < 32; i++) o[i] = 0.f;
      lacc[0] = lacc[1] = lacc[2] = lacc[3] = 0.f;
    }
    const size_t it = ((size_t)bq * a.hkv + h) * a.nsplit + sp;
#pragma unroll
    for (int hh = 0; hh < 2; hh++) {
      const int row = 16 * w + g + 8 * hh;
      float* wo = a.wo + (it * 64 + row) * D;
#pragma unroll
      for (int j = 0; j < 8; j++) *(float2*)(wo + 8 * j + 2 * t) = make_float2(o[4 * j + 2 * hh], o[4 * j + 2 * hh + 1]);
      if (t == 0) { a.wml[(it * 64 + row) * 2] = mx[hh]; a.wml[(it * 64 + row) * 2 + 1] = lacc[2 * hh]; }
    }
  }
  // ---- split merge (last nmg arrivers; consumer threads only do the work, everyone joins the barriers)
  PS(4);
  __syncthreads();
  __shared__ int rank_s;
  if (tid == 0) { int v; asm volatile("atom.add.acq_rel.gpu.global.s32 %0, [%1], 1;\n" : "=r"(v) : "l"(f.cnt + bq) : "memory"); rank_s = v; }
  __syncthreads();
  const int nmg = f.nmg, rank = rank_s - (a.nsplit - nmg);
  PS(5);
  if (rank < 0) return;
  const int R = 128 / nmg, G = R >= 16 ? 1 : 16 / R, RP = 16 / G;
  const int d4 = tid & 15, sl = (tid >> 4) & 15, kg = sl % G, rl = sl / G;
  float gv0[4];
  {
    const int rr = rank + nmg * rl, r = rr & 63, l = t0 + (r >> 3), hd = (rr >> 6) * 8 + (r & 7);
    if (!prod && kg == 0 && (r >> 3) < lt) {
      const uint2 u = *(const uint2*)(f.qkv + ((size_t)b * a.L + l) * (2 * 16 + 4) * D + 20 * D + hd * D + d4 * 4);
      const __nv_bfloat16* bb = (const __nv_bfloat16*)&u;
#pragma unroll
      for (int i = 0; i < 4; i++) gv0[i] = __bfloat162float(bb[i]);
    }
  }
  if (tid == 0) {
    int v;
    do { asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(f.cnt + bq) : "memory"); if (v < a.nsplit) __nanosleep(32); } while (v < a.nsplit);
  }
  __syncthreads();
  PS(6);
  float* xo = (float*)smem + ((sb - (uint32_t)__cvta_generic_to_shared(smem)) >> 2);
  float* xm = xo + 16 * 64;
  for (int p0 = 0; p0 < R; p0 += RP) {
    const int rr = rank + nmg * (p0 + rl), hh = rr >> 6, r = rr & 63, base = (bq * a.hkv + hh) * a.nsplit;
    float Mr = -INFINITY, Lr = 0.f;
    float4 a4 = make_float4(0.f, 0.f, 0.f, 0.f);
    if (!prod)
      mrg8(kg, G, a.nsplit, [&](int k2) { return __ldcg((const float2*)(a.wml + ((size_t)(base + k2) * 64 + r) * 2)); },
           [&](int k2) { return __ldcg((const float4*)(a.wo + ((size_t)(base + k2) * 64 + r) * D) + d4); }, Mr, Lr, a4);
    if (G > 1) {
      if (!prod) { ((float4*)(xo + sl * 64))[d4] = a4; if (d4 == 0) { xm[sl] = Mr; xm[16 + sl] = Lr; } }
      __syncthreads();
      if (!prod && kg == 0) {
        float M2 = -INFINITY;
        for (int q = 0; q < G; q++) M2 = fmaxf(M2, xm[sl + q]);
        a4 = make_float4(0.f, 0.f, 0.f, 0.f); Lr = 0.f;
        for (int q = 0; q < G; q++) {
          const float mq = xm[sl + q], c1 = mq == -INFINITY ? 0.f : ex2(mq - M2);
          const float4 v = ((const float4*)(xo + (sl + q) * 64))[d4];
          Lr = fmaf(xm[16 + sl + q], c1, Lr);
          a4.x = fmaf(v.x, c1, a4.x); a4.y = fmaf(v.y, c1, a4.y); a4.z = fmaf(v.z, c1, a4.z); a4.w = fmaf(v.w, c1, a4.w);
        }
      }
    }
    if (!prod && kg == 0) {
      const float inv = f.vscale / Lr;
      a4.x *= inv; a4.y *= inv; a4.z *= inv; a4.w *= inv;
      float ss = a4.x * a4.x + a4.y * a4.y + a4.z * a4.z + a4.w * a4.w;
#pragma unroll
      for (int x = 1; x < 16; x <<= 1) ss += __shfl_xor_sync(0xffffu << (lane & 16), ss, x);
      const float rn = rsqrtf(ss / D + 1e-5f);
      const int l = t0 + (r >> 3), hd = hh * 8 + (r & 7);
      if ((r >> 3) < lt) {
        const __nv_bfloat16* gt = f.qkv + ((size_t)b * a.L + l) * (2 * 16 + 4) * D + 20 * D + hd * D + d4 * 4;
        const float vals[4] = {a4.x, a4.y, a4.z, a4.w};
        __nv_bfloat16 ob[4];
#pragma unroll
        for (int i2 = 0; i2 < 4; i2++) ob[i2] = __float2bfloat16(vals[i2] * rn / (1.f + expf(-(p0 == 0 ? gv0[i2] : __bfloat162float(gt[i2])))));
        *(uint2*)(f.out + ((size_t)b * a.L + l) * 16 * D + hd * D + d4 * 4) = *(uint2*)ob;
      }
    }
    if (G > 1) __syncthreads();
  }
  PS(7);
  const int nbq = a.B * nqb;
  if (tid == 0 && atomicAdd(f.cnt + nbq + bq, 1) == nmg - 1) { f.cnt[bq] = 0; f.cnt[nbq + bq] = 0; }
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
  int B, ps, maxp, window, NS, fl; float theta, qsc, qk, kinv, vinv, vscale; unsigned long long* prof; const float* invf;
};
#ifdef PROF
__device__ __forceinline__ unsigned long long gtm() { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
#define PT(k) do { if ((threadIdx.x & 31) == 0) p.prof[(blockIdx.x * 8 + (threadIdx.x >> 5)) * 16 + (k)] = gtm(); } while (0)
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

template <bool CL, bool MU>
#ifndef FDMINB
#define FDMINB 1
#endif
__global__ void __launch_bounds__(256, FDMINB) fdec(F p) {
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, t = lane & 3;
#ifdef FD_EMPTY
  if (tid == 0 && p.NS < 0) p.out[0] = __float2bfloat16(0.f);
  return;
#endif
  PT(0);
  int b = blockIdx.x / p.NS, s = blockIdx.x % p.NS, nsp = p.NS;
  if (p.fl & 8) {  // full layers, B <= 32: splits balanced over the batch's total keys (p.NS = most splits per request)
    const int kb_ = lane < p.B ? p.seq[lane] + 1 : 0, G = gridDim.x;
    const int K = __reduce_add_sync(0xffffffffu, kb_), mxk = __reduce_max_sync(0xffffffffu, kb_);
    int lo = max((K + G - 1) / G, (mxk + p.NS - 1) / p.NS), hi = max(lo, mxk);
    while (lo < hi) {
      const int c = (lo + hi) >> 1;
      if ((int)__reduce_add_sync(0xffffffffu, lane < p.B ? (kb_ + c - 1) / c : 0) <= G) hi = c; else lo = c + 1;
    }
    const int nl = lane < p.B ? (kb_ + lo - 1) / lo : 0;
    int pre = nl;
#pragma unroll
    for (int o2 = 1; o2 < 32; o2 <<= 1) { const int v2 = __shfl_up_sync(0xffffffffu, pre, o2); if (lane >= o2) pre += v2; }
    const int bb = __popc(__ballot_sync(0xffffffffu, lane < p.B && pre <= (int)blockIdx.x));
    if (bb >= p.B) return;  // spare CTA
    nsp = __shfl_sync(0xffffffffu, nl, bb);
    s = (int)blockIdx.x - (__shfl_sync(0xffffffffu, pre, bb) - nsp);
    b = bb;
  }
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
  const uint2 gpre = (p.fl & 1) ? *(const uint2*)(row0 + 20 * D + tid * 4) : make_uint2(0, 0);  // gate rows for the output
  const int seq = p.seq[b];
  // fl & 2 (full layers): splits over the page table's capacity, independent of seq, so the page-table loads below do not
  // wait for it; keys past seq are cut by we
  const int maxk = p.maxp * p.ps, stt = (p.fl & 2) && !p.window;
  const int k0 = p.window ? max(0, seq - p.window + 1) : 0, span = stt ? maxk : seq + 1 - k0;
  const int cs = k0 + (int)((long long)span * s / nsp), ce = k0 + (int)((long long)span * (s + 1) / nsp);
  const int ws = cs + (int)((long long)(ce - cs) * w / 8), we0 = cs + (int)((long long)(ce - cs) * (w + 1) / 8);
  const int we = stt ? min(we0, seq + 1) : we0;
  const __nv_bfloat16* row = p.qkv + (size_t)b * (2 * 16 + 4) * D;
  extern __shared__ __align__(16) uint8_t smem[];
  const uint32_t sb = (uint32_t)__cvta_generic_to_shared(smem);
  const uint32_t tab = sb + 8 * RBF * 4096;  // new token's K row (128 B) then V row (128 B)
  const int* ptb = p.pt + (size_t)b * p.maxp;
  const uint32_t ring = sb + w * RBF * 4096;
  // cache row of key ws..: -1 past the range, -2 for the new token (taken from the table)
  auto rowsfor = [&](int kb) -> int {
    const int kp = kb + (lane & 15), kc = min(kp, maxk - 1);
    const int raw = ptb[kc / p.ps] * p.ps + kc % p.ps;  // address independent of seq (fl & 2): issued before seq lands
    return (kb < we && kp < we) ? (kp >= seq ? -2 : raw) : -1;
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
        asm volatile("cp.async.cg.shared.global" FDH " [%0], [%1], 16, %2;\n" :: "r"(dd), "l"(p.kc + off), "r"(rw >= 0 ? 16 : 0));
        asm volatile("cp.async.cg.shared.global" FDH " [%0], [%1], 16, %2;\n" :: "r"(dd + 2048), "l"(p.vc + off), "r"(rw >= 0 ? 16 : 0));
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
  PT(8);
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
  const uint32_t gsm = tab + 256 + 8 * 2048;  // 16 x 64 gate values (bf16), prefetched
  if (p.fl & 1) asm volatile("st.shared.v2.u32 [%0], {%1,%2};\n" :: "r"(gsm + tid * 8), "r"(gpre.x), "r"(gpre.y));
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
  auto gate4 = [&](int rr, int d4_, float* gv) {
    if (p.fl & 1) {
      uint2 u; asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];\n" : "=r"(u.x), "=r"(u.y) : "r"(gsm + (rr * 64 + d4_ * 4) * 2));
      const __nv_bfloat16* bb = (const __nv_bfloat16*)&u;
#pragma unroll
      for (int i = 0; i < 4; i++) gv[i] = __bfloat162float(bb[i]);
    } else {
      const __nv_bfloat16* gt = row + 20 * D + rr * D + d4_ * 4;
#pragma unroll
      for (int i = 0; i < 4; i++) gv[i] = __bfloat162float(gt[i]);
    }
  };
  auto finish = [&](float4 a4, float Lr) {  // row r = q head, 16 threads (one half-warp) hold its 64 dims
    const float inv = p.vscale / Lr;
    a4.x *= inv; a4.y *= inv; a4.z *= inv; a4.w *= inv;
    float ss = a4.x * a4.x + a4.y * a4.y + a4.z * a4.z + a4.w * a4.w;
#pragma unroll
    for (int x = 1; x < 16; x <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, x);
    const float rn = rsqrtf(ss / D + 1e-5f);
    float gv[4]; gate4(r, dv, gv);
    const float vals[4] = {a4.x, a4.y, a4.z, a4.w};
    __nv_bfloat16 ob[4];
#pragma unroll
    for (int i = 0; i < 4; i++) ob[i] = __float2bfloat16(vals[i] * rn / (1.f + expf(-gv[i])));
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
  if (nsp == 1) { finish(acc, L); PT(5); return; }
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
      float gv[4]; gate4(hr, d4, gv);
      const float vals[4] = {a4.x, a4.y, a4.z, a4.w};
      __nv_bfloat16 ob[4];
#pragma unroll
      for (int i = 0; i < 4; i++) ob[i] = __float2bfloat16(vals[i] * rn / (1.f + expf(-gv[i])));
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
  const int nmg = nsp >= 16 ? 16 : (nsp >= 8 ? 8 : (nsp >= 4 ? 4 : (nsp >= 2 ? 2 : 1))), rank = rank_s - (nsp - nmg);
  if (rank < 0) return;
  if (tid == 0) { int v; do { asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(p.cnt + b) : "memory"); if (v < nsp) __nanosleep(32); } while (v < nsp); }
  __syncthreads();
  PT(7);  // the acquire spin + CTA barrier order the other splits' partials before the reads below (no fence.sc)
  asm volatile("cp.async.wait_all;\n" ::: "memory");
  float* xs = (float*)smem;  // [16 (rl, kg)][64] weighted O, [16] sums, [16] max
  const int d4 = tid & 15, kg = (tid >> 4) & (nmg - 1), rl = tid / (16 * nmg), slot = rl * nmg + kg;
  const int hr = rank + rl * nmg;
  const float* mlp = p.wml + ((size_t)b * p.NS * 16 + hr) * 2;
  float Mg = -INFINITY;
  float4 og = make_float4(0.f, 0.f, 0.f, 0.f);
  float lg = 0.f, Mr = -INFINITY;
  if (p.fl & 4) {  // one pass: running max per split group (one L2 round trip, one barrier fewer)
    mrg8(kg, nmg, nsp, [&](int k) { return __ldcg((const float2*)(mlp + (size_t)k * 32)); },
         [&](int k) { return __ldcg((const float4*)(p.wo + (((size_t)b * p.NS + k) * 16 + hr) * D) + d4); }, Mg, lg, og);
    PT(9);
    if (d4 == 0) xs[1024 + 16 + slot] = Mg;
    __syncthreads();
    for (int k = 0; k < nmg; k++) Mr = fmaxf(Mr, xs[1024 + 16 + rl * nmg + k]);
    const float c1 = Mg == -INFINITY ? 0.f : ex2(Mg - Mr);
    lg *= c1; og.x *= c1; og.y *= c1; og.z *= c1; og.w *= c1;
  } else {
  if (MU) {  // independent loads in flight together (the plain loops below wait out one L2 round trip per split)
#pragma unroll 8
    for (int k = kg; k < nsp; k += nmg) Mg = fmaxf(Mg, __ldcg(mlp + (size_t)k * 32));
  } else
  for (int k = kg; k < nsp; k += nmg) Mg = fmaxf(Mg, __ldcg(mlp + (size_t)k * 32));
  if (d4 == 0) xs[1024 + 16 + slot] = Mg;
  __syncthreads();
  PT(9);
  for (int k = 0; k < nmg; k++) Mr = fmaxf(Mr, xs[1024 + 16 + rl * nmg + k]);
  if (MU) {
#pragma unroll 8
  for (int k = kg; k < nsp; k += nmg) {
    const float2 mlv = __ldcg((const float2*)(mlp + (size_t)k * 32));
    const float4 v = __ldcg((const float4*)(p.wo + (((size_t)b * p.NS + k) * 16 + hr) * D) + d4);
    const float wt = mlv.x == -INFINITY ? 0.f : ex2(mlv.x - Mr);
    lg += wt * mlv.y;
    og.x += wt * v.x; og.y += wt * v.y; og.z += wt * v.z; og.w += wt * v.w;
  }  } else
  for (int k = kg; k < nsp; k += nmg) {
    const float2 mlv = __ldcg((const float2*)(mlp + (size_t)k * 32));
    const float4 v = __ldcg((const float4*)(p.wo + (((size_t)b * p.NS + k) * 16 + hr) * D) + d4);
    if (mlv.x == -INFINITY) continue;
    const float wt = ex2(mlv.x - Mr);
    lg += wt * mlv.y;
    og.x += wt * v.x; og.y += wt * v.y; og.z += wt * v.z; og.w += wt * v.w;
  }
  }
  PT(10);
  ((float4*)(xs + slot * 64))[d4] = og;
  if (d4 == 0) xs[1024 + slot] = lg;
  __syncthreads();
  PT(11);
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
    PT(12);
    float gv[4]; gate4(hr, d4, gv);
    const float vals[4] = {a4.x, a4.y, a4.z, a4.w};
    __nv_bfloat16 ob[4];
#pragma unroll
    for (int i = 0; i < 4; i++) ob[i] = __float2bfloat16(vals[i] * rn / (1.f + expf(-gv[i])));
    PT(13);
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
              double qk_factor, double sm_scale, double kv_scale, int64_t NS, torch::Tensor prof, torch::Tensor invf, int64_t mu, int64_t fl) {
  F p;
  p.qkv = (const __nv_bfloat16*)qkv.data_ptr(); p.loc = loc.data_ptr<int64_t>(); p.kc = (uint8_t*)kc.data_ptr(); p.vc = (uint8_t*)vc.data_ptr();
  p.pt = pt.data_ptr<int>(); p.seq = seq.data_ptr<int>(); p.wo = wo.data_ptr<float>(); p.wml = wml.data_ptr<float>(); p.cnt = cnt.data_ptr<int>();
  p.out = (__nv_bfloat16*)out.data_ptr(); p.B = seq.size(0); p.ps = ps; p.maxp = pt.size(1); p.window = window; p.NS = NS; p.fl = (int)fl;
  p.theta = theta; p.qsc = qk_factor; p.qk = (float)(sm_scale * 1.4426950408889634); p.kinv = 1.f / kv_scale; p.vinv = 1.f / kv_scale; p.vscale = kv_scale;
  p.prof = prof.numel() ? (unsigned long long*)prof.data_ptr() : nullptr; p.invf = invf.data_ptr<float>();
  const int sm = max(8 * RBF * 4096 + 256 + 8 * 2048 + 2048, (512 + 256 * D) * 4);  // + prefetched gate rows  // ring + table + page-table rings, or the merge scratch
  static bool once = false;
  if (!once) {
    cudaFuncSetAttribute(fdec<false, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm); cudaFuncSetAttribute(fdec<false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
    cudaFuncSetAttribute(fdec<true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
    cudaFuncSetAttribute(fdec<true, true>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    once = true;
  }
  static int major = 0; if (!major) { int dev; cudaGetDevice(&dev); cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev); }
  if (CLUS && !(fl & 8) && major >= 9 && NS > 1 && NS <= 16 && (NS & (NS - 1)) == 0 && p.B * NS <= 64) {  // clusters fragment GPCs: small grids only
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(p.B * NS); cfg.blockDim = dim3(256); cfg.dynamicSmemBytes = sm; cfg.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = NS; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
    cfg.attrs = at; cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, fdec<true, true>, p);
  } else if (fl & 8) {  // balanced: one CTA per resident slot
    static int capd = 0;
    if (!capd) { int dev, nsm, per; cudaGetDevice(&dev); cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
                 cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, fdec<false, true>, 256, sm); capd = nsm * per; }
    fdec<false, true><<<capd, 256, sm, at::cuda::getCurrentCUDAStream()>>>(p);
  } else if (mu) fdec<false, true><<<p.B * NS, 256, sm, at::cuda::getCurrentCUDAStream()>>>(p);
  else fdec<false, false><<<p.B * NS, 256, sm, at::cuda::getCurrentCUDAStream()>>>(p);
}

template <int V>
static void awfu(int ps, int g, int smw, cudaStream_t st, const A& a, const WF& f) {
  static bool o = false;
  if (!o) { cudaFuncSetAttribute(attnw<true, true, true, V>, cudaFuncAttributeMaxDynamicSharedMemorySize, smw); o = true; }
  if (ps == 1) attnw<true, true, true, V><<<g, 256, smw, st>>>(a, f); else attnw<false, true, true, 0><<<g, 256, smw, st>>>(a, f);
}
#ifndef AWV_ALL
#define AWV_ALL 0
#endif
void awfu_v(int v, int ps, int g, int smw, cudaStream_t st, const A& a, const WF& f) {
  switch (v) {
    case 8: awfu<8>(ps, g, smw, st, a, f); break;
    case 512: awfu<512>(ps, g, smw, st, a, f); break;
    case 520: awfu<520>(ps, g, smw, st, a, f); break;
#if AWV_ALL
    case 524: awfu<524>(ps, g, smw, st, a, f); break;
    case 1: awfu<1>(ps, g, smw, st, a, f); break;
    case 2: awfu<2>(ps, g, smw, st, a, f); break;
    case 3: awfu<3>(ps, g, smw, st, a, f); break;
    case 4: awfu<4>(ps, g, smw, st, a, f); break;
    case 5: awfu<5>(ps, g, smw, st, a, f); break;
    case 9: awfu<9>(ps, g, smw, st, a, f); break;
    case 10: awfu<10>(ps, g, smw, st, a, f); break;
    case 11: awfu<11>(ps, g, smw, st, a, f); break;
    case 12: awfu<12>(ps, g, smw, st, a, f); break;
    case 13: awfu<13>(ps, g, smw, st, a, f); break;
#endif
#ifdef AWV_ABL
    case 16: awfu<16>(ps, g, smw, st, a, f); break;
    case 32: awfu<32>(ps, g, smw, st, a, f); break;
    case 48: awfu<48>(ps, g, smw, st, a, f); break;
    case 64: awfu<64>(ps, g, smw, st, a, f); break;
    case 128: awfu<128>(ps, g, smw, st, a, f); break;
    case 192: awfu<192>(ps, g, smw, st, a, f); break;
    case 256: awfu<256>(ps, g, smw, st, a, f); break;
    case 240: awfu<240>(ps, g, smw, st, a, f); break;
    case 496: awfu<496>(ps, g, smw, st, a, f); break;
#endif
    default: awfu<0>(ps, g, smw, st, a, f);
  }
}

template <int X>
static void aw39f(int ps, int g, int smw, cudaStream_t st, const A& a, const WF& f) {
  const int smx = smw;
  static bool o = false;
  if (!o) { cudaFuncSetAttribute(attnw39<true, true, true, X>, cudaFuncAttributeMaxDynamicSharedMemorySize, smx); cudaFuncSetAttribute(attnw39<false, true, true, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, smw); o = true; }
  if (ps == 1) attnw39<true, true, true, X><<<g, 256, smx, st>>>(a, f); else attnw39<false, true, true, 0><<<g, 256, smw, st>>>(a, f);
}
#ifndef AWX_ALL
#define AWX_ALL 0
#endif
// x39 grafts on the v39 kernel: 4 fast softmax, 8 prologue overlap, 32 whole-row copies (CTA barrier), 64 L2::128B hint, 256 balanced
void aw39_x(int x, int ps, int g, int smw, cudaStream_t st, const A& a, const WF& f) {
  switch (x) {
    case 8: aw39f<8>(ps, g, smw, st, a, f); break;
    case 12: aw39f<12>(ps, g, smw, st, a, f); break;
    case 64: aw39f<64>(ps, g, smw, st, a, f); break;
    case 72: aw39f<72>(ps, g, smw, st, a, f); break;
    case 76: aw39f<76>(ps, g, smw, st, a, f); break;
    case 320: aw39f<320>(ps, g, smw, st, a, f); break;
    case 332: aw39f<332>(ps, g, smw, st, a, f); break;
#if AWX_ALL
    case 4: aw39f<4>(ps, g, smw, st, a, f); break;
    case 32: aw39f<32>(ps, g, smw, st, a, f); break;
    case 96: aw39f<96>(ps, g, smw, st, a, f); break;
    case 108: aw39f<108>(ps, g, smw, st, a, f); break;
    case 264: aw39f<264>(ps, g, smw, st, a, f); break;
#endif
    default: aw39f<0>(ps, g, smw, st, a, f);
  }
}

void run(torch::Tensor qkv, torch::Tensor pos, torch::Tensor loc, torch::Tensor kc, torch::Tensor vc, torch::Tensor ksc,
         torch::Tensor vsc, torch::Tensor pt, torch::Tensor seq, torch::Tensor qbuf, torch::Tensor wo, torch::Tensor wml,
         torch::Tensor out, int64_t hq, int64_t hkv, int64_t ps, int64_t window, double theta, double qk_factor,
         double sm_scale, double kv_scale, int64_t nsplit, torch::Tensor invf, torch::Tensor cnt, int64_t var, int64_t nmgcap, int64_t fumax, int64_t x39, torch::Tensor prof, int64_t mfl) {
  auto st = at::cuda::getCurrentCUDAStream();
  const int B = seq.size(0), toks = qkv.size(0) * qkv.size(1), L = toks / B;
  const bool V8 = L == 8 && hkv == 2 && hq == 16;
  const bool PF = L > 8 && hkv == 2 && hq == 16;  // prefill blocks: prep, then 8-token query blocks on wgmma
  static int major = 0; if (!major) { int dev; cudaGetDevice(&dev); cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev); }
  if (var == -3 && L == 1 && hkv == 2 && hq == 16 && ps == 1 && (major == 9 || WGEMU)) {  // decode on wgmma (attnd)
    const int nl = (major == 9 && !WGEMU) ? 2 : 1;
    A a;
    a.q = nullptr; a.q8 = nullptr; a.kc = (const uint8_t*)kc.data_ptr(); a.vc = (const uint8_t*)vc.data_ptr();
    a.pt = pt.data_ptr<int>(); a.seq = seq.data_ptr<int>(); a.wo = wo.data_ptr<float>(); a.wml = wml.data_ptr<float>();
    a.B = B; a.L = 1; a.hq = hq; a.hkv = hkv; a.ps = ps; a.maxp = pt.size(1); a.window = window; a.nsplit = nsplit; a.split = 0; a.rows = 8;
    a.qk = (float)(sm_scale * 1.4426950408889634 * qk_factor);
    WF f{(const __nv_bfloat16*)qkv.data_ptr(), loc.data_ptr<int64_t>(), invf.data_ptr<float>(), cnt.data_ptr<int>(), (__nv_bfloat16*)out.data_ptr(),
         (float)theta, (float)(1.0 / kv_scale), (float)(1.0 / kv_scale), (float)kv_scale, 2, prof.numel() ? (unsigned long long*)prof.data_ptr() : nullptr, (int)mfl};
    const int smd = nl * (DRS * 16384 + 32768 + 4096 + 2048 + 2048) + 256 + nl * 1024 + 1024;
    const int g = B * nsplit / nl;
    if (nl == 1) adf<1, 0>(g, smd, st, a, f);
    else if (x39 & 64) adf<2, 64>(g, smd, st, a, f);
    else adf<2, 0>(g, smd, st, a, f);
    return;
  }
  const bool Q8 = (V8 || PF) && (major == 9 || WGEMU) && WGV;
  const bool FU = Q8 && V8 && FUSEV && nsplit <= fumax;  // one launch: prologue, attention, merge + gate
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
      const int smw = RSW * 16384 + 8192 + (VDB ? 32768 : 16384) + 4096 + 1024 + 1024;
      static bool ow = false;
      if (!ow) {
        cudaFuncSetAttribute(attnw<false, false, false, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, smw); cudaFuncSetAttribute(attnw<true, false, false, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, smw);
        cudaFuncSetAttribute(attnw<true, true, true, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, smw); cudaFuncSetAttribute(attnw<false, true, true, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, smw);
        cudaFuncSetAttribute(attnw<false, false, true, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, smw);
        ow = true;
      }
      static int cap = 0;  // co-resident CTAs: many spinning mergers only when every CTA of the grid is resident
      if (!cap) { int dev, nsm, per; cudaGetDevice(&dev); cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
                  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, attnw<true, true, true, 0>, 256, smw); cap = nsm * per; }
      const int grid = B * ((L + 7) / 8) * nsplit;
      const int smx = RSW * 16384 + 8192 + 32768 + 1024 + 8 * (2 * RSW + 2) + 1024;
      int capl = cap;
      if (var >= 4096) {
        static int capx = 0;
        if (!capx) {
          cudaFuncSetAttribute(attnx<true, true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smx); cudaFuncSetAttribute(attnx<true, false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smx);
          cudaFuncSetAttribute(attnx<true, true, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smx); cudaFuncSetAttribute(attnx<true, false, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smx);
          cudaFuncSetAttribute(attnx<false, false, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smx);
          int dev, nsm, per; cudaGetDevice(&dev); cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
          cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, attnx<true, true, true>, 288, smx); capx = nsm * per;
        }
        capl = min(cap, capx);
      }
      const int nmg = min(grid <= capl ? (int)nmgcap : 8, 1 << (31 - __builtin_clz((unsigned)nsplit)));
      WF f{(const __nv_bfloat16*)qkv.data_ptr(), loc.data_ptr<int64_t>(), invf.data_ptr<float>(), cnt.data_ptr<int>(), (__nv_bfloat16*)out.data_ptr(),
           (float)theta, (float)(1.0 / kv_scale), (float)(1.0 / kv_scale), (float)kv_scale, nmg, prof.numel() ? (unsigned long long*)prof.data_ptr() : nullptr, (int)mfl};
      if (var >= 4096 && FU) {  // warp-specialized: producer warp gathers + converts V, consumer warpgroups do wgmma + softmax
        if (ps != 1) attnx<false, false, false><<<B * nsplit, 288, smx, st>>>(a, f);
        else if (var & 8) { if (var & 4) attnx<true, true, true><<<B * nsplit, 288, smx, st>>>(a, f); else attnx<true, false, true><<<B * nsplit, 288, smx, st>>>(a, f); }
        else if (var & 4) attnx<true, true, false><<<B * nsplit, 288, smx, st>>>(a, f);
        else attnx<true, false, false><<<B * nsplit, 288, smx, st>>>(a, f);
        return;
      }
      if (var == -2 && FU && ps == 1 && L == 8) {  // warp-specialized: producer warpgroup + 2 lanes of consumer warpgroups
        const int nl = (major == 9 && !WGEMU) ? 2 : 1;
        const int smp = nl * (PRS * 16384 + 8192 + 32768 + 2048 + ((x39 & 128) ? 2048 : 0)) + 1024 + 256 + 1024;
        static int nsm = 0; if (!nsm) { int dev; cudaGetDevice(&dev); cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev); }
        ap_x(nl, (int)x39, (x39 & 256) ? nsm : B * nsplit / nl, smp, st, a, f);  // balanced: one CTA per SM
        return;
      }
      if (var < 0) {  // v39 kernel, grafts x39
        if (FU) { aw39_x((int)x39, (int)ps, ((x39 & 256) && ps == 1) ? cap : B * nsplit, smw, st, a, f); return; }  // balanced: one CTA per resident slot
        static bool o39 = false;
        if (!o39) {
          cudaFuncSetAttribute(attnw39<true, false, false, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, smw); cudaFuncSetAttribute(attnw39<false, false, false, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, smw);
          cudaFuncSetAttribute(attnw39<true, false, true, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, smw); cudaFuncSetAttribute(attnw39<false, false, true, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, smw);
          o39 = true;
        }
        if (PF) {
          const int g = B * ((L + 7) / 8) * nsplit;
          if (ps == 1) attnw39<true, false, true, 0><<<g, 256, smw, st>>>(a, f); else attnw39<false, false, true, 0><<<g, 256, smw, st>>>(a, f);
          return;
        }
        if (ps == 1) attnw39<true, false, false, 0><<<B * nsplit, 256, smw, st>>>(a, f); else attnw39<false, false, false, 0><<<B * nsplit, 256, smw, st>>>(a, f);
      } else {
      if (FU) { awfu_v((int)var, (int)ps, B * nsplit, smw, st, a, f); return; }
      if (PF) {
        const int g = B * ((L + 7) / 8) * nsplit;
        attnw<false, false, true, 0><<<g, 256, smw, st>>>(a, f);
        return;
      }
      if (ps == 1) attnw<true, false, false, 0><<<B * nsplit, 256, smw, st>>>(a, f); else attnw<false, false, false, 0><<<B * nsplit, 256, smw, st>>>(a, f);
      }
    } else if (ps == 1) attn9<true><<<B * ((L + 7) / 8) * nsplit, 128, sm8, st>>>(a); else attn9<false><<<B * ((L + 7) / 8) * nsplit, 128, sm8, st>>>(a);
  } else if (mt <= 1) attn<1><<<blocks, 256, sm, st>>>(a);
  else if (mt <= 2) attn<2><<<blocks, 256, sm, st>>>(a);
  else attn<4><<<blocks, 256, sm, st>>>(a);
  if (nsplit > 32) comb2<<<B * L * hq, 1024, 0, st>>>(a, (const __nv_bfloat16*)qkv.data_ptr(), (__nv_bfloat16*)out.data_ptr(), (float)kv_scale);
  else comb<<<(B * L * hq + 7) / 8, 256, 0, st>>>(a, (const __nv_bfloat16*)qkv.data_ptr(), (__nv_bfloat16*)out.data_ptr(), (float)kv_scale);
}
"""

_mod = None
_DBG = []
_NOPROF = None
_INV = {}


def _invf(theta, dev):
    """RoPE inverse frequencies exactly as the reference builds them (fp32), cached per theta"""
    k = (float(theta), str(dev))
    if k not in _INV:
        t = theta if theta else 1.0
        _INV[k] = (1.0 / (t ** (torch.arange(0, 64, 2, device=dev).float() / 64))).contiguous()
    return _INV[k]
_BUF = {}
_NSDIV = 128
# verify (L = 8) per shape (B, sliding): attnw variant bits (1 copy, 2 QK-first overlap, 4 fast softmax, 8 prologue overlap),
# keys per split, merger cap, largest split count that runs fused (one launch); var -1 = the v39 kernel; v39 = (-1, 256, 8, 32)
# x39 (var -1 only): grafts on the v39 kernel: 4 softmax fast path, 8 prologue overlap
# mfl & 1: split merge loads 8 splits at a time (chunked one pass) instead of one guarded L2 round trip per split
# var >= 4096: warp-specialized kernel (attnx; var & 4 fast softmax, var & 8 producer converts V; mfl & 2 seq-independent splits on full layers)
VDEF = (-1, 256, 8, 32, 0, 1)
# v48 x39 bits: 64 L2::128B prefetch hint on the K/V row copies (each warpgroup copies only its head's 64 B half of a 128 B
# row, so without it DRAM sees 64 B gathers), 256 full-layer splits balanced over the batch's total keys (grid = resident CTAs)
VCFG = {(1, True): (-1, 128, 8, 1 << 30, 8, 1), (1, False): (-2, 304, 128, 1 << 30, 68, 1),  # v1f: attnp, 264 lanes, 128 grouped mergers  # (new-kernel V 7/14/15 trip ptxas 12.9 here)
        (16, True): (-2, 128, 16, 1 << 30, 32836, 1), (32, True): (-2, 256, 8, 1 << 30, 32836, 1),  # v16s: 16 grouped mergers  # x 32768: early query loads
        (16, False): (-2, 256, 8, 1 << 30, 324, 1), (32, False): (-2, 256, 8, 1 << 30, 324, 1)}
# v51: x39 256 on attnp = lanes balanced over the batch's total keys on all SMs (full layers), 128 = producer converts V
# var -2: attnp (warp-specialized: producer warpgroup + 2 split lanes x 2 consumer warpgroups per CTA, setmaxnreg 32 / 112),
# x39 bits there: 4 fast softmax, 64 L2::128B hint
DCFG = {}  # decode (B, sliding) -> unrolled cross-CTA merge (default 1)
# decode (B, sliding) -> fdec flags: 1 gate rows prefetched at kernel start, 2 seq-independent splits (full layers),
# 4 one-pass cross-CTA merge
DFL = {}
DFLDEF = {True: 5, False: 5}
DACFG = {(16, True): (64, 64), (32, True): (64, 64)}  # decode (B, sliding) -> (keys per split, x39 (64: L2::128B hint)) for attnd; absent / DAOVR = () -> fdec
DAOVR = None
DFLOVR = None
_VPROF = None


def _DFL(B, sliding):
    return DFLOVR if DFLOVR is not None else DFL.get((B, sliding), DFLDEF[sliding] | (2 if (B == 1 and not sliding) else 0))
VOVR = None


def _vcfg(B, sliding):
    return VOVR if VOVR is not None else VCFG.get((B, sliding), VDEF)
OVR = {}  # (kind, B, sliding) -> split count override (tuning)


def _big():
    pr = torch.cuda.get_device_properties(0)
    sm = getattr(pr, "shared_memory_per_block_optin", None)
    return sm >= 180000 if sm else (pr.major, pr.minor) in ((9, 0), (10, 0))


def _hopper():
    return torch.cuda.get_device_capability() == (9, 0)


def _gencode():
    """sm_90a (wgmma) on Hopper; elsewhere torch's own arch flags"""
    return ["-gencode=arch=compute_90a,code=sm_90a"] if torch.cuda.get_device_capability() == (9, 0) else []


def _ext():
    global _mod
    if _mod is None:
        rb, rb2, rb3, rbf = (8, 6, 20, 5) if _big() else (4, 3, 8, 2)
        _mod = load_inline("attn_v59d_%d" % rb + "".join(_DBG).replace("-", "_").replace("=", ""),
                           cpp_sources="void gt_res(torch::Tensor); void prep_only(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,double,double,torch::Tensor); void run_fdec(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,double,double,double,double,int64_t,torch::Tensor,torch::Tensor,int64_t,int64_t); void run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t,double,double,double,double,int64_t,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t,torch::Tensor,int64_t);",
                           cuda_sources=CUDA, functions=["run", "run_fdec", "prep_only", "gt_res"], extra_cuda_cflags=["-O3", "-lineinfo", "-DRB=%d" % rb, "-DRB2=%d" % rb2, "-DRB3=%d" % rb3, "-DRBF=%d" % rbf] + _DBG + _gencode(), verbose=False)
    return _mod


DIAG = False


def _diag(dev):
    """timeline of the fused decode kernel on synthetic batch-1 inputs (sliding and full), printed once"""
    global _mod, _DBG, _NOPROF
    keep = (_mod, _DBG)
    _mod, _DBG = None, _DBG + ["-DPROF"]
    for window, theta, ctx in ((2048, 500000.0, 80000), (0, 0.0, 80000), (2048, 500000.0, 1000)):
        for B in (1, 32):
            mp = ctx + 8
            pt = torch.randperm(B * mp, device=dev, dtype=torch.int64).to(torch.int32).view(B, mp)
            kc = torch.randn((B * mp, 1, 2, 64), device=dev).to(torch.float8_e4m3fn); vc = kc.clone()
            seq = torch.full((B,), ctx, dtype=torch.int32, device=dev)
            loc = pt[:, ctx].long()
            qkv = torch.randn((B, 1, 2304), device=dev).bfloat16()
            span = min(mp, window + 1) if window else mp
            NS = max(1, min(132 // B, (span + _NSDIV - 1) // _NSDIV))
            if NS <= 17 and B * min(NS, 16) <= 64: NS = 1 << (min(NS, 16).bit_length() - 1)
            wo = torch.empty((B * NS * 16 * 64,), device=dev); wml = torch.empty((B * NS * 32,), device=dev)
            cnt = torch.zeros((2 * B,), dtype=torch.int32, device=dev); out = torch.empty((B, 1, 1024), dtype=torch.bfloat16, device=dev)
            prof = torch.zeros((B * NS * 8 * 16,), dtype=torch.int64, device=dev)
            fl = torch.ones(2 ** 26, device=dev)
            for it in range(3):
                fl.sum()
                _ext().run_fdec(qkv, loc, kc, vc, pt, seq, wo, wml, cnt, out, 1, window, theta, 5.47, 0.125, 1.0, NS, prof, _invf(theta, dev), 1, _DFL(B, bool(window)))
            torch.cuda.synchronize()
            P = prof.view(-1, 16).cpu().tolist()
            t0 = min(r[0] for r in P)
            col = lambda k: [(r[k] - t0) / 1e3 for r in P if r[k]]
            mx = lambda k: max(col(k)) if col(k) else -1
            print("FDIAG win %d B %d NS %d fl %d: start max %.1f issued %.1f prolog %.1f sync %.1f loop-end p50 %.1f max %.1f cta-merge %.1f ranked %.1f all-in %.1f "
                  "pass1 %.1f pass2 %.1f sync2 %.1f norm %.1f gate %.1f last-end %.1f" % (
                window, B, NS, _DFL(B, bool(window)), mx(0), mx(8), mx(1), mx(2), sorted(col(3))[len(col(3)) // 2], mx(3), mx(4), mx(5), mx(7),
                mx(9), mx(10), mx(11), mx(12), mx(13), mx(6)), flush=True)
    _mod, _DBG = keep


FA3PF = True
_FA3 = []


def _fa3():
    if not _FA3:
        try:
            from sgl_kernel.flash_attn import flash_attn_with_kvcache
            _FA3.append(flash_attn_with_kvcache)
        except Exception:
            _FA3.append(None)
    return _FA3[0]


def _prefill_fa3(data, cfg, B, L, hq, hkv, qkv, dev):
    q8 = torch.empty((B * L, hq, 64), dtype=torch.uint8, device=dev)
    _ext().prep_only(qkv, data["positions"], data["cache_loc"], data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"], q8,
                     hq, hkv, cfg["rope_theta"], cfg["qk_scale_factor"], _invf(cfg["rope_theta"], dev))
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
    o = _fa3()(q=q8.view(torch.float8_e4m3fn), k_cache=data["k_cache"], v_cache=data["v_cache"], page_table=data["page_table"],
               cache_seqlens=csl, cu_seqlens_q=cuq, max_seqlen_q=L, softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"], causal=True,
               window_size=(win - 1, 0) if win else (-1, -1), scheduler_metadata=data.get("scheduler_metadata"), **desc)
    if isinstance(o, tuple):
        o = o[0]
    x = o.float().view(B * L, hq, 64)
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
    g = qkv.reshape(B * L, -1)[:, (hq + 2 * hkv) * 64:].float().view(B * L, hq, 64)
    return (x * torch.sigmoid(g)).to(torch.bfloat16).view(B, L, hq * 64)


def _kernel_fn(data):
    global DIAG
    if DIAG:
        DIAG = False
        _diag(data["qkv"].device)
    cfg = data["config"]
    B, L, hq, hkv = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"]
    qkv = data["qkv"]
    dev = qkv.device
    dac = (DAOVR if DAOVR is not None else DACFG.get((B, bool(cfg["window"])))) if L == 1 else None
    if dac and _hopper() and hq == 16 and hkv == 2 and cfg["page_size"] == 1:  # decode on wgmma (attnd, sm_90 only): keys as M
        kps, x39 = dac
        maxkeys = data["page_table"].shape[1]
        span = min(maxkeys, cfg["window"] + 1) if cfg["window"] else maxkeys
        nl = 2 if (_hopper() and torch.cuda.get_device_capability() == (9, 0)) else 1
        sms = torch.cuda.get_device_properties(0).multi_processor_count
        nsplit = max(nl, min((span + kps - 1) // kps, nl * (sms // B)))
        nsplit -= nsplit % nl
        key = ("ad", B, nsplit)
        if key not in _BUF:
            _BUF[key] = (torch.empty((B * nsplit * 16 * 64,), dtype=torch.float32, device=dev),
                         torch.empty((B * nsplit * 16 * 2,), dtype=torch.float32, device=dev),
                         torch.zeros((2 * B,), dtype=torch.int32, device=dev), torch.empty(0, device=dev))
        wo, wml, cnt, qb = _BUF[key]
        out = torch.empty((B, 1, hq * 64), dtype=torch.bfloat16, device=dev)
        ks = float(cfg["kv_scale"])
        _ext().run(qkv, data["positions"], data["cache_loc"], data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"],
                   data["page_table"], data["seq_lens"], qb, wo, wml, out, hq, hkv, 1, cfg["window"], cfg["rope_theta"],
                   cfg["qk_scale_factor"], cfg["sm_scale"] * ks, ks, nsplit, _invf(cfg["rope_theta"], dev), cnt, -3, 2, 1 << 30, x39,
                   _VPROF if _VPROF is not None else torch.empty(0, device=dev), 1)
        return out
    if L == 1 and hq == 16 and hkv == 2 and cfg["head_dim"] == 64:  # fused single-launch decode
        maxkeys = data["page_table"].shape[1] * cfg["page_size"]
        span = min(maxkeys, cfg["window"] + 1) if cfg["window"] else maxkeys
        NS = max(1, min(132 // B, (span + _NSDIV - 1) // _NSDIV))  # keys per CTA >= _NSDIV, one wave
        if _hopper() and NS <= 17 and B * min(NS, 16) <= 64:  # one cluster per request: split merge in distributed smem
            NS = 1 << (min(NS, 16).bit_length() - 1)
        NS = OVR.get(("d", B, bool(cfg["window"])), NS)
        if _DFL(B, bool(cfg["window"])) & 8:  # balanced splits: NS = most splits one request may get (buffer stride)
            NS = min(64, 2 * ((132 + B - 1) // B))
        key = ("f", B, NS)
        if key not in _BUF:
            _BUF[key] = (torch.empty((B * NS * 16 * 64,), dtype=torch.float32, device=dev),
                         torch.empty((B * NS * 16 * 2,), dtype=torch.float32, device=dev),
                         torch.zeros((2 * B,), dtype=torch.int32, device=dev))
        wo, wml, cnt = _BUF[key]
        out = torch.empty((B, 1, hq * 64), dtype=torch.bfloat16, device=dev)
        ks = float(cfg["kv_scale"])
        _ext().run_fdec(qkv, data["cache_loc"], data["k_cache"], data["v_cache"], data["page_table"], data["seq_lens"], wo, wml, cnt,
                        out, cfg["page_size"], cfg["window"], cfg["rope_theta"], cfg["qk_scale_factor"], cfg["sm_scale"] * ks, ks, NS,
                        torch.empty(0, device=dev), _invf(cfg["rope_theta"], dev), DCFG.get((B, bool(cfg["window"])), 1), _DFL(B, bool(cfg["window"])))
        return out
    if L > 8 and FA3PF and _fa3() is not None and _hopper():  # prefill: our prologue (norm, RoPE, cache store), FA3, our norm + gate
        return _prefill_fa3(data, cfg, B, L, hq, hkv, qkv, dev)
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
        var, kps, nmgcap, fumax, x39, mfl = _vcfg(B, bool(cfg["window"]))
        nsplit = max(1, min((span + 127) // 128, (132 * 3) // B)) if not _hopper() else max(1, min((span + kps - 1) // kps, (132 * 2) // B))
        if var == -2:  # warp-specialized kernel: 2 splits per CTA on Hopper, one CTA per SM (spinning mergers stay co-resident)
            _nl = 2 if torch.cuda.get_device_capability() == (9, 0) else 1  # lanes per CTA (C++: 2 on sm_90, 1 under emulation)
            _sm = torch.cuda.get_device_properties(0).multi_processor_count  # one CTA per SM: spinning mergers stay co-resident
            nsplit = max(_nl, min(nsplit + (nsplit & 1) if _nl == 2 else nsplit, _nl * (_sm // B)))
            if x39 & 256:  # balanced lanes over all SMs: nsplit = most splits one request may get (buffer stride)
                nsplit = min(64, 2 * ((264 + B - 1) // B))
        if x39 & 256:  # balanced splits (full layers, B <= 32): nsplit = most splits one request may get (buffer stride)
            nsplit = min(64, 2 * ((132 * 2 + B - 1) // B))
    else:
        nsplit = max(1, min((span + 127) // 128, (132 * 8 + B * hkv - 1) // (B * hkv)))
    if L != 8 or hkv != 2 or hq != 16:
        var, nmgcap, fumax, x39, mfl = -1, 8, 32, 0, 1
    nsplit = OVR.get(("v" if L == 8 else "o", B, bool(cfg["window"])), nsplit)
    rows = L * (hq // hkv)
    nqb = (L + 7) // 8
    key = (B, L, hq, hkv, nsplit)
    if key not in _BUF:
        _BUF[key] = (torch.empty((B * L * hq * 64,), dtype=torch.float16, device=dev),
                     torch.empty((B * nqb * hkv * nsplit * 64 * 64,), dtype=torch.float32, device=dev),
                     torch.empty((B * nqb * hkv * nsplit * 64 * 2,), dtype=torch.float32, device=dev),
                     torch.zeros((2 * B * nqb,), dtype=torch.int32, device=dev))
    qbuf, wo, wml, cnt = _BUF[key]
    out = torch.empty((B, L, hq * 64), dtype=torch.bfloat16, device=dev)
    ks = float(cfg["kv_scale"])
    # k_scale enters the logits: fold it into the query scale (qk_scale_factor already in q)
    _ext().run(qkv, data["positions"], data["cache_loc"], data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"],
               data["page_table"], data["seq_lens"], qbuf, wo, wml, out, hq, hkv, ps, cfg["window"], cfg["rope_theta"],
               cfg["qk_scale_factor"], cfg["sm_scale"] * ks, ks, nsplit, _invf(cfg["rope_theta"], dev), cnt, var, nmgcap, fumax, x39, _VPROF if _VPROF is not None else torch.empty(0, device=dev), mfl)
    return out



# ---- diag only (never submitted): H200 verify timelines (launch helpers now have internal linkage, so the PROF module
# gets its own smem attributes), per-step loop cost, v59: narrowed balanced-split search, consumer setmaxnreg after the query prologue, v16s 16 mergers; no-setmaxnreg (x 65536), closed-form split map (x 131072) and early consumer setmaxnreg.inc (x 262144) A/Bs; phase clocks
_DDONE = set()
_GTR = []
_GFL = []


def _gt(fn, reps=20):
    if not _GFL:
        _GFL.append(torch.ones(64 << 20, dtype=torch.float32, device="cuda"))
    s_ = torch.cuda.Stream(); s_.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s_):
        for _ in range(2): fn()
    torch.cuda.current_stream().wait_stream(s_); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    ts = []
    for _ in range(reps + 3):
        _GFL[0].sum()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record(); g.replay(); e1.record(); e1.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3)
    ts = sorted(ts[3:])
    return ts[len(ts) // 2]


def _vtimeline(data, tag, warm=False):
    global _mod, _DBG, _VPROF
    keep = (_mod, _DBG)
    _mod, _DBG = None, _DBG + ["-DPROF"]
    try:
        _VPROF = torch.zeros((65536 + 16384,), dtype=torch.int64, device="cuda")
        for it_ in range(3):
            _VPROF.zero_()
            if _GFL and not warm: _GFL[0].sum()  # warm: code, params and page table stay in L2 from the previous call
            _kernel_fn(data)
        torch.cuda.synchronize()
        V = _VPROF.cpu()
        P = [r for r in V[:65536].view(-1, 8).tolist() if r[0]]
        W = [r for r in V[65536:65536 + 2048].view(-1, 4).tolist() if r[2]]
        Z = [r for r in V[65536 + 2048:65536 + 4096].view(-1, 8).tolist() if r[0] or r[4]]
        Y = V[65536 + 4096:].view(-1, 2, 8).tolist()
        t0 = min(r[0] for r in P)
        out, med = [], {}
        for k_, nm in enumerate(["start", "q", "prolog", "loop", "parts", "ranked", "all-in", "end"]):
            c = sorted((r[k_] - t0) / 1e3 for r in P if r[k_])
            if c:
                out.append("%s %.1f/%.1f" % (nm, c[len(c) // 2], c[-1])); med[nm] = c[len(c) // 2]
        cfg = data["config"]
        keys = (data["seq_lens"].long() + cfg["q_len"]).clamp(max=(cfg["window"] + cfg["q_len"]) if cfg["window"] else 1 << 40).sum().item()
        nct = sum(1 for r in P if r[3])
        if nct and "loop" in med and "prolog" in med:
            st = keys / 64.0 / nct
            out.append("| %d CTAs %.1f steps/CTA %.0f ns/step" % (nct, st, (med["loop"] - med["prolog"]) * 1e3 / max(st, 1)))
        if W:
            q = lambda k, p: sorted(r[k] for r in W)[min(len(W) - 1, int(p * len(W)))] / 1e3
            out.append("| lanes %d cons-wait-full %.1f/%.1f prod-wait-empty %.1f/%.1f cons-loop %.1f/%.1f prod-loop %.1f/%.1f us (p50/max)" % (
                len(W), q(0, .5), q(0, 1), q(1, .5), q(1, 1), q(2, .5), q(2, 1), q(3, .5), q(3, 1)))
        if Z:
            def zq(k):
                c = sorted((r[k] - t0) / 1e3 for r in Z if r[k])
                return "%.1f/%.1f" % (c[len(c) // 2], c[-1]) if c else "-"
            out.append("| prod start %s seq %s pt %s kv0 %s cons start %s q %s own %s data0 %s (p50/max)" % tuple(zq(k) for k in range(8)))
        for role, nm in ((0, "cons"), (1, "prod")):
            R = [y[role] for y in Y if y[role][7] == 1]
            if R:
                md = lambda f: sorted(f(r) for r in R)[len(R) // 2]
                out.append("| %s entry %.2f pre-bar %.2f bar %.2f setmaxnreg %.2f us; clk %d / %d / %d" % (
                    nm, md(lambda r: (r[0] - t0) / 1e3), md(lambda r: (r[1] - r[0]) / 1e3), md(lambda r: (r[2] - r[1]) / 1e3),
                    md(lambda r: (r[3] - r[2]) / 1e3), md(lambda r: r[4]), md(lambda r: r[5]), md(lambda r: r[6])))
        print("VDIAG %s%s: %s" % (tag, " warm" if warm else "", " ".join(out)), flush=True)
    except Exception as e:
        print("VDIAG %s ERR %s" % (tag, str(e)[:80]), flush=True)
    _VPROF = None
    _mod, _DBG = keep


def _dsweep(data):
    global DFLOVR, VOVR
    cfg = data["config"]
    B, L, ps = cfg["batch"], cfg["q_len"], cfg["page_size"]
    mk = data["page_table"].shape[1] * ps
    if ps != 1 or cfg["num_q_heads"] != 16 or not (80000 <= mk <= 86100) or not _hopper(): return
    sl = bool(cfg["window"])
    key = (B, L, sl)
    if key in _DDONE: return
    _DDONE.add(key)
    tag = "%s%d%s" % ("d" if L == 1 else "v", B, "s" if sl else "f")

    def run(lbl):
        try:
            print("WD %s %s %.2f" % (tag, lbl, _gt(lambda: _kernel_fn(data))), flush=True)
        except Exception as e:
            print("WD %s %s ERR %s" % (tag, lbl, str(e)[:60]), flush=True)

    if L == 1:
        return
    if not _GTR:
        try:
            o = torch.zeros(32, dtype=torch.int64, device="cuda"); _ext().gt_res(o); torch.cuda.synchronize()
            print("GTRES step ns / clk: " + " ".join("%d/%d" % (o[2 * k].item(), o[2 * k + 1].item()) for k in range(16)), flush=True)
        except Exception as e:
            print("GTRES ERR %s" % str(e)[:80], flush=True)
        _GTR.append(1)
    big = 1 << 30
    d = _vcfg(B, sl)
    run("DEF%s" % (d,))
    vs = []
    if B == 1 and not sl:
        vs = [(-2, 304, 128, big, 65604, 1), (-2, 304, 128, big, 262212, 1)]
    elif B == 16 and sl:
        vs = [(-2, 128, 16, big, 98372, 1), (-2, 128, 16, big, 294980, 1), (-2, 128, 8, big, 32836, 1)]
    elif B == 32 and sl:
        vs = [(-2, 256, 8, big, 98372, 1)]
    elif B > 1:
        vs = [(-2, 256, 8, big, 131396, 1)]
    for c in vs:
        VOVR = c; run("cfg%s" % (c,))
    VOVR = None
    if B == 1 and sl:
        return
    _vtimeline(data, "%s DEF" % tag)
    for c in vs[:2]:
        VOVR = c; _vtimeline(data, "%s %s" % (tag, c)); VOVR = None


def kernel_fn(data):
    if not torch.cuda.is_current_stream_capturing():
        _dsweep(data)
    return _kernel_fn(data)
