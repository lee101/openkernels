
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#define D 64
#define LOG2E 1.4426950408889634f

// ---------------------------------------------------------------- prep: one CTA (64 x Hq+2Hkv threads) per new token
__global__ void prep(const __nv_bfloat16* qkv, const int64_t* pos, const int64_t* loc, uint8_t* kc, uint8_t* vc,
                     const float* ksc, const float* vsc, __half* qo, int hq, int hkv, float theta, float qsc) {
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
    const int i = d >> 1;
    const float inv = 1.0f / powf(theta, (float)(2 * i) / D);
    const float ang = (float)pos[tok] * inv;
    float sn, cs; sincosf(ang, &sn, &cs);
    const float other = __shfl_xor_sync(0xffffffffu, x, 1);
    x = (d & 1) ? other * sn + x * cs : x * cs - other * sn;
  }
  if (isq) qo[((size_t)tok * hq + hd) * D + d] = __float2half_rn(x);
  else {
    const int h = hd - hq;
    __nv_fp8_e4m3 f(x / *ksc);
    kc[(size_t)loc[tok] * hkv * D + h * D + d] = f.__x;
  }
}

// ---------------------------------------------------------------- attention: one warp per (request, kv head, split)
__device__ __forceinline__ void mma16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t e2h(uint32_t two) {  // 2 x e4m3 (low 16 bits) -> f16x2
  __half2_raw r = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(two & 0xffff), __NV_E4M3);
  return *(uint32_t*)&r;
}
__device__ __forceinline__ uint32_t pk(float a, float b) { __half2 h = __floats2half2_rn(a, b); return *(uint32_t*)&h; }
__device__ __forceinline__ uint4 ldg16(const void* p) { return __ldg((const uint4*)p); }
__device__ __forceinline__ uint2 ldg8(const void* p) { return __ldg((const uint2*)p); }

struct A {
  const __half* q; const uint8_t* kc; const uint8_t* vc; const int* pt; const int* seq; float* wo; float* wml;
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
        const float sc = nm == -INFINITY ? 1.f : exp2f(mx[m][hh] - nm);
        mx[m][hh] = nm;
        float p0 = nm == -INFINITY ? 0.f : exp2f(s[m][0][hh * 2] - nm), p1 = nm == -INFINITY ? 0.f : exp2f(s[m][0][hh * 2 + 1] - nm);
        float p2 = nm == -INFINITY ? 0.f : exp2f(s[m][1][hh * 2] - nm), p3 = nm == -INFINITY ? 0.f : exp2f(s[m][1][hh * 2 + 1] - nm);
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
      const float nm = fmaxf(mx[h], bm), sc = exp2f(mx[h] - nm);  // nm finite: every block has a valid key
      mx[h] = nm;
      const float p0 = exp2f(v[0] - nm), p1 = exp2f(v[1] - nm), p2 = exp2f(v[2] - nm), p3 = exp2f(v[3] - nm);
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
#define RB3 8
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
  const int nb = (ke - ks + 15) >> 4;
  int rnext = rowof(ks + (RB3 - 1) * 16);
#pragma unroll
  for (int i = 0; i < RB3 - 1; i++) issue(i, rowof(ks + i * 16));
  for (int bi = 0, kb = ks; bi < nb; bi++, kb += 16) {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(RB3 - 2));
    __syncthreads();  // block bi visible to all warps; slot (bi - 1) % RB3 free for refill
    const int r2 = rowof(kb + RB3 * 16);
    issue((bi + RB3 - 1) % RB3, rnext);
    rnext = r2;
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
      const float sc = nm == -INFINITY ? 1.f : exp2f(mx[hh] - nm);
      mx[hh] = nm;
      float pp[4];
#pragma unroll
      for (int i = 0; i < 4; i++) pp[i] = nm == -INFINITY ? 0.f : exp2f(v[i] - nm);
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

// ---------------------------------------------------------------- combine v2: one CTA (256 threads) per (token, q head)
__global__ void __launch_bounds__(256) comb2(A a, const __nv_bfloat16* qkv, __nv_bfloat16* out, float vscale) {
  const int tid = threadIdx.x, wi = blockIdx.x;
  const int hd = wi % a.hq, tl = wi / a.hq, l = tl % a.L, b = tl / a.L;
  const int grp = a.hq / a.hkv, h = hd / grp, r = l * grp + hd % grp;
  const size_t base = ((size_t)b * a.hkv + h) * a.nsplit;
  __shared__ float red[256];
  __shared__ float4 vec[16][16];
  float M = -INFINITY;
  for (int s = tid; s < a.nsplit; s += 256) M = fmaxf(M, a.wml[((base + s) * a.rows + r) * 2]);
  red[tid] = M; __syncthreads();
  for (int x = 128; x; x >>= 1) { if (tid < x) red[tid] = fmaxf(red[tid], red[tid + x]); __syncthreads(); }
  M = red[0]; __syncthreads();
  const int sg = tid >> 4, dv = tid & 15;  // 16 split groups x 16 float4 of the 64 dims
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f); float Ls = 0.f;
  for (int s = sg; s < a.nsplit; s += 16) {
    const float* ml = a.wml + ((base + s) * a.rows + r) * 2;
    const float m = ml[0];
    if (m == -INFINITY) continue;
    const float w = exp2f(m - M);
    if (dv == 0) Ls += w * ml[1];
    const float4 v = ((const float4*)(a.wo + ((base + s) * a.rows + r) * D))[dv];
    acc.x += w * v.x; acc.y += w * v.y; acc.z += w * v.z; acc.w += w * v.w;
  }
  vec[sg][dv] = acc;
  red[tid] = Ls;
  __syncthreads();
  if (tid < 16) {
    float4 v = vec[0][tid];
    for (int k = 1; k < 16; k++) { const float4 u = vec[k][tid]; v.x += u.x; v.y += u.y; v.z += u.z; v.w += u.w; }
    vec[0][tid] = v;
  }
  if (tid == 0) { float L2 = 0.f; for (int k = 0; k < 16; k++) L2 += red[k * 16]; red[0] = L2; }
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

void run(torch::Tensor qkv, torch::Tensor pos, torch::Tensor loc, torch::Tensor kc, torch::Tensor vc, torch::Tensor ksc,
         torch::Tensor vsc, torch::Tensor pt, torch::Tensor seq, torch::Tensor qbuf, torch::Tensor wo, torch::Tensor wml,
         torch::Tensor out, int64_t hq, int64_t hkv, int64_t ps, int64_t window, double theta, double qk_factor,
         double sm_scale, double kv_scale, int64_t nsplit) {
  auto st = at::cuda::getCurrentCUDAStream();
  const int B = seq.size(0), toks = qkv.size(0) * qkv.size(1), L = toks / B;
  prep<<<dim3(toks, hq + 2 * hkv), D, 0, st>>>((const __nv_bfloat16*)qkv.data_ptr(), pos.data_ptr<int64_t>(), loc.data_ptr<int64_t>(),
      (uint8_t*)kc.data_ptr(), (uint8_t*)vc.data_ptr(), ksc.data_ptr<float>(), vsc.data_ptr<float>(), (__half*)qbuf.data_ptr(),
      (int)hq, (int)hkv, (float)theta, (float)qk_factor);
#ifdef ONLYPREP
  return;
#endif
  A a;
  a.q = (const __half*)qbuf.data_ptr(); a.kc = (const uint8_t*)kc.data_ptr(); a.vc = (const uint8_t*)vc.data_ptr();
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
  } else if (L == 8 && hkv == 2 && hq == 16) {
    const int sm3 = RB3 * 4096;
    attn3<<<B * nsplit, 256, sm3, st>>>(a);
  } else if (mt <= 1) attn<1><<<blocks, 256, sm, st>>>(a);
  else if (mt <= 2) attn<2><<<blocks, 256, sm, st>>>(a);
  else attn<4><<<blocks, 256, sm, st>>>(a);
  if (nsplit > 32) comb2<<<B * L * hq, 256, 0, st>>>(a, (const __nv_bfloat16*)qkv.data_ptr(), (__nv_bfloat16*)out.data_ptr(), (float)kv_scale);
  else comb<<<(B * L * hq + 7) / 8, 256, 0, st>>>(a, (const __nv_bfloat16*)qkv.data_ptr(), (__nv_bfloat16*)out.data_ptr(), (float)kv_scale);
}
