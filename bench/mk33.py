"""v33 = v30 with a lighter v17-path tile machinery (the path that runs at M <= 64):
- K-split groups without a group barrier: each warp drops its partial in smem (3 rotating buffers) and the last arriver
  (smem atomic) finishes the tile; others move on. Tile ids exchanged through an 8-deep tagged ring (readers spin on the tag).
- W2 dependency: each warp remembers which experts' W13 it has already seen complete (no L2 round trip per W2 tile).
- queue heads fetched before the list fill (latency overlaps it); combine reads routes and weights from smem.
- MDM diag modes (1 W13 only, 2 W2 only, 3 skeleton, 4 prologue + combine, 5 empty) and the MD line from prepare()."""
D = '/vfast/data/code/openkernels/.claude/worktrees/agent-a17114b9f73577b71/muse_moe_w4a16/'
S = open(D + 'submission_v30.py').read()
a = S.index('CUDA = r"""') + 11
b = S.index('"""\n', a)
C = S[a:b]
def rep(o, n, cnt=1):
    global C
    assert C.count(o) == cnt, (C.count(o), o[:90])
    C = C.replace(o, n)
rep('#define RED (2 * 2 * 32 * 16)  // per-warp split-K partial slots: 2 tile parities x NT <= 2 float4 per lane',
    '#ifndef NRB\n#define NRB 3\n#endif\n#define RED (NRB * 2 * 32 * 16)  // per-warp split-K partial slots: NRB rotating tile buffers x NT <= 2 float4 per lane')
rep('struct Sm { int cnt[NLOC], off[NLOC], fill[NLOC], act[NLOC], na, ks, maxc; int ids[16][4]; };',
    'struct Sm { int cnt[NLOC], off[NLOC], fill[NLOC], act[NLOC], na, ks, maxc; int ids[16][8]; int gcnt[16][NRB], ggen[16][NRB]; };')
rep('  __device__ __forceinline__ int tid_of(int i) const { return ((volatile int*)sm->ids[gi])[i & 3]; }',
    '''  __device__ __forceinline__ int tid_of(int i) const {  // tagged: (tile index & 0x7fff) << 16 | unit
    const volatile int* q = sm->ids[gi] + (i & 7);
    int v = *q;
    while ((v >> 16) != (i & 0x7fff)) v = *q;
    return v & 0xffff;
  }''')
# epilogue: last arriver instead of group barrier
rep('''  if (NT <= 2 && C.ks > 1) {  // group split-K: partials meet in smem (double-buffered by tile parity)
    const int wid = threadIdx.x >> 5, ks = C.ks, w0 = wid - C.gj;
    float4* red = (float4*)(smem + (size_t)WPC * S * STG + WPC * SCR + WPC * S * 8) + (ti & 1) * WPC * 64;
#pragma unroll
    for (int n = 0; n < NT; n++) red[(wid * 2 + n) * 32 + lane] = make_float4(f[n][0], f[n][1], f[n][2], f[n][3]);
    gbar(1 + C.gi, ks * 32);
    if (C.gj != (ti & (ks - 1))) return false;
    for (int o = w0; o < w0 + ks; o++) {''', '''  if (NT <= 2 && C.ks > 1) {  // group split-K: partials meet in smem; the last arriving warp finishes, nobody waits
    const int wid = threadIdx.x >> 5, ks = C.ks, w0 = wid - C.gj, bi = ti % NRB, use = ti / NRB;
    float4* red = (float4*)(smem + (size_t)WPC * S * STG + WPC * SCR + WPC * S * 8) + bi * WPC * 64;
    if (lane == 0) { const volatile int* gg = &C.sm->ggen[C.gi][bi]; while (*gg != use) {} }  // buffer free: tile ti - NRB finished
    __syncwarp();
#pragma unroll
    for (int n = 0; n < NT; n++) red[(wid * 2 + n) * 32 + lane] = make_float4(f[n][0], f[n][1], f[n][2], f[n][3]);
    __syncwarp();
    int last = 0;
    if (lane == 0) { __threadfence_block(); last = atomicAdd(&C.sm->gcnt[C.gi][bi], 1) == ks - 1; }
    if (!__shfl_sync(0xffffffffu, last, 0)) return false;
    __threadfence_block();
    for (int o = w0; o < w0 + ks; o++) {''')
rep('''      for (int n = 0; n < NT; n++) { float4 v = red[(o * 2 + n) * 32 + lane]; f[n][0] += v.x; f[n][1] += v.y; f[n][2] += v.z; f[n][3] += v.w; }
    }
  }''', '''      for (int n = 0; n < NT; n++) { float4 v = red[(o * 2 + n) * 32 + lane]; f[n][0] += v.x; f[n][1] += v.y; f[n][2] += v.z; f[n][3] += v.w; }
    }
    __syncwarp();
    if (lane == 0) { C.sm->gcnt[C.gi][bi] = 0; __threadfence_block(); ((volatile int*)C.sm->ggen[C.gi])[bi] = use + 1; }
  }''')
# prologue: init the new smem state, tws, early queue heads
rep('  if (tid < NLOC) sm->cnt[tid] = 0;\n  __syncthreads();', '''  if (tid < NLOC) sm->cnt[tid] = 0;
  if (tid < 16 * NRB) { (&sm->gcnt[0][0])[tid] = 0; (&sm->ggen[0][0])[tid] = 0; }
  if (tid < 16 * 8) (&sm->ids[0][0])[tid] = -1;
  __syncthreads();''')
rep('for (int i = tid; i < R; i += blockDim.x) { unsigned e = p.ids[i] - p.lo; lid[i] = e < NLOC ? e : 0xff; if (e < NLOC) atomicAdd(&sm->cnt[e], 1); }',
    'for (int i = tid; i < R; i += blockDim.x) { unsigned e = p.ids[i] - p.lo; lid[i] = e < NLOC ? e : 0xff; if (e < NLOC) atomicAdd(&sm->cnt[e], 1); }')
rep('''  __syncthreads();
  for (int i = tid; i < R; i += blockDim.x) { unsigned e = lid[i]; if (e < NLOC) list[atomicAdd(&sm->fill[e], 1)] = i; }
  const int ks = sm->ks, na = sm->na;''', '''  __syncthreads();
  const int ks = sm->ks, na = sm->na;
  if (wid % ks == 0 && lane == 0) { const int gi = wid / ks; sm->ids[gi][0] = atomicAdd(p.ctr + QI, 1) + QOFS0(na); sm->ids[gi][1] = (1 << 16) | (atomicAdd(p.ctr + QI, 1) + QOFS0(na)); }
  for (int i = tid; i < R; i += blockDim.x) { unsigned e = lid[i]; if (e < NLOC) list[atomicAdd(&sm->fill[e], 1)] = i; }''')
rep('  if (lead) { sm->ids[C.gi][0] = atomicAdd(p.ctr + QI, 1); sm->ids[C.gi][1] = atomicAdd(p.ctr + QI, 1); }\n', '')
rep('    if (lead) nxt = atomicAdd(p.ctr + QI, 1);  // tile for i + 2; latency hidden behind this tile',
    '    if (lead) nxt = atomicAdd(p.ctr + QI, 1) + QOFS0(na);  // tile for i + 2; latency hidden behind this tile')
rep('    if (lead) sm->ids[C.gi][(i + 2) & 3] = nxt;  // read 2 tiles later, after the group barrier (ks > 1) / syncwarp',
    '    if (lead) ((volatile int*)sm->ids[C.gi])[(i + 2) & 7] = (((i + 2) & 0x7fff) << 16) | nxt;  // tagged: readers spin until it lands')
# W2 dependency cache
rep('''    if (g2) {
      if (lane == 0) while (ldacq(p.ctr + e) < T1E) __nanosleep(32);
      __syncwarp();''', '''    if (g2) {
      if (!(dok >> e & 1)) {
        if (lane == 0 && MDM != 2) while (ldacq(p.ctr + e) < T1E) __nanosleep(32);
        __syncwarp();
        dok |= 1u << e;
      }''')
rep('  for (int i = 0;; i++) {\n    const int u = C.tid_of(i);', '  unsigned dok = 0;\n  for (int i = 0;; i++) {\n    const int u = C.tid_of(i);')
# combine from smem
rep('''      const int r = t * 8 + k;
      unsigned e = p.ids[r] - p.lo;
      if (e < NLOC) {
        float cf = p.tw[r] * rsqrtf(__ldcg(p.ss + r) * (1.f / HD) + 1e-8f);''', '''      const int r = t * 8 + k;
      if (lid[r] < NLOC) {
        float cf = p.tw[r] * rsqrtf(__ldcg(p.ss + r) * (1.f / HD) + 1e-8f);''')
# diag modes
rep('#ifndef KSF\n', '#ifndef MDM\n#define MDM 0  // diag: 1 W13 only, 2 W2 only, 3 skeleton (no TMA, no math), 4 prologue + combine, 5 empty\n#endif\n#define QOFS0(na) (MDM == 2 ? (na) * T1E : 0)\n#ifndef KSF\n')
rep('      if (lane == 0) tma_ld(ring + slot * STG, src, bars + slot * 8);', '#if MDM != 3\n      if (lane == 0) tma_ld(ring + slot * STG, src, bars + slot * 8);\n#endif')
rep('    mb_wait(bars + slot * 8, (q / S) & 1);', '#if MDM != 3\n    mb_wait(bars + slot * 8, (q / S) & 1);\n#endif')
rep('''    const uint32_t slot = C.acquire();
    const uint2 sh0''', '''    const uint32_t slot = C.acquire();
#if MDM == 3
    C.release(); continue;
#endif
    const uint2 sh0''')
rep('''  constexpr int S = NSTG;
  extern __shared__ __align__(16) uint8_t smem[];''', '''  constexpr int S = NSTG;
#if MDM == 5
  return;
#endif
  extern __shared__ __align__(16) uint8_t smem[];''')
rep('  C.N1 = na * T1E; C.N = C.N1 + na * T2E;', '''  C.N1 = na * T1E; C.N = C.N1 + na * T2E;
#if MDM == 1
  C.N = C.N1;
#elif MDM == 4
  C.N = 0;
#endif''')
rep('  const int T2 = sm->na * T2E;', '  const int T2 = (MDM == 1 || MDM == 4) ? 0 : sm->na * T2E;')
S = S[:a] + C + S[b:]
S = S.replace('moe_w4a16_v30', 'moe_w4a16_v33')
S = S.replace('"""v30: per-M choice', '"""v33: v30 with a lighter v17-path tile machinery: no group barrier (last-arriving warp finishes a K-split tile),\ntagged tile-id ring, cached W2 dependencies, early queue heads, combine from smem. v30: per-M choice', 1)
for o, n in (('"v30 tune ', '"v33 tune '), ('"v30 prepare', '"v33 prepare')):
    assert o in S
    S = S.replace(o, n)
# MD diag from prepare (as v30d), modes built from this file's CUDA
mk = open(D + 'bench/mkdiag.py').read()
md = mk[mk.index("S += '''") + 8:mk.index("'''\nS = S.replace('''    if TUNE")]
S += md.replace('moe_w4a16_v30dD_', 'moe_w4a16_v33D_')
o = '''    if TUNE and FORCE is None and el < 200:
        st["tune"] = _tune(st)
    return st'''
assert o in S
S = S.replace(o, '''    if TUNE and FORCE is None and el < 200:
        st["tune"] = _tune(st)
    if MDIAG:
        try:
            if time.time() - _T0 < 200: _diag_md(st)
            else: print("MD skipped: prepare at %.0fs" % (time.time() - _T0), flush=True)
        except Exception as ex:
            print("MD failed", ex, flush=True)
    return st''')
S = S.replace('TUNE = True\n', 'TUNE = True\nMDIAG = True  # print the MD line (v17 path: full / W13 / W2 / skeleton / prologue / empty) from prepare()\n', 1)
n0 = S.count('"-DNSTG=%d" % stg')
S = S.replace('"-DNSTG=%d" % stg', '"-DNSTG=%d" % stg, "-DNRB=%d" % (3 if _big() else 2)')
print('nrb flags', n0)
S = S.replace('MDIAG = True  #', 'MDIAG = bool(int(__import__("os").environ.get("MDIAG", "1")))  #')
open(D + 'submission_v33.py', 'w').write(S)
print('ok')
