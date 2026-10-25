"""v32 = v30 + ring-path K-chunk dependencies: W13 completion is counted per (expert, 512-column chunk of inter) and the
producer waits per W2 stage for just the chunk that stage reads; W2 units are queued K-part-major. Removes the whole-expert
W13 -> W2 barrier (H100 PROF bs1: ~11 us producer dep-wait)."""
D = '/vfast/data/code/openkernels/.claude/worktrees/agent-a17114b9f73577b71/muse_moe_w4a16/'
S = open(D + 'submission_v30.py').read()
a = S.index('CUDA_L = r"""'); b = S.index('"""\n', a + 13)
L = S[a:b]
def rep(o, n):
    global L
    assert L.count(o) == 1, o[:90]
    L = L.replace(o, n)
rep('#define C_UC 64', '#define C_UC 64\n#define NCHK 8  // W13 -> W2 dependency granularity: inter columns per chunk = 4096 / NCHK\n#define C_DEP (C_UC + 2 * NLOC * 64 * 8)\n#ifndef KMAJ\n#define KMAJ 1\n#endif')
rep('int cnt[NLOC], off[NLOC], act[NLOC], np[NLOC], base[NLOC];', 'int cnt[NLOC], off[NLOC], act[NLOC], np[NLOC], base[NLOC], depc[NLOC];')
rep('  if (tid < NLOC) sm->base[tid] = 0;', '  if (tid < NLOC) { sm->base[tid] = 0; sm->depc[tid] = 0; }')
rep('red_rel(p.ctr + e, 1)', 'red_rel(p.ctr + C_DEP + e * NCHK + h.rb / (NRB1 / NCHK), 1)')
rep('red_rel(p.ctr + (h.g ? C_W2 : e), NCW)', 'red_rel(p.ctr + (h.g ? C_W2 : C_DEP + e * NCHK + h.rb / (NRB1 / NCHK)), NCW)')
rep('kp = v % ks; v /= ks; ps = v % np; rb = v / np;', 'if (g && KMAJ) { const int per = NRB2 * np; kp = v / per; v %= per; ps = v % np; rb = v / np; }\n  else { kp = v % ks; v /= ks; ps = v % np; rb = v / np; }')
rep('for (int i = 0; i <= C_DONE; i++) p.ctr[i] = 0;', 'for (int i = 0; i <= C_DONE; i++) p.ctr[i] = 0;\n      for (int i = 0; i < NLOC * NCHK; i++) p.ctr[C_DEP + i] = 0;')
i0 = L.index('        if (g && i == 0 && !(depok >> e & 1)) {')
i1 = L.index('        for (int rw = 0; rw < ntok; rw++) {', i0)
L = L[:i0] + '''        if (g) {  // wait only for the W13 chunk of inter this stage reads
#ifdef PROF
          long long c1 = clock64();
#endif
          const int ch = s * NCHK / NHS;
          if (lane == 0 && sm->depc[e] <= ch) {
            const int tgt = (NRB1 / NCHK) * sm->np[e] * NCW;
            for (int c = sm->depc[e]; c <= ch; c++) while (ldacq(p.ctr + C_DEP + e * NCHK + c) < tgt) __nanosleep(32);
            sm->depc[e] = ch + 1;
          }
          __syncwarp();
#ifdef PROF
          wdep += clock64() - c1;
#endif
        }
''' + L[i1:]
L = L.replace('    unsigned depok = 0;\n', '')
S = S[:a] + L + S[b:]
S = S.replace('moe_w4a16_v30', 'moe_w4a16_v32')
S = S.replace('"""v30: per-M choice', '"""v32: v30 + ring-path W13 -> W2 dependencies per 512-column chunk of inter (checked per W2 stage; W2 units K-part-major).\nv30: per-M choice', 1)
for o, n in (('"ctrL": torch.zeros(64 + 2 * 32 * 64 * 8,', '"ctrL": torch.zeros(64 + 2 * 32 * 64 * 8 + 32 * 8,'), ('"v30 tune ', '"v32 tune '), ('"v30 prepare', '"v32 prepare')):
    assert o in S, o
    S = S.replace(o, n)
open(D + 'submission_v32.py', 'w').write(S)
print('ok')
