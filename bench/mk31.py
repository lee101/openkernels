"""v31 = v30 + a high-occupancy ring variant for small M (2 CTAs/SM x 16 consumer warps x 1 tile, TP 8/16: 32 dequant warps
per SM instead of 16), added to the per-M autotune."""
D = '/vfast/data/code/openkernels/.claude/worktrees/agent-a17114b9f73577b71/muse_moe_w4a16/'
S = open(D + 'submission_v30.py').read()
def rep(o, n, cnt=1):
    global S
    assert o in S, o[:90]
    S = S.replace(o, n, cnt)
rep('''      if (nt == 1) stage<1>(acc, wsm, xs, lane);
      else if (nt == 2) stage<2>(acc, wsm, xs, lane);
      else if (nt == 3) stage<3>(acc, wsm, xs, lane);
      else stage<4>(acc, wsm, xs, lane);''', '''      if (nt == 1 || TP == 8) stage<1>(acc, wsm, xs, lane);
#if TP >= 16
      else if (nt == 2 || TP == 16) stage<2>(acc, wsm, xs, lane);
#endif
#if TP >= 32
      else if (nt == 3) stage<3>(acc, wsm, xs, lane);
      else stage<4>(acc, wsm, xs, lane);
#endif''')
rep('''        if (nt == 1) epilogue<1>(p, sm, list, acc, h, wid, lane, pend);
        else if (nt == 2) epilogue<2>(p, sm, list, acc, h, wid, lane, pend);
        else if (nt == 3) epilogue<3>(p, sm, list, acc, h, wid, lane, pend);
        else epilogue<4>(p, sm, list, acc, h, wid, lane, pend);''', '''        if (nt == 1 || TP == 8) epilogue<1>(p, sm, list, acc, h, wid, lane, pend);
#if TP >= 16
        else if (nt == 2 || TP == 16) epilogue<2>(p, sm, list, acc, h, wid, lane, pend);
#endif
#if TP >= 32
        else if (nt == 3) epilogue<3>(p, sm, list, acc, h, wid, lane, pend);
        else epilogue<4>(p, sm, list, acc, h, wid, lane, pend);
#endif''')
S = S.replace('moe_w4a16_v30', 'moe_w4a16_v31')
a = S.index('CUDA_L = r"""'); b = S.index('"""\n', a + 13)
L = S[a:b]
assert L.count('constexpr int CH = NT == 1 ? 4 : (NT == 2 ? 2 : 1);') == 2
L = L.replace('constexpr int CH = NT == 1 ? 4 : (NT == 2 ? 2 : 1);', 'constexpr int CH = NT == 1 ? CHN1 : (NT == 2 ? 2 : 1);')
L = L.replace('#ifndef LAG', '#ifndef CHN1\n#define CHN1 4\n#endif\n#ifndef LAG', 1)
assert 'moe_kernel(P p) {' in L
L = L.replace('moe_kernel(P p) {', 'moe_kernel(const __grid_constant__ P p) {')
L = L.replace('TORCH_CHECK(nb >= CPB, "CPB not resident");', 'if (nb < CPB) { cudaFuncAttributes fa; cudaFuncGetAttributes(&fa, moe_kernel); TORCH_CHECK(false, "CPB not resident: blocks ", nb, " smem ", smem_bytes(M), " regs ", fa.numRegs); }')
S = S[:a] + L + S[b:]
rep('"""v30: per-M choice', '"""v31: v30 + ring variant "h" (2 CTAs/SM x 16 consumer warps, one 16-row tile each, TP 16) for M <= 16.\nv30: per-M choice')
# variant registry: nx1 flag -> generic variant key
rep('''def _extL(nx1=0):
    if nx1 not in _modL:
        fl = (["-DRPW=2", "-DNSLOT=4"] if _big() else ["-DRPW=1", "-DNSLOT=2"]) + (["-DNX1"] if nx1 else []) + list(_DBG)''', '''RING_VARIANTS = {0: ([], 0), 1: (["-DNX1"], 0), 2: (["-DNX1", "-DNCW=16", "-DRPW=1", "-DCPB=2", "-DNSLOT=2", "-DTP=16", "-DCHN1=2"], 16)}  # flags, max M


def _extL(nx1=0):
    if nx1 not in _modL:
        vf = RING_VARIANTS[nx1][0]
        if not _big(): vf = [{"-DTP=16": "-DTP=8", "-DNSLOT=2": "-DNSLOT=1"}.get(d, d) for d in vf]  # sm_86 smem: 2 CTAs of 16 consumers fit only with TP 8
        base = (["-DRPW=2", "-DNSLOT=4"] if _big() else ["-DRPW=1", "-DNSLOT=2"]) if "-DRPW=1" not in vf else ([] if _big() else ["-DNSLOT=1"])
        fl = [d for d in base if not any(x.startswith(d.split("=")[0] + "=") for x in vf)] + vf + list(_DBG)''')
rep('''    for nx in (0, 1):
        for ks in ((1, 2, 4, 8) if M <= KSL_MAX else (1,)):
            c.append(("ring", ks, nx))''', '''    for nx, (_, mmax) in RING_VARIANTS.items():
        if mmax and M > mmax: continue
        for ks in ((1, 2, 4, 8) if M <= KSL_MAX else (1,)):
            c.append(("ring", ks, nx))''')
rep("[lambda: _extL(0), lambda: _extL(1)] + [lambda k=k: _extS(k) for k in V17_KSF]", "[lambda v=v: _extL(v) for v in RING_VARIANTS] + [lambda k=k: _extS(k) for k in V17_KSF]")
rep("ThreadPoolExecutor(5)", "ThreadPoolExecutor(6)")
rep('''f"M{M}:" + ",".join(f"{c[0][0]}{c[1]}{'n' if c[2] else ''}=''', '''f"M{M}:" + ",".join(f"{c[0][0]}{c[1]}{'nh'[c[2] - 1] if c[2] else ''}=''')
rep('''+ f"->{best[0]}{best[1]}{'n' if best[2] else ''}")''', '''+ f"->{best[0]}{best[1]}{'nh'[best[2] - 1] if best[2] else ''}")''')
S = S.replace('"v30 tune ', '"v31 tune ').replace('"v30 prepare', '"v31 prepare')
open(D + 'submission_v31.py', 'w').write(S)
print('ok')
