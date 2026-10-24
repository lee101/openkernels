"""Calibration cases: real H200 measurements (judge `okc check` medians, data/h200_measured.json) and pod microbenchmarks /
ablations (README.md, docs/ideas.md, notes/h200_session1.md), each paired with a model description built only from
shapes, launch geometry and SASS loop counts (vc/sass.py on the judged source). Builders double as templates for new ideas."""
import json, math
from pathlib import Path
from .model import Kernel, Phase, SM

DATA = json.loads((Path(__file__).resolve().parent.parent / "data" / "h200_measured.json").read_text())
MB = 1e6

# ---- SASS features (data/sass_db.json, from vc/sass.py on the judged sources, sm_90a, CUDA 12.9):
# (instrs, 1-warp cycles, tensor SM-cycles) per loop iteration; straight-line 1-warp cycles and static size per kernel
from . import sassdb as S
_k = lambda n: S.get(n)
FDEC_STEP, FDEC_T1, FDEC_TC = S.loop("attn_v38:void fdec<false>(F)", lambda l: l["mma"] > 0)        # 16-token step per warp
AW_STEP, AW_T1, AW_TENSOR = S.loop("attn_v38:void attnw<true, true, true>(A, WF)", lambda l: l["wgmma"] > 0)  # 64-key step per warp
AW_STEP_V35, AW_T1_V35, AW_TENSOR_V35 = S.loop("attn_v35:void attnw<true, true, true>(A, WF)", lambda l: l["wgmma"] > 0)
MOE = {(lg, nt): S.loop(f"moe_v17_{'large' if lg else 'small'}:moe_kernel(P)", lambda l, nt=nt: l["mma"] == 32 * nt)
       for lg in (False, True) for nt in ((1, 2, 4, 8) if lg else (1, 2, 4))}  # per-warp 4.6 KB stage by NT (tokens/8 per pass)
SKINNY_STEP, SKINNY_T1, _ = S.loop("proj_v11:void skinny<1, false, 2, 2>(Q)", lambda l: l["mma"] > 0)
RTR_LOOP, RTR_T1, _ = S.loop("proj_v11:void rtr<1>(Q)", lambda l: l["n"] > 100)
_K = {"fdec": "attn_v38:void fdec<false>(F)", "attnw": "attn_v38:void attnw<true, true, true>(A, WF)", "attnw_nf": "attn_v38:void attnw<true, false, false>(A, WF)",
      "moe": "moe_v17_small:moe_kernel(P)", "moe_large": "moe_v17_large:moe_kernel(P)", "skinny": "proj_v11:void skinny<1, false, 2, 2>(Q)",
      "rtr": "proj_v11:void rtr<1>(Q)", "fin": "proj_v11:fin(Q)"}
WFIX = {k: _k(v)["path_cyc"] for k, v in _K.items()} | {"small": 1500, "nvjet": 3000}  # nvjet / small helper kernels: guesses
CODE = {k: _k(v)["code"] for k, v in _K.items()} | {"small": 300, "nvjet": 1500}


def judge(prob, ver, case_sub):
    for f, d in DATA[prob].items():
        if f.startswith(ver + "."):
            for c, x in d["cases"].items():
                if all(s in c for s in case_sub.split("|")): return x
    raise KeyError((prob, ver, case_sub))


# ------------------------------------------------------------------ builders
def tma_stream(nbytes, warps=16, stages=2, stage=4608):
    """single persistent TMA ring stream (bf16_proj pod microbench)."""
    return [Kernel("tma_stream", [Phase("stream", hbm=nbytes, inflight=warps * stages * stage)], code=CODE["small"], wfix=WFIX["small"])]


def cublas_gemv(nbytes, splitk_out=0, flops=0):
    """cuBLAS nvjet skinny GEMM (weights streamed once, wgmma) + optional split-K reduce kernel (fp32 partials)."""
    ks = [Kernel("nvjet", [Phase("stream", hbm=nbytes, tensor=flops / 4096)], code=CODE["nvjet"], wfix=WFIX["nvjet"])]
    if splitk_out: ks.append(Kernel("splitKreduce", [Phase("reduce", l2=splitk_out, l2chain=1)], code=CODE["small"], wfix=WFIX["small"]))
    return ks


def skinny(nbytes, m, n, chunk=4096, fin=True):
    """bf16_proj stream-K GEMV: 132 CTAs x 16 warps, 2-stage TMA rings, fp32 atomics into ws, then fin kernel."""
    chunks = nbytes / chunk
    ks = [Kernel("skinny", [Phase("stream", hbm=nbytes, issue=chunks * SKINNY_STEP, inflight=16 * 2 * chunk, wcyc=chunks / (SM * 16) * SKINNY_T1),
                            Phase("atomics", l2=m * n * 4 * 2, l2chain=1)], code=CODE["skinny"], wfix=WFIX["skinny"])]
    if fin: ks.append(Kernel("fin", [Phase("fin", l2=m * n * 8, l2chain=1)], launches=0, code=CODE["fin"], wfix=WFIX["fin"]))  # PDL launch
    return ks


def rtr(m):
    """router fp32 W [256, 8192] as hi+lo bf16 (8.4 MB): CTA = 16-row tile x 16 K-chunks, smem warp reduce, last-arriver."""
    nbytes = 256 * 8192 * 4
    return [Kernel("rtr", [Phase("load", hbm=nbytes, sms=128, issue=nbytes / 512 / 16 * RTR_LOOP, wcyc=nbytes / 512 / 16 / (128 * 16) * RTR_T1),
                           Phase("reduce", syncs=1, l2chain=1)], code=CODE["rtr"], wfix=WFIX["rtr"])]


def kv_tokens(B, ctx, sliding, L=1):
    """(total tokens read, max/mean imbalance): agentic spread uniform ctx +/- 7.5% for B > 1."""
    if sliding: return B * (2048 + L), 1.0
    if B == 1: return ctx + L, 1.0
    return B * (ctx + L), 1 + 0.075 * (B - 1) / (B + 1)


def fdec(B, ctx, sliding, step=FDEC_STEP):
    """fused decode attention: CTA per (request, split), 8 warps x 4 cp.async 4 KB slots in flight, 16 tokens per warp step,
    online softmax, CTA merge in smem, cross-CTA merge (last arriver) or cluster DSMEM merge."""
    span = 2049 if sliding else ctx + 8
    NS = max(1, min(132 // B, -(-span // 128)))
    if NS <= 17 and B * min(NS, 16) <= 64: NS = 1 << (min(NS, 16).bit_length() - 1)
    tok, imb = kv_tokens(B, ctx, sliding)
    ctas = B * NS
    steps = tok / 16
    return [Kernel(f"fdec B{B} NS{NS}", code=CODE["fdec"], wfix=WFIX["fdec"], phases=[
        Phase("prologue", chain=2),  # seq_len -> page table (-> first KV rows = the main phase's fill latency)
        Phase("main", g128=tok * 256, sms=min(SM, ctas), inflight=8 * 4 * 4096, issue=steps * step, imb=imb,
              wcyc=steps / (ctas * 8) * FDEC_T1 * step / FDEC_STEP, tensor=steps * FDEC_TC),
        Phase("merge", syncs=1 if NS > 1 else 0, l2chain=2, l2_sm=NS * 16 * 66 * 4 if NS > 1 else 0)])]


def attnw(B, ctx, sliding, fused=True, step=AW_STEP, t1=AW_T1, tc=AW_TENSOR):
    """verify attention (8 queries x 16 heads): CTA per (request, split), 2 warpgroups (one per KV head), 2 CTAs/SM,
    64-key steps: FP8 QK wgmma from the raw cp.async ring, f16 PV wgmma, softmax; split merge in-kernel (fused) or comb2."""
    span = 2056 if sliding else ctx + 8
    nsplit = max(1, min(-(-span // 256), 264 // B))
    tok, imb = kv_tokens(B, ctx, sliding, 8)
    ctas = B * nsplit
    wg_steps = 2 * tok / 64
    sms = min(SM, ctas)
    main = Phase("main", g128=tok * 256, sms=sms, inflight=2 * 4 * 8192, issue=wg_steps * 4 * step, tensor=wg_steps * 4 * tc, imb=imb,
                 wcyc=wg_steps / (ctas * 2) * t1)
    merge = Phase("merge", syncs=1, l2chain=2, l2_sm=nsplit * 2 * (64 * 64 + 128) * 4) if (fused and nsplit > 1) else Phase("merge")
    ks = [Kernel(f"attnw B{B} ns{nsplit}", [Phase("prologue", chain=2), main, merge], code=CODE["attnw"], wfix=WFIX["attnw" if fused else "attnw_nf"])]
    if not fused:
        ks = [Kernel("prep", [Phase("prep", l2chain=2)], code=CODE["small"], wfix=WFIX["small"])] + ks + [Kernel("comb2", [Phase("comb2", l2=B * 8 * 16 * 64 * 4 * nsplit, l2chain=2)], code=CODE["small"], wfix=WFIX["small"])]
    return ks


EXPERT = 3 * 4096 * 4096 * 0.5625  # NVFP4 bytes per expert (w13 + w2 + fp8 scales) = 28.3 MB
MOE_PAIRS = {1: 2.25, 8: 12.9, 16: 25.3, 32: 51.4, 128: 209, 256: 393}  # local (token, expert) pairs, reference generator


def moe(bs, sol_us, fence=True, dequant=True):
    """persistent NVFP4 MoE (v17): 132 CTAs, tile queue over 16-row tiles x 4096 K (8 TMA stages of 4.6 KB), NVFP4 dequant
    + mma.sync; small M: 16 warps x 2 stages, split-K ks warps per tile; M > 64: 8 warps x 4 stages, up to 64 tokens per pass.
    G1 (w13, 512 tiles/expert) -> G2 (w2, 256 tiles/expert) dependency, then norm/weight epilogue."""
    e = sol_us * 4.8e12 / 1e6 / EXPERT
    pairs = MOE_PAIRS[bs]; tpe = pairs / e
    large = bs > 64
    if large:
        wpc, ks = 8, 1
        passes = max(1.0, tpe / 64); nt = min(8, max(1, round(tpe / passes / 8)))
    else:
        wpc = 16
        tw = e * 768 / 132
        ks = 4 if tw < 32 else (2 if tw < 96 else 1)
        passes, nt = 1.0, min(4, max(1, math.ceil(tpe / 8)))
    nt = 1 << (nt - 1).bit_length()
    n_i, t1, tc = MOE[(large, nt)]
    stage = n_i if dequant else n_i - 560   # FAKEDQ ablation: drop the ~560 dequant instrs/stage (LOP3/SHF/HMUL2/PRMT)
    t1 *= stage / n_i
    par = SM * wpc / ks
    def g(name, tiles, frac):
        return Phase(name, hbm=e * EXPERT * frac, l2=(tiles * passes * tpe * 8192 if large else 0), inflight=wpc * (4 if large else 2) * 4608,
                     issue=tiles * passes * 8 * stage, tensor=tiles * passes * 8 * tc, units=tiles * passes, par=par, dynamic=True,
                     unit_issue=8 / ks * stage, unit_cyc=8 / ks * t1, wps=wpc / 4, unit_sync=1 if fence else 0)  # fence.acq_rel + release add per tile
    return [Kernel(f"moe bs{bs} e{e:.1f} ks{ks} nt{nt}", code=CODE["moe_large" if large else "moe"], wfix=WFIX["moe_large" if large else "moe"], phases=[
        Phase("routing", chain=2, l2chain=2),
        g("G1", e * 512, 2 / 3), g("G2", e * 256, 1 / 3),
        Phase("epilogue", syncs=1, l2chain=3)])]


# ------------------------------------------------------------------ cases
LAYER_BYTES = {"qkv_gate": 2304 * 8192 * 2, "o": 8192 * 1024 * 2, "pooler": 4096 * 8192 * 2, "unpooler": 8192 * 4096 * 2}
LAYER_N = {"qkv_gate": 2304, "o": 8192, "pooler": 4096, "unpooler": 8192}


def cases():
    C = []
    add = lambda group, name, ks, meas, src, mode="judge": C.append({"group": group, "name": name, "ks": ks, "meas": meas, "src": src, "mode": mode})
    # microbenchmarks (pod, graph replay minus empty graph)
    for nb, t in ((16.8e6, 5.7), (37.7e6, 10.0), (67.1e6, 16.5)):
        add("ubench", f"tma stream {nb / MB:.1f} MB", tma_stream(nb), t, "notes/h200_session1.md", "graph-empty")
    # bf16_proj v11: cuBLAS (nvjet) and ours
    V = ("muse_bf16_proj", "submission_v11")
    for layer, m, load in (("o", 1, "decode 1"), ("o", 16, "decode 16"), ("o", 32, "decode 32"), ("o", 8, "verify 1"),
                           ("qkv_gate", 16, "decode 16"), ("qkv_gate", 32, "decode 32"), ("qkv_gate", 8, "verify 1"),
                           ("pooler", 16, "decode 16"), ("pooler", 32, "decode 32"), ("pooler", 8, "verify 1"),
                           ("unpooler", 16, "decode 16"), ("unpooler", 32, "decode 32"),
                           ("o", 128, "verify 16"), ("unpooler", 128, "verify 16"), ("pooler", 128, "verify 16"),
                           ("o", 256, "verify 32"), ("unpooler", 256, "verify 32"), ("pooler", 256, "verify 32"), ("qkv_gate", 256, "verify 32")):
        j = judge(*V, f"layer={layer} |load={load} ")
        sk = sum(1 for k in j["kernels"] if "splitK" in k[0])
        add("cublas", f"{layer} m{m}", cublas_gemv(LAYER_BYTES[layer] * (1 + m / LAYER_N[layer]), m * LAYER_N[layer] * 4 * 4 if sk else 0, 2 * max(m, 8) * LAYER_BYTES[layer] / 2), j["yours"], f"{V[1]} judge")
    for layer, m, load, nb in (("qkv_gate", 1, "decode 1", LAYER_BYTES["qkv_gate"]), ("unpooler", 1, "decode 1", 1024 * 4096 * 2)):
        j = judge(*V, f"layer={layer} |load={load} ")
        add("bf16 ours", f"skinny {layer} m{m}", skinny(nb, m, LAYER_N[layer] if layer != "unpooler" else 1024), j["yours"], f"{V[1]} judge")
    for m, load in ((1, "decode 1"), (16, "decode 16"), (32, "decode 32"), (8, "verify 1")):
        j = judge(*V, f"layer=router |load={load} ")
        add("bf16 ours", f"rtr router m{m}", rtr(m), j["yours"], f"{V[1]} judge")
    # attention v38
    A = ("muse_attention_decode", "submission_v38")
    for B in (1, 16, 32):
        for sl in (True, False):
            lay = "sliding" if sl else "full"
            j = judge(*A, f"batch={B} |layer={lay} |load=decode")
            add("attention", f"fdec decode B{B} {lay}", fdec(B, 80000, sl), j["yours"], f"{A[1]} judge")
            j = judge(*A, f"batch={B} |layer={lay} |load=verify")
            add("attention", f"attnw verify B{B} {lay}", attnw(B, 80000, sl, fused=(B > 1 or sl)), j["yours"], f"{A[1]} judge")
    # moe v17
    M = ("muse_moe_w4a16", "submission_v17")
    for bs, load in ((1, "decode 1"), (16, "decode 16"), (32, "decode 32"), (8, "verify 1"), (128, "verify 16"), (256, "verify 32")):
        j = judge(*M, f"bs={bs} |load={load} ")
        add("moe", f"moe bs{bs}", moe(bs, j["sol"]), j["yours"], f"{M[1]} judge")
    return C


# ------------------------------------------------------------------ relative cases (variant A vs B, measured ratio B/A)
def deltas():
    D = []
    add = lambda name, a, b, ratio, src: D.append({"name": name, "a": a, "b": b, "meas": ratio, "src": src})
    for B, full, noload, nomma, both in ((16, 164, 145, 154, 139), (32, 313, 277, 294, 264)):
        base = attnw(B, 80000, False)
        nl = attnw(B, 80000, False); nl[0].phases[1].g128 = 0
        nm = attnw(B, 80000, False); nm[0].phases[1].tensor = 0; nm[0].phases[1].issue *= (AW_STEP - 10) / AW_STEP
        bo = attnw(B, 80000, False); bo[0].phases[1].g128 = 0; bo[0].phases[1].tensor = 0; bo[0].phases[1].issue *= (AW_STEP - 10) / AW_STEP
        add(f"attnw b{B} verify full: no KV loads", base, nl, noload / full, "ideas.md 2026-10-25 pod")
        add(f"attnw b{B} verify full: no MMA", base, nm, nomma / full, "ideas.md 2026-10-25 pod")
        add(f"attnw b{B} verify full: neither", base, bo, both / full, "ideas.md 2026-10-25 pod")
    for B, a, b in ((16, 160.8, 156.9), (32, 302.2, 293.7)):
        add(f"attnw b{B} verify full: loop 786->735 SASS (v37->v38)", attnw(B, 80000, False, step=AW_STEP_V35, t1=AW_T1_V35, tc=AW_TENSOR_V35), attnw(B, 80000, False), b / a, "README v38 judge")
    j1, j9 = judge("muse_moe_w4a16", "submission_v17", "bs=1 |"), None
    add("moe bs1: FAKEDQ (no dequant ALU)", moe(1, j1["sol"]), moe(1, j1["sol"], dequant=False), 0.82, "ideas.md v9 DIAG -18%")
    return D
