"""v30 = v29 + ring-path lookahead-1 queue fetch (NX1) + prepare()-time autotune of (path, K split, NX1) per M bucket on the
judge GPU (synthetic routings, graph replays, cold L2)."""
D = '/vfast/data/code/openkernels/.claude/worktrees/agent-a17114b9f73577b71/muse_moe_w4a16/'
S = open(D + 'submission_v29.py').read()
def rep(o, n, cnt=1):
    global S
    assert o in S, o[:90]
    S = S.replace(o, n, cnt)
# NX1: the ring producer reserves its next unit only while issuing the last stage of the current one (shorter end tail)
rep('''  int u = 0, nx = 0;
  if (tid == NCW * 32) { u = atomicAdd(p.ctr + C_Q, 1); nx = atomicAdd(p.ctr + C_Q, 1); }  // queue heads early: latency hides behind routing''',
'''  int u = 0, nx = 0;
#ifdef NX1
  if (tid == NCW * 32) u = atomicAdd(p.ctr + C_Q, 1);
#else
  if (tid == NCW * 32) { u = atomicAdd(p.ctr + C_Q, 1); nx = atomicAdd(p.ctr + C_Q, 1); }  // queue heads early: latency hides behind routing
#endif''')
rep('''      for (int i = 0; i < nst; i++, seq++) {
        const int slot = seq % NSLOT, s = kp * nst + i;''', '''      for (int i = 0; i < nst; i++, seq++) {
        const int slot = seq % NSLOT, s = kp * nst + i;
#ifdef NX1
        if (i == nst - 1 && lane == 0) nx = atomicAdd(p.ctr + C_Q, 1);
#endif''')
rep('''      u = nx;
      if (lane == 0 && nx < ntot) nx = atomicAdd(p.ctr + C_Q, 1);
      nx = __shfl_sync(0xffffffffu, nx, 0);''', '''#ifdef NX1
      u = __shfl_sync(0xffffffffu, nx, 0);
#else
      u = nx;
      if (lane == 0 && nx < ntot) nx = atomicAdd(p.ctr + C_Q, 1);
      nx = __shfl_sync(0xffffffffu, nx, 0);
#endif''')
S = S.replace('moe_w4a16_v29', 'moe_w4a16_v30')
rep('"""v29: M <= MBIG runs v28', '"""v30: per-M choice (autotuned in prepare on the judge GPU) between v28 and the ring path below with K split 1-8 and\nlookahead-1 queue fetch. v29: M <= MBIG runs v28')
a = S.index('\n\nMBIG = 64\n_modL = None')
S = S[:a] + '''

import time
from concurrent.futures import ThreadPoolExecutor
_modL = {}
TUNE_M = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)
KSL_MAX = 64  # ring K-split workspace covers M <= 64
TUNE = True
V17_KSF = (0, 1, 4)  # v17 path: 0 = its own K-split rule (4/2/1 by tiles per warp group), else forced
FORCE = None  # e.g. ("ring", 2, 1) or ("v17", 0, 0): skip the tuned table


def _extL(nx1=0):
    if nx1 not in _modL:
        fl = (["-DRPW=2", "-DNSLOT=4"] if _big() else ["-DRPW=1", "-DNSLOT=2"]) + (["-DNX1"] if nx1 else []) + list(_DBG)
        _modL[nx1] = load_inline("moe_w4a16_v30L" + "".join(fl).replace("-", "_").replace("=", ""),
                                 cpp_sources="void moe(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,torch::Tensor);",
                                 cuda_sources=CUDA_L, functions=["moe"], extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"] + fl, verbose=False)
    return _modL[nx1]


def _ring(data, state, ks, nx1):
    x = data["hidden_states"].contiguous()
    M = x.shape[0]
    ids = data["topk_ids"].to(torch.int32).contiguous()
    tw = data["topk_weights"].float().contiguous()
    inter = torch.empty((M * 8, 4096), dtype=torch.bfloat16, device=x.device)
    y = torch.empty((M * 8, 4096), dtype=torch.bfloat16, device=x.device)
    ss = torch.empty((M * 8,), dtype=torch.float32, device=x.device)
    out = torch.empty_like(x)
    _extL(nx1).moe(x, ids, tw, state["w1"], state["w2"], state["g1"], state["g2"], state["nw"], inter, y, ss, state["ctrL"],
                   state["wsL"], out, data["config"]["local_start"], ks if M <= KSL_MAX else 1, state["prof"])
    return out


_modS = {}


def _extS(ksf):
    if not ksf: return _ext()
    if ksf not in _modS:
        wpc, stg = (16, 2) if _big() else (8, 2)
        _modS[ksf] = load_inline("moe_w4a16_v30S_%d_%d_k%d" % (wpc, stg, ksf),
                                 cpp_sources="void moe(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,torch::Tensor);",
                                 cuda_sources=CUDA, functions=["moe"],
                                 extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo", "-DWPC=%d" % wpc, "-DNSTG=%d" % stg, "-DKSF=%d" % ksf] + list(_DBG), verbose=False)
    return _modS[ksf]


def _small(data, state, ksf):
    x = data["hidden_states"].contiguous()
    M = x.shape[0]
    ids = data["topk_ids"].to(torch.int32).contiguous()
    tw = data["topk_weights"].float().contiguous()
    inter = torch.empty((M * 8, 4096), dtype=torch.bfloat16, device=x.device)
    y = torch.empty((M * 8, 4096), dtype=torch.bfloat16, device=x.device)
    ss = torch.empty((M * 8,), dtype=torch.float32, device=x.device)
    out = torch.empty_like(x)
    _extS(ksf).moe(x, ids, tw, state["w1"], state["w2"], state["g1"], state["g2"], state["nw"], inter, y, ss, state["ctr"],
                   state["tcnt"], state["ws"], out, data["config"]["local_start"], state["prof"])
    return out


def _run(cfg, data, state):
    return _small(data, state, cfg[1]) if cfg[0] == "v17" else _ring(data, state, cfg[1], cfg[2])


def _default(M):
    return ("v17", 0, 0) if M <= 64 else ("ring", 1, 0)


def _cands(M):
    c = [("v17", k, 0) for k in V17_KSF] if M <= 64 else []
    for nx in (0, 1):
        for ks in ((1, 2, 4, 8) if M <= KSL_MAX else (1,)):
            c.append(("ring", ks, nx))
    return c


def _synth(M, seed, dev):
    """busiest-GPU top-8 routing with heavy-tailed expert popularity (the judge's soft routing shape), random activations"""
    g = torch.Generator(device=dev).manual_seed(seed)
    sc = torch.rand((M, 256), generator=g, device=dev)
    z = torch.randn((256,), generator=g, device=dev)
    u = torch.rand((256,), generator=g, device=dev).clamp(min=1e-12)
    r = (sc + 0.03 * z / (-u.log()).sqrt()).topk(8, dim=-1)
    ids = r.indices
    ids = (ids - 32 * torch.bincount((ids // 32).flatten(), minlength=8).argmax()) % 256
    return {"hidden_states": torch.randn((M, 4096), generator=g, device=dev).to(torch.bfloat16), "topk_ids": ids.to(torch.int32),
            "topk_weights": (r.values / r.values.sum(-1, keepdim=True)).float(), "config": {"local_start": 0}}


def _tune(state):
    dev = state["w1"].device
    flush = torch.empty(2 ** 27, device=dev); sink = torch.empty((), device=dev)
    table, lines = {}, []
    for M in TUNE_M:
        res = {}
        for seed in (11, 12):
            data = _synth(M, seed + M, dev)
            for cfg in _cands(M):
                try:
                    sr = torch.cuda.Stream(); sr.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(sr):
                        for _ in range(2): _run(cfg, data, state)
                    torch.cuda.current_stream().wait_stream(sr); torch.cuda.synchronize()
                    gr = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(gr): _run(cfg, data, state)
                    ts = []
                    for _ in range(12):
                        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                        sink.copy_(flush.sum()); e0.record(); gr.replay(); e1.record(); torch.cuda.synchronize()
                        ts.append(e0.elapsed_time(e1) * 1e3)
                    res[cfg] = res.get(cfg, 0.0) + sorted(ts)[len(ts) // 2]
                    del gr
                except Exception as ex:
                    res[cfg] = float("inf"); print("tune", M, cfg, "failed", ex, flush=True)
        d = _default(M)
        best = min(res, key=res.get)
        if res[best] > 0.98 * res.get(d, float("inf")): best = d
        table[M] = best
        lines.append(f"M{M}:" + ",".join(f"{c[0][0]}{c[1]}{'n' if c[2] else ''}={res[c] / 2:.0f}" for c in res) + f"->{best[0]}{best[1]}{'n' if best[2] else ''}")
    print("v30 tune " + " | ".join(lines), flush=True)
    torch.cuda.empty_cache()
    return table


def kernel_fn(data, state):
    M = data["hidden_states"].shape[0]
    if FORCE is not None: return _run(FORCE, data, state)
    tb = state.get("tune") or {}
    b = next((m for m in TUNE_M if m >= M), TUNE_M[-1])
    return _run(tb.get(b, _default(M)), data, state)
'''
rep('    _ext(); _extL()\n', '    with ThreadPoolExecutor(5) as ex: list(ex.map(lambda f: f(), [lambda: _extL(0), lambda: _extL(1)] + [lambda k=k: _extS(k) for k in V17_KSF]))\n')
rep('"wsL": torch.zeros((1,), dtype=torch.float32, device=dev)}', '"wsL": torch.zeros((KSL_MAX * 8 * 12288,), dtype=torch.float32, device=dev)}')
rep('    if DIAG:\n        _diag(st, weights)\n    return st', '    if DIAG:\n        _diag(st, weights)\n    el = time.time() - _T0\n    print("v30 prepare %.0fs before tuning" % el, flush=True)\n    if TUNE and FORCE is None and el < 200:\n        st["tune"] = _tune(st)\n    return st')
rep('def prepare(weights):\n', 'def prepare(weights):\n    global _T0\n    _T0 = time.time()\n')
rep('from torch.utils.cpp_extension import load_inline\n', 'from torch.utils.cpp_extension import load_inline\nimport time\n_T0 = 0.0\n')
open(D + 'submission_v30.py', 'w').write(S)
print('ok')
