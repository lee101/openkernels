P_GVC = {
    ("qkv_gate", 1): [((1, 8, 16, 1, 1, 0), (1,))],
    ("qkv_gate", 2): [((1, 4, 16, 2, 1, 0), (1,)), ((1, 4, 8, 4, 1, 2), (3,))],
    ("o", 1): [((1, 8, 2, 1, 1, 0), (1, 3))],
    ("o", 2): [((1, 8, 2, 1, 1, 0), (1,))],
    ("pooler", 1): [((2, 16, 8, 1, 1, 0), (3,)), ((2, 8, 16, 1, 1, 0), (0,))],
    ("pooler", 2): [((2, 8, 16, 1, 1, 0), (1,))],
    ("pooler", 4): [((1, 8, 16, 1, 1, 2), (0, 1))],
    ("unpooler_s", 1): [((1, 16, 4, 1, 1, 0), (1, 0))],
    ("unpooler", 2): [((1, 4, 16, 1, 1, 2), (1, 0))],
    ("unpooler", 4): [((1, 4, 16, 1, 1, 2), (1, 0)), ((2, 4, 16, 1, 1, 2), (1,))],
    ("router", 1): [((1, 8, 2, 8, 1, 0), (0,)), ((1, 8, 1, 16, 1, 0), (0,))],
    ("router", 2): [((1, 8, 2, 8, 1, 0), (0,))],
    ("router", 4): [((1, 4, 2, 16, 1, 0), (0,)), ((1, 8, 2, 8, 1, 0), (0,))],
}
P_GVI = P__gvinst()
P16 = True


def P__cublas(layer, m):
    return layer != "router" and m in (32, 128, 256)


def P__lt(x, w, wt, out, n, i):
    P__ext().lt_mm(x.contiguous(), w, wt, out, n, i)
    return out


def P__v12(x, layer, n, state):
    m = x.shape[0]
    if layer != "router":
        i = state["lt"].get((layer, m))
        if i is None:
            return x @ state[layer + "_raw"][:n].T
        return P__lt(x, state[layer + "_raw"], state[layer + "_t"], torch.empty((m, n), dtype=torch.bfloat16, device=x.device), n, i)
    out = torch.empty((m, n), dtype=torch.float32, device=x.device)
    A, A2 = state["router"]
    (P__ext().run3 if m <= 32 else P__ext().run)(A, A2, x.contiguous(), out, state["ws"], state["cnt"], n)
    return out


_prepare0, _proj0 = prepare, _proj


def _precompile():
    import threading
    fs = [A__ext, lambda: M__ext(M_LARGE), lambda: M__ext(False), _gext, P__ext] + ([P__gvext] if P16 else [])
    th = [threading.Thread(target=f) for f in fs]
    for t in th:
        t.start()
    for t in th:
        t.join()


def _rkey(T):
    return P__bk(T) if T <= 32 else T


def _rcands(st, T):
    p16, c = st["p16"], {}
    if T <= 32:
        c["rtr"] = lambda h, lg: P__ext().run3(st["rA"], st["rA2"], h, lg, st["pws"], st["pcnt"], 256)
        p = p16["gv"].get(("router", P__bk(T)))
        if p is not None:
            c["gv"] = lambda h, lg: P__gvcall(p16, p, h, lg)
    else:
        c["skinny"] = lambda h, lg: P__ext().run(st["rA"], st["rA2"], h, lg, st["pws"], st["pcnt"], 256)
        i = p16["lt"].get(("router", T))
        if i is not None:
            def lt(h, lg):
                P__ext().lt_side(1)
                a = P__lt(h, p16["router_hl"], p16["router_hl_t"], torch.empty((T, 512), dtype=torch.float32, device=h.device), 512, i)
                P__ext().lt_side(0)
                torch.add(a[:, :256], a[:, 256:], out=lg)
            c["lt"] = lt
    return c


def _rsel(st):
    """router choice per token bucket, timed as the layer runs it: on the side stream, overlapping the pooler"""
    flush = torch.empty(64 << 20, dtype=torch.float32, device="cuda")
    st["rsel"] = {}
    for T in (1, 8, 16, 32, 128, 256):
        c = _rcands(st, T)
        if len(c) < 2:
            continue
        h = torch.randn((T, 8192), device="cuda").to(torch.bfloat16)
        lg = torch.empty((T, 256), dtype=torch.float32, device="cuda")
        def pair(f):
            def run():
                cur, side = torch.cuda.current_stream(), st["side"]
                side.wait_stream(cur)
                with torch.cuda.stream(side):
                    f(h, lg)
                _proj(h, st, "pooler", 4096)
                cur.wait_stream(side)
            return run
        ref = (h.double() @ st["router"].double().T).float()
        ts = {}
        for k, f in c.items():
            f(h, lg); torch.cuda.synchronize()
            if not bool(torch.isfinite(lg).all()) or float((lg - ref).norm() / ref.norm()) > 1e-4:
                continue
            ts[k] = min(P__tm(pair(f), flush, 21) for _ in range(2))
        if ts:
            st["rsel"][T] = min(ts, key=ts.get)
        print("router T%d %s -> %s" % (T, " ".join("%s:%.2f" % (k, v * 1e3) for k, v in ts.items()), st["rsel"].get(T)), flush=True)
    del flush


def prepare(weights):
    import time
    t = time.time()
    _precompile()
    print("compile %.1fs" % (time.time() - t), flush=True)
    st = _prepare0(weights)
    if P16:
        t = time.time()
        st["p16"] = P_prepare({k: weights[k] for k in ("qkv_gate", "o", "pooler", "unpooler", "router")})
        _rsel(st)
        torch.cuda.synchronize()
        print("p19 prepare %.1fs gv %s lt %s" % (time.time() - t, sorted("%s/%d" % k for k in st["p16"]["gv"]), sorted("%s/%d" % k for k in st["p16"]["lt"])), flush=True)
    return st


def _proj(x, st, name, n):
    if not P16:
        return _proj0(x, st, name, n)
    T = x.shape[0]
    p = st["p16"]["gv"].get((name, P__bk(T))) if T <= 32 else None
    if p is None or p[4] != n:
        return P__v12(x, name, n, st["p16"])
    out = torch.empty((T, n), dtype=torch.bfloat16, device=x.device)
    P__gvcall(st["p16"], p, x.contiguous(), out)
    return out


def _router(h, lg, st):
    T = h.shape[0]
    k = st["rsel"].get(_rkey(T)) if P16 and (T <= 32 or T in (128, 256)) else None
    if k is not None:
        _rcands(st, T)[k](h, lg)
    else:
        (P__ext().run3 if T <= 32 else P__ext().run)(st["rA"], st["rA2"], h, lg, st["pws"], st["pcnt"], 256)
