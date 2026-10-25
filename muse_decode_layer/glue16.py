P_TUNE = False
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
    ("unpooler", 4): [((1, 4, 16, 1, 1, 2), (1, 0))],
}
P_GVI = P__gvinst()
P16 = True
RTR16 = True


def P__v12(x, layer, n, state):
    if layer != "router":
        return x @ state[layer + "_raw"][:n].T
    out = torch.empty((x.shape[0], n), dtype=torch.float32, device=x.device)
    A, A2 = state["router"]
    (P__ext().run3 if x.shape[0] <= 32 and RTR16 else P__ext().run)(A, A2, x.contiguous(), out, state["ws"], state["cnt"], n)
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


def prepare(weights):
    import time
    t = time.time()
    _precompile()
    print("compile %.1fs" % (time.time() - t), flush=True)
    st = _prepare0(weights)
    if P16:
        t = time.time()
        st["p16"] = P_prepare({k: weights[k] for k in ("qkv_gate", "o", "pooler", "unpooler", "router")})
        torch.cuda.synchronize()
        print("p16 prepare %.1fs plans %s" % (time.time() - t, sorted("%s/%d" % k for k in st["p16"]["gv"])), flush=True)
    return st


def _p16(st, name, T):
    return st["p16"]["gv"].get((name, P__bk(T))) if P16 and T <= 32 else None


def _proj(x, st, name, n):
    T = x.shape[0]
    p = _p16(st, name, T)
    if p is None or p[4] != n:
        return _proj0(x, st, name, n)
    out = torch.empty((T, n), dtype=torch.bfloat16, device=x.device)
    P__gvcall(st["p16"], p, x.contiguous(), out)
    return out


def _router(h, lg, st):
    T = h.shape[0]
    p = _p16(st, "router", T)
    if p is not None:
        P__gvcall(st["p16"], p, h, lg)
    else:
        (P__ext().run3 if T <= 32 and RTR16 else P__ext().run)(st["rA"], st["rA2"], h, lg, st["pws"], st["pcnt"], 256)
