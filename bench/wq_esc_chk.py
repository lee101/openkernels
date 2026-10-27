"""validate the wgq pack + per-decoder-warp escape streams by emulating the kernel's decoder walk in torch"""
import importlib.util, sys, torch
sys.path.insert(0, "/vfast/data/code/openkernels/.claude/worktrees/agent-aa2d7d5da3e8f8442/muse_bf16_proj")
spec = importlib.util.spec_from_file_location("s", sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
import reference as ref
ws = ref.generate_weights(ref.WEIGHTS_SEED)
for (key, bm), geos in m.WQB.items():
    w = ws[key]
    N, K = w.shape
    KC = K // 64
    for geo in geos:
        st = {"wq": {}}
        plan = m._wqplan(st, key, w, geo)
        if plan is None:
            print(key, bm, geo, "plan None"); continue
        TN, QS, S, NWG = geo
        BM, tpb, U = 64 * NWG, NWG * 4, NWG
        pk, e, T0 = plan[1], plan[2], plan[3]
        c = st["wq"][(key, N)]
        # decode without escapes, then apply the per-warp streams exactly as the kernel walks them
        data = pk.view(N // BM, KC, tpb, 3, 32, 16).permute(0, 2, 1, 3, 4, 5).reshape(N // 16, KC, 3, 32, 16)
        val = m._unc12(data.contiguous(), c[1], torch.zeros((0, 6), dtype=torch.long, device=w.device), 1)  # [T, KC, 4, 32, 8]
        E = e.view(N // BM, S, 4, 32).long() & 0xffffffff
        c0s = [KC * k // S for k in range(S)]
        n_applied = 0
        for rb in range(N // BM):
            for ks in range(S):
                nk = KC * (ks + 1) // S - c0s[ks]
                for dw in range(4):
                    lst = E[rb, ks, dw].tolist()
                    ep = 0
                    nx = lst[0]
                    for cc in range(nk):
                        for i in range(U):
                            while ep < 32 and nx != 0xffffffff and (nx >> 20) == cc * U + i:
                                lane, rr, half, xr = (nx >> 15) & 31, (nx >> 11) & 15, (nx >> 10) & 1, nx & 0x7f
                                uu = dw + 4 * i
                                j, v = rr // 4, (rr % 4) * 2 + half
                                val[rb * tpb + uu, c0s[ks] + cc, j, lane, v] ^= xr << 8
                                n_applied += 1
                                ep += 1
                                nx = lst[ep] if ep < 32 else 0xffffffff
                    assert ep == 32 or nx == 0xffffffff, ("unconsumed", key, geo, rb, ks, dw)
        # val -> W layout: [T, KC, j, g, t, 8] natural-k order
        v8 = val.view(N // 16, KC, 4, 8, 4, 8)
        W = torch.empty((N // 16, 2, 8, KC, 4, 2, 4, 2), dtype=torch.int32, device=w.device)
        order = [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 0, 1), (0, 1, 0), (0, 1, 1), (1, 1, 0), (1, 1, 1)]
        for idx, (hf, h, ee) in enumerate(order):
            W[:, hf, :, :, :, h, :, ee] = v8[..., idx].permute(0, 3, 1, 2, 4)
        ok = torch.equal(W.view(N, K) & 0xffff, w.view(torch.int16).int() & 0xffff)
        print(key, bm, geo, "escapes", len(c[2]), "applied", n_applied, "exact", ok, flush=True)
