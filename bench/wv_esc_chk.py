"""validate the wgv (v26) pack + per-consumer-warp escape streams by emulating the kernel's two-half decode walk in torch.
python bench/wv_esc_chk.py muse_bf16_proj/submission_v26diag.py"""
import importlib.util, os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../muse_bf16_proj")
spec = importlib.util.spec_from_file_location("s", sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
import reference as ref
ws = ref.generate_weights(ref.WEIGHTS_SEED)
order = [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 0, 1), (0, 1, 0), (0, 1, 1), (1, 1, 0), (1, 1, 1)]
for (key, bm), geos in m.WVB.items():
    w = ws[key]
    N, K = w.shape
    KC = K // 64
    for geo in geos:
        st = {"wq": {}}
        plan = m._wvplan(st, key, w, geo)
        if plan is None:
            print(key, bm, geo, "plan None"); continue
        TN, QS, S, NWG = geo
        BM, tpb = 64 * NWG, NWG * 4
        pk, e = plan[1], plan[2]
        c = st["wq"][(key, N)]
        data = pk.view(N // BM, KC, tpb, 3, 32, 16).permute(0, 2, 1, 3, 4, 5).reshape(N // 16, KC, 3, 32, 16)
        val = m._unc12(data.contiguous(), c[1], torch.zeros((0, 6), dtype=torch.long, device=w.device), 1)
        E = (e.view(N // BM, S, tpb, 32).long() & 0xffffffff).cpu()
        c0s = [KC * k // S for k in range(S)]
        upd = []
        for rb in range(N // BM):
            for ks in range(S):
                nk = KC * (ks + 1) // S - c0s[ks]
                for wi in range(tpb):
                    lst = E[rb, ks, wi].tolist()
                    ep, nx = 0, lst[0]
                    for cc in range(nk):
                        for half in (0, 1):
                            while ep < 32 and nx != 0xffffffff and (nx >> 20) == cc and (half or ((nx >> 11) & 15) < 8):
                                rr = (nx >> 11) & 15
                                upd.append((rb * tpb + wi, c0s[ks] + cc, rr // 4, (nx >> 15) & 31, (rr % 4) * 2 + ((nx >> 10) & 1), nx & 0x7f))
                                ep += 1
                                nx = lst[ep] if ep < 32 else 0xffffffff
                    assert ep == 32 or nx == 0xffffffff, ("unconsumed", key, geo, rb, ks, wi)
        if upd:
            u = torch.tensor(upd, device=w.device)
            val[u[:, 0], u[:, 1], u[:, 2], u[:, 3], u[:, 4]] ^= u[:, 5] << 8
        v8 = val.view(N // 16, KC, 4, 8, 4, 8)
        W = torch.empty((N // 16, 2, 8, KC, 4, 2, 4, 2), dtype=torch.int32, device=w.device)
        for idx, (hf, h, ee) in enumerate(order):
            W[:, hf, :, :, :, h, :, ee] = v8[..., idx].permute(0, 3, 1, 2, 4)
        ok = torch.equal(W.view(N, K) & 0xffff, w.view(torch.int16).int() & 0xffff)
        print(key, bm, geo, "escapes", len(c[2]), "applied", len(upd), "exact", ok, flush=True)
