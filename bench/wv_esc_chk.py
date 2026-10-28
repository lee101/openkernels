"""validate the wgv (v26/v27) packs + per-consumer-warp escape streams by emulating the kernel's two-half decode walk in torch
(router: hi + lo units, matrix bit 9; C12 0: plain BF16 fragments). python bench/wv_esc_chk.py muse_bf16_proj/submission_v27diag.py"""
import importlib.util, os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../muse_bf16_proj")
spec = importlib.util.spec_from_file_location("s", sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
import reference as ref
ws = ref.generate_weights(ref.WEIGHTS_SEED)
order = [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 0, 1), (0, 1, 0), (0, 1, 1), (1, 1, 0), (1, 1, 1)]


def unnat(v8, N, K):  # [T, KC, 4, 32, 8] int values -> [N, K]
    KC = K // 64
    v8 = v8.view(N // 16, KC, 4, 8, 4, 8)
    W = torch.empty((N // 16, 2, 8, KC, 4, 2, 4, 2), dtype=torch.int32, device=v8.device)
    for idx, (hf, h, ee) in enumerate(order):
        W[:, hf, :, :, :, h, :, ee] = v8[..., idx].permute(0, 3, 1, 2, 4)
    return W.view(N, K) & 0xffff


only = sys.argv[2] if len(sys.argv) > 2 else None
for (key, bm), geos in m.WVB.items():
    if only and key != only:
        continue
    w = ws[key]
    N, K = w.shape
    KC = K // 64
    for geo in geos:
        st = {"wq": {}}
        plan = m._wvplan(st, key, w, geo)
        if plan is None:
            print(key, bm, geo, "plan None"); continue
        TN, QS, S, NWG, HL, C12 = geo[:6]
        BM, tpb, NA = 64 * NWG, NWG * 4, 2 if HL else 1
        pk, e = plan[1], plan[2]
        mats = [w.to(torch.bfloat16), (w - w.to(torch.bfloat16).float()).to(torch.bfloat16)] if HL else [w]
        oks = []
        if not C12:
            for h in range(NA):
                raw = pk[:, :, h].reshape(N // BM, KC, tpb, 2048).permute(0, 2, 1, 3).reshape(N // 16, KC, 4, 32, 8, 2)
                v = (raw[..., 0].int() | (raw[..., 1].int() << 8)).view(N // 16, KC, 4, 32, 8)
                oks.append(torch.equal(unnat(v, N, K), mats[h].view(torch.int16).int() & 0xffff))
            print(key, bm, geo, "raw exact", all(oks), flush=True)
            continue
        tabs = [plan[3], plan[6]]
        vals = []
        for h in range(NA):
            data = pk[:, :, h].reshape(N // BM, KC, tpb, 3, 32, 16).permute(0, 2, 1, 3, 4, 5).reshape(N // 16, KC, 3, 32, 16)
            vals.append(m._unc12(data.contiguous(), tabs[h], torch.zeros((0, 6), dtype=torch.long, device=w.device), 1))
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
                                upd.append(((nx >> 9) & 1, rb * tpb + wi, c0s[ks] + cc, rr // 4, (nx >> 15) & 31, (rr % 4) * 2 + ((nx >> 10) & 1), nx & 0x7f))
                                ep += 1
                                nx = lst[ep] if ep < 32 else 0xffffffff
                    assert ep == 32 or nx == 0xffffffff, ("unconsumed", key, geo, rb, ks, wi)
        if upd:
            u = torch.tensor(upd, device=w.device)
            for h in range(NA):
                uh = u[u[:, 0] == h]
                vals[h][uh[:, 1], uh[:, 2], uh[:, 3], uh[:, 4], uh[:, 5]] ^= uh[:, 6] << 8
        for h in range(NA):
            oks.append(torch.equal(unnat(vals[h], N, K), mats[h].view(torch.int16).int() & 0xffff))
        print(key, bm, geo, "applied", len(upd), "exact", all(oks), flush=True)
