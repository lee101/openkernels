"""Calibrate model constants against H200 measurements; report per-case error (fit, leave-one-out, leave-one-group-out)."""
import json, math, sys
import numpy as np
from scipy.optimize import least_squares
from . import model
from .cases import cases, deltas

FIT = ["bw_hbm", "bw_l2", "lat", "t_kernel", "t_sync", "t_graph", "fstraight", "bw_l2_sm"]  # ipc, nsm, icache pinned from H200 measurements (see model.P0)
BOUNDS = {"bw_hbm": (3.0e12, 4.8e12), "bw_l2": (3e12, 20e12), "lat": (0.3, 2.5), "t_kernel": (0.3, 4), "t_sync": (0.2, 6),
          "ipc": (0.2, 1.0), "nsm": (1.5, 30.0), "t_graph": (0.0, 3.0), "icache": (0.0, 3.0), "fstraight": (0.0, 1.5), "bw_l2_sm": (30e9, 400e9)}
SCALE = {"bw_hbm": 1e12, "bw_l2": 1e12, "bw_l2_sm": 1e9}


PRIOR = {"bw_hbm": (4.2e12, 0.08), "bw_l2": (7e12, 0.5), "lat": (0.9, 0.3), "t_kernel": (1.5, 0.5), "t_sync": (1.5, 0.5),
         "ipc": (0.9, 0.2), "nsm": (4.0, 0.7), "t_graph": (1.0, 0.5), "icache": (0.4, 1.0), "fstraight": (0.5, 0.7), "bw_l2_sm": (117e9, 0.5)}  # (center, log-width): weak MAP priors from pod notes
PRIOR_W = 0.15


def pred1(c, p):
    if c.get("mode") == "graph-empty":  # graph replay minus an empty graph: no per-call or launch fixed cost
        return model.predict(c["ks"], p, call=False) - p["t_kernel"] * sum(k.launches for k in c["ks"])
    return model.predict(c["ks"], p)


def pred_all(C, p): return np.array([pred1(c, p) for c in C])


def fit(C, p0, keys=FIT):
    x0 = [p0[k] / SCALE.get(k, 1) for k in keys]
    lo = [BOUNDS[k][0] / SCALE.get(k, 1) for k in keys]; hi = [BOUNDS[k][1] / SCALE.get(k, 1) for k in keys]
    meas = np.array([c["meas"] for c in C])
    ng = {}
    for c in C: ng[c["group"]] = ng.get(c["group"], 0) + 1
    w = np.array([1 / math.sqrt(ng[c["group"]]) for c in C]) * math.sqrt(len(C) / len(ng))  # each family weighs the same
    def res(x):
        p = dict(p0); p.update({k: v * SCALE.get(k, 1) for k, v in zip(keys, x)})
        reg = [PRIOR_W * math.log(max(p[k], 1e-6) / PRIOR[k][0]) / PRIOR[k][1] for k in keys if k in PRIOR]
        return np.concatenate([w * np.log(np.maximum(pred_all(C, p), 1e-3) / meas), reg])
    r = least_squares(res, np.clip(x0, lo, hi), bounds=(lo, hi), loss="linear")
    p = dict(p0); p.update({k: v * SCALE.get(k, 1) for k, v in zip(keys, r.x)})
    return p


def stats(err):
    a = np.abs(err)
    return f"median |err| {np.median(a) * 100:.1f}%  mean {np.mean(a) * 100:.1f}%  max {np.max(a) * 100:.1f}%  within 20%: {np.mean(a < 0.2) * 100:.0f}%"


def main(argv):
    C = cases(); D = deltas()
    p0 = dict(model.P0)
    do_fit = "--fit" in argv
    hold = set(a.split("=", 1)[1] for a in argv if a.startswith("--hold=")) if any(a.startswith("--hold=") for a in argv) else {"moe"}
    hold.discard("none")
    CF = [c for c in C if c["group"] not in hold]  # held-out families are validation only (MoE: known unmodeled per-stage stalls)
    p = fit(CF, p0) if do_fit else p0
    pr = pred_all(C, p); meas = np.array([c["meas"] for c in C]); err = pr / meas - 1
    loo = logo = None
    if "--cv" in argv:
        loo = np.array([pred1(c, fit([d for d in CF if d is not c], p)) / c["meas"] - 1 for c in C])
        groups = sorted(set(c["group"] for c in C))
        gp = {g: fit([c for c in CF if c["group"] != g], p) for g in groups}
        logo = np.array([pred1(c, gp[c["group"]]) / c["meas"] - 1 for c in C])
    rows = []
    for i, c in enumerate(C):
        _, parts = model.kernel_time(c["ks"][0] if len(c["ks"]) == 1 else max(c["ks"], key=lambda k: model.kernel_time(k, p)), p, True)
        bound = max(parts, key=lambda x: x[0])[1]["bound"]
        rows.append([c["group"], c["name"], c["meas"], pr[i], err[i], loo[i] if loo is not None else None, logo[i] if logo is not None else None, bound])
    print(f"{'group':10} {'case':28} {'meas us':>8} {'pred us':>8} {'err':>7} {'LOO':>7} {'LOGO':>7}  bound")
    for g, n, m, q, e, l, lg, b in rows:
        print(f"{g:10} {n:28} {m:8.1f} {q:8.1f} {e * 100:6.1f}% " + (f"{l * 100:6.1f}% {lg * 100:6.1f}%" if l is not None else "") + f"  {b}")
    print(f"fit on {len(CF)} cases (held out: {', '.join(sorted(hold)) or 'none'})")
    print("fit:", stats(err))
    if loo is not None: print("LOO:", stats(loo)); print("LOGO:", stats(logo))
    for g in sorted(set(c["group"] for c in C)):
        idx = [i for i, c in enumerate(C) if c["group"] == g]
        print(f"  {g:10} n={len(idx):2} fit {np.median(np.abs(err[idx])) * 100:5.1f}%" + (f"  LOGO {np.median(np.abs(logo[idx])) * 100:5.1f}%" if logo is not None else ""))
    print("\nrelative (variant/base ratio):")
    de = []
    for d in D:
        r = model.predict(d["b"], p) / model.predict(d["a"], p); de.append(r / d["meas"] - 1)
        print(f"  {d['name']:52} meas {d['meas']:.3f} pred {r:.3f}")
    print("params:", json.dumps({k: (f"{v:.3g}") for k, v in p.items()}))
    if do_fit and "--save" in argv:
        (model._PJ).write_text(json.dumps({k: p[k] for k in FIT}, indent=1)); print("saved", model._PJ)
    if "--json" in argv:
        out = {"params": p, "rows": [dict(zip(["group", "case", "meas", "pred", "err", "loo", "logo", "bound"], r)) for r in rows],
               "deltas": [{"name": d["name"], "meas": d["meas"], "pred": model.predict(d["b"], p) / model.predict(d["a"], p)} for d in D]}
        print(json.dumps(out, default=float))


if __name__ == "__main__":
    main(sys.argv[1:])
