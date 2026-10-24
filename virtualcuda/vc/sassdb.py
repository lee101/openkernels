"""SASS-derived features of the judged kernels (loop issue counts, 1-warp cycles/iteration, straight-line cycles, code size),
computed with vc/sass.py and cached in data/sass_db.json so calibration is reproducible without recompiling.
`python3 -m vc.sassdb --refresh` recompiles (needs nvcc + torch headers, ~2 min)."""
import json, statistics, sys
from pathlib import Path
from . import sass

R = Path(__file__).resolve().parent.parent
DB = R / "data" / "sass_db.json"
A38 = ["-DRB=8", "-DRB2=6", "-DRB3=20", "-DRBF=5"]
SRC = {  # name: (source, defs, function regex)
    "attn_v38": ("muse_attention_decode/submission_v38.py", A38, r"^void (attnw|fdec)<"),
    "attn_v35": ("muse_attention_decode/submission_v35.py", A38, r"^void (attnw|fdec)<"),
    "moe_v17_small": ("muse_moe_w4a16/submission_v17.py", ["-DWPC=16", "-DNSTG=2"], r"^moe_kernel"),
    "moe_v17_large": ("muse_moe_w4a16/submission_v17.py", ["-DWPC=8", "-DNSTG=4", "-DNTM=8", "-DSCR=2048"], r"^moe_kernel"),
    "proj_v11": ("muse_bf16_proj/submission_v11.py", ["-DWPC=16"], r"^void (skinny<1, false, 2, 2>|rtr<1>)|^fin"),
}


def build():
    db = {"gmem_latency_cycles": sass.VARLAT["gmem"], "kernels": {}}
    for key, (src, defs, fn) in SRC.items():
        for e in sass.analyze(str(R.parent / src), defs, func=fn):
            db["kernels"][f"{key}:{e['func']}"] = {
                "src": src, "defs": defs, "regs": e["regs"], "code": e["total"]["n"], "straight_cyc": e["straight"]["warp_cycles"], "path_cyc": e["straight"]["path_cycles"],
                "loops": [{k: l[k] for k in ("start", "n", "warp_cycles", "inner_big", "tensor_sm_cycles")} | {"mma": l["pipes"].get("mma", 0), "wgmma": l["pipes"].get("wgmma", 0)} for l in e["loops"]]}
    DB.write_text(json.dumps(db, indent=1))
    print("wrote", DB, len(db["kernels"]), "kernels")


_db = None


def get(name):
    global _db
    if _db is None: _db = json.loads(DB.read_text())
    return _db["kernels"][name]


def loop(name, pick):
    """pick(loop) -> bool; returns (instrs, 1-warp cycles, tensor SM-cycles/warp) averaged over matching innermost loops."""
    L = [l for l in get(name)["loops"] if l["inner_big"] == 0 and pick(l)]
    if not L: raise KeyError((name, "no loop"))
    m = max(l["n"] for l in L); L = [l for l in L if l["n"] >= 0.9 * m]
    return statistics.mean(l["n"] for l in L), statistics.mean(l["warp_cycles"] for l in L), statistics.mean(l["tensor_sm_cycles"] for l in L)


if __name__ == "__main__":
    if "--refresh" in sys.argv: build()
    for k, v in json.loads(DB.read_text())["kernels"].items():
        big = sorted(v["loops"], key=lambda l: -l["n"])[:3]
        print(f"{k[:60]:60} code {v['code']:5} straight {v['straight_cyc']:7.0f} path {v['path_cyc']:7.0f} cyc | " + " ".join(f"[{l['n']}i {l['warp_cycles']:.0f}c mma{l['mma']}/{l['wgmma']}]" for l in big if l["inner_big"] == 0))
