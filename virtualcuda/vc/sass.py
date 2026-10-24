"""SASS extraction: compile CUDA (or a submission .py's CUDA strings) for sm_90a, parse cuobjdump SASS,
classify every instruction by H200 pipe, find loops (backward branches) and report per-loop mixes.
Static counts: conditional blocks inside a loop are counted as if always executed."""
import hashlib, json, os, re, subprocess, sys, sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache"
CUDA = next(Path(c) for c in (os.environ.get("VC_CUDA", ""), "/usr/local/cuda-12", os.environ.get("CUDA_HOME", ""), "/usr/local/cuda") if c and Path(c, "bin/nvcc").exists())

# per-warp cost in cycles on its SM sub-partition (scheduler), Hopper; smem/tensor costs are SM-wide (see model.py)
PIPE = [  # (regex on base opcode, pipe)
    (r"^[HQIB]GMMA$", "wgmma"), (r"^(HMMA|IMMA|BMMA|DMMA)$", "mma"),
    (r"^(FFMA|FADD|FMUL|FFMA32I|FADD32I|FMUL32I|HFMA2|HADD2|HMUL2|HFMA2MMA|FSWZADD|FCHK)$", "fma"),
    (r"^(IMAD|IMUL|IMADSP|IMAD32I|IMUL32I|IDP|IDP4A|VIMNMX3)$", "imad"),
    (r"^(MUFU)$", "mufu"),
    (r"^(F2F|F2I|I2F|I2FP|F2IP|FRND|DADD|DFMA|DMUL)$", "xu"),
    (r"^(F2FP)$", "f2fp"),
    (r"^(LDS|STS|LDSM|STSM|ATOMS|LD|ST)$", "smem"),
    (r"^(LDG|STG|LDGSTS|ATOM|ATOMG|RED|REDG|LDC|CCTL|LDL|STL|LDGDEPBAR|ULDC|ULDG)$", "gmem"),
    (r"^(SHFL)$", "shfl"),
    (r"^(UTMALDG|UTMASTG|UTMAPF|UTMAREDG|UBLKCP|UBLKPF|UBLKRED|UTMACCTL|UTMACMDFLUSH)$", "tma"),
    (r"^(SYNCS)$", "mbar"),
    (r"^(BAR|BARRIER|MEMBAR|FENCE|ERRBAR|DEPBAR|WARPGROUP|WARPSYNC|NANOSLEEP|ACQBULK|ELECT|ARRIVES)$", "sync"),
    (r"^(BRA|BRX|JMP|JMX|CALL|RET|EXIT|BSSY|BSYNC|BREAK|BPT|YIELD|KILL|NOP|WARPGROUPSET|ACQSHMINIT)$", "ctrl"),
    (r"^U", "uniform"),
]
PIPE_RE = [(re.compile(p), n) for p, n in PIPE]
# cycles per warp-instruction on one sub-partition (32 lanes); ALU int/logic is 16 lanes on Hopper
SUB_COST = {"fma": 1, "imad": 2, "alu": 2, "mufu": 8, "xu": 8, "f2fp": 2, "shfl": 2, "uniform": 1, "ctrl": 1, "sync": 1,
            "gmem": 1, "tma": 1, "mbar": 1, "smem": 1, "mma": 0, "wgmma": 0}


def classify(op):
    base = op.split(".")[0]
    for r, n in PIPE_RE:
        if r.match(base): return n
    return "alu"


def mma_cycles(op):
    """SM-wide tensor cycles of one warp-instruction (H200: 4096 dense bf16 flop/clk/SM; mma.sync measured ~66% of that)."""
    base = op.split(".")[0]
    if base in ("HGMMA", "QGMMA", "IGMMA", "BGMMA"):
        m = re.search(r"\.64x(\d+)x(\d+)", op)
        n = int(m.group(1)) if m else 64
        return n / 2 / 4  # one warpgroup = 4 warps share the instruction; whole-WG op takes N/2 SM cycles
    if base == "HMMA":
        if ".16816" in op: return 4096 / 2700
        if ".1688" in op: return 2048 / 2700
        return 4096 / 2700
    if base == "IMMA":
        return (8192 if ".16832" in op else 4096) / 5400
    return 0.0


def smem_bytes(op):
    base = op.split(".")[0]
    if base == "LDSM":
        m = re.search(r"\.M88\.(\d)", op); return 128 * int(m.group(1)) if m else 512
    if base in ("LDS", "STS", "LD", "ST", "STSM", "ATOMS"):
        m = re.search(r"\.(128|64|U8|S8|U16|S16|32)\b", op)
        w = {"128": 16, "64": 8, "U8": 1, "S8": 1, "U16": 2, "S16": 2}.get(m.group(1) if m else "", 4)
        return 32 * w
    return 0


def extract_cuda(py):
    src = Path(py).read_text()
    parts = re.findall(r'^\w*CUDA\w* = r"""\n(.*?)"""', src, re.S | re.M)
    if not parts: sys.exit(f"no CUDA strings in {py}")
    return "\n".join(parts)


def compile_cubin(src, defs=(), arch="sm_90a", extra=()):
    """src: .cu or .py path. Returns cubin path (cached by content+flags)."""
    p = Path(src)
    code = extract_cuda(p) if p.suffix == ".py" else p.read_text()
    flags = [f"-gencode=arch=compute_{arch[3:]},code={arch}", "-O3", "--use_fast_math", "-lineinfo", "-std=c++17", *defs, *extra]
    h = hashlib.sha1((code + " ".join(flags)).encode()).hexdigest()[:16]
    CACHE.mkdir(exist_ok=True)
    cu, cub = CACHE / f"{p.stem}.{h}.cu", CACHE / f"{p.stem}.{h}.{arch}.cubin"
    if cub.exists(): return cub
    cu.write_text(code)
    inc = []
    if "torch/extension.h" in code or "ATen" in code:
        import torch.utils.cpp_extension as ce
        inc = [f"-I{d}" for d in ce.include_paths()] + [f"-I{sysconfig.get_paths()['include']}"]
    cmd = [str(CUDA / "bin/nvcc"), "-cubin", "-o", str(cub), *flags, *inc, "-diag-suppress", "607,1886,177,550", str(cu)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode: sys.exit("nvcc failed:\n" + r.stderr[-3000:])
    return cub


LINE = re.compile(r"^\s+/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;\s*(/\*\s*0x[0-9a-f]+\s*\*/)?\s*$")
HI = re.compile(r"^\s+/\*\s*0x([0-9a-f]{16})\s*\*/\s*$")


def ctrl(hi):
    """Volta+ control bits (high word bits 41..61): stall, yield, write/read scoreboard, wait mask, reuse."""
    return {"stall": (hi >> 41) & 0xF, "yld": (hi >> 45) & 1, "wbar": (hi >> 46) & 7, "rbar": (hi >> 49) & 7, "wait": (hi >> 52) & 0x3F}


def parse(cubin):
    """-> {func: {"instrs": [(pc, pred, op, args)], "regs":, "smem":}}"""
    out = subprocess.run([str(CUDA / "bin/cuobjdump"), "-sass", str(cubin)], capture_output=True, text=True).stdout
    funcs, cur = {}, None
    for l in out.splitlines():
        m = re.match(r"\s+Function : (\S+)", l)
        if m: cur = funcs.setdefault(m.group(1), {"instrs": []}); continue
        m = LINE.match(l)
        if m and cur is not None:
            body = m.group(2)
            pm = re.match(r"(@!?U?P[T0-9]+)\s+(.*)", body)
            pred, body = (pm.group(1), pm.group(2)) if pm else ("", body)
            op, _, args = body.partition(" ")
            cur["instrs"].append((int(m.group(1), 16), pred, op, args.strip()))
            continue
        h = HI.match(l)
        if h and cur is not None and cur["instrs"] and len(cur["instrs"][-1]) == 4:
            cur["instrs"][-1] = cur["instrs"][-1] + (ctrl(int(h.group(1), 16)),)
    res = subprocess.run([str(CUDA / "bin/cuobjdump"), "-res-usage", str(cubin)], capture_output=True, text=True).stdout
    for f, r in re.findall(r"Function (\S+):\s*\n\s*(REG:.*)", res):
        if f in funcs:
            funcs[f]["regs"] = int(re.search(r"REG:(\d+)", r).group(1)); funcs[f]["smem"] = int(re.search(r"SHARED:(\d+)", r).group(1))
    return funcs


def demangle(names):
    r = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True, text=True)
    return dict(zip(names, r.stdout.splitlines())) if r.returncode == 0 else {n: n for n in names}


# variable-latency producers (scoreboard waits), cycles; fixed-latency ops are covered by the encoded stall counts
VARLAT = {"smem": 33, "gmem": 250, "mufu": 18, "xu": 18, "shfl": 24, "mma": 32, "mbar": 40, "tma": 40, "sync": 30, "alu": 12, "f2fp": 12,
          "fma": 12, "imad": 12, "uniform": 12, "ctrl": 12, "wgmma": 0}
WGMMA_WAIT = 96  # WARPGROUP.DEPBAR after an issued GMMA (64xNx16, N=64)


def warp_cycles(instrs, iters=3):
    """single-warp in-order time per loop iteration (steady state of `iters` passes): encoded stalls + scoreboard waits."""
    if not instrs or len(instrs[0]) < 5: return 0
    t, ready, marks, gm = 0, [0] * 6, [], False
    for _ in range(iters):
        marks.append(t)
        for ins in instrs:
            pc, pred, op, args, c = ins
            ti = max([t] + [ready[b] for b in range(6) if c["wait"] >> b & 1])
            b = op.split(".")[0]
            if b in ("HGMMA", "QGMMA", "IGMMA", "BGMMA"): gm = True
            if op.startswith("WARPGROUP.DEPBAR") and gm: ti += WGMMA_WAIT; gm = False
            if c["wbar"] < 6:
                lat = VARLAT.get(classify(op), 12)
                if b == "LDG" and "LDGSTS" not in op: lat = VARLAT["gmem"]
                ready[c["wbar"]] = ti + lat
            if c["rbar"] < 6: ready[c["rbar"]] = max(ready[c["rbar"]], ti + 12)
            t = ti + max(c["stall"], 1)
    return (t - marks[1]) / (iters - 1) if iters > 1 else t


def mix(instrs):
    """pipe counts + derived per-warp cost numbers for a list of instrs."""
    c, ops = {}, {}
    tensor = smem = 0.0
    for ins in instrs:
        op = ins[2]
        p = classify(op); c[p] = c.get(p, 0) + 1
        b = op.split(".")[0]; ops[b] = ops.get(b, 0) + 1
        tensor += mma_cycles(op); smem += smem_bytes(op)
    sub = sum(SUB_COST.get(p, 2) * n for p, n in c.items() if p not in ("fma", "imad", "alu", "f2fp", "mufu", "xu"))
    pipes = {p: SUB_COST[p] * c.get(p, 0) for p in ("fma", "imad", "alu", "f2fp", "mufu", "xu", "shfl")}
    return {"n": len(instrs), "warp_cycles": round(warp_cycles(instrs), 1), "stall_sum": sum(i[4]["stall"] for i in instrs if len(i) > 4), "pipes": c, "top": dict(sorted(ops.items(), key=lambda x: -x[1])[:12]),
            "issue": len(instrs), "pipe_cycles": pipes, "tensor_sm_cycles": round(tensor, 2), "smem_bytes": smem}


def cfg(instrs):
    """basic blocks + successor edges from cuobjdump SASS (absolute branch targets)."""
    idx = {ins[0]: i for i, ins in enumerate(instrs)}
    lead = {0}
    tgt = {}
    for i, (pc, pred, op, args, *_) in enumerate(instrs):
        b = op.split(".")[0]
        if b in ("BRA", "EXIT", "RET", "BRX", "JMP"):
            m = re.search(r"0x([0-9a-f]+)", args) if b == "BRA" else None
            if m and int(m.group(1), 16) in idx: tgt[i] = idx[int(m.group(1), 16)]; lead.add(tgt[i])
            if i + 1 < len(instrs): lead.add(i + 1)
    lead = sorted(lead)
    blocks = [(a, (lead[k + 1] if k + 1 < len(lead) else len(instrs)) - 1) for k, a in enumerate(lead)]
    bid = {a: k for k, (a, _) in enumerate(blocks)}
    succ = []
    for a, e in blocks:
        pc, pred, op, args = instrs[e][:4]; b = op.split(".")[0]; s_ = []
        if e in tgt: s_.append(bid[tgt[e]])
        uncond = b in ("EXIT", "RET", "BRX", "JMP") or (b == "BRA" and (not pred or pred == "@PT") and "P" not in args.split(",")[0])
        if not uncond and e + 1 < len(instrs): s_.append(bid[e + 1])
        if b == "EXIT" and pred and e + 1 < len(instrs): s_ = [bid[e + 1]]
        succ.append(s_)
    return blocks, succ


def longest_path(instrs, inloop):
    """longest entry->exit path (1-warp cycles) through the CFG outside loops: alternatives (if/else, template-uniform
    runtime branches) count once instead of summed. Loop bodies are excluded (modeled per iteration)."""
    blocks, succ = cfg(instrs)
    n = len(blocks)
    w = []
    for a, e in blocks:
        body = [instrs[i] for i in range(a, e + 1) if i not in inloop]
        w.append(warp_cycles(body, iters=1) if body else 0)
    order, seen, st = [], [0] * n, [(0, 0)]
    while st:  # iterative DFS post-order, ignoring back edges
        u, i = st.pop()
        if i == 0:
            if seen[u]: continue
            seen[u] = 1
        if i < len(succ[u]):
            st.append((u, i + 1)); v = succ[u][i]
            if not seen[v]: st.append((v, 0))
        else:
            seen[u] = 2; order.append(u)
    pos = {u: k for k, u in enumerate(order)}
    best = [0.0] * n
    for u in order:  # reverse topological via post-order: successors first
        nxt = [best[v] for v in succ[u] if pos.get(v, -1) < pos[u]]
        best[u] = w[u] + (max(nxt) if nxt else 0)
    return best[0]


def loops(instrs):
    """natural loops (back edges to a dominating header) -> [{start, end, depth, blocks}]"""
    blocks, succ = cfg(instrs)
    n = len(blocks)
    pred = [[] for _ in range(n)]
    for u, ss in enumerate(succ):
        for v in ss: pred[v].append(u)
    reach = [False] * n; st = [0]; reach[0] = True
    while st:
        u = st.pop()
        for v in succ[u]:
            if not reach[v]: reach[v] = True; st.append(v)
    full = set(i for i in range(n) if reach[i])
    dom = [set(full) if reach[i] else set() for i in range(n)]; dom[0] = {0}
    changed = True
    while changed:
        changed = False
        for v in range(1, n):
            if not reach[v]: continue
            ps = [dom[u] for u in pred[v] if reach[u]]
            nd = (set.intersection(*ps) if ps else set()) | {v}
            if nd != dom[v]: dom[v] = nd; changed = True
    L = {}
    for u in range(n):
        for h in succ[u]:
            if reach[u] and h in dom[u]:
                body = {h, u}; st = [u]
                while st:
                    x = st.pop()
                    if x == h: continue
                    for q in pred[x]:
                        if q not in body and reach[q]: body.add(q); st.append(q)
                L.setdefault(h, set()).update(body)
    out = []
    for h, body in L.items():
        depth = sum(1 for h2, b2 in L.items() if h2 != h and body < b2)
        inner = sum(1 for h2, b2 in L.items() if h2 != h and b2 < body)
        idxs = sorted(i for b in body for i in range(blocks[b][0], blocks[b][1] + 1))
        big = sum(1 for h2, b2 in L.items() if h2 != h and b2 < body and sum(blocks[b][1] - blocks[b][0] + 1 for b in b2) >= 20)
        out.append({"start": instrs[blocks[h][0]][0], "end": max(instrs[i][0] for i in idxs), "depth": depth, "inner": inner, "inner_big": big, "idx": idxs})
    return sorted(out, key=lambda x: x["start"])


def analyze(src, defs=(), arch="sm_90a", func=None, extra=()):
    cub = compile_cubin(src, defs, arch, extra)
    funcs = parse(cub)
    dm = demangle(list(funcs))
    res = []
    for f, d in funcs.items():
        name = dm.get(f, f)
        if func and not re.search(func, name) and not re.search(func, f): continue
        ins = d["instrs"]
        # drop trailing BRA-to-self + NOP padding
        while ins and ins[-1][2] in ("NOP",) or (ins and ins[-1][2] == "BRA" and str(hex(ins[-1][0]))[2:] in ins[-1][3]): ins = ins[:-1]
        ent = {"func": name, "mangled": f, "regs": d.get("regs"), "smem_static": d.get("smem"), "total": mix(ins), "loops": []}
        L = loops(ins)
        inloop = set(i for lp in L for i in lp["idx"])
        straight = [ins[i] for i in range(len(ins)) if i not in inloop]
        ent["straight"] = {"n": len(straight), "warp_cycles": round(warp_cycles(straight, iters=1), 1), "path_cycles": round(longest_path(ins, inloop), 1)}
        for lp in L:
            body = [ins[i] for i in lp.pop("idx")]
            ent["loops"].append({**lp, "start": hex(lp["start"]), "end": hex(lp["end"]), **mix(body)})
        res.append(ent)
    return res


def fmt(res, top_loops=6, min_n=20):
    out = []
    for e in res:
        t = e["total"]
        out.append(f"== {e['func'][:110]}\n   regs {e['regs']} smem_static {e['smem_static']} | static instrs {t['n']} | {pipes_str(t['pipes'])}\n"
                   f"   outside loops: {e['straight']['n']} instrs, 1-warp {e['straight']['warp_cycles']:.0f} cyc all paths, {e['straight']['path_cycles']:.0f} cyc longest path")
        for lp in sorted([l for l in e["loops"] if l["n"] >= min_n], key=lambda x: (x["inner_big"] > 0, -x["n"]))[:top_loops]:
            out.append(f"   loop {lp['start']}-{lp['end']} depth {lp['depth']} inner {lp['inner']}/{lp['inner_big']}: {lp['n']} instrs, 1-warp {lp['warp_cycles']:.0f} cyc/iter (issue-saturating warps/sched {lp['warp_cycles'] / max(lp['n'], 1):.1f}), tensor {lp['tensor_sm_cycles']} SM-cyc, "
                       f"smem {lp['smem_bytes']:.0f} B | {pipes_str(lp['pipes'])}\n     top: {' '.join(f'{k}:{v}' for k, v in lp['top'].items())}")
    return "\n".join(out)


def pipes_str(c):
    return " ".join(f"{k}:{v}" for k, v in sorted(c.items(), key=lambda x: -x[1]))


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="vc.py sass", description=__doc__)
    ap.add_argument("src"); ap.add_argument("-D", action="append", default=[]); ap.add_argument("--arch", default="sm_90a")
    ap.add_argument("--func"); ap.add_argument("--json", action="store_true"); ap.add_argument("--loops", type=int, default=6)
    a = ap.parse_args(argv)
    r = analyze(a.src, [f"-D{d}" for d in a.D], a.arch, a.func)
    print(json.dumps(r, indent=1) if a.json else fmt(r, a.loops))


if __name__ == "__main__":
    main()
