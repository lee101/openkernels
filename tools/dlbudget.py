"""per-load component budget: dl judge profile (ours, per kernel -> component) vs standalone SGLang component baselines
usage: dlbudget.py <dl profile .txt> <attention .txt> <moe .txt> <bf16_proj .txt>  (okc profiles/submission_*.txt)"""
import re, sys
dl, att, moe, prj = sys.argv[1:5]


def table(path):
    rows = {}
    for l in open(path):
        m = re.search(r"load=(\w+ \d+).*?\s([\d.]+)\s+([\d.]+)\s+[\d.]+x", l)
        if m:
            lay = re.search(r"layer=(\w+)", l)
            rows.setdefault((m.group(1), lay.group(1) if lay else ""), []).append((float(m.group(2)), float(m.group(3))))
    return rows


def comp(k):
    if re.match(r"(void )?(moe_|xgather)", k): return "moe"
    if re.match(r"(void )?(fdec|attn|prep|comb)", k): return "attn"
    if re.match(r"(void )?(resid|rmsg|route|l2pf|spin)", k): return "glue"
    return "proj"


dlt, A, M, P = table(dl), table(att), table(moe), table(prj)
prof, cur = {}, None
for l in open(dl):
    m = re.match(r"profile \d+: .*layer=(\w+) load=(\w+ \d+)", l)
    if m: cur = (m.group(2), m.group(1)); prof[cur] = {}; continue
    m = re.match(r"(.+?)\s+([\d.]+) µs\s+[\d.]+%\s+\d+$", l.rstrip())
    if cur and m and not l.startswith(("total", "GPU busy")):
        c = comp(m.group(1)); prof[cur][c] = prof[cur].get(c, 0.0) + float(m.group(2))
print(f"{'load':18} {'sgl':>6} {'ours':>6} | {'attn s/o':>11} {'moe s/o':>11} {'proj s/o':>11} {'glue s/o':>11}")
for (ld, lay), v in dlt.items():
    s, o = v[0]
    sa = A.get((ld, lay), [(0, 0)])[0][0]
    sm = M.get((ld, ""), [(0, 0)])[0][0]
    sp = sum(P[(ld, x)][0][0] for x in ("qkv_gate", "o", "pooler", "unpooler"))  # router overlaps the pooler (side stream) in both
    sg = s - sa - sm - sp
    pr = prof.get((ld, lay), {})
    print(f"{ld + ' ' + lay[:2]:18} {s:6.0f} {o:6.0f} | {sa:5.0f}/{pr.get('attn', 0):5.0f} {sm:5.0f}/{pr.get('moe', 0):5.0f} {sp:5.0f}/{pr.get('proj', 0):5.0f} {sg:5.0f}/{pr.get('glue', 0):5.0f}")
