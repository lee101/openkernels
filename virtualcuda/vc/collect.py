"""Snapshot judge profiles (muse_*/profiles/*.txt, gitignored) into data/h200_measured.json (committed ground truth)."""
import json, re, sys
from pathlib import Path

R = Path(__file__).resolve().parent.parent.parent
OUT = Path(__file__).resolve().parent.parent / "data" / "h200_measured.json"
KEEP = {  # problem: judged files used for calibration
    "muse_attention_decode": ["submission_v38", "submission_v35", "submission_v33"],
    "muse_moe_w4a16": ["submission_v17", "submission_v14", "submission_v9", "submission_v18", "submission_v20"],
    "muse_bf16_proj": ["submission_v11", "submission_v12", "submission_v10"],
    "muse_decode_layer": ["submission_v7", "submission_v9"],
    "muse_prefill_layer": ["submission_v6"],
}


def parse(f):
    t = f.read_text()
    head = dict(re.findall(r"^(submission|speedup|status)\s+(.*)$", t, re.M))
    cases, order = {}, []
    blk = t.split("scored shapes", 1)[1].split("\nprofile 0:", 1)[0] if "scored shapes" in t else ""
    for l in blk.splitlines()[2:]:
        m = re.match(r"^(.*?)\s{2,}([\d.]+)\s+([\d.]+)\s+([\d.]+)x\s+([\d.]+)", l)
        if m:
            cases[m.group(1).strip()] = {"sglang": float(m.group(2)), "yours": float(m.group(3)), "sol": float(m.group(5)), "kernels": []}
            order.append(m.group(1).strip())
    for m in re.finditer(r"^profile (\d+): (.*?)\n(.*?)(?=^profile \d+:|\Z)", t, re.M | re.S):
        name = m.group(2).strip()
        if name not in cases: continue
        for l in m.group(3).splitlines():
            km = re.match(r"^(\S.*?)\s{2,}([\d.]+) µs\s+[\d.]+%\s+(\d+)", l)
            if km: cases[name]["kernels"].append([km.group(1).strip(), float(km.group(2)), int(km.group(3))])
            bm = re.match(r"^GPU busy.*?([\d.]+) µs", l)
            if bm: cases[name]["busy"] = float(bm.group(1))
    return {"speedup": float(head.get("speedup", "nan")), "cases": cases}


def main():
    data = {}
    for prob, vers in KEEP.items():
        for f in sorted((R / prob / "profiles").glob("*.txt")):
            v = f.name.split(".")[0]
            if v in vers:
                d = parse(f)
                if d["cases"]: data.setdefault(prob, {})[f"{v}.{f.name.split('.')[1]}"] = d
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1))
    print("wrote", OUT, sum(len(v) for v in data.values()), "profiles")


if __name__ == "__main__":
    main()
