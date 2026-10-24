"""cu.py extract|embed [problem ...]: CUDA sources of the submissions as standalone .cu files (cu/<problem>__<VAR>.cu).
extract: submission.py -> cu/ (for analyzers, reviews, nvcc/cuk). embed: cu/ -> submission.py (submission stays one file).
Each .cu gets a prelude that defines the -D macros the submission passes, so `nvcc -c` works standalone."""
import re, sys
from pathlib import Path
R = Path(__file__).resolve().parent.parent
PROBS = ["muse_attention_decode", "muse_moe_w4a16", "muse_bf16_proj", "muse_decode_layer"]
PAT = re.compile(r'^(\w*CUDA) = r"""\n(.*?)"""', re.S | re.M)
TAG = "// ---- cu.py: end of standalone prelude ----\n"


def extract(p):
    src = (R / p / ("glue.py" if p == "muse_decode_layer" else "submission.py")).read_text()
    for m in PAT.finditer(src):
        out = R / "cu" / f"{p}__{m.group(1)}.cu"
        out.write_text(f"// extracted from {p} ({m.group(1)}); edit then `python3 tools/cu.py embed {p}`\n{TAG}" + m.group(2))
        print("wrote", out.relative_to(R), m.group(2).count("\n"), "lines")


def embed(p):
    f = R / p / ("glue.py" if p == "muse_decode_layer" else "submission.py")
    src = f.read_text()
    for cu in sorted((R / "cu").glob(f"{p}__*.cu")):
        var = cu.stem.split("__", 1)[1]
        body = cu.read_text().split(TAG, 1)[1]
        new, n = re.subn(rf'^{var} = r"""\n.*?"""', lambda _: f'{var} = r"""\n{body}"""', src, flags=re.S | re.M)
        if n != 1: sys.exit(f"{var} not found in {f}")
        src = new
    f.write_text(src)
    print("embedded into", f.relative_to(R))


if __name__ == "__main__":
    cmd, probs = sys.argv[1], sys.argv[2:] or PROBS
    for p in probs: (extract if cmd == "extract" else embed)(p)
