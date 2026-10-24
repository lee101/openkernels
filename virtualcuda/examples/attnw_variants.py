"""Triage example: verify attention (attnw) variants at B=32, 80K full context, using the calibrated builders.
`python3 vc.py est examples/attnw_variants.py [-v]` prints each variant's predicted H200 time and its ratio to the first."""
from vc.cases import attnw, fdec, AW_STEP, AW_T1, AW_TENSOR


def kernels():
    v = {}
    v["v38 baseline (735 instr/step)"] = attnw(32, 80000, False)
    v["-50 SASS/step (e.g. drop per-step row max)"] = attnw(32, 80000, False, step=AW_STEP - 50, t1=AW_T1 - 120)
    v["-150 SASS/step"] = attnw(32, 80000, False, step=AW_STEP - 150, t1=AW_T1 - 300)
    k = attnw(32, 80000, False); k[0].phases[1].g128 /= 2  # e.g. FP8 K only (half the KV bytes) - hypothetical
    v["half the KV bytes"] = k
    k = attnw(32, 80000, False); k[0].phases[1].tensor *= 2
    v["2x tensor work (e.g. bf16 QK instead of fp8)"] = k
    return v
