"""Template for a brand-new kernel idea, described from first principles (no code yet).
Fill bytes per memory level, warp-instructions (from a prototype's `vc.py sass` loop count x iterations, or a guess),
tensor SM-cycles (= FLOPs / 4096 for dense bf16 wgmma, / 8192 for fp8), dependent latency chains and grid syncs.

Example: MoE large-M (bs=256) with CTA-shared activation tiles: weights streamed once (523 MB), activation L2 traffic
cut 4x (2.5 GB -> 0.62 GB), wgmma instead of mma.sync, same dequant ALU (~560 instrs per 4.6 KB stage per warp)."""
from vc.model import Kernel, Phase

E, TILES, STAGES = 18.5, 18.5 * 768, 8
DQ = 560  # dequant warp-instrs per stage (vc.py sass on moe_kernel: LOP3/SHF/HMUL2/PRMT)


def kernels():
    cur = Kernel("moe large-M today", code=12120, wfix=6084, phases=[
        Phase("routing", chain=2, l2chain=2),
        Phase("experts", hbm=523e6, l2=2.47e9, issue=TILES * 1.3 * STAGES * 1109, tensor=TILES * 1.3 * STAGES * 388,
              units=TILES * 1.3, par=1056, dynamic=True, unit_cyc=STAGES * 3971, wps=2, unit_sync=1),
        Phase("epilogue", syncs=1, l2chain=3)])
    new = Kernel("moe large-M shared-x + wgmma", code=9000, wfix=6000, phases=[
        Phase("routing", chain=2, l2chain=2),
        Phase("experts", hbm=523e6, l2=0.62e9, issue=TILES * STAGES * (DQ + 120), tensor=TILES * STAGES * 256,
              units=TILES / 4, par=264, dynamic=True, unit_cyc=STAGES * 1500, wps=2, unit_sync=1),  # 4 tiles per CTA share x
        Phase("epilogue", syncs=1, l2chain=3)])
    return {"today (model)": [cur], "idea": [new]}
