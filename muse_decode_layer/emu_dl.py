# sm_86 check of a composed decode layer (env DL_SUB, default submission_v23.py) with the sm_90a attention paths
# (attnp / attnd / wgmma) emulated (-DWGEMU=1, small ring); DL_EMU=0 runs it native. Prints prepare time and GPU memory.
# usage: DL_SUB=submission_v23.py python3 okb.py muse_decode_layer -s emu_dl.py ...
import importlib.util, os, time
from pathlib import Path
import torch

_spec = importlib.util.spec_from_file_location("dl_emu", Path(__file__).with_name(os.environ.get("DL_SUB", "submission_v23.py")))
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
if os.environ.get("DL_EMU", "1") != "0":
    _m.A__DBG = ["-DWGEMU=1", "-DRSW=2"]
    if os.environ.get("DL_HOPPER") == "1":  # attention dispatch as on sm_90 (e.g. v74 s1k: S1CL off on sm_86 = grid-barrier emulation)
        _m.A__hopper = lambda: True
if os.environ.get("DL_FUSE") is not None and hasattr(_m, "DL_FUSE"):  # v31+: glue fusion on / off
    _m.DL_FUSE = os.environ["DL_FUSE"] != "0"
kernel_fn = _m.kernel_fn


def prepare(w):
    torch.cuda.reset_peak_memory_stats()
    m0, t0 = torch.cuda.memory_allocated(), time.time()
    st = _m.prepare(w)
    torch.cuda.synchronize()
    if os.environ.get("DL_WX"):  # force the MoE wx path (e.g. DL_WX=-DRPW=2) for every bucket >= 16
        for b in (16, 32, 128, 256):
            st["moe"]["tune"][b] = ("wx", 0, 0, os.environ["DL_WX"])
        print("DL_WX", st["moe"]["tune"], flush=True)
    if os.environ.get("DL_G1") is not None:  # force the MoE g1 path at M1 (DL_G1="" or "-DDQHI")
        st["moe"]["tune"][1] = ("g1", 0, 0, os.environ["DL_G1"])
        print("DL_G1", st["moe"]["tune"], flush=True)
    if os.environ.get("DL_GK") is not None:  # force the MoE gk path at M8 / M16 (DL_GK="-DSW=4 -DKS2=2" etc.)
        for b in (8, 16):
            st["moe"]["tune"][b] = ("gk", 0, 0, os.environ["DL_GK"])
        print("DL_GK", st["moe"]["tune"], flush=True)
    if os.environ.get("DL_PICKS"):  # generic: "1=gk:-DSW=4 -DKS2=1 -DMAXM=8;16=gk:-DSW=4 -DKS2=2" (path:defines, path in g1/gk/wx)
        for it in os.environ["DL_PICKS"].split(";"):
            b, c = it.split("=", 1)
            kind, x = c.split(":", 1)
            if kind == "fs":  # "32=fs:1,3," -> ("fs", 1, 3, "")
                k, sm, ex = x.split(",", 2)
                st["moe"]["tune"][int(b)] = ("fs", int(k), int(sm), ex)
            elif kind in ("ring", "v17"):  # "32=ring:1,0"
                a, c = x.split(",")
                st["moe"]["tune"][int(b)] = (kind, int(a), int(c))
            else:
                st["moe"]["tune"][int(b)] = (kind, 0, 0, x)
        print("DL_PICKS", st["moe"]["tune"], flush=True)
    print("DLPREP %.1fs state %.2f GB peak %.2f GB (above the %.2f GB before)" % (time.time() - t0, (torch.cuda.memory_allocated() - m0) / 2**30,
          (torch.cuda.max_memory_allocated() - m0) / 2**30, m0 / 2**30), flush=True)
    return st
