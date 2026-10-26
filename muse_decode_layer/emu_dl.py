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
    print("DLPREP %.1fs state %.2f GB peak %.2f GB (above the %.2f GB before)" % (time.time() - t0, (torch.cuda.memory_allocated() - m0) / 2**30,
          (torch.cuda.max_memory_allocated() - m0) / 2**30, m0 / 2**30), flush=True)
    return st
