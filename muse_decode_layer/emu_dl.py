# sm_86 check of a composed decode layer (env DL_SUB, default submission_v22.py) with the sm_90a attention paths
# (attnp / attnd / wgmma) emulated (-DWGEMU=1, small ring). usage: DL_SUB=submission_v22.py python3 okb.py muse_decode_layer -s emu_dl.py ...
import importlib.util, os
from pathlib import Path

_spec = importlib.util.spec_from_file_location("dl_emu", Path(__file__).with_name(os.environ.get("DL_SUB", "submission_v22.py")))
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
_m.A__DBG = ["-DWGEMU=1", "-DRSW=2"]
prepare, kernel_fn = _m.prepare, _m.kernel_fn
