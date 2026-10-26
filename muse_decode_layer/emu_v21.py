# sm_86 check of submission_v21.py with the sm_90a attention verify path (attnp / wgmma) emulated (-DWGEMU=1, small ring)
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("dl_v21", Path(__file__).with_name("submission_v21.py"))
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
_m.A__DBG = ["-DWGEMU=1", "-DRSW=2"]
prepare, kernel_fn = _m.prepare, _m.kernel_fn
