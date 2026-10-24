"""python3 -m pytest -q virtualcuda/tests  (or python3 virtualcuda/tests/test_vc.py)"""
import shutil, sys, tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vc import model, sass, calib
from vc.cases import cases, deltas


def test_model_monotone():
    k = lambda b: [model.Kernel("s", [model.Phase(hbm=b)])]
    assert model.predict(k(64e6)) > model.predict(k(32e6)) > model.predict(k(1e6))
    # 64 MB at ~4.2 TB/s is ~15 us of transfer plus fixed costs
    assert 14 < model.predict(k(64e6)) < 22


def test_latency_chain_adds():
    a = model.Kernel("a", [model.Phase(hbm=1e6)]); b = model.Kernel("b", [model.Phase(hbm=1e6, chain=2)])
    assert abs(model.predict(b) - model.predict(a) - 2 * model.P0["lat"]) < 1e-9


def test_calibration_accuracy():
    C = [c for c in cases() if c["group"] != "moe"]
    err = calib.pred_all(C, model.P0) / np.array([c["meas"] for c in C]) - 1
    assert np.median(np.abs(err)) < 0.15, np.median(np.abs(err))


def test_sass_loop_and_pipes():
    if not shutil.which(str(sass.CUDA / "bin/nvcc")): return
    src = r'''
    __global__ void k(const float* a, float* o, int n) {
      float s = 0.f;
      for (int i = threadIdx.x; i < n; i += blockDim.x) s = fmaf(a[i], 1.5f, s) + __expf(s);
      o[threadIdx.x] = s;
    }'''
    d = Path(tempfile.mkdtemp()); f = d / "t.cu"; f.write_text(src)
    r = sass.analyze(str(f))
    e = [x for x in r if "k(" in x["func"]][0]
    assert e["loops"], "loop not found"
    lp = max(e["loops"], key=lambda l: l["n"])
    assert lp["pipes"].get("mufu", 0) >= 1 and lp["pipes"].get("gmem", 0) >= 1 and lp["warp_cycles"] > lp["n"]


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("ok", n)
