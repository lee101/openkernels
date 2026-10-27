"""run a bf16_proj submission's sm_90 extension build branches on a non-Hopper box (compile + link + import, no launch):
_wbext() with the device capability faked to 9.0 (the sm_90a wgmma extension, exactly the judge's code path) and, with --gv,
_gvext() built for TORCH_CUDA_ARCH_LIST=9.0. Exit code 1 if a build branch fails (e.g. a Python error in the source generator).
python bench/wb_compile_chk.py muse_bf16_proj/submission_v27diag.py [--gv] [--dry]   (--dry: prepare() with a stub wgmma extension)"""
import importlib.util, os, sys, time
import torch

sub = sys.argv[1]
spec = importlib.util.spec_from_file_location("sub_chk", os.path.abspath(sub)); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
real = torch.cuda.get_device_capability
torch.cuda.get_device_capability = lambda *a, **k: (9, 0)
t = time.time()
try:
    ext = m._wbext()
finally:
    torch.cuda.get_device_capability = real
ok = bool(ext) and all(hasattr(ext, f) for f in ("wbrun", "wqrun", "wvrun"))
print("wb sm_90a build+import:", "OK" if ok else "FAILED", "%.0fs" % (time.time() - t), flush=True)
if "--gv" in sys.argv:
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
    t = time.time()
    try:
        g = m._gvext()
        gok = hasattr(g, "gvrun")
    except Exception as e:
        print("gv:", str(e)[-500:]); gok = False
    print("gv sm_90 build+import:", "OK" if gok else "FAILED", "%.0fs" % (time.time() - t), flush=True)
    ok = ok and gok
if "--dry" in sys.argv:  # run prepare() with a stub sm_90 extension (every wgmma launch "fails"): exercises the Hopper-only plan code
    class _Stub:
        def __getattr__(self, n):
            return lambda *a: -9
    m._wbmod[0] = _Stub()
    sys.path.insert(0, os.path.dirname(os.path.abspath(sub)))
    import reference as ref
    t = time.time()
    try:
        m.prepare(ref.generate_weights(ref.WEIGHTS_SEED)); dok = True
    except Exception as e:
        import traceback; traceback.print_exc(); dok = False
    print("prepare with stub sm_90 ext:", "OK" if dok else "FAILED", "%.0fs" % (time.time() - t), flush=True)
    ok = ok and dok
sys.exit(0 if ok else 1)
