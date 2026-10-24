#!/usr/bin/env python3
"""virtualcuda: estimate H200 (sm_90) kernel time without an H200.
  vc.py sass <src.cu|submission.py> [-D X=1 ...] [--func REGEX] [--json]   SASS mix, loops, 1-warp cycles (sm_90a)
  vc.py est <spec.py|spec.json> [--set k=v ...]                            analytical H200 estimate (calibrated)
  vc.py calib [--fit] [--cv] [--save] [--hold=moe|none]                     calibration table vs real H200 numbers
  vc.py ncu2spec <ncu.csv> [--sms N]                                        H200 estimate from an ncu (sm_86, daisy) CSV
  vc.py cases                                                               list the calibration cases (templates)
Simulator wrappers (Accel-Sim SM90_H200 on sm_86 NVBit traces): sim/trace.sh, sim/run_sim.sh; ncu: sim/ncu_daisy*.sh"""
import importlib.util, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vc import model


def load_spec(path):
    p = Path(path)
    if p.suffix == ".json": return model.from_dict(json.loads(p.read_text()))
    spec = importlib.util.spec_from_file_location("vc_spec", p); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    ks = m.kernels() if hasattr(m, "kernels") else m.KERNELS
    return ks


def params(argv):
    p = dict(model.P0)
    for a in argv:
        if "=" in a and not a.startswith("-"):
            k, v = a.split("=", 1); p[k] = float(v)
    return p


def est(argv):
    ks = load_spec(argv[0]); p = params([a for a in argv[1:] if a != "--set"])
    if isinstance(ks, dict):  # {"variant": [kernels]} -> side-by-side
        base = None
        for name, k in ks.items():
            t = model.predict(k, p); base = base or t
            print(f"{name:40} {t:9.2f} us  ({t / base:.3f}x of first)")
        if "-v" in argv:
            for name, k in ks.items(): print(f"--- {name}\n" + model.report(k, p))
        return
    print(model.report(ks, p))


def ncu2spec(argv):
    """ncu CSV (one row per metric) -> per-kernel H200 Phase from measured bytes and executed instructions."""
    import csv
    rows = list(csv.reader(open(argv[0])))
    hdr = rows[0]; ki, mi, vi, gi = hdr.index("Kernel Name"), hdr.index("Metric Name"), hdr.index("Metric Value"), hdr.index("ID")
    K = {}
    for r in rows[1:]:
        if len(r) < len(hdr): continue
        try: v = float(r[vi].replace(",", ""))
        except ValueError: continue
        K.setdefault((r[gi], r[ki]), {})[r[mi]] = v
    p = dict(model.P0)
    for (i, name), m in K.items():
        warp_inst = m.get("sm__inst_executed.sum", 0)
        rd, wr, l2 = m.get("dram__bytes_read.sum", 0), m.get("dram__bytes_write.sum", 0), m.get("lts__t_bytes.sum", 0)
        grid = m.get("launch__grid_size", 132)
        tensor = m.get("smsp__inst_executed_pipe_tensor.sum", 0) * 4096 / 2700  # sm_86 HMMA.16816 count -> H200 mma.sync SM-cycles
        ph = model.Phase("ncu", hbm=rd + wr, l2=max(0, l2 - rd - wr), issue=warp_inst, tensor=tensor, sms=min(132, grid),
                         pipes={"xu": m.get("smsp__inst_executed_pipe_xu.sum", 0)})
        k = model.Kernel(name[:60], [ph])
        t3090 = m.get("gpu__time_duration.sum", 0) / 1e3
        print(f"[{i}] {name[:70]}\n    3090: {t3090:.2f} us, DRAM {(rd + wr) / 1e6:.1f} MB, L2 {l2 / 1e6:.1f} MB, {warp_inst / 1e6:.2f} M warp-instr"
              f" -> H200 estimate {model.predict(k, p):.2f} us (judge-style)")
        if "-v" in argv: print(model.report(k, p))


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"): print(__doc__); return
    cmd, argv = sys.argv[1], sys.argv[2:]
    if cmd == "sass":
        from vc import sass
        sass.main(argv)
    elif cmd == "est": est(argv)
    elif cmd == "calib":
        from vc import calib; calib.main(argv)
    elif cmd == "ncu2spec": ncu2spec(argv)
    elif cmd == "cases":
        from vc.cases import cases
        for c in cases(): print(f"{c['group']:10} {c['name']:30} meas {c['meas']:7.1f} us  {c['src']}")
    else: print(__doc__)


if __name__ == "__main__":
    main()
