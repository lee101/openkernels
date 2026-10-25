"""spill instructions inside / near the HMMA region of a cubin: python3 bench/spills.py k.o [k2.o ...]"""
import subprocess, sys
for f in sys.argv[1:]:
    L = subprocess.run(["cuobjdump", "-sass", f], capture_output=True, text=True).stdout.splitlines()
    h = [i for i, l in enumerate(L) if "HMMA" in l or "HGMMA" in l]
    sp = [i for i, l in enumerate(L) if "LDL" in l or "STL" in l]
    inside = sum(1 for i in sp if h and h[0] <= i <= h[-1])
    print(f, "spill instrs", len(sp), "inside mma span", inside, "mma span lines", (h[-1] - h[0]) if h else 0)
