import json, sys
d = json.load(open(sys.argv[1]))
r = d.get("result") or {}
print(d.get("status"), r.get("status"), "passed", r.get("passed"), "speedup", r.get("speedup"), r.get("speedup_ci"), (r.get("error") or "")[:300])
for x in r.get("records", []):
    if x["event"] == "test" and x.get("status") != "pass": print("FAIL", x["case"], x.get("message", "")[:300])
    if x["event"] == "benchmark":
        c = x["case"]
        nm = " ".join(f"{k}={v}" for k, v in c.items() if k in ("load", "layer", "batch", "q_len", "m"))
        print(f"  {nm:48} sgl {x['baseline_ns']/1e3:8.1f} ours {x['median_ns']/1e3:8.1f} x{x['speedup']:.3f}")
lt = r.get("log_tail") or ""
for l in str(lt).splitlines():
    if any(k in l for k in sys.argv[2:]): print(l[:300])
