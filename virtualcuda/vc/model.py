"""Analytical H200 (sm_90, 132 SMs) kernel-time model: roofline per phase + latency chains + granularity tails.

A kernel = fixed launch/ramp cost + sequential phases. A phase:
  memory   hbm (streamed/TMA bytes), g64/g128/g256 (random row-gather bytes by row size), l2 (L2-hit bytes)
  compute  issue (warp-instructions), pipes {pipe: warp-instr} (SUB_COST cycles each), tensor (SM-cycles; see sass.mma_cycles)
  spread   sms (SMs doing the work), wps (warps per scheduler, for unit latency), inflight (bytes in flight per SM)
  serial   wcyc (cycles the critical warp spends serially in this phase: iterations x 1-warp cycles/iter from vc sass)
  tails    units/par/dynamic (work items over workers: wave quantization or dynamic-queue tail), unit_issue (warp-instr
           one worker issues per unit), unit_cyc (1-warp latency-bound cycles per unit), wps (warps sharing a scheduler),
           imb (max/mean load imbalance)
  per-SM   l2_sm (bytes a single SM moves serially, e.g. a last-arriver merge reading every split's partials)
  latency  chain (dependent DRAM round trips), l2chain (dependent L2 round trips), syncs (grid-level syncs/merges), us (extra fixed us)
Times in microseconds."""
import math
from dataclasses import dataclass, field, asdict

SM, CLK, SUBP = 132, 1.83e9, 4
HBM_PEAK = 4.8e12

# defaults = calibrated values (vc/params.json overrides after `vc calib --fit`)
P0 = {
    "bw_hbm": 4.2e12,    # streamed DRAM read, B/s
    "bw_g64": 2.25e12,   # random 64 B rows (H200 bench/gather.cu)
    "bw_g128": 3.0e12,   # random 128 B rows
    "bw_g256": 3.3e12,   # random 256 B rows
    "bw_l2": 7.0e12,     # L2-hit read bandwidth
    "bw_l2_sm": 117e9,   # one SM's L2 read bandwidth (64 B/clk), B/s
    "sat": 1.6,          # HBM saturates with SM/sat... i.e. bw scales as min(1, sat * sms / 132)
    "lat": 0.9,          # dependent DRAM round trip, us (cold L2/TLB)
    "lat_ld": 2.0,       # loaded DRAM latency for Little's law (bytes in flight / latency), us
    "l2lat": 0.35,       # dependent L2-hit round trip, us
    "t_kernel": 1.6,     # per-kernel fixed GPU time (launch ramp, first wave start, drain), us
    "t_sync": 1.5,       # grid-level sync / cross-CTA merge step (atomic + flag spin + reload), us
    "ipc": 1.0,          # warp-instr issue per scheduler per clock (pinned: H200 awprof, attnw 4 WG/SM ~2850 cyc per 2940-instr round)
    "nsm": 3.0,          # smooth-max exponent memory vs issue/tensor throughput (pinned: attnw no-KV-load ablation -11.6%)
    "icache": 0.6,       # cold-code cost per 1000 static SASS instrs, us (pinned: MoE 8.4K instrs, ~5 us first-iteration stall, H200 DIAG)
    "fstraight": 0.5,    # share of the all-paths straight-line 1-warp cycles on the critical path
    "t_graph": 1.0,      # per timed call: judge time (graph replay - 4 us) minus summed kernel time, us
}
import json as _json
from pathlib import Path as _Path
_PJ = _Path(__file__).resolve().parent / "params.json"
if _PJ.exists(): P0.update(_json.loads(_PJ.read_text()))


@dataclass
class Phase:
    name: str = "main"
    hbm: float = 0; g64: float = 0; g128: float = 0; g256: float = 0; l2: float = 0
    issue: float = 0; pipes: dict = field(default_factory=dict); tensor: float = 0; wcyc: float = 0
    sms: float = SM; wps: float = 0; inflight: float = 0
    units: float = 0; par: float = 0; dynamic: bool = False; unit_issue: float = 0; unit_cyc: float = 0; unit_sync: float = 0; imb: float = 1.0
    l2_sm: float = 0  # bytes one SM must move by itself (e.g. last-arriver merge reading all partials), at per-SM L2 bandwidth
    chain: float = 0; l2chain: float = 0; syncs: float = 0; us: float = 0


@dataclass
class Kernel:
    name: str
    phases: list
    launches: int = 1   # 0 = launch cost hidden (PDL / programmatic dependent launch)
    code: float = 0     # static SASS instructions (cold instruction-cache fetch at kernel start)
    wfix: float = 0     # 1-warp cycles of the code outside loops (vc sass "straight-line", all paths): serial prologue/epilogue


def phase_time(ph, p=P0, detail=False):
    sms = max(1.0, min(SM, ph.sms))
    frac = min(1.0, p["sat"] * sms / SM)
    bw = lambda b: b * frac
    t_hbm = ph.hbm / bw(p["bw_hbm"]) + ph.g64 / bw(p["bw_g64"]) + ph.g128 / bw(p["bw_g128"]) + ph.g256 / bw(p["bw_g256"])
    if ph.inflight:  # Little's law: per-SM bytes in flight / loaded latency
        lit_bw = ph.inflight * sms / (p["lat_ld"] * 1e-6)
        t_hbm = max(t_hbm, (ph.hbm + ph.g64 + ph.g128 + ph.g256) / lit_bw)
    t_mem = max(t_hbm, ph.l2 / p["bw_l2"]) * 1e6
    sched = sms * SUBP
    t_issue = ph.issue / sched / (p["ipc"] * CLK) * 1e6
    from .sass import SUB_COST
    t_pipe = max([v * SUB_COST.get(k, 1) / sched / CLK * 1e6 for k, v in ph.pipes.items()] + [0])
    t_tensor = ph.tensor / sms / CLK * 1e6
    t_serial = ph.wcyc / CLK * 1e6
    t_tp = max(t_issue, t_pipe, t_tensor)
    n = p["nsm"]  # smooth max: memory, issue/pipe throughput and the critical warp's serial time overlap imperfectly
    t_thr = max((t_mem ** n + t_tp ** n) ** (1 / n), t_serial * ph.imb)  # serial (latency-bound) warps overlap memory fully
    t = t_thr
    if ph.units and ph.par:
        waves = ph.units / ph.par
        u_thr = t_thr / max(waves, 1e-9)
        u_lat = max(ph.unit_issue * max(ph.wps, 1) / p["ipc"], ph.unit_cyc) / CLK * 1e6 + ph.unit_sync * p["t_sync"]
        u = max(u_thr if waves >= 1 else 0, u_lat)
        t = max(t_thr, (waves + 0.5) * u if ph.dynamic else math.ceil(waves - 1e-9) * u)
    fill = p["lat"] if (ph.hbm + ph.g64 + ph.g128 + ph.g256) > 0 else 0  # first bytes arrive one DRAM latency after the phase starts
    t_lat = fill + ph.l2_sm / p["bw_l2_sm"] * 1e6 + ph.chain * p["lat"] + ph.l2chain * p["l2lat"] + ph.syncs * p["t_sync"] + ph.us
    tot = t + t_lat
    if detail:
        bound = max((("mem", t_mem), ("issue", t_issue), ("pipe", t_pipe), ("tensor", t_tensor), ("serial", t_serial)), key=lambda x: x[1])[0]
        if t > t_thr * 1.05: bound += "+tail"
        if t_lat > t: bound = "latency"
        return tot, dict(mem=t_mem, issue=t_issue, pipe=t_pipe, tensor=t_tensor, serial=t_serial, thr=t_thr, tail=t - t_thr, lat=t_lat, bound=bound)
    return tot


def kernel_time(k, p=P0, detail=False):
    parts = [phase_time(ph, p, True) for ph in k.phases]
    t = k.launches * p["t_kernel"] + p["icache"] * k.code / 1000 + p["fstraight"] * k.wfix / CLK * 1e6 + sum(x[0] for x in parts)
    return (t, parts) if detail else t


def predict(ks, p=P0, call=True):
    """ks: Kernel or list of Kernels run back to back (no overlap); call=True adds the per-call graph overhead (judge time)."""
    ks = ks if isinstance(ks, (list, tuple)) else [ks]
    return sum(kernel_time(k, p) for k in ks) + (p["t_graph"] if call else 0)


def report(ks, p=P0):
    ks = ks if isinstance(ks, (list, tuple)) else [ks]
    out, tot = [], 0
    for k in ks:
        t, parts = kernel_time(k, p, True); tot += t
        out.append(f"{k.name}: {t:.2f} us (launch {k.launches * p['t_kernel']:.1f}, cold code {p['icache'] * k.code / 1000:.1f}, straight-line {p['fstraight'] * k.wfix / CLK * 1e6:.1f})")
        for ph, (pt, d) in zip(k.phases, parts):
            out.append(f"  {ph.name:10} {pt:8.2f} us  [{d['bound']}] mem {d['mem']:.2f} issue {d['issue']:.2f} pipe {d['pipe']:.2f} "
                       f"tensor {d['tensor']:.2f} serial {d['serial']:.2f} tail {d['tail']:.2f} lat {d['lat']:.2f}")
    out.append(f"total {tot:.2f} us kernels, {tot + p['t_graph']:.2f} us judge-style (+{p['t_graph']:.1f} per call)")
    return "\n".join(out)


def from_dict(d):
    """{"name":..., "launches":1, "phases":[{...Phase fields...}]} or a list of those -> Kernel(s)"""
    if isinstance(d, list): return [from_dict(x) for x in d]
    return Kernel(d.get("name", "kernel"), [Phase(**ph) for ph in d.get("phases", [{}])], d.get("launches", 1), d.get("code", 0), d.get("wfix", 0))


def to_dict(k):
    return {"name": k.name, "launches": k.launches, "code": k.code, "wfix": k.wfix, "phases": [asdict(ph) for ph in k.phases]}
