# H200 pod session 1 (2026-10-23, ~45 min, ~$4)
- RunPod: ncu counters blocked (ERR_NVGPUCTRPERM even as root). Only timeline/microbench.
- loaded SM clock 1.76-1.83 GHz; cuk bw: DRAM read 4.2 TB/s (87% of 4.8), mma.sync bf16 645 TF, graph node 0.5us
- moe v3 (16 warps x 2 TMA stages): spans bs1 57us, bs8 115, bs32 213, bs256 375 (after ldca for G2 B)
- FAKEDQ (no dequant ALU) only -12%; FAKEDQ+NOMMA bs32 still 181us, bs1 47us => not ALU-bound
- 12 warps x 3 stages: slower. Full-stage B prefetch (D=7): no change.
- interleaving G2(e) right after G1(e+1) in round-robin order: worse (same-round dependency waits)

# check results log (H200, speedup vs SGLang)
- v1 1.069 | v2 race fail | v2.1 1.203 | v3 smem split ks forced 1.144 | ks by balance 1.204 | stream-K 1.255
- v4 dynamic queue + group barrier split-K + acq_rel 1.339 (bs1 1.47 bs16 1.36 bs32 1.14 bs8 1.41 bs128 1.46 bs256 1.24)
- v5 async last-arriver reduction 1.147 (worse everywhere, even ks=1 shapes) -> reverted
- v6 (warp-specialized producer/consumer ring) judge 1.188; pod head-to-head (read-only flush): v4 bs1 57.8 bs16 125 bs32 226 bs8 110 bs128 320 bs256 378 vs v6 59.5/161/312/145/386/528 -> v4 kept
- okb flush fixed: was writing 1.5GB before each replay (write-back stole HBM BW); now read-only like the judge
- total pod spend ~ $14 (3 sessions)
