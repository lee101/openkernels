# H200 pod session 1 (2026-10-23, ~45 min, ~$4)
- RunPod: ncu counters blocked (ERR_NVGPUCTRPERM even as root). Only timeline/microbench.
- loaded SM clock 1.76-1.83 GHz; cuk bw: DRAM read 4.2 TB/s (87% of 4.8), mma.sync bf16 645 TF, graph node 0.5us
- moe v3 (16 warps x 2 TMA stages): spans bs1 57us, bs8 115, bs32 213, bs256 375 (after ldca for G2 B)
- FAKEDQ (no dequant ALU) only -12%; FAKEDQ+NOMMA bs32 still 181us, bs1 47us => not ALU-bound
- 12 warps x 3 stages: slower. Full-stage B prefetch (D=7): no change.
- interleaving G2(e) right after G1(e+1) in round-robin order: worse (same-round dependency waits)
