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

# bf16_proj (H200 checks)
- per-kernel GPU (cuBLAS): qkv m1 9.4+1.7 splitK reduce, pooler 18.6+1.6, o 5.2, unp-slice 4.4+1.4; judge ~= kernels + 1.5-2us
- pure TMA stream single launch (graph-empty): 8.4MB 2.0, 16.8MB 5.7, 37.7MB 10.0, 67MB 16.5us (16w x2 x4.6KB best; 2KB stages slower)
- v1 0.84 | v2 CTA smem 0.78 | v3 0.66 | v4 TMA 6x2KB 0.86 | TWOK 0.86 | v5 4KB stages + red + PDL fin 0.94
- sweep m1 (graph-empty): cublas qkv 15.5 poo 21.7 o 7.4 unp 8.4; ours RSN2/CUN2 qkv 14.6 poo 21.5 o 9.2 unp 7.4
- router hi/lo: 1.3-3.25x at m<=32; m=128/256 router path still slower than SGLang split2

# attention (H200 checks)
- SGLang/shape: sliding 20.6-34.6us, full b1 31, b16 201, b32 384 (FA3 ~1.7TB/s)
- random-gather bench: 64B rows 2.1-2.4TB/s, 128B 2.9-3.1, 256B 3.2-3.4
- v1 0.39 | v2 ring 0.49 | v3 both-heads full-row decode 0.58 (full decode 1.48/1.68) | v4 CTA verify + comb2 0.69
- v5 0.68 | ex2.approx 0.72 | verify one-wave grid 0.81 | SB=4 steps 0.82
- verify bottleneck: F2FP e4m3->f16 4x redundant per CTA (~376us at b32) > HMMA (~145us); plan: FP8 mma (Q,P e4m3) or convert once
- sliding shapes need a single fused launch (prep+attn+combine+gate) to approach leader (2.37x)

# 2026-10-24 sessions 4-13 (judge checks)
- attention: v17 wgmma verify 1.334 | v21 fused verify launch 1.361 (sl decode b1 1.56, b16 1.53, b32 1.23; full ~1.33-1.47; verify sl 1.40-1.47, full 1.16)
  - mma.sync e4m3 is emulated on sm_90 (F2FP+HMMA): no gain; wgmma FP8 QK from raw ring + f16 PV (V converted) works
  - L2 bulk prefetch ahead of the ring hurts badly; random 128B gathers ~2.5TB/s ceiling
  - fused decode timeline: prologue 3.5-9us (dependent seq->ptb->KV chain, cold code), merge 3-6us dominate b1/b32 sliding
  - cluster DSMEM merge: small gain at b1, terrible for B*NS>64 (GPC fragmentation -> second wave)
- moe: v7 adaptive ks 1.356 | v8 early slot refill 1.350 | v9 large-M build (8 warps x 4 stages, NT<=8) 1.371 | v10 lookahead-3 1.306 (balance loss)
  - inner loop streams 3.9TB/s steady; losses: ptxas warp-aggregated atomicAdd + shfl stalls on the queue atomic (use atom.inc asm), G1->G2 deps (per-quarter counters), startup
- decode layer: v0 0.90 | v1 1.222
- bf16_proj: v6 deep per-CTA ring GEMV 0.83 (worse than v5 0.94)
- prefill: v0 failed on H200 (attnw partial layout rows; fixed), pod torch lacks _grouped_mm and mm(out_dtype) -> fallbacks
- s15: attention v31 (page-table smem rings): decode full b16 148->117.5us, b32 277->206.5 (okb); verify full b32 340->323
- s15: bf16_proj v7 (in-kernel finalize) slower than v5 at every shape -> off
- s15: w4wg prototype correct after 1 KB stage alignment fix; 1.2 ms for 32 experts x 8192 x 4096 at ~20 tok/expert (latency-bound, 3-stage ring, serialized dequant/wgmma)
