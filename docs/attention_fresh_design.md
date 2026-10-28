# swk: batched sliding decode (d16s / d32s) without lockstep chunks or a global merge

What actually limits these shapes (H200 facts from docs/ideas.md): the data is tiny (8.4 / 16.8 MB = 2.2 / 4.3 us at 3.87 TB/s) and
every failed or current kernel is latency-chain bound, not bandwidth bound:
- the per-64-key step costs 0.65-1.1 us whether the data is already in smem or not (f1k: loop - kv-issued = 11 us for 10 chunks in all
  three stream modes, even F1bulk where every row had landed; s2k ~1 us / chunk; attnd 740-880 ns / step). It is a dependent chain
  (mbarrier wait -> shared convert -> bar.sync -> mma -> shuffles -> exp -> mma -> bar.sync) run by all warps of a CTA in lockstep, so
  the SM idles. The aggregate work is small: ~90 instructions per 16 keys per kv head = ~6K per SM on d32s (~0.8 us of issue).
- the cross-CTA merge after the slowest split costs 2.7-3.5 us (attnd d16s loop-end max 6.5 -> end 9.2; even CM's DSMEM tail was 2.4).
- clusters failed only on GPC packing (size 8 -> 15, 4 -> 30, 3 -> 39, 2 -> 66 co-resident at 1 CTA / SM), not on DSMEM cost.

Design (submission_v84diag.py, swk<NH>, flag SW, off by default):
1. Clusters sized to fit one wave: one cluster per (request, kv head): B 16 -> NC 3 (32 clusters <= 39, 96 SMs), B 32 -> NC 2
   (64 <= 66, 128 SMs); alternative NH 2 (both heads, full 128 B rows): B 16 NC 6 (16 <= 17). No global atomics, spins or counters.
2. Warp-independent chunks: 16 warps, each owns <= 4 chunks of 16 keys of one head; seq -> page table (2 LDG per lane) -> shuffles ->
   every K / V load of the warp issued at once (<= 64 KB per SM in flight on d32s) straight into registers. No smem ring, no mbarrier,
   no CTA barrier until the merge, so the 16 chains overlap instead of serializing.
3. Raw E4M3 bytes are the mma fragments: S^T = K Q^T with keys as M (no padding at 8 query heads; head dims permuted so lane (g, c)'s
   16 B row slice is its A fragment for all 4 k-steps), O^T = V^T P^T with dims as M (lane loads 8 B of 4 keys; 2 PRMT + 4 cvt per
   m-tile), P^T moved from C to B layout by 2 movmatrix. mma.sync f16 (L 1 compute is ~0.3 GFLOP: no wgmma needed).
4. One-pass softmax per warp (all its keys are in registers), warp partials -> smem -> one __syncthreads combine, ranks 1.. push
   (O, m, l) rows to rank 0 over DSMEM, one cluster barrier, rank 0 merges NC sources, RMSNorm, gate (prefetched at entry).

Per-phase budget, us of kernel time from launch (attnd today: prolog 2.0-2.4, loop end 5.5-6.5 / 8.1-9.0, end 9.2 / 11.8):
| phase | d16s (NC 3) | d32s (NC 2) | basis |
|---|---|---|---|
| seq -> page table -> first KV row | 1.2 | 1.2 | LAT probe: seq->pt->kv +1.0-1.25 over an empty kernel; q / new-KV prologue hidden under the pt trip |
| stream the rest | 2.4-3.0 | 4.4 | 87 / 128 KB per SM at 29-36 GB/s per SM (3.87 TB/s / 132 is the floor rate) |
| last chunk's chain after its data lands | 0.4 | 0.4 | ~700 clk: 16 cvt, 4 dependent mma, 3 shuffles, ex2, 2 movmatrix, 4 mma |
| CTA combine (barrier + 16-source LDS) | 0.2 | 0.2 | |
| DSMEM push + cluster barrier + owner merge, norm, gate, store | 0.5 | 0.5 | CM measured cl-bar1 -> cl-bar2 ~0.8 incl. skew |
| total kernel / judge (kernel + ~2) | 4.7-5.3 / ~7.0 | 6.7 / ~8.8 | today 8.9 / 10.9 and 11.8 / 14.0 |

Honest estimate: -30-38 % on d16s / d32s, not -50 %: halving needs judge 5.4 / 7.0, i.e. kernel ~3.5 / 5, below chain + data at the
ceiling (3.4 / 5.5). Score: d16 load x1.08, d32 x1.065 -> geomean ~2.04 (from 1.99). Risks: (a) the per-SM gather rate with 96 SMs
(B 16) may stay at 29 GB/s/SM (then NH 2 NC 6 is no better; NC 4 would need 32 clusters > 30); (b) 64 regs of in-flight data + 8 q
fragments per thread under the 128-register cap of 512 threads (ptxas report below); (c) half-row (64 B) loads rely on the L2::128B
hint so DRAM sees 128 B gathers.
Next if it wins: the same skeleton for v16s / v32s (64 query rows per kv head: 4 warps share each chunk's rows through L1, or a
64-row wgmma tile per warpgroup), and d1f with per-warp prefetching streams (no lockstep) + last-16-arrivers parallel merge
(budget ~1.2 + 5.3 / 0.85 + 0.4 + 1.5 = 9.3 us kernel vs 15.5).
Status: built, off by default (SW = False). sm_90a: swk<1> / swk<2> 116 registers, 0 spills, all K / V LDG (LTC128B) issued before
the first HMMA. sm_86 (cluster exchange through global memory): 140 reference checks pass (B 2-32, agentic / mixed / short contexts,
edges, kv_scale 1/64, every (NH, NC) in {1,2} x {2,3,4,6,8}; err 0.004-0.0057, rel vs default <= 4e-4), 300-replay fresh-input
graph stress d16s / d32s clean. The DSMEM / cluster-barrier path itself only compiles here: the v84diag ST lines are its first test.
