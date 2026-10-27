# muse_attention_decode: where would a 2.37 come from?

Ours: v56, judge 1.8429. Leader: 2.37, which is 1.29x faster than us in geomean.

## 1. Byte floors per shape

Sources: `muse_attention_decode/reference.py`, the v56 judge profile (`profiles/submission_v56.1b3eda41addb4119.txt`).

- The KV cache is FP8 E4M3 in the judge (`FP8 = torch.float8_e4m3fn`, `_fill_cache`). TASK.md: "The KV cache stays FP8". SGLang's FA3 reads the same FP8 rows. Each token is one random 128 B K row (both kv heads) plus one random 128 B V row (page size 1, `randperm` pool), so every key costs 256 B of random 128 B gathers.
- Keys attended: sliding = window + L - 1 (2048 decode, 2055 verify); full = context + L (lengths are agentic: uniform in 80000 +-7.5 %, mean 80000).
- Minimal bytes = keys x (256 B KV + 4 B page table) + the qkv / out rows (< 0.3 MB). The 128 B gather ceiling on the H200 is 3.87 TB/s (v49diag GB lines: 264-396 CTAs, any ring depth >= 3); 4.8 TB/s is the HBM peak.

| shape | keys | MB | floor @3.87 us | floor @4.8 us | SGLang | ours (v56) | ours / floor @3.87 |
|---|---|---|---|---|---|---|---|
| d1s | 2048 | 0.54 | 0.14 | 0.11 | 20.91 | 8.62 | 61.9 |
| d1f | 80001 | 20.81 | 5.38 | 4.33 | 31.55 | 17.45 | 3.2 |
| d16s | 32768 | 8.63 | 2.23 | 1.80 | 27.01 | 11.21 | 5.0 |
| d16f | 1280016 | 332.91 | 86.02 | 69.36 | 201.55 | 110.48 | 1.3 |
| d32s | 65536 | 17.25 | 4.46 | 3.59 | 31.47 | 13.57 | 3.0 |
| d32f | 2560032 | 665.82 | 172.05 | 138.71 | 382.86 | 200.99 | 1.2 |
| v1s | 2055 | 0.59 | 0.15 | 0.12 | 22.85 | 12.42 | 81.8 |
| v1f | 80008 | 20.86 | 5.39 | 4.34 | 36.03 | 21.56 | 4.0 |
| v16s | 32880 | 9.40 | 2.43 | 1.96 | 29.38 | 16.74 | 6.9 |
| v16f | 1280128 | 333.69 | 86.22 | 69.52 | 205.26 | 130.86 | 1.5 |
| v32s | 65760 | 18.80 | 4.86 | 3.92 | 34.27 | 19.84 | 4.1 |
| v32f | 2560256 | 667.37 | 172.45 | 139.04 | 390.16 | 243.31 | 1.4 |

A load is 46 x sliding + 16 x full, and the score is the geomean of SGLang/ours over the 6 loads (paired-ratio medians aside).

Ours per load (d1, d16, d32, v1, v16, v32): 2.17, 1.96, 1.97, 1.78, 1.62, 1.63, geomean 1.84.

## 2. What a 2.37 looks like

Model: every shape costs floor @3.87 / efficiency + a fixed per-call latency F (launch, dependent DRAM trips, split merge).

| model | d1 | d16 | d32 | v1 | v16 | v32 | gm |
|---|---|---|---|---|---|---|---|
| ours (v56) | 2.17 | 1.96 | 1.97 | 1.78 | 1.62 | 1.63 | 1.84 |
| floor + 8 us | 2.49 | 2.26 | 2.19 | 2.76 | 2.33 | 2.25 | 2.37 |
| floor / 0.85 + 6 us | 3.05 | 2.12 | 1.97 | 3.38 | 2.18 | 2.01 | 2.39 |
| floor / 0.85 + 4 us | 4.11 | 2.25 | 2.03 | 4.55 | 2.31 | 2.08 | 2.73 |
| ours, full shapes at floor + 3 us | | | | | | | 2.29 |
| ours, sliding shapes at floor + 4 us | | | | | | | 2.31 |
| ours, full / 1.5 | | | | | | | 2.37 (d32f would be below its floor) |

- 2.37 needs no exploit. "85 % of the gather ceiling on the big shapes plus ~6 us per call" is enough, and so is "floor plus 8 us everywhere".
- Under the 85 %/6 us model our decode full shapes are already there (d16f 110 vs 107, d32f 201 vs 208). The gaps, as ours / model time per shape: v1s 2.0x, v16s 1.9x, v1f 1.75x, v32s 1.7x, d1f 1.42x, d1s 1.39x, d16s 1.3x, v16f 1.22x, d32s 1.2x, v32f 1.16x.
- Per load (ours / model): **v1 1.93x**, d1 1.40x, v16 1.35x, v32 1.24x, d16 1.08x, d32 1.0x. The leader must be winning mostly where we are latency-bound: batch-1 verify and decode, then the sliding verify shapes. The large full shapes are within 15-25 % of what a 2.37 needs.

## 3. Mechanisms, checked against our timelines

- **K/V reuse across the 8 query tokens.** Already done: one 64-row (8 tokens x 8 heads) wgmma tile per kv head per key block in both verify kernels, so verify reads each key once.
- **FP8 dequant cost.** QK runs on FP8 wgmma straight from the gathered rows (no K conversion). V is converted to f16 once per key block; removing that conversion (ablation, H100 v32f) saved 1.5 %. Not a lever.
- **Bandwidth on the full shapes.** Decode full is at 3.0-3.3 TB/s and verify full at 2.6-2.7 TB/s of the 3.87 ceiling. The warp-specialized verify (attnp) is consumer-bound on its per-step dependent chain (H200: consumers wait for data 7 % of the loop, the producer waits for free slots 12-15 %). Raising it to ~3.3 TB/s is worth v16f 131 -> ~107 and v32f 243 -> ~209 (v16 x1.22, v32 x1.16 on their loads).
- **Fixed latency on the small shapes** (H200 VDIAG, us from the first CTA's start, v56):
  - v1s (attnw39, 17 CTAs x 2 steps): prologue 2.8, loop to 5.7 (1.5 us per 64-key step on a lone CTA), partial stores and arrival 6.0 -> 6.9, last arrival 8.0, merge 8.0 -> 10.7.
  - v16s (attnp): producer page table 2.6, first KV 3.9, consumer query tile 4.3, loop to 8.7, merge to 15.5.
  - v1f (attnp, 264 lanes x 4.75 steps): first KV 4.3, loop to 13.1, merge 14.6 -> 20.3.
  - The chain seq -> page table -> KV is three dependent DRAM trips (~0.7-1 us each cold). The merge costs 2.5-5 us: partial stores, an acq_rel arrival atomic, waiting for the slowest split, then 1-3 serial L2 round trips over the partials.
  - "start" (1.3-2.0 us after the first stamp in every attnp shape) was not launch latency. ptxas moved the split range computation, which needs seq (one cold DRAM load), above the kernel's first `__syncthreads` (an `R2UR` of seq right after its `LDG`). So every warp waited a DRAM trip at the barrier, and the consumers' query loads were only issued after it. v57 fixes this (section 4).
- **Fewer splits and a cheaper merge.** At B = 1 a split's loop is latency-bound (~1-1.5 us per step), so fewer, longer splits make the loop longer. Our v1s / v1f merges cost 2.7-5.7 us; a DSMEM cluster merge lost earlier (v48: 16-CTA clusters forced a third step on some splits). The plausible leader shape for B = 1 is one key block per CTA (33 CTAs for v1s, 1251 key blocks spread over 132 SMs for v1f) plus a merge that is one L2 round trip.

## 4. v57: the change built

- The seq hoist: in attnp and attnd, seq is now read after the first CTA barrier through a non-uniform zero loaded from shared memory (`seq = seq_ld + lds(mbar + 248 + 4 (tid & 1))`). A plain `asm volatile` mov was not enough: ptxas still moved seq into a uniform register before the barrier. SASS of attnp x32836 / x68 and attnd x64 now has the first use of seq after `BAR.SYNC 0x0` and the `LDS`. The BAL variant (v16f / v32f) still needs every request's length before its split map; that one is inherent.
- Expected: about -1 us on v1f, v16s, v32s, d16s, d32s (the producer's page-table load starts at kernel start + one DRAM trip instead of + two), and a little on v16f / v32f.
- Also: attnd is selected only on sm_90 (or the wgmma emulation), so v53+ no longer crash at decode 16 on other GPUs (`_hopper()` guard, as in the dl build).
- v57diag adds per-lane sub-stamps and B = 1 sliding candidates: attnp at 64 / 128 keys per split with early loads and a grouped merge of 32 / 16 mergers, against attnw39.

## 5. Next, by value

1. B = 1 verify (v1 load 1.93x from the model): one key block per CTA plus a one-round-trip merge for v1s, and a shorter merge for v1f (parallel over splits and rows already; the remaining cost is the arrival atomic and the slowest split).
2. Sliding verify B16 / B32: the same fixed-latency items (merge 4-7 us of 16-20).
3. Verify full: consumer chain per step (latency-bound at ~1.4 us per lane step); needs more independent work per warp (QK of step i + 1 during softmax of step i), which does not fit in 112 registers as built.

## 2026-10-27: gap to the 3.03 leader (we are 1.85, v66diag per-shape times)

Load time = 46 x sliding + 16 x full; the score is the geomean over the 6 loads.

| load | SGLang | ours | x | sliding share of ours |
|---|---|---|---|---|
| d1 | 1462 | 678 | 2.16 | 59 % |
| d16 | 4453 | 2263 | 1.97 | 22 % |
| d32 | 7619 | 3884 | 1.96 | 17 % |
| v1 | 1648 | 911 | 1.81 | 64 % |
| v16 | 4643 | 2865 | 1.62 | 27 % |
| v32 | 7845 | 4826 | 1.63 | 19 % |

3.03 on every load needs roughly: full B16 / B32 at ~1.2x the 4.8 TB/s floor (76 / 152 us, i.e. ~4 TB/s, about the measured 3.87 TB/s random-row gather ceiling) with sliding at 2-7 us; B = 1 at ~9-10 us sliding and ~5-8 us full. Model: sliding at a fixed 4 us and full at 1.1x / 1.3x floor gives 4.07 / 3.62; 5 us and 1.3x gives 3.38. So the leader is near the gather ceiling on full shapes and ~2-4x below our latency on sliding / B = 1.

Where we are vs that target: v16f 131 vs ~80 (1.6x), v32f 244 vs ~160 (1.5x), d16f 110 / d32f 203 (1.3-1.4x); v32s 20, v16s 17, v1s 12.7, v1f 20.5, d1f 17.4 vs ~5-9 (2-3x); d16s 10.9, d32s 14, d1s 8.7 (1.5-2.5x).

The common limiter on all six verify shapes is the attnp consumer step (~1.5 us per 64-key lane step, a serial chain; v66 ablations): v16f / v32f directly, and v1s / v1f / v16s / v32s through 4-10 steps per lane plus prolog (~2.3 us) and merge tail (~3-5 us). Per SM the step's resource floors are ~1200 issue cycles and 1024 MUFU cycles per 4-warpgroup round vs ~2900 measured, so overlap (ping-pong, FP8 PV, more concurrent chains) is worth up to ~2x there. Decode full (fdec 2.9-3.3 TB/s) is closest to the ceiling.

TASK.md levers: FP8 P / V is allowed (queries may be E4M3; FA3 with an FP8 cache runs PV in FP8), so v67 FP8 PV is legal. No prepare step and "fresh inputs: don't cache anything derived from data", so no KV relayout or cached page-table products. Cache rows are a random permutation (reference.generate_input randperm), so every key is a 128 B random gather: the 3.87 TB/s ceiling stands. The graph's fixed ~4 us is subtracted on both sides; a raw seq -> page table -> KV chain is ~1.2-1.7 us after that, so a sliding shape could reach ~3-4 us.
