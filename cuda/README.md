# Local 3090 Ti sanity checks

Small standalone CUDA probes for the ideas in `../docs/ideas.md`. These are
**not** H200 reproductions: the local card is sm_86 (no TMA, WGMMA, FP8
tensor cores), 6 MB L2, 100 KB shared. They validate mechanisms and relative
direction, not absolute H200 numbers.

```bash
./run.sh          # builds and writes results.txt (GPU idle-ish)
make clean
```

## Results (RTX 3090 Ti, 2026-10-24, `results.txt`)

### `gather.cu` — paged-cache row size vs runs

| row | random | runs4 | runs16 | seq |
|---:|---:|---:|---:|---:|
| 64 B | 531 GB/s | 881 | 896 | 815 |
| 128 B | 829 | 878 | 878 | 860 |
| 256 B | 864 | 782 | 796 | 797 |

Takeaways:

- 64 B random rows lose ~40% of peak; coalescing runs recovers ~1.65x.
- 128 B random rows are already at 829 GB/s; runs add only ~6%.
- 256 B random rows are at 864 GB/s; runs do not help.
- `fdec` loads 128 B per token row (both KV heads). So **A2 is modest for the
  current 128 B path**; it matters if any path drops to 64 B per head, or if
  page-table lookups, not row bytes, are the limiter.

### `cvt.cu` — FP8→F16 conversion vs HMMA

| mode | time |
|---|---:|
| MMA only | 2.29 ms |
| conversion, 1x | 16.44 ms |
| conversion, 4x | 35.59 ms |
| conversion + MMA, 1x | 5.48 ms |
| conversion + MMA, 2x | 12.01 ms |
| conversion + MMA, 4x | 36.51 ms |

On sm86 the FP8 conversion is emulated, so this exaggerates the cost versus
H200's native conversion. The direction still matters: 4x redundant
conversion is far worse than converting once. Do not use the absolute ratio as
an H200 prediction.

### `reuse.cu` — stream a 36 KB weight tile 4x vs hold it

1024 tiles × 36 KB, 84 CTAs, 4 token passes:

```
stream    0.06 ms   issued 151.0 MB (4 passes; later passes hit L2)
resident  0.04 ms   issued  37.7 MB (one HBM read, then shared)
speedup 1.47x (memory-only)
```

This supports M1, but the gain is ~1.5x in a pure memory model, not 4x: with
one 36 KB tile per CTA, the re-streamed passes are L2 hits. In the real kernel,
TMA issue overhead and compute will reduce the gain further. Measure on H200
before the full-tile redesign.

### `split.cu` — split count vs merge cost

Pure KV-read model (no attention math):

```
B=32 span=80K: ns=1 922 GB/s, ns=4 870, ns=8 760
B=1  span=80K: ns=16 401 GB/s, ns=32/64/132 ~588
B=1  span=2K:  ns=8 51, ns=16/32 ~51-54
B=1  span=1M:  ns=16 504, ns=32 898, ns=64 857, ns=132 876
```

Takeaways:

- B=1 long context needs at least ~32 splits; 132 is not better in this
  read-only model.
- B=32 is already bandwidth-saturated at 32 CTAs; more splits only add merge
  overhead here. Attention math may still need the extra CTAs, so sweep it.
- Current heuristics (`NSDIV=128`, 132/B) are defensible for B=1 full, but a
  32/64-split arm should be checked per load.

## What this changes

- Keep **M1** high priority, but call the expected memory-only gain ~1.5x.
- Lower **A2** from med-high to med; 128 B rows are already near the roof.
- Lower **A3** confidence: 32 splits are enough for B=1 long context in a
  read-only model; test merge/compute before changing the heuristic.
- **A1** stays high, but the sm86 conversion ratio is not an H200 prediction.
- Add a check for the new page-table smem ring: the page table is ~4 B/token,
  so 80K tokens is 320 KB and likely L2-resident; staging removes dependent
  loads but adds copies. Measure it separately from row coalescing.
