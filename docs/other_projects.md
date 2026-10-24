# Other projects (agent passes, 2026-10-24)

## omniserve-native — branch perf/agent-1024, worktree ../omniserve-native-opt (not pushed/deployed)
- 7cc4da6 C Nyquist notch 9.2 -> 2.9 ms/1024^2 (byte-identical)
- bb85047 numpy notch 40 -> 26 ms; drop base64 decode for byte count
- 7462d5e matte pass 1.56 -> 1.15 ms (fused resizes, fewer launches, no stream sync under g_lock; OMATTE_CUDA_SYNC=1 restores)
- 62435db matte 1.15 -> 0.69 ms (register arrays; not byte-identical, max abs 5.6e-5, revert alone if exactness needed)
- Next (needs quiet GPU): overlap CPU encode/notch with next request at cog_pod/handler lock level; BiRefNet torch.compile/attention path; tiled matte sweeps; llama.cpp backend sampling.

## manifoldgen-site — branch perf/agent-1024, worktree ../manifoldgen-site-opt (not pushed/deployed)
- 158f1da h3 router: encode prompt once + cached anchor embeddings (~16x fewer encodes); 60 s workersMax cache
- f314faa same cache on swap/animation submit path
- 3de1ef2 h3-swap cold start: weights fetch overlaps ComfyUI boot (~15-40 s/cold worker); concurrent media downloads
- 6d5cff6 h3-orbit same overlap
- 9bc15ef h3-swap/control: boot-time preload thread
- a758118 voice-enhancer: YuE weight hash warm-up overlaps job start (~27 s first render)
- Next: keep YuE alive between batches; reuse ABC plan across seeds; fp8/vllm for YuE AR; cache text conditioning across chunks.

## Kernel passes by main session (2026-10-24 evening, local 3090 Ti)

### LLM decode — llama.cpp branch perf/decode-kernels (worktree ../llama.cpp-opt, base = llama.cpp-omni + its local patches)
- gemma-4-E4B Q8_0 tg128: 120 (omni default) -> 128 (GGML_CUDA_GRAPH_OPT=1, env only) -> 131 t/s (+ commits below). Greedy output bit-identical (md5 over 200 tokens).
- 81c1e4f06 rms_norm keeps rows <= 4*block in registers: fused norm 4.07->2.71 us, ~290 us/token.
- e49c629b7 mmvq 2 rows/block at ncols_dst=1 (per-row sum order unchanged): fused FFN gate/up GEMV 73.4->63.0 us. 4 rows/block is worse.
- Depth 4K/16K: +3% holds (125/119 vs 121/116). Attention is ~9% at 16K; decode is GEMV + launch-floor bound (~1235 kernels/token).
- Next: fold quantize_q8_1 into the norm when it only feeds Q8 GEMVs; GRAPH_OPT into the prod drop-in after omniserve parity run.

### Z-Image turbo DiT — stable-diffusion.cpp branch perf/dit-fusion (worktree ../sdcpp-opt, base = sd.cpp-prod incl. its uncommitted prod diffs)
- DiT step 1024^2: 807 -> 758 (norm folds) -> 688 ms (rope_pe). 3-step sampling 2.61 -> 2.27 s (-13%). Image: rope_pe bit-identical; norm fold PSNR 36.7 dB vs prod (rounding only, visually identical).
- z_image: (1+scale) and tanh(gate) folded into RMSNorm weights; ggml-cuda sinks RMS_NORM next to its MUL so RMS_NORM+MUL(+ADD) fuses (buffer-overlap guarded; GGML_CUDA_NO_NORM_SINK=1 disables).
- ggml_rope_pe (GGML_OP_ROPE mode 128, CPU+CUDA): precomputed-table rope in one pass (SD_NO_ROPE_PE=1 disables). Qwen-Image uses the same Rope::attention, so ra2 gets it too.
- Remaining per step: MMQ 359 ms (~47% of int8 peak), FA 181 ms (~60% of fp32-accum peak), elementwise ~150 ms (kv_scale scale/cast passes, silu-gate, quantize).

### Qwen-Image 2.1 (ra2 path) — stable-diffusion.cpp-master branch perf/master-dit-fusion (worktree ../sdmaster-opt, ggml branch perf/fusion)
- 1024^2, 6 steps, 3090 Ti: total 12.2 -> 10.7 s (bit-identical image); DiT window 1934 -> 1786 ms; VAE decode 3.75 -> 2.83 s (2.44 s with --vae-conv-direct, PSNR 58.7 dB).
- ggml-cuda: NORM+MUL(+ADD) fusion (adaLN modulate in one pass), NORM sink; concat in one launch when contiguous (Wan VAE decode issued 181k concat launches, one per channel slice).
- Flux::modulate = x*(1+scale)+shift; rope via ggml_rope_pe (shared with Z-Image); Wan VAE single-frame causal Conv3d -> Conv2d with last temporal tap (exact; also handles singleton-kt exports).
- Remaining VAE: 2D im2col 0.92 s, cpy 0.6 s (DupUp3D permute conts), GEMM 0.36 s. omniserve has no env for conv-direct yet (OMNISERVE_NATIVE_SD_VAE_CONV_DIRECT would need adding in backend_sd.c).
- Note: sd.cpp-prod (Z-Image) and sd.cpp-master share one git repo; worktree branches must have unique names.
