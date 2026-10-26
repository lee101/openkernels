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

### PROD DEPLOY 2026-10-24 09:18 UTC — ra2 (omniserve-native-qwen, 5090)
- Lib: /nvme0n1-disk/models/omniserve-native/qwen-image-2.1/lib/libstable-diffusion-9bc7b35-cfgend-fusion.so (md5 b76bb585...), built from /nvme0n1-disk/code/sdcpp-ra2-fusion branch perf/ra2-fusion-20261024 (= live 9bc7b35-cfgend tree incl. its uncommitted cfgend/guidance + ggml local diffs, plus rope_pe, NORM fusion/sink, Flux modulate, Wan single-frame conv2d).
- Validation on the 5090: sd-cli with live lib vs new lib, same seed: bit-identical PNGs (512^2 4 steps, 512^2 8 steps, 1024^2 8 steps). nsys kernel time 1024^2 8 steps: DiT 3959 -> 3517 ms, VAE 2396 -> 2047 ms (-12% total). Wall A/B too noisy under live load.
- Drop-in /etc/systemd/system/omniserve-native-qwen.service.d/zzzzzzzzzz-fusion-20261024.conf. Rollback: delete it, daemon-reload, restart omniserve-native-qwen.
- Post-deploy: active, process maps the fusion lib, /health 200, authenticated 512^2 generation 200 in 7.8 s.

### PROD DEPLOY 2026-10-24 09:38 UTC — omniserve-native main gateway (5090): Z-Image lib + LLM ggml libs
- Drop-in /etc/systemd/system/omniserve-native.service.d/zzzzzzzzzzzz-fusion-20261024.conf: OMNISERVE_NATIVE_SD_LIB=/nvme0n1-disk/code/omniserve-native-deploy/sd-20261024-fusion/libstable-diffusion.so (md5 ba1573d6...), LD_LIBRARY_PATH=/sdb-disk/code/llama.cpp-omniserve-opt/build/bin. Rollback: delete it, daemon-reload, restart omniserve-native.
- Z-Image source /nvme0n1-disk/code/sdcpp-zimage-fusion branch perf/zimage-fusion-20261024 = exact sd-20261022-fuse recipe (md5-verified base) + adaLN fold, rope_pe, kv-scale folded flash attention. 5090 A/B: PSNR 39 dB, elementwise kernels ~1.9 s -> 0.63 s per 8-step image, ~15% less DiT kernel time.
- LLM source /sdb-disk/code/llama.cpp-omniserve-opt branch perf/decode-kernels-20261024: rms_norm register rows + mmvq 2 rows/block; bit-identical (md5) on the 5090; mmvq FFN 41.4->36.8 us, down 15.8->12.0 us, rms_norm 3.79->1.78 us.
- Live after restart: health 200 in 5 s; background 1024^2 s=8 gen median ~4.5 s -> ~3.5 s (no LoRA) / ~4.1 s (LoRA), all ok=1; chat 200, coherent.

## H3 video on sd.cpp vs the ComfyUI cog (2026-10-25)
- Branch `perf/h3-20261025` (sdmaster-h3opt, ggml `perf/h3-fusion`): 25-29% faster per video at small sizes on the 3090 Ti (448x256x22, 640x352x39; DiT launches 8878 -> ~1400/step), bit-identical (or 55-58 dB with VAE flash attention).
- Production setting on a RunPod RTX 5090 (same GPU class as the live `gpu-rtx5090` cog), 896x672x124 frames, 20 steps, res_multistep/simple, cfg 1, Q4_K_M DiT + Q4_K_M Qwen3-VL TE, --offload-to-cpu, --diffusion-fa: total 313 s (text encode 12.5 s, sampling 244 s = 12.2 s/step, VAE decode 54 s); with --cache-mode spectrum 207 s (sampling 141 s, decode 45 s).
- Live ComfyUI cog (int8_convrot DiT, NVFP4 TE, Sol-Attn INT8 + Spectrum) on the 5090: 48-72 s generation (manifoldgen docs/old-anime-1920-results, same size/frames). sd.cpp is 3-5x slower at production size: attention-bound at ~73K tokens (ggml FA is fp16 mma; the cog uses INT8 attention + step caching), plus a 54 s VAE decode. Not deployed.
- 5090 (RunPod) production-size follow-up, same prompt/seed: SageAttention (`--sage-attn`) 14.6 -> 9.1 s/step; DiT+VAE resident (`--params-backend te=cpu`, no --offload-to-cpu) 7.1 s/step and VAE decode 42 -> 29 s (offload re-streamed 34 GB of VAE weights per decode). 20 steps + `--cache-mode spectrum`: 121.7 s total (TE 18.4, sampling 72.4, decode 29.0) vs 313 s baseline; without spectrum 288.6 s. Remaining gap to the ComfyUI cog (~60 s): TE runs from host-mapped params (16 GB staged in 0.36 s, then 12.5 s compute = PCIe reads; on-GPU TE 5.6 s but auto-fit never evicts it before the DiT, OOM at block 37), VAE decode 29 s, DiT 7.1 s/step. CUDA graphs: VAE decode slower (per-tile recapture).

### H3 sd-server warm bench (RunPod 5090, 2026-10-24)
896x672x124f, 20 steps res_multistep, spectrum, --diffusion-fa --sage-attn --params-backend te=cpu. sd-server has no -M flag.
- e2e: cold 126.0s, warm-same 110.4s, warm-new 110.3s (~8s HTTP/encode overhead).
- warm stages: TE 7.0s (cold 17.6s), sampling 68.5s, VAE decode 25.3s, generate 102s.
- Gap vs cog (48-72s): sampling ~3.4s/step and VAE 25s. Next: int8/fp8 DiT GEMMs instead of Q4_K MMQ, fused VAE decode, TE on GPU then evict.

### H3 sd.cpp pass 2 (2026-10-26, local 3090 Ti) — sdmaster-h3opt perf/h3-20261025 @135c287, ggml perf/h3-fusion @latest
A/B vs 345a185 (same flags: --diffusion-fa --sage-attn --params-backend te=cpu, Q4_K_M DiT, seed 42):
- 640x480x56: DiT step 3.40 -> 3.03 s/it (latent bit-identical); warm VAE decode 4.63 -> 2.49 s, first 8.32 -> 3.60 s; audio decode 4.15 -> 0.87 s; all frames pixel-identical.
- 896x672x124 (production size), 2 steps: DiT 17.87 -> 16.73 s/it (MMQ 49%, sage f16 41% at 23.7K tokens); warm VAE decode 35.4 -> 19.0 s (140 tiles); frames pixel-identical.
- VAE: host side was ~35% of decode (serial cache-hostile tile merge, growing per-chunk concat, runner_end + workspace realloc per chunk, cudaGetDeviceProperties per graph); now parallel/recycled. GPU side: F16 cuBLAS GEMM epilogue fusion (bias, bias+swiglu->f16, bias+gated residual), RMS_NORM->ROPE(PE) fusion, K/V written once as F16 FA operands. SD_H3_VAE_SAGE=1 (opt-in): another -8%, PSNR 56-58 dB.
- DiT: sage path builds q/k/v straight from the packed QKV (was slice/rope/concat/permute/scale/cast); segment graphs now carry use counts (all CUDA fusions were silently off in segmented/streamed compute).
- int8: sd.cpp already loads the cog's ComfyUI int8_convrot DiT (Comfy-Org/MiniMax-H3 diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors, sha256 e889202c..., local copy /vfast/data/h3work/weights). cuBLAS int8 GEMM is 1.9-2.1x Q4_K MMQ on Ampere (all 4 H3 shapes, 6.6K and 23.7K rows); DiT GEMM time 2.0 -> 0.86 s/step at 640x480x56. Added GEMM->swiglu->convrot-quantize and GEMM->gated-residual int8 epilogues (bit-identical). 21 GB weights do not fit a 24 GB card with compute; on a 5090 run with --params-backend te=cpu,vae=cpu.
- TE: warm sd-server request 1.2 s (640x480) / 2.2 s (896x672, 51 streamed segments) on the 3090 Ti; the 5090's 7.0 s warm TE was not reproduced locally.
- Remaining gap to the cog: Sol-Attn (block-sparse int8 attention, ~35-40% of a 5090 DiT eval is dense sage), cog Spectrum is more aggressive (degree 1, warmup 1).
