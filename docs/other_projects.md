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
- Follow-up (same day): RMS_NORM+MUL -> F16 GEMM operand fusion for the VAE (frames identical to unfused; not timed, GPU shared with other jobs). SageAttention "fp16+fp32" PV accumulation (inst buffer) on Ampere was 1.4-2.4x slower than f32 accumulation: reverted. 5090 validation pod not run: awaiting the user's direct confirmation (plan: (a) 345a185 Q4_K, (b) new Q4_K, (c) new int8 te=cpu,vae=cpu, (d) (c)+SD_H3_VAE_SAGE=1; sd-server 896x672x124 20 steps spectrum, 2 requests each, 50 min TTL).

## daisy deploy (RTX 3090 sm_86, 2026-10-26 ~23:00 NZDT)
- Inventory before: both lanes image-only (omniserve built WITH_LLAMA=OFF / no libllama loaded; no llama.cpp on daisy, so LLM decode libs N/A).
  - omniserve-native :8791 Z-Image: bin 20261022-fuse (main 02a69a1), lib 20261022-fuse = /mnt/fast/code/sdcpp-prod-20261022 (byte-identical to sdcpp-opt base 689e7f7 + ggml ada26aab), te=cpu, cuDNN VAE.
  - omniserve-native-qwen :8792 ra2: bin /home/lee/omniserve-native-tiers/build-main (7213f12), lib /home/lee/sdcpp-ra2 (10-02 build of local stable-diffusion.cpp-master c0d6c1e / ggml f583f393 = sdmaster-opt base).
- Built on daisy (cuda 12.9, sm_86, same cmake flags as prior builds): omniserve main a406606 at /mnt/fast/code/omniserve-native-20261026 (build-z, build-q); libs /mnt/fast/code/sdcpp-zimage-exact-20261026, sdcpp-zimage-opt-20261026, sdcpp-ra2-opt-20261026 (git repos, base + applied patches). Artifacts + drop-ins + A/B scripts: /mnt/fast/code/omniserve-native-deploy/20261026-latest/{zimage,qwen,ab,nsys}.
- ra2: lib = sdmaster-opt 433ff24 + ggml perf/fusion 5b698ffc. omniserve A/B on :18791 (old bin+lib / new bin+old lib / new bin+new lib), 4 requests (turbo LoRA 512/1024/768x1024 + 30-step base): pixel md5 identical. gen 1024^2 13.63 -> 12.68 s, 768x1024 10.05 -> 9.64 s, base 512 5.37 -> 5.16 s (single samples).
- Z-Image: lib "exact" = ggml sink/rope_pe/rope_pe_ext + sd rope_pe (a5d5cff). Pixel-identical (4 omniserve requests + sd-cli). nsys --cuda-graph-trace=node, sd-cli 1024^2 9 steps: kernel 11.74/12.30 -> 11.20/11.53 s, sampling 13.2 -> 12.5 s. Served wall time not measurably different (GPU warms to 74 C and drifts 10 -> 16 s per image over back-to-back runs; new vs old binary equal within drift).
- Rejected: sdcpp-opt adaLN fold (ebf4fff) + kv_scale FA (74bb3dd, depends on fold): PSNR vs prod 45.7 / 37.2 / 23.0 / 21.1 dB on the 4 daisy requests (9 steps; divergence, not rounding).
- Drop-ins: /etc/systemd/system/omniserve-native.service.d/zzz-latest-20261026.conf (WorkingDirectory=/mnt/fast/code/omniserve-native-20261026, new bin+lib), /etc/systemd/system/omniserve-native-qwen.service.d/zzz-latest-20261026.conf (new bin+lib). Post-deploy: health 200, authenticated 1024^2 generations 200 on both.
- Rollback (each service): sudo rm <drop-in>; sudo systemctl daemon-reload; sudo systemctl restart <service>.

## Image-gen kernel pass 2 (2026-10-26, local 3090 Ti, sm_86) — Z-Image (sdcpp-opt) + Qwen-Image 2.1 VAE (sdmaster-opt)
All changes bit-identical (md5) to the previous graphs; every fusion has an env off-switch. Timing = nsys kernel sums;
the local GPU and daisy were shared with other jobs, so "min-est" = per-(kernel, grid) minimum x count (contention-robust).
Bench scripts/artifacts: /vfast/data/imgopt (zbench.sh, zq4.sh, lqv.sh, qrun.sh, zstep.py, qvae3.py).

### Z-Image turbo — sdcpp-opt perf/dit-fusion @8406ede, ggml branch perf/zimage-dit-fusion @18f1c867 (lib md5 b11ace08...)
- Prod config (Q4_K DiT + Flux ae VAE, 1024^2, 8 steps): DiT step min-est 616 -> 577 ms (raw 728 -> 670); VAE decode GPU 442 -> 330 ms
  (im2col 172 -> 81, bias add+convert 82 -> 61). Q8_0 DiT: 589 -> 560 ms (raw 713 -> 652). md5: Q4_K 5c46c96b..., Q8_0 262480b0...
  identical base vs new; Q4_K 4-step on daisy also identical.
- Fusions: RMS_NORM[->MUL]->ROPE_PE for q/k (port of h3 80d01170); RMS_NORM->MUL->MMQ writes the q8_1 activation directly (D4 and DS4,
  ffn_norm1 quantized once for w1+w3); SWIGLU[->SCALE]->MMQ quantizes silu(x)*g without the hidden tensor; w2's 1/128 output
  rescale folded into ffn_norm2 eps (exact pow2); tiled im2col 3x3/1x1 + half2; F16 GEMM -> row-bias ADD in the output conversion;
  conv2d N=1 skips the no-op permute copy. Remaining per step: MMQ 333 ms, FA 178 ms, elementwise ~50 ms (was ~110).
- MMQ experiments (no gain, reverted): int->float via magic-number add instead of I2F (-6%: MMQ is not I2F bound); mmq_x cap
  96/64/48 (-6/-20/-30%). q8_0 MMQ is ~33% of int8 peak on GA102 at these shapes (1 CTA/SM, 58 KB smem, no cp.async pipeline);
  needs a pipelined kernel or repacked aligned q8_0 weights. FA 42.8 TFLOPS (60% of fp32-acc peak); faster options change numerics.

### Qwen-Image 2.1 VAE — sdmaster-opt perf/master-dit-fusion @2ca573c, ggml perf/fusion @d38023c9 (lib md5 9543c59e...)
- Warm VAE decode 1024^2 (tile 32, overlap 0.5, 9 tiles): wall 2.62 -> 1.05 s; kernel min-est 1809 -> 766 ms (raw 2428 -> 1012).
  Full pipeline (6 steps) cold decode 3.62 -> 1.77 s. md5 03b3fb13... identical (full pipeline) and 9834df69... (VAE-only latent).
- Per stage (min-est ms): im2col 921 -> 139; cpy (permute conts) 251 -> 26; RMS_norm+mul+silu 113 -> 114 (now incl. both permutes);
  bias add + f16->f32 130 -> 89; concat 45 -> 7; GEMM ~330 unchanged.
- How: ggml_rms_norm_ch (RMS over ne[3], CUDA kernel reproduces rms_norm_f32's reduction order; fuses gamma + SiLU; smem column
  cache / float2 variant); tiled im2col; single-frame DupUp3D = nearest upscale of a channel-strided view; conv2d N=1 copy skip +
  GEMM row-bias epilogue; H3 ports: parallel tile split/merge (18459fd tiling.cpp) and cached cudaDeviceProp (799b5c01).
  Off-switches: SD_NO_NORM_CH, SD_NO_DUPUP_FAST, SD_NO_CONV2D_N1, GGML_CUDA_NO_IM2COL_TILED, GGML_CUDA_NO_GEMM_ROWBIAS,
  GGML_CUDA_NO_NORM_CH2. Bench hooks: SD_DECODE_LATENT_OUT/IN, SD_VAE_DECODE_REPEAT.
- --vae-conv-direct is now a regression (existing implicit-GEMM kernel 3.3 s vs ~0.47 s im2col+GEMM, and not bit-identical).
  omniserve branch perf/vae-conv-direct-20261026 (worktree ../omniserve-native-vaeconv @e0939cf) adds
  OMNISERVE_NATIVE_SD_VAE_CONV_DIRECT (default off); not recommended to enable.
- Remaining VAE: cuBLAS F16 GEMM 330 ms, im2col 139, norm 114, bias/convert 89, residual adds 42.

### Deploy notes (for the deploy agent; nothing deployed by this pass)
- Both branches build on the earlier opt branches, not on the exact live trees: port the commits onto the live source recipe
  (prod-deploys memory), rebuild for the target arch, then md5-compare same-seed images (expected identical).
- sdmaster-opt ggml_rms_norm_ch adds an op_params flag to GGML_OP_RMS_NORM; only CUDA and CPU implement it (sd.cpp uses it only on
  those backends).

## LLM decode pass 2 (2026-10-27, local 3090 Ti sm_86) — llama.cpp-opt branch perf/decode-kernels @5749d573b (on top of e49c629b7, not pushed/deployed)
- Served LLMs (omniserve-native repo config): main gateway = gemma-roleplay-v2-q8_0.gguf (Gemma-4 E4B arch, Q8_0), KV q8_0 + FA (forced), ctx 4096, 1 context, LLM_PERMITS 1 => batch-1 decode; swap profiles Boulesis 26B-A4B Q4_K_S (+MTP Q8_0), G4 31B IQ4_XS. Qwen3-4B / Qwen3-VL-8B Q4_K_M are sd.cpp text encoders, not llama.cpp decode; daisy has no llama.cpp. Proxies: gemma-4-E4B-it-Q8_0 (same arch/quant), Qwen3-4B-Instruct-2507-Q4_K_M (Q4_K/Q6_K GEMVs).
- Commits (default behaviour bit-identical):
  - d330b3acd q8_1 activations from producers (rms_norm[+mul[+add[+scalar mul]]], gelu*mul, hadamard fwht) so mmvq skips quantize_q8_1 (Q/K/V share one copy); head_prep kernel = rms_norm[*w] -> rope neox -> hadamard (+ the KV SET_ROWS, q8_0/f16/f32) per head; GELU in the unary*mul fusion.
  - 2829bfff1 two-slot persistent q8_1 cache; q8_1 output from the fused gate/up GLU GEMV (last-block quantize; opt-in since 5749d573b).
  - 61ed999b5 graph_optimize moves K/V MUL_MATs next to Q (same src1/type/K), one mmvq launch for all three.
  - e0aab0399 opt-in FA flags (below).
  - 5749d573b fixes an in-place race in the multi-block rms_norm+q8 kernel present in d330b3acd..e0aab0399 (gemma post-norms are allocated in place; md5 over 200 tokens did not catch it, 8K-token decode KLD did) => deploy the branch head only; adds opt-in GGML_CUDA_MMVQ_NW_TUNE.
- nsys (tg128, per-position median, us/token, gemma-4-E4B q8_0 KV, depth 0): e49c629b7 -> 5749d573b: total 8051 -> 7392, kernels 1394 -> 754. GEMV 5882 -> 5837 (302 -> 254 launches; SWA Q+K+V 7.7+3.3+3.3 -> 11.6), quantize_q8_1 378 -> 68 (301 -> 44), rms_norm 658 -> 529 (302 -> 212), rope/hadamard/set_rows 345 -> 346 (246 -> 132), elementwise 250 -> 69 (180 -> 49), FA 537 -> 543. Depth 4096 FA is 936 us/token (global D=512 layers: dequant of all K/V to f16 + MMA every token).
- tg128 t/s (llama-bench -fa 1 -ctk/-ctv q8_0, GRAPH_OPT unset, 3 interleaved rounds, shared GPU):
  - gemma-4-E4B Q8_0: d0 117.2 -> 126.1 (+7.6%); d4096 109.2 -> 119.0 (+9.0%); d16384 97.9 -> 105.7 (+8.0%).
    + FA flags: 128.5 / 124.1 / 118.7; + FA flags + NW_TUNE: 130.0 / 126.0 / 119.4 (+10.9% / +15.4% / +22%).
  - Qwen3-4B Q4_K_M: d0 207.0 -> 224.4 (+8.4%), + NW_TUNE 238.5 (+15.2%); d4096 173.4 -> 184.6, + NW_TUNE 192.6.
  - batched decode aggregate t/s (llama-batched-bench npp 256 ntg 64, npl 1/2/4/8, gemma): 110.1/215.1/411.1/665.3 -> 118.9/231.2/427.0/698.8 (+8/+7.5/+3.9/+5%); FA flags 122.5/232.5/436.0/695.8.
- Parity (default settings): greedy md5 over 200 tokens identical vs e49c629b7 for gemma (q8_0 KV 86c3a50c79b6, f16 KV e672e90a70b3, FA off f3a8046e0811, CUDA graphs off, GRAPH_OPT=1) and Qwen3-4B (q8_0 335e61bc3ba0, f16 825ff9939efd); 1500-token continuation of a 1000-token prompt md5 3d1c8fa99533 identical in 3/3 runs each. Stronger check: llama-perplexity -ub 1 (8192 decode tokens, ctx 4096, Shakespeare, --kl-divergence vs e49c629b7 logits): KLD 0 / same-top 100% in 7 of 8 gemma runs and the Qwen run; one gemma run showed Δp RMS 0.002% (KLD 1e-6, no top-1 change) — not reproduced in 7 reruns, cause not found.
- Opt-in, not bit-identical (quality = KLD over the same 8192 decode tokens vs an f16-KV MMA prefill reference; for scale f16 vs q8_0 KV differ by KLD 0.0065 on gemma, 0.0024 on Qwen):
  - GGML_CUDA_FA_VEC512=1 (gemma global layers D=512 q8_0: vec kernel on the quantized cache) + GGML_CUDA_FA_VEC_SPLIT=1 (vec kernel KV tiles of 128 instead of D): FA kernel time/token d0 533 -> 432 us, d4096 948 -> 559 us. KLD vs ref: baseline decode 0.00488, FA flags 0.00408 (closer to the reference; same-top 96.53 -> 96.68%).
  - GGML_CUDA_MMVQ_NW_TUNE=1: 5 warps instead of 4 at one column when K splits into whole iterations (K=2560: Q4_K/Q6_K 10 superblocks, Q8_0 80 blocks). Qwen Q4_K gate/up 42.5 -> 37.3 us. KLD vs ref: gemma 0.00497 (baseline 0.00488, within ±0.00024), Qwen 0.00458 (baseline 0.00495; same-top 96.36 -> 96.90%). SPLIT is a no-op on Qwen (D=128) at these depths.
- Not kept: chained post-norm + next pre-norm in one kernel (5.1 us vs 2.6 + 2.2, and not bit-identical); GLU q8_1 output default-on (fence/atomic tail ~= saved launch, opt-in GGML_CUDA_GLU_Q8=1).
- GRAPH_OPT=1 on gemma-4: the QKV concurrency events never validate (overlapping write ranges), but the interleaved node order stays and breaks the Q/K norm fusions (1048 -> 1096 kernels/token on the pre-pass build). Leave GRAPH_OPT unset for gemma.
- Off switches (all default on): GGML_CUDA_NO_NORM_Q8, GGML_CUDA_NO_HEAD_PREP, GGML_CUDA_NO_HEAD_SET_ROWS, GGML_CUDA_NO_MMVQ_MULTI.
- Remaining per token (gemma d0, 7.4 ms): GEMV 5.84 ms (gate/up 2.6 at ~87% of 1008 GB/s, down/O 1.68, lm_head 0.75 at 94%); Q8_0 rows are 34-byte blocks (2-byte aligned loads) — a repacked aligned Q8_0 layout is the next big lever but changes weight storage. Norms 0.53 ms (in-place post-norms need the single-block kernel, 2.96 us). FA vec at short depth is latency-bound (8 blocks/token/layer without SPLIT).
- Deploy (user decides; nothing deployed): port e49c629b7..5749d573b onto /sdb-disk/code/llama.cpp-omniserve-opt (perf/decode-kernels-20261024 already has the two earlier commits), build sm_120 (changed files compile for sm_120 with CUDA 12.9), validate with the same md5 recipe on the 5090 (expect identical with flags unset) plus ideally the -ub 1 KLD run. Flags to consider in the drop-in after review: GGML_CUDA_FA_VEC512=1 GGML_CUDA_FA_VEC_SPLIT=1 (gemma, largest gain at 4K ctx); NW_TUNE only matters for K-quant models.

### Q8_0 aligned repack (2026-10-28, local 3090 Ti) — llama.cpp-opt branch perf/q8-repack @7f327f636 (on 5749d573b, pushed to `lee`, not deployed)
- Load-time repack of Q8_0 MUL_MAT weights (ne0 % 256 == 0, ne0 >= 1024) into a CUDA<n>_Q8R buffer type: each row keeps its 34*nb bytes as [32*nb quants][nb fp16 scales], so quants are 32-byte aligned. mmvq reads them with 8-byte loads (pseudo type GGML_TYPE_Q8_0R, same thread->block/chunk mapping and sum order as Q8_0 => bit-identical). MMQ gets its own Q8_0R instance with a 4-byte-aligned tile loader. set/get_tensor convert, so host readers still see plain Q8_0. supports_op keeps every other op off repacked tensors, LoRA tensors go to the default buffer, and any other MUL_MAT path unpacks to a pool buffer. Default on, GGML_CUDA_Q8_REPACK=0 disables. Same VRAM, load time unchanged (~2 s warm).
- Parity: greedy md5 over 200 tokens is identical to 5749d573b for gemma-4-E4B q8_0 KV (86c3a50c79b6), f16 KV (e672e90a70b3) and FA off (f3a8046e0811). llama-perplexity logits files with -ub 1/4/7/512 (repack on vs off) are byte-identical. Qwen3-4B Q4_K_M has no Q8_0 weights, so it is a no-op (335e61bc3ba0).
- nsys, gemma-4-E4B tg128 d0, us/token, per-position median (5749d573b -> repack):
  - fused gate/up: 2652 -> 2558 (63.2 -> 60.9 us/call, 881 -> 915 GB/s = 91% of 1008).
  - down/O: 1681 -> 1675 (down 31.3 -> 31.1 us = 89%; O 7.5 us = ~74%, latency/launch bound).
  - lm_head: 752 -> 752 (94%).
  - QKV multi: 314 -> 310.
  - other small GEMVs: about unchanged. proj (K=256) is excluded: the small-K variant was slower repacked.
  - Totals: GEMV 5772 -> 5668 (-104 us, -1.8%); all kernels 7380 -> 7273 us/token (-1.4%).
- tg128 t/s (llama-bench, 6x5 interleaved samples, max): d0 125.3 -> 127.9 (+2%). d4096 and pp64/pp512 were inside the noise: other jobs held the GPU at 95-100% intermittently and the host was at load ~28. Prefill MMQ kernel time is about -2% on nsys. An earlier runtime-tagged loader inside the Q8_0 MMQ kernel cost the plain Q8_0 path ~4%, which is why the repacked layout has its own instance.
- Finding: 2-byte-aligned loads were not the bottleneck. The big GEMVs already ran at 88-94% of peak (L1 merges the 16-bit loads into full sectors). Tried and dropped:
  - a 16-byte paired layout for the 4-warp mapping: same decode, ~2x MMQ sector waste (prefill -20%);
  - prefetching tail blocks;
  - L1::no_allocate loads.
  What remains is per-kernel fixed cost on small GEMVs (O, QKV, inp_gate: ~1.5 us each). Only fusion, or PDL on sm_90+/sm_120, removes it.
- Deploy notes: port 7f327f636 onto the live tree after the decode-kernels commits. It adds q8r.cu/.cuh, a new MMQ instance file and llama loader/adapter hooks. Build for sm_120 and validate with the md5 recipe (expect identical, flag on and off). The gain on the 5090 is unmeasured and probably similar or smaller. GGML_CUDA_Q8_REPACK=0 is the rollback switch with no rebuild. Risk areas: tensor-parallel/2-D set_tensor paths assert on repacked tensors (not used by omniserve), and the MTP Q8_0 draft model will also be repacked.

### Z-Image MMQ pipelining (2026-10-28, local 3090 Ti) — sdcpp-opt perf/mmq-pipe @e1a5797, ggml perf/mmq-pipe @03c485ec (pushed to `lee`, not deployed)
- Result: no speedup. The pipelined kernel is committed opt-in only (GGML_CUDA_MMQ_PIPE=1, default off). The default path is unchanged.
  Images are md5-identical with the flag unset and set: Q4_K prod 8-step 5c46c96b, Q8_0 3-step 262480b0.
  e2e nsys min-est per DiT step (GPU shared, ±1-2%): Q4_K base 625/625, branch default 633/622, pipe 638/641 (MMQ 361 -> 373 ms).
  Q8_0: base 607, pipe 646 (MMQ 325-347 -> 402).
- Limiter (no ncu: RmProfilingAdminOnly=1). Ablation of the stock kernel at 10240x4128x3840, Q8_0/Q4_K:
  full 2.67/2.96 ms; tile loads skipped 1.90/2.36; vec_dot skipped 1.58/1.37. So load and math do not overlap.
  But the board is power-capped. The full kernel holds 450 W at ~1.9 GHz (throttle reason SW power cap), and the compute-only
  variant already draws 446 W. Overlapping the loads only moves energy around, so the time stays the same.
  Microbench (pure mma.m16n8k32 s8): 330 TOPS at ~370 W. With the Q8_0 epilogue (I2FP+FMUL+FFMA per element): 214 TOPS at 404 W.
  MMQ runs at ~113-125 TOPS.
- Tried; all bit-identical (hash of the full output vs base across many shapes, incl. need_check, stream-k and batched),
  none faster in interleaved sustained A/B:
  - cp.async 2-stage staging of raw Q8_0 rows plus unpack, 8 warps: Q8_0 +12-20%, Q4_K +5%.
  - Q4_K: 3-stage, nibbles read with ldmatrix straight from the staged rows (no qs unpack): ~0-2%.
  - 16 warps/CTA (128 regs): compute-only got slower (Q8_0 1.91 -> 2.25 ms).
  - 256-row Q4_K tiles (-33% L2->SM bytes): -8..+7% burst, +7% sustained.
  - y tile only, cp.async double-buffered inside the stock kernel: neutral.
  - Q4_K y (d,s) pre-converted to float2 (drops 2 HADD2 per IMMA): +2%.
  - Grouped tile raster (L2 locality, G=4..80): neutral/worse.
  - 2 CTAs/SM at mmq_x=64: about neutral.
  - Not bit-identical, timing only: dropping the Q4_K min-term FFMA gave only -2%. The epilogue math is not the lever either.
- Takeaways for the 5090 (575 W, sm_120): the opt-in kernel might behave differently there, but this is unmeasured. On GA102 the
  remaining MMQ levers are fewer joules per output (fewer MACs, i.e. lower-bit weights or fewer DiT tokens), not scheduling.
  test-backend-ops GGML_PERF_DIT_TOKENS now also covers Q4_K/Q6_K weights. A bit-exact checker (hash of the CUDA mul_mat output) is in the
  agent scratchpad only.

### omniserve-native LLM sampling (2026-10-28, local 3090 Ti) — branch perf/llm-sampling @ origin (not deployed)
- Commit: one commit cherry-picked onto origin/main (the local wip base carried an unpushed commit that the gitleaks hook flags; the local copy is perf/llm-sampling-local). Model: prod gemma-roleplay-v2-q8_0 + E4B MTP head, prod batching env, MTP draft 2. Libs: llama.cpp-latest (omni) and llama.cpp-opt (prod lineage).
- Finding: with the existing exact cut active, local `sample=` is already 0.33-0.45 ms/step (gather 190-250 us/row, chain 15-35 us/row). Prod's 3.75 ms (gather 0.70 + chain 1.95 ms/row) matches the full-vocab path: with LLM_PREFILTER=0 it is gather 201 + chain 439 us/row, 1.17 ms/step here, and the 5090 host's CPU is about 4x slower under load. max_new is clamped to the 4096 ctx, so the old k > 8192 bail never fires. Prod requests are therefore probably skipping the cut: min_probability > 0, repetition_penalty < 1, or negative frequency/presence penalties. Check the request params arriving at :8791 before expecting the plan's +20%.
- Change: (1) The cut window is top_k + distinct history tokens and grows as tokens are accepted. It no longer includes + max_new, and the 8192 cap is gone. (2) The chain is applied one sampler at a time. If the post-penalty top (k+1) contains a tied logit, the row is redone on the old request-wide set. That makes the tokens bit-identical to the old path by construction. Ties occur on about 0.3% of rows. (3) Opt-in LLM_GPU_CUT=K (current llama.cpp API only, needs n_outputs_max_per_seq): a per-sequence backend sampler keeps the full row and adds a ggml_top_k of each vocab half. The host reads 2K (id, logit) pairs and keeps the ones above the larger half-minimum; it falls back to the host cut if fewer than k remain. (4) Fixed a latent overflow: a capped verify-row gather was being redone into the 32K vcand buffer.
- Parity (fresh servers, unique prompts, SHARE_PREFIX=0): 24 prompts x {greedy, temp 0.9 seed 7+i} x {DRY 0.8, rep 1.1, both}, 192 tokens. 144/144 md5-identical to base for the new CPU path (latest and prod-lineage libs), for GPU_CUT=1024, and with PAR_GATHER=1. Long generations (max_tokens 8192, 4 x 3 configs, ~27K tokens): 12/12 identical with GPU_CUT. Caveat: with prefix sharing on, back-to-back runs land in different prefix-cache states (slot release timing), so outputs vary even base vs base. Parity needs unique prompts or SHARE_PREFIX=0.
- Sampling per step (engine trace): batch 1, 192 tokens: base 0.33-0.45 ms, new CPU 0.33-0.42, GPU_CUT 0.04-0.09 (gather 4-6 us/row). Long generations: base 0.53-1.78 ms (the k bound with max_new inflates the set), GPU_CUT 0.05-0.09. Batch 4-8: base 0.74-0.85, GPU_CUT 0.10-0.12. Chain 20-37 us/row in all variants. Decode time with GPU_CUT is unchanged within noise (steady 9.8-9.9 ms vs 9.8-10.0). The host gather is DRAM-bound: reading the DMA'd 1 MB row costs ~120 us cold even when quiet, and prefetching or the thread pool did not help (PAR_GATHER under load: 1.2-2.5 ms/step, worse).
- Tokens/s: wall clock is unusable here. Another job held the GPU at 98% SM, and the same config ranged 14-131 tok/s. Trace-derived numbers from the quietest windows, batch 1, s/g_dry: base d2 11.02 ms/step -> 149 tok/s; GPU_CUT d2 10.86 ms -> 149; base d1 10.15 ms -> 143; GPU_CUT d1 9.96 ms -> 146. The GPU_CUT step-time saving (~0.35 ms/step, ~3%) is real but hidden in dec noise. Batch 4/8 wall: base 208-212 / 308-325, GPU_CUT 177-260 / 237-343 (noise).
- MTP draft A/B (same tokens, md5-identical d1 vs d2): position-1 acceptance 45%; position 2 is accepted 37-41% of the time position 1 was, 17-19% overall (plan: 11% on prod). Tokens/step: 1.45 (d1) vs 1.64 (d2). Step cost: +0.9 ms for d2 (draft decode +0.55 ms, verify row and sampling). Net here: d2 ~ +2-4%, within noise. On prod, each extra verify row costs ~2.6 ms of CPU sampling on the full-vocab path, and position-2 acceptance is lower, so set MTP_DRAFT=1 there unless GPU_CUT is on (or the full-vocab fallback cause is fixed). With cheap sampling, d1 and d2 are a wash; keep d2 only if the 5090 shows the same position-2 acceptance as here.

## Z-Image runtime-LoRA fusion (2026-10-28, local 3090 Ti sm_86) — sdcpp-opt perf/zimage-lora-fusion @bb9255c, ggml perf/zimage-lora-fusion @4b347416 (pushed to lee; not deployed)
Worktree /vfast/data/code/sdcpp-lora (base perf/dit-fusion 8406ede / ggml 18f1c867). LoRA = prod `the_look` (rank 32, bf16, 240 linears incl. adaLN; copied to /vfast/data/imgopt/lora/the_look.safetensors).
- Metric (Q4_K DiT + Flux ae, 1024^2, 8 steps, --vae-tiling, kernel min-est over 3 interleaved rounds, LoRA minus no-LoRA per step):
  old runtime path **98.1 ms/step** -> fused **57.4 ms/step** (-41%). Sampling wall with LoRA 9.11 -> 8.77 s (no-LoRA 8.2 s). No-LoRA graph/time unchanged.
  Target <= 25 ms/step not reached on the 3090 Ti. Remaining: MMQ epilogue +33 ms, downs 18 ms (my exact kernel 9.0 for qkv + w1/w3, cuBLAS 9.3 for out + w2), w2 f32 dual write 6.3 ms.
  Projection for the 5090 (TF32 tensor 2.6x, bandwidth 1.8x of the 3090 Ti): roughly 20-25 ms/step vs ~70 now; not measured.
- Where the old 98 ms went: cuBLAS TF32 up-GEMMs with beta=1 onto the base output (read+write of every output, ~50 ms), downs 18 ms, and lost
  pass-2 fusions (w2 1/128 scale passes 17, swiglu materialized 17, norm->q8_1 lost 13). Per-step concat/pad/scale of weights was only ~2 ms.
- What changed:
  - sd.cpp: LoRA weights prepared once per multiplier (stacked q/k/v downs, scaled block-diagonal ups, same ops as before) into a backend buffer
    (+~360 MB VRAM); the graph emits ADD(diff, base) with the LoRA GEMMs before the base GEMM; w2 keeps the 1/128 scaled output with the diff
    taken from the scaled activation (exact, pow2); w1/w3 share one R=64 down; Q8_0 pow2 activation-scale skip as in the direct path.
  - ggml-cuda: (1) MMQ LoRA epilogue: dst = mmq + U·lx computed with m16n8k8 TF32 MMA, k ascending, RNA operands, one add — bit-equal to cuBLAS TF32
    SGEMM beta=1; all-zero U chunks skipped (q/k/v block diagonal) via a chunk mask; noinline call after write_back (inline versions push the MMQ
    main loop to 255 regs and slow it). (2) RMS_NORM->MUL and SWIGLU->SCALE producers feeding MMQs + LoRA downs: one kernel writes the shared
    q8_1; for the norms the downs read raw rows + per-row norm scale (no f32 normed rows). (3) lora_down_tf32: exact non-split TF32 down GEMM.
- Parity: md5-identical to the previous runtime path at 1024^2 for 6 prompts (4 scenes + 2 portrait/anime) x scales 0.6/0.8, at 1344x768,
  Q8_0 DiT, LoRA rescale 0.6->0.8->0.6 and LoRA off/on in one sd ctx (library harness), and no-LoRA. **Not** identical at 768x1024 (and so
  1024x768): there cuBLAS itself picks FP32 SIMT SGEMM for the out/w2 ups and split-K=3 for the qkv down, so the old bits are a cuBLAS-heuristic
  artifact; PSNR 24.9 dB, same composition. Sensitivity reference: changing only the TF32 operand rounding of the LoRA up (truncate vs RNA)
  gives 31 dB at 1024^2, i.e. nothing short of bit-exact can meet a 45 dB bar at 8 steps.
- Off switches: SD_NO_LORA_FUSE=1 (exact previous graph), SD_NO_LORA_GROUP=1 (w1/w3 separate downs), GGML_CUDA_NO_LORA_FUSE=1,
  GGML_CUDA_NO_LORA_EPI=1, GGML_CUDA_NO_LORA_DOWN=1. With only the ggml switch the grouped w1/w3 down goes to cuBLAS (M=64, different kernel): valid
  TF32 but not md5-equal to the old path; use SD_NO_LORA_FUSE=1 for the exact old behaviour.
- Tests/bench: test-backend-ops `-o LORA_LINEAR` (eval vs CPU, need_check and 2-chunk cases), GGML_PERF_LORA_TOKENS=4128 perf mode;
  /vfast/data/imgopt/{zl.sh,zagg.py,zgrid.py,parity.sh,lorab3.sh}.
- Deploy notes (for the deploy agent): port both commits onto the live Z-Image recipe (needs the pass-2 fusions it builds on, i.e. task 2 first),
  build sm_120, then on the 5090 compare old vs new lib md5 at each prod size with LoRA 0.6/0.8. cuBLAS heuristics on sm_120 differ, so expect
  identity only where cuBLAS used non-split TF32 GEMMs; where it differs, decide on visual A/B (the fused path is deterministic and shape-independent).
  VRAM +~360 MB per loaded LoRA. Daisy could not be used for timing (19 GB resident, 1024^2 OOMs).
