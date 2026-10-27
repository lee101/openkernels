# Production inference stack: survey and optimization plan (2026-10-28)

Sources: live unit files and drop-ins on prod (5090, `systemctl cat`, read-only), the gateway journal (Z-Image `[image]` lines, 2026-10-27 17:00 to 21:50 UTC, plus 48 h counts), `/metrics` on :8791/:8792, `/nvme0n1-disk/data/omniserve-frontier/gateway-ra2.jsonl`, `/var/log/supervisor/text-generator-*.log`, and the art-farm journal. Also omniserve-native `performance/*.md` and README, and `docs/other_projects.md`.
No local GPU runs were made: the 3090 Ti was at 100% utilization with 18.8 of 24.5 GB in use. All numbers below come from prod logs or earlier passes.

## Where the 5090 goes

| Lane | Prod volume | GPU share |
|---|---|---|
| Z-Image (`:8791`, embedded sd.cpp) | ~600 img/h, ~99% `background` tier (art farm via cutedsl :8100 plus other producers) | **~63% of wall time** (600 x ~3.8 s) |
| Kokoro TTS (`:9085`, via `:9083`) | ~550 req/day | <1% |
| LLM gemma-roleplay E4B (`:8791`) | 8.1K generated tokens in 36 h (engine counters) | <1% |
| Qwen-Image 2.1 ra2 (`:8792`) | ~90 req/week (73 background turbo, 14 free, 1 paid in the week of 10-22) | ~1-2% |
| BiRefNet / RVM (`:9094`) | about 1 job per 48 h | ~0 |
| ASR | local worker disabled; router `:9096` falls back to `:9083`, where `STT_BACKEND=gemini` (cloud). No STT lines in 3 days of logs. | 0 |
| YuE (`:9106`) | local worker inactive since 10-23; all music runs on RunPod serverless | 0 local |
| H3 video | RunPod serverless 5090 cog (ComfyUI, int8_convrot DiT, Sol-Attn, Spectrum); fal H3 Max for plain requests | 0 local |

The 5090 has 7.5 GB free with every tenant warm. Gateway 5xx: 982 out of 50.8K in 36 h, almost all `503 /v1/images/generations` caused by VRAM-lease denials (978 denied leases), about 20 s apart, which matches the art farm's `--retry-delay 20`. Daisy (3090) is image-only overflow and nearly idle (27 Z-Image requests in 48 h).

## Workload table

| Workload | Model / quant / runtime (prod) | Request shape | Current perf (measured) | Top levers | Est. gain | Effort |
|---|---|---|---|---|---|---|
| **Z-Image** | Z-Image-Turbo DiT Q4_K GGUF, Qwen3-4B Q4_K_M text encoder, Flux `ae` VAE (cuDNN conv, tiles 80x80). sd.cpp lib `sd-20261024-fusion`, embedded in the C gateway `deploy-prod-20261026-l0`, resident on the 5090. LoRA `the_look` (rank 32, bf16, 240 linears) is auto-routed at scale 0.6 or 0.8 and applied **at runtime**, because the weights are quantized. | 8 steps, batch 1 (`SD_MAX_BATCH=1`). 1024² is 65% of requests, the rest 768x1024, 1024x768, 768x1344 and 1344x768. LoRA on 69%. WebP output. | gen p50 / p90 at 1024²: **3.41 / 3.45 s** without LoRA, **4.01 / 4.08 s** with LoRA. 768x1024: 2.57 s / 3.13 s. 1344x768: 3.54 s / 4.14 s. WebP encode 105-160 ms. Teleport cache hit 0% (prompts are unique). The farm client sees `generate` = 11-16 s per image, i.e. 7-12 s of queueing behind other producers plus proxying through :8100. Moderation (`NSFW_DEVICE=cpu`) adds 4-8 s per image. | (1) Fuse the runtime LoRA path. It costs +0.56-0.61 s per image (15-18%); the math is ~2.5 ms per step, so the overhead is launches, casts and adds. (2) Port the pass-2 DiT/VAE fusions (bit-identical; not on prod yet). (3) Later: FP8 or int8 tensor-core GEMMs on sm_120 instead of Q4_K MMQ (~58% of each step). | (1) -0.4 to -0.5 s on LoRA images, ~5% of total 5090 time. (2) DiT step -6%, VAE -25% (3090 Ti min-est). (3) -15 to -25% of a step. | (1) M (2) S-M (3) H; (3) needs ~2.5 GB more VRAM and a quality gate |
| **ra2 Qwen-Image 2.1** | Q4_K_M DiT, Qwen3-VL-8B Q4_K_M text encoder plus mmproj, bf16 VAE. Lib `9bc7b35-cfgend-fusion`. `SD_PARAMS_BACKEND=*=cpu`, so weights stream from host every step. Easycache; 6-step turbo LoRA for background; `MIN_STEPS=30` for HQ; edit uses 16 steps with cfg until 0.5. | 1024², n=1. netwrck renders up to 4 images in series. | Week of 10-22, p50 exec: background **6.45 s**; free **43.6 s** (p90 57 s; probably edits since the 10-23 editcfg deploy); paid 33.5 s. Daisy 3090, same lane: free 1024² at 30 steps in **13.4-14.0 s**. Earlier weeks: free 5-10 s. | (1) Route HQ and edit to daisy when it is idle; the frontier/overflow plumbing already exists. (2) Keep part of the DiT resident inside the 10-11 GB lease instead of `*=cpu`. (3) netwrck: request all n images in parallel instead of in series. | (1) roughly 40 s down to 15-20 s on HQ/edit. (2) -30 to -50% step time. | (1) S (2) M |
| **LLM chat (local)** | `gemma-roleplay-v2-q8_0.gguf` (Gemma-4 E4B, Q8_0). KV cache q8_0 with flash attention, ctx 4096, full GPU (NGL 99). Batched engine with 32 contexts and 6 KV sequences. MTP head Q8_0 with draft 2. DRY sampling at 0.8; repetition-penalty window 0. llama.cpp libs `llama.cpp-omniserve-opt` (decode pass 1 only). Swap profiles: Boulesis 26B-A4B Q4_K_S, G4 31B IQ4_XS. | netwrck default lane: `max_tokens` 192, temperature 0.9, **not streamed**. Manim: `max_tokens` 6000, tier `sub`. | Engine trace: steady decode **11.19 ms** per step at 1 row. **Sampling 3.75 ms per step**: the CPU sampler chain takes 1.95 ms and the logits gather 0.70 ms per row over the 262K vocab. MTP: 2877 drafted, 828 accepted (29%); position 1 accepts ~52%, position 2 ~11%. | (1) Exact pruned sampling: penalties and DRY only *lower* logits, so the post-penalty top-k is contained in the pre-penalty top-(k+\|P\|) plus the penalized set P. A GPU top-K and gathering only those rows makes the result bit-identical. (2) Set `MTP_DRAFT=1` (README measured +12% at draft 1, nothing more at draft 2). (3) Port decode pass 2 plus the Q8 repack (+8-10% decode, bit-identical). | (1) step 14.9 -> ~11.7 ms, **+20-25% tokens/s** at batch 1. (2) +3-8%. (3) +8-10%. Low traffic, so this is mostly a latency win. | (1) M (2) S (3) S-M |
| **netwrck chat UX** | Default lane goes to the gateway or api.text-generator.io; routed and paid lanes go to OpenRouter, DeepSeek, Gemini, xAI and Anthropic. **All non-streamed** (`chat_enhanced.go:1196,1770`, 3 tries with backoff). | 192 to 5096 tokens. | TTFT equals full generation time: ~2.5-3 s for 192 tokens locally, tens of seconds on long routed replies. | Stream SSE end to end. The gateway and app.nz already stream; `live.go` already has an SSE writer. | perceived TTFT -> ~0.3-1 s | M |
| **TTS (server)** | Kokoro-82M, torch fp32, CUDA, in the `text-generator-tts-gpu` sidecar :9085 (`TTS_GPU_CONCURRENCY=2`). Supertonic ONNX (CUDA EP, 4 steps) is loaded but gets almost no server traffic. | All logged traffic uses Kokoro voices (af_nicole, af, am_michael, af_sarah). chars p50 120, p90 717, max 2820. Returns a whole WAV; not streamed. | took p50 **0.38 s**, p90 1.34 s, p99 3.43 s. RTF p50 0.032. Under 100 chars: 0.24 s. The compat layer adds ~10 ms. Every call also logs a failed Stripe credential lookup. | (1) Sentence-chunked streaming so the first chunk plays early. (2) fp16 or bf16 plus CUDA graphs or compile for the decoder. (3) Cache phonemizer output; drop the per-request credential lookup. | TTFA 0.38 -> ~0.12 s; p90 1.34 -> ~0.3 s to first audio | M |
| **TTS (browser, netwrck)** | Kokoro ONNX (WebGPU fp32, or wasm q8) first, then Supertonic via onnxruntime-web 1.23 (wasm, **`numThreads=1`**, 4 steps), then the server. | 180-char chunks. | Not measured. | Raise wasm `numThreads` when `crossOriginIsolated`; serve q8/fp16 ONNX on WebGPU. | 2-4x on wasm devices | S |
| **ASR** | Selected model is whisper-small.en (`docs/asr-model-selection.md`), but `omniserve-asr-worker` is **disabled**. The router goes straight to the :9083 fallback, which calls **Gemini** in the cloud. Parakeet-ctc-0.6b scored 100% WER through the transformers path. | multipart audio | No local runtime; no STT traffic in 3 days of logs. | If volume grows, enable a CTranslate2 whisper-small.en int8_float16 lazy worker (~0.5 GB). | removes a cloud dependency | S |
| **H3 video** | RunPod serverless ComfyUI cog `h3-cog:cu130-…-r7` on 5090s: int8_convrot DiT, NVFP4 text encoder, Sol-Attn, Spectrum. Endpoints have `workersMin` 0, **`idleTimeout` 5 s**, FlashBoot. | 896x672 (or 16:9), 5 s at 24 fps (~124 frames), 20 steps, audio on, AV1 output to R2. | Warm: execution 56-68 s (generation 48-59 s). **Cold: delay 28 s and execution 160 s for 61 s of real work**, so ~100 s is model load. Polling stacks up: RunPod 2 s, app.nz 2 s, manifoldgen 2 s, browser 2.5-3 s. The sd.cpp port is 2x slower than the cog (121.7 s vs ~60 s) and not deployed. | (1) Cold start: preload weights from a network volume or snapshot, overlap the load with ComfyUI boot (already done for swap/orbit in `manifoldgen-site-opt`, not for t2v), and use a session-aware `idleTimeout` (60-120 s). (2) Collapse the polling: webhook or long-poll instead of 2 s polls at 3 layers. | Cold requests 190 -> ~90 s; -2 to -4 s per request from polling | M |
| **YuE music** | YuE2-3B bf16 AR/NAR with an fp32 VAE, 32 ODE steps, torch. RunPod serverless only. | Clip 1500 semantic tokens; song 9000. | Clip: cold **250 s** caller wall time (178 s of it queue and init), warm **32 s** (planning 5.1, semantic 11.7, NAR 3.5, VAE 3.4). | Keep-warm or FlashBoot snapshot; fp8 or vLLM for the AR stage. | cold 250 -> ~60 s | M |
| **BiRefNet / RVM** | BiRefNet at 1024, torch.compile, autocast, channels_last. RVM resnet50 fp32 TorchScript. | 1024² mattes | 335 ms mean at 1 permit, 2.98 img/s; 4.95 img/s at 2 permits. Traffic is near zero. | none needed | | |
| **Embeddings** | gte-modernbert-base Q8_0, embedded in the gateway | single string | 36.5 ms mean; the traffic is mostly monitor pings | none needed | | |

## Ranked top-8 tasks (one agent each)

1. **Z-Image runtime-LoRA fusion.** Repo `sdcpp-opt` plus ggml, new branch `perf/zimage-lora-fusion` off `perf/dit-fusion`.
   - Work: batch the LoRA A-projections that share an input (q/k/v; w1/w3), fold `multiplier` into B at load, and add B·(xA) inside the base GEMM's output conversion or epilogue instead of separate mul_mat, scale, cast and add nodes. Keep the 0.6↔0.8 rescale cheap.
   - Metric: nsys kernel time per step with LoRA minus without LoRA, at 1024², Q4_K, the `the_look` LoRA (copy from prod `.lora_cache`). Target overhead ≤ 25 ms per step (now about 70 ms on the 5090).
   - Parity: same seed, LoRA vs the current at-runtime path. Expect md5-identical; accept PSNR ≥ 45 dB if summation order changes. Use the 4 prompts in `/vfast/data/imgopt` plus 2 portrait prompts that trigger `people-anime`, at scales 0.6 and 0.8. Add an off-switch env.

2. **Port Z-Image pass-2 fusions to the live prod recipe for sm_120.** Repo `sdcpp-zimage-fusion` (live tree, branch `perf/zimage-fusion-20261024`), new branch `perf/zimage-pass2-sm120`.
   - Work: cherry-pick sdcpp-opt `8406ede` plus ggml `perf/zimage-dit-fusion` (`18f1c867`), skipping the adaLN fold and kv_scale commits that were rejected on daisy. Build for sm_120.
   - Metric: nsys DiT step and VAE decode kernel time vs the live `sd-20261024-fusion` lib with the prod flags (cuDNN conv, `VAE_CONV_DIRECT=1`, tiles 80x80). Target ≥ 5% DiT and ≥ 20% VAE.
   - Parity: same `sd-cli` binary with `LD_LIBRARY_PATH` A/B, same seed, md5-identical for Q4_K at 1024² with 8 steps, with and without LoRA. Validate on the 3090 Ti; the 5090 step is for the deploy agent. Do not deploy.

3. **Exact pruned sampling plus MTP draft A/B in the omniserve LLM engine.** Repo `omniserve-native`, branch `perf/llm-sampling`.
   - Work: for each row, take top-(k + |P|) logits on the GPU (or after a single D2H of a GPU top-K), union them with the penalty/DRY history tokens P, and run the existing llama.cpp chain on that candidate set. Fall back to the full vocab when `top_k` ≤ 0, or when a logit-bias or grammar sampler is present.
   - Metric: engine trace `sample=` at ≤ 0.6 ms per step (now 3.75 ms) and `chain` at ≤ 0.2 ms per row. tokens/s at batch 1, 4 and 8 on gemma E4B Q8_0.
   - Parity: token-sequence md5 identical to the current build over ≥ 20 prompts × {greedy, temp 0.9 with a fixed seed} × {DRY 0.8, rep 1.1, both}. Also A/B `MTP_DRAFT=1` vs 2 (tokens/s and acceptance), and report which to set.

4. **netwrck chat streaming.** Repo `netwrck`, branch `perf/chat-sse`.
   - Work: add `stream:true` to the default and routed chat calls in `search_server_go/chat_enhanced.go` (lines 1196, 1770 and the default lane at 1575-1630). Relay SSE to the browser using the existing `live.go` writer pattern. Keep the current retry only for failures before the first byte.
   - Metric: TTFT measured at the browser for the default lane (target < 1 s) and for routed models; total time unchanged.
   - Parity: the final message text, moderation, credits and DB records are identical to the non-streamed path in the existing e2e tests (`model_generation_e2e_test.go`). Add a non-SSE fallback.

5. **H3 cold-start and polling.** Repos `manifoldgen-site` (branch `perf/h3-cold`) and the sibling `h3-cog` if needed.
   - Work: (a) pull RunPod job stats (delayTime, executionTime vs `metrics.total_seconds`) for the last 2 weeks and report the cold fraction. (b) Apply the weights-overlap/preload pattern already in `manifoldgen-site-opt` (3de1ef2, 6d5cff6) to the t2v handler, or put the 21 GB int8 weights on a network volume. (c) Propose an idleTimeout or active-session warm policy, with the $ cost from the cost guard. (d) Replace the 2 s polls with RunPod webhook or `/stream`, or with one long-poll layer.
   - Metric: cold execution minus `total_seconds` (now ~100 s; target < 30 s); end-to-end p50 for warm requests.
   - Parity: frames identical for the same seed, warm vs cold; no change to the generation config.

6. **ra2 HQ/edit latency.** Repo `omniserve-native`, branch `perf/ra2-residency`.
   - Work: (a) frontier policy: send `free`/`paid` HQ (≥ 16 steps) and edit requests to daisy `:8792` when its `/status` shows it idle, through the existing `omniserve-runpod-overflow` adapter. (b) Measure partial DiT residency (first N blocks on the GPU within the 10-11 GB lease) vs `*=cpu` on the 3090 Ti with the same lib.
   - Metric: per-step time and exec_ms p50 for 30-step 1024² and 16-step edit. Target ≤ 20 s (now 33-57 s on prod).
   - Parity: same lib and seed. Residency changes should be md5-identical on the same GPU; across 3090 and 5090, PSNR ≥ 40 dB.

7. **Kokoro TTS first-audio latency.** Repo `text-generator.io` (the prod checkout `/nvme0n1-disk/code/text-generator.io` is the live source; local `/vfast/data/code/text-generatorio` is stale at Oct 2, so sync it first), branch `perf/tts-stream`.
   - Work: add a chunked or streaming response to `tts_gpu_server.py` and `/v1/audio/speech`, splitting on sentences and emitting the first chunk early. Try bf16 or fp16 for the Kokoro decoder plus CUDA graphs per length bucket. Remove the per-request Stripe credential lookup that fails on every internal call.
   - Metric: time to first audio byte, p50 and p90, on the logged char-length distribution (p50 120, p90 717 chars). Target p50 ≤ 0.15 s.
   - Parity: concatenated streamed audio matches the non-streamed waveform (max abs diff ≤ 1 LSB in fp32 mode). In bf16 mode, mel-spectrogram L1 and a listening check.

8. **Port llama.cpp decode pass 2 plus Q8 repack to the live LLM libs.** Repo `llama.cpp-omniserve-opt` (`/sdb-disk/...`; local equivalent `llama.cpp-opt`), new branch `perf/decode-pass2-sm120` on `perf/decode-kernels-20261024`.
   - Work: cherry-pick `e49c629b7..5749d573b` and `7f327f636`, then build for sm_120.
   - Metric: llama-bench tg128 at d0 and d4096 for gemma E4B Q8_0 with q8_0 KV, flags unset and with `GGML_CUDA_FA_VEC512=1 GGML_CUDA_FA_VEC_SPLIT=1`. Target +8% at d0.
   - Parity: greedy md5 over 200 tokens identical to the live lib (flags unset; repack on and off), plus a `llama-perplexity -ub 1` KLD run (expect 0). The MTP head is Q8_0, so it gets repacked too: include a spec-decode md5 run.

Not in the top 8:
- FP8/int8 DiT GEMMs on sm_120 for Z-Image: the largest remaining kernel lever, but high effort plus a VRAM and quality gate.
- Moving the art-farm NSFW moderation (4-8 s CPU per image) to the GPU or the gateway's `/nsfw_detect`.
- Turning the background-tier VRAM-lease 503s into queueing.
- YuE keep-warm.
- Raising the browser Supertonic wasm thread count.
