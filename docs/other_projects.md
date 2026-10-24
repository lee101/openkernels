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
