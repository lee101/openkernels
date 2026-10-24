# openkernels — Open Kernel Challenge (Muse Spark Open, H200)

Kernel track closes 2026-11-04 23:59 PT. Scores are speedup over SGLang (geomean over shapes) from `okc check` (off-board) runs.

## Best so far (judge, `okc check`)

| problem | best file | speedup | leaderboard top | notes |
|---|---|---|---|---|
| muse_attention_decode | `submission_v31.py` | **1.498** | 2.37 | fdec (fused decode, per-warp cp.async rings, smem page-table rings) + attnw (wgmma verify/prefill) |
| muse_moe_w4a16 | `submission_v9.py` | **1.371** (v12 1.360) | 2.33 | persistent kernel, NVFP4 dequant 11 ops/8 values, mma.sync, TMA rings, dynamic tile queue |
| muse_decode_layer | `submission_v2.py` | **1.265** | 1.86 | composed from attention/moe/proj (`tools/build_dl.py`) + `glue.py` |
| muse_bf16_proj | `submission_v5.py` | 0.94 (v8 pending) | 1.45 | v8: cuBLAS for m 8-32 on qkv/o/pooler; v9 (working file): dedicated router kernel `rtr` |
| muse_prefill_layer | `submission_v1.py` | 0.80 | 1.70 | glue_pf.py; MoE at T~750 and 8-token-block attention dominate |

`submission.py` in each directory is the working copy (may be ahead of the best judged file).

## Per-shape weak spots (latest judge)
- attention v31: verify full b16/b32 1.19x (172/326 µs vs decode full 117/210 µs on the same bytes) -> attnw step loop was CTA-barrier bound; working copy decouples the two KV-head warpgroups.
- moe v12: bs1 48 µs vs SOL 11; bs256 455 µs (~1.2-1.7 TB/s). Ablations (H200): removing activation L2 traffic -20% at bs128/256, removing dequant -15%; both -35%. `moe_large` (CTA-shared activation tiles + wgmma RS) is correct but slower so far (WGL=False).
- bf16_proj: our skinny kernel loses to cuBLAS at m 8-32 except router; router `rtr` kernel 13.3/14.9 -> 10.0/11.0 µs (m 16/32, pod incl. ~4 µs overhead).

## Layout
- `muse_*/submission.py` working kernels (single-file, CUDA in Python strings); `submission_vN.py` judged snapshots; `profiles/` judge profiles (gitignored).
- `okb.py` local judge-like harness (`--no-time`, `--stress N`, `--kprof`); `rp.py` RunPod H200 pods with TTL/budget guards; `bench/pod_s*.sh` batched pod sessions.
- `tools/cu.py extract|embed` CUDA <-> `cu/*.cu` for static analysis; `tools/build_dl.py` builds decode/prefill layers; `tools/showchk.py` summarizes check JSON.
- `docs/ideas.md` findings and ideas log; `docs/other_projects.md` omniserve-native / manifoldgen-site agent passes; `notes/h200_session1.md` H200 session notes.

## Workflow
Local RTX 3090 Ti = correctness only (shared GPU). H200 timing via short RunPod pods (no perf counters there either). Judge: `okc check <problem> file.py --json` (10-20/day). Never `okc submit` until confident of #1.

## Security and secrets

`RUNPOD_API_KEY` and `RUNPOD_ACCOUNT_ID` live in a gitignored `.env` at the repo root; real environment variables take precedence and no key is tracked. `notes/rp_watchdog.log` (pod ids) and `notes/rp_budget.json` (spend state) are local-only and gitignored.

Secret scanning uses [gitleaks](https://github.com/gitleaks/gitleaks) with the default ruleset (`.gitleaks.toml`):

```sh
gitleaks git --redact --verbose    # full history
pre-commit install                 # scan staged changes on every commit
```

CI runs the same full-history scan on every push and pull request (`.github/workflows/gitleaks.yml`).

## License

Apache-2.0 (see `LICENSE`). Copyright 2026 Lee Penkman.
