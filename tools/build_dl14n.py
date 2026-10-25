"""build muse_decode_layer/submission_v14.py: build_dl19.py + programmatic dependent launch (PDL) on every decode-layer kernel.
Kernels get griddepcontrol.wait after their input-independent prologue (then launch_dependents); launches carry
cudaLaunchAttributeProgrammaticStreamSerialization (sm_90+). Component sources are patched here, not edited."""
import re, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); import compose
R = Path(__file__).resolve().parent.parent
out_dl = R / (sys.argv[1] if len(sys.argv) > 1 else "muse_decode_layer/submission_v14.py")
TMP = Path(tempfile.mkdtemp())

MAC = r"""
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define PDL_WAIT() asm volatile("griddepcontrol.wait;\n" ::: "memory")
#define PDL_GO() asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory")
#else
#define PDL_WAIT()
#define PDL_GO()
#endif
#ifndef PDL
#define PDL 0
#endif
static inline int pdl_on() { static int v = -1; if (v < 0) { int d, mj; cudaGetDevice(&d); cudaDeviceGetAttribute(&mj, cudaDevAttrComputeCapabilityMajor, d); v = PDL && mj >= 9; } return v; }
template <typename... K, typename... T>
static inline void pdlk(void (*k)(K...), dim3 g, dim3 b, size_t sm, cudaStream_t st, T... a) {
  cudaLaunchConfig_t c = {}; c.gridDim = g; c.blockDim = b; c.dynamicSmemBytes = sm; c.stream = st;
  cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[0].val.programmaticStreamSerializationAllowed = 1;
  c.attrs = at; c.numAttrs = pdl_on();
  cudaLaunchKernelEx(&c, k, a...);
}
"""
TOP = " PDL_WAIT(); PDL_GO();"


def sub(s, a, b, n=1):
    assert s.count(a) == n, (a[:80], s.count(a))
    return s.replace(a, b)


def launches(s, skip=()):
    """kernel<<<g, b, sm, st>>>(args) -> pdlk(kernel, g, b, sm, st, args)"""
    def f(m):
        if m.group(1).split("<")[0] in skip:
            return m.group(0)
        assert m.group(2).count(",") >= 3 or "(" in m.group(2), m.group(0)
        return "pdlk(%s, %s, " % (m.group(1), m.group(2))
    return re.sub(r"\b([A-Za-z_]\w*(?:<[^<>;()]*>)?)<<<([^;]*?)>>>\(", f, s)


def tops(s, skip=()):
    """griddepcontrol.wait + launch_dependents at the top of every kernel body not in skip"""
    out, i = [], 0
    for m in re.finditer(r"__global__\s+void\s+(?:__launch_bounds__\([^)]*\)\s*)?(\w+)\s*\(", s):
        j = m.end()
        nb, ns = s.find("{", j), s.find(";", j)
        if m.group(1) in skip or nb < 0 or (0 <= ns < nb):
            continue
        out.append(s[i:nb + 1] + TOP); i = nb + 1
    return "".join(out) + s[i:]


def section(s, start, end):
    a = s.index(start); b = s.index(end, a)
    return s[:a], s[a:b], s[b:]


def p_attn(s):
    s = sub(s, "#include <ATen/cuda/CUDAContext.h>\n", "#include <ATen/cuda/CUDAContext.h>\n" + MAC)
    s = sub(s, '"attn_v45_%d"', '"attn_v45p_%d"')
    s = sub(s, """  uint4 qraw[2][2];
#pragma unroll
  for (int h = 0; h < 2; h++) { const uint4* qp = (const uint4*)(row0 + (h * 8 + g) * D + t * 16); qraw[h][0] = qp[0]; qraw[h][1] = qp[1]; }
""", "  uint4 qraw[2][2];\n")
    s = sub(s, """  uint2 kraw = make_uint2(0, 0), vraw = make_uint2(0, 0);
  if (w == 0) {
    const int h = lane >> 4, d0 = (lane & 15) * 4;
    kraw = *(const uint2*)(row0 + 16 * D + h * D + d0); vraw = *(const uint2*)(row0 + 18 * D + h * D + d0);
  }
  const uint2 gpre = (p.fl & 1) ? *(const uint2*)(row0 + 20 * D + tid * 4) : make_uint2(0, 0);  // gate rows for the output
""", "  uint2 kraw = make_uint2(0, 0), vraw = make_uint2(0, 0), gpre = make_uint2(0, 0);\n")
    s = sub(s, "  for (int i = 0; i < RBF - 1; i++) issue(i, rpro[i], false);  // KV loads start before the query prologue\n",
            """  for (int i = 0; i < RBF - 1; i++) issue(i, rpro[i], false);  // KV loads start before the query prologue
  PDL_WAIT(); PDL_GO();  // seq, page table and the first KV groups above read only step inputs; qkv comes from the projection
#pragma unroll
  for (int h = 0; h < 2; h++) { const uint4* qp = (const uint4*)(row0 + (h * 8 + g) * D + t * 16); qraw[h][0] = qp[0]; qraw[h][1] = qp[1]; }
  if (w == 0) {
    const int h = lane >> 4, d0 = (lane & 15) * 4;
    kraw = *(const uint2*)(row0 + 16 * D + h * D + d0); vraw = *(const uint2*)(row0 + 18 * D + h * D + d0);
  }
  if (p.fl & 1) gpre = *(const uint2*)(row0 + 20 * D + tid * 4);
""")
    s = sub(s, """    cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = NS; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
    cfg.attrs = at; cfg.numAttrs = 1;""", """    cudaLaunchAttribute at[2]; at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = NS; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
    at[1].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[1].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at; cfg.numAttrs = 1 + pdl_on();""")
    s = tops(s, skip=("fdec",))
    return launches(s)


def p_moe(s):
    s = sub(s, "#include <ATen/cuda/CUDAContext.h>\n", "#include <ATen/cuda/CUDAContext.h>\n" + MAC)
    s = sub(s, '"moe_w4a16_v17_%d_%d%s%s"', '"moe_w4a16_v17p_%d_%d%s%s"')
    s = sub(s, "  if (tid < NLOC) sm->cnt[tid] = 0;\n  __syncthreads();\n", "  if (tid < NLOC) sm->cnt[tid] = 0;\n  __syncthreads();\n  PDL_WAIT(); PDL_GO();\n")
    s = tops(s, skip=("moe_kernel",))
    return launches(s)


def p_proj(s):
    for a, b in (("static void* g_ltws = nullptr;", "static void* g_ltws = nullptr; static void* g_ltws2 = nullptr; static int g_side = 0;"),
                 ("cudaMalloc(&g_ltws, LTWS); }", "cudaMalloc(&g_ltws, LTWS); cudaMalloc(&g_ltws2, LTWS); }"),
                 ("&al.algo, g_ltws, LTWS, at::cuda::getCurrentCUDAStream());\n}", "&al.algo, g_side ? g_ltws2 : g_ltws, LTWS, at::cuda::getCurrentCUDAStream());\n}\nvoid lt_side(int64_t s) { g_side = (int)s; }"),
                 ('"lt_cfg"]', '"lt_cfg", "lt_side"]'), ('"skinny_v19_%d"', '"skinny_v19dlp_%d"'), ('"gv19_%x"', '"gv19p_%x"'),
                 ("\nstruct Q {", MAC + "\nstruct Q {"), ("\nstruct G {", MAC + "\nstruct G {"),
                 ("  for (int j = 0; j < 4; j++) { a[j] = ldnc(ah + j * 512); h[j] = ldnc(al + j * 512); }\n",
                  "  for (int j = 0; j < 4; j++) { a[j] = ldnc(ah + j * 512); h[j] = ldnc(al + j * 512); }\n  PDL_WAIT(); PDL_GO();\n"),
                 ("rtr<1><<<grid, 512, 0, st>>>(q); else if (nt <= 2) rtr<2><<<grid, 512, 0, st>>>(q); else rtr<4><<<grid, 512, 0, st>>>(q);",
                  "pdlk(rtr<1>, grid, 512, 0, st, q); else if (nt <= 2) pdlk(rtr<2>, grid, 512, 0, st, q); else pdlk(rtr<4>, grid, 512, 0, st, q);"),
                 ("__uint_as_float(u.z), __uint_as_float(u.w)); }\n  float acc[R][M];", "__uint_as_float(u.z), __uint_as_float(u.w)); }\n  PDL_WAIT(); PDL_GO();\n  float acc[R][M];"),
                 ("""  auto ld = [&](int s, int u) {""", """  auto ldw = [&](int s, int u) {
#pragma unroll
    for (int j = 0; j < RT; j++)
#pragma unroll
      for (int h = 0; h < NA; h++)
#pragma unroll
        for (int i = 0; i < NL; i++) wb[s][j][h][i] = gnc((h ? p.W2 : p.W) + wo + j * js + (u * NL + i) * 32);
  };
  auto ldx = [&](int s, int u) {
#pragma unroll
    for (int n = 0; n < MT; n++) { xb[s % XB][n][0] = gx(xp[n] + u * 64); xb[s % XB][n][1] = gx(xp[n] + u * 64 + 8); }
  };
  auto ld = [&](int s, int u) {"""),
                 ("  for (int s = 0; s < ST; s++) if (s < KU) ld(s, s);\n", """  for (int s = 0; s < ST; s++) if (s < KU) ldw(s, s);
  PDL_WAIT(); PDL_GO();  // weights above are input-independent; x comes from the previous kernel
#pragma unroll
  for (int s = 0; s < ST; s++) if (s < KU && !XS) ldx(s, s);
"""),
                 ("""  cudaLaunchAttribute at[1];
  if (S > 1 && cc >= 9) { at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = 1; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = S; cfg.attrs = at; cfg.numAttrs = 1; }""",
                  """  cudaLaunchAttribute at[2]; cfg.attrs = at; cfg.numAttrs = 0;
  if (S > 1 && cc >= 9) { at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = 1; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = S; cfg.numAttrs = 1; }
  if (pdl_on()) { at[cfg.numAttrs].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[cfg.numAttrs].val.programmaticStreamSerializationAllowed = 1; cfg.numAttrs++; }""")):
        s = sub(s, a, b)
    for k in ("rr<1, 1>", "rr<1, 2>", "rr<8, 2>", "rr<8, 4>"):
        s = re.sub(re.escape(k) + r"<<<([^;]*?)>>>\(", lambda m: "pdlk(%s, %s, " % (k, m.group(1)), s)
    assert "rr<1, 1><<<" not in s
    return s


def p_glue(s):
    s = sub(s, "#include <ATen/cuda/CUDAContext.h>\n", "#include <ATen/cuda/CUDAContext.h>\n" + MAC)
    s = sub(s, '"dl_glue_v4"', '"dl_glue_v4p"')
    s = sub(s, """  if (in) araw = __ldg((const uint4*)(a + (size_t)r * na + c0));
  ld8(s + (size_t)r * H + c0, sv); ld8(al + c0, av); ld8(be + c0, bv); ld8(gn + c0, gv);
  if (ga && in) ld8(ga + c0, gav);
""", """  ld8(al + c0, av); ld8(be + c0, bv); ld8(gn + c0, gv);
  if (ga && in) ld8(ga + c0, gav);
  PDL_WAIT(); PDL_GO();
  if (in) araw = __ldg((const uint4*)(a + (size_t)r * na + c0));
  ld8(s + (size_t)r * H + c0, sv);
""")
    s = tops(s, skip=("resid",))
    s = launches(s)
    rc = 'P__ext().run(st["rA"], st["rA2"], h, lg, st["pws"], st["pcnt"], 256)'
    return sub(s, rc, "_router(h, lg, st)")


def tmpf(name, text):
    p = TMP / name; p.write_text(text); return p


head = '''# okc: problem=muse_decode_layer
"""Decoder layer: c12 register-direct GEMV / cuBLAS projections + fused split-KV FP8 attention + fused residual norms + NVFP4 W4A16 MoE; PDL across every kernel."""
from typing import TypedDict
import torch
from torch.utils.cpp_extension import load_inline

'''
plain = (R / "muse_decode_layer/submission_plain.py").read_text().splitlines(True)
cls = "".join(plain[15:75])
parts = [head, compose.rename(tmpf("a.py", p_attn((R / "muse_attention_decode/submission.py").read_text())), "A_"), "\n",
         compose.rename(tmpf("m.py", p_moe((R / "muse_moe_w4a16/submission.py").read_text())), "M_"), "\n",
         compose.rename(tmpf("p.py", p_proj((R / "muse_bf16_proj/submission_v19.py").read_text())), "P_"), "\n", cls, "\n",
         p_glue((R / "muse_decode_layer/glue.py").read_text()), "\n", (R / "muse_prefill_layer/glue_pf.py").read_text(), "\n",
         (R / "muse_decode_layer/glue19.py").read_text()]
out = "".join(parts)
compile(out, out_dl.name, "exec")
out_dl.write_text(out)
print("wrote", out_dl, len(out), "pdlk", out.count("pdlk("), "waits", out.count("PDL_WAIT();"))
