"""build muse_decode_layer/submission_v32.py: build_dl31 with attention v74 (s1k: 16-CTA cluster batch-1 sliding kernel, launched
with the cluster attribute only, so no PDL attribute; in BIG so it never triggers dependents early).
build_dl31 -> submission_v31.py: build_dl28 (default attention v69f / MoE v52, wx pruned to v52's WXG2 picks) + glue fused into the GEMV
epilogues behind DL_FUSE: o-proj -> resid1, pooler -> rmsg, unpooler -> resid2. At every exit of the gv kernel the
last-arriving CTA (per token block, self-cleaning counter) runs the row epilogue; a single CTA per token block only pays at T = 1,
so DL_FUSE_T = (1,).
build_dl28: build_dl27 with attention v65 / MoE v48 by default, pruned to v48's picks (gk only, no g1).
build_dl27: build_dl26 with MoE v47 by default, pruned to v47's judge picks (+ fs f1s3 for the
M32 pin: v47 retired fs from its candidates, but CUDA_F / _fs / its state are unchanged from v46).
build_dl26: build_dl25 with MoE v46 / attention v60 by default; MoE gk (CUDA_K, M <= 16, 3 grid
barriers) gets the g1 / fs PDL treatment (wait at the top, never trigger); the MoE tune deadlines count from the decode layer's
prepare start (DL_T0 / DL_TSTART / DL_TEND) so the whole prepare stays inside 300 s.
build_dl25: build_dl24 with MoE v45 by default; its g1 path (CUDA_G, M <= 2, one CTA per SM,
grid barriers) gets the fs treatment: PDL wait at the top, launched with the attribute, never triggers.
build_dl24: build_dl23 + MoE M32 pinned to the fused split-K fs f1s3 (v22's in-layer pick; v23's
r1 lost 16-37 us at decode32 on the judge although the synthetic in-layer tune had them tied), kept unless the in-layer tune
measures another candidate faster by more than DL_PIN_MARGIN; the in-layer tune times are kept in state and the picks printed
(DLMOE line in the prepare log).
build_dl23: build_dl22 parameterized on the attention and MoE sources (default attention v56,
MoE v43). MoE wx (CUDA_X: xgather + moe_x, one CTA per SM) is left without PDL (plain launches, fully serialized; it only runs
at M >= 16 where the policy has PDL off anyway) and is pre-compiled with the other extensions.
build_dl22: build_dl21 with the attention source as a parameter.
build_dl21: build_dl20 with attention v50 (attnp: no early trigger, like attnw) and MoE v34
(fs fused split-K: PDL wait only, no trigger, grid barrier safe). Base: build_dl15.py b (PDL for T <= 8, late triggers) with bf16_proj v22
(wgb/wgr wgmma routers, gv push split-K + uncapped regs). The router runs on the side stream next to the pooler, so it gets
its own gv/wgr workspace and counters (p16r). wgr spins on per-tile counters (whole grid co-resident): it triggers its
dependents only after its counter wait, so a dependent grid can never take SMs from a not-yet-resident wgr CTA.
usage: build_dl32.py [attention submission] [moe submission] [out]"""
import re, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); import compose
R = Path(__file__).resolve().parent.parent
VAR = "b"
POL = {"a": "{k: T <= 8 for k in _PDLX}", "b": "{k: T <= 8 for k in _PDLX}", "c": "{k: k != 'moe' for k in _PDLX}"}[VAR]
LATE = int(VAR == "b")
ATTN = R / (sys.argv[1] if len(sys.argv) > 1 else "muse_attention_decode/submission_v74.py")
MOE = R / (sys.argv[2] if len(sys.argv) > 2 else "muse_moe_w4a16/submission_v52.py")
out_dl = R / (sys.argv[3] if len(sys.argv) > 3 else "muse_decode_layer/submission_v32.py")
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
#define PDL 1
#endif
#define PDL_LATE @@LATE@@
#define PDL_GO_EARLY() do { if (!PDL_LATE) PDL_GO(); } while (0)
#define PDL_GO_LATE() do { if (PDL_LATE) PDL_GO(); } while (0)
static int g_pdl = 1;
void pdl_set(int64_t v) { g_pdl = (int)v; }
static inline int pdl_on() { static int a = -1; if (a < 0) { int d, mj; cudaGetDevice(&d); cudaDeviceGetAttribute(&mj, cudaDevAttrComputeCapabilityMajor, d); a = mj >= 9; } return PDL && a && g_pdl; }
template <typename... K, typename... T>
static inline void pdlk(void (*k)(K...), dim3 g, dim3 b, size_t sm, cudaStream_t st, T... a) {
  cudaLaunchConfig_t c = {}; c.gridDim = g; c.blockDim = b; c.dynamicSmemBytes = sm; c.stream = st;
  cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[0].val.programmaticStreamSerializationAllowed = 1;
  c.attrs = at; c.numAttrs = pdl_on();
  cudaLaunchKernelEx(&c, k, a...);
}
""".replace("@@LATE@@", str(LATE))
TOP = " PDL_WAIT(); PDL_GO();"
TOPL = " PDL_WAIT(); PDL_GO_EARLY();"
BIG = ("attnw", "attnw39", "attnx", "attnp", "attnd", "s1k")  # cross-CTA merge spins: never trigger dependents early (PDL_LATE)


def sub(s, a, b, n=1):
    assert s.count(a) == n, (a[:80], s.count(a), n)
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
    for m in re.finditer(r"__global__\s+void\s+(?:__launch_bounds__\((?:[^()]|\([^()]*\))*\)\s*)?(\w+)\s*\(", s):  # nested parens: attnp
        j = m.end()
        nb, ns = s.find("{", j), s.find(";", j)
        if m.group(1) in skip or nb < 0 or (0 <= ns < nb):
            continue
        out.append(s[i:nb + 1] + (TOPL if m.group(1) in BIG else TOP)); i = nb + 1
    return "".join(out) + s[i:]


def section(s, start, end):
    a = s.index(start); b = s.index(end, a)
    return s[:a], s[a:b], s[b:]


def p_attn(s):
    s = sub(s, "#include <ATen/cuda/CUDAContext.h>\n", "#include <ATen/cuda/CUDAContext.h>\n" + MAC)
    # v52+: attnd (decode on wgmma) only exists on sm_90 or under WGEMU; elsewhere the host fell through to a path with
    # attnd-sized buffers (illegal address on the 3090). No change on Hopper.
    da = '    dac = (DAOVR if DAOVR is not None else DACFG.get((B, bool(cfg["window"])))) if L == 1 else None\n'
    if da in s:
        s = sub(s, da, da + '    if dac and torch.cuda.get_device_capability()[0] != 9 and "-DWGEMU=1" not in _DBG:\n        dac = None\n')
    if "template <int DEPTH, bool PDL>" in s:  # v65 latk (diag) template parameter named like the PDL macro
        s = sub(s, "template <int DEPTH, bool PDL>", "template <int DEPTH, bool PDLD>")
        s = sub(s, '  if (PDL) asm volatile("griddepcontrol.wait;" ::: "memory");', '  if (PDLD) asm volatile("griddepcontrol.wait;" ::: "memory");')
    nm = re.findall(r'load_inline\("(attn_v\w+?)_%d"', s)
    assert len(nm) == 1, nm
    s = sub(s, '"%s_%%d"' % nm[0], '"%sp%d_%%d"' % (nm[0], LATE))
    s = sub(s, 'cpp_sources="', 'cpp_sources="void pdl_set(int64_t); ')  # the attention module's one load_inline
    fl = re.findall(r'functions=\["run", [^\]]*\]', s)
    assert len(fl) == 1, fl
    s = sub(s, fl[0], fl[0][:-1] + ', "pdl_set"]')
    s = sub(s, '  PT(3);\n  asm volatile("cp.async.wait_group 0;\\n");\n', '  PDL_GO_LATE();\n  PT(3);\n  asm volatile("cp.async.wait_group 0;\\n");\n')
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
  PDL_WAIT(); PDL_GO_EARLY();  // seq, page table and the first KV groups above read only step inputs; qkv comes from the projection
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
    """MoE v30: PDL on the v17-style kernel (CUDA: _ext / _extS builds); the CTA-ring kernel (CUDA_L) is left as is"""
    i = s.index('CUDA_L = r"""')
    s, rest = s[:i], s[i:]
    s = sub(s, "#include <ATen/cuda/CUDAContext.h>\n", "#include <ATen/cuda/CUDAContext.h>\n" + MAC)
    rest = sub(rest, 'load_inline("moe_w4a16_v30_%d_%d%s%s" % (wpc, stg, "_L" if large else "", "".join(_DBG).replace("-", "_").replace("=", "")),\n                        cpp_sources="void moe(',
               'load_inline("moe_w4a16_v30p%d_%%d_%%d%%s%%s" %% (wpc, stg, "_L" if large else "", "".join(_DBG).replace("-", "_").replace("=", "")),\n                        cpp_sources="void pdl_set(int64_t); void moe(' % LATE)
    rest = sub(rest, '"moe_w4a16_v30S_%d_%d_k%d" % (wpc, stg, ksf),\n                                 cpp_sources="void moe(',
               '"moe_w4a16_v30Sp%d_%%d_%%d_k%%d" %% (wpc, stg, ksf),\n                                 cpp_sources="void pdl_set(int64_t); void moe(' % LATE)
    rest = sub(rest, 'cuda_sources=CUDA, functions=["moe"]', 'cuda_sources=CUDA, functions=["moe", "pdl_set"]', 2)
    s = sub(s, "  if (tid < NLOC) sm->cnt[tid] = 0;\n  __syncthreads();\n", "  if (tid < NLOC) sm->cnt[tid] = 0;\n  __syncthreads();\n  PDL_WAIT(); PDL_GO_EARLY();\n")
    s = sub(s, "  const int T2 = sm->na * T2E;\n", "  PDL_GO_LATE();\n  const int T2 = sm->na * T2E;\n")
    s = tops(s, skip=("moe_kernel",))
    # v34 fs (CUDA_F): one CTA per SM with a grid barrier. PDL wait only, never a trigger: it may start on SMs freed by its
    # predecessors (which finish on their own), but no dependent can launch, and take SMs, before every fs CTA has exited.
    rest = sub(rest, 'CUDA_F = r"""\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n',
               'CUDA_F = r"""\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n' + MAC)
    rest = sub(rest, "__global__ void __launch_bounds__(WPC * 32, 1) moe_f(P p) {\n", "__global__ void __launch_bounds__(WPC * 32, 1) moe_f(P p) {\n  PDL_WAIT();\n")
    rest = sub(rest, "  moe_f<<<g_gridF, WPC * 32, g_smemF, at::cuda::getCurrentCUDAStream()>>>(p);\n",
               "  pdlk(moe_f, dim3(g_gridF), dim3(WPC * 32), (size_t)g_smemF, at::cuda::getCurrentCUDAStream(), p);\n")
    fn = re.findall(r'load_inline\("(moe_w4a16_v\d+F)" \+ "".join\(fl\)', rest)
    assert len(fn) == 1, fn
    rest = sub(rest, 'load_inline("%s" + "".join(fl).replace("-", "_").replace("=", ""),\n                               cpp_sources="void moe(' % fn[0],
               'load_inline("%sp" + "".join(fl).replace("-", "_").replace("=", ""),\n                               cpp_sources="void pdl_set(int64_t); void moe(' % fn[0])
    rest = sub(rest, 'cuda_sources=CUDA_F, functions=["moe"]', 'cuda_sources=CUDA_F, functions=["moe", "pdl_set"]')
    for tag, kern in (("G", "moe_g"), ("K", "moe_k")):  # v45 g1 / v46 gk: grid barriers, so as fs: PDL wait only, never trigger
        if 'CUDA_%s = r"""' % tag not in rest:
            continue
        rest = sub(rest, 'CUDA_%s = r"""#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n' % tag,
                   'CUDA_%s = r"""#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n' % tag + MAC)
        rest = sub(rest, "__global__ void __launch_bounds__(NW * 32, 1) %s(P p) {\n" % kern, "__global__ void __launch_bounds__(NW * 32, 1) %s(P p) {\n  PDL_WAIT();\n" % kern)
        rest = sub(rest, "  %s<<<g_grid%s, NW * 32, g_smem%s, at::cuda::getCurrentCUDAStream()>>>(p);\n" % (kern, tag, tag),
                   "  pdlk(%s, dim3(g_grid%s), dim3(NW * 32), (size_t)g_smem%s, at::cuda::getCurrentCUDAStream(), p);\n" % (kern, tag, tag))
        gn = re.findall(r'load_inline\("(moe_w4a16_v\d+%s)" \+ "".join\(fl\)' % tag, rest)
        assert len(gn) == 1, gn
        rest = sub(rest, 'load_inline("%s" + "".join(fl).replace("-", "_").replace("=", ""),\n                                   cpp_sources="void moe(' % gn[0],
                   'load_inline("%sp" + "".join(fl).replace("-", "_").replace("=", ""),\n                                   cpp_sources="void pdl_set(int64_t); void moe(' % gn[0])
        rest = sub(rest, 'cuda_sources=CUDA_%s, functions=["moe"]' % tag, 'cuda_sources=CUDA_%s, functions=["moe", "pdl_set"]' % tag)
    # prepare budget: tune deadlines measured from the decode layer's prepare start (set by its prepare), not the MoE's own
    rest = sub(rest, "def prepare(weights):\n    global _T0\n    _T0 = time.time()\n",
               "DL_T0, DL_TSTART, DL_TEND = None, 200, 270  # decode layer: its prepare start, last tune start, last bucket start (s)\n\n\n"
               "def prepare(weights):\n    global _T0\n    _T0 = time.time() if DL_T0 is None else DL_T0\n")
    if "and el < 200:\n        st[\"tune\"]" in rest:
        rest = sub(rest, "and el < 200:\n        st[\"tune\"]", "and el < DL_TSTART:\n        st[\"tune\"]")
    if "        if time.time() - _T0 > 270:" in rest:
        rest = sub(rest, "        if time.time() - _T0 > 270:", "        if time.time() - _T0 > DL_TEND:")
    rest = sub(rest, "        table[M] = best\n", "        table[M] = best\n        state.setdefault(\"tune_us\", {})[M] = {c: res[c] / 2 for c in res}\n")
    return launches(s) + rest


def p_proj(s):
    for a, b in (("static void* g_ltws = nullptr;", "static void* g_ltws = nullptr; static void* g_ltws2 = nullptr; static int g_side = 0;"),
                 ("cudaMalloc(&g_ltws, LTWS); }", "cudaMalloc(&g_ltws, LTWS); cudaMalloc(&g_ltws2, LTWS); }"),
                 ("&al.algo, g_ltws, LTWS, at::cuda::getCurrentCUDAStream());\n}", "&al.algo, g_side ? g_ltws2 : g_ltws, LTWS, at::cuda::getCurrentCUDAStream());\n}\nvoid lt_side(int64_t s) { g_side = (int)s; }"),
                 ('"lt_cfg"]', '"lt_cfg", "lt_side", "pdl_set"]'), ('"skinny_v21_%d"', '"skinny_v21dlp%d_%%d"' % LATE), ('"gv21_%x"', '"gv21p%d_%%x"' % LATE),
                 ('cpp_sources=LTCPP + "void run(', 'cpp_sources=LTCPP + "void pdl_set(int64_t); void run('),
                 ('cpp_sources="int64_t gvrun(', 'cpp_sources="void pdl_set(int64_t); int64_t gvrun('), ('functions=["gvrun", "rrrun"]', 'functions=["gvrun", "rrrun", "pdl_set"]'),
                 ("  float4 f[RT][MT];\n", "  PDL_GO_LATE();\n  float4 f[RT][MT];\n"),
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
  PDL_WAIT(); PDL_GO_EARLY();  // weights above are input-independent; x comes from the previous kernel
#pragma unroll
  for (int s = 0; s < ST; s++) if (s < KU && !XS) ldx(s, s);
"""),
                 ("""  cudaLaunchAttribute at[1];
  if (S > 1 && cc >= 9) { at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = 1; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = S; cfg.attrs = at; cfg.numAttrs = 1; }""",
                  """  cudaLaunchAttribute at[2]; cfg.attrs = at; cfg.numAttrs = 0;
  if (S > 1 && cc >= 9) { at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = 1; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = S; cfg.numAttrs = 1; }
  if (pdl_on()) { at[cfg.numAttrs].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[cfg.numAttrs].val.programmaticStreamSerializationAllowed = 1; cfg.numAttrs++; }""")):
        s = sub(s, a, b)
    s = p_wb(s)
    for k in ("rr<1, 1>", "rr<1, 2>", "rr<8, 2>", "rr<8, 4>"):
        s = re.sub(re.escape(k) + r"<<<([^;]*?)>>>\(", lambda m: "pdlk(%s, %s, " % (k, m.group(1)), s)
    assert "rr<1, 1><<<" not in s
    return s


WLD = """      for (int c = 0; c < nk; c++) {
        const int s = c % QS;
        if (c >= QS) bwait(empty + s * 8, ((c / QS) - 1) & 1);
        const uint32_t bar = full + s * 8, dst = sb + s * SBA;
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\\n" :: "r"(bar), "n"(SB) : "memory");
"""


def p_wb(s):
    """v22 wgb / wgr: first ring stages of W issued before griddepcontrol.wait (x after it); wgb triggers after its main loop,
    wgr only after its counter wait (co-residency: see the module docstring)"""
    s = sub(s, "\nstruct BP {", MAC + "\nstruct BP {")
    s = sub(s, '"wb22_%x"', '"wb22p%d_%%x"' % LATE)
    s = sub(s, 'cpp_sources="int64_t wbrun(', 'cpp_sources="void pdl_set(int64_t); int64_t wbrun(')
    s = sub(s, 'functions=["wbrun"]', 'functions=["wbrun", "pdl_set"]')
    for wl, xl in (("""        btma(dst + XT, &wm, (c0 + c) * 64, rb * BN, bar, pw);
        if (HL) btma(dst + XT + WT, &wm, (c0 + c) * 64, p.NH + rb * BN, bar, pw);
""", "        btma(dst, &xm, (c0 + c) * 64, tb * TM, bar, px);\n"), ("""        btma(dst + XT, &wm, (c0 + c) * 64, rb * BN, bar, pw);
        btma(dst + XT + WT, &wm, (c0 + c) * 64, p.NH + rb * BN, bar, pw);
""", "        btma(dst, &xm, (c0 + c) * 64, tb * TM, bar, px);\n")):
        old = WLD + (xl + wl if "HL" in wl else wl + xl) + "      }\n"
        new = ("      auto ldw = [&](int c) {\n        const int s = c % QS;\n        const uint32_t bar = full + s * 8, dst = sb + s * SBA;\n"
               + WLD.splitlines(True)[4] + wl + "      };\n"
               "      for (int c = 0; c < nk && c < QS; c++) ldw(c);  // weights do not depend on the previous kernel\n"
               "      PDL_WAIT();\n"
               "      for (int c = 0; c < nk; c++) {\n        const int s = c % QS;\n"
               "        if (c >= QS) { bwait(empty + s * 8, ((c / QS) - 1) & 1); ldw(c); }\n"
               "        const uint32_t bar = full + s * 8, dst = sb + s * SBA;\n" + xl + "      }\n")
        s = sub(s, old, new)
    s = sub(s, """    }
  } else {
    for (int c = 0; c < nk; c++) {
      const int s = c % QS;
      bwait(full + s * 8, (c / QS) & 1);""", """    } else PDL_WAIT();
  } else {
    PDL_WAIT();
    for (int c = 0; c < nk; c++) {
      const int s = c % QS;
      bwait(full + s * 8, (c / QS) & 1);""", 2)
    s = sub(s, "  __syncthreads();\n  const int tk0 = wg * 64 + (w & 3) * 16 + g;\n  O* out = (O*)p.out;\n",
            "  __syncthreads();\n  PDL_GO();\n  const int tk0 = wg * 64 + (w & 3) * 16 + g;\n  O* out = (O*)p.out;\n")
    s = sub(s, """    do { asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\\n" : "=r"(v) : "l"(p.cnt + ti) : "memory"); } while (v - tgt < 0);
  }
  __syncthreads();
""", """    do { asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\\n" : "=r"(v) : "l"(p.cnt + ti) : "memory"); } while (v - tgt < 0);
  }
  __syncthreads();
  PDL_GO();  // only after the counter wait: dependents launch once every CTA got here, i.e. the whole grid is resident
""")
    s = sub(s, """  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = 1; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = S;
  cfg.attrs = at; cfg.numAttrs = S > 1 ? 1 : 0;
""", """  cudaLaunchAttribute at[2];
  at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = 1; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = S;
  cfg.attrs = at; cfg.numAttrs = S > 1 ? 1 : 0;
  if (pdl_on()) { at[cfg.numAttrs].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[cfg.numAttrs].val.programmaticStreamSerializationAllowed = 1; cfg.numAttrs++; }
""")
    return sub(s, "  wgr<BN, QS, S, NWG><<<grid, NWG * 128 + 32, sm, st>>>(xm, wm, p);\n", "  pdlk(wgr<BN, QS, S, NWG>, grid, dim3(NWG * 128 + 32), sm, st, xm, wm, p);\n")


def p_g19(s):
    """router on the side stream: own gv / wgr workspace + counters (p16r); wgr / gv candidates at T 128 / 256;
    v22 GEMV candidates: qkv m16 / m32 push (V 4) + uncapped (V 8), router push"""
    s = sub(s, """    ("qkv_gate", 2): [((1, 4, 16, 2, 1, 0), (1,)), ((1, 4, 8, 4, 1, 2), (3,))],
""", """    ("qkv_gate", 2): [((1, 4, 16, 2, 1, 0), (1,)), ((1, 4, 8, 4, 1, 2), (3,)), ((1, 4, 16, 2, 1, 2, 1), (1, 5, 13)), ((1, 4, 8, 4, 1, 2, 1), (1, 5, 13))],
    ("qkv_gate", 4): [((1, 4, 8, 4, 1, 2, 1), (1, 5, 13)), ((1, 4, 16, 2, 1, 2, 1), (1, 5, 13))],
""")
    s = sub(s, """    ("router", 1): [((1, 8, 2, 8, 1, 0), (0,)), ((1, 8, 1, 16, 1, 0), (0,))],
    ("router", 2): [((1, 8, 2, 8, 1, 0), (0,))],
    ("router", 4): [((1, 4, 2, 16, 1, 0), (0,)), ((1, 8, 2, 8, 1, 0), (0,))],
""", """    ("router", 1): [((1, 8, 2, 8, 1, 0), (0, 4)), ((1, 8, 1, 16, 1, 0), (0,))],
    ("router", 2): [((1, 8, 2, 8, 1, 0), (0, 4))],
    ("router", 4): [((1, 4, 2, 16, 1, 0), (0, 4)), ((1, 8, 2, 8, 1, 0), (0, 4))],
""")
    s = sub(s, '            c["gv"] = lambda h, lg: P__gvcall(p16, p, h, lg)\n', '            c["gv"] = lambda h, lg: P__gvcall(st["p16r"], p, h, lg)\n')
    s = sub(s, """            c["lt"] = lt
    return c
""", """            c["lt"] = lt
        p = p16["gv"].get(("router", T))
        if p is not None:
            c["gv"] = lambda h, lg: P__gvcall(st["p16r"], p, h, lg)
    return c
""")
    s = sub(s, """        st["p16"] = P_prepare({k: weights[k] for k in ("qkv_gate", "o", "pooler", "unpooler", "router")})
""", """        st["p16"] = P_prepare({k: weights[k] for k in ("qkv_gate", "o", "pooler", "unpooler", "router")})
        dv = weights["router"].device
        st["p16r"] = dict(st["p16"], gvws=torch.zeros((1 << 22,), dtype=torch.float32, device=dv),
                          gvcnt=torch.zeros((1 << 16,), dtype=torch.int32, device=dv), wrcnt=torch.zeros((4096,), dtype=torch.int32, device=dv))
""")
    return s


EPI = r'''
template <int NT>
__device__ __forceinline__ float ebsum(float v, float* red) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  __syncthreads();
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < NT / 32; i++) s += red[i];
  return s;
}
__device__ __forceinline__ void eld8(const float* q, float* v) { const float4 x = __ldg((const float4*)q), y = __ldg((const float4*)q + 1); v[0] = x.x; v[1] = x.y; v[2] = x.z; v[3] = x.w; v[4] = y.x; v[5] = y.y; v[6] = y.z; v[7] = y.w; }
// fused glue (DL_FUSE): the last CTA of the grid to finish a token block runs the row op on the block's bf16 outputs (read back
// through L2). epi 1 / 3: resid (s' = al * s + be * rms(out * ga, 1e-8), eout = bf16(rms(s', 1e-5) * egn), eso = s', H = 8192);
// epi 2: rmsg (eout = bf16(rms(out, 1e-5) * egn) over N columns)
template <int NT, int TK>
__device__ __forceinline__ void gv_epi(const G& p) {
  __shared__ int elast;
  __shared__ float ered[32];
  __syncthreads();
  if (threadIdx.x == 0) { __threadfence(); elast = atomicAdd(p.ecnt + blockIdx.x, 1) == (int)(gridDim.y * gridDim.z) - 1; }
  __syncthreads();
  if (!elast) return;
  __threadfence();
  const int r0 = blockIdx.x * TK, r1 = min(p.m, r0 + TK);
  for (int r = r0; r < r1; r++) {
    const __nv_bfloat16* a = (const __nv_bfloat16*)p.out + (size_t)r * p.N;
    if (p.epi == 2) {
      float ss = 0.f;
      for (int c = threadIdx.x * 8; c < p.N; c += NT * 8) {
        const uint4 u = __ldcg((const uint4*)(a + c)); const __nv_bfloat16* b = (const __nv_bfloat16*)&u;
#pragma unroll
        for (int i = 0; i < 8; i++) { const float x = __bfloat162float(b[i]); ss += x * x; }
      }
      const float rr = rsqrtf(ebsum<NT>(ss, ered) / p.N + 1e-5f);
      for (int c = threadIdx.x * 8; c < p.N; c += NT * 8) {
        const uint4 u = __ldcg((const uint4*)(a + c)); const __nv_bfloat16* b = (const __nv_bfloat16*)&u;
        float gv8[8]; eld8(p.egn + c, gv8);
        __nv_bfloat16 o8[8];
#pragma unroll
        for (int i = 0; i < 8; i++) o8[i] = __float2bfloat16(__bfloat162float(b[i]) * rr * gv8[i]);
        *(uint4*)(p.eout + (size_t)r * p.N + c) = *(uint4*)o8;
      }
      continue;
    }
    constexpr int HE = 8192;
    float ss = 0.f;
    for (int c = threadIdx.x * 8; c < p.N; c += NT * 8) {
      const uint4 u = __ldcg((const uint4*)(a + c)); const __nv_bfloat16* b = (const __nv_bfloat16*)&u;
      float g8[8]; if (p.ega) eld8(p.ega + c, g8);
#pragma unroll
      for (int i = 0; i < 8; i++) { const float x = __bfloat162float(b[i]) * (p.ega ? g8[i] : 1.f); ss += x * x; }
    }
    const float ra = rsqrtf(ebsum<NT>(ss, ered) / HE + 1e-8f);
    auto vrow = [&](int c, float* v) {
      float x[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, sv[8], av[8], bv[8];
      if (c < p.N) {
        const uint4 u = __ldcg((const uint4*)(a + c)); const __nv_bfloat16* b = (const __nv_bfloat16*)&u;
        float g8[8]; if (p.ega) eld8(p.ega + c, g8);
#pragma unroll
        for (int i = 0; i < 8; i++) x[i] = __bfloat162float(b[i]) * (p.ega ? g8[i] : 1.f);
      }
      eld8(p.es + (size_t)r * HE + c, sv); eld8(p.eal + c, av); eld8(p.ebe + c, bv);
#pragma unroll
      for (int i = 0; i < 8; i++) v[i] = av[i] * sv[i] + bv[i] * (x[i] * ra);
    };
    float s2 = 0.f;
    for (int c = threadIdx.x * 8; c < HE; c += NT * 8) {
      float v[8]; vrow(c, v);
#pragma unroll
      for (int i = 0; i < 8; i++) s2 += v[i] * v[i];
    }
    const float rs = rsqrtf(ebsum<NT>(s2, ered) / HE + 1e-5f);
    for (int c = threadIdx.x * 8; c < HE; c += NT * 8) {
      float v[8], gv8[8]; vrow(c, v); eld8(p.egn + c, gv8);
      __nv_bfloat16 o8[8];
#pragma unroll
      for (int i = 0; i < 8; i++) o8[i] = __float2bfloat16(v[i] * rs * gv8[i]);
      float4* sp = (float4*)(p.eso + (size_t)r * HE + c);
      sp[0] = make_float4(v[0], v[1], v[2], v[3]); sp[1] = make_float4(v[4], v[5], v[6], v[7]);
      *(uint4*)(p.eout + (size_t)r * HE + c) = *(uint4*)o8;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) p.ecnt[blockIdx.x] = 0;
}
'''


def p_epi(s):
    """DL_FUSE: gv runs the last-arriver glue epilogue at its exits; host gv_epi() arms it for the next gvrun"""
    s = sub(s, "struct G { const uint4* W; const uint4* W2; const __nv_bfloat16* x; void* out; const uint32_t* esc; float4* ws; int* cnt; int m, N, K, KC; uint32_t T0, T1, U0, U1; };",
            "struct G { const uint4* W; const uint4* W2; const __nv_bfloat16* x; void* out; const uint32_t* esc; float4* ws; int* cnt; int m, N, K, KC; uint32_t T0, T1, U0, U1;\n"
            "  int epi; int* ecnt; const float* es; const float* eal; const float* ebe; const float* egn; const float* ega; float* eso; __nv_bfloat16* eout; };")
    # the epilogue runs at every (CTA-uniform) exit of gv, compiled only into the MT 1 instantiations (T = 1); the kernel body is
    # otherwise untouched (a gv_body + wrapper split made ptxas spill in the MT 4 kernels)
    k0 = "__global__ void __launch_bounds__(WR * WK * 32, XS == 2 && !(V & 8) && WR * WK * 32 < 512 ? 512 / (WR * WK * 32) : 1) gv(G p) {"
    s = sub(s, "\ntemplate <int FMT, int MT, int ST, int WR, int WK, int KU, int S, int RT, int XS, int V>\n" + k0,
            EPI + "#define GV_EPI() do { if constexpr (MT == 1) { if (p0.epi) gv_epi<WR * WK * 32, MT * 8>(p0); } } while (0)\n"
            "\ntemplate <int FMT, int MT, int ST, int WR, int WK, int KU, int S, int RT, int XS, int V>\n" + k0)
    a = s.index(k0); b = s.index("static int64_t gvl(", a)
    body = s[a:b]
    assert body.count("return;") == 5, body.count("return;")
    body = body.replace("return;", "{ GV_EPI(); return; }")
    body = sub(body, "  const int bz = blockIdx.x, bx = blockIdx.y, by = blockIdx.z, GX = gridDim.y;\n",
               "  const int bz = blockIdx.x, bx = blockIdx.y, by = blockIdx.z, GX = gridDim.y;\n  const G p0 = p;  // unshifted, for the fused epilogue\n")
    body = sub(body, "#endif\n}\n\ntemplate <int FMT, int MT, int ST, int WR, int WK, int KU, int S, int RT, int XS, int V>\n",
               "#endif\n  GV_EPI();\n}\n\ntemplate <int FMT, int MT, int ST, int WR, int WK, int KU, int S, int RT, int XS, int V>\n")
    s = s[:a] + body + s[b:]
    s = sub(s, "int64_t gvrun(int64_t inst, torch::Tensor W, torch::Tensor W2, torch::Tensor x, torch::Tensor out, torch::Tensor esc, torch::Tensor ws, torch::Tensor cnt, int64_t T, int64_t U) {\n  G p;\n",
            "static G g_epi = {};\n"
            "void gv_epi(int64_t kind, torch::Tensor cnt, torch::Tensor s, torch::Tensor al, torch::Tensor be, torch::Tensor gn, torch::Tensor ga, torch::Tensor so, torch::Tensor out) {\n"
            "  auto fp = [](torch::Tensor t) { return t.numel() ? t.data_ptr<float>() : (float*)nullptr; };\n"
            "  g_epi.epi = (int)kind; g_epi.ecnt = cnt.data_ptr<int>(); g_epi.es = fp(s); g_epi.eal = fp(al); g_epi.ebe = fp(be); g_epi.egn = fp(gn);\n"
            "  g_epi.ega = fp(ga); g_epi.eso = fp(so); g_epi.eout = (__nv_bfloat16*)out.data_ptr();\n}\n"
            "int64_t gvrun(int64_t inst, torch::Tensor W, torch::Tensor W2, torch::Tensor x, torch::Tensor out, torch::Tensor esc, torch::Tensor ws, torch::Tensor cnt, int64_t T, int64_t U) {\n"
            "  G p = g_epi;\n  g_epi.epi = 0;\n")
    s = sub(s, 'cpp_sources="void pdl_set(int64_t); int64_t gvrun(', 'cpp_sources="void pdl_set(int64_t); void gv_epi(int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor); int64_t gvrun(')
    s = sub(s, 'functions=["gvrun", "rrrun", "pdl_set"]', 'functions=["gvrun", "rrrun", "pdl_set", "gv_epi"]')
    return sub(s, '"gv21p%d_%%x"' % LATE, '"gv21p%de_%%x"' % LATE)


def p_fuse(s):
    """kernel_fn: the three glue launches become epilogues when _fproj finds a gv plan for the shape (else unchanged)"""
    s = sub(s, """    a = _proj(att, st, "o", 8192)
    g = _gext()
    s1 = torch.empty((T, 8192), dtype=torch.float32, device=dev)
    h = torch.empty((T, 8192), dtype=torch.bfloat16, device=dev)
    g.resid1(a, data["stream"], st["a1"], st["b1"], st["g_pre_ffn_norm"], s1, h)
""", """    g = _gext()
    s1 = torch.empty((T, 8192), dtype=torch.float32, device=dev)
    h = torch.empty((T, 8192), dtype=torch.bfloat16, device=dev)
    if not _fproj(att, st, "o", 8192, 1, (data["stream"], st["a1"], st["b1"], st["g_pre_ffn_norm"], None, s1, h)):
        a = _proj(att, st, "o", 8192)
        g.resid1(a, data["stream"], st["a1"], st["b1"], st["g_pre_ffn_norm"], s1, h)
""")
    s = sub(s, """    pr = _proj(h, st, "pooler", 4096)
    p = torch.empty_like(pr)
    g.rmsgain(pr, st["g_pre_expert_norm"], p)
""", """    p = torch.empty((T, 4096), dtype=torch.bfloat16, device=dev)
    if not _fproj(h, st, "pooler", 4096, 2, (None, None, None, st["g_pre_expert_norm"], None, None, p)):
        pr = _proj(h, st, "pooler", 4096)
        g.rmsgain(pr, st["g_pre_expert_norm"], p)
""")
    return sub(s, """    f = _proj(y, st, "unpooler", 1024 if T <= 8 else 8192)
    s2 = torch.empty_like(s1)
    hid = torch.empty_like(h)
    g.resid2(f, st["g_post_ffn_norm"], s1, st["a2"], st["b2"], st["g_next_pre_attn_norm"], s2, hid)
    return hid, s2, ids
""", """    s2 = torch.empty_like(s1)
    hid = torch.empty_like(h)
    if not _fproj(y, st, "unpooler", 1024 if T <= 8 else 8192, 3, (s1, st["a2"], st["b2"], st["g_next_pre_attn_norm"], st["g_post_ffn_norm"], s2, hid)):
        f = _proj(y, st, "unpooler", 1024 if T <= 8 else 8192)
        g.resid2(f, st["g_post_ffn_norm"], s1, st["a2"], st["b2"], st["g_next_pre_attn_norm"], s2, hid)
    return hid, s2, ids
""")


def p_glue(s):
    s = sub(s, "#include <ATen/cuda/CUDAContext.h>\n", "#include <ATen/cuda/CUDAContext.h>\n" + MAC)
    s = sub(s, '"dl_glue_v4"', '"dl_glue_v4p%d"' % LATE)
    s = sub(s, 'cpp_sources="void resid1(', 'cpp_sources="void pdl_set(int64_t); void resid1(')
    s = sub(s, 'functions=["resid1",', 'functions=["pdl_set", "resid1",')
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
    ma = 'p, loc, tw, ms["w1"], ms["w2"], ms["g1"], ms["g2"], ms["nw"], inter, y2, ss, '
    s = sub(s, 'M__ext(M_LARGE and T > 64).moe(' + ma + 'ms["ctr"], ms["tcnt"], ms["ws"], y, 0, ms["prof"])',
            'y = M_kernel_fn({"hidden_states": p, "topk_ids": loc, "topk_weights": tw, "config": {"local_start": 0}}, ms)  # MoE v30: per-M tuned kernel')
    s = sub(s, "_gext(); A__ext(); P__ext(); M__ext(M_LARGE)", "_gext(); A__ext(); P__ext(); M__ext(False)")
    return sub(s, rc, "_router(h, lg, st)")


def tmpf(name, text):
    p = TMP / name; p.write_text(text); return p


POLICY = '''
M_TUNE_M = (1, 8, 16, 32, 128, 256)  # MoE v30 per-M tune: the layer's token counts only (buckets cover the tests' T)

_PDLX = ("attn", "glue", "gv", "rtr", "moe")


def _pdlpol(T):
    return @@POL@@


_kernel_fn19 = kernel_fn


def kernel_fn(data, st):
    pol = _pdlpol(data["config"]["tokens"])
    A__ext().pdl_set(int(pol["attn"])); _gext().pdl_set(int(pol["glue"])); P__gvext().pdl_set(int(pol["gv"])); P__ext().pdl_set(int(pol["rtr"]))
    if P__wbext():
        P__wbext().pdl_set(int(pol["gv"]))
    for k in M_V17_KSF:
        M__extS(k).pdl_set(int(pol["moe"]))
    for c in M_FS_CFG:
        M__extF(*c).pdl_set(int(pol["moe"]))
    for c in globals().get("M_G_CFG", ()):
        M__extG(c).pdl_set(int(pol["moe"]))
    for c in globals().get("M_K_CFG", ()):
        M__extK(c).pdl_set(int(pol["moe"]))
    return _kernel_fn19(data, st)


# MoE builds pruned to what can win in the layer (MoE v48d judge tune: M1 / M8 gk SW 4 KS2 1 MAXM 8 (36 / 83; g1 RT 16 38 at M1),
# M16 gk SW 4 KS2 2, M32 r1, M128 r1 (wx RPW 2 within 2%), M256 wx NCW 16): v17 KSF 0 (v0, the default), fs f1s3 (M32 pin), ring L(0) / L(1)
DL_PRUNE = True
if DL_PRUNE:
    M_V17_KSF = (0,)
    M_FS_CFG = ((1, 3, ""),)
    if "M_G_CFG" in globals(): M_G_CFG = ()
    if "M_K_CFG" in globals(): M_K_CFG = ("-DSW=4 -DKS2=1 -DMAXM=8", "-DSW=4 -DKS2=2")
    if "M_WX_CFG" in globals(): M_WX_CFG = (("-DNCW=16", "-DWXG2"), ("-DNCW=16", "-DN8STEP", "-DWXG2"))  # v52d M128 / M256 picks
DL_PIN = {32: ("fs", 1, 3, "")}  # MoE bucket -> config (judge v22 vs v23 decode32: fs f1s3 219.7 / 383.2 us vs r1 236.1 / 420.2)
DL_PIN_MARGIN = 0.92  # keep the pin unless the in-layer tune's best is faster than the pin by more than 8%
for _c in DL_PIN.values():
    if _c[0] == "fs" and _c[1:] not in M_FS_CFG:
        M_FS_CFG = M_FS_CFG + (_c[1:],)  # a tune candidate too (built in the precompile)
_prepare_dl = prepare


DL_TSTART, DL_TEND = 230, 240  # MoE tune: start only before 230 s into the layer's prepare, no new bucket after 240 s
                                # (judge v25: compile 148 s, p19 prepare 22 s after the MoE)


DL_FUSE = True  # glue in the GEMV epilogues (o -> resid1, pooler -> rmsg, unpooler -> resid2)
DL_FUSE_T = (1,)  # one CTA per token block runs the row op: only pays when the block is a single token


def _fproj(x, st, name, n, kind, args):
    T = x.shape[0]
    if not DL_FUSE or T not in DL_FUSE_T:
        return False
    p = st["p16"]["gv"].get((name, P__bk(T)))
    if p is None or p[4] != n or p[0] >= 1000:
        return False
    out = torch.empty((T, n), dtype=torch.bfloat16, device=x.device)
    e = st["none_f"]
    P__gvext().gv_epi(kind, st["ecnt"][kind], *[e if a is None else a for a in args])
    return P__gvcall(st["p16"], p, x.contiguous(), out) == 0


def prepare(weights):
    import time
    M_DL_T0, M_DL_TSTART, M_DL_TEND = time.time(), DL_TSTART, DL_TEND
    globals().update(M_DL_T0=M_DL_T0, M_DL_TSTART=M_DL_TSTART, M_DL_TEND=M_DL_TEND)
    st = _prepare_dl(weights)
    st["ecnt"] = {k: torch.zeros(4096, dtype=torch.int32, device="cuda") for k in (1, 2, 3)}
    st["none_f"] = torch.empty(0, dtype=torch.float32, device="cuda")
    print("DLPREPARE %.1fs" % (time.time() - M_DL_T0), flush=True)
    ms = st["moe"]
    tb, us = ms.get("tune"), ms.get("tune_us") or {}
    if tb is not None:
        for b, c in DL_PIN.items():
            r = us.get(b, {})
            best = min(r, key=r.get) if r else None
            if c in r and not (best != c and r[best] < DL_PIN_MARGIN * r[c]):
                tb[b] = c
        print("DLMOE picks " + " ".join("M%d:%s%s" % (b, M__nm(c), "(%.0f)" % us[b][c] if c in us.get(b, {}) else "") for b, c in sorted(tb.items())), flush=True)
    return st


def _precompile():
    import threading, time
    t0 = time.time()
    th = [threading.Thread(target=f) for f in [A__ext, lambda: M__extL(0), lambda: M__extL(1), _gext, P__ext, P__gvext, P__wbext] + [lambda k=k: M__extS(k) for k in M_V17_KSF] + [lambda c=c: M__extF(*c) for c in M_FS_CFG] + ([lambda c=c: M__extX(c) for c in M_WX_CFG] if "M__extX" in globals() else []) + [lambda c=c: M__extG(c) for c in globals().get("M_G_CFG", ())] + [lambda c=c: M__extK(c) for c in globals().get("M_K_CFG", ())]]
    for t in th:
        t.start()
    for t in th:
        t.join()
    print("DLCOMPILE %.1fs (%d extension builds)" % (time.time() - t0, len(th)), flush=True)
'''.replace("@@POL@@", POL)
head = '''# okc: problem=muse_decode_layer
"""Decoder layer (v20: bf16_proj v22): c12 register-direct GEMV / wgmma routers / cuBLAS projections + fused split-KV FP8 attention + fused residual norms + NVFP4 W4A16 MoE; PDL selectable per (kernel, T)."""
from typing import TypedDict
import torch
from torch.utils.cpp_extension import load_inline

'''
plain = (R / "muse_decode_layer/submission_plain.py").read_text().splitlines(True)
cls = "".join(plain[15:75])
parts = [head, compose.rename(tmpf("a.py", p_attn(ATTN.read_text())), "A_"), "\n",
         compose.rename(tmpf("m.py", p_moe(MOE.read_text())), "M_"), "\n",
         compose.rename(tmpf("p.py", p_epi(p_proj((R / "muse_bf16_proj/submission_v22.py").read_text()))), "P_"), "\n", cls, "\n",
         p_fuse(p_glue((R / "muse_decode_layer/glue.py").read_text())), "\n", (R / "muse_prefill_layer/glue_pf.py").read_text(), "\n",
         p_g19((R / "muse_decode_layer/glue19.py").read_text()), POLICY]
out = "".join(parts)
compile(out, out_dl.name, "exec")
out_dl.write_text(out)
print("wrote", out_dl, len(out), "pdlk", out.count("pdlk("), "waits", out.count("PDL_WAIT();"))
