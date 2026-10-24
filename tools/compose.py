"""compose.py: inline a component submission into another file with its top-level names prefixed.
usage (library): rename(src_path, prefix) -> source text"""
import ast, io, tokenize


def toplevel(src):
    names = set()
    for n in ast.parse(src).body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)): names.add(n.name)
        elif isinstance(n, (ast.Assign, ast.AnnAssign)):
            for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                for x in ast.walk(t):
                    if isinstance(x, ast.Name): names.add(x.id)
    return names


def rename(path, prefix, keep=("Config", "Inputs", "Weights")):
    src = open(path).read()
    lines = src.splitlines(True)
    if lines and lines[0].startswith("# okc:"): src = "".join(lines[1:])
    names = toplevel(src) - set(keep)
    out = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.NAME and tok.string in names: tok = tok._replace(string=prefix + tok.string)
        out.append(tok)
    return tokenize.untokenize(out)
