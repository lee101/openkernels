import re, sys, collections
L = open(sys.argv[1]).read().splitlines()
ins = []
for l in L:
    m = re.match(r'\s+/\*([0-9a-f]+)\*/\s+(.*?);', l)
    if m:
        ins.append((int(m.group(1), 16), m.group(2).strip()))
# loops: backward BRA targets
lab = {}
for i, l in enumerate(L):
    m = re.match(r'(\.L_x_\d+):', l.strip())
    if m:
        # address of next instruction
        for l2 in L[i + 1:]:
            m2 = re.match(r'\s+/\*([0-9a-f]+)\*/', l2)
            if m2:
                lab[m.group(1)] = int(m2.group(1), 16); break
loops = []
for a, s in ins:
    m = re.search(r'BRA\s+(?:`\()?(\.L_x_\d+)', s)
    if m and m.group(1) in lab and lab[m.group(1)] < a:
        loops.append((lab[m.group(1)], a))
for lo, hi in sorted(set(loops)):
    body = [s for a, s in ins if lo <= a <= hi]
    nh = sum(1 for s in body if 'HMMA' in s)
    if nh < 8:
        continue
    c = collections.Counter()
    for s in body:
        op = s.split()[0]
        if op.startswith('@'):
            op = s.split()[1]
        c[op.split('.')[0]] += 1
    tot = sum(c.values())
    print(f"loop {lo:x}-{hi:x}: {tot} instrs, HMMA {nh}, per HMMA {tot / nh:.1f}: " + " ".join(f"{k}:{v}" for k, v in c.most_common(14)))
