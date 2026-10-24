#!/bin/bash
# ncu_daisy_py.sh <problem> <submission.py> <bench_index> <kernel-regex>: run one scored shape of a submission on daisy's
# RTX 3090 (sm_86 build of the same source) under ncu, metrics for kernels matching the regex. CSV -> stdout.
# Copies only the problem dir files it needs + notes/<problem>.json + sim/okc_case.py to /tmp on daisy. Needs DAISY_SUDO_PW.
set -e
[ -n "$DAISY_SUDO_PW" ] || { echo "set DAISY_SUDO_PW (sudo password on daisy)" >&2; exit 1; }
H=${DAISY_HOST:-lee@daisy}; C=/usr/local/cuda-12.8; D=/tmp/vc_ncu_$USER/okc
R=$(cd "$(dirname "$0")/../.." && pwd); P=$1; S=$2; I=$3; K=$4
M=$(sed -n 's/^M="\(.*\)"$/\1/p;/^M="/,/"$/p' "$(dirname "$0")/ncu_daisy.sh" | tr -d '\\\n"' | sed 's/^M=//')
ssh -o BatchMode=yes $H "mkdir -p $D/$P $D/notes $D/virtualcuda/sim"
scp -q "$R/$P/reference.py" "$R/$P/checks.py" "$S" $H:$D/$P/ 2>/dev/null || scp -q "$R/$P/reference.py" "$S" $H:$D/$P/
[ -f "$R/$P/nvfp4.py" ] && scp -q "$R/$P/nvfp4.py" $H:$D/$P/
scp -q "$R/notes/$P.json" $H:$D/notes/; scp -q "$R/virtualcuda/sim/okc_case.py" $H:$D/virtualcuda/sim/
ssh -o BatchMode=yes $H "cd $D && export CUDA_HOME=$C PATH=$C/bin:\$PATH TORCH_CUDA_ARCH_LIST=8.6 && python3 virtualcuda/sim/okc_case.py $P $P/$(basename $S) $I >/dev/null 2>&1; \
  sudo -S -p '' env PATH=\$PATH CUDA_HOME=$C TORCH_CUDA_ARCH_LIST=8.6 OKC_NOWARM=1 HOME=\$HOME TORCH_EXTENSIONS_DIR=\$HOME/.cache/torch_extensions \
  $C/bin/ncu --csv -k regex:'$K' --metrics $M python3 virtualcuda/sim/okc_case.py $P $P/$(basename $S) $I" <<< "$DAISY_SUDO_PW" | rg '^"'
