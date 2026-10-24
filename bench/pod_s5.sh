#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0a
cd /workspace/ok/muse_attention_decode
for nd in 512 256 128; do sed "s/^_NSDIV = 512/_NSDIV = $nd/" submission.py > sub_nd$nd.py; done
cd /workspace/ok
for nd in 512 256 128; do echo "== decode NSDIV $nd"; python okb.py muse_attention_decode -s sub_nd$nd.py --no-check --reps 30 --only decode 2>&1 | grep -E '^decode' | awk '{print $1,$2,$NF}' | tr '\n' ' '; echo; done
for v in "" "-DMINB=2" "-DNOGROW=1" "-DRB8=4" "-DRB8=8 -DMINB=2" "-DVALU=1"; do
  echo "== verify [$v]"; OKB_DBG="$v" python okb.py muse_attention_decode --no-check --reps 20 --only verify 2>&1 | grep -E '^verify' | awk '{print $1,$2,$NF}' | tr '\n' ' '; echo
done
echo "== decode layer"; python okb.py muse_decode_layer --no-check --reps 20 --kprof 2>&1 | grep -v -i warn | grep -vE "capture_end"
