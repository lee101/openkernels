#!/bin/bash
# builds Accel-Sim (trace-driven) + GPGPU-Sim (accel-sim dev fork, has SM90_H100/H200 cfgs) under virtualcuda/ext
set -e
H=$(cd "$(dirname "$0")/.." && pwd); E=$H/ext; P=$E/prefix/usr
[ -d $E/accel-sim-framework ] || git clone -q --depth 1 https://github.com/accel-sim/accel-sim-framework.git $E/accel-sim-framework
[ -d $E/gpgpu-sim ] || git clone -q https://github.com/accel-sim/gpgpu-sim_distribution.git $E/gpgpu-sim
if ! command -v bison >/dev/null || ! command -v makedepend >/dev/null || ! command -v m4 >/dev/null || [ ! -f /usr/include/zstd.h ]; then  # no sudo: unpack debs
  mkdir -p $E/debs && (cd $E/debs && apt-get download bison flex libfl-dev libfl2 libbison-dev xutils-dev libzstd-dev m4 libsigsegv2 && for d in *.deb; do dpkg -x $d $E/prefix; done)
  export PATH=$P/bin:$PATH BISON_PKGDATADIR=$P/share/bison M4=$P/bin/m4 LIBRARY_PATH=$P/lib/x86_64-linux-gnu:$LIBRARY_PATH LD_LIBRARY_PATH=$P/lib/x86_64-linux-gnu:$E/prefix/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH CPATH=$P/include:$CPATH
fi
export CPATH=$E/accel-sim-framework/util/tracer_nvbit/tracer_tool:$CPATH
export CUDA_INSTALL_PATH=${CUDA_INSTALL_PATH:-/usr/local/cuda-12}
G=$E/accel-sim-framework/gpu-simulator
ln -sfn $E/gpgpu-sim $G/gpgpu-sim
sed -i 's#-o $(BIN_DIR)/accel-sim.out accel-sim.cc main.cc *$#-o $(BIN_DIR)/accel-sim.out accel-sim.cc main.cc -lzstd -lz#' $G/Makefile  # link order fix
cd $G && set +e && source setup_environment.sh && set -e
make -j${J:-16} -C $G > $E/make.log 2>&1 || { rg -n -m5 " error|Error [0-9]|undefined reference" $E/make.log; exit 1; }
ls -la $G/bin/release/accel-sim.out
