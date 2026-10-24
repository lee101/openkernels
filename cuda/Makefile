NVCC ?= nvcc
ARCH ?= sm_86
FLAGS := -O3 -arch=$(ARCH)

all: gather reuse cvt split

%: %.cu
	$(NVCC) $(FLAGS) $< -o $@

clean:
	rm -f gather reuse cvt split

.PHONY: all clean
