#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH
cd /workspace/ok
timeout 900 python bench/wgg.py 2>&1 | grep -v -i warn | tail -25
