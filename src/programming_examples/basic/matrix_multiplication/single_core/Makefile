#===- Makefile ----------------------------------------------000----===//
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# (c) Copyright 2023 Advanced Micro Devices, Inc.
#===----------------------------------------------------------------------===//
# Modified in 2026 
# Original source: https://github.com/Xilinx/mlir-aie/tree/v1.3.2/programming_examples/basic/matrix_multiplication/single_core/Makefile

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
subdir=single_core
targetname=single_core
aie_py_src=${targetname}_iron_ozaki.py
xclbins_dir?=ozaki_xclbins


# M, K, N: final GEMM dimensions
K?=2048
M?=${K}
N=${K}

# m, k, n: single AIE core GEMM dimensions
m?=16
k?=16
n?=16

# ozaki params
num_splits?=20
bps?=4
dtype_in?=i8
dtype_out?=i32
b_col_maj?=0
verify?=0
full_dump?=0
verbosity?=0
num_log_tiles?=2
opt_perf=1
use_chess?=0
run_log_file=oz_dump_K${K}_b${bps}_d${num_splits}.log

# for random matrices generation in accuracy tests 
fi?=4
seed?=12445

kernels=mm_ozaki
kernel_target_suffix=_${m}x${k}x${n}_b${bps}_d${num_splits}
target_suffix=${M}x${K}x${N}_${m}x${k}x${n}_b${bps}_d${num_splits}
aieargs+=-m $m -k $k -n $n --b-col-maj ${b_col_maj}
runargs+=--b_col_maj ${b_col_maj} --verbosity ${verbosity}
debug?=0

# set this flag to 1 for linear buffer allocation
# else, 0 for bank-aware
use_linear_buf_alloc?=1

ifeq (${use_linear_buf_alloc}, 1)
buffer_aloc_flag=basic-sequential
else
buffer_aloc_flag=bank-aware
endif

KERNEL_DEFINES=-D${dtype_in}_${dtype_out}_ONLY -DDIM_K_GLOBAL=${K} -DDIM_M=${m} -DDIM_K=${k} -DDIM_N=${n} -DBITS_PER_SLICE=${bps} -DNUM_SPLITS=${num_splits} -DB_COL_MAJ -DVECTORIZED_ONLY -DOPT_PERF_ENABLED

$(info aie_py_src = $(aie_py_src))
runargs+=--num_splits=${num_splits} --bps=${bps} --mt=${m} --kt=${k} --nt=${n} --fi=${fi} --seed=${seed} --verbosity=${verbosity} --verify=${verify} --full_dump=${full_dump}
aieargs+=--num_splits ${num_splits} --bps ${bps} --dtype_in=${dtype_in}

include ${srcdir}/../makefile-common

ifeq (${full_dump}, 1)
CMAKE_BUILD_TYPE=Debug
KERNEL_DEFINES+=-DDEBUG
runargs+=--num_log_tiles=${num_log_tiles} 
aieargs+=--full_dump ${full_dump} --num_log_tiles ${num_log_tiles}
else
define XCLBIN_BUILD_POST
	@echo "Adding/Replacing generated xclbin and instructions file to ${xclbins_dir}"
	@mkdir -p ${xclbins_dir}
	@cp ${xclbin_target} ${xclbins_dir}
	@cp ${insts_target} ${xclbins_dir}
endef
CMAKE_BUILD_TYPE=Release
endif

build/${kernels}${kernel_target_suffix}.o: ${kernels_dir}/${kernels}.cc
	mkdir -p ${@D}
	cd ${@D} && ${KERNEL_CC} ${KERNEL_CFLAGS} ${KERNEL_DEFINES} -c $< -o ${@F}

runspecific: ${targetname}.exe
	${powershell} ./${targetname}.exe -x ${xclbins_dir}/${xclbin_target:build/%=%} -i ${xclbins_dir}/${insts_target:build/%=%} -k MLIR_AIE -M $M -K $K -N $N ${runargs} > ${run_log_file}
	tail -n 200 ${run_log_file} | grep Error
	
