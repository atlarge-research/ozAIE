##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# 
##===----------------------------------------------------------------------===##
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
subdir=ozaie
targetname=oztest
CMAKE_BUILD_TYPE=Debug

oztest: oztest.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake -E env CXXFLAGS="-std=c++23 -DDTYPE_IN=${dtype_in_cpp} -DDTYPE_OUT=${dtype_out_cpp} -DDTYPE_ACC=${dtype_acc_cpp}" \
		cmake ./.. \
		-D CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
		-D CMAKE_C_COMPILER=${CC} -D CMAKE_CXX_COMPILER=${CXX} -DTARGET_NAME=oztest -Dsubdir=ozaie
	cd _build && ${powershell} cmake --build . --config DEBUG
ifeq "${powershell}" "powershell.exe"
	cp _build/oztest.exe $@
else
	cp _build/oztest $@ 
endif

.PHONY: clean
clean:
	rm -rf build _build ${targetname}.exe oztest
