# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 Advanced Micro Devices, Inc.

# parameters
# -DXRT_INC_DIR: Full path to src/runtime_src/core/include in XRT cloned repo
# -DXRT_LIB_DIR: Path to xrt_coreutil.lib
# -DTARGET_NAME: Target name to be built
#----------------------------------------------------------------------------------------
# Modified in 2026.
# Original source: https://github.com/Xilinx/mlir-aie/tree/v1.3.2/programming_examples/basic/matrix_multiplication/CMakeLists.txt


# cmake needs this line
cmake_minimum_required(VERSION 3.30)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

include(../../mlir_aie_init.cmake)
mlir_aie_init_example()
project(${ProjectName})
include(../../common.cmake)

add_executable(${currentTarget}
    ${subdir}/test.cpp
)

target_include_directories (${currentTarget} PUBLIC 
    ${XRT_INC_DIR}
)

target_link_directories(${currentTarget} PUBLIC
    ${XRT_LIB_DIR}
)

find_package(PkgConfig REQUIRED)
pkg_check_modules(MPFR REQUIRED IMPORTED_TARGET mpfr)
pkg_check_modules(BLIS REQUIRED IMPORTED_TARGET blis)

# target_link_libraries(${currentTarget} PUBLIC
#     xrt_coreutil
#     PkgConfig::MPFR
#     PkgConfig::BLIS
# )

if(n_aie_cols OR (DEFINED ENV{BLIS_NUM_THREADS} AND "$ENV{BLIS_NUM_THREADS}" GREATER 1))
    message(STATUS "Multi-core build: n_aie_cols=${n_aie_cols}, BLIS_NUM_THREADS=$ENV{BLIS_NUM_THREADS}")
    set(BLIS_LIBRARY blis-mt)
else()
    message(STATUS "Single-core build")
    set(BLIS_LIBRARY blis)
endif()

message(STATUS "Using BLIS library: ${BLIS_LIBRARY}")

target_link_directories(${currentTarget} PUBLIC
    ${BLIS_LIBRARY_DIRS}
)

target_link_libraries(${currentTarget} PUBLIC
    xrt_coreutil
    PkgConfig::MPFR
    ${BLIS_LIBRARY}
)


target_link_test_utils(${currentTarget})