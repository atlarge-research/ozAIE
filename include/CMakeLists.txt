# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# cmake needs this line
cmake_minimum_required(VERSION 3.30)

set(_OZAIE_INCLUDE "${CMAKE_CURRENT_LIST_DIR}")
set(_MLIR_AIE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../mlir-aie-v1.3.2/programming_examples")

include("${_MLIR_AIE_ROOT}/mlir_aie_init.cmake")
mlir_aie_init_example()

include("${_MLIR_AIE_ROOT}/common.cmake")

find_package(PkgConfig REQUIRED)

function(use_ozaie target)
    target_include_directories(${target} PUBLIC
	"${_OZAIE_INCLUDE}"
        "${XRT_INC_DIR}"
    )

    target_link_directories(${target} PUBLIC
        ${XRT_LIB_DIR}
    )

    target_link_libraries(${target} PUBLIC
        xrt_coreutil
    )

    target_link_test_utils(${target})
endfunction()
