# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# cmake needs this line
cmake_minimum_required(VERSION 3.30)

project(oztest LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

add_executable(oztest oztest.cpp)

include("$ENV{OZAIE_CMAKE_INCLUDE}")

use_ozaie(oztest)
