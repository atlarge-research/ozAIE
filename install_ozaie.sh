#!/usr/bin/env bash 

#!/bin/bash
##===- install_ozaie.sh  --*- Script -*-===##
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# # (c) Copyright 2023 Advanced Micro Devices, Inc.
##===-----------------------------------------------------------------------===##
# 
# Modified in 2026
# Original Source: https://github.com/Xilinx/mlir-aie/blob/v1.3.2/utils/quick_setup.sh
# This script is a modified version maintained independently of the original
# project. Changes have been made for specific development requirements.
# Usage: ./install_ozaie.sh
##===----------------------------------------------------------------------===##
MLIR_VERSION=v1.3.2
MLIR_REPO_DIR=mlir-aie-${MLIR_VERSION}
MAKE_XCLBIN_PATH=$PWD/${MLIR_REPO_DIR}/programming_examples/basic/matrix_multiplication/whole_array

mkdir ${MLIR_REPO_DIR}
git clone --branch ${MLIR_VERSION} --depth 1 https://github.com/Xilinx/mlir-aie.git ${MLIR_REPO_DIR}

# original MLIR quick_setup.sh workflow starts here
echo "Setting up RyzenAI developement tools..."
if [ -z "${WSL_DISTRO_NAME-}" ]; then
  XRTSMI=`which xrt-smi`
  if ! test -f "$XRTSMI"; then 
    echo "xrt-smi not found. Is XRT installed?"
    return 1
  fi
  NPU=`xrt-smi examine | grep -E "NPU Phoenix|NPU Strix|NPU Strix Halo|NPU Krackan|RyzenAI-npu[1456]"`
  if echo "$NPU" | grep -qE "NPU Phoenix|NPU Strix|NPU Strix Halo|NPU Krackan|RyzenAI-npu[1456]"; then
    echo "AMD XDNA NPU found: "
    echo $NPU
  else
    echo "NPU not found. Is the amdxdna driver installed?"
    return 1
  fi
else
  echo "Environment is WSL"
  NPU="${NPU:-$(/mnt/c/Windows/System32/AMD/xrt-smi.exe examine 2>/dev/null | tr -d '\r' | grep -E 'NPU Phoenix|NPU Strix|NPU Strix Halo|NPU Krackan|RyzenAI-npu[1456]' || true)}"
fi
# Check if the current environment is NPU2
# npu4 => Strix, npu5 => Strix Halo, npu6 => Krackan
if echo "$NPU" | grep -qiE "NPU Strix|NPU Strix Halo|NPU Krackan|RyzenAI-npu[456]"; then
    export NPU2=1
else
    export NPU2=0
fi

#sudo apt-get update
#sudo DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential clang clang-14 lld lld-14 cmake ninja-build python3-venv python3-pip uuid-dev

if hash python3.12; then
   echo "Using python version `python3.12 --version`"
   my_python=python3.12
else
  echo "This script requires python3.12"
   return 1
fi

pushd ${MLIR_REPO_DIR}
# If an install is already present, remove it and start from a clean slate
$my_python -m venv ironenv
source ironenv/bin/activate
python3 -m pip install --upgrade pip

# upgrade cmake >= 3.30
python3 -m pip install --upgrade cmake

python3 -m pip install mlir_aie -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/${MLIR_VERSION}
export MLIR_AIE_INSTALL_DIR="$(pip show mlir_aie | grep ^Location: | awk '{print $2}')/mlir_aie"

# we instead install a specfific peano version used in the last stages of development
wget https://github.com/Xilinx/mlir-aie/raw/ef24eba6a31114b04970a2a965ab466d34ea67ab/utils/peano-requirements.txt -P utils
python3 -m pip install -r utils/peano-requirements.txt
export PEANO_INSTALL_DIR="$(pip show llvm-aie | grep ^Location: | awk '{print $2}')/llvm-aie"

pip install pre-commit

# This installs the pre-commit hooks defined in .pre-commit-config.yaml
pre-commit install --hook-type pre-commit --hook-type pre-push

# Setup environment
source utils/env_setup.sh

pushd programming_examples
echo "PATH              : $PATH"
echo "LD_LIBRARY_PATH   : $LD_LIBRARY_PATH"
echo "PYTHONPATH        : $PYTHONPATH"

popd
popd
echo $PWD
mkdir -p xclbins
cp -r src/* ${MLIR_REPO_DIR}/

echo "source ${MLIR_REPO_DIR}/ironenv/bin/activate" > ozaie_env_setup.sh
echo "source /opt/xilinx/xrt/setup.sh" >> ozaie_env_setup.sh
echo "source ${MLIR_REPO_DIR}/utils/env_setup.sh" >> ozaie_env_setup.sh 
echo "export OZAIE_XCLBIN_MAKE_PATH=${MAKE_XCLBIN_PATH}" >> ozaie_env_setup.sh
echo "export OZAIE_CMAKE_INCLUDE=${PWD}/include/CMakeLists.txt" >> ozaie_env_setup.sh
echo "source ~/aocl/5.0.0/gcc/amd-libs.cfg" >>  ozaie_env_setup.sh



