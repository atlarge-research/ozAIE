//===- oz_aie.h ----------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file contains exposed API methods for setting up and calling 
// emulated DGEMM based on the Ozaki scheme on AIE2 and AIE2P architectures of NVIDIA Ryzen NPUs

#pragma once

#include <bits/stdc++.h>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <bit>
#include <format>
#include <string>
#include <utility>
#include <cmath>

#include "test_utils.h"

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

namespace oz_aie {
  inline std::string Node = "MLIR_AIE";
  inline xrt::bo bo_instr;
  inline xrt::bo bo_a;
  inline xrt::bo bo_b;
  inline xrt::bo bo_out;
  inline xrt::bo bo_trace;
  inline xrt::bo bo_tmp1;
  inline int instr_size;
  xrt::hw_context context;
  inline xrt::kernel kernel;
  std::vector<uint32_t> instr_v;
  
  inline constexpr char make_fmt[] = "make -C {} n_aie_cols={} M={} K={} N={} bps={} num_splits={} xclbins_dir={}";
  inline constexpr char config_str_fmt[] = "{}_{}x{}x{}_{}x{}x{}_b{}_d{}_{}c.{}";
  inline constexpr char XCLBIN_MAKE_PATH_VAR[] = "OZAIE_XCLBIN_MAKE_PATH";
  inline constexpr char DEV_VAR[] = "NPU2";
  inline const std::string bin_prefix = "final";
  inline const std::string instr_prefix = "insts";
  inline const std::string xclbin_ext = "xclbin";
  inline const std::string instr_ext = "txt";
  inline const std::string xclbins_relative_path = "/../xclbins/";
  inline const int tile_size = 16;
  inline int device_type = -1;

  // bits per slice
  int b;
  // number of splits
  int d;

  std::pair<int, std::pair<int,int>> accuracy_configs[] = {
      {7, {3, 29} },
      {8, {3, 28} },
      {9, {3, 27} },
      {10, {3, 27} },
      {11, {3, 26} },
      {12, {3, 25} },
      {13, {3, 25} },
      {14, {4, 18} }
  };

  std::pair<int, std::pair<int,int>> perf_configs[] = {
      {7, {7, 8} },
      {8, {7, 8}  },
      {9, {7, 8}  },
      {10, {7, 8}  },
      {11, {7, 8}  },
      {12, {7, 8}  },
      {13, {7, 8}  },
      {14, {7, 8}  }
  };

  enum Mode { accurate=0, fast=1 };

  /// @brief This method convert the Multi-INT64 C Matrix from the NPU to the double-precision C matrix
  /// @param C the multi-INT64 (per element) result C matrix from the NPU
  /// @param result address for the post-conversion result double-precision C matrix 
  /// @param M 
  /// @param N 
  /// @param K 
  /// @param num_splits number of splits used in the Ozaki scheme configuration
  /// @param bits_per_slice bits per slice used in the Ozaki scheme configuration
  void convert_paired_i64_to_double_matrix(int64_t* C, double *result, int M, int N, int K, int num_splits, int bits_per_slice) {
    constexpr __int128_t sign_mask_128 = static_cast<__int128_t>(1) << 127;
    //constexpr int64_t scale_back_bias = static_cast<int64_t>(2 - 2 * 1023) << 51;
    const int total_fraction_space = 2 * (bits_per_slice * num_splits);
    const int fraction_ommitted = bits_per_slice * (num_splits - 1);
    int num_grp_splits[2];
    int num_splits_alignable = 31 / bits_per_slice + 1;

    int numerator = bits_per_slice * (num_splits + 1) - 51 + std::countr_zero(static_cast<uint>(K));
    int min_possible_bins_in_first_group = numerator < 0? 1 : (numerator / bits_per_slice + 1);
    num_grp_splits[0] = std::min(static_cast<signed>(num_splits), std::max(min_possible_bins_in_first_group, num_splits_alignable));
    num_grp_splits[1] = num_splits - num_grp_splits[0];

    //std::cout << "Num of splits in first = " << num_grp_splits[0] << "," << num_grp_splits[1] << "\n";

    for (int row = 0; row < M; row++) {
      for (int col = 0; col < N; col++) {
        __int128_t quad;
        int64_t low = C[row*(N*2)+col*2];
        int64_t high = C[row*(N*2)+col*2+1];

        // restore bits for the higher i64 (preserves sign)
        int64_t high_rbits = (high >> 63) << (64-13);
        // scale back exponent value from npu
        int64_t scale_back = (high & 0x7FF8000000000000L) + (static_cast<int64_t>(2) << 51);
        scale_back -= (static_cast<int64_t>(2 * 1023) << 51);
        scale_back <<=1;
        high = (high & 0x0007FFFFFFFFFFFFL) | high_rbits;

        quad = static_cast<__int128_t>(low);
        quad += static_cast<__int128_t>(high) << (num_grp_splits[0] * bits_per_slice);

        __int128_t quad_sign = (quad & sign_mask_128);
        quad *= (quad_sign >> 126) | 1;    // take absolute

        uint64_t ul_high = static_cast<uint64_t>(quad >> 64);
        uint64_t ul_low = static_cast<uint64_t>((quad << 64) >> 64);
        int lz = std::countl_zero(ul_high);

        if (lz == 64)
        {
            lz += std::countl_zero(ul_low);
        }

        int64_t exponent = (128 - lz) - (total_fraction_space - fraction_ommitted)- 1;
        int shift = (128 - lz) - 53;
        
        uint64_t lu = (shift > 0) ? static_cast<uint64_t>(quad >> shift) : static_cast<uint64_t>(quad << -shift);

        if (lu > 0) {
          lu -= 0x0010000000000000UL; // remove implicit one if present
          lu += static_cast<uint64_t>(quad_sign >> 64) + (static_cast<uint64_t>(exponent+1023) << 52) + static_cast<uint64_t>(scale_back);  // add sign and mantissa
        }

        result[row*N+col] =  std::bit_cast<double>(lu);
      }
    }

  }

  /// @brief clean up the previous setup state (buffers, kernel etc) associated with the NPU dispatch
  void cleanup_xrt()
  {
    // Release buffers first.
    bo_trace = xrt::bo{};
    bo_tmp1  = xrt::bo{};
    bo_out   = xrt::bo{};
    bo_b     = xrt::bo{};
    bo_a     = xrt::bo{};
    bo_instr = xrt::bo{};

    // Release the kernel before its context.
    kernel  = xrt::kernel{};
    context = xrt::hw_context{};

    // Optional: release the instruction data in host memory.
    instr_v.clear();
    instr_size = 0;
}

  /// @brief Setup a new M x N x K matmul configuration of the Ozaki scheme-based emulation with the specified accuracy or performance
  /// mode. This method must be called once whenever a different matmul dimension or mode is required in the DGEMM (see ozaie_DGEMM method below)
  /// If the binaries are available, this method reuses those to setup the NPU dispatch otherwise it invokes a rebuild. In the latter case,
  /// the method may take more time depending on the compilation time of the builds.
  ///  Only square matrices are currently supported with dimension K between 128 and 2048 (inclusive)
  /// @param M 
  /// @param N 
  /// @param K 
  /// @param  
  /// @return 
  int32_t ozAIE_setup(int M, int N, int K, enum Mode mode) {
    std::string make_path;
    int log2K = int(std::log2(K));
    int verbosity = 0;

    if (log2K < 7 || log2K > 11 || (M != K) || (N != K)) {
      std::cerr << "Only square matrices are currently supported" << std::endl;
      std::cerr << "Single side dimension : power of 2 between 128 and 2048 (inclusive)" << std::endl;
      return 1;
    }

    if (mode==Mode::accurate) {
      b = accuracy_configs[log2K - 7].second.first;
      d = accuracy_configs[log2K - 7].second.second;
    } else {
      b = perf_configs[log2K - 7].second.first;
      d = perf_configs[log2K - 7].second.second;
    }

    cleanup_xrt();

    // fetch the xclbin makefile location from enviroment 
    if (const char* value = std::getenv(XCLBIN_MAKE_PATH_VAR)) {
        make_path = value;
    } else {
        std::cerr << "No " << XCLBIN_MAKE_PATH_VAR << " environment variable exists" << std::endl;
        return 2;
    }

    // if device type has not been identified
    if (device_type < 0) {
      if (const char* value = std::getenv(DEV_VAR)) {
          device_type = std::stoi(value);
      } else {
          std::cerr << "No " << DEV_VAR << " environment variable exists to identify NPU type" << std::endl;
          return 2;
      }
    }
    
    // use all 4 columns on NPU1 and all 8 columns on NPU2 for maximum performance
    int n_aie_cols = device_type==0? 4 : 8;

    std::string bins_full_path = std::filesystem::path(__FILE__).parent_path().string() + xclbins_relative_path;
    std::string instr_file = bins_full_path + std::format(config_str_fmt, 
        instr_prefix, M, K, N, tile_size, tile_size, tile_size, b, d, n_aie_cols, instr_ext);
    std::string xclbin_file = bins_full_path + std::format(config_str_fmt, 
        bin_prefix, M, K, N, tile_size, tile_size, tile_size, b, d, n_aie_cols, xclbin_ext);

    // build the instructions and xclbin if configuration not available
    if (!std::filesystem::exists(instr_file) || !std::filesystem::exists(xclbin_file)) {
        std::string command = std::format(make_fmt, make_path.c_str(), 
          n_aie_cols, M, K, N, b, d, 
          bins_full_path
        );
        std::system((command + " clean > /dev/null 2>&1").c_str());
        int build_status = std::system((command + " > /dev/null 2>&1").c_str());

        if (build_status != -1 && WIFEXITED(build_status) && WEXITSTATUS(build_status) == 0) {
            // check files again. if still no files available, return error
            if (!std::filesystem::exists(instr_file) || !std::filesystem::exists(xclbin_file)) {
              std::cerr << "Build command: " << command << std::endl;
              std::cerr << "No binaries created in " << bins_full_path << std::endl;
              return 3;
            }
        } else {
            std::cerr << "Build of fresh binaries failed" << std::endl;
            return 4;
        }  
    } 

    instr_v = test_utils::load_instr_binary(instr_file);
    instr_size = instr_v.size();

    // Start the XRT test code
    // Get a device handle
    unsigned int device_index = 0;
    auto device = xrt::device(device_index);

    // Load the xclbin
    auto xclbin = xrt::xclbin(xclbin_file);

    // Get the kernel from the xclbin
    auto xkernels = xclbin.get_kernels();
    auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                                [Node,verbosity](xrt::xclbin::kernel &k) {
                                  auto name = k.get_name();
                                  return name.rfind(Node, 0) == 0;
                                });
    auto kernelName = xkernel.get_name();
    // register bin
    device.register_xclbin(xclbin);
    // get a hardware context
    context = xrt::hw_context(device, xclbin.get_uuid());  
    // get a kernel handle
    kernel = xrt::kernel(context, kernelName);

    bo_instr = xrt::bo(device, instr_size * sizeof(int),
                            XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    bo_a =
        xrt::bo(device, M * K * sizeof(double), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    bo_b =
        xrt::bo(device, K * N * sizeof(double), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
    bo_out =
        xrt::bo(device, M * N * 2 * sizeof(std::int64_t), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

    bo_tmp1 = xrt::bo(device, 1, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));

    int trace_size = 0;
    // Workaround so we declare a really small trace buffer when one is not used
    int tmp_trace_size = (trace_size > 0) ? trace_size : 1;
    bo_trace = xrt::bo(device, tmp_trace_size*4, XRT_BO_FLAGS_HOST_ONLY,
                            kernel.group_id(7));

    // // Instruction buffer for DMA configurations
    void *bufInstr = bo_instr.map<void *>();
    memcpy(bufInstr, instr_v.data(), instr_size * sizeof(int));
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    return 0;
  }

  /// @brief DGEMM call for M x N x K matmul dimensions. The dimensions should match with the 
  /// most recent configuration setup using the ozAIE_Setup method (see above).
  /// @param M 
  /// @param N 
  /// @param K 
  /// @param A 
  /// @param B 
  /// @param C 
  /// @return 
  int32_t ozAIE_DGEMM(int M, int N, int K, const double *A, const double *B, double *C) {
      int trace_size = 0;
      double *bufA = bo_a.map<double *>();
      double *bufB = bo_b.map<double *>();

      memcpy(bufA, A, M * K * sizeof(double));
      memcpy(bufB, B, K * N * sizeof(double));

      // Initialize outputs; bufOut is results matrix plus tracing info
      char *bufOut = bo_out.map<char *>();
      // the output C matrix from NPU is in multi-INT64 format with 2 INT64s per C element
      memset(bufOut, 0, M * N * 2 * sizeof(std::int64_t));

      char *bufTrace = bo_trace.map<char *>();
      if (trace_size > 0)
        memset(bufTrace, 0, trace_size);

      bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      if (trace_size > 0)
        bo_trace.sync(XCL_BO_SYNC_BO_TO_DEVICE);

      unsigned int opcode = 3;
      auto run = kernel(opcode, bo_instr, instr_size, bo_a, bo_b, bo_out,
                        bo_tmp1, bo_trace);
      ert_cmd_state r = run.wait();
      if (r != ERT_CMD_STATE_COMPLETED) {
        return 3;
      }

      bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      if (trace_size > 0)
        bo_trace.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      
      convert_paired_i64_to_double_matrix((int64_t*) bufOut, C, M, N, K, d, b);

      return 0;
    }
}
