//===- test.cpp -------------------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// 
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
// Modified in 2026 
// Original source: https://github.com/Xilinx/mlir-aie/tree/v1.3.2/programming_examples/basic/matrix_multiplication/test.cpp
// Modified for the ozaki scheme-based DGEMM under programming_examples/basic/matrix_multiplication/
// Verification has been extended to use emulated high-precision using MPFR library
// evaluation of Ozaki scheme-based DGEMM with AOCL-BLIS baseline and the naive DGEMM implementation
// Accuracy verification againt MPFR lib-based high-precision emulated results

#include "cxxopts.hpp"
#include <bits/stdc++.h>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdfloat>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "blis.h"
#include "cblas.h"

#include "common.h"

#ifndef DATATYPES_USING_DEFINED
#define DATATYPES_USING_DEFINED
//#ifndef DTYPE_IN
//#define DTYPE_IN std::bfloat16_t
#define DTYPE_IN double
//#endif
//#ifndef DTYPE_OUT
//#define DTYPE_OUT std::bfloat16_t
#define DTYPE_OUT std::int64_t
//#endif
// #ifndef DTYPE_ACC
#define DTYPE_ACC std::int64_t
// #endif
using A_DATATYPE = DTYPE_IN;
using B_DATATYPE = DTYPE_IN;
using C_DATATYPE = DTYPE_OUT;
using ACC_DATATYPE = DTYPE_ACC;
#endif

#define XSTR(X) STR(X)
#define STR(X) #X

constexpr long long verify_stochastic_threshold = 2048LL * 2048 * 2048;
constexpr int verify_stochastic_n_samples = 1000;

// Verification tolerance
// See "Note on Numerical Tolerances" in README.md
double abs_tol = matmul_common::get_abs_tol<double>();
double rel_tol = matmul_common::get_rel_tol<double>();

int main(int argc, const char *argv[]) {
  // Program arguments parsing
  cxxopts::Options options("Matrix Matrix Multiplication Test");
  cxxopts::ParseResult vm;
  matmul_common::add_default_options(options);

  matmul_common::parse_options(argc, argv, options, vm);
  int verbosity = vm["verbosity"].as<int>();
  int do_verify = vm["verify"].as<bool>();
  int do_full_dump = vm["full_dump"].as<bool>();
  int n_iterations = vm["iters"].as<int>();
  int n_warmup_iterations = vm["warmup"].as<int>();
  int n_iterations_dgemm = vm["iters_dgemm"].as<int>();
  int n_warmup_iterations_dgemm = vm["warmup_dgemm"].as<int>();
  int n_iterations_blis = vm["iters_blis"].as<int>();
  int n_warmup_iterations_blis = vm["warmup_blis"].as<int>();
  int cold_cache_blis = vm["cold_cache_blis"].as<bool>();
  int trace_size = vm["trace_sz"].as<int>();
  int b_col_maj = vm["b_col_maj"].as<int>();
  int c_col_maj = vm["c_col_maj"].as<int>();
  int num_splits = vm["num_splits"].as<int>();
  int bits_per_slice = vm["bps"].as<int>();
  int n_bits_ref_precision = vm["n_bits_ref_precision"].as<int>();
  int seed = vm["seed"].as<int>();
  double fi = vm["fi"].as<double>();
  int num_log_tiles = vm["num_log_tiles"].as<int>();
  std::string devicename = vm["dev"].as<std::string>();

  dim_t blis_nt = bli_thread_get_num_threads();

  // Fix the seed to ensure reproducibility in CI.
  srand(1726250518); // srand(time(NULL));

  int M = vm["M"].as<int>();
  int K = vm["K"].as<int>();
  int N = vm["N"].as<int>();

  int mt = vm["mt"].as<int>();
  int kt = vm["kt"].as<int>();
  int nt = vm["nt"].as<int>();

  
  bool do_verify_stochastic =
      (long long)M * N * K > verify_stochastic_threshold;

  if (verbosity >= 1) {
    std::cout << "Matrix size " << M << "x" << K << "x" << N << std::endl;
  }

  
  int A_VOLUME = M * K;
  int B_VOLUME = K * N;
  int C_VOLUME = M * N * 2;

  if (do_full_dump) {
      int K_div_k = K / kt;
      int dump_length_factor = (1 + K_div_k * (2 + num_splits) + num_splits + 1 + num_log_tiles);
      int n_tiles = (M * N) / (mt * nt);
      C_VOLUME = n_tiles * mt * dump_length_factor * (nt * 2);

  }



  std::cout << "Splits " << num_splits << " Bits per slice " << bits_per_slice << std::endl;
  std::cout << "NPU Device " << devicename << std::endl;
  std::cout << "Iters " << n_iterations << "," << n_iterations_blis << "," << n_iterations_dgemm << std::endl;
  std::cout << "Cold cache blis: " << cold_cache_blis << std::endl;
  std::cout << "Number of BLIS threads: " << blis_nt << std::endl;


  if (do_full_dump)
  {
    std::cout << "LogTiles " << num_log_tiles << std::endl;
  }

  size_t A_SIZE = (A_VOLUME * sizeof(A_DATATYPE));
  size_t B_SIZE = (B_VOLUME * sizeof(B_DATATYPE));
  size_t C_SIZE = (C_VOLUME * sizeof(C_DATATYPE));

  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());

  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << "\n";

  // Start the XRT test code
  // Get a device handle
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  // Load the xclbin
  if (verbosity >= 1)
    std::cout << "Loading xclbin: " << vm["xclbin"].as<std::string>() << "\n";
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());

  if (verbosity >= 1)
    std::cout << "Kernel opcode: " << vm["kernel"].as<std::string>() << "\n";
  std::string Node = vm["kernel"].as<std::string>();

  // Get the kernel from the xclbin
  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [Node, verbosity](xrt::xclbin::kernel &k) {
                                 auto name = k.get_name();
                                 if (verbosity >= 1) {
                                   std::cout << "Name: " << name << std::endl;
                                 }
                                 return name.rfind(Node, 0) == 0;
                               });
  auto kernelName = xkernel.get_name();

  if (verbosity >= 1)
    std::cout << "Registering xclbin: " << vm["xclbin"].as<std::string>()
              << "\n";

  device.register_xclbin(xclbin);

  // get a hardware context
  if (verbosity >= 1)
    std::cout << "Getting hardware context.\n";
  xrt::hw_context context(device, xclbin.get_uuid());

  // get a kernel handle
  if (verbosity >= 1)
    std::cout << "Getting handle to kernel:" << kernelName << "\n";
  auto kernel = xrt::kernel(context, kernelName);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_a =
      xrt::bo(device, A_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_b =
      xrt::bo(device, B_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_out =
      xrt::bo(device, C_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

  auto bo_tmp1 = xrt::bo(device, 1, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));

  // Workaround so we declare a really small trace buffer when one is not used
  int tmp_trace_size = (trace_size > 0) ? trace_size : 1;
  auto bo_trace = xrt::bo(device, tmp_trace_size*4, XRT_BO_FLAGS_HOST_ONLY,
                          kernel.group_id(7));

  if (verbosity >= 1) {
    std::cout << "Writing data into buffer objects.\n";
  }

  constexpr uint64_t dbl_bits =  0b0'01111111111'1111111111111111111111111111111111111111111111111111ULL;
  constexpr uint64_t dbl_bits2 = 0b0'01111111111'0100001011010001100001010100010001000010110100011000ULL;
  
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> uniform_dist(-0.5, 0.5);
  std::normal_distribution<double> norm_dist(0.0, 1.0);
  uint64_t test_convert = 0;

  A_DATATYPE *bufA = bo_a.map<A_DATATYPE *>();
  std::vector<A_DATATYPE> AVec(A_VOLUME);
  for (int i = 0; i < A_VOLUME; i++) {
    //AVec[i] = std::bit_cast<double>(((dbl_bits << 12) >> 12) + (1023UL << 52));
    AVec[i] = uniform_dist(rng) * std::exp(fi * norm_dist(rng));
  }

  memcpy(bufA, AVec.data(), (AVec.size() * sizeof(A_DATATYPE)));
  B_DATATYPE *bufB = bo_b.map<B_DATATYPE *>();
  std::vector<B_DATATYPE> BVec(B_VOLUME);
  for (int i = 0; i < B_VOLUME; i++) {
    //BVec[i] = std::bit_cast<double>(((dbl_bits << 12) >> 12) + (1023UL << 52) + (1UL << 63));
    BVec[i] = uniform_dist(rng) * std::exp(fi * norm_dist(rng));
  }
  
  memcpy(bufB, BVec.data(), (BVec.size() * sizeof(B_DATATYPE)));

  // Initialize outputs; bufOut is results matrix plus tracing info
  char *bufOut = bo_out.map<char *>();
  std::vector<C_DATATYPE> CVec(C_VOLUME);
  memset(bufOut, 0, C_SIZE);

  char *bufTrace = bo_trace.map<char *>();
  if (trace_size > 0)
    memset(bufTrace, 0, trace_size);

  if (verbosity >= 2) {
    std::cout << "DTYPE_IN  = " XSTR(DTYPE_IN) "\n";
    std::cout << "DTYPE_OUT = " XSTR(DTYPE_OUT) "\n";
    //std::cout << "Verification tolerance " << abs_tol << " absolute, "
    //          << rel_tol << " relative.\n";
    std::cout << "A = \n";
    matmul_common::print_matrix(AVec, K, M, K);
    std::cout << "B = \n";
    matmul_common::print_matrix(BVec, N, M, N);
  }

  // Instruction buffer for DMA configuration
  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  if (trace_size > 0)
    bo_trace.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned num_iter = n_iterations + n_warmup_iterations;
  float npu_time_total = 0;
  float npu_time_min = 9999999;
  float npu_time_max = 0;
  float dgemm_time_total = 0;
  float dgemm_time_min = 9999999;
  float dgemm_time_max = 0;
  double alpha = 1.0d;
  double beta = 0.0d;
  int lda = M;
  int ldb = M;
  int ldc = M;
  float blis_dgemm_time_total = 0;
  float blis_dgemm_time_min = 9999999;
  float blis_dgemm_time_max = 0;
  std::vector<float> iter_times_npu;
  std::vector<float> iter_times_cpu;
  std::vector<float> iter_times_ozaki;
  std::vector<float> iter_times_DGEMM;
  std::vector<float> iter_times_BLIS_DGEMM;

  std::vector<double> CVec_DGEMM(M * N);
  std::vector<double> CVec_BLIS_DGEMM(M * N);
  std::vector<double> CVec_double(M * N);

  std::vector<std::vector<double>> input_A_blis;
  std::vector<std::vector<double>> input_B_blis;
  std::vector<std::vector<double>> output_C_blis;

  int errors = 0;
  float macs = 2.0 * float(M) * float(K) * float(N);

  for (unsigned iter = 0; iter < num_iter; iter++) {

    if (verbosity >= 1) {
      std::cout << "Running Kernel (iteration " << iter << ").\n";
    }
    auto start = std::chrono::high_resolution_clock::now();
    unsigned int opcode = 3;
    auto run = kernel(opcode, bo_instr, instr_v.size(), bo_a, bo_b, bo_out,
                      bo_tmp1, bo_trace);
    ert_cmd_state r = run.wait();
    if (r != ERT_CMD_STATE_COMPLETED) {
      std::cout << "Kernel did not complete. Returned status: " << r << "\n";
      return 1;
    }

    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    if (trace_size > 0)
      bo_trace.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    auto stop = std::chrono::high_resolution_clock::now();
    
    auto cpu_phase_start = std::chrono::high_resolution_clock::now();
   
    matmul_common::convert_paired_i64_to_double_matrix((int64_t*) bufOut, CVec_double.data(), M, N, K, num_splits, bits_per_slice);
    
    auto cpu_phase_stop = std::chrono::high_resolution_clock::now();
    
    memcpy(CVec.data(), bufOut, (CVec.size() * sizeof(C_DATATYPE)));

    if (iter < n_warmup_iterations) {
      /* Warmup iterations do not count towards average runtime. */
      continue;
    }

    float npu_time =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
            .count();

    float cpu_phase_time =
        std::chrono::duration_cast<std::chrono::microseconds>(cpu_phase_stop - cpu_phase_start)
            .count();
    

    iter_times_npu.push_back(npu_time);
    iter_times_cpu.push_back(cpu_phase_time);

    npu_time += cpu_phase_time;

    npu_time_total += npu_time;
    npu_time_min = (npu_time < npu_time_min) ? npu_time : npu_time_min;
    npu_time_max = (npu_time > npu_time_max) ? npu_time : npu_time_max;
  }

  for (unsigned iter = 0; iter < (n_iterations_dgemm + n_warmup_iterations_dgemm); iter++) {

    if (verbosity >= 1) {
      std::cout << "Running DGEMM (iteration " << iter << ").\n";
    }
    auto dgemm_start = std::chrono::high_resolution_clock::now();
    
    // the second normal DGEMM baseline
    matmul_common::matmul_DGEMM(M, N, K, AVec,
          BVec, CVec_DGEMM, b_col_maj, c_col_maj);

    auto dgemm_stop = std::chrono::high_resolution_clock::now();

    if (iter < n_warmup_iterations_dgemm) {
      /* Warmup iterations do not count towards average runtime. */
      continue;
    }

    float dgemm_time =
        std::chrono::duration_cast<std::chrono::microseconds>(dgemm_stop - dgemm_start)
            .count();
    
    iter_times_DGEMM.push_back(dgemm_time);

    dgemm_time_total += dgemm_time;
    dgemm_time_min = (dgemm_time < dgemm_time_min) ? dgemm_time : dgemm_time_min;
    dgemm_time_max = (dgemm_time > dgemm_time_max) ? dgemm_time : dgemm_time_max;
  }

  if (cold_cache_blis>0) {
      input_A_blis.reserve(n_iterations_blis);
      input_B_blis.reserve(n_iterations_blis);
      output_C_blis.reserve(n_iterations_blis);

      // prepeare random set of inputs to reduce warm-cache effect of BLIS
      for (unsigned iter = 0; iter < n_iterations_blis; iter++) {
        std::vector<double> a_rand(A_VOLUME);
        std::vector<double> b_rand(B_VOLUME);
        std::vector<double> c_rand_out(M*N);

        for (int i = 0; i < A_VOLUME; i++) {
            a_rand[i] = uniform_dist(rng) * std::exp(fi * norm_dist(rng));
        }

        for (int i = 0; i < B_VOLUME; i++) {
            b_rand[i] = uniform_dist(rng) * std::exp(fi * norm_dist(rng));
        }

        input_A_blis.push_back(std::move(a_rand));
        input_B_blis.push_back(std::move(b_rand));
        output_C_blis.push_back(std::move(c_rand_out));
      }
  }

  for (unsigned iter = 0; iter < (n_iterations_blis + n_warmup_iterations_blis); iter++) {
    double* a_in;
    double* b_in;
    double* c_out;

    if (verbosity >= 0) {
      std::cout << "Running AOCL-BLIS DGEMM (iteration " << iter << ").\n";
    }

    if (cold_cache_blis==0 || iter < n_warmup_iterations_blis) {
        a_in = AVec.data();
        b_in = BVec.data();
        c_out = CVec_BLIS_DGEMM.data();
    } else {
        a_in = input_A_blis[iter - n_warmup_iterations_blis].data();
        b_in = input_B_blis[iter - n_warmup_iterations_blis].data();
        c_out = output_C_blis[iter - n_warmup_iterations_blis].data();
    }


    auto blis_dgemm_start = std::chrono::high_resolution_clock::now();

    // the second normal DGEMM baseline
    cblas_dgemm(
          CblasRowMajor, CblasNoTrans, CblasNoTrans, 
          M, N, K, 
          alpha,a_in,lda, b_in,ldb, 
          beta,c_out,ldc);

    auto blis_dgemm_stop = std::chrono::high_resolution_clock::now();

    if (iter < n_warmup_iterations_blis) {
      /* Warmup iterations do not count towards average runtime. */
      continue;
    }

    float blis_dgemm_time =
        std::chrono::duration_cast<std::chrono::microseconds>(blis_dgemm_stop - blis_dgemm_start)
            .count();
    
    iter_times_BLIS_DGEMM.push_back(blis_dgemm_time);

    blis_dgemm_time_total += blis_dgemm_time;
    blis_dgemm_time_min = (blis_dgemm_time < blis_dgemm_time_min) ? blis_dgemm_time : blis_dgemm_time_min;
    blis_dgemm_time_max = (blis_dgemm_time > blis_dgemm_time_max) ? blis_dgemm_time : blis_dgemm_time_max;
  }

    if (verbosity >= 1) {
      std::cout << "C = \n";
      matmul_common::print_matrix(CVec_double, N, M, N);
    }

    if (do_full_dump || do_verify) {
         std::cout << "A INT64 = \n";
         matmul_common::print_matrix(AVec, K, M, K, true);
         std::cout << "B INT64 = \n";
         matmul_common::print_matrix(BVec, K, M, K, true);
    }

    if (do_full_dump) {
        std::cout << "FULL DUMP = \n";
        matmul_common::print_matrix(CVec, nt*2, C_VOLUME / (nt*2), nt*2);
    } 

    if (do_verify) {
      std::cout << "C Multi-INT64 = \n";
      matmul_common::print_matrix(CVec, N*2, C_VOLUME / (N*2), N*2);

      if (do_verify_stochastic) {
            std::cout << "Verifying " << verify_stochastic_n_samples
                      << " random samples against reference matmul ..."
                      << std::endl;
      } else {
            std::cout << "Verifying against reference matmul ..." << std::endl;
      }
      
      
      

      auto vstart = std::chrono::system_clock::now();

      if (do_verify_stochastic) {
        matmul_common::verify_stochastic(
            M, N, K, AVec, BVec, CVec_BLIS_DGEMM, "BLIS_DGEMM", verify_stochastic_n_samples, n_bits_ref_precision, verbosity,
            abs_tol, rel_tol, b_col_maj, c_col_maj);

        matmul_common::verify_stochastic(
            M, N, K, AVec, BVec, CVec_DGEMM, "DGEMM", verify_stochastic_n_samples, n_bits_ref_precision, verbosity,
            abs_tol, rel_tol, b_col_maj, c_col_maj);

        errors = matmul_common::verify_stochastic(
            M, N, K, AVec, BVec, CVec_double, "Ozaki", verify_stochastic_n_samples, n_bits_ref_precision, verbosity,
            abs_tol, rel_tol, b_col_maj, c_col_maj);
      } else {
         // BLIS DGEMM comparison against MPFR
        matmul_common::verify(
            M, N, K, AVec, BVec, CVec_BLIS_DGEMM, "BLIS_DGEMM", abs_tol, rel_tol, n_bits_ref_precision, verbosity, b_col_maj,
            c_col_maj);


        // DGEMM comparison against MPFR
        matmul_common::verify(
            M, N, K, AVec, BVec, CVec_DGEMM, "DGEMM", abs_tol, rel_tol, n_bits_ref_precision, verbosity, b_col_maj,
            c_col_maj);

        // actual comparision against MPFR
        errors = matmul_common::verify(
            M, N, K, AVec, BVec, CVec_double, "Ozaki", abs_tol, rel_tol, n_bits_ref_precision, verbosity, b_col_maj,
            c_col_maj);
      }

      auto vstop = std::chrono::system_clock::now();
      float vtime =
          std::chrono::duration_cast<std::chrono::seconds>(vstop - vstart)
              .count();
      if (verbosity >= 1) {
        std::cout << "Verify time: " << vtime << " s." << std::endl;
      }
    } else {
      if (verbosity >= 1)
        std::cout << "WARNING: matmul results not verified." << std::endl;
    }

    

  // Only write out trace of last iteration.
  if (trace_size > 0) {
    matmul_common::write_out_trace((char *)bufTrace, trace_size,
                                   vm["trace_file"].as<std::string>());
  }

  for (int iter_time_idx=0; iter_time_idx< n_iterations; iter_time_idx++) {
      iter_times_ozaki.push_back(iter_times_npu[iter_time_idx] + iter_times_cpu[iter_time_idx]);
  }

  if (verbosity>=1) {
    for (int iter_time_idx=0; iter_time_idx< n_iterations; iter_time_idx++) {
      std::cout << "NPU Iter# " << iter_time_idx << " : " << iter_times_npu[iter_time_idx] << std::endl;
    }
    for (int iter_time_idx=0; iter_time_idx< n_iterations; iter_time_idx++) {
      std::cout << "CPU Iter# " << iter_time_idx << " : " << iter_times_cpu[iter_time_idx] << std::endl;
    }
    for (int iter_time_idx=0; iter_time_idx< n_iterations; iter_time_idx++) {
      std::cout << "Ozaki Iter# " << iter_time_idx << " : " << iter_times_ozaki[iter_time_idx] << std::endl;
    }
    for (int iter_time_idx=0; iter_time_idx< n_iterations_dgemm; iter_time_idx++) {
      std::cout << "DGEMM Iter# " << iter_time_idx << " : " << iter_times_DGEMM[iter_time_idx] << std::endl;
    }
    for (int iter_time_idx=0; iter_time_idx< n_iterations_blis; iter_time_idx++) {
      std::cout << "BLIS DGEMM Iter# " << iter_time_idx << " : " << iter_times_BLIS_DGEMM[iter_time_idx] << std::endl;
    }
  }



  std::cout << std::endl
            << "Median DGEMM matmul time: " << matmul_common::median(iter_times_DGEMM) << "us."
            << std::endl;
  std::cout << std::endl
            << "Avg DGEMM matmul time: " << dgemm_time_total / n_iterations_dgemm << "us."
            << std::endl;

  std::cout << std::endl
            << "Median BLIS_DGEMM matmul time: " << matmul_common::median(iter_times_BLIS_DGEMM) << "us."
            << std::endl;
  std::cout << std::endl
            << "Avg BLIS_DGEMM matmul time: " << blis_dgemm_time_total / n_iterations_blis << "us."
            << std::endl;

  std::cout << std::endl
            << "Median NPU time: " << matmul_common::median(iter_times_npu)
              << "us." << std::endl;

  std::cout << std::endl
            << "Median CPU time: " << matmul_common::median(iter_times_cpu)
              << "us." << std::endl;

  std::cout << std::endl
            << "Median Ozaki matmul time: " << matmul_common::median(iter_times_ozaki)
              << "us." << std::endl;
  std::cout << std::endl
            << "Avg Ozaki matmul time: " << npu_time_total / n_iterations << "us."
            << std::endl;
 
  std::cout << "Median DGEMM gflops: "
  << macs / (1000 * matmul_common::median(iter_times_DGEMM)) << std::endl;

  std::cout << "Median BLIS_DGEMM gflops: "
  << macs / (1000 * matmul_common::median(iter_times_BLIS_DGEMM)) << std::endl;       

   std::cout << "Median Ozaki gflops: "
            << macs / (1000 * matmul_common::median(iter_times_ozaki)) << std::endl;   
  std::cout << "Avg Ozaki gflops: "
            << macs / (1000 * npu_time_total / n_iterations) << std::endl;
 


  if (do_verify) {
      if (!errors) {
        std::cout << "\nPASS!\n\n";
        return 0;
      } else {
        std::cout << "\nError count: " << errors;
        if (do_verify_stochastic) {
          std::cout << " (out of " << verify_stochastic_n_samples
                    << " random samples)";
        }
        std::cout << "\n\n";

        std::cout << "\nFailed.\n\n";
        return 1;
      }
  }
}
