//===- common.h ----------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// (c) Copyright 2024 Advanced Micro Devices, Inc.
//===----------------------------------------------------------------------===//
// Modified in 2026 
// Original source: https://github.com/Xilinx/mlir-aie/tree/v1.3.2/programming_examples/basic/matrix_multiplication/common.h
// Modified for the ozaki scheme-based DGEMM under programming_examples/basic/matrix_multiplication/
// Verification has been extended to use emulated high-precision using MPFR library
// additional command-line parameters have been added for evaluation of Ozaki scheme-based DGEMM

#ifndef MATRIX_MULTIPLICATION_H
#define MATRIX_MULTIPLICATION_H

#include <algorithm>
#include <bits/stdc++.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <optional>
#include <ostream>
#include <stdfloat>
#include <bit>
#include <mpfr.h>
#include <gmp.h>

#include "test_utils.h"

namespace matmul_common {

// --------------------------------------------------------------------------
// Command Line Argument Handling
// --------------------------------------------------------------------------

void add_default_options(cxxopts::Options &options) {
  options.add_options()("help,h", "produce help message")(
      "xclbin,x", "the input xclbin path", cxxopts::value<std::string>())(
      "kernel,k", "the kernel name in the XCLBIN (for instance PP_PRE_FD)",
      cxxopts::value<std::string>())("verbosity,v",
                                     "the verbosity of the output",
                                     cxxopts::value<int>()->default_value("0"))(
      "instr,i",
      "path of file containing userspace instructions sent to the NPU",
      cxxopts::value<std::string>())(
      "verify", "whether to verify the AIE computed output",
      cxxopts::value<bool>()->default_value("true"))(
      "rows,M", "Matrix size M", cxxopts::value<int>()->default_value("512"))(
      "inner,K", "Matrix size K", cxxopts::value<int>()->default_value("512"))(
      "columns,N", "Matrix size N",
      cxxopts::value<int>()->default_value("512"))(
      "tile_rows,mt", "Tile rows mt", cxxopts::value<int>()->default_value("32"))(
      "tile_inner,kt", "Tile inner kt", cxxopts::value<int>()->default_value("32"))(
      "tile_columns,nt", "Tile cols nt", cxxopts::value<int>()->default_value("32"))(
        "iters", "number of iterations",
      cxxopts::value<int>()->default_value("1"))(
        "iters_dgemm", "number of iterations for DGEMM",
      cxxopts::value<int>()->default_value("1"))(
        "iters_blis", "number of iterations BLIS DGEMM",
      cxxopts::value<int>()->default_value("1"))(
        "cold_cache_blis", "change inputs and output address space every iteration for BLIS DGEMM",
      cxxopts::value<bool>()->default_value("false"))(
      "warmup", "number of warmup iterations",
      cxxopts::value<int>()->default_value("0"))(
      "warmup_dgemm", "number of warmup iterations for single-core DGEMM",
      cxxopts::value<int>()->default_value("0"))(
      "warmup_blis", "number of warmup iterations for single/multi-core BLIS DGEMM",
      cxxopts::value<int>()->default_value("0"))(
      "trace_sz,t", "trace size", cxxopts::value<int>()->default_value("0"))(
      "trace_file", "where to store trace output",
      cxxopts::value<std::string>()->default_value("trace.txt"))(
      "b_col_maj", "Is B matrix in colum-major format?",
      cxxopts::value<int>()->default_value("0"))(
      "c_col_maj", "Is C matrix in colum-major format?",
      cxxopts::value<int>()->default_value("0"))(
      "num_splits", "number of splits?",
      cxxopts::value<int>()->default_value("0"))(
      "bps", "bits per slice?",
      cxxopts::value<int>()->default_value("0"))(
      "n_bits_ref_precision", "number of bits for reference precision for accuracy tests?",
      cxxopts::value<int>()->default_value("106"))(
      "fi", "exponent distribution?",
      cxxopts::value<double>()->default_value("0.1"))(
      "seed", "seed for random matrix generation?",
      cxxopts::value<int>()->default_value("12345"))(
      "full_dump", "whether to do a full dump in output",
      cxxopts::value<bool>()->default_value("false"))(
      "num_log_tiles", "number of free logging tiles in full dump",
      cxxopts::value<int>()->default_value("1"))(
      "devicename,dev", "npu1/npu2", cxxopts::value<std::string>());
}

void parse_options(int argc, const char *argv[], cxxopts::Options &options,
                   cxxopts::ParseResult &result) {
  try {
    result = options.parse(argc, argv);

    if (result.count("help")) {
      std::cout << options.help() << "\n";
      std::exit(1);
    }

    // Check required options
    if (!result.count("xclbin") || !result.count("kernel") ||
        !result.count("instr")) {
      std::cerr << "Error: Required options missing\n\n";
      std::cerr << "Usage:\n" << options.help() << "\n";
      std::exit(1);
    }

  } catch (const cxxopts::exceptions::parsing &e) {
    std::cerr << e.what() << "\n\n";
    std::cerr << "Usage:\n" << options.help() << "\n";
    std::exit(1);
  }
}

// --------------------------------------------------------------------------
// Matrix / Float / Math
// --------------------------------------------------------------------------

template <typename T>
static inline T get_random();

template <>
std::int16_t get_random<std::int16_t>() {
  return (std::int16_t)rand() % 0x10000;
}

template <>
int8_t get_random<int8_t>() {
  return (int8_t)rand() % 0x100;
}

template <>
std::bfloat16_t get_random<std::bfloat16_t>() {
  // Random numbers should NOT be uniformly between 0 and 1, because that
  // would make the matrix product AB always close to 1.
  return std::bfloat16_t(4.0 * (float)rand() / (float)(RAND_MAX));
}

/**
 * Converts a given double precision matrix to MPFR format matrix with specified precision
 */
std::vector<mpfr_t> to_MPFR(int M, int N, const std::vector<double>& in_mat, int n_bits_ref_precision) {
    std::vector<mpfr_t> result(M*N);

    for (int row = 0; row < M; row++) {
      for (int col = 0; col < N; col++) {
          //mpfr_t elem;
          //mpfr_init2(elem, n_bits_ref_precision);
          //mpfr_set_d(elem, in_mat[row * N + N], MPFR_RNDN);
          //result[row * N + col] = elem;
          mpfr_init2(result[row * N + col], n_bits_ref_precision);
          mpfr_set_d(result[row * N + col], in_mat[row * N + col], MPFR_RNDN);    
      }
    }

    return result;
} 

/** Multiplies given double precision matrices using the precision specified in bits using MPFR api  */
void matmul_MPFR(int M, int N, int K, const std::vector<double> A,
            const std::vector<double> B, std::vector<double> &C, int b_col_maj,
            int c_col_maj, int n_bits_per_ref_precision) {
  std::vector<mpfr_t> A_DD = to_MPFR(M, N, A, n_bits_per_ref_precision);
  std::vector<mpfr_t> B_DD = to_MPFR(M, N, B, n_bits_per_ref_precision);
  mpfr_t prod;
  mpfr_t running_sum;
  mpfr_init2(prod, n_bits_per_ref_precision);
  mpfr_init2(running_sum, n_bits_per_ref_precision);


  for (int row = 0; row < M; row++) {
    for (int col = 0; col < N; col++) {
      
      mpfr_set_d(running_sum, 0.0, MPFR_RNDN);
      
      for (int k = 0; k < K; k++) {
        
        mpfr_set_d(prod, 0.0, MPFR_RNDN);
        
        if (!b_col_maj) {  
            mpfr_mul(prod, A_DD[row * K + k], B_DD[k * N + col], MPFR_RNDN);
        } else {
            mpfr_mul(prod, A_DD[row * K + k], B_DD[k + col * K], MPFR_RNDN);
        }

        mpfr_add(running_sum, running_sum, prod, MPFR_RNDN);
      }

      if (!c_col_maj) {
        C[row * N + col] = mpfr_get_d(running_sum, MPFR_RNDN);
      } else {
        C[row + col * M] = mpfr_get_d(running_sum, MPFR_RNDN);
      }
    }
  }

  for (int row = 0; row < M; row++) {
    for (int col = 0; col < N; col++) {
      mpfr_clear(A_DD[row * N + col]);
      mpfr_clear(B_DD[row * N + col]);
    }
  }

  mpfr_free_cache ();
}

double mul_acc_MPFR(int M, int N, int K, int row, int col, const std::vector<double> A,
             const std::vector<double> B, int b_col_maj, int n_bits_per_ref_precision) {
  mpfr_t elem_a, elem_b, prod, running_sum;
  mpfr_inits2(n_bits_per_ref_precision, elem_a, elem_b, prod, running_sum, (mpfr_ptr) 0 );
  mpfr_set_d(running_sum, 0.0, MPFR_RNDN);

  for (int k = 0; k < K; k++) {
    mpfr_set_d(prod, 0.0, MPFR_RNDN);
    mpfr_set_d(elem_a, A[row * K + k], MPFR_RNDN);

    if (!b_col_maj) {    
        mpfr_set_d(elem_b, B[k * N + col], MPFR_RNDN);
    } else {
       mpfr_set_d(elem_b, B[k + col * K], MPFR_RNDN);
    }

    mpfr_mul(prod, elem_a, elem_b, MPFR_RNDN);
    mpfr_add(running_sum, running_sum, prod, MPFR_RNDN);
  }

  return mpfr_get_d(running_sum, MPFR_RNDN);
}

void matmul_DGEMM(int M, int N, int K, const std::vector<double> A,
            const std::vector<double> B, std::vector<double> &C, int b_col_maj,
            int c_col_maj) {
  for (int row = 0; row < M; row++) {
    for (int col = 0; col < N; col++) {
      double running_sum = double(0.0d);
      for (int k = 0; k < K; k++) {
        if (!b_col_maj) {
          running_sum += A[row * K + k]*B[k * N + col];
        } else {
          running_sum += A[row * K + k]*B[k + col * K];
        }
      }
      if (!c_col_maj) {
        C[row * N + col] = running_sum;
      } else {
        C[row + col * M] = running_sum;
      }
    }
  }
}

double mul_acc_DGEMM(int M, int N, int K, int row, int col, const std::vector<double> A,
             const std::vector<double> B, int b_col_maj) {
  double running_sum = double(0.0d);
  for (int k = 0; k < K; k++) {
    if (!b_col_maj) {
      running_sum += A[row * K + k]*B[k * N + col];
    } else {
       running_sum += A[row * K + k]*B[k + col * K];
    }
  }
  return running_sum;
}


// nearly_equal function adapted from Stack Overflow, License CC BY-SA 4.0
// Original author: P-Gn
// Source: https://stackoverflow.com/a/32334103
bool nearly_equal(float a, float b, float epsilon = 128 * FLT_EPSILON,
                  float abs_th = FLT_MIN)
// those defaults are arbitrary and could be removed
{
  assert(std::numeric_limits<float>::epsilon() <= epsilon);
  assert(epsilon < 1.f);

  if (a == b)
    return true;

  auto diff = std::abs(a - b);
  auto norm =
      std::min((std::abs(a) + std::abs(b)), std::numeric_limits<float>::max());
  // or even faster: std::min(std::abs(a + b),
  // std::numeric_limits<float>::max()); keeping this commented out until I
  // update figures below
  return diff < std::max(abs_th, epsilon * norm);
}

// nearly_equal function adapted from original common.h file from AMD 
// Source: https://github.com/Xilinx/mlir-aie/blob/v1.2.0/programming_examples/basic/matrix_multiplication/common.h
bool nearly_equal(double a, double b, double epsilon = 128 * DBL_EPSILON,
                  double abs_th = DBL_MIN)
// those defaults are arbitrary and could be removed
{
  assert(std::numeric_limits<double>::epsilon() <= epsilon);
  assert(epsilon < 1.f);

  if (a == b)
    return true;

  auto diff = std::abs(a - b);
  auto norm =
      std::min((std::abs(a) + std::abs(b)), std::numeric_limits<double>::max());
  // or even faster: std::min(std::abs(a + b),
  // std::numeric_limits<float>::max()); keeping this commented out until I
  // update figures below
  return diff < std::max(abs_th, epsilon * norm);
}

template <typename T>
static inline float get_abs_tol();
template <typename T>
static inline float get_rel_tol();

template <>
float get_abs_tol<std::int16_t>() {
  return 0.0;
}

template <>
float get_abs_tol<std::int32_t>() {
  return 0.0;
}

template <>
float get_abs_tol<std::bfloat16_t>() {
  return 0.5;
}

template <>
float get_abs_tol<float>() {
  return 0.5;
}

template <>
float get_abs_tol<double>() {
  return 0.5;
}


template <>
float get_abs_tol<int8_t>() {
  return 0;
}
template <>
float get_abs_tol<int64_t>() {
  return 0;
}

template <>
float get_rel_tol<std::int16_t>() {
  return 0.0;
}

template <>
float get_rel_tol<std::int32_t>() {
  return 0.0;
}

template <>
float get_rel_tol<std::int64_t>() {
  return 0.0;
}

template <>
float get_rel_tol<std::bfloat16_t>() {
  return 0.05;
}

template <>
float get_rel_tol<float>() {
  return 0.05;
}

template <>
float get_rel_tol<double>() {
  return 0.05;
}

template <>
float get_rel_tol<int8_t>() {
  return 0;
}

template <typename T>
void print_matrix(const std::vector<T> matrix, int n_cols,
                  int n_printable_rows = 10, int n_printable_cols = 10,
                  bool cast_as_long = false,
                  std::ostream &ostream = std::cout,
                  const char col_sep[] = "  ", const char elide_sym[] = " ... ",
                  int w = -1) {
  assert(matrix.size() % n_cols == 0);

  auto maxima = std::minmax_element(matrix.begin(), matrix.end());
  //T max_val = std::max(*maxima.first, (T)std::abs(*maxima.second));
  size_t n_digits = 12; //log10(max_val);
  
  if (w == -1) {
    w = n_digits;
  }
  int n_rows = matrix.size() / n_cols;

  n_printable_rows = std::min(n_rows, n_printable_rows);
  n_printable_cols = std::min(n_cols, n_printable_cols);

  const bool elide_rows = false; // n_printable_rows < n_rows;
  const bool elide_cols = false; // n_printable_cols < n_cols;

  if (elide_rows || elide_cols) {
    w = std::max((int)w, (int)strlen(elide_sym));
  }

  w += 3; // for decimal point and two decimal digits
  ostream << std::fixed << std::setprecision(20);

#define print_row(what)                                                        \
  for (int col = 0; col < n_printable_cols; col++) {                 \
    ostream << std::right << std::setw(w) << what;                 \
    ostream << std::setw(0) << col_sep;                               \
  }

  // if (elide_cols) {                                                            \
  //   ostream << std::setw(0) << elide_sym;                                      \
  // }                                                                            \
  // for (int i = 0; i < n_printable_cols / 2; i++) {                             \
  //   int col = elide_cols? (n_cols - n_printable_cols / 2 + i) : (n_printable_cols + 1) / 2; \
  //   ostream << std::right << std::setw(w) << (what);                           \
  //   ostream << std::setw(0) << col_sep;                                        \
  // }

  for (int row = 0; row < (n_printable_rows + 1) / 2; row++) {
    if constexpr (sizeof(T)==8) {
      if (cast_as_long) {
        print_row(std::bit_cast<std::int64_t>(matrix[row * n_cols + col]));
      }
      else 
      {
        print_row(matrix[row * n_cols + col]);
      }
    }
    else {
      print_row(matrix[row * n_cols + col]);
    }
    ostream << std::endl;
  }

  if (elide_rows) {
    print_row(elide_sym);
    ostream << std::endl;
  }
  for (int i = 0; i < n_printable_rows / 2; i++) {
    int row = elide_rows? (n_rows - n_printable_rows / 2 + i) : ((n_printable_rows + 1) / 2 + i);
    if constexpr (sizeof(T)==8) {
      if (cast_as_long) {
        print_row(std::bit_cast<std::int64_t>(matrix[row * n_cols + col]));
      }
      else 
      {
        print_row(matrix[row * n_cols + col]);
      }
    }
    else {
      print_row(matrix[row * n_cols + col]);
    }
    ostream << std::endl;
  }

#undef print_row
}

// int8_t aka char will not print as a number but as a character; specialize
// print_matrix<int8_t> to cast to int16_t first so everything prints as numbers
template <>
void print_matrix(const std::vector<int8_t> matrix, int n_cols,
                  int n_printable_rows, int n_printable_cols,
                  bool cast_as_long,
                  std::ostream &ostream, const char col_sep[],
                  const char elide_sym[], int w) {
  std::vector<int16_t> cast_matrix(matrix.size());
  for (uint i = 0; i < matrix.size(); i++) {
    cast_matrix[i] = (int16_t)matrix[i];
  }
  print_matrix(cast_matrix, n_cols, n_printable_rows, n_printable_cols, false, ostream,
               col_sep, elide_sym, w);
}

constexpr int max_printable_errors = 32;

template <typename Tout>
struct error {
  int row;
  int col;
  Tout expected;
  Tout actual;
};

template <typename Tout>
std::optional<struct error<Tout>>
verify_single(std::ostream &os, int row, int col, Tout expected, Tout actual,
              Tout abs_tol, Tout rel_tol) {
  bool match = expected == actual;
  if (abs_tol > 0 || rel_tol > 0) {
    // Allow for some tolerance for FP data types
    match = nearly_equal(expected, actual, rel_tol, abs_tol);
  }
  if (!match) {
    return (struct error<Tout>){row, col, expected, actual};
  }
  return std::nullopt;
}

template <typename Tout>
void print_error_summary(std::ostream &os, int n_errors,
                         std::vector<struct error<Tout>> &errors,
                         Tout max_rel_error, const std::string& label) {
  for (struct error<Tout> &err : errors) {
    os << "[" << std::setw(5) << err.row << ", " << std::setw(5) << err.col
       << "] " << std::setw(10) << std::setprecision(5) << std::fixed
       << (double)err.actual << " =!= " << std::setw(10) << std::setprecision(5)
       << std::fixed << (double)err.expected << std::endl;
  }
  if (n_errors > max_printable_errors) {
    os << "...and " << std::setw(0) << n_errors - max_printable_errors
       << " further errors." << std::endl;
  }
  if (n_errors >= 0) {
    os << label << " Maximum relative error: " << std::setw(20) << std::scientific << std::setprecision(10)
       << max_rel_error  << std::endl;
  }
}

void print_progress_bar(std::ostream &os, double progress, int len = 75) {
  os << "\r" << std::string((int)(progress * len), '|')
     << std::string(len - (int)(progress * len), ' ') << std::setw(4)
     << std::fixed << std::setprecision(0) << progress * 100 << "%"
     << "\r";
}

float median(std::vector<float> values) {
  if (values.empty())
        return 0.0f;

    std::sort(values.begin(), values.end());

    size_t n = values.size();

    if (n % 2 == 0)
        return (values[n / 2 - 1] + values[n / 2]) / 2.0f;
    else
        return values[n / 2];
}

int verify(int M, int N, int K, std::vector<double> A, std::vector<double> B,
           const std::vector<double>& C, const std::string& label,
           double abs_tol, double rel_tol,
           int n_bits_ref_precision = 106,
           int verbosity = 0, 
           int b_col_maj = 0, int c_col_maj = 0) {
  int n_errors = 0;
  std::vector<struct error<double>> errors;
  double max_rel_error = std::numeric_limits<double>::lowest();
  struct error<double> max_error;
  // use the specified number of precision bits for the reference emulated higher precision product matrix using MPFR
  // only calculate it once to persist between calls (NOTE: keeping it static assumes that A,B,M,N and K do not change when this method is invoked)
  static std::vector<double> CRef;

  if (CRef.empty()) {
    CRef.resize(M * N);
    matmul_MPFR(M, N, K, A, B, CRef, b_col_maj, c_col_maj, n_bits_ref_precision);
  }

  for (int row = 0; row < M; row++) {
    for (int col = 0; col < N; col++) {
      double actual = C[row * N + col];
      double ref = CRef[row * N + col];

      std::optional<struct error<double>> error = verify_single<double>(std::cout, row, col, ref, actual, abs_tol, rel_tol);
      
      if (error.has_value()) {
        if (n_errors < max_printable_errors) {
          errors.push_back(*error);
        }
        n_errors++;
      }

      double rel_error = std::abs(actual - ref) / std::abs(ref);

      if (rel_error > max_rel_error) {
          max_rel_error = rel_error;
      }
    }
  }

  if (n_errors)
    errors.push_back(max_error);
  print_error_summary(std::cout, n_errors, errors, max_rel_error, label);

  if (n_errors >= 0) {
    std::cout << std::endl << "Reference:" << std::endl;
    matmul_common::print_matrix(CRef, N);
    std::cout << std::endl << "Output:" << std::endl;
    matmul_common::print_matrix(C, N);
  }

  std::cout << std::endl << label << " Relative Error:" << max_rel_error << "  " << std::abs(C[0] - CRef[0]) / std::abs(CRef[0]) << std::endl;

  return n_errors;
}

int verify_stochastic(int M, int N, int K, std::vector<double> A,
                      std::vector<double> B, std::vector<double> C, const std::string& label, int n_samples,
                      int n_bits_ref_precision = 106,
                      int verbosity = 0, double abs_tol = 0.5,
                      double rel_tol = 0.05, int b_col_maj = 0,
                      int c_col_maj = 0) {
  std::mt19937 rng;
  auto rows = std::views::iota(0, M);
  auto cols = std::views::iota(0, N);
  auto sampled_rows = std::vector<int>(n_samples);
  auto sampled_cols = std::vector<int>(n_samples);

  std::ranges::sample(rows, sampled_rows.begin(), n_samples, rng);
  std::ranges::sample(cols, sampled_cols.begin(), n_samples, rng);

  int n_errors = 0;
  std::vector<struct error<double>> errors;
  double max_rel_error = std::numeric_limits<double>::lowest();
  double progress = 0;
  for (std::tuple<size_t, std::tuple<int &, int &>> cell :
       std::views::enumerate(std::views::zip(sampled_rows, sampled_cols))) {
    int i = std::get<0>(cell);
    int row = std::get<0>(std::get<1>(cell));
    int col = std::get<1>(std::get<1>(cell));
    if (verbosity >= 1 &&
        (int)(progress * 100) < (int)((double)i / n_samples * 100)) {
      // Only print progress bar if percentage changed
      progress = (double)i / n_samples;
      print_progress_bar(std::cerr, progress);
    }
    double ref = mul_acc_MPFR(M, N, K, row, col, A, B, b_col_maj, n_bits_ref_precision);
    double actual;
    if (!c_col_maj) {
      actual = C[row * N + col];
    } else {
      actual = C[row + col * M];
    }
    std::optional<struct error<double>> error =
        verify_single<double>(std::cout, row, col, ref, actual, abs_tol, rel_tol);
    if (error.has_value()) {
        if (n_errors < max_printable_errors) {
          errors.push_back(*error);
        }
        n_errors++;
      }

      double rel_error = std::abs(actual - ref) / std::abs(ref);

      if (rel_error > max_rel_error) {
          max_rel_error = rel_error;
      }
  }
  std::cout << std::endl;

  print_error_summary(std::cout, n_errors, errors, max_rel_error, label);
  return n_errors;
}

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

  std::cout << "Num of splits in first = " << num_grp_splits[0] << "," << num_grp_splits[1] << "\n";

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


// --------------------------------------------------------------------------
// Tracing
// --------------------------------------------------------------------------
void write_out_trace(char *traceOutPtr, size_t trace_size, std::string path) {
  std::ofstream fout(path);
  uint32_t *traceOut = (uint32_t *)traceOutPtr;
  for (int i = 0; i < trace_size / sizeof(traceOut[0]); i++) {
    fout << std::setfill('0') << std::setw(8) << std::hex << (int)traceOut[i];
    fout << std::endl;
  }
}

} // namespace matmul_common

#endif
