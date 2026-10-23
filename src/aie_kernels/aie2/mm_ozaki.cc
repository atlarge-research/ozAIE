//===- mm_ozaki.cc ----------------XDNA compatibility---------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//# (c) Copyright 2023 Advanced Micro Devices, Inc.
//===----------------------------------------------------------------------===//
// Modified in 2026 
// Original source: https://github.com/Xilinx/mlir-aie/tree/v1.3.2/aie_kernels/aie2/mm.cc

#define NOCPP

#include <stdio.h>
#include <stdlib.h>

#define REL_WRITE 0
#define REL_READ 1

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <aie_api/utils.hpp>

#include "mm.cc"

extern "C" {

#ifdef i4_i32_ONLY
#define combos(X) X(int4, i4, int32, i32, 4, 16, 8)
#endif

}

/*
XDNA1 does not support aie::exp2 vector operation. This method is a workaround
*/
template <unsigned v_elems>
inline aie::vector<uint32, v_elems> exp2(const aie::vector<uint32, v_elems>& v_pow) {
      aie::vector<uint32, v_elems> result;
      result = to_fixed<uint32>(aie::upshift(aie::add(v_pow, 127u), 23).template cast_to<float>());
    // workaround as experienced on XDNA1. Value of 2^31 saturates at INT32 max even when converted to uint32 from float 
      return aie::select(result, (1u << 31), aie::ge(v_pow, 31u));
}

template <int M, int N>
void zero_vectorized(int64_t *__restrict c) {
  constexpr int r = 256 / (sizeof(int64_t) * 8); // one 256 bit store unit
  static_assert((M * N) % r == 0);
  const aie::vector<int32, r*2> zeros = aie::zeros<int32, r*2>();
  const int64_t *__restrict c_end = c + M * N;

  for (; c < c_end; c += r) {
    aie::store_v((int32_t *) c, zeros);
  }
  
}

#include "../generic/ozaki_base.cc"

/* This mul kernel is based on thr 4x2 mmul kernel from AMD MLIR-AIE project with
  added support for int8 x int4 multiplication mode on AIE-ML/XDNA1
 */
template <typename T_in, typename T_out, unsigned rowA, unsigned colA,
          unsigned colB, unsigned r, unsigned s, unsigned t,
          bool b_row_maj = true, bool c_row_maj = true>
static inline void matmul_vectorized_4x2_mmul_int4_support(const T_in *__restrict pA,
                                              const T_in *__restrict pB,
                                              T_out *__restrict pC) {
  using MMUL_A_T = std::conditional_t<
    std::is_same_v<T_in, int4_t>,
    int8_t,
    T_in
  >;

  using MMUL = aie::mmul<
      r, s, t,
      MMUL_A_T,
      T_in,
      accauto
  >;

  auto outer_body = [&](unsigned z) [[gnu::always_inline]] {
    T_out *__restrict pC1;
    T_out *__restrict pC2;
    T_out *__restrict pC3;
    T_out *__restrict pC4;

    if constexpr (c_row_maj) {
      pC1 = pC + (z * colB + 0) * MMUL::size_C;
      pC2 = pC + ((z + 1) * colB + 0) * MMUL::size_C;
      pC3 = pC + ((z + 2) * colB + 0) * MMUL::size_C;
      pC4 = pC + ((z + 3) * colB + 0) * MMUL::size_C;
    }

    for (unsigned j = 0; j < colB; j += 2)
#ifdef OPT_PERF_ENABLED
      AIE_LOOP_FLATTEN
#endif
      {
        if constexpr (!c_row_maj) {
          pC1 = pC + j * rowA * MMUL::size_C + z * MMUL::size_C;
          pC2 = pC + (j + 1) * rowA * MMUL::size_C + z * MMUL::size_C;
        }

        const T_in *__restrict pA1 = pA + ( (z * colA + 0) * MMUL::size_A / (std::is_same_v<T_in, int4_t>? 2 : 1));
        const T_in *__restrict pA2 = pA + ( ((z + 1) * colA + 0) * MMUL::size_A / (std::is_same_v<T_in, int4_t>? 2 : 1));
        const T_in *__restrict pA3 = pA + ( ((z + 2) * colA + 0) * MMUL::size_A / (std::is_same_v<T_in, int4_t>? 2 : 1));
        const T_in *__restrict pA4 = pA + ( ((z + 3) * colA + 0) * MMUL::size_A / (std::is_same_v<T_in, int4_t>? 2 : 1));

        const T_in *__restrict pB1;
        const T_in *__restrict pB2;
        if constexpr (b_row_maj) {
          pB1 = pB + ((j)*MMUL::size_B / (std::is_same_v<T_in, int4_t>? 2 : 1));
          pB2 = pB + ((j + 1)*MMUL::size_B / (std::is_same_v<T_in, int4_t>? 2 : 1));
        } else {
          pB1 = pB + ((j * colA) * MMUL::size_B / (std::is_same_v<T_in, int4_t>? 2 : 1));
          pB2 = pB + (( (j + 1) * colA) * MMUL::size_B / (std::is_same_v<T_in, int4_t>? 2 : 1));
        }

        aie::vector<MMUL_A_T, MMUL::size_A> A01;
        aie::vector<MMUL_A_T, MMUL::size_A> A11;
        aie::vector<MMUL_A_T, MMUL::size_A> A21;
        aie::vector<MMUL_A_T, MMUL::size_A> A31;
        aie::vector<T_in, MMUL::size_B> B0;
        aie::vector<T_in, MMUL::size_B> B1;

        aie::vector<T_out, MMUL::size_C> acc_C00;
        aie::vector<T_out, MMUL::size_C> acc_C01;
        aie::vector<T_out, MMUL::size_C> acc_C10;
        aie::vector<T_out, MMUL::size_C> acc_C11;
        aie::vector<T_out, MMUL::size_C> acc_C20;
        aie::vector<T_out, MMUL::size_C> acc_C21;
        aie::vector<T_out, MMUL::size_C> acc_C30;
        aie::vector<T_out, MMUL::size_C> acc_C31;

        if constexpr (c_row_maj) {
          acc_C00 = aie::load_v<MMUL::size_C>(pC1);
          acc_C01 = aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C);
          acc_C10 = aie::load_v<MMUL::size_C>(pC2);
          acc_C11 = aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C);
          acc_C20 = aie::load_v<MMUL::size_C>(pC3);
          acc_C21 = aie::load_v<MMUL::size_C>(pC3 + MMUL::size_C);
          acc_C30 = aie::load_v<MMUL::size_C>(pC4);
          acc_C31 = aie::load_v<MMUL::size_C>(pC4 + MMUL::size_C);
        } else {
          acc_C00 = aie::transpose(aie::load_v<MMUL::size_C>(pC1), t, r);
          acc_C01 = aie::transpose(aie::load_v<MMUL::size_C>(pC2), t, r);
          acc_C10 = aie::transpose(
              aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C), t, r);
          acc_C11 = aie::transpose(
              aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C), t, r);
          acc_C20 = aie::transpose(
              aie::load_v<MMUL::size_C>(pC1 + 2 * MMUL::size_C), t, r);
          acc_C21 = aie::transpose(
              aie::load_v<MMUL::size_C>(pC2 + 2 * MMUL::size_C), t, r);
          acc_C30 = aie::transpose(
              aie::load_v<MMUL::size_C>(pC1 + 3 * MMUL::size_C), t, r);
          acc_C31 = aie::transpose(
              aie::load_v<MMUL::size_C>(pC2 + 3 * MMUL::size_C), t, r);
        }

        MMUL C00(acc_C00);
        MMUL C01(acc_C01);
        MMUL C10(acc_C10);
        MMUL C11(acc_C11);
        MMUL C20(acc_C20);
        MMUL C21(acc_C21);
        MMUL C30(acc_C30);
        MMUL C31(acc_C31);

        for (unsigned i = 0; i < colA; i += 1)
#ifdef OPT_PERF_ENABLED
          AIE_LOOP_FLATTEN
#endif
          {
            if constexpr (std::is_same_v<T_in, int4_t>) {
                // unpack int4 to int8 for int8 x int4 multiplication mode
                A01 = aie::load_v<MMUL::size_A>(pA1).unpack();
                pA1 += MMUL::size_A / 2;
                A11 = aie::load_v<MMUL::size_A>(pA2).unpack();;
                pA2 += MMUL::size_A / 2;
                A21 = aie::load_v<MMUL::size_A>(pA3).unpack();;
                pA3 += MMUL::size_A / 2;
                A31 = aie::load_v<MMUL::size_A>(pA4).unpack();;
                pA4 += MMUL::size_A / 2;
            }
            else 
            {
                A01 = aie::load_v<MMUL::size_A>(pA1);
                pA1 += MMUL::size_A;
                A11 = aie::load_v<MMUL::size_A>(pA2);
                pA2 += MMUL::size_A;
                A21 = aie::load_v<MMUL::size_A>(pA3);
                pA3 += MMUL::size_A;
                A31 = aie::load_v<MMUL::size_A>(pA4);
                pA4 += MMUL::size_A;
            }

            if constexpr (b_row_maj) {
              B0 = aie::load_v<MMUL::size_B>(pB1);
            } else {
              B0 = aie::transpose(aie::load_v<MMUL::size_B>(pB1), t, s);             
            }

            C00.mac(A01, B0);
            
            C01.mac(A01, B1);
            
            C10.mac(A11, B0);
            
            C11.mac(A11, B1);
            
            C20.mac(A21, B0);
            
            C21.mac(A21, B1);
            
            C30.mac(A31, B0);
            
            C31.mac(A31, B1);
            
          }

        if constexpr (c_row_maj) {
          aie::store_v(pC1, C00.template to_vector<T_out>());
          pC1 += MMUL::size_C;
          aie::store_v(pC1, C01.template to_vector<T_out>());
          pC1 += MMUL::size_C;
          aie::store_v(pC2, C10.template to_vector<T_out>());
          pC2 += MMUL::size_C;
          aie::store_v(pC2, C11.template to_vector<T_out>());
          pC2 += MMUL::size_C;
          aie::store_v(pC3, C20.template to_vector<T_out>());
          pC3 += MMUL::size_C;
          aie::store_v(pC3, C21.template to_vector<T_out>());
          pC3 += MMUL::size_C;
          aie::store_v(pC4, C30.template to_vector<T_out>());
          pC4 += MMUL::size_C;
          aie::store_v(pC4, C31.template to_vector<T_out>());
          pC4 += MMUL::size_C;
        } else {
          aie::store_v(pC1,
                       aie::transpose(C00.template to_vector<T_out>(), r, t));
          pC1 += MMUL::size_C;
          aie::store_v(pC2,
                       aie::transpose(C01.template to_vector<T_out>(), r, t));
          pC2 += MMUL::size_C;
          aie::store_v(pC1,
                       aie::transpose(C10.template to_vector<T_out>(), r, t));
          pC1 += MMUL::size_C;
          aie::store_v(pC2,
                       aie::transpose(C11.template to_vector<T_out>(), r, t));
          pC2 += MMUL::size_C;
          aie::store_v(pC1,
                       aie::transpose(C20.template to_vector<T_out>(), r, t));
          pC1 += MMUL::size_C;
          aie::store_v(pC2,
                       aie::transpose(C21.template to_vector<T_out>(), r, t));
          pC2 += MMUL::size_C;
          aie::store_v(pC1,
                       aie::transpose(C30.template to_vector<T_out>(), r, t));
          pC1 += MMUL::size_C;
          aie::store_v(pC2,
                       aie::transpose(C31.template to_vector<T_out>(), r, t));
          pC2 += MMUL::size_C;
        }
      }
  };

  constexpr unsigned outer_iters = rowA / 4;
  if constexpr (outer_iters >= 4) {
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(4)
    for (unsigned z = 0; z < rowA; z += 4)
      outer_body(z);
  } else if constexpr (outer_iters >= 2) {
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(2)
    for (unsigned z = 0; z < rowA; z += 4)
      outer_body(z);
  } else {
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (unsigned z = 0; z < rowA; z += 4)
      outer_body(z);
  }

}

template <unsigned m, unsigned k, unsigned n>
static inline void matmul_vectorized_4x16x8_i4_i32(const int4_t *__restrict p_splits_A,
                                                  const int4_t *__restrict p_splits_B,
                                                  int32_t *__restrict pC) {
  constexpr int r = 4;
  constexpr int s = 16;
  constexpr int t = 8;

  static_assert(m % (4 * r) == 0);
  static_assert(k % s == 0);
  static_assert(n % (2 * t) == 0);

  return matmul_vectorized_4x2_mmul_int4_support<int4, int32, (m / r), (k / s), (n / t), r,
                                    s, t, is_b_row_maj>(p_splits_A, p_splits_B, pC);
}

template <typename T_in, typename T_out, unsigned m, unsigned k, unsigned n, 
    unsigned bits_per_slice, unsigned num_splits>
static inline void ozaki_matmul(const T_in * p_splits_A, \
  const T_in * p_splits_B,\
  T_out * p_bins_C, int tile, int mul_iter, int64_t *__restrict log_buff) {
  
  // if ith split of a is multiplied by jth split of b where total number of splits are s, 
  // then the following pattern represents these s(s+1)/2 inner products (i,j): 
  // (The remaining possible products omitted do not contribute to the final accuracy)
  // 
  // (s-1),0
  // (s-2),0   (s-2),1
  // . . . . . . . . 
  // . . . . . . . . 
  // (s-n),0   (s-n),1    (s-n),(n-1)

  constexpr int size_tile_a = m*k;
  constexpr int size_tile_b = k*n;
  constexpr int size_tile_c = m*n;

  const T_in * a = p_splits_A;
  const T_in * b = p_splits_B; 
  T_out * c = p_bins_C;

  /* event0(); */ /* muls_cycles_measure_start*/

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_RANGE(NUM_SPLITS,NUM_SPLITS)
  for (int a_split_no=0; a_split_no<num_splits; a_split_no++) {
    b = p_splits_B;
    c = p_bins_C + (a_split_no * size_tile_c);

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(1, NUM_SPLITS)
    for (int b_split_no=0; b_split_no<num_splits-a_split_no; b_split_no++) {
      // if int4 splits are passed in, use the int4 mul mode
      if constexpr (std::is_same_v<T_in, int4_t>) {
        matmul_vectorized_4x16x8_i4_i32<m ,k ,n>(a, b, c);
      }
      // else use the int8 mul mode
      else 
      {
        matmul_vectorized_4x8x8_i8_i32<m, k, n>(a, b, c);
      }
      // for int4 splits, we have two int4s packed in one byte , so we move the pointer by half tile size. Also for A splits below
      b += ( size_tile_b / (std::is_same_v<T_in, int4_t>? 2 : 1) );
      c += size_tile_c;  
    }

    a += ( size_tile_a / (std::is_same_v<T_in, int4_t>? 2 : 1) );
  }

  /* event1(); */ /* muls_cycles_measure_start*/
}

extern "C" {

#define ozaki_matmul_vectorized_c_func(ctype_in, mlir_type_in, ctype_out,            \
                                 mlir_type_out, r, s, t)                       \
void ozaki_matmul_##mlir_type_in##_##mlir_type_out##_double_s(ctype_in *a_splits_in, ctype_in *b_splits_in, \
                                              ctype_out *c_bins_buff, int tile, int mul_iter, int64_t* log_buff) {             \
  ozaki_matmul<ctype_in, ctype_out, DIM_M, DIM_K, DIM_N, BITS_PER_SLICE, NUM_SPLITS>(a_splits_in, b_splits_in, c_bins_buff, tile, mul_iter, log_buff);                               \
} 

combos(ozaki_matmul_vectorized_c_func) 

} // extern "C"