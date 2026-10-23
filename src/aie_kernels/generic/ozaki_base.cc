//===- ozaki_base.cc ----------------------------------------------000---*- C++ -*-===//
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

#ifdef DEBUG
constexpr bool kDebug = true;
#else
constexpr bool kDebug = false;
#endif

#ifndef NUM_SPLITS
#define NUM_SPLITS 8
#endif

#ifndef BITS_PER_SLICE
#define BITS_PER_SLICE 7
#endif

constexpr int LOG2K = __builtin_ctz(DIM_K_GLOBAL);

template <typename T, int M, int N>
void zero_vectorized(T *__restrict c, int n_times) {
  for (int i=0; i<n_times; i++)
  {
    zero_vectorized<T,M,N>(c);
    c += M*N;
  }
}

template <int M, int N>
void zero_vectorized(int64_t *__restrict c, int n_times) {
  for (int i=0; i<n_times; i++)
  {
    zero_vectorized<M, N>(c);
    c += M*N;
  }
}

template <unsigned m = DIM_M, typename T_vec, unsigned nElems>
void log(int tag, const aie::vector<T_vec, nElems> &v, int64_t * buff, bool tag_lines = true, bool reset_flag = false, bool testing = false) {
  constexpr int unpack_factor =  std::is_same_v<T_vec, int4_t>? 8 : (sizeof(int32_t) / sizeof(T_vec));
  constexpr int log_width = 32 * (sizeof(int64_t) / sizeof(int32_t));
  constexpr int log_display_width = m;  // set so that easily seen at required zoom level in terminal
  constexpr int n_lines = (nElems + log_display_width - 1) / log_display_width;
  constexpr int n_elems_per_line = nElems / n_lines;
  constexpr int max_lines_per_log = 200; 

  int32_t *__restrict pBuff = (int32_t *) buff;
  aie::vector<int32, 8> v_header;

  if constexpr(kDebug) {

      std::pair<aie::vector<int32,n_elems_per_line>,aie::vector<int32,n_elems_per_line>> iv_pair;

      if (buff==0) {
        return;
      }

      v_header = aie::load_v<8>(pBuff);
      int32_t start_line = v_header.get(0);
      v_header = aie::zeros<int32, 8>();

      if (start_line==0 || start_line>max_lines_per_log || reset_flag) {
        start_line=1;
      }

      if (reset_flag) {
        zero_vectorized<DIM_M, DIM_N>(buff,2);
      }

      int32_t * line_ptr;

      aie::vector<int32, nElems> v_full;
      aie::vector<int32, n_elems_per_line> v_line;

      if constexpr (unpack_factor==1) {
        if constexpr (std::is_same_v<T_vec, uint32>) {
          v_full = v.template cast_to<int32>();
        }
        else 
        {
          v_full = v;
        }
      }

      if constexpr (unpack_factor==2) {
        v_full = v.template cast_to<int16>().unpack();
      }

      if constexpr (unpack_factor==4) {
        v_full = v.template cast_to<int8>().unpack().unpack();    
      }

      if constexpr (unpack_factor==8) {
        v_full = v.template cast_to<int4>().unpack().unpack().unpack();    
      }

      line_ptr = pBuff + (start_line * log_width);
      if (tag_lines) {
          v_header.set(tag, 0);
          aie::store_v(line_ptr, v_header);
          line_ptr += log_width;
      }

      int elem_no = 0;

      for (int l=0; l < n_lines; l++) {
        iv_pair = aie::interleave_zip(
          v_full.template extract<n_elems_per_line>(l), 
          aie::select(0, -1, aie::lt(v_full.template extract<n_elems_per_line>(l), 0)), 
          1);

        aie::store_v(line_ptr, iv_pair.first);
        aie::store_v(line_ptr + n_elems_per_line, iv_pair.second);

        line_ptr += log_width;
      }

      v_header.set((line_ptr - pBuff) / log_width, 0);
      //v_header.set(start_line + (n_lines + 1), 0);
      aie::store_v(pBuff, v_header);
  }
}

template <typename T_out, unsigned m, unsigned n, unsigned inner_m, unsigned inner_n, unsigned bits_per_slice, unsigned num_splits>
static inline void ozaki_vectorized_sum_upshift(const T_out * const __restrict pC_bins, \
  int64_t * const __restrict pC, int16_t *__restrict row_max_exp, int16_t *__restrict col_max_exp, int64_t * __restrict log_buff) {
    constexpr int size_c_tile = m*n;
    constexpr int max_vector_bit_width = 1024;
    constexpr int vec_factor = max_vector_bit_width / sizeof(T_out);
    constexpr int doubles_per_loop = 32;
    
    constexpr int n_bins_in_first_group = [] {
        int min_bins_alignable_in_first_group = 31 / bits_per_slice + 1;
        // at least this number of bins should be summed in first INT64 to leave space of 12 bits for embedding
        // scaleback in the second (higher) INT64
        int numerator = bits_per_slice * (num_splits + 1) - 51 + LOG2K;
        int min_possible_bins_in_first_group = numerator < 0? 1 : (numerator / bits_per_slice + 1);
        return std::min(static_cast<signed>(num_splits), std::max(min_possible_bins_in_first_group, min_bins_alignable_in_first_group));
    }();

    constexpr int num_sum_groups = (num_splits == n_bins_in_first_group)? 1 : 2;
    constexpr int n_splits_in_group[] = { n_bins_in_first_group, static_cast<signed>(num_splits) - n_bins_in_first_group }; 

    constexpr int tile_size = m*n;
    constexpr int inner_tile_size = inner_m * inner_n;
    constexpr int n_loads_per_inner_tile = inner_tile_size / doubles_per_loop;
    constexpr int max_i16_per_vector = max_vector_bit_width / 16; 
    constexpr int row_exp_max_load_batch_size = (m * inner_n < max_i16_per_vector? m : max_i16_per_vector / inner_n);
    constexpr int col_exp_max_load_batch_size = (n * inner_m < max_i16_per_vector? n : max_i16_per_vector / inner_m);
    constexpr int row_exp_max_replicate_factor = inner_n;
    constexpr int col_exp_max_replicate_factor = inner_m;
    constexpr int row_exp_max_pattern_size = doubles_per_loop <= inner_tile_size ? doubles_per_loop : inner_tile_size;
    constexpr int col_exp_max_pattern_size = doubles_per_loop <= inner_tile_size ? inner_tile_size : doubles_per_loop;
    constexpr int n_row_inner_batches = (row_exp_max_load_batch_size * row_exp_max_replicate_factor) / row_exp_max_pattern_size;
    constexpr int n_col_inner_batches = (col_exp_max_load_batch_size * col_exp_max_replicate_factor) / col_exp_max_pattern_size;

    constexpr int n_inner_tiles_per_row_exp_max_load = (row_exp_max_load_batch_size * row_exp_max_replicate_factor) / inner_tile_size;
    constexpr int n_inner_tiles_per_col_exp_max_load = (col_exp_max_load_batch_size * col_exp_max_replicate_factor) / inner_tile_size;

    constexpr int col_step_size = col_exp_max_pattern_size;
    // if there are inner steps within a tile in the inner most loop, then dont need to move ahead by 
    // the full row of inner tiles. Instead, adding one more inner step means that the current tile is already
    // out
    constexpr int row_step_size = doubles_per_loop < inner_tile_size ? 
      (doubles_per_loop + (n / inner_n - 1) * inner_tile_size)  : ((n / inner_n) * inner_tile_size) ;

    aie::mask<32> mask_hi_32 = aie::mask<32>::from_uint32(0xAAAAAAAAu);

    aie::mask<32> mask_neg;
    aie::vector<int32, 32> v_strip;
    aie::vector<uint32, 32> v_strip_ui32;
    aie::vector<uint32, 32> v_exp_embed;
    std::pair<aie::vector<uint32,32>,aie::vector<uint32,32>> v_exp_iv;
    std::pair<aie::vector<int16,32>,aie::vector<int16,32>> v_exp_batch_iv;
    std::pair<aie::vector<int32,32>,aie::vector<int32,32>> v_strip_iv1;
    std::pair<aie::vector<int32,32>,aie::vector<int32,32>> v_strip_iv2;
    aie::accum<acc64,16> acc_c[4];
    aie::accum<acc64,16> acc;
       
    int32_t *pC1 = (int32_t*) pC;
        // start from the last split reverse as the smaller int64 component is filled first
    const T_out *__restrict pCB1 = pC_bins + (num_splits - 1) * size_c_tile;

    std::pair<
          aie::vector<int16_t, col_exp_max_load_batch_size*col_exp_max_replicate_factor/2>,
          aie::vector<int16_t, col_exp_max_load_batch_size*col_exp_max_replicate_factor/2>> v_col_max_exp_strip_iv;

    /* event0(); */ /*sum_cycles_measure_start*/

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(2)
    // first load the column-wise max exponents batch
    for (int col_exp_max_batch=0; col_exp_max_batch < n / col_exp_max_load_batch_size; col_exp_max_batch++) {
        // load from max and grow_replicate inner_m times and then interleave so that the layout corresponds to actual c_tiles 
        aie::vector<int16_t, col_exp_max_load_batch_size*col_exp_max_replicate_factor> v_col_exp_max_batch = \
        aie::load_v<col_exp_max_load_batch_size>(col_max_exp + col_exp_max_batch*col_exp_max_load_batch_size).template grow_replicate<col_exp_max_load_batch_size*col_exp_max_replicate_factor>();

        if constexpr (col_exp_max_load_batch_size > inner_n)
        {
            int iv_step = col_exp_max_load_batch_size;

            // do an interleave loop here 
            aie::vector<int16_t, col_exp_max_load_batch_size*col_exp_max_replicate_factor/2> v_col_exp_batch_half1 = v_col_exp_max_batch.template extract<col_exp_max_load_batch_size*col_exp_max_replicate_factor/2>(0);
            aie::vector<int16_t, col_exp_max_load_batch_size*col_exp_max_replicate_factor/2> v_col_exp_batch_half2 = v_col_exp_max_batch.template extract<col_exp_max_load_batch_size*col_exp_max_replicate_factor/2>(1);

            while (iv_step > inner_n) {
              iv_step = col_exp_max_load_batch_size / 2;
              v_exp_batch_iv = aie::interleave_unzip(v_col_exp_batch_half1, v_col_exp_batch_half2, iv_step);
              v_col_exp_batch_half1 = v_exp_batch_iv.first;
              v_col_exp_batch_half2 = v_exp_batch_iv.second;
            }

            v_exp_batch_iv = aie::interleave_zip(v_col_exp_batch_half1, v_col_exp_batch_half2, inner_m*inner_n);
            v_col_exp_max_batch = aie::concat(v_exp_batch_iv.first, v_exp_batch_iv.second);
        }

        // for each column-wise max exponents batch, do each row-wise max exponents batch
        for (int row_exp_max_batch=0; row_exp_max_batch < m / row_exp_max_load_batch_size; row_exp_max_batch++) {
            // load from max and grow_replicate inner_n times and then transpose so that the layout corresponds to actual c_tiles 
            aie::vector<int16_t, row_exp_max_load_batch_size*row_exp_max_replicate_factor> v_row_exp_max_batch = \
            aie::load_v<row_exp_max_load_batch_size>(row_max_exp + row_exp_max_batch*row_exp_max_load_batch_size).template grow_replicate<row_exp_max_load_batch_size*row_exp_max_replicate_factor>();
        
            v_row_exp_max_batch = aie::transpose(v_row_exp_max_batch, inner_n , row_exp_max_load_batch_size); 

            for (int col_inner_batch_idx=0; col_inner_batch_idx < n_col_inner_batches; col_inner_batch_idx++) {
                aie::vector<int16_t, doubles_per_loop> v_col_max_exp;
                aie::vector<int16_t, col_exp_max_pattern_size> v_col_max_batch = v_col_exp_max_batch.template extract<col_exp_max_pattern_size>(col_inner_batch_idx);

                v_col_max_exp = v_col_max_batch.template extract<doubles_per_loop>(0);

                // offset from base of the start row of the current outer row-wise max exp batach 
                // and the start column of the current outer col-wise max exp batch
                int offset = (row_exp_max_batch * n_inner_tiles_per_row_exp_max_load) * (n / inner_n) * inner_tile_size;
                offset += (col_exp_max_batch * n_inner_tiles_per_col_exp_max_load) * inner_tile_size + (col_inner_batch_idx * col_step_size);
                // reset bins location to the last split at the offset
                pCB1 = pC_bins + (num_splits - 1) * size_c_tile + offset;

                // same offset for C output buffer
                pC1 =  (int32_t*) (pC + offset * sizeof(int64_t) / sizeof(int32_t));

                for (int row_inner_batch_idx=0; row_inner_batch_idx < n_row_inner_batches; row_inner_batch_idx++) {
                    int32_t *pC_write = pC1; 

                    aie::vector<int16_t, doubles_per_loop> v_row_max_exp;
                    aie::vector<int16_t, row_exp_max_pattern_size> v_row_max_batch = v_row_exp_max_batch.template extract<row_exp_max_pattern_size>(row_inner_batch_idx);

                    if constexpr (row_exp_max_pattern_size < doubles_per_loop) {
                      v_row_max_exp = v_row_max_batch.template grow<doubles_per_loop>();
                    } else {
                      v_row_max_exp = v_row_max_batch.template extract<doubles_per_loop>(0);
                    }         

                    acc_c[0] = aie::zeros<acc64,16>();
                    acc_c[1] = aie::zeros<acc64,16>();
                    acc_c[2] = aie::zeros<acc64,16>();
                    acc_c[3] = aie::zeros<acc64,16>();

                    // in case of 2 groups, the lower half bins (group 0) are accumulated in the first 2 accumulators and the higher
                    // half bins (group 1) are accumulated in the remaining 2 accumulators. 
                    for (int sum_group=0; sum_group<num_sum_groups; sum_group++) {

                      for (int i=0; i<n_splits_in_group[sum_group]; i++) {
                        v_strip = aie::load_v<32>(pCB1);
                        pCB1 -= size_c_tile;

                        acc.from_vector(v_strip.extract<16>(0), bits_per_slice * i);   // first 16 int32 values from bin
                        acc_c[sum_group*2] = aie::add(acc_c[sum_group*2], acc);
                        acc.from_vector(v_strip.extract<16>(1), bits_per_slice * i);   // last 16 int32 values from bin
                        acc_c[sum_group*2+1] = aie::add(acc_c[sum_group*2+1], acc);
                      }
                    }

                    // get the exponent scale-backs 
                    aie::vector<uint32, doubles_per_loop> scale_back_exp = aie::add(v_row_max_exp, v_col_max_exp).unpack().cast_to<uint32>();

                    scale_back_exp = aie::upshift(scale_back_exp, 32-13); 
                    v_exp_iv = aie::interleave_zip(aie::zeros<uint32, 32>(), scale_back_exp, 1);

                    for (int store_iter=0; store_iter<2; store_iter++) {
                        // take the higher i64 component for embedding scale back inside
                        v_strip_ui32 = acc_c[store_iter+2].cast_to<acc32>().template to_vector<uint32>();
                        
                        // from the higher i64, take only the upper int32 to embedd the scale back
                        // reason for comment : the final merge will anyways revert the lower i32 back once 
                        // scale back is embedded into the higher i32
                        // v_exp_embed = aie::select(aie::zeros<uint32, 32>(), v_strip_ui32, mask_hi_32);
                        // make zeroed-out space of upper 12 bits (excluding sign bit) for embedding
                        v_exp_embed = aie::add(
                          aie::logical_downshift(aie::upshift(v_strip_ui32, 13), 13), // zero out space for embed
                          aie::upshift(aie::logical_downshift(v_strip_ui32, 31), 31) // retain sign bit
                        );

                        // embed scale back 
                        v_exp_embed = aie::add(v_exp_embed, store_iter==0? v_exp_iv.first : v_exp_iv.second);
                        

                        // merge the lower i32 part of higher i64 back
                        v_exp_embed = aie::select(v_strip_ui32, v_exp_embed, mask_hi_32);

                        // corresponding lower and higher i64 values for the same C element grouped as pairs
                        v_strip_iv1 = aie::interleave_zip(
                          acc_c[store_iter].cast_to<acc32>().template to_vector<int32>(),
                          v_exp_embed.cast_to<int32>(), 2);

                        aie::store_v(pC_write, v_strip_iv1.first);
                        pC_write += 32;
                        aie::store_v(pC_write, v_strip_iv1.second);
                        pC_write += 32;
                    }

                    int inner_step = ( (row_inner_batch_idx+1) % ((inner_tile_size + doubles_per_loop - 1)/ doubles_per_loop) ) * doubles_per_loop;
                    pCB1 += (num_splits * size_c_tile) + inner_step + (inner_step==0? row_step_size : 0);
                    pC1 += (inner_step + (inner_step==0? row_step_size : 0)) * (sizeof(int64_t) / sizeof(int32_t) * 2);  // 2 x int64 (eq. 4 x int32s) per each int32 (bin) summation
                }
          }
        
      }
    }

    /* event1(); */ /*sum_cycles_measure_stop*/
}

template <unsigned m, unsigned k, unsigned inner_m, unsigned inner_k>
static void ozaki_get_max_exponent_inner_tiled_wise_half_half(const int32_t *__restrict pA, int16_t *__restrict max_exp_A, int64_t* log_buff = 0) {
  constexpr int n_inner_tile_rows = m / inner_m;
  constexpr int n_inner_tile_cols = k / inner_k;

  // The lower and higher words of FP64 are seperated into two regions of the tile space. 
  constexpr int tile_size = m * k;   // (m * k) is actually half the total INT32s in the buffer 
  constexpr int inner_tile_size = inner_m * inner_k;

  constexpr int vec_factor = [] {
    int result = n_inner_tile_cols==1?  inner_tile_size : (inner_tile_size * n_inner_tile_cols / 2);
    int n_loads =  n_inner_tile_cols==1? 1 : 2;
    // restrict load limit to reduce register pressure. There are total 24 256-bit registers on XDNA
    int max_int32s_per_load = ((24 * 256 / n_loads) / 32);
    if (result >= max_int32s_per_load) {
      result = result / 2;
    }
  
    return result;
  }();

  constexpr int n_intra_inner_tile_steps = inner_tile_size / vec_factor;

  // step size for next tile load within inner batches
  constexpr int inner_step_size_to_next_inner_tile = inner_tile_size;
  
  // step size for the base pointer on which inner load pointer(s) are based
  constexpr int outer_step_size = (n_intra_inner_tile_steps > 1? vec_factor + (n_inner_tile_cols - 1) * inner_tile_size : n_inner_tile_cols * inner_tile_size);
  
  // number of elements in loaded vector over which an element-wise max consitutes the same row or column
  constexpr int max_group_dim = inner_k;

  constexpr int n_inner_batches = n_inner_tile_cols==1? 1 : (n_inner_tile_cols / 2);
  constexpr int n_outer_batches = n_inner_tile_rows * n_intra_inner_tile_steps;

  constexpr int n_exponent_bits = 11;

  aie::vector<int32,vec_factor> v_hi_tile1;
  aie::vector<int32,vec_factor> v_hi_tile2;
  aie::vector<int32,vec_factor> v_max;

  aie::vector<int32,vec_factor> v_zeros = aie::zeros<int32, vec_factor>();

  // start from the second half of the buffer (i.e, from the higher words)
  const int32_t *__restrict pA_outer_base = pA + tile_size;
  const int32_t *__restrict pA_load = pA_outer_base;

  int16_t *__restrict ps = max_exp_A;

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_RANGE(1,4)
  for (int outer_batch_idx=0; outer_batch_idx < n_outer_batches; outer_batch_idx++) {
    v_max = v_zeros;

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(1,2)
    // there should be at least 2 loads to get element-wise max (see static assertion above)
    for (int inner_batch_idx=0; inner_batch_idx < n_inner_batches; inner_batch_idx++) {
        // ************load first tile (not necessarily full inner tile) in the current row of inner tiles**********
         v_hi_tile1 = aie::load_v<vec_factor>(pA_load);
        pA_load += vec_factor;

        // get the exponents 
        v_hi_tile1 = aie::upshift(v_hi_tile1, 1);
        v_hi_tile1 = aie::logical_downshift(v_hi_tile1, 32 - n_exponent_bits);

        if constexpr (n_inner_tile_cols>1) {
          // **********load second tile (not necessarily full inner tile) from the current column or row*********
          v_hi_tile2 = aie::load_v<vec_factor>(pA_load);  
          pA_load += inner_step_size_to_next_inner_tile;

          v_hi_tile2 = aie::upshift(v_hi_tile2, 1);
          v_hi_tile2 = aie::logical_downshift(v_hi_tile2, 32 - n_exponent_bits);
          
          v_max = aie::max(v_max, aie::max(v_hi_tile1, v_hi_tile2));
        }
        else {
          v_max = v_hi_tile1;
        }
      }

      for (int max_group_idx=0; max_group_idx < vec_factor / max_group_dim; max_group_idx++) {
          aie::vector<int32, max_group_dim> current_group = v_max.template extract<max_group_dim>(max_group_idx);
          *ps = std::max(*ps, static_cast<int16_t>(aie::reduce_max(current_group)) );
          ++ps;
      }

    // move to next row of outer batches
    if constexpr (n_intra_inner_tile_steps>1) {
      int intra_tile_step = (outer_batch_idx + 1) % n_intra_inner_tile_steps == 0? 0 : vec_factor;
      pA_outer_base += intra_tile_step + (intra_tile_step==0? outer_step_size : 0);
    } else {
      pA_outer_base += outer_step_size;
    }

    pA_load = pA_outer_base;
  }

}

/*
This method used acc64 lanes in accumulator instead of int32 lanes of vectors to conduct the splits. This version
supports mantissa coverages of up to 127 bits
This revision of the original acc64 based splitting method assumes buffer of input matrix is divided into two halves
first half : lower words of FP64 in the same inner tiles layout
second half : higher words of FP64 in the same inner tiles layout
*/
template <unsigned doubles_per_loop, unsigned bits_per_slice, unsigned num_splits, typename T_in> 
static void extract_splits_acc64_half_half(const int32_t *__restrict pA_lo, const int32_t *__restrict pA_hi, T_in *__restrict ps, int step_size, int inner_tile_size, int tile_size, aie::vector<int16_t, doubles_per_loop> &v_max_exp, int n_batches, int tile, int mul_iter, int exp_batch, int inner_batch_idx,  int64_t *__restrict log_buff) 
{
   constexpr int min_vector_bit_width = 128;
   constexpr int vec_factor = doubles_per_loop;
    // Bits to avoid : 
    // (1) sign bit . as causes ones when right shifting and then need to be removed
    // (2) 3 bits     In AIE-ML\XDNA1 , avoiding these three bits make sure we dont downshift from accumulator by greater than 59 (unsafe shift range)
    //                We also used this for XDNA2 to keep safe. If we only avoid the sign bit on XDNA2, we faced similar issues
    // (4) mod bits   To avoid handling partial split across accumulators seperately to simplify logic
    constexpr int bits_to_avoid = 1 + (bits_per_slice==3 ? 3 : 0);

    constexpr int n_valid_bits_per_acc[] = { 
        64 - bits_to_avoid - ((64 - bits_to_avoid) % bits_per_slice), 
        64 - bits_to_avoid
    };
    constexpr int n_acc64 = (num_splits * bits_per_slice) <= n_valid_bits_per_acc[0] ? 1 : 2;
    constexpr int n_splits_per_acc[] = { std::min(static_cast<signed>(num_splits), n_valid_bits_per_acc[0] / static_cast<signed>(bits_per_slice)), static_cast<signed>(num_splits) - n_valid_bits_per_acc[0] / static_cast<signed>(bits_per_slice) };

   const uint32_t *__restrict pA_lo1 = (uint32_t __aie_dm_resource_a *) pA_lo;
   const uint32_t *__restrict pA_hi1 = (uint32_t __aie_dm_resource_a *) pA_hi;

   // for the sign 
  aie::mask<vec_factor> is_neg;

  // for implicit 1 checks
  aie::mask<vec_factor> is_implicit_one_set; 
  aie::mask<vec_factor> is_sub; // is subnormal number?
  // accumulators for keeping shifted mantissa 
  aie::accum<acc64, vec_factor> acc_split[n_acc64];
  aie::accum<acc64, vec_factor> acc_tmp = aie::zeros<acc64, vec_factor>();

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_RANGE(1,2)
  for (int iter=0; iter<n_batches; iter++) {          
      aie::vector<T_in, vec_factor > v_split; 
      // for the exponent biases for mantissa shifts
      aie::vector<uint32, vec_factor  > v_exp_bias;
      aie::vector<uint32, vec_factor  > v_exp_bias_pow2;     
      {
          aie::vector<uint32,vec_factor> v_upshift_sets[n_acc64 * 2]; 
          {
              aie::vector<uint32, vec_factor > v_hi;
              aie::vector<uint32, vec_factor > v_lo;
              v_hi = aie::load_v<vec_factor>(pA_hi1);
              v_lo = aie::load_v<vec_factor>(pA_lo1);       
              
              // capture signs of each double from the higher words first
              is_neg = aie::lt(v_hi.template cast_to<int32>(), 0);

              int start_bit_offset = 11;
              int n_slice_bits = 32 - start_bit_offset;    // no of bits containing slices (excludes the sign and exponent bits except implicit one bit)

              // get exponents
              v_exp_bias = aie::upshift(v_hi, 1);
              v_exp_bias = aie::logical_downshift(v_exp_bias, 32 - 11);

              // capture subnormals
              is_sub = aie::eq(v_exp_bias, 0u);
              // get exponent biases relative to the row/col max
              v_exp_bias = aie::sub(v_max_exp.unpack().template cast_to<uint32>(), v_exp_bias);
              
              // as the signs and the bias of exponents have been extracted, the sign and exponent bits are not needed now in the original higher words
              // removing exponents to simplify mantissa shifting 
              v_hi = aie::upshift(v_hi, 12);
              v_hi = aie::logical_downshift(v_hi, 12);
              // add implicit one if not subnormal
              v_hi = aie::add(v_hi, aie::select(1u << (32 - (11 + 1)), 0u, is_sub));

              // *********************************SETUP PHASE*****************************************************

              // first align the 53-bit mantissa with the msb of the higher word 
              //                              Higher 32-bit                   Lower-32 bit
              // e.g original :  00000000000111111111111111111111 11111111111111111111100000000000
              //     aligned  :  11111111111111111111111111111111 11111111110000000000000000000000
              v_hi = aie::upshift(v_hi, 11);
              v_hi = aie::add(aie::logical_downshift(v_lo, 32 - 11), v_hi);
              
              v_lo = aie::upshift(v_lo, 11);

              // keep 4 int32-words according to the exponent bias range thresholds: 0, 32, 64, 96
              // we choose this ranges as we can only shift a 32-bit part by upto 31 bits most using uint32
              // in aie::mac operation. (for 32 bit shifting we do a 31 bit shift and then add to emulate 32 bit shift)
              // these 4 32-bit words represent the state of the 2 accumulators when the bias is exactly at these
              // above thresholds
              v_upshift_sets[0] = aie::select(v_hi, 0u, aie::ge(v_exp_bias, 32u));
              v_upshift_sets[1] = aie::select(v_lo, v_hi, aie::ge(v_exp_bias, 32u));
              v_upshift_sets[1] = aie::select(v_upshift_sets[1], 0u, aie::ge(v_exp_bias, 64u));

              if constexpr (n_acc64==2) {
                  v_upshift_sets[2] = aie::select(0u, v_lo , aie::ge(v_exp_bias, 32u));
                  v_upshift_sets[2] = aie::select(v_upshift_sets[2], v_hi, aie::ge(v_exp_bias, 64u));
                  v_upshift_sets[2] = aie::select(v_upshift_sets[2], 0u, aie::ge(v_exp_bias, 96u));

                  v_upshift_sets[3] = aie::select(0u, v_lo , aie::ge(v_exp_bias, 64u));
                  v_upshift_sets[3] = aie::select(v_upshift_sets[3], v_hi, aie::ge(v_exp_bias, 96u));
              }
          } // scope ends to relieve register pressure : v_hi and v_lo not needed any more
    
          // *********************************MANTISSA SHIFTS*****************************************************

          // clamp values to 0-31
          v_exp_bias = aie::downshift(aie::upshift(v_exp_bias, 32-5), 32-5);
          // switch to power of 2 for mac-based shifts using accumulators
          v_exp_bias_pow2 = exp2(aie::sub(32u, v_exp_bias));
        
          // important to zero out the accumulators for next batch to avoid stray bits from last iter
          acc_split[0] = aie::zeros<acc64,vec_factor>();
          if constexpr (n_acc64==2) {
            acc_split[1] = aie::zeros<acc64,vec_factor>();
          }

          // incorporate shifted mantissas from the set 32-bit partitions and the relative exp biases
          for (int acc_idx=0; acc_idx<n_acc64; acc_idx++) {
              acc_split[acc_idx] = aie::mac(acc_split[acc_idx], v_upshift_sets[acc_idx*2], v_exp_bias_pow2);            
              acc_split[acc_idx] = aie::mac(acc_split[acc_idx], aie::select(0u, v_upshift_sets[acc_idx*2], aie::eq(v_exp_bias,0u)), v_exp_bias_pow2);

              acc_tmp = aie::mul(v_upshift_sets[acc_idx * 2 + 1], v_exp_bias_pow2);
              acc_tmp = aie::mac(acc_tmp, aie::select(0u, v_upshift_sets[acc_idx * 2 + 1], aie::eq(v_exp_bias,0u)), v_exp_bias_pow2);

              acc_split[acc_idx] = aie::add(acc_split[acc_idx], acc_tmp.template to_vector_sign<uint32>(false, 32));

              if constexpr (n_acc64==2) {
                if (acc_idx==0) {
                  acc_split[acc_idx + 1].from_vector_sign(acc_tmp.template to_vector<uint32>() , false, 32);
                }
              }
          }
      } // scope ends to relieve register pressure as the the set of 4 32-bit parts no longer needed 

      // shift the mantissa out across the two acc64 accumulators to allow straightforward split extraction
      // avoiding sign bit in accumulator's arithmetic right shift and unsafe shifts possible with XDNA1 otherwise
      {
          aie::vector<uint32,vec_factor> v_tmp;

          const int n_bits_shifted_out_acc1 = (64 - n_valid_bits_per_acc[0]);
          const int n_bits_shifted_out_acc2 = (64 - n_valid_bits_per_acc[0]) + (64 - n_valid_bits_per_acc[1]);

          v_tmp = acc_split[0].template to_vector<uint32>(32);
          v_tmp = aie::logical_downshift(v_tmp, n_bits_shifted_out_acc1);
          acc_tmp.from_vector(v_tmp, 32);
          v_tmp = acc_split[0].template to_vector<uint32>(n_bits_shifted_out_acc1);
          acc_tmp = aie::add(acc_tmp, v_tmp);

          v_tmp = acc_split[0].template to_vector<uint32>();
          acc_split[0] = acc_tmp;
  
          if constexpr (n_acc64==2) {
              v_tmp = aie::upshift(v_tmp, 32 - n_bits_shifted_out_acc1);
              acc_tmp.from_vector(v_tmp, 32 - (64 - n_valid_bits_per_acc[1]));

              acc_tmp = aie::mac(acc_tmp, acc_split[1].template to_vector<uint32>(32), 1u << (32 - n_bits_shifted_out_acc2));
              acc_tmp = aie::add(acc_tmp, aie::logical_downshift(acc_split[1].template to_vector<uint32>(), n_bits_shifted_out_acc2));

              acc_split[1] = acc_tmp;
          }
      }
      
      // *********************************SPLITS*****************************************************

      int split_no=0;

      AIE_PREPARE_FOR_PIPELINING
      AIE_LOOP_RANGE(1,2)
      // extract splits starting with the higher accumulator and then the lower one
      for (int acc_idx=0; acc_idx<n_acc64  ;acc_idx++) {
          AIE_PREPARE_FOR_PIPELINING
          AIE_LOOP_RANGE(1,20)
          for (int local_split_no = 0; local_split_no < n_splits_per_acc[acc_idx]; local_split_no++, split_no++) {
            int shift = n_valid_bits_per_acc[acc_idx] - bits_per_slice * (local_split_no + 1);
            v_split = acc_split[acc_idx].template to_vector<T_in>(shift);
            acc_tmp.from_vector(v_split, shift);
            acc_split[acc_idx] = aie::sub(acc_split[acc_idx], acc_tmp);  

            // apply sign to slices
            v_split = aie::select(v_split, aie::neg(v_split), is_neg);

            aie::store_v(ps, v_split);
            ps += (tile_size / (std::is_same_v<T_in, int4_t>? 2 : 1));
          }
      }

      pA_lo1 += step_size;
      pA_hi1 += step_size;

      // reset to the first split buffer for next batch
      ps -= (num_splits * (tile_size / (std::is_same_v<T_in, int4_t>? 2 : 1) ) ) ;
      ps += (step_size / (std::is_same_v<T_in, int4_t>? 2 : 1));

    } // inner batch iteration ends
}

template <unsigned m, unsigned k, unsigned inner_m, unsigned inner_k, unsigned bits_per_slice, unsigned num_splits, typename T_in>
static inline void ozaki_split_vectorized_revised_half_half(const int32_t * pA, T_in * p_split_A, int16_t *__restrict max_A, int tile, int mul_iter, int64_t *__restrict log_buff = 0) {
  constexpr int min_vector_bit_width = 128;
  constexpr int min_load_size_i16s = min_vector_bit_width / 16;
  constexpr int doubles_per_loop = std::is_same_v<T_in, int4_t>? 32 : 16;
  constexpr int tile_size = m*k;
  constexpr int inner_tile_size = inner_m * inner_k;
  constexpr int exp_max_load_batch_size = std::is_same_v<T_in, int4_t>? 2 : 8;
  constexpr int n_exp_max = m;
  constexpr int exp_max_replicate_factor = inner_k;
  // we can traverse using pattern less than or equal to the inner tile size but never larger than an inner tile size
  // as that will now allow to align over contiguous set of tiles. (a "rows" batch)
  constexpr int exp_max_pattern_size = (doubles_per_loop <= inner_tile_size) ? doubles_per_loop  : inner_tile_size;
  constexpr int n_inner_batches = (exp_max_load_batch_size * exp_max_replicate_factor) / exp_max_pattern_size;
  constexpr int step_size = doubles_per_loop <= inner_tile_size ? inner_tile_size : doubles_per_loop;
  constexpr int n_doubles_per_loop_batches_per_inner_batch = (k / inner_k * inner_tile_size / step_size);

  constexpr int vec_factor = doubles_per_loop;

  const int32_t *pA_lo;  // points into the first half for lower words of FP64
  const int32_t *pA_hi;  // points into the second half for higher words of FP64
  
  T_in *__restrict ps;

  // IMPORTANT : splits will not work properly without these. setting it here once instead of in each
  // call to the actual split extraction helper function to avoid overhead of these set methods.
  aie::set_rounding(aie::rounding_mode::floor);
  aie::set_saturation(aie::saturation_mode::none);

  // 00000000
  // 00000000
  // 00000000
  // 00000000
  // 00000000
  // 00000000
  // 00000000
  // 00000000
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_RANGE(2, 8)
  for (int exp_batch=0; exp_batch < n_exp_max / exp_max_load_batch_size; exp_batch++) {
    aie::vector<int16_t, exp_max_load_batch_size*exp_max_replicate_factor> v_exp_max_batch; 
    int extract_idx_exp_max_batch = 0;

    {
        aie::vector<int16_t, min_load_size_i16s*exp_max_replicate_factor> v_exp_max_initial_load;

        int offset_for_initial_load =  exp_max_load_batch_size >= min_load_size_i16s? (exp_batch*exp_max_load_batch_size) :
             ( ( (exp_batch*exp_max_load_batch_size) / min_load_size_i16s) * min_load_size_i16s);      
        extract_idx_exp_max_batch =  exp_max_load_batch_size >= min_load_size_i16s? 0 : (exp_batch % (min_load_size_i16s / exp_max_load_batch_size) );

        // transpose of vectors with more than 64 elements not supported on XDNA1, so we need to emulate using interleave_zip
        // Note: such tile sizes (elements > 64) happens with int4 multiplication modes so performance is not a concern in these cases for now as 
        // the multiplications are already high due to lower bits per slice and more splits. We use the int4 modes mainly for reaching
        // more accuracy while restricting bin-wise summation to INT64s=2 per C element in summation phase. 
        if constexpr (aie::arch::is(aie::arch::ArchVersion::AIE_ML) && (min_vector_bit_width / 16 * exp_max_replicate_factor)==128) {
            aie::vector<int16_t, (min_load_size_i16s)*exp_max_replicate_factor/2> v_exp_max_initial_load_half = 
                  aie::load_v<min_load_size_i16s>(max_A + offset_for_initial_load).template grow_replicate<(min_load_size_i16s)*exp_max_replicate_factor/2>();
            
            v_exp_max_initial_load_half = aie::transpose(v_exp_max_initial_load_half, exp_max_replicate_factor/2, (min_load_size_i16s));
            
            std::pair<
                aie::vector<int16_t, ((min_load_size_i16s)*exp_max_replicate_factor)/2>,
                aie::vector<int16_t, ((min_load_size_i16s)*exp_max_replicate_factor)/2>
            > iv_transpose_half_half = aie::interleave_zip(v_exp_max_initial_load_half, v_exp_max_initial_load_half, exp_max_replicate_factor/2);
            
            v_exp_max_initial_load = aie::concat(iv_transpose_half_half.first, iv_transpose_half_half.second);
        }
        else 
        {
          // load from max and grow_replicate inner_k times and then transpose 
          v_exp_max_initial_load = 
              aie::load_v<min_load_size_i16s>(max_A + exp_batch*exp_max_load_batch_size).template grow_replicate<(min_load_size_i16s)*exp_max_replicate_factor>();

          v_exp_max_initial_load = aie::transpose(v_exp_max_initial_load, exp_max_replicate_factor , (min_load_size_i16s));
        }

        v_exp_max_batch = v_exp_max_initial_load.template extract<exp_max_load_batch_size*exp_max_replicate_factor>(extract_idx_exp_max_batch);
    }
    
    // 00000000
    // 00000000
    // 00000000
    // 00000000
    // 11111111
    // 11111111
    // 11111111
    // 11111111
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(2, 8)
    for (int inner_batch_idx=0; inner_batch_idx < n_inner_batches; inner_batch_idx++) {
        aie::vector<int16_t, doubles_per_loop> v_max_exp;

        // scope here so that v_max_batch can be released to reduce register pressure
        // we only need v_max_exp for the inner most doubles per loop batches
        {
            aie::vector<int16_t, exp_max_pattern_size> v_max_batch = v_exp_max_batch.template extract<exp_max_pattern_size>(inner_batch_idx);

            if constexpr (exp_max_pattern_size < doubles_per_loop) {
              v_max_exp = v_max_batch.template grow_replicate<doubles_per_loop>();
            } else {
              v_max_exp = v_max_batch.template extract<doubles_per_loop>(0);
            }
        }
         
        int total_inner_batches = (exp_batch * n_inner_batches) + inner_batch_idx;
        int current_tile = total_inner_batches / 
            (exp_max_pattern_size <= inner_tile_size? 
                    (inner_tile_size / exp_max_pattern_size) : 
                    (exp_max_pattern_size / inner_tile_size)
            );
          
        int start_offset = k / inner_k * inner_tile_size * current_tile;

        // the step size within a tile applies if the pattern match for each "rows" batch is smaller than inner tile .
        int intra_tile_step = exp_max_pattern_size >= inner_tile_size? 0 : 
            ( (total_inner_batches % (inner_tile_size / exp_max_pattern_size)) * exp_max_pattern_size);

        pA_lo = (pA + start_offset + intra_tile_step);
        pA_hi = (pA + tile_size + start_offset + intra_tile_step);
        ps = p_split_A + ((start_offset + intra_tile_step) / (std::is_same_v<T_in, int4_t>? 2 : 1));    

        // seperated this into a helper func so as to avoid overflowing program memory. Here, template-based multiple versions
        // based on matrix type (A or B) are avoided by explicitly passing in values for otherwise constexpr which are dependent
        // on the matrix type
        extract_splits_acc64_half_half<doubles_per_loop, bits_per_slice, num_splits, T_in>(pA_lo, pA_hi, ps, 
          step_size, inner_tile_size, tile_size, v_max_exp, n_doubles_per_loop_batches_per_inner_batch, tile, mul_iter, exp_batch, inner_batch_idx, log_buff);    
          
    } 
  }
  
}

extern "C" {

#define ozaki_max_exp_vectorized_c_func(ctype_in, mlir_type_in, ctype_out,      \
                                 mlir_type_out, r, s, t)                       \
  void ozaki_get_max_exponent_A_inner_tiled_wise(int32_t *a_in, int16_t *max_exp_out, int64_t* log_buff=0) {  \
      /* event0(); */  /* max_exp_a_cycles_measure_start */ \
      ozaki_get_max_exponent_inner_tiled_wise_half_half<DIM_M, DIM_K, r, s>(a_in, max_exp_out, log_buff); \
      /* event1(); */  /* max_exp_a_cycles_measure_stop */ \
  } \
  void ozaki_get_max_exponent_B_inner_tiled_wise(int32_t *b_in, int16_t *max_exp_out, int64_t* log_buff=0) {  \
      /* event0(); */  /* max_exp_b_cycles_measure_start */ \
      ozaki_get_max_exponent_inner_tiled_wise_half_half<DIM_N, DIM_K, t, s>(b_in, max_exp_out, log_buff); \
      /* event1(); */  /* max_exp_b_cycles_measure_stop */ \
  }

#define ozaki_split_vectorized_c_func(ctype_in, mlir_type_in, ctype_out,      \
                                 mlir_type_out, r, s, t)                       \
  void ozaki_split_vectorized_revised_A_##mlir_type_in(int32_t *a_in, ctype_in *c_out, int16_t* max_exp_buff, int tile, int mul_iter, int64_t* log_out) {  \
    /* event0(); */  /* splits_a_cycles_measure_start */ \
    ozaki_split_vectorized_revised_half_half<      \
        DIM_M, DIM_K, r, s, BITS_PER_SLICE, NUM_SPLITS, ctype_in>(a_in, c_out, max_exp_buff, tile, mul_iter, log_out);                               \
    /* event1(); */  /* splits_a_cycles_measure_stop */ \
  } \
  void ozaki_split_vectorized_revised_B_##mlir_type_in(int32_t *a_in, ctype_in *c_out, int16_t* max_exp_buff, int tile, int mul_iter, int64_t* log_out) {  \
    /* event0(); */  /* splits_b_cycles_measure_start */ \
    ozaki_split_vectorized_revised_half_half<      \
        DIM_N, DIM_K, t, s, BITS_PER_SLICE, NUM_SPLITS, ctype_in>(a_in, c_out, max_exp_buff, tile, mul_iter, log_out);    \ 
    /* event1(); */  /* splits_b_cycles_measure_stop */ \
  }

#define ozaki_sum_vectorized_c_func(ctype_in, mlir_type_in, ctype_out,\
                                mlir_type_out, r, s, t)                       \
  void ozaki_sum_vectorized_upshift_##mlir_type_out(ctype_out *c_bins, int64_t *c_out, int16_t *row_max_exp, int16_t *col_max_exp, int64_t *log_buff) { \
    ozaki_vectorized_sum_upshift<ctype_out, DIM_M, DIM_N, r, t, BITS_PER_SLICE, NUM_SPLITS>(c_bins, c_out, row_max_exp, col_max_exp, log_buff); \
  } \

#define ozaki_zero_vectorized_c_func(ctype_in, mlir_type_in, ctype_out,              \
                               mlir_type_out, r, s, t)                         \
  void zero_n_times_##mlir_type_out(ctype_out *c_out, int n_times) {                       \
    zero_vectorized<ctype_out, DIM_M, DIM_N>(c_out, n_times);                           \
  } \
  void zero_n_times_i64(int64_t *c_out, int n_times) {                       \
    zero_vectorized<DIM_M, DIM_N>(c_out, n_times);                           \
  } \
  void zero_n_times_log(int64_t *c_out, int n_times) {                       \
    zero_vectorized<DIM_M, DIM_N>(c_out, n_times);                           \
  } \
  void zero_max_exp_buffer_A(int16_t *buff) {                                \
    aie::store_v(buff, aie::zeros<int16_t, DIM_M>());                           \
  } \
  void zero_max_exp_buffer_B(int16_t *buff) {                                \
    aie::store_v(buff, aie::zeros<int16_t, DIM_N>());                           \
  } \

combos(ozaki_zero_vectorized_c_func) 
combos(ozaki_max_exp_vectorized_c_func)
combos(ozaki_split_vectorized_c_func) 
combos(ozaki_sum_vectorized_c_func)

} // extern "C"