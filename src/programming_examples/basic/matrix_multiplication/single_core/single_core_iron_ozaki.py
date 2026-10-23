#===- single_core_iron_ozaki.py ----------------------------------------------000---*- python -*-===//
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## (c) Copyright 2023 Advanced Micro Devices, Inc.
#===----------------------------------------------------------------------===//
# Modified in 2026
# Original source: https://github.com/Xilinx/mlir-aie/tree/v1.3.2/programming_examples/basic/matrix_multiplication/single_core/single_core_iron.py

import argparse
import numpy as np
import math
import aie.extras.types as extratypes
import aie.extras.dialects.arith as arith

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker, Buffer, str_to_dtype
from aie.iron.device import NPU1, NPU2, Tile
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessSequence, TensorTiler2D, TensorAccessPattern
from aie.extras.dialects.arith import index_cast, constant, cmpi, CmpIPredicate, remui, remsi
from aie.helpers.dialects.scf import if_


microkernel_mac_dim_map = {
    "npu": {
        "i4": (4, 16, 8),
        "i8": (4, 8, 8),
        "i16": (4, 4, 4),
    },
    "npu2": {
        "i4": (4,16,16),
        "i8": (8, 8, 8),
        "i16": (4, 4, 8),
    },
}


def main():
    argparser = argparse.ArgumentParser(
        prog="AIE Matrix Multiplication MLIR Design (Single Core)",
        description="Emits MLIR code for a matrix multiplication design of the given input size",
    )
    argparser.add_argument("--dev", type=str, choices=["npu", "npu2"], default="npu")
    argparser.add_argument("-M", type=int, default=256)
    argparser.add_argument("-K", type=int, default=256)
    argparser.add_argument("-N", type=int, default=256)
    argparser.add_argument("-m", type=int, default=64)
    argparser.add_argument("-k", type=int, default=64)
    argparser.add_argument("-n", type=int, default=32)
    argparser.add_argument(
         "--bps", type=int, default=7
    )
    argparser.add_argument(
        "--num_splits", type=int, default=8
    )

    argparser.add_argument(
        "--dtype_in", type=str, choices=["i4", "i8", "i16"], default="i8"
    )
    
    argparser.add_argument(
        "--dtype_out",
        type=str,
        choices=["i32", "i64"],
        default="i32"
    )
    argparser.add_argument("--b-col-maj", type=int, choices=[0, 1], default=0)
    argparser.add_argument("--trace_size", type=int, default=0)
    argparser.add_argument(
        "--generate-taps",
        action="store_true",
        help="Generate TensorAccessPatterns, a Python object to represent each data transfer"
        "of the input/output matrices. These objects can be used for visualization.",
    )
    argparser.add_argument(
            "--full_dump",
            type=int,
            default=0,
            help="Whether to do a full dump of all phases in the C output. Useful for reference check for debugging.",
        )

    argparser.add_argument(
                "--num_log_tiles",
                type=int,
                default=1,
                help="How many tiles allocated for logging info from SIMD kernels when doing full dump?",
            )


    args = argparser.parse_args()
    maybe_module = my_matmul(
        args.dev,
        args.M,
        args.K,
        args.N,
        args.m,
        args.k,
        args.n,
        args.bps,
        args.num_splits,
        args.dtype_in,
        args.dtype_out,
        args.b_col_maj,
        args.trace_size,
        args.generate_taps,
        args.full_dump,
        args.num_log_tiles
    )
    if args.generate_taps:
        # maybe_module is actually taps
        return maybe_module
    else:
        # print mlir
        print(maybe_module)


# Need ceildiv to capture partial tiling patterns
def ceildiv(a, b):
    return (a + b - 1) // b

def my_matmul(
    dev,
    M,
    K,
    N,
    m,
    k,
    n,
    bps_str,
    num_splits,
    dtype_in_str,
    dtype_out_str,
    b_col_maj,
    trace_size,
    generate_taps=False,
    full_dump=0,
    num_log_tiles=1
):

    assert M % m == 0
    assert K % k == 0
    assert N % n == 0

    bps = np.int32(bps_str)

    # accumulation type
    dtype_out = str_to_dtype(dtype_out_str)

    if (dtype_in_str=="i4"):
        dtype_in = np.int8
    else: 
        dtype_in = str_to_dtype(dtype_in_str)
    
    # r, s, t are the dimensions required by the microkernel MAC instructions.
    mac_dims = microkernel_mac_dim_map[dev][dtype_in_str]
    r, s, t = mac_dims

    assert m % r == 0
    assert k % s == 0
    assert n % t == 0

    vectorized = True
    enable_tracing = True if trace_size > 0 else False

    assert np.issubdtype(dtype_in, np.integer) == np.issubdtype(
        dtype_out, np.integer
    ), f"Input dtype ({dtype_in}) and output dtype ({dtype_out}) must either both be integral or both be float"
    assert (
        np.dtype(dtype_out).itemsize >= np.dtype(dtype_in).itemsize
    ), f"Output dtype ({dtype_out}) must be equal or larger to input dtype ({dtype_in})"

    M_div_m = M // m
    K_div_k = K // k
    N_div_n = N // n
    tiles = M_div_m * N_div_n

    # number of dump tiles to include per a C-tile output. Applied only when using full dumps (full_dump=1) for debugging
    dump_length_factor = (1 + K_div_k * (2 + num_splits) + num_splits + 1 + num_log_tiles)

    # These will hold TensorAccessPattern objects that represent the runtime
    # npu_dma_memcpy_nd operations of this design. They are only used if generate_taps is true
    A_taps = []
    B_taps = []
    C_taps = []

    # Define tensor types
    A_ty = np.ndarray[(M * K,), np.dtype[np.float64]]
    B_ty = np.ndarray[(K * N * 2,), np.dtype[np.int32]]
    if full_dump:
        C_ty = np.ndarray[(tiles * m * dump_length_factor * n * 2,), np.dtype[np.int64]]
    else:
        C_ty = np.ndarray[(M * N * 2,), np.dtype[np.int64]]

    a_exp_ty = np.ndarray[(m * k * 2,), np.dtype[np.int32]]
    a_max_ty = np.ndarray[(m,), np.dtype[np.int16]]
    a_ty = np.ndarray[(m, k * 2), np.dtype[np.int32]]

    b_exp_ty = np.ndarray[(n * k * 2,), np.dtype[np.int32]]    # column major
    b_max_ty = np.ndarray[(n,), np.dtype[np.int16]]
    b_ty = np.ndarray[(k, n * 2), np.dtype[np.int32]]
    
    c_ty = np.ndarray[(m, n * 2), np.dtype[np.int64]]
    c_bins_ty = np.ndarray[(num_splits, m, n), np.dtype[dtype_out]]

    splits_ty_a = np.ndarray[(num_splits, m, k // (2 if dtype_in_str=="i4" else 1)), np.dtype[dtype_in]]
    splits_ty_b = np.ndarray[(num_splits, k, n // (2 if dtype_in_str=="i4" else 1)), np.dtype[dtype_in]]

    # only for full dumps for debugging
    log_ty = np.ndarray[(((m*num_log_tiles) if full_dump else 1), (n * 2) if full_dump else 16), np.dtype[np.int64]]
    kernel_file = f"mm_ozaki_{m}x{k}x{n}_b{bps}_d{num_splits}.o"

    # AIE Core Function declarations
    func_type = ""
    zero_kernels = {
        t_in : Kernel(
            f"zero_n_times_{func_type}{t_in}", 
            kernel_file, [ty, np.int32]
        )
        for t_in, ty in [(dtype_out_str, c_bins_ty),("i64", c_ty)]
    } 

    zero_log_kernel = Kernel(
            f"zero_n_times_log", 
            kernel_file, [log_ty, np.int32])

    zero_max_buff_kernels = {
        mat : Kernel(
            f"zero_max_exp_buffer_{mat}",
            kernel_file,
            [max_ty],
        )
        for mat, max_ty in (("A", a_max_ty), ("B", b_max_ty))
    }

    matmul_vectorized_func_name = f"ozaki_matmul_{dtype_in_str}_{dtype_out_str}_double_s"
    matmul_kernel = Kernel(
        matmul_vectorized_func_name,
        kernel_file,
        [splits_ty_a, splits_ty_b, c_bins_ty, np.int32, np.int32, log_ty]
    )
    matmul_sum_vectorized_func_name = f"ozaki_sum_vectorized_upshift_{dtype_out_str}"
    sum_kernel = Kernel(
        matmul_sum_vectorized_func_name,
        kernel_file,
        [c_bins_ty, c_ty, a_max_ty, b_max_ty, log_ty],
    )

    max_exp_kernels = {
        mat : Kernel(
            f"ozaki_get_max_exponent_{mat}_inner_tiled_wise",
            kernel_file,
            [exp_ty, max_ty, log_ty],
        )
        for mat, exp_ty, max_ty in (("A", a_ty, a_max_ty), ("B", b_ty, b_max_ty))
    }

    matmul_split_vectorized_func_name = f"ozaki_split_vectorized_revised_{{mat}}_{dtype_in_str}"
    split_kernels = {
        mat : Kernel(
            matmul_split_vectorized_func_name.format(mat=mat),
            kernel_file,
            [ty, splits_ty, max_ty, np.int32, np.int32, log_ty]
        ) for mat, ty, splits_ty, max_ty in (("A", a_ty, splits_ty_a, a_max_ty), ("B", b_ty, splits_ty_a, b_max_ty))
    }

    inA = ObjectFifo(a_ty, name="inA")
    a_dims = None
    if vectorized:
         a_dims = [
             (m // r, r * k * 2), 
             (k // s, s * 2), 
             (r, k * 2), 
             (s * 2, 1)
         ]
         a_dims_from = [
            (m * k, 1),
            (2, m * k),
            (1, 1)
         ]
    memA = inA.cons().forward(name="memA", dims_to_stream=a_dims, dims_from_stream=a_dims_from)    

    inB = ObjectFifo(b_ty, name="inB")
    # we need different set of transformations (at memory tile DMA and receiving core) 
    # based on whether B was initially column major or not
    if b_col_maj:
        b_dims = [(n // t, t * k * 2), (k // s, s * 2), (t, k * 2), (s * 2, 1)]
        b_dims_from = [(n * k, 1), (2, n * k), (1, 1)]
    else:
        b_dims = [(n // t, t * k * 2), (k * 2  // s, s), (t, k*2), (s, 1)]
        b_dims_from = [ (n // t, k * t), (2, n * k), (k * t, 1) ]

    memB = inB.cons().forward(name="memB", dims_to_stream=b_dims, dims_from_stream=b_dims_from)

    # Output C
    memC = ObjectFifo(c_ty, name="memC", depth=(2 if not full_dump else 3))
    c_dims = None
    if vectorized:
        c_dims = [(m // r, r * n * 2), (r, t * 2), ((n * 2) // (t * 2), r * (t * 2)), ((t * 2), 1)]
    if full_dump:
        c_dims = None
    outC = memC.cons().forward(name="outC", dims_to_stream=c_dims)

    cBinsBuffer = Buffer(c_bins_ty, name="cBinsBuff", initial_value=np.array([0 for i in range(m * n * num_splits)], dtype=np.int32))
    splitsBufferA = Buffer(splits_ty_a, name="splitsBuffA")
    splitsBufferB = Buffer(splits_ty_b, name="splitsBuffB")
    maxBufferA = Buffer(a_max_ty, name="maxBuffA", initial_value=np.array([0 for i in range(m)], dtype=np.int16))
    maxBufferB = Buffer(b_max_ty, name="maxBuffB", initial_value=np.array([0 for i in range(n)], dtype=np.int16))
    logBuffer = Buffer(log_ty, name="logBuffer", initial_value=np.array([0 for i in range((m*num_log_tiles*n*2) if full_dump else (1*16))], dtype=np.int64))

    # Task each core will run
    def core_fn(
        of_a,
        of_b,
        of_c, 
        max_buff_A, 
        max_buff_B,
        split_buff_a,
        split_buff_b,
        c_bins_buff, 
        log_buff, 
        zero_cbins,
        zero_c_out,
        zero_log,
        zero_max_buffA,
        zero_max_buffB,
        max_expA, 
        max_expB,
        split_A,
        split_B, 
        matmul, 
        sum
    ):
        for t in range_(tiles) if tiles > 1 else range(1):  # issue #1547
            if full_dump:
                zero_log(log_buff, 2*num_log_tiles)

            # max exponent of A's rows from a row of A tiles needs to be refreshed for every row of C tiles produced as the row 
            # of A tiles used is the same
            zero = arith.constant(0, type=extratypes.index())
            r = remsi(t, constant(N_div_n, extratypes.index()))
            is_max_exp_a_refresh = arith.cmpi("eq", r, zero)      
            with if_(is_max_exp_a_refresh) as if_op1:
                zero_max_buffA(max_buff_A)
                for _ in range_(K_div_k) if K_div_k > 1 else range(1):
                    elem_in_a_exp = of_a.acquire(1)
                    max_expA(elem_in_a_exp, max_buff_A, log_buff)
                    of_a.release(1)        

            # max exponent of B's columns needs to be refreshed for every C tile produced as columns of B change
            # for every produced tile
            zero_max_buffB(max_buff_B)
            for b_iter in range_(K_div_k) if K_div_k > 1 else range(1):
                elem_in_b_exp = of_b.acquire(1)
                max_expB(elem_in_b_exp, max_buff_B, log_buff)
                of_b.release(1)
            
            # output tile component for max buff A and B values (FOR DUMP ONLY)
            if (full_dump):
                elem_out = of_c.acquire(1)
                zero_c_out(elem_out, 2)
                for i in range_(m): 
                    ic = index_cast(i, extratypes.i32()) 
                    elem_out[0, i] = arith.extsi(in_=max_buff_A[ic], out=extratypes.i64())
                    elem_out[3, i] = arith.extsi(in_=max_buff_B[ic], out=extratypes.i64())
                of_c.release(1)

            zero_cbins(c_bins_buff, num_splits)
            # issue #1547
            for mul_iter in range_(K_div_k) if K_div_k > 1 else range(1):
                # main_tile_to_save = arith.constant(0, type=extratypes.index())
                # mul_iter_to_save = arith.constant(0, type=extratypes.index())
                # save_split_for_main_tile = arith.cmpi("eq", t, main_tile_to_save)
                # mul_iter_to_mul = arith.cmpi("eq", mul_iter, mul_iter_to_save)
                # cond = arith.andi(save_split_for_main_tile, mul_iter_to_mul)

                elem_in_a = of_a.acquire(1)            
                elem_in_b = of_b.acquire(1)

                # output tile components for A- and B-tiles (FOR DUMP ONLY)
                if (full_dump):
                    elem_outs = of_c.acquire(2)
                    zero_c_out(elem_outs[0], 2)
                    zero_c_out(elem_outs[1], 2)
                    for i in range_(k):
                        for j in range_(n*2):
                            ic = index_cast(i, extratypes.i32())
                            jc = index_cast(j, extratypes.i32())
                            (elem_outs[0])[i, j] = arith.extsi(in_=elem_in_a[ic, jc], out=extratypes.i64())
                            (elem_outs[1])[i, j] = arith.extsi(in_=elem_in_b[ic, jc], out=extratypes.i64())
                    of_c.release(2)

                # with if_(cond) as if_op10:
                split_A(elem_in_a, split_buff_a, max_buff_A, t, mul_iter, log_buff)
                split_B(elem_in_b, split_buff_b, max_buff_B, t, mul_iter, log_buff)
                of_a.release(1)
                of_b.release(1)
                
                # output tile components for splits of A and B (FOR DUMP ONLY)
                if (full_dump):
                    for split_no in range_(num_splits):
                        sn_jc = index_cast(split_no, extratypes.i32())
                        elem_out = of_c.acquire(1)
                        zero_c_out(elem_out, 2)
                        for i in range_(k):
                            for j in range_(n // (2 if dtype_in_str=="i4" else 1)):
                                ic = index_cast(i, extratypes.i32())
                                jc = index_cast(j, extratypes.i32())
                                elem_out[i, j] = arith.extsi(in_=split_buff_a[sn_jc, ic, jc], out=extratypes.i64())
                                elem_out[i, n+j] = arith.extsi(in_=split_buff_b[sn_jc, ic, jc], out=extratypes.i64())
                        of_c.release(1)

                
                
                # with if_(cond) as if_op2:
                matmul(split_buff_a, split_buff_b, c_bins_buff, t, mul_iter, log_buff)

            # output tile component for bins (FOR DUMP ONLY)
            if (full_dump):
                for split_no in range_(num_splits):
                    sn_jc = index_cast(split_no, extratypes.i32())
                    elem_out = of_c.acquire(1)
                    zero_c_out(elem_out, 2)
                    for i in range_(m):
                        for j in range_(n):
                            ic = index_cast(i, extratypes.i32())
                            jc = index_cast(j, extratypes.i32())
                            elem_out[i, j] = arith.extsi(in_=c_bins_buff[sn_jc, ic, jc], out=extratypes.i64())
                    of_c.release(1)
            

            elem_out = of_c.acquire(1)
            zero_c_out(elem_out, 2)
            sum(c_bins_buff, elem_out, max_buff_A, max_buff_B, log_buff)
            of_c.release(1)

            # additional log buffer tile to log things from SIMD kernel for debugging (FOR DUMP ONLY)
            if (full_dump):
                for l_idx in range_(num_log_tiles):
                    elem_out = of_c.acquire(1)
                    zero_c_out(elem_out, 2)
                    for i in range_(m):
                        for j in range_(n*2):
                            elem_out[i, j] = log_buff[m*l_idx+i,j]
                    of_c.release(1)

    # Create worker from task
    worker = Worker(
        core_fn,
        [   
            memA.cons(),
            memB.cons(),
            memC.prod(), 
            maxBufferA, 
            maxBufferB,
            splitsBufferA,
            splitsBufferB,
            cBinsBuffer, 
            logBuffer,
            zero_kernels[dtype_out_str], 
            zero_kernels["i64"], 
            zero_log_kernel,
            zero_max_buff_kernels["A"],
            zero_max_buff_kernels["B"],
            max_exp_kernels["A"], 
            max_exp_kernels["B"],
            split_kernels["A"], 
            split_kernels["B"], 
            matmul_kernel, 
            sum_kernel
        ],
        stack_size=3072,
        tile = Tile(0,2)
    )

    # only do these number of columns in one transfer block
    # at a time before synchronizing, so we can reuse BDs. 
    # Each transfer block consists of one ping and one pong transfer
    # so we wait on the previous while the current is being streamed in parallel for latency hiding
    tb_max_n_cols = 4 if (N_div_n >= 8) else 2

    A_tiles = TensorTiler2D.group_tiler(
        (M, K), (m, k), (1, K_div_k), pattern_repeat=(N_div_n + 1), prune_step=False
    )

    # There is only one access pattern for B - it tiles the entire matrix in (k x n) tiles.
    # Note: we changed here so that the SIMD kernels always receive inner tiles of B in column major format
    # irrespective the input B is in row-major or column-major.
    if b_col_maj:
        B_Tiles = TensorTiler2D.group_tiler(
            (N, K), (n, k), (1, K_div_k), prune_step=False, pattern_repeat=2
        )
    else:
        B_Tiles = TensorTiler2D.group_tiler(
            (K, N*2),
            (k, n*2),
            (K_div_k, 1),
            # Note: with FP64 elements, enabling tile_col_major below results in the lower and higher 32-bit words 
            # to be considered as seperate columns rather than same (perhaps because AIE works on 32-bit boundaries) 
            # so its not a true column major layout
            # we have dealt with it over the DMA transformation at the memory tile and the receiving compute tile
            # to obtain the true column major layout for FP64 tiles
            tile_col_major=True, 
            tile_group_col_major=True,
            iter_col_major=True,
            prune_step=False,
            pattern_repeat=2
        )

    C_tiles = TensorTiler2D.group_tiler(
        (tiles * m * dump_length_factor, n * 2) if full_dump else (M, N*2), # For full sequential dump
        (m, n * 2),
        (dump_length_factor * tb_max_n_cols // 2, 1) if full_dump else (1, tb_max_n_cols // 2)
        # prune_step=False
    )
    c_index = 0

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(A_ty, B_ty, C_ty) as (A, B, C):
        if enable_tracing:
            rt.enable_trace(trace_size, workers=[worker])
        rt.start(worker)

        # Task groups will be used to determine when to sync/await/free DMA runtime ops
        tgs = []
        row_tgs = []
        for row in range(M // m):
            row_tgs.append(rt.task_group())
            rt.fill(inA.prod(), A, tap=A_tiles[row], task_group=row_tgs[-1])
        
            # A loop is exploited heres so that the same column of B can be repeated twice
            # once for the maximum exponents phase (pre-requisite for splits phase) 
            # and second for the rest of the ozaki scheme's
            # phases
            for tb_col in range(ceildiv(N // n, tb_max_n_cols)):

                for pingpong in [0, 1]:
                    col_base = tb_col * tb_max_n_cols + pingpong * tb_max_n_cols // 2
                    current_tb_n_cols = min([tb_max_n_cols // 2, N // n - col_base])
        
                    if current_tb_n_cols<=0:
                        # At the very last iteration, we may not need a 'pong' iteration
                        break

                    tgs.append(rt.task_group())

                    rt.drain(outC.cons(), C, tap=C_tiles[c_index], task_group=tgs[-1], wait=True)
                    c_index += 1    

                    for tile_col in range(current_tb_n_cols):                   
                        rt.fill(inB.prod(), B, tap=B_Tiles[col_base + tile_col], task_group=tgs[-1])  
                        
                    if len(tgs) > 1:
                        rt.finish_task_group(tgs[-2])
                        del tgs[-2]
        
            if len(row_tgs)>1:
                rt.finish_task_group(row_tgs[-2])
                del row_tgs[-2]
        
        # finish the last ping pong task
        # for tg in tgs:
        rt.finish_task_group(tgs[-1])
        # for tg in row_tgs:
        rt.finish_task_group(row_tgs[-1])

    if generate_taps:
        # If generate taps is true, return a representation of tensor access patterns
        # representing all the npu_dma_memcpy_nd runtime sequence operations per input/ouput tensor.
        return (
            TensorAccessSequence.from_taps(A_taps),
            TensorAccessSequence.from_taps(B_taps),
            TensorAccessSequence.from_taps(C_taps),
        )

    # Create the program from the device type and runtime
    if dev == "npu":
        dev_ty = NPU1()
    else:
        dev_ty = NPU2()
    my_program = Program(dev_ty, rt)

    # Place components (assign them resources on the device) and generate an MLIR module
    module = my_program.resolve_program()
    return module


if __name__ == "__main__":
    main()
