#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2025 Advanced Micro Devices, Inc. or its affiliates
import argparse
import numpy as np
import math
import aie.extras.types as extratypes
import aie.extras.dialects.arith as arith
import aie.utils.trace.utils as trace_utils
import aie.utils.trace.events as trace_events

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker, Buffer, str_to_dtype
from aie.iron.device import NPU1Col1, NPU1Col2, NPU1, NPU2, Tile
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessSequence, TensorTiler2D
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
        prog="AIE Matrix Multiplication MLIR Design (Whole Array)",
        description="Emits MLIR code for a matrix multiplication design of the given input size",
    )
    argparser.add_argument("--dev", type=str, choices=["npu", "npu2"], default="npu")
    argparser.add_argument("-M", type=int, default=512)
    argparser.add_argument("-K", type=int, default=512)
    argparser.add_argument("-N", type=int, default=512)
    argparser.add_argument("-m", type=int, default=64)
    argparser.add_argument("-k", type=int, default=64)
    argparser.add_argument("-n", type=int, default=32)
    argparser.add_argument("--n-aie-cols", type=int, choices=[1, 2, 4, 8], default=4)
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
        args.n_aie_cols,
        args.dtype_in,
        args.dtype_out,
        args.b_col_maj,
        args.trace_size,
        args.generate_taps,
        args.full_dump,
        args.num_log_tiles
    )
    if args.generate_taps:
        return maybe_module
    else:
        print(maybe_module)


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
    n_aie_cols,
    dtype_in_str,
    dtype_out_str,
    b_col_maj,
    trace_size,
    generate_taps=False,
    full_dump=0,
    num_log_tiles=1
):
    n_aie_rows = 4
    n_aie_cores = n_aie_rows * n_aie_cols

    bps = np.int32(bps_str)

    # accumulation type
    dtype_out = str_to_dtype(dtype_out_str)

    if (dtype_in_str=="i4"):
        dtype_in = np.int8
    else: 
        dtype_in = str_to_dtype(dtype_in_str)
    vectorized = True
    enable_tracing = True if trace_size > 0 else False

    assert np.issubdtype(dtype_in, np.integer) == np.issubdtype(
        dtype_out, np.integer
    ), f"Input dtype ({dtype_in}) and output dtype ({dtype_out}) must either both be integral or both be float"
    assert (
        np.dtype(dtype_out).itemsize >= np.dtype(dtype_in).itemsize
    ), f"Output dtype ({dtype_out}) must be equal or larger to input dtype ({dtype_in})"

    # number of dump tiles to include per a C-tile output. Applied only when using full dumps (full_dump=1) for debugging
    dump_length_factor = (1 + (K // k) * (2 + num_splits) + num_splits + 1 + num_log_tiles)

    # r, s, t are the dimensions required by the microkernel MAC instructions.
    mac_dims = microkernel_mac_dim_map[dev][dtype_in_str]
    r, s, t = mac_dims

    # npu is a 4 row x 4 col array
    if dev == "npu" and n_aie_cols > 4:
        raise AssertionError("Invalid configuration: NPU (Phoenix/Hawk) has 4 columns")
    # npu2 is a 4 row x 8 col array
    if dev == "npu2" and n_aie_cols > 8:
        raise AssertionError(
            "Invalid configuration: NPU2 (Strix/Strix Halo/Krackan) has 8 columns"
        )

    # Input matrix A:
    # Conceptually, we divide input A into (m * n_rows, k)-sized blocks. These
    # blocks are _broadcast_ across AIE core columns, then _distributed_ across
    # rows, s.t. each of the n_rows compute cores in a column receives a
    # contiguous (m, k)-sized block of A.
    assert (
        M % (m * n_aie_rows) == 0
    ), """A must be tileable into (m * n_aie_rows, k)-sized blocks"""

    # Both A and B are tiled in the K dimension into size k.
    assert K % k == 0

    # Input matrix B:
    # Conceptually, we do the same as with A, but instead of broadcasting
    # across columns we broadcast across rows and distribute across columns.
    assert (
        N % (n * n_aie_cols) == 0
    ), """B must be tileable into (k, n * n_aie_cols)-sized blocks"""

    assert m % r == 0
    assert k % s == 0
    assert n % t == 0

    # If you get errors during CDO generation due to running out of program
    # memory, it may be because too much code is generated due to ObjectFIFO
    # loop unrollings. Reducing the depth to 1 here will work around that at
    # a big performance cost.
    fifo_depth = 2

    n_C_tiles = (M // m) * (N // n)
    n_tiles_per_core = n_C_tiles // n_aie_cores

    # When using more AIE columns than n_aie_rows (4) (applicable to NPU2),
    # restrict the number of shim/mem tiles to n_aie_rows,
    # since we have only n_aie_rows row tiles for matrix A
    if n_aie_cols > n_aie_rows:
        n_shim_mem_A = n_aie_rows
    # When using n_aie_rows (4) or less AIE columns (both NPU and NPU2),
    # the number of shim/mem tiles are equal to n_aie_cols.
    # We use the distribute pattern in object FIFO (see linking for A below),
    # since we have n_aie_rows (4) row tiles for matrix A
    else:
        n_shim_mem_A = n_aie_cols

    # Integer division when n_aie_cols < 4, otherwise set to 1
    n_A_tiles_per_shim = n_aie_rows // n_aie_cols if n_aie_cols < 4 else 1

    if dev == "npu":
        if n_aie_cols == 1:
            dev_ty = NPU1Col1()
        elif n_aie_cols == 2:
            dev_ty = NPU1Col2()
        elif n_aie_cols == 4:
            dev_ty = NPU1()
    else:
        dev_ty = NPU2()

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
    A_l2_ty = np.ndarray[(m * k * 2 * n_A_tiles_per_shim,), np.dtype[np.int32]]
    B_l2_ty = np.ndarray[(k * n * 2,), np.dtype[np.int32]]
    C_l2_ty = np.ndarray[(m * n * 2 * n_aie_rows,), np.dtype[np.int64]]
    A_l1_ty = np.ndarray[(m, k * 2), np.dtype[np.int32]]
    B_l1_ty = np.ndarray[(k, n * 2), np.dtype[np.int32]]
    C_l1_ty = np.ndarray[(m, n*2), np.dtype[np.int64]]
    C_bins_ty = np.ndarray[(m * n * num_splits,), np.dtype[dtype_out]]

    a_max_ty = np.ndarray[(m,), np.dtype[np.int16]]
    b_max_ty = np.ndarray[(n,), np.dtype[np.int16]]
    splits_ty_a = np.ndarray[(num_splits, m, k // (2 if dtype_in_str=="i4" else 1)), np.dtype[dtype_in]]
    splits_ty_b = np.ndarray[(num_splits, k, n // (2 if dtype_in_str=="i4" else 1)), np.dtype[dtype_in]]

    # only for full dumps for debugging
    log_ty = np.ndarray[(((m*num_log_tiles) if full_dump else 1), (n * 2) if full_dump else 16), np.dtype[np.int64]]
    kernel_file = f"mm_ozaki_{m}x{k}x{n}_b{bps}_d{num_splits}.o"

    # AIE Core Function declarations
    # AIE Core Function declarations
    func_type = "" 
    zero_kernels = {
        t_in : Kernel(
            f"zero_n_times_{func_type}{t_in}", 
            kernel_file, [ty, np.int32]
        )
        for t_in, ty in [(dtype_out_str, C_bins_ty),("i64", C_l1_ty)]
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
        [splits_ty_a, splits_ty_b, C_bins_ty, log_ty]
    )
    matmul_sum_vectorized_func_name = f"ozaki_sum_vectorized_upshift_{dtype_out_str}"
    sum_kernel = Kernel(
        matmul_sum_vectorized_func_name,
        kernel_file,
        [C_bins_ty, C_l1_ty, a_max_ty, b_max_ty, log_ty],
    )

    max_exp_kernels = {
        mat : Kernel(
            f"ozaki_get_max_exponent_{mat}_inner_tiled_wise",
            kernel_file,
            [exp_ty, max_ty, log_ty],
        )
        for mat, exp_ty, max_ty in (("A", A_l1_ty, a_max_ty), ("B", B_l1_ty, b_max_ty))
    }

    matmul_split_vectorized_func_name = f"ozaki_split_vectorized_revised_{{mat}}_{dtype_in_str}"
    split_kernels = {
        mat : Kernel(
            matmul_split_vectorized_func_name.format(mat=mat),
            kernel_file,
            [ty, splits_ty, max_ty,np.int32, np.int32, log_ty]
        ) for mat, ty, splits_ty, max_ty in (("A", A_l1_ty, splits_ty_a, a_max_ty), ("B", B_l1_ty, splits_ty_a, b_max_ty))
    }

    cBinsBuffers = []
    splitsBuffersA = []
    splitsBuffersB = []
    maxBuffersA = []
    maxBuffersB = []
    logBuffers = []
    
    for i in range(n_aie_cores):
        cBinsBuffers.append(Buffer(C_bins_ty, initial_value=np.array([0]*(m*n*num_splits), dtype=dtype_out)))
        splitsBuffersA.append(Buffer(splits_ty_a, initial_value=np.array([0]*(m*k*num_splits), dtype=dtype_in)))
        splitsBuffersB.append(Buffer(splits_ty_b, initial_value=np.array([0]*(k*n*num_splits), dtype=dtype_in)))
        maxBuffersA.append(Buffer(a_max_ty, initial_value=np.array([0]*m, dtype=np.int16)))
        maxBuffersB.append(Buffer(b_max_ty, initial_value=np.array([0]*n, dtype=np.int16)))
        logBuffers.append(Buffer(log_ty, initial_value=np.array([0 for i in range((m*num_log_tiles*n*2) if full_dump else 1*16)], dtype=np.int64)))

    # Tile declarations as tile[row][col]
    tiles = [[(col, row) for col in range(0, n_aie_cols)] for row in range(0, 6)]
    core_tiles = tiles[2:]

    # AIE-array data movement with object fifos
    A_l3l2_fifos = [None] * n_shim_mem_A
    A_l2l1_fifos = [None] * n_aie_rows

    B_l3l2_fifos = [None] * n_aie_cols
    B_l2l1_fifos = [None] * n_aie_cols

    C_l1l2_fifos = [[None] * n_aie_cols for _ in range(n_aie_rows)]
    C_l2l3_fifos = [None] * n_aie_cols

    # Input A
    for i in range(n_shim_mem_A):
        A_l3l2_fifos[i] = ObjectFifo(A_l2_ty, name=f"A_L3L2_{i}", depth=fifo_depth)
        # If n_shim_mem_A == n_rows, n_A_tiles_per_shim is 1 and
        # this simply links a_l3l2_fifos[i] to a_l2l1_fifos[i] directly,
        # If n_shim_mem_A < n_rows, each column receives multiple rows of
        # tiles; distribute it along rows of AIE cores.
        start_row = i * n_A_tiles_per_shim
        stop_row = start_row + n_A_tiles_per_shim
        of_offsets = [m * k * 2 * j for j in range(stop_row - start_row)]
        a_dims_to_stream = [
            [
                (m // r, r * (k * 2)),
                (k // s, s * 2),
                (r, k * 2),
                (s * 2, 1),
            ]
        ] * (stop_row - start_row)
        a_dims_from_stream = [
            [
                 (m * k, 1),
                 (2, m * k),
                (1, 1)
            ]
        ] * (stop_row - start_row)
        a_tmp_fifos = (
            A_l3l2_fifos[i]
            .cons()
            .split(
                of_offsets,
                obj_types=[A_l1_ty] * (stop_row - start_row),
                names=[f"A_L2L1_{row}" for row in range(start_row, stop_row)],
                dims_to_stream=a_dims_to_stream,
                dims_from_stream=a_dims_from_stream
            )
        )

        for j in range(stop_row - start_row):
            A_l2l1_fifos[j + start_row] = a_tmp_fifos[j]

    # Input B
    for col in range(n_aie_cols):
        B_l3l2_fifos[col] = ObjectFifo(B_l2_ty, name=f"B_L3L2_{col}", depth=fifo_depth)
        if b_col_maj:
            b_dims_to_stream = [(n // t, t * k * 2), (k // s, s * 2), (t, k * 2), (s * 2, 1)]
            b_dims_from_stream = [(n * k, 1), (2, n * k), (1, 1)]
        else:
            b_dims_to_stream = [(n // t, t * k * 2), (k * 2  // s, s), (t, k*2), (s, 1)]
            b_dims_from_stream = [ (n // t, k * t), (2, n * k), (k * t, 1) ]
        B_l2l1_fifos[col] = (
            B_l3l2_fifos[col]
            .cons()
            .forward(
                obj_type=B_l1_ty,
                name=f"B_L2L1_{col}",
                dims_to_stream=b_dims_to_stream,
                dims_from_stream=b_dims_from_stream 
            )
        )

        # Output C
        C_l2l3_fifos[col] = ObjectFifo(
            C_l2_ty,
            name=f"C_L2L3_{col}",
            depth=fifo_depth,
            dims_to_stream=[(m // r, r * n * 2), (r, t*2), (n // t, r * t * 2), (t*2, 1)],
        )
        of_offsets = [m * n * 2 * i for i in range(n_aie_rows)]

        # join along one column
        c_tmp_fifos = (
            C_l2l3_fifos[col]
            .prod()
            .join(
                of_offsets,
                obj_types=[C_l1_ty] * n_aie_rows,
                names=[f"C_L1L2_{col}_{row}" for row in range(n_aie_rows)],
                depths=[fifo_depth] * n_aie_rows,
            )
        )
        for j in range(n_aie_rows):
            C_l1l2_fifos[j][col] = c_tmp_fifos[j]

    # Tasks for each worker to perform
    def core_fn(
        in_a,
        in_b,
        out_c, 
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
        loop = range(1)  # Workaround for issue #1547
        if n_tiles_per_core > 1:
            loop = range_(n_tiles_per_core)
        for t in loop:
            zero_cbins(c_bins_buff, num_splits)

            # max exponent of A's rows from a row of A tiles needs to be refreshed once for every row of C tiles produced as the row 
            # of A tiles used is the same
            zero = arith.constant(0, type=extratypes.index())
            r = remsi(t, constant((N // n // n_aie_cols), extratypes.index()))
            is_max_exp_a_refresh = arith.cmpi("eq", r, zero)      
            with if_(is_max_exp_a_refresh) as if_op1:
                zero_max_buffA(max_buff_A)
                for _ in range_(K // k) if (K // k) > 1 else range(1):
                    elem_in_a_exp = in_a.acquire(1)
                    max_expA(elem_in_a_exp, max_buff_A, log_buff)
                    in_a.release(1)

            # max exponent of B's columns needs to be refreshed for every C tile produced as columns of B change
            # every tile produced
            zero_max_buffB(max_buff_B)
            for _ in range_(K // k) if (K // k) > 1 else range(1):
                elem_in_b_exp = in_b.acquire(1)
                max_expB(elem_in_b_exp, max_buff_B, log_buff)
                in_b.release(1)

             # output tile component for max buff B (FOR DUMP ONLY)
            if (full_dump):
                elem_out = out_c.acquire(1)
                for i in range_(m): 
                    ic = index_cast(i, extratypes.i32()) 
                    elem_out[0, i] = arith.extsi(in_=max_buff_A[ic], out=extratypes.i64())
                    elem_out[3, i] = arith.extsi(in_=max_buff_B[ic], out=extratypes.i64())
                out_c.release(1)

            for _ in range_(K // k):
                elem_in_a = in_a.acquire(1)
                elem_in_b = in_b.acquire(1)

                # output tile components for A- and B-tiles (FOR DUMP ONLY)
                if (full_dump):
                    elem_outs = out_c.acquire(2)
                    zero_c_out(elem_outs[0], 2)
                    zero_c_out(elem_outs[1], 2)
                    for i in range_(k):
                        for j in range_(n*2):
                            ic = index_cast(i, extratypes.i32())
                            jc = index_cast(j, extratypes.i32())
                            (elem_outs[0])[i, j] = arith.extsi(in_=elem_in_a[ic, jc], out=extratypes.i64())
                            (elem_outs[1])[i, j] = arith.extsi(in_=elem_in_b[ic, jc], out=extratypes.i64())
                    out_c.release(2)

                split_A(elem_in_a, split_buff_a, max_buff_A, t, 0, log_buff)
                split_B(elem_in_b, split_buff_b, max_buff_B, t, 0, log_buff)
                in_a.release(1)
                in_b.release(1)

                # output tile components for splits of A and B (FOR DUMP ONLY)
                if (full_dump):
                    for split_no in range_(num_splits):
                        sn_jc = index_cast(split_no, extratypes.i32())
                        elem_out = out_c.acquire(1)
                        zero_c_out(elem_out, 2)
                        for i in range_(k):
                            for j in range_(n // (2 if dtype_in_str=="i4" else 1)):
                                ic = index_cast(i, extratypes.i32())
                                jc = index_cast(j, extratypes.i32())
                                elem_out[i, j] = arith.extsi(in_=split_buff_a[sn_jc, ic, jc], out=extratypes.i64())
                                elem_out[i, n+j] = arith.extsi(in_=split_buff_b[sn_jc, ic, jc], out=extratypes.i64())
                        out_c.release(1)

                matmul(split_buff_a, split_buff_b, c_bins_buff, log_buff)

             # output tile component for bins (FOR DUMP ONLY)
            if (full_dump):
                for split_no in range_(num_splits):
                    sn_jc = index_cast(split_no, extratypes.i32())
                    elem_out = out_c.acquire(1)
                    zero_c_out(elem_out, 2)
                    for i in range_(m):
                        for j in range_(n):
                            ic = index_cast(i, extratypes.i32())
                            jc = index_cast(j, extratypes.i32())
                            elem_out[i, j] = arith.extsi(in_=c_bins_buff[sn_jc, ic, jc], out=extratypes.i64())
                    out_c.release(1)

            elem_out = out_c.acquire(1)
            zero_c_out(elem_out, 2)
            sum(c_bins_buff, elem_out, max_buff_A, max_buff_B, log_buff)
            out_c.release(1)

            # additional log buffer tile to log things from SIMD kernel for debugging (FOR DUMP ONLY)
            if (full_dump):
                for l_idx in range_(num_log_tiles):
                    elem_out = out_c.acquire(1)
                    zero_c_out(elem_out, 2)
                    for i in range_(m):
                        for j in range_(n*2):
                            elem_out[i, j] = log_buff[m*l_idx+i,j]
                    out_c.release(1)

    # Set up compute tiles
    workers = []
    for row in range(n_aie_rows):
        for col in range(n_aie_cols):
            tile_col, tile_row = core_tiles[row][col]
            workers.append(
                Worker(
                    core_fn,
                    [
                        A_l2l1_fifos[row].cons(),
                        B_l2l1_fifos[col].cons(),
                        C_l1l2_fifos[row][col].prod(),
                        maxBuffersA[row*n_aie_cols+col], 
                        maxBuffersB[row*n_aie_cols+col], 
                        splitsBuffersA[row*n_aie_cols+col], 
                        splitsBuffersB[row*n_aie_cols+col], 
                        cBinsBuffers[row*n_aie_cols+col], 
                        logBuffers[row*n_aie_cols+col],
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
                    stack_size=3072
                )
            )

    # We are limited in the number of BDs. After synchronizing, we can reuse BDs.
    # We only transfer 6 rows of tiles at once before starting a new transfer block.
    # tb = transfer block; block of transfers before sync call
    tb_max_n_rows = 4
    tb_n_rows = tb_max_n_rows // 2

    tb_max_n_cols = 4
    tb_n_cols = (tb_max_n_cols // 2) 

    if (N // n // n_aie_cols) < tb_n_cols:
        tb_n_cols = (N // n // n_aie_cols)

    # Define tensor access patterns (tiling) for A, B, and C
    A_tiles = TensorTiler2D.group_tiler(
        (M, K),  # Size of A matrix
        (m * n_A_tiles_per_shim, k),  # Size of A (smallest) tile
        (1, K // k),  # Size of "group" of tiles
        # Repeat data so can distribute across whole column
        pattern_repeat=N // n // n_aie_cols + 1,
        # pattern_repeat=tb_n_cols,
        prune_step=False,
    )
    if b_col_maj:
        B_tiles = TensorTiler2D.step_tiler(
            (N, K * 2),  # Size of B matrix
            (n, k * 2),  # Size of B tile
            # Number of tiles per transfer in each dimension (whole col, partial row)
            tile_group_repeats=(1, K // k),
            tile_group_steps=(1, 1),
            prune_step=False,
            pattern_repeat=2
        )
    else:
        B_tiles = TensorTiler2D.step_tiler(
            (K, N * 2),  # Size of B matrix
            (k, n * 2),  # Size of B tile
            # Number of tiles per transfer in each dimension (whole col, partial row)
            # use a loop to send steps of columns as we need to repeat every column twice before sending the
            # nest
            tile_group_repeats=(K // k, 1),
            # Contiguous tile group in col, but send every n_aie_cols-th tile in the row
            # use a loop to send steps of columns as we need to repeat every column twice before sending the
            # nest
            tile_group_steps=(1, 1),
            tile_col_major=True,
            tile_group_col_major=True,  # Send all tiles in column before moving on to next column
            prune_step=False,
            pattern_repeat=2
        )

    C_tiles = TensorTiler2D.step_tiler(
        (n_C_tiles * m * dump_length_factor, n * 2) if full_dump else (M, N*2), # For full sequential dump
        (m, n * 2),  # Size of C tile
        # Number of tiles per transfer in each dimension (partial col, partial row)
        tile_group_repeats=( (dump_length_factor * n_aie_rows * tb_n_cols, 1) if full_dump else (n_aie_rows, tb_n_cols) ),

        # Collect every n_aie_cols row at a time (mirroring how we sent in B data)
        tile_group_steps=( (1, 1) if full_dump else (1, n_aie_cols) ),
        tile_group_col_major=True,
        prune_step=False
    )
    c_index = 0

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(A_ty, B_ty, C_ty) as (A, B, C):
        # rt.enable_trace(
        # trace_size = trace_size,
        # workers=[workers[0]]
        # # ddr_id = 4,
        # # coretile_events = [
        # #          trace_events.CoreEvent.INSTR_EVENT_0,
        # #          trace_events.CoreEvent.INSTR_EVENT_1]
        # #         trace_events.CoreEvent.INSTR_VECTOR,
        # #         trace_events.CoreEvent.MEMORY_STALL,
        # #         trace_events.CoreEvent.STREAM_STALL,
        # #         trace_events.CoreEvent.LOCK_STALL,
        # #         trace_events.CoreEvent.ACTIVE,
        # #         trace_events.CoreEvent.DISABLED]
        # )

        rt.start(*workers)

        # Task groups will be used to determine when to sync/await/free DMA runtime ops
        tgs = []
        row_tgs = []
        for row in range(M // m // n_aie_rows):

            row_tgs.append(rt.task_group())

            for col in range(n_shim_mem_A):

                tile_offset = (
                    row * n_shim_mem_A + col
                ) % len(A_tiles)
             
                rt.fill(
                    A_l3l2_fifos[col].prod(),
                    A,
                    tap=A_tiles[tile_offset],
                    task_group=row_tgs[-1],
                )

            # A loop is exploited heres so that the same column of B can be repeated twice
            # once for the maximum exponents phase (pre-requisite for splits phase) 
            # and second for the rest of the ozaki scheme's
            # phases
            for tb_col in range(ceildiv(N // n // n_aie_cols, tb_max_n_cols)):

                for pingpong in [0, 1]:
                    col_base = tb_col * tb_max_n_cols + pingpong * tb_max_n_cols // 2
                    current_tb_n_cols = min(
                        [tb_max_n_cols // 2, N // n // n_aie_cols - col_base]
                    )

                    if current_tb_n_cols<=0:
                        # At the very last iteration, we may not need a 'pong' iteration
                        break

                    tgs.append(rt.task_group())

                    for col in range(n_aie_cols):
                        # B input transfer:
                        # Transfer the first a (n)-wide block of columns of B,
                        # Then transfer the (n_aie_columns)-th such block, and so on.
                        # Each shim will start at a different column offset.
                        # For example, shim 0 may transfer the tiles marked 0 below,
                        # and shim 1 may transfer the tiles marked 1.
                        #
                        #             N
                        #      ----------------
                        #     |0011    0011    |
                        #     |0011    0011    |
                        #     |0011    0011    |
                        # K   |0011    0011    |
                        #     |0011    0011    |
                        #     |0011    0011    |
                        #     |0011    0011    |
                        #     |0011    0011    |
                        #      ----------------
                        rt.drain(
                            C_l2l3_fifos[col].cons(),
                            C,
                            tap=C_tiles[c_index],
                            wait=True,
                            task_group=tgs[-1],
                        )    
                        c_index += 1                 

                        for tile_col in range(current_tb_n_cols):

                            rt.fill(
                                B_l3l2_fifos[col].prod(),
                                B,
                                tap=B_tiles[((col_base + tile_col) * n_aie_cols + col) % len(B_tiles)],
                                task_group=tgs[-1]
                            )

                    # if row > 0 or tb_col > 0 or (tb_col == 0 and pingpong > 0):
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
    my_program = Program(dev_ty, rt)

    # Place components (assign them resources on the device) and generate an MLIR module
    module = my_program.resolve_program()
    return module


if __name__ == "__main__":
    main()
