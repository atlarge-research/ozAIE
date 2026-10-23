Oz-AIE

Oz-AIE is a reusable double-precision general matrix multiplication (DGEMM) library targeting AMD Ryzen processors with AI Engine (AIE) architecture, including XDNA1 and XDNA2 platforms.

This repository contains the library installation scripts, NPU instruction binaries, and experimental workflow used to reproduce the accuracy and performance results. The experiments measure numerical accuracy, execution time, and hardware cycle counts. MATLAB scripts are provided to generate plots from collected measurements; the experiment executable can also generate plots directly.

Compilation environment:	GCC 13.3.0, XRT 2.26.0, MLIR-AIE v1.3.2, MPFR 4.2.2, AOCL-BLIS 5.0.0
Experiment build environment:	Visual Studio Community 2022 (version 17.14 or later), .NET 8.0 SDK
Plot generation:	MATLAB R2023b
Runtime environment:	Windows 11 and Ubuntu 24.04.3 (Linux kernel 6.11 or later)
Metrics:	Numerical accuracy, execution time, and hardware cycles on AIE


Oz-AIE provides a reusable DGEMM library for AMD Ryzen processors equipped with XDNA1 or XDNA2 AI Engine architecture.

The repository includes an installation script that installs the library dependencies and headers.

A separate experimental workflow is available in the Experiments directory. It builds a measurement-collection executable, runs tests remotely on configured XDNA platforms, collects measurement data, and generates plots.

Hardware Requirements
An AMD Ryzen processor with AI Engine architecture, using either XDNA1 or XDNA2.
A supported Linux environment on the target platform.
A Windows machine for building and running the measurements and generating the plots. (experiments)

For examples of supported platforms, see the MLIR-AIE v1.3.2 programming guide.

Software Dependencies
Library and Target-Platform Dependencies
Dependency	Version	Purpose
GCC	13.3.0	C++ compilation
XRT	2.26.0	Runtime and driver support for the NPU
MLIR-AIE	v1.3.2	AIE compilation and binary generation
MPFR	4.2.2	Reference calculations for accuracy measurements
AOCL-BLIS	5.0.0	Reference implementation for performance measurements

Oz-AIE primarily depends on MLIR-AIE v1.3.2 and its associated dependencies.

The following additional requirements apply when running experiments:

Accuracy measurements: MPFR 4.2.2 must be installed on the target XDNA platform.
Performance measurements: AOCL-BLIS 5.0.0 must be installed and configured with OpenMP enabled.

The XRT driver must be installed separately. Refer to the MLIR-AIE v1.3.2 documentation for installation instructions.

Experiment and Plotting Dependencies
Windows 11
Visual Studio Community 2022, version 17.14 or later
.NET 8.0 SDK
MATLAB R2023b
Getting Started

Clone the repository:

git clone https://github.com/atlarge-research/ozAIE.git
cd ozAIE

Install the Oz-AIE Library

Run the installation script from the repository root:

./install_ozaie.sh

The installation script:

Installs MLIR-AIE v1.3.2 and its dependencies.
Generates the NPU instruction binaries for the default 
(𝐾,𝑙𝑚)
 configurations, covering the available accuracy- and performance-oriented modes.
Installs the library headers and configures the relevant environment variables.

Ensure that the XRT driver is installed separately before using the library.

Library Usage

After installation, include the Oz-AIE headers in your C++ project and use the library to perform DGEMM operations.
```C++
#include "oz_aie.h"

#include <vector>

int main() {
    const int M = 256;
    const int N = 512;
    const int K = 128;

    // Initialize the library for M x N x K multiplication under accuracy mode
    ozAIE_setup(M, N, K, oz_aie::Mode::accurate);   

    // Allocate and initialize matrices
    std::vector<double> A(M * K, 1.0);
    std::vector<double> B(K * N, 1.0);
    std::vector<double> C(M * N, 0.0);

    // Get pointers to the matrix data
    double* a_in = A.data();
    double* b_in = B.data();
    double* c_out = C.data();

    // Perform matrix multiplication: C = A * B
    ozAIE_DGEMM(M, N, K, a_in, b_in, c_out);

    return 0;
}
```

The following example selects the performance-oriented mode for a given matrix size:
```C
ozAIE_setup(M, N, K, oz_aie::Mode::fast);
```

Use the appropriate setup mode depending on whether numerical accuracy or execution performance is the primary objective. Refer to example/oztest.cpp for a complete usage example.

