Oz-AIE

Oz-AIE is a reusable double-precision general matrix multiplication (DGEMM) emulation library based on Ozaki scheme for XDNA1 and XDNA2 architecture AMD Ryzen NPU architectures  XDNA1 and XDNA2.

This repository contains the library installation scripts, NPU instruction binaries, and experimental workflow used to reproduce the accuracy and performance results. The experiments measure numerical accuracy, execution time, and hardware cycle counts. MATLAB scripts are provided to generate plots from collected measurements; the experiment executable can also generate plots directly.

| Category | Details |
|---|---|
| Compilation Environment | GCC 13.3.0, XRT 2.26.0, MLIR-AIE v1.3.2, MPFR 4.2.2, AOCL-BLIS 5.0.0 |
| Experiment Build Environment | Visual Studio Community 2022 (version 17.14 or later), .NET 8.0 SDK |
| Plot Generation | MATLAB R2023b |
| Runtime Environment | Windows 11 and Ubuntu 24.04.3 (Linux kernel 6.11 or later) |
| Metrics | Numerical accuracy, execution time, and hardware cycles on AIE |


Oz-AIE provides a reusable DGEMM library for AMD Ryzen processors equipped with XDNA1 or XDNA2 AI Engine architecture.

The repository includes an installation script that installs the library dependencies and headers.

A separate experimental workflow is available in the Experiments directory. It builds a measurement-collection executable, runs tests remotely on configured XDNA platforms, collects measurement data, and generates plots.

Hardware Requirements
An AMD Ryzen processor with AI Engine architecture, using either XDNA1 or XDNA2.
A supported Linux environment on the target platform.
A Windows machine for building and running the measurements and generating the plots. (experiments)

For examples of supported platforms, see the MLIR-AIE v1.3.2 programming guide.

### Library and Target-Platform Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| GCC | 13.3.0 | C++ compilation |
| XRT | 2.26.0 | Runtime and driver support for the Ryzen NPU |
| MLIR-AIE | v1.3.2 | Compilation for binaries for Ryzen NPUs |
| MPFR | 4.2.2 | High-precision Emulated DGEMM Baseline  for accuracy comparison |
| AOCL-BLIS | 5.0.0 | CPU Baseline for performance comparison |


Oz-AIE primarily depends on MLIR-AIE v1.3.2 and its associated dependencies.

The following additional requirements apply when running experiments (see README under Experiments folder)  but are not needed for the library usage itself:

Accuracy measurements: MPFR 4.2.2 must be installed on the target XDNA platform.
Performance measurements: AOCL-BLIS 5.0.0 must be installed and configured with OpenMP enabled.

The Xilinx XRT driver must be installed separately. Refer to the MLIR-AIE v1.3.2 documentation for installation instructions https://github.com/Xilinx/mlir-aie/tree/v1.3.2.

### Install the Oz-AIE Library

Clone the repository:

```bash
git clone https://github.com/atlarge-research/ozAIE.git
cd ozAIE
```

Ensure that the XRT driver (see above) is installed separately before using the installation script.

Run the installation script at the repository root:

```bash
./install_ozaie.sh
```

The installation script:

Installs MLIR-AIE v1.3.2 and its dependencies. Installs the library headers and configures the relevant environment variables.

Library Usage

After installation, include the Oz-AIE header in your C++ source and use the library for DGEMM. A minimum example is given under the "example" directory. The CMakeLists.txt under this folder shows how to refer to the headers and link to the CMake compilation hierarchy of the library to compile the client-side cpp. The necessary path environment variables are created by the above-mentioned installation script. 
```C++
#include "oz_aie.h"

#include <vector>

int main() {
    const int M = 256;
    const int N = 512;
    const int K = 128;

    // Initialize the library for M x N x K multiplication under accuracy mode
    int status = ozAIE_setup(M, N, K, oz_aie::Mode::accurate);  
    
    if (status>0) return 1;   // error in setup . error description goes to stderr

    // Allocate and initialize matrices
    std::vector<double> A(M * K, 1.0);
    std::vector<double> B(K * N, 1.0);
    std::vector<double> C(M * N, 0.0);

    // Get pointers to the matrix data
    double* a_in = A.data();
    double* b_in = B.data();
    double* c_out = C.data();

    // Perform matrix multiplication: C = A * B
    int dgemm_status = ozAIE_DGEMM(M, N, K, a_in, b_in, c_out);

    return dgemm_status;
}
```

The following example selects the performance-oriented mode for a given matrix size:
```C++
ozAIE_setup(M, N, K, oz_aie::Mode::fast);
```

Use the appropriate setup mode depending on whether numerical accuracy or execution performance is the primary objective. Refer to example/oztest.cpp for a complete usage example.
