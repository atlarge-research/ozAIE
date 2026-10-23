#include "oz_aie.h"

 #include <vector>

 int main() {
    const int M = 256;
    const int N = 512;
    const int K = 128;

    // Setup for M x N x K multiplication under accuracy mode
    int setup_status = oz_aie::ozAIE_setup(M, N, K, oz_aie::Mode::accurate);

    if (setup_status>0) {
      return setup_status;
    }

    // Allocate and initialize matrices
    std::vector<double> A(M * K, 1.0);
    std::vector<double> B(K * N, 1.0);
    std::vector<double> C(M * N, 0.0);

    // Get pointers to the matrix data
    double* a_in = A.data();
    double* b_in = B.data();
    double* c_out = C.data();

    // Perform matrix multiplication: C = A * B
    return oz_aie::ozAIE_DGEMM(M, N, K, a_in, b_in, c_out);
 }