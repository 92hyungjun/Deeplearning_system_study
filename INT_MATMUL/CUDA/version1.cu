#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <cstring>

// Matrix dimensions
const int M = 8;  // Rows of A and D
const int N = 8;  // Columns of B and D
const int K = 32; // Columns of A and rows of B

// Kernel to perform GEMM using Tensor Core MMA
__global__ void mma_test_u4(int *A, int *B, int *C, int *output) {
    int accumulate[2] = {0, 0};
    int frag_A = A[threadIdx.x];
    int frag_B = B[threadIdx.x];
    int frag_C[2] = {0, 0}; // C is always initialized to 0

    // Perform MMA operation using inline assembly
    asm volatile(
        "mma.sync.aligned.m8n8k32.row.col.s32.u4.u4.s32 "
        "{%0, %1}, {%2}, {%3}, {%4, %5};\n"
        : "=r"(accumulate[0]), "=r"(accumulate[1])
        : "r"(frag_A), "r"(frag_B), 
          "r"(frag_C[0]), "r"(frag_C[1])
    );

    // Store the result in the output array
    output[threadIdx.x * 2] = accumulate[0];
    output[threadIdx.x * 2 + 1] = accumulate[1];
}

// Reference CPU kernel to compute GEMM
void reference_cpu(int *A, int *B, int *C, int *output) {
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            int acc = 0;
            for (int i = 0; i < K; i++) {
                acc += A[row * K + i] * B[i * N + col];
            }
            output[row * N + col] = acc; // C is always 0
        }
    }
}

// Utility function to print a matrix
void print_matrix(const std::string &description, int *arr, int rows, int cols) {
    std::cout << "----- " << description << " -----" << '\n';
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << std::setw(4) << arr[i * cols + j] << ' ';
        }
        std::cout << '\n';
    }
}

// Utility function to fill a matrix with random values in the range [0, 3]
void fill_random_matrix(int *arr, int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = rand() % 4; // Random values between 0 and 3
    }
}

// Utility function to fill a matrix with zeros
void fill_zeros(int *arr, int size) {
    std::memset(arr, 0, size * sizeof(int));
}

// Utility function to transpose a matrix
void transpose_matrix(int *src, int *dest, int rows, int cols) {
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            dest[col * rows + row] = src[row * cols + col];
        }
    }
}

// Utility function to pack a matrix into 4-bit compressed form
void pack_matrix(int *src, int *dest, int rows, int cols) {
    int packed_idx = 0;
    for (int i = 0; i < rows * cols; i += 8) {
        dest[packed_idx++] = (src[i] & 0xF) |
                             ((src[i + 1] & 0xF) << 4) |
                             ((src[i + 2] & 0xF) << 8) |
                             ((src[i + 3] & 0xF) << 12) |
                             ((src[i + 4] & 0xF) << 16) |
                             ((src[i + 5] & 0xF) << 20) |
                             ((src[i + 6] & 0xF) << 24) |
                             ((src[i + 7] & 0xF) << 28);
    }
}

// Function to initialize input matrices
void initialize_inputs(int *A, int *B, int *C) {
    fill_random_matrix(A, M * K); // Fill A with random values
    fill_random_matrix(B, K * N); // Fill B with random values
    fill_zeros(C, M * N);         // Fill C with zeros
}

int main() {
    srand(time(NULL));

    // Allocate memory for inputs and outputs
    int *input_A = new int[M * K];
    int *input_B = new int[K * N];
    int *input_C = new int[M * N];

    // Initialize inputs
    initialize_inputs(input_A, input_B, input_C);
    // Print input and packed matrices
    print_matrix("Matrix A (Unpacked)", input_A, M, K);
    print_matrix("Matrix B (Unpacked)", input_B, K, N);
    print_matrix("Matrix C (Unpacked)", input_C, M, N);

    // Transpose and pack matrices
    int *transposed_B = new int[K * N];
    transpose_matrix(input_B, transposed_B, K, N);

    int *packed_A = new int[(M * K) / 8];
    int *packed_B = new int[(K * N) / 8];
    pack_matrix(input_A, packed_A, M, K);
    pack_matrix(transposed_B, packed_B, K, N);

    // Allocate GPU memory
    int *d_A, *d_B, *d_C, *d_D;
    cudaMalloc(&d_A, (M * K) / 8 * sizeof(int));
    cudaMalloc(&d_B, (K * N) / 8 * sizeof(int));
    cudaMalloc(&d_C, M * N * sizeof(int));
    cudaMalloc(&d_D, M * N * sizeof(int));

    // Copy data to GPU
    cudaMemcpy(d_A, packed_A, (M * K) / 8 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, packed_B, (K * N) / 8 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, input_C, M * N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch MMA kernel
    mma_test_u4<<<1, 32>>>(d_A, d_B, d_C, d_D);

    // Wait for device to complete
    cudaDeviceSynchronize();

    // Copy results back to CPU
    int *output_D = new int[M * N];
    cudaMemcpy(output_D, d_D, M * N * sizeof(int), cudaMemcpyDeviceToHost);

    // Run reference CPU kernel
    int *output_D_ref = new int[M * N];
    reference_cpu(input_A, input_B, input_C, output_D_ref);

    // Print output matrices
    print_matrix("Matrix D (MMA Result)", output_D, M, N);
    print_matrix("Matrix D_ref (CPU Reference)", output_D_ref, M, N);

    // Validate results
    bool valid = true;
    for (int i = 0; i < M * N; i++) {
        if (output_D[i] != output_D_ref[i]) {
            std::cout << "Mismatch at index " << i << ": GPU " << output_D[i] << " != CPU " << output_D_ref[i] << '\n';
            valid = false;
        }
    }
    std::cout << (valid ? "PASS" : "FAIL") << '\n';

    // Free memory
    delete[] input_A;
    delete[] input_B;
    delete[] input_C;
    delete[] transposed_B;
    delete[] packed_A;
    delete[] packed_B;
    delete[] output_D;
    delete[] output_D_ref;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);

    return 0;
}
