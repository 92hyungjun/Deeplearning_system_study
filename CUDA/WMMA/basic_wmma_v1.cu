#include <iostream>
#include <cuda_fp16.h>
#include <mma.h>
#include <cassert>

using namespace nvcuda;
using namespace std;

static const int M = 128, N = 128, K = 48;
static const int BLOCK_M = 16;
static const int BLOCK_N = 16;

static const int WMMA_M = 16;
static const int WMMA_K = 16;
static const int WMMA_N = 16;

__global__ void wmma_matmul_kernel(__half *A, __half *B, __half*C) {
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    int a_offset = blockIdx.x * (BLOCK_M * K);
    __half *A_tile = A + a_offset;

    int b_offset = blockIdx.y * (BLOCK_N * K);
    __half *B_tile = B + b_offset;    

    for (int tile_idx = 0; tile_idx < K; tile_idx += WMMA_K) {
        wmma::load_matrix_sync(a_frag, A_tile, K);
        wmma::load_matrix_sync(b_frag, B_tile, K);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        A_tile += WMMA_K;
        B_tile += WMMA_K;
    }

    int c_row_offset = blockIdx.x * BLOCK_M * N;
    int c_col_offset = blockIdx.y * BLOCK_N;
    
    __half *C_tile = C + c_row_offset + c_col_offset;
    wmma::store_matrix_sync(C_tile, c_frag, N, wmma::mem_row_major);
}

void cpu_matmul(const __half *A, const __half *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += __half2float(A[i * K + k]) * __half2float(B[k * N + j]);
            }
            C[i * N + j] = sum;
        }
    }
}

void transpose_matrix(__half *src, __half *dest, int rows, int cols) {
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            dest[col * rows + row] = src[row * cols + col];
        }
    }
}


int main() {
    __half h_A[M * K], h_B[K * N];
    float h_C[M * N], h_C_ref[M * N];

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            h_A[i * K + j] = rand() % 3;
            //h_A[i * K + j] =  i * K + j;
        }
    }

    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            h_B[i * N + j] = rand() % 3;
            //h_B[i * N + j] = i * N + j;
        }
    }

    cout << "Matrix A:" << endl;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            cout << __half2float(h_A[i * K + j]) << " ";
        }
        cout << endl;
    }

    cout << "\nMatrix B:" << endl;
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            cout << __half2float(h_B[i * N + j]) << " ";
        }
        cout << endl;
    }

    __half transposed_B[K * N];
    transpose_matrix(h_B, transposed_B, K, N);

    __half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(__half));
    cudaMalloc(&d_B, K * N * sizeof(__half));
    cudaMalloc(&d_C, M * N * sizeof(__half));

    // Copy host matrices to device
    cudaMemcpy(d_A, h_A, M * K * sizeof(__half), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_B, h_B, K * N * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, transposed_B, K * N * sizeof(__half), cudaMemcpyHostToDevice);

    dim3 threads(32, 1, 1);
    //M,N
    int GRID_M = M / BLOCK_M;
    int GRID_N = N / BLOCK_N;
    dim3 blocks(GRID_M, GRID_N, 1);
    wmma_matmul_kernel<<<blocks, threads>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    __half h_C_half[M * N];
    cudaMemcpy(h_C_half, d_C, M * N * sizeof(__half), cudaMemcpyDeviceToHost);
    for (int i = 0; i < M * N; i++) h_C[i] = __half2float(h_C_half[i]);

    cpu_matmul(h_A, h_B, h_C_ref, M, N, K);

    cout << "\nMatrix C (GPU):" << endl;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            cout << h_C[i * N + j] << " ";
        }
        cout << endl;
    }

    cout << "\nMatrix C (CPU Reference):" << endl;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            cout << h_C_ref[i * N + j] << " ";
        }
        cout << endl;
    }

    // Verify correctness
    for (int i = 0; i < M * N; i++) {
        assert(abs(h_C[i] - h_C_ref[i]) < 1e-2);
    }

    cout << "\nResults match!" << endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
