#include <iostream>
#include <cuda_fp16.h>
#include <mma.h>
#include <cassert>

using namespace nvcuda;
using namespace std;

static const int NUM_THREADS = 256;

//M,N must be multiples of BLOCK _M, BLOCK _N now
static const int M = 512, N = 768, K = 512;

static const int WMMA_M = 16;
static const int WMMA_K = 16;
static const int WMMA_N = 16;

static const int WARPS_M = 4; // tunable
static const int M_PER_WARPS = WMMA_M * 2; // 32
static const int BLOCK_M = M_PER_WARPS * WARPS_M;

static const int WARPS_N = 2; // tunable
static const int N_PER_WARPS = WMMA_N * 4; // 64
static const int BLOCK_N = N_PER_WARPS * WARPS_N;


__global__ void wmma_matmul_kernel(__half *A, __half *B, __half*C) {
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> c_frag;

    int warp_id = threadIdx.x / 32;
    int m_warp = warp_id / 4; // Warp group row index
    int n_warp = warp_id % 4; // Warp group column index

    int A_block_offset = blockIdx.x * (BLOCK_M * K);
    int A_warp_offset = m_warp * (WMMA_M * K);

    int B_block_offset = blockIdx.y * (BLOCK_N * K);
    int B_warp_offset = (n_warp * WMMA_N * K);

    for( int tile_m_idx = 0; tile_m_idx < WARPS_M; tile_m_idx++ ) {
        int A_tile_offset = tile_m_idx * (M_PER_WARPS * K);

        for( int tile_n_idx = 0; tile_n_idx < WARPS_N; tile_n_idx++ ) {
            __half *A_tile = A + A_block_offset + A_warp_offset + A_tile_offset;

            int B_tile_offset = tile_n_idx * (N_PER_WARPS * K);
            __half *B_tile = B + B_block_offset + B_warp_offset + B_tile_offset;
            wmma::fill_fragment(c_frag, 0.0f);

            for (int tile_idx = 0; tile_idx < K; tile_idx += WMMA_K) {
                wmma::load_matrix_sync(a_frag, A_tile, K);
                wmma::load_matrix_sync(b_frag, B_tile, K);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                A_tile += WMMA_K;
                B_tile += WMMA_K;
            }

            int C_block_row_offset = blockIdx.x * BLOCK_M * N;
            int C_warp_row_offset = m_warp * WMMA_M * N;
            int C_tile_row_offset = tile_m_idx * M_PER_WARPS * N;

            int C_block_col_offset = blockIdx.y * BLOCK_N;
            int c_warp_col_offset = n_warp * WMMA_N;
            int C_tile_col_offset = tile_n_idx * N_PER_WARPS;

            __half *C_tile = C + C_block_row_offset + C_block_col_offset +
                                C_warp_row_offset + c_warp_col_offset +
                                C_tile_row_offset + C_tile_col_offset;
            wmma::store_matrix_sync(C_tile, c_frag, N, wmma::mem_row_major);
        }
    }
}

void print_matrix(const __half *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            cout << __half2float(matrix[i * cols + j]) << " ";
        }
        cout << endl;
    }
}

void print_matrix(const float *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            cout << matrix[i * cols + j] << " ";
        }
        cout << endl;
    }
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

void fill_random_matrix(__half *matrix, int rows, int cols, int max_value = 3) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i * cols + j] = rand() % max_value;
        }
    }
}


int main() {
    __half *h_A = new __half[M * K];
    __half *h_B = new __half[K * N];
    float *h_C = new float[M * N];
    float *h_C_ref = new float[M * N];

    fill_random_matrix(h_A, M, K);
    fill_random_matrix(h_B, K, N);

    __half *transposed_B = new __half[K * N];
    transpose_matrix(h_B, transposed_B, K, N);
    
    __half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(__half));
    cudaMalloc(&d_B, K * N * sizeof(__half));
    cudaMalloc(&d_C, M * N * sizeof(__half));

    cudaMemcpy(d_A, h_A, M * K * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, transposed_B, K * N * sizeof(__half), cudaMemcpyHostToDevice);

    dim3 threads(NUM_THREADS, 1, 1);
    int GRID_M = M / BLOCK_M;
    int GRID_N = N / BLOCK_N;
    dim3 blocks(GRID_M, GRID_N, 1);
    cout << "GRID M : " << GRID_M << " GRID_N : " << GRID_N <<  endl;
    int CNT = 1;
    for(int i = 0; i < CNT; i++){
        wmma_matmul_kernel<<<blocks, threads>>>(d_A, d_B, d_C);
    }
    cudaDeviceSynchronize();

    __half *h_C_half = new __half[M * N];
    cudaMemcpy(h_C_half, d_C, M * N * sizeof(__half), cudaMemcpyDeviceToHost);
    for (int i = 0; i < M * N; i++) h_C[i] = __half2float(h_C_half[i]);

    bool cpu_test = true;
    if (cpu_test) {
        cout << "Matrix A:" << endl;
        print_matrix(h_A, M, K);

        cout << "\nMatrix B:" << endl;
        print_matrix(h_B, K, N);

        cpu_matmul(h_A, h_B, h_C_ref, M, N, K);

        cout << "\nMatrix C (GPU):" << endl;
        print_matrix(h_C, M, N);

        cout << "\nMatrix C (CPU Reference):" << endl;
        print_matrix(h_C_ref, M, N);

        for (int i = 0; i < M * N; i++) {
            assert(abs(h_C[i] - h_C_ref[i]) < 1e-2);
        }
        cout << "\nResults match!" << endl;
    }

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_ref;
    delete[] h_C_half;
    delete[] transposed_B;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
