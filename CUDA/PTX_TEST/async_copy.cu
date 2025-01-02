#include <stdio.h>
#include <cuda.h>

#define SIZE 64
#define COPY_LEN 16
#define NUM_THREADS 2

__global__ void copy_async_example(char *src, char *dst) {
    __shared__ char shared_mem[SIZE];

    int idx = threadIdx.x * COPY_LEN;
    
    unsigned int shared_ptr = __cvta_generic_to_shared(shared_mem + idx);
    char* target = src + idx;
    asm volatile ("cp.async.cg.shared.global [%0], [%1], %2;" 
                    :
                    : "r"(shared_ptr), "l"(target), "n"(COPY_LEN));
    asm volatile ("cp.async.commit_group;");
    asm volatile ("cp.async.wait_group 0;");

    int offset = COPY_LEN * blockDim.x;
    shared_ptr += offset;
    target += offset;
    asm volatile ("cp.async.cg.shared.global [%0], [%1], %2;" 
                    :
                    : "r"(shared_ptr), "l"(target), "n"(COPY_LEN));
    asm volatile ("cp.async.commit_group;");
    asm volatile ("cp.async.wait_group 0;");

    for(int i = 0; i < COPY_LEN; i++){
        int ii = idx + i;
        dst[ii] = shared_mem[ii];
    }

    for(int i = 0; i < COPY_LEN; i++){
        int ii = idx + i + offset;
        dst[ii] = shared_mem[ii];
    }
}


int main() {
    const int array_size = SIZE * sizeof(char);
    char *h_src = (char *)malloc(array_size);
    char *h_dst = (char *)malloc(array_size);

    // Initialize the source array with some values
    for (int i = 0; i < SIZE; i++) {
        h_src[i] = (i+1) % 256; // Limit values to fit in a char
    }

    char *d_src, *d_dst;
    cudaMalloc(&d_src, array_size);
    cudaMalloc(&d_dst, array_size);

    cudaMemcpy(d_src, h_src, array_size, cudaMemcpyHostToDevice);

    // Launch kernel with one block and SIZE threads
    copy_async_example<<<1, NUM_THREADS>>>(d_src, d_dst);

    cudaMemcpy(h_dst, d_dst, array_size, cudaMemcpyDeviceToHost);

    // Print the destination array
    for (int i = 0; i < SIZE; i++) {
        printf("h_dst[%d] = %d\n", i, h_dst[i]);
    }

    cudaFree(d_src);
    cudaFree(d_dst);
    free(h_src);
    free(h_dst);

    return 0;
}
