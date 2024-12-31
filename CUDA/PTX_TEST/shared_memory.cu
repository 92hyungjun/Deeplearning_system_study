#include <stdio.h>
#include <stdint.h>

__device__ uint32_t cast_smem_ptr_to_uint(void const* const ptr) {
    uint32_t smem_ptr;
    asm(
        "{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }\n"
        : "=r"(smem_ptr) : "l"(ptr));
    return smem_ptr;
}

// CUDA kernel using PTX assembly to store and load from shared memory
__global__ void sharedMemoryKernel(int *input, int *output) {
    // Declare shared memory array with volatile keyword to prevent optimization
    __shared__ int sharedMem[32];

    // Each thread gets its thread ID
    int tid = threadIdx.x;

    // Convert shared memory pointer to 32-bit shared pointer
    //https://github.com/NVIDIA/cutlass/blob/bf9da7b76c766d7ee7d536afc77880a4ef1f1156/include/cute/arch/util.hpp#L119
    //uint32_t shared_ptr = cast_smem_ptr_to_uint(&sharedMem[tid]);
    //or
    uint32_t shared_ptr = __cvta_generic_to_shared(&sharedMem[tid]);

    // Store the input value into shared memory using PTX
    asm volatile (
        "st.shared.s32 [%0], %1;\n" // Store the value as int
        :
        : "r"(shared_ptr), "r"(input[tid])
    );

    // Synchronize threads to ensure all writes are visible
    __syncthreads();

    // Load the value back from shared memory using PTX
    int value;
    asm volatile (
        "ld.shared.s32 %0, [%1];\n" // Load the value as int
        : "=r"(value)
        : "r"(shared_ptr)
    );

    // Store the loaded value into the output array
    output[tid] = value;
}

int main() {
    const int numThreads = 32;
    int hostInput[numThreads], hostOutput[numThreads];
    int *deviceInput, *deviceOutput;

    // Initialize input array on the host
    for (int i = 0; i < numThreads; i++) {
        hostInput[i] = i * 3; // Example initialization
    }

    // Allocate memory on the device
    cudaMalloc(&deviceInput, numThreads * sizeof(int));
    cudaMalloc(&deviceOutput, numThreads * sizeof(int));

    // Copy input data to the device
    cudaMemcpy(deviceInput, hostInput, numThreads * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel with 1 block and 32 threads
    sharedMemoryKernel<<<1, numThreads>>>(deviceInput, deviceOutput);

    // Copy the results back to the host
    cudaMemcpy(hostOutput, deviceOutput, numThreads * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the results
    for (int i = 0; i < numThreads; i++) {
        printf("Thread %d: %d\n", i, hostOutput[i]);
    }

    // Free device memory
    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    return 0;
}

