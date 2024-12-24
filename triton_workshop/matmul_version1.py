#based on https://github.com/triton-lang/triton/tree/main/python/tutorials

import torch
import triton
import triton.language as tl
import time


@triton.jit
def matmul_kernel(
    x_ptr,
    y_ptr,
    z_ptr,
    M_SIZE: tl.constexpr,
    K_SIZE: tl.constexpr,
    N_SIZE: tl.constexpr,
    M_BLOCK_SIZE: tl.constexpr,
    K_BLOCK_SIZE: tl.constexpr,
    N_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    num_n_blocks = tl.cdiv(N_SIZE, N_BLOCK_SIZE)
    m_block = pid // num_n_blocks
    n_block = pid % num_n_blocks

    m_offsets = tl.arange(0, M_BLOCK_SIZE) + m_block * M_BLOCK_SIZE
    n_offsets = tl.arange(0, N_BLOCK_SIZE) + n_block * N_BLOCK_SIZE
    k_offsets = tl.arange(0, K_BLOCK_SIZE)

    x_ptrs = x_ptr + m_offsets[:, None] * K_SIZE + k_offsets[None, :]
    y_ptrs = y_ptr + k_offsets[:, None] * N_SIZE + n_offsets[None, :]
    z_ptrs = z_ptr + m_offsets[:, None] * N_SIZE + n_offsets[None, :]

    accumulator = tl.zeros((M_BLOCK_SIZE, N_BLOCK_SIZE), tl.float32)
    for k in range(0, K_SIZE, K_BLOCK_SIZE):
        x_mask = (m_offsets[:, None] < M_SIZE) & (k_offsets[None, :]+k < K_SIZE)
        y_mask = (k_offsets[:, None]+k < K_SIZE) & (n_offsets[None, :] < N_SIZE)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        y = tl.load(y_ptrs, mask=y_mask, other=0.0)
        accumulator = tl.dot(x, y, accumulator)
        x_ptrs += K_BLOCK_SIZE
        y_ptrs += K_BLOCK_SIZE * N_SIZE

    c = accumulator.to(tl.float16)
    z_mask = (m_offsets[:, None] < M_SIZE) & (n_offsets[None, :] < N_SIZE)
    tl.store(z_ptrs, c, mask=z_mask)


def matmul(x, y):
    m_size, k_size = x.shape
    _, n_size = y.shape
    z = torch.empty(m_size, n_size, device="cuda", dtype=torch.float16)

    def grid(meta):
        return (triton.cdiv(m_size, meta["M_BLOCK_SIZE"]) * 
                triton.cdiv(n_size, meta["N_BLOCK_SIZE"]),)

    K_BLOCK = 16
    M_BLOCK = 16
    N_BLOCK = 16
    kernel_info = matmul_kernel[grid](x, y, z, m_size, k_size, n_size, M_BLOCK, K_BLOCK, N_BLOCK, num_warps=1)
    #export TRITON_DEBUG=1
    #print(kernel_info.asm.keys())
    return z


def main():
    M = 31
    K = 31
    N = 31
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    y = torch.randn(K, N, device="cuda", dtype=torch.float16)

    a = matmul(x, y)
    b = torch.matmul(x, y)
    torch.cuda.synchronize()

    def time_check(fn, fn_name, CNT=5):
        s = time.time()
        torch.cuda.synchronize()
        for i in range(CNT):
            fn(x,y)
        torch.cuda.synchronize()
        e = time.time()
        average = (e-s)/CNT
        print(f"{fn_name} mm {average}")
        
    #time_check(torch.matmul, "torch")
    #time_check(matmul, "triton")

    if torch.allclose(a, b, atol=0.0001):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")
        #torch.set_printoptions(threshold=10_000)
        torch.set_printoptions(linewidth=200)
        print(f"triton")
        print(f"{a}")
        print(f"torch")
        print(f"{b}")


if __name__ == "__main__":
    main()
