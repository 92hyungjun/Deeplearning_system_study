#based on https://github.com/triton-lang/triton/tree/main/python/tutorials

import torch
import triton
import triton.language as tl
from triton.runtime import driver


def naive_softmax(x):
    """Compute row-wise softmax of X using native pytorch

    We subtract the maximum element in order to avoid overflows. Softmax is invariant to
    this shift.
    """
    # read  MN elements ; write M  elements
    x_max = x.max(dim=1)[0]
    # read MN + M elements ; write MN elements
    z = x - x_max[:, None]
    # read  MN elements ; write MN elements
    numerator = torch.exp(z)
    # read  MN elements ; write M  elements
    denominator = numerator.sum(dim=1)
    # read MN + M elements ; write MN elements
    ret = numerator / denominator[:, None]
    # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
    return ret


@triton.jit
def softmax_kernel(output_ptr, input_ptr, n_eles, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0) 
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    input_ptrs = input_ptr + offsets

    mask = offsets < n_eles
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    output_ptrs = output_ptr + offsets
    tl.store(output_ptrs, softmax_output, mask=mask)


def softmax(x):
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    num_programs = n_rows
    n_elements = y.numel()
    softmax_kernel[(num_programs, 1, 1)](
        y,
        x,
        n_elements,
        BLOCK_SIZE=n_cols,
        num_stages=4
    )
    return y


torch.manual_seed(0)
x = torch.randn(512, 2048, device='cuda')
for i in range(10):
    y_triton = softmax(x)
    naive = naive_softmax(x)
    y_torch = torch.softmax(x, axis=1)
torch.cuda.synchronize()
print(f"triton : {y_triton}")
print(f"torch  : {y_torch} ")
print(f"naive  : {naive}")
assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)
