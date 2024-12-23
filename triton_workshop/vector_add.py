#based on https://github.com/triton-lang/triton/tree/main/python/tutorials
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               ):
    pid = tl.program_id(axis=0) 
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    def grid(meta):
        g_size = (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
        return g_size
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=32, num_warps=1, num_stages=1)
    return output

input_shapes = (128,1)
test_dtype = torch.uint8
x = torch.randint(0,2, input_shapes, device='cuda', dtype=test_dtype)
y = torch.randint(0,2, input_shapes, device='cuda', dtype=test_dtype)

output_torch = x + y
output_triton = add(x, y)

print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')
