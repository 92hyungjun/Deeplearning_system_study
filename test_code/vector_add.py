import torch
import triton
import triton.language as tl

SS = 1024
in_1 = (SS, SS)
t1 = torch.rand(in_1, device="cuda", requires_grad=False, dtype=torch.half)
t2 = torch.rand(in_1, device="cuda", requires_grad=False, dtype=torch.half)


@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)



def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    print(f"n ele : {n_elements}")
    def my_test_fn(meta):
        print(f"meta : {meta.keys()}")
        print(f"block size : {meta['BLOCK_SIZE']}")
        g_size = (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
        print(f"grid size : {g_size}")
        return g_size
    grid = my_test_fn
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024, num_warps=1, num_stages=2)
    #add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=32)
    return output


torch.manual_seed(0)
input_shapes = (1024,1)
test_dtype = torch.uint8
x = torch.randint(0,2, input_shapes, device='cuda', dtype=test_dtype)
y = torch.randint(0,2, input_shapes, device='cuda', dtype=test_dtype)

torch.matmul(t1, t2)
output_torch = x + y
#print(f"{x}")
#print(f"{y}")
#print(f"{output_torch}")
#torch.matmul(t1, t2)
print("##########")
output_triton = add(x, y)
#print(f"{x}")
#print(f"{y}")
#print(f"{output_triton}")
torch.matmul(t1, t2)
print("############# END ##############")

print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')
