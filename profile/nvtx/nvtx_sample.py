import torch
from torch.cuda import nvtx

#nsys profile -t cuda,nvtx -o /work/debug_nvvp/_%p.qdstrm python nvtx_sample.py

s1 = (256, 256)
t1 = torch.rand(s1, device="cuda", requires_grad=False, dtype=torch.half)

s2 = (512, 512)
t2 = torch.rand(s2, device="cuda", requires_grad=False, dtype=torch.half)

s3 = (1024, 1024)
t3 = torch.rand(s3, device="cuda", requires_grad=False, dtype=torch.half)

#warm-up
for i in range(2):
    torch.matmul(t1, t1)
    torch.matmul(t2, t2)
    torch.matmul(t3, t3)

nvtx.range_push(f"MM")
nvtx.range_push(f"m1")
torch.matmul(t1, t1)
nvtx.range_pop() # m1

nvtx.range_push(f"m2")
torch.matmul(t2, t2)
nvtx.range_pop() # m2

nvtx.range_push(f"m3")
torch.matmul(t3, t3)
nvtx.range_pop() # m3
nvtx.range_pop() # MM

torch.cuda.synchronize()
