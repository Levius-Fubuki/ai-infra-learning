# canary.py —— 亲眼看见无 mask 的越界写污染邻居显存
import torch, triton, triton.language as tl

@triton.jit
def add_nomask(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, tl.load(x_ptr + offs) + tl.load(y_ptr + offs))

N, B = 4097, 4096                    # grid=1,一次越界写 4095 个 float ≈ 16 KB
x, y = torch.rand(N, device="cuda"), torch.rand(N, device="cuda")
out   = torch.empty_like(x)
victim = torch.full((2048,), -12345.0, device="cuda")   # 金丝雀:塞满哨兵值
add_nomask[triton.cdiv(N, B),](x, y, out, BLOCK=B)
torch.cuda.synchronize()
print("out 正确? ", torch.allclose(out, x + y))
bitten = (victim != -12345.0).sum().item()
print(f"victim 被咬 {bitten}/2048 口 → OOB 写实锤")