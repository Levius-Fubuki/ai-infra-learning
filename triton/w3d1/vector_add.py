# vector_add.py —— W3D1: 第一个 Triton kernel 🎉
# 运行:python3 vector_add.py(对,不用 nvcc,保存即跑)
import torch
import triton
import triton.language as tl

print(f"triton={triton.__version__}  torch={torch.__version__}  "
      f"GPU={torch.cuda.get_device_name(0)}")

# ───────── GPU 侧:kernel(≈ .cu 里的 __global__ 函数)─────────
@triton.jit                                          # ≈ __global__:标记"GPU 函数"
def add_kernel(x_ptr, y_ptr, output_ptr,             # 门牌号:张量首地址(≈ float*)
               n_elements,                           # ≈ int n(运行期标量)
               BLOCK_SIZE: tl.constexpr):            # 编译期常量:换值=重编一个新版本
    pid  = tl.program_id(axis=0)                     # ≈ blockIdx.x
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # ⭐ 一次算一整块索引(向量!)
    mask = offs < n_elements                         # ≈ if(i<n) 的向量版

    x      = tl.load(x_ptr + offs)        # 批量读(mask=F 的位置不读)
    y      = tl.load(y_ptr + offs)
    output = x + y                                   # NumPy 风格:整块相加
    tl.store(output_ptr + offs, output)   # 批量写(mask=F 的位置不写)

# ───────── CPU 侧:launcher(≈ main() 的六步骨架,现在只剩三行)─────────
def add(x, y, BLOCK_SIZE=1024, verbose=True):
    output = torch.empty_like(x)                     # ≈ cudaMalloc(free 交给 GC)
    n = output.numel()
    grid = (n//BLOCK_SIZE,)             # ≈ (N + B-1)//B 向上取整,一模一样
    if verbose:
        print(f"  N={n:>9}  BLOCK={BLOCK_SIZE:>5}  grid={grid[0]:>4} programs")
    add_kernel[grid](x, y, output, n, BLOCK_SIZE=BLOCK_SIZE)  # ≈ kernel<<<grid>>>(...)
    return output

# ───────── ① 验证门(Week2 铁律照旧:先对再快)─────────
print("\n[1] 正确性(含 N=4097 非整除,考验 mask):")
for N in (4096, 4097):
    x, y = torch.rand(N, device="cuda"), torch.rand(N, device="cuda")
    out, ref = add(x, y), x + y                      # torch 自带 GPU 加法当参考答案
    err = (out - ref).abs().max().item()
    print(f"  N={N}: {'✅ PASS' if torch.allclose(out, ref) else '❌ FAIL'}"
          f"  max|err|={err:.2e}")

# ───────── ② 跑大计时(warmup + cudaEvent,Week2 手法照搬)─────────
print("\n[2] 2^26 元素,BLOCK_SIZE 单变量扫描:")
N = 2 ** 26                                          # 6710 万元素;读x+读y+写out
x, y = torch.rand(N, device="cuda"), torch.rand(N, device="cuda")

def bench(fn):
    fn()                                             # warmup(首次调用含 JIT 编译)
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(10):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / 10                    # 每次平均 ms

for B in (256, 1024, 4096):
    ms = bench(lambda: add(x, y, BLOCK_SIZE=B, verbose=False))
    gb = 3 * N * 4 / 1e9                             # 移动字节数:读x+读y+写out
    print(f"  BLOCK={B:>5}: {ms:.3f} ms  →  {gb/ms*1000:.0f} GB/s")
ms = bench(lambda: x + y)
print(f"  torch 原生 a+b: {ms:.3f} ms  →  {3*N*4/1e9/ms*1000:.0f} GB/s")