# matmul_fp16.py —— W3D5:fp16 tensor core + 大分块(黄金组合)
# 运行:python3 matmul_fp16.py
import torch
import triton
import triton.language as tl

print(f"triton={triton.__version__}  GPU={torch.cuda.get_device_name(0)}")

@triton.jit
def matmul_fp16_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                       stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                       GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)   # ⭐ 累加器 fp32 保精度
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k < (K - k * BLOCK_K)
        a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)   # fp16 输入
        b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)
        acc = tl.dot(a, b, acc)        # fp16×fp16→fp32,自动走 fp16 tensor core
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc)              # 结果 fp32

matmul_fp16 = triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,   'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128,  'BLOCK_N': 64,  'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256,  'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128,  'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=1, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)(matmul_fp16_kernel)

def matmul(a, b):                    # a,b 是 fp16,返回 fp32
    M, K = a.shape; K2, N = b.shape; assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    matmul_fp16[grid](a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c.stride(0), c.stride(1))
    return c

def matmul_fixed(a, b, BM, BN, BK, num_warps=8, num_stages=2):   # 手动指定做实验
    M, K = a.shape; K2, N = b.shape; assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    matmul_fp16_kernel[grid](a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=8,
        num_warps=num_warps, num_stages=num_stages)
    return c

def bench(fn):
    fn()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(10): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / 10

# ───────── [1] 验证门(fp16 vs fp32 参考,rtol=1e-2)─────────
print("\n[1] 验证门(fp16 in, fp32 acc, vs fp32 参考):")
for n in (256, 1024, 4096):
    a, b = torch.randn(n, n, device="cuda"), torch.randn(n, n, device="cuda")
    ah, bh = a.half(), b.half()
    c = matmul(ah, bh)
    ref16 = (ah @ bh).float()                  # cuBLAS fp16,现在 fp32 累加
    ref32 = a @ b
    e16 = (c - ref16).abs().max().item()
    e32 = (c - ref32).abs().max().item()
    rel16 = e16 / ref16.abs().max().item()
    rel32 = e32 / ref32.abs().max().item()
    print(f"  {n}³: 实现正确性 e16={e16:.2e}(rel{rel16:.1e},{'PASS' if rel16<1e-3 else 'FAIL'})"
          f"   精度预算 e32={e32:.2e}(rel{rel32:.1e},固有)")
    print(f"  {n}³: c.half() vs cuBLAS fp16 = {(c.half().float() - ref16).abs().max().item():.2e}")
    
# ───────── [2] 4096³ 跑分:fp16 vs fp32 vs cuBLAS ─────────
print("\n[2] 4096³ 跑分:")
n = 4096
a, b = torch.randn(n, n, device="cuda"), torch.randn(n, n, device="cuda")
ah, bh = a.half(), b.half()
FLOP = 2 * n**3
t = bench(lambda: matmul(ah, bh))
print(f"  Triton fp16 autotune: {t:.3f} ms → {FLOP/(t*1e-3)/1e12:.2f} TFLOPS")
t_ref = bench(lambda: a @ b)
print(f"  cuBLAS fp32         : {t_ref:.3f} ms → {FLOP/(t_ref*1e-3)/1e12:.2f} TFLOPS")
t_refh = bench(lambda: (ah @ bh).float())
print(f"  cuBLAS fp16         : {t_refh:.3f} ms → {FLOP/(t_refh*1e-3)/1e12:.2f} TFLOPS")
print("  锚点:D4 fp32 autotune 19.91 · Week2 k5 8.57 · fp32 峰 35.6 · fp16 tensor 峰 142")

# ───────── [3] 昨天不可能今天可能:128×128×64 扫 num_stages ─────────
print("\n[3] 128×128×64 扫 num_stages(fp16;fp32 昨天只能 ns=1):")
for ns in (1, 2, 3, 4):
    try:
        t = bench(lambda: matmul_fixed(ah, bh, 128, 128, 64, num_warps=8, num_stages=ns))
        print(f"  ns={ns}: {t:.3f} ms → {FLOP/(t*1e-3)/1e12:.2f} TFLOPS")
    except Exception as ex:
        print(f"  ns={ns}: 失败 {type(ex).__name__} {str(ex)[:60]}")

# ───────── [4] 精度阶梯:fp64 vs fp32 vs fp16 ─────────
print("\n[4] 精度阶梯(256³,以 fp64 为真值):")
n = 256
a, b = torch.randn(n, n, device="cuda"), torch.randn(n, n, device="cuda")
c64 = a.double() @ b.double()
e32 = (a @ b - c64).abs().max().item()
e16 = (matmul(a.half(), b.half()) - c64).abs().max().item()
print(f"  fp32 最大误差 = {e32:.2e}   fp16 = {e16:.2e}   → fp16 是 fp32 的 {e16/max(e32,1e-30):.0f}×")