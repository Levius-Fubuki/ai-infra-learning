# matmul_autotune.py —— W3D4:autotune 自动扫分块参数(你的 k9 一站式补课)
# 运行:python3 matmul_autotune.py
import torch
import triton
import triton.language as tl

print(f"triton={triton.__version__}  GPU={torch.cuda.get_device_name(0)}")

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                  GROUP_M: tl.constexpr, ALLOW_TF32: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    # ⭐ group ordering:相邻 program 处理相邻 C 块 → A/B 的 L2 复用
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k < (K - k * BLOCK_K)              # K 轴 mask(Day3 教训)
        a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)
        acc = tl.dot(a, b, acc, allow_tf32=ALLOW_TF32)   # IEEE:精度诚实
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc)

# ── ① 套上 autotune(纯函数式包一层,原 kernel 还能直接手动调)──
matmul_autotuned = triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=1, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)(matmul_kernel)

def matmul(a, b):                       # autotune 版(默认 IEEE)
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    matmul_autotuned[grid](a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c.stride(0), c.stride(1), ALLOW_TF32=False)
    return c

def matmul_fixed(a, b, BM, BN, BK, num_warps=8, num_stages=2):   # 手动指定,做实验
    M, K = a.shape; K2, N = b.shape; assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    matmul_kernel[grid](a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=8, ALLOW_TF32=False,
        num_warps=num_warps, num_stages=num_stages)
    return c

def bench(fn):
    fn()                                # warmup(autotune 首调会 BENCHMARKING 全部 config)
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(10): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / 10

# ───────── [1] 验证门(铁律:先对再快;autotune 版,IEEE)─────────
print("\n[1] 验证门(autotune 版):")
for n in (256, 1024):
    a, b = torch.randn(n, n, device="cuda"), torch.randn(n, n, device="cuda")
    c, ref = matmul(a, b), a @ b
    print(f"  {n}³: {'PASS' if torch.allclose(c, ref, atol=1e-4) else 'FAIL'}  max|err|={(c-ref).abs().max().item():.2e}")

# ───────── [2] 4096³ autotune 跑分(首调会打印缓存表,请截下来)─────────
print("\n[2] 4096³ autotune:")
n = 4096
a, b = torch.randn(n, n, device="cuda"), torch.randn(n, n, device="cuda")
t = bench(lambda: matmul(a, b))
print(f"  autotuned: {t:.3f} ms → {2*n**3/(t*1e-3)/1e12:.2f} TFLOPS")
t_ref = bench(lambda: a @ b)
print(f"  cuBLAS   : {t_ref:.3f} ms → {2*n**3/(t_ref*1e-3)/1e12:.2f} TFLOPS")
print("  轨迹:naive 0.51 → tiled 2.88 → k4 3.97 → D3 4.03 → autotune ? → k5 8.57 → cuBLAS 24")

# ───────── [3] num_stages 扫描(64×64×64,双缓冲)─────────
print("\n[3] num_stages 扫描(64×64×64, num_warps=8):")
for ns in (1, 2, 3):
    t = bench(lambda: matmul_fixed(a, b, 64, 64, 64, num_warps=8, num_stages=ns))
    print(f"  num_stages={ns}: {t:.3f} ms → {2*n**3/(t*1e-3)/1e12:.2f} TFLOPS")

# ───────── [4] shared 墙:128×128×64 fp32 配 ns=2 ─────────
print("\n[4] shared 墙(128×128×64, 试着 ns=2):")
try:
    t = bench(lambda: matmul_fixed(a, b, 128, 128, 64, num_warps=8, num_stages=2))
    print(f"  竟然成功: {t:.3f} ms → {2*n**3/(t*1e-3)/1e12:.2f} TFLOPS(看它是不是默默降了 stages)")
except Exception as ex:
    print("  如预期失败:", type(ex).__name__, str(ex)[:100])

# ───────── [5] BLOCK 扫描:验证 AI≈B/4 → 带宽帽翻倍 ─────────
print("\n[5] BLOCK 扫描(方块, BK=32, ns=2):")
for B in (32, 64, 128):
    t = bench(lambda: matmul_fixed(a, b, B, B, 32, num_warps=8, num_stages=2))
    print(f"  BLOCK={B:3}: {t:.3f} ms → {2*n**3/(t*1e-3)/1e12:.2f} TFLOPS")