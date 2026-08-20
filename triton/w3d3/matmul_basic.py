# matmul_basic.py —— W3D3:裸版 Triton matmul(BLOCK=16 压线对照 Week2 tiled)
# 运行:python3 matmul_basic.py
import torch
import triton
import triton.language as tl

print(f"triton={triton.__version__}  GPU={torch.cuda.get_device_name(0)}")

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr,
                  M, N, K,
                  stride_am, stride_ak,   # A: (M,K) 行主序
                  stride_bk, stride_bn,   # B: (K,N) 行主序
                  stride_cm, stride_cn,   # C: (M,N)
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                  ALLOW_TF32: tl.constexpr):          # ⭐ 新增开关
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m             # 2D 网格压平成 1D 再拆:先定行块
    pid_n = pid // num_pid_m            #                        再定列块

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)     # C tile 的行坐标(向量,16)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)     # C tile 的列坐标(向量,16)
    offs_k = tl.arange(0, BLOCK_K)                        # K 方向游标(向量,16)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)  # 累加器,fp32 保精度

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)             # (BM,BK) 块
        b = tl.load(b_ptrs)             # (BK,BN) 块
        acc = tl.dot(a, b, acc, allow_tf32=ALLOW_TF32)    # ⭐ 开关接进 dot
        a_ptrs += BLOCK_K * stride_ak   # K 游标推进:A 向下移 BK 行
        b_ptrs += BLOCK_K * stride_bk   #           B 向右移 BK 列

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc)

def matmul(a, b, allow_tf32=True):                       # ⭐ 对外暴露
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "内维不匹配"
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = (triton.cdiv(M, 16) * triton.cdiv(N, 16),)    # 1D grid 压平
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=16, BLOCK_N=16, BLOCK_K=16,
        ALLOW_TF32=allow_tf32,
        num_warps=4,
    )
    return c

def bench(fn):
    fn()                                   # warmup(每个新 constexpr 组合一次)
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(10): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / 10

FLOP = lambda n: 2 * n**3

# ───────── [1] TF32 与 IEEE 验证门对照(铁律:先对再快)─────────
print("\n[1] 验证门 TF32 vs IEEE:")
for name, atf in [("TF32(默认)", True), ("IEEE(full)", False)]:
    for n in (256, 1024):
        a, b = torch.randn(n, n, device="cuda"), torch.randn(n, n, device="cuda")
        c, ref = matmul(a, b, allow_tf32=atf), a @ b
        ok = torch.allclose(c, ref, atol=1e-4)
        print(f"  {name:<10} {n}³: {'PASS' if ok else 'FAIL'}  max|err|={(c-ref).abs().max().item():.2e}")

# ───────── [2] 4096³ 跑分:TF32 / IEEE / cuBLAS ─────────
print("\n[2] 4096³ 跑分:")
n = 4096
a, b = torch.randn(n, n, device="cuda"), torch.randn(n, n, device="cuda")
for name, atf in [("Triton TF32 ", True), ("Triton IEEE ", False)]:
    t = bench(lambda: matmul(a, b, allow_tf32=atf))
    print(f"  {name}: {t:.3f} ms  →  {FLOP(n)/(t*1e-3)/1e12:.2f} TFLOPS")
t_ref = bench(lambda: a @ b)
print(f"  torch cuBLAS: {t_ref:.3f} ms  →  {FLOP(n)/(t_ref*1e-3)/1e12:.2f} TFLOPS")

# 你的 Week2 战绩(对照锚点)
print("  ── Week2 锚点 ──")
for nm, tf in [("naive", 0.514), ("tiled(16)", 2.88), ("k4 regtile", 3.97), ("k5 2D(128/8)", 8.57)]:
    print(f"    {nm:<14}: {tf} TFLOPS")

# ───────── [3] 修罗场:K=100 非整除(选做,先预测再跑)─────────
print("\n[3] 修罗场:K=100 非整除:")
ka, kb = torch.randn(128, 100, device="cuda"), torch.randn(100, 128, device="cuda")
kc = matmul(ka, kb)
ok = torch.allclose(kc, ka @ kb, atol=1e-3)
print(f"  结果: {'PASS' if ok else 'FAIL'}  max|err|={(kc - ka@kb).abs().max().item():.2e}")

# ───────── [4] 附赠:给 matmul 加 K 轴 mask,补上 [3] 的洞 ─────────
# (先别改主 kernel,单独一个带 mask 的版本,验证 K=100 能救回来)
@triton.jit
def matmul_kernel_masked(a_ptr, b_ptr, c_ptr, M, N, K,
                         stride_am, stride_ak, stride_bk, stride_bn,
                         stride_cm, stride_cn,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                         BLOCK_K: tl.constexpr, ALLOW_TF32: tl.constexpr):
    pid = tl.program_id(0)
    pid_m = pid % tl.cdiv(M, BLOCK_M)
    pid_n = pid // tl.cdiv(M, BLOCK_M)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k < (K - k * BLOCK_K)            # ⭐ K 轴 mask
        a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)
        acc = tl.dot(a, b, acc, allow_tf32=ALLOW_TF32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc)

def matmul_masked(a, b):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = (triton.cdiv(M, 16) * triton.cdiv(N, 16),)
    matmul_kernel_masked[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=16, BLOCK_N=16, BLOCK_K=16, ALLOW_TF32=False)
    return c

print("\n[4] K-mask 修复版(K=100):")
kc2 = matmul_masked(ka, kb)
ok = torch.allclose(kc2, ka @ kb, atol=1e-4)
print(f"  K=100: {'PASS' if ok else 'FAIL'}  max|err|={(kc2 - ka@kb).abs().max().item():.2e}")
kc3 = matmul_masked(torch.randn(1024, 1024, device="cuda"), torch.randn(1024, 1024, device="cuda"))