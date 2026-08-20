 # bench_configs.py —— 手动扫 8 个 config,打"成绩单"(替代 autotune 的 BENCHMARKING 表)
    # 运行:python3 -u bench_configs.py
import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                    GROUP_M: tl.constexpr, ALLOW_TF32: tl.constexpr):
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k < (K - k * BLOCK_K)
        a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)
        acc = tl.dot(a, b, acc, allow_tf32=ALLOW_TF32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc)

def matmul_fixed(a, b, BM, BN, BK, num_warps=8, num_stages=2):
    M, K = a.shape; K2, N = b.shape; assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    matmul_kernel[grid](a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=8, ALLOW_TF32=False,
        num_warps=num_warps, num_stages=num_stages)
    return c

# ── 与 autotune configs 完全一致的 8 个候选 ──
CONFIGS = [
    dict(BM=64,  BN=64,  BK=32, ns=2, nw=4),
    dict(BM=64,  BN=64,  BK=32, ns=4, nw=4),
    dict(BM=64,  BN=64,  BK=64, ns=2, nw=8),
    dict(BM=128, BN=128, BK=32, ns=2, nw=8),
    dict(BM=128, BN=128, BK=32, ns=3, nw=8),
    dict(BM=128, BN=64,  BK=32, ns=3, nw=4),
    dict(BM=64,  BN=128, BK=32, ns=3, nw=4),
    dict(BM=128, BN=128, BK=64, ns=1, nw=8),
]

n = 4096
a = torch.randn(n, n, device="cuda")
b = torch.randn(n, n, device="cuda")
ref = a @ b
FLOP = 2 * n**3

rows = []
print(f"扫 8 个 config @ {n}³(IEEE 全精度):\n")
for i, cfg in enumerate(CONFIGS):
    fn = lambda: matmul_fixed(a, b, cfg["BM"], cfg["BN"], cfg["BK"],
                              num_warps=cfg["nw"], num_stages=cfg["ns"])
    fn()                                        # warmup(含 JIT 编译)
    ms = triton.testing.do_bench(fn)            # 自动多次计时取稳定均值
    tf = FLOP / (ms * 1e-3) / 1e12
    c = matmul_fixed(a, b, cfg["BM"], cfg["BN"], cfg["BK"],
                     num_warps=cfg["nw"], num_stages=cfg["ns"])
    ok = torch.allclose(c, ref, atol=1e-4)
    rows.append((ms, i, cfg, tf, ok))
    print(f"  Config {i}: BM={cfg['BM']:3} BN={cfg['BN']:3} BK={cfg['BK']:3} "
          f"ns={cfg['ns']} nw={cfg['nw']} → {ms:.3f} ms  {tf:6.2f} TFLOPS  {'PASS' if ok else 'FAIL'}")

rows.sort()
ms, i, cfg, tf, ok = rows[0]
print(f"\n🏆 最优 Config {i}: BM={cfg['BM']} BN={cfg['BN']} BK={cfg['BK']} "
      f"ns={cfg['ns']} nw={cfg['nw']} → {ms:.3f} ms  {tf:.2f} TFLOPS")

t = triton.testing.do_bench(lambda: a @ b)
print(f"参考 cuBLAS: {t:.3f} ms → {FLOP/(t*1e-3)/1e12:.2f} TFLOPS")