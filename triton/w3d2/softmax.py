# softmax.py —— W3D2:行块思维 + 归约树转世 + 数值稳定性
# 运行:python3 softmax.py
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

print(f"triton={triton.__version__}  GPU={torch.cuda.get_device_name(0)}")

@triton.jit
def softmax_kernel(out_ptr, in_ptr, stride_in, stride_out, n_cols,
                   BLOCK_SIZE: tl.constexpr, STABLE: tl.constexpr, GHOST: tl.constexpr):
    row  = tl.program_id(0)                        # ⭐ 1 program = 1 整行
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    in_ptrs = in_ptr + row * stride_in + cols      # 2D 寻址:行起点 + 列偏移
    if GHOST:                                      # 修罗场:哨兵换成 0.0
        x = tl.load(in_ptrs, mask=mask, other=0.0)
    else:
        x = tl.load(in_ptrs, mask=mask, other=-float('inf'))  # ⭐ 双中性哨兵

    if STABLE:                                     # 数值稳定:减 max
        x = x - tl.max(x, axis=0)                  # tl.max = Week2 归约树转世
    num = tl.exp(x)
    denom = tl.sum(num, axis=0)                    # 第二棵树
    y = num / denom

    tl.store(out_ptr + row * stride_out + cols, y, mask=mask)

def softmax(x, num_warps=4, STABLE=True, GHOST=False):
    M, N = x.shape
    out = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(N)              # N=4000 → 4096(非2幂的正规解法)
    softmax_kernel[(M,)](out, x, x.stride(0), out.stride(0), N,
                         BLOCK_SIZE=BLOCK, STABLE=STABLE, GHOST=GHOST,
                         num_warps=num_warps)
    return out

def bench(fn):
    fn()                                           # warmup(每个新 constexpr 组合一次)
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(10): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / 10

# ───────── [1] 正确性:正方阵 + 非整除(考验 mask 哨兵)─────────
print("\n[1] 正确性:")
for shape in [(4096, 4096), (1024, 4000)]:
    x = torch.rand(*shape, device="cuda") * 10
    out, ref = softmax(x), F.softmax(x, dim=-1)
    err = (out - ref).abs().max().item()
    print(f"  {shape}: {'PASS' if torch.allclose(out, ref, atol=1e-6) else 'FAIL'}"
          f"  max|err|={err:.2e}")

# ───────── [2] 数值稳定性:同一份数据,稳定 vs 不稳定 ─────────
print("\n[2] 数值稳定性(输入 ~[999,1000]):")
x = torch.rand(3, 4, device="cuda") + 999
print("  F.softmax :", [f"{v:.3f}" for v in F.softmax(x, dim=-1)[0].tolist()])
print("  稳定版    :", [f"{v:.3f}" for v in softmax(x, STABLE=True)[0].tolist()])
print("  不稳定版  :", softmax(x, STABLE=False)[0].tolist())

# ───────── [3] 修罗场:哨兵 0.0 的"幽灵"(数据全负 + mask 尾巴长)─────────
print("\n[3] 幽灵哨兵(N=100→BLOCK=128,数据全在 [-101,-100]):")
x = -torch.rand(2, 100, device="cuda") - 100
good, ghost = softmax(x, GHOST=False), softmax(x, GHOST=True)
print("  other=-inf:", [f"{v:.4f}" for v in good[0, :4].tolist()],
      " 行和 =", f"{good[0].sum().item():.4f}")
print("  other=0.0 :", [f"{v:.2e}" for v in ghost[0, :4].tolist()],
      " 行和 =", f"{ghost[0].sum().item():.2e}")

# ───────── [4] 跑大:4096×4096 三选手 ─────────
print("\n[4] 4096×4096 计时(有用字节口径:读+写各 64MiB):")
M = N = 4096
BYTES = 2 * M * N * 4                              # 有用搬运:读 x + 写 y
x = torch.rand(M, N, device="cuda")
def naive(x):                                      # 未融合:5 个 torch kernel 接力
    m = x.max(dim=-1, keepdim=True)[0]
    num = torch.exp(x - m)
    return num / num.sum(dim=-1, keepdim=True)

for name, fn in [("Triton   ", softmax), ("F.softmax", lambda v: F.softmax(v, dim=-1)),
                 ("naive    ", naive)]:
    t = bench(lambda: fn(x))
    print(f"  {name}: {t:.3f} ms   {BYTES/1e6/t:.0f} GB/s(有用口径)   vs Triton "
          f"{'—' if name.strip()=='Triton' else ''}")
print(f"  理论地板(读+写各一遍)= {BYTES/936e9*1e3:.3f} ms")

# ───────── [5] num_warps 初见(其余全不动)─────────
print("\n[5] num_warps 扫描:")
for w in (4, 8, 16):
    print(f"  num_warps={w:2}: {bench(lambda: softmax(x, num_warps=w)):.3f} ms")