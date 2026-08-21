# flash_attn_fwd.py —— W3D6:简化版 Flash-Attention 前向(非因果,单头,head_dim=64)
# 运行:python3 flash_attn_fwd.py
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

print(f"triton={triton.__version__}  GPU={torch.cuda.get_device_name(0)}")

BLOCK_M, BLOCK_N, HEAD_DIM = 64, 64, 64

@triton.jit
def attn_fwd(Q, K, V, sm_scale, O, N,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr):
    """
    简化版 Flash-Attention 前向(非因果,单头)。
    参数说明(以 4096 序列、head_dim=64 为例):
    Q : 查询矩阵首地址(门牌号,不是张量本体),显存 (N, 64) fp16,行主序
        每个 program 只取其中 BLOCK_M=64 行(对应 64 个输出 token)
    K : 键矩阵首地址,显存 (N, 64) fp16,行主序
        在循环里分块流式读,每块 BLOCK_N=64 行(=64 个 key)
    V : 值矩阵首地址,显存 (N, 64) fp16,行主序
        与 K 同步分块流式读,形状 (BLOCK_N, HEAD_DIM)
    sm_scale : 缩放因子 = 1/sqrt(head_dim),运行期标量
        head_dim=64 → 1/8 = 0.125,乘到点积上防数值爆炸(attention 的定义)
    O : 输出首地址,显存 (N, 64) **fp32**(精度防线,和 acc 同 dtype)
        每个 program 写回自己那 BLOCK_M=64 行
    N : 序列长度,运行期标量
        决定 K/V 分块循环圈数 = N/BLOCK_N(如 4096/64 = 64 圈)
    BLOCK_M : 编译期常量,本 program 一次处理的 Q 行数(64)
        grid = N/BLOCK_M 个 program,每个算 64 行输出
    BLOCK_N : 编译期常量,每次循环扫过的 K/V 行数(64)
        圈数 = N/BLOCK_N;S/P 块 (BLOCK_M, BLOCK_N) 只在寄存器里存在,不落显存
    HEAD_DIM : 编译期常量,每行特征数(64)
        tl.arange(0,64) 要求是 2 的幂(非 2 幂会编译报错);
        行主序下行步长=HEAD_DIM,所有寻址都乘它
    备注:
    - Q/K/V 传的都是首地址(Triton 从 torch 张量取 .data_ptr()),指针算术按"元素"计
    - 三个 BLOCK 都是 constexpr:换值=编译一个全新 kernel(首次调用会 JIT)
    - 简化版假设 N 整除 64 且输入连续,故全程免 mask(库版要带)
    """
    start_m = tl.program_id(0)                    # ⭐ 1 program = 64 行 Q
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)  # 64 个行号
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = Q + offs_m[:, None] * HEAD_DIM + offs_d[None, :]
    k_ptrs = K + offs_n[None, :] * HEAD_DIM + offs_d[:, None]   # 转置:(HEAD_DIM, BLOCK_N)
    v_ptrs = V + offs_n[:, None] * HEAD_DIM + offs_d[None, :]   # (BLOCK_N, HEAD_DIM)

    q = tl.load(q_ptrs)                       # (BLOCK_M, HEAD_DIM) fp16
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)  # 滚动输出
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)           # 在线行和
    m_i = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)  # 在线 max

    for start_n in range(0, N, BLOCK_N):      # ⭐ 沿 N 轴扫,K/V 分块推进
        k = tl.load(k_ptrs)                   # (HEAD_DIM, BLOCK_N)
        v = tl.load(v_ptrs)                   # (BLOCK_N, HEAD_DIM)
        qk = tl.dot(q, k) * sm_scale          # (BLOCK_M, BLOCK_N) 打分
        m_ij = tl.max(qk, 1)                  # 本块行 max
        m_new = tl.maximum(m_i, m_ij)         # ① 在线 max
        alpha = tl.exp(m_i - m_new)           # ② 旧贡献缩放
        p = tl.exp(qk - m_new[:, None])       # 本块概率 (BLOCK_M, BLOCK_N)
        l_i = l_i * alpha + tl.sum(p, 1)      # ③ 在线行和
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)  # ③ rescale + 累加
        m_i = m_new
        k_ptrs += BLOCK_N * HEAD_DIM          # 推进到下一个 K/V 块
        v_ptrs += BLOCK_N * HEAD_DIM

    acc = acc / l_i[:, None]                  # 扫完,一次除完
    tl.store(O + offs_m[:, None] * HEAD_DIM + offs_d[None, :], acc)  # 输出 fp32

def flash_attn(q, k, v, sm_scale=None):
    N, d = q.shape
    assert d == HEAD_DIM and N % BLOCK_M == 0 and N % BLOCK_N == 0, "要求 head_dim=64 且 N 整除 64"
    if sm_scale is None:
        sm_scale = 1.0 / (d ** 0.5)
    out = torch.empty((N, d), device=q.device, dtype=torch.float32)
    grid = (N // BLOCK_M,)
    attn_fwd[grid](q, k, v, sm_scale, out, N,
                   BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=d, num_warps=4)
    return out

def ref_attn_fp32(q, k, v, sm_scale):       # 真值:显式 softmax,fp32
    S = (q.float() @ k.float().transpose(0, 1)) * sm_scale
    P = torch.softmax(S, dim=-1)
    return P @ v.float()

def naive_attn_fp16(q, k, v, sm_scale):     # 朴素版:落显存,fp16(生产基线)
    S = (q @ k.transpose(0, 1)) * sm_scale
    P = torch.softmax(S, dim=-1)
    return P @ v

def bench(fn):
    fn()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(10): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / 10

# ───────── [1] 正确性(flash vs fp32 显式参考)─────────
print("\n[1] 正确性(flash vs fp32 显式参考):")
for N in (1024, 2048, 4096):
    q, k, v = (torch.randn(N, 64, device="cuda", dtype=torch.float16) for _ in range(3))
    sm = 1.0 / 8.0
    out = flash_attn(q, k, v, sm)
    ref = ref_attn_fp32(q, k, v, sm)
    rel = (out - ref).abs().max().item() / ref.abs().max().item()
    ok = torch.allclose(out, ref, rtol=2e-2, atol=2e-2)
    print(f"  N={N:5}: {'PASS' if ok else 'FAIL'}  rel={rel:.1e}")

# ───────── [2] 对比 torch.sdpa(生产参考,它内部可能就是 flash)─────────
print("\n[2] vs torch.sdpa:")
for N in (1024, 4096):
    q, k, v = (torch.randn(N, 64, device="cuda", dtype=torch.float16) for _ in range(3))
    sm = 1.0 / 8.0
    out = flash_attn(q, k, v, sm)
    sdpa = F.scaled_dot_product_attention(q[None], k[None], v[None], scale=sm)[0].float()
    rel = (out - sdpa).abs().max().item() / sdpa.abs().max().item()
    print(f"  N={N:5}: flash vs sdpa rel={rel:.1e}")

# ───────── [3] seq_len 扫描:Triton vs 朴素 vs sdpa ─────────
print("\n[3] seq_len 扫描(耗时 ms):")
print("   N      Triton   朴素(落显存)    sdpa")
for N in (1024, 2048, 4096, 8192):
    q, k, v = (torch.randn(N, 64, device="cuda", dtype=torch.float16) for _ in range(3))
    sm = 1.0 / 8.0
    t_tr = bench(lambda: flash_attn(q, k, v, sm))
    t_naive = bench(lambda: naive_attn_fp16(q, k, v, sm))
    t_sdpa = bench(lambda: F.scaled_dot_product_attention(q[None], k[None], v[None], scale=sm))
    print(f"  {N:5}   {t_tr:7.3f}   {t_naive:7.3f}       {t_sdpa:7.3f}")