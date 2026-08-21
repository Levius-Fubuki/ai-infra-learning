---
title: Week 3 笔记(Day6):Flash-Attention 前向——online softmax
phase: 0
week: 3
period: 2026-08-21
tags:
  - ai-infra
  - 笔记
  - triton
created: 2026-08-21
---

# Week 3 笔记(Day6):Flash-Attention 前向——online softmax

> [!info] 笔记体例
> 概念块沿用四段式 **是什么 / 有什么用 / 怎么用 / 坑**;代码块采用 **逐行讲解 + 具体数字举例**。当日五题完整批改存档于 §四(我的答案 → 对错 → 详解)。
> 关联 [[Week3笔记-Day5-fp16与tensor-core]] · [[Week3笔记-Day2-softmax归约与融合]] · [[Week3笔记-Day4-autotune与分块参数]] · [[阶段0-执行计划]]。

> [!success] Day6 成绩
> - `flash_attn_fwd.py` 跑通(单头 head_dim=64、非因果):验证 PASS rel 1.5e-4/7e-5/1.8e-4;vs torch.sdpa rel 3e-4 一致。
> - **N=8192:Triton 0.356 ms vs 朴素 1.088 ms(3.1×)** vs torch.sdpa 3.324 ms(9.3×,有水分,见 §三.3)。
> - Triton @8192 = 4N²d = 17.2 GFLOP → **48 TFLOPS**(fp16 tensor 峰 142 的 34%)= **compute-bound**(memory 地板仅 ~4µs)。
> - naive = **memory-bound**(512 MiB 中间 / 1.088ms ≈ 471 GB/s 贴带宽)。
> - 五题批改:Q1 半对 / Q2 ✓ / Q3 半对 / Q4 ✗ / Q5 ✓。

> [!warning] 本篇的"性能数字"怎么读
> **9.3× vs sdpa 有水分**:torch.sdpa 对单个 (N,64) fp16 张量走了低效 backend(dispatch 开销),3.3ms 不代表 cuBLAS 级调优——**受控对比是 naive 的 3.1×**,9.3× 只能当"特殊形状恰好跑赢库函数"的案例,不能当"Triton 碾压 PyTorch flash"的结论。另外**三条曲线都是 O(N²)**:attention 的 FLOP 本来就 O(N²d),Flash 换的不是复杂度,是**常数**(去掉 4N² 中间流量)和**哪顶帽先顶**(memory→compute)。

---

## 〇、attention 公式与朴素版的病根

单个注意力头(`head_dim=64`,非因果):

```
S = Q @ Kᵀ · (1/√64)     打分 (N×N)
P = softmax(S)             概率 (N×N)
O = P @ V                  加权和 (N×d)

朴素版病根:中间张量 S 和 P 都是 N×N,必须整个落显存
  N=8192 → S 写 1 遍 + 读 1 遍 + P 写 1 遍 + 读 1 遍 = 4N² 次搬运
  fp16 = 4×67M×2 = 512 MiB 纯中间流量 → 512/936 ≈ 0.57 ms 只搬不做别的
  而 Q/K/V 本身才 ~2 MB——99% 的搬运全浪费在中间结果
```

**Flash 核心思想一句话:S/P 的分块永远不落显存,算完的块直接 rescale 进累加器,随扫随丢。** Day3-5 的"寄存器/shared 换显存带宽",这次省的是 O(N²) 中间流量——比 matmul 的复用更狠。

---

## 一、online softmax:分母不能一次算,就边走边修正 ⭐(Q2 主场)

Day2 学过 softmax 要先减 max。但**不能先看完整行再算 max——因为那要整行落显存**。解法:分块扫过去,滚动维护三个量,每遇更大的 max 就把旧的按比例缩小:

```
朴素(全量,要整行在手里):               Flash(在线,逐块扫 N 轴):
m = max(整行 S)                            m_i:见过的最大分数(running max)
P = exp(S−m) / Σexp(S−m)                   l_i:见过的指数和(running sum)
O = P @ V                                  O:已累计的输出(running output)

每扫到一个新块 S_t:
  ① m_new = max(m_i, 本块行 max)        ← 发现更大的 max
  ② α = exp(m_i − m_new)                 ← 旧贡献的缩放因子(≤1)
  ③ O ← O×α + exp(S_t − m_new) @ V_t     ← 旧输出按新分母缩放 + 新块并进去
     l ← l×α + Σ exp(S_t − m_new)
最后:O / l                                 ← 扫完整行才一次归一
```

**为什么 α 必须做**(Q2 详解):softmax 分母是全行指数和。扫到第 2 块时发现最大分在块 3,那第 1、2 块的贡献都必须按 `exp(旧max − 新max)` 缩小,否则比例错。`m_new > m_i` 时 `α < 1`(对旧 acc/l 做衰减);`m_new == m_i` 时 `α = 1`(不动)。

---

## 二、代码逐行精读

### 变量字典(以 BLOCK_M=BLOCK_N=64, HEAD_DIM=64 为例)

| 变量/表达式 | 是什么 | 值/含义 |
|---|---|---|
| `start_m` | program 编号 = Q 行块 | 0 ~ N/64−1 |
| `offs_m/n/d` | 行/列/特征坐标向量 | 各 64 个 |
| `q_ptrs` | Q 块地址 | (64,64),**只 load 一次** |
| `k_ptrs` | **转置** K 块地址 | (64,64),`offs_d[:,None]+offs_n[None,:]` → k[d,n]=K[n,d] |
| `qk` | 打分块 | `tl.dot(q,k)·sm_scale`(BLOCK_M,BLOCK_N) |
| `m_i` / `l_i` | 在线 max / 在线行和 | (BLOCK_M,) 向量,滚动的 |
| `alpha` | 旧贡献缩放 | `exp(m_i − m_new)`,≤1 |
| `p` | 本块概率 | `exp(qk − m_new)`,cast 成 fp16 再 dot |
| `acc` | 滚动输出 | (BLOCK_M,64) fp32,rescale+累加 |
| `sm_scale` | 1/√d | 1/8 = 0.125 |

### 一个具体的数怎么算出来

```
program 0 处理 Q 的 0..63 行;沿 N 轴 128 圈(BLOCK_N=64,N=8192)
第 0 圈(n=0..63):
  qk = Q[0..63,:] @ K[0..63,:]ᵀ · 0.125   → (64,64) 分数,假设最大 m_ij=1.2
  m_new = max(−inf, 1.2) = 1.2;α = exp(−inf−1.2) = 0 → 旧 acc(0) 归零,正确
  p = exp(qk − 1.2);acc += p16 @ V[0..63,:]
  l_i = Σp
第 3 圈(n=192..255),假设本块最大 1.8 > m_i=1.2:
  α = exp(1.2 − 1.8) = 0.55 → 前 3 块的 acc/l 全乘 0.55(它们的分母从 1.2 升到 1.8)
  acc += exp(qk−1.8)16 @ V;...
扫完 128 圈:O = acc / l_i → 与朴素 softmax 完全一致(数值舍入量级见 [1])
```

---

## 三、实测与分析

### 3.1 成绩表

| N | Triton | 朴素(落显存) | sdpa | Triton/naive |
|---|---|---|---|---|
| 1024 | 0.033 | 0.094 | 0.210 | 2.8× |
| 2048 | 0.053 | 0.140 | 0.316 | 2.6× |
| 4096 | 0.104 | 0.369 | 0.888 | 3.5× |
| 8192 | 0.356 | 1.088 | 3.324 | **3.1×** |

### 3.2 Triton 是 compute-bound,不是 memory-bound(Q5 定案)

```
Triton @8192:FLOP = 4N²d = 4×67.1M×64 = 17.2 GFLOP
  0.356ms → 48 TFLOPS = fp16 tensor 峰 142 的 34%
  memory 地板:Q/K/V 全量 ~4MB ÷ 936 = ~4µs → 完全不是瓶颈
  → 时间由 FLOP(和 softmax/调度 overhead)决定 = compute-bound

朴素 @8192:512 MiB 中间 ÷ 1.088ms ≈ 471 GB/s = 峰值 50%
  → 时间由 4N² 中间流量决定 = memory-bound
```

**关键认知**:attention 的 FLOP 本来就 O(N²d),**Flash 不改变复杂度**——它去掉 4N² 中间流量,把瓶颈从 memory 翻到 compute(常数变小 + 帽切换)。这就是 [3] 里 Triton 只比 naive 快 3× 而不是 10× 的原因:两边都在 O(N²) 爬,只是 Triton 的斜率更低。

### 3.3 ⚠️ sdpa 的 9.3× 别当战功

torch.sdpa 对单个 (N,64) fp16 张量走的是低效 backend(dispatch 开销大,可能落 math 路径),3.3ms 远非 cuBLAS 级调优。**受控对比只有 naive 的 3.1×**——那是你自己写的两个 kernel、同一数据、唯一变量是"落不落中间"。9.3× 只能写进"特殊形状恰好跑赢库函数"的观察,不能写成"Triton 碾压 PyTorch flash"。

---

## 四、当日预测题复盘(五题:我的答案 → 批改 → 详解)

| 题 | 考点 | 我的答案 | 判定 | 核心缺漏 |
|---|---|---|---|---|
| Q1 | 访存账+加速比 | "K/V 共 2MB 进 L2" | 半对 | 只给了 L2 洞察,没算账、没给加速比 |
| Q2 | online softmax | "m_new 更大时 α<1,削弱旧贡献" | ✓ 满分 | — |
| Q3 | 为什么 p 用 fp16 | "混合精度,乘积 fp16 累加 fp32" | 半对 | 说了是什么,没答"为什么"(见下) |
| Q4 | head_dim 约束 | "没有 mask 会读到垃圾值" | ✗ | 机制错:是编译期 2 的幂报错,不是垃圾 |
| Q5 | 瓶颈判定 | "flash compute-bound, naive memory-bound" | ✓ 满分 | 补:两者都 O(N²),flash 换常数和帽 |

### Q1:预测 N=8192 的加速比(展开版)

```
朴素:中间 S+P = 4N² fp16 = 512 MiB → 纯搬运 512/936 ≈ 0.57ms
     + Q/K/V 读 ~4MB → 总量 ~0.57ms 起;实测 1.088ms(≈2×,softmax/launch/重叠 overhead)
Triton:Q 读一次(1MB);K/V 每 Q 块重读一次,但 K+V=2MB **留在 6MB L2** → DRAM 只付首读 2MB
     → 有效 DRAM ~4MB(4µs);瓶颈换成 17.2G FLOP ≈ 0.25ms(f16 峰)~0.36ms(带 overhead)
加速比 ≈ 1.088 / 0.356 ≈ 3×
```

你抓住的"K/V 进 L2"是对的方向——但完整的账要算到"3× 来自砍掉 512MiB 中间,而不是复杂度变了"。

### Q2:✓ 满分

`m_new > m_i` 时 `α = exp(m_i−m_new) < 1`,对旧 `acc` 和 `l` 整体衰减——因为新分母(指数和)把旧贡献的占比挤小了,必须按比例缩回来。

### Q3:半对——"为什么"的完整回答

你说"乘积 fp16 累加 fp32"是对的,但那是**怎么做**,不是**为什么**:

```
① tl.dot 的两个操作数必须同 dtype(MMA 指令要求)→ p 要和 v 一样是 fp16;
② tensor core 只收 fp16/bf16/tf32 → p 保持 fp32 只能走 TF32/FMA(Day3/5 教训);
③ acc 保持 fp32 = Day5 的黄金组合:舍入只发生在输入(p 和 v),累积不涨;
④ 代价可接受:p ∈[0,1],cast fp16 舍入 ~2⁻¹¹ ≈ 5e-4,低于 fp16 输入本身的噪声地板。
```

### Q4:✗ 修正——head_dim=80 是编译期报错,不是垃圾值

`tl.arange(0, HEAD_DIM)` 要求 **2 的幂**(Day1 修罗场 A:JIT 编译期 `ValueError: arange's range must be a power of 2`)。head_dim=80 → **编译直接失败**,根本到不了"读到垃圾"那一步。

你描述的"垃圾值"场景发生在另一种情况:**用 `next_power_of_2(80)=128` 强行扩到 128、却忘了在 `offs_d` 上加 mask**——那才会读到 80..127 的越界垃圾。真库代码(官方教程)不靠 2 的幂,而是**把 head_dim 切成 16 的块循环**:`for offs_d in range(0, head_dim, 16)` 一段段 load——这样任意 head_dim 都行,而且省 shared。

### Q5:✓ 满分,补一句

flash compute-bound / naive memory-bound,方向全对。补的认知:两条曲线都在 O(N²) 爬,Triton 只换常数和哪顶帽先顶——**这正是"优化 memory-bound kernel 的核心是少搬字节"的收官句**(Day2 融合、Day4 复用、Day6 在线 softmax 是同一招的三个变体)。

---

## 五、指标小抄(Day5 表续,新增 4 行)

| 指标 | 公式 | 本篇实例 |
|---|---|---|
| 朴素中间流量 | 4N² × dtype 字节 | 512 MiB @8192 fp16 |
| flash 有效 DRAM | Q/K/V 首读 ≈ 4MB(K/V 进 L2) | ~4µs,非瓶颈 |
| flash 实际 TFLOPS | 4N²d ÷ 时间 | 48 TFLOPS @8192 |
| 加速比(受控) | naive_time / flash_time | 3.1× @8192 |
| 与 sdpa 的差距(有水分) | sdpa_time / flash_time | 9.3×(不引为普遍结论) |

---

## 六、坑合集(Day6)

1. **sdpa 数字有水分**:单形状 dispatch 开销大,9.3× 别当普遍结论;受控对比只有 naive 3.1×。
2. **head_dim 必须是 2 的幂否则编译失败**(`tl.arange` 硬规定);真代码用 16 块循环处理任意 head_dim。
3. **`next_power_of_2` 扩维必须配 mask**,否则读到垃圾(Day1 幽灵哨兵同族)。
4. **online softmax 的 α 是正确性不是优化**:漏了它,结果会错(比例不齐)。
5. **p cast fp16 是必须**(tensor core 同 dtype),不是偷懒——但 acc 必须 fp32,否则误差随 K 涨。
6. **Flash 不改变 O(N²)**:省的是常数,别对面试说"Flash 把复杂度降到 O(N)"。
7. **`m_i` 初始 `-inf`**:双中性哨兵(Day2)在滚动 max 里又出场,首轮 α=0 自动归零旧贡献。

---

## 七、自测(闭卷过一遍)

1. 朴素版 attention 的中间流量多少?为什么它是瓶颈?(答:4N² fp16 = 512MiB@8192;Q/K/V 才 2MB)
2. online softmax 维护哪三个滚动量?α 什么时候 <1、做什么?(答:m/l/O;m_new>m_i 时 α<1,衰减旧贡献)
3. 为什么 p 必须 cast fp16 而 acc 必须 fp32?(答:tensor core 同 dtype + 累加不涨误差)
4. head_dim=80 会怎样?真代码怎么处理任意 head_dim?(答:编译期报错;切成 16 的块循环)
5. flash attention 是 compute 还是 memory bound?naive 呢?为什么两者都 O(N²)?(答:compute/memory;FLOP 本就 O(N²d),flash 换常数和帽)
6. 3.1× 和 9.3× 哪个是受控结论?为什么?(答:3.1×;sdpa 走了低效 backend)
7. Triton @8192 的 TFLOPS 和占峰?(答:17.2G/0.356ms = 48T = 142 的 34%)

---

## 八、与后续日子的连接

- **Day7 复习+面试预演**:默写 CUDA↔Triton 对照表、online softmax 三步、"会 CUDA 为什么还要 Triton"(用自己的数据:手写 k5 8.57 → Triton autotune 20→fp16 72)、flash-attn 为什么快(少搬 4N²)+ 3.1× 受控数字;LC C++ 2 道保持手感。
- **Week3 收官成绩单**:naive 0.51 → tiled 2.88 → k4 3.97 → autotune fp32 19.91 → fp16 72.04 → flash-attn 3.1×naive。
- **W4 vLLM 实战预告**:PagedAttention 就是"flash-attention 的 cache 版"——今天的分块+在线思想直接平移。

> 上一篇:[[Week3笔记-Day5-fp16与tensor-core]] · 下一篇:Day7 复习+面试预演(待续)
