---
title: Week 2 笔记(Day4-7):matmul 优化梯子——从 0.5 到 4 TFLOPS
phase: 0
week: 2
period: 2026-08-14 ~ 2026-08-16
tags:
  - ai-infra
  - 笔记
  - cuda
  - matmul
  - profiling
created: 2026-08-16
---

# Week 2 笔记(Day4-7):matmul 优化梯子——从 0.5 到 4 TFLOPS

> [!info] 笔记体例
> 概念块沿用四段式 **是什么 / 有什么用 / 怎么用 / 坑**;代码逐行讲解、每个优化都带**具体数字对比**。
> 同系列前篇:`code/ai-infra-learning/notes/Week2笔记-Day1-3-CUDA线程层级与归约.md`(线程层级/归约/指标基础在那篇)。
> 关联 [[阶段0-执行计划]] · [[个人求职路线-AI Infra]] · [[国内大厂AI-Infra岗位JD分析]]。

> [!success] Day4-7 成果
> - 三天三级:**naive 0.514 → tiled 2.88 → regtile 3.97 TFLOPS**(4096³),共 **7.7×**;
> - 拿到测量工具链:**cudaEvent + nsys + 手动 Roofline**(ncu 在 AutoDL 被权限挡死,绕行);
> - 攒下两个"**又快又错**"法医案例(508=2×254 累加器未清零、610 TFLOPS 假数据)——比跑分本身更值钱。

**怎么复现**(AutoDL 3090,CUDA 12.4):

```bash
nvcc -O3 -arch=sm_86 -o matmul_naive  matmul_naive.cu     # Day4
nvcc -O3 -arch=sm_86 -o matmul_tiled  matmul_tiled.cu     # Day5
nvcc -O3 -arch=sm_86 -o matmul_regtile matmul_regtile.cu  # Day7
nsys profile -o naive ./matmul_naive                       # Day6
nsys stats --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum naive.nsys-rep
```

---

## 〇、总览:优化梯子与实测全家福

### 0.1 梯子表(精读 Simon Boehm《How to Optimize a CUDA Matmul》并亲手复现前 4 级)

| 级 | 优化 | 一句话说清改动 | 3090 实测@4096³ | 文章 A6000@4092² | 状态 |
|---|---|---|---|---|---|
| k1 | naive | 每线程算 1 个 C 元素,直接读全局内存 | **0.514**(1.4%) | 0.31 | ✅ |
| k2 | coalescing | 调整线程→数据映射,让相邻线程读相邻地址 | (我的 k1 写法已含半个 k2) | 1.99 | 半含 |
| k3 | SMEM tiling | block 合伙把数据搬进共享内存再算 | **2.878**(8.1%) | 2.98 | ✅ 打平 |
| k4 | 1D 寄存器分块 | 每线程算 TM=8 个 C,中间结果住寄存器 | **3.97**(11.2%) | 8.47 | ✅ 差距=k5/k6 的活 |
| k5 | 2D 寄存器分块 | 每线程算 TM×TN=8×8 方块 | — | 15.97 | ⬅ 下一课 |
| k6 | float4 向量化 | 一次装载 16 字节 | — | 18.24 | 之后 |
| k9 | 自动调参 | 扫参数空间找最优配置 | — | 19.72 | 做过迷你版 |
| k10 | warp 分块 | 块内再按 warp 切分 | — | 21.78 | 之后 |
| — | tensor core | 用矩阵专用指令 | — | (文章标 WIP) | 只讲不写 |
| k0 | cuBLAS | 库的天花板 | — | 23.25 | 参照物 |

> A6000 与 3090 同为 GA102(84 vs 82 个 SM,FP32 峰值几乎相同),文章数字可直接对照。

### 0.2 存储层级:三天其实只在爬一座金字塔

```
GMEM(936 GB/s,几百 ns)                      ← Day4 naive 直接在这算
  │  block 合作搬运 + __syncthreads(Day5 tiling)
  ▼
SMEM(~19 TB/s,几十 ns,block 私有)           ← Day5 tiled 在这算
  │  每线程取 1 个数复用 8 次 + acc[] 寄存器(Day7)
  ▼
寄存器(最快,线程私有,零同步)                ← Day7 regtile 的累加器住这
  │
  ▼
FMA 计算单元
```

**每往上一层,数据离计算单元更近、复用次数更多、需要的手续(搬运/同步)越多。**

### 0.3 两尺寸全家福:Week2 最重要的实验结论

| kernel | AI(FLOP/B) | Roofline 帽 | 1024³ | 4096³ | 变化 | 解读 |
|---|---|---|---|---|---|---|
| naive | 0.25 | 自有低帽(烂访问) | 0.515 | 0.514 | **不动** | 早已贴住自己的帽 |
| tiled | 4 | 3.74 | 2.82 | 2.878 | **不动** | 贴带宽帽(77%),尺寸不改变贴墙事实 |
| regtile | 16 | ~15 | 2.94 | **3.97** | **+35%** | 唯一动了的:小尺寸饿着,大尺寸吃到新帽 |

> [!tip] 尺寸方法论(一句话)
> **Roofline 帽(强度)不随矩阵大小变;尺寸改变的只有一件事——你有没有足够的并行度(波数)去吃到帽。**
> 波数 waves = 总 block 数 ÷ SM 数(82)。regtile@1024³ 只有 (1024/64)²=256 个 block ≈ **3 波**,启动/排队的摊销都压不平;@4096³ 有 (4096/64)²=4096 个 block ≈ **50 波**,才把 16 的 AI 帽吃出一部分。另一个因素是 L2(6MB):1024³ 时 A+B+C 共 12MB,局部性高、L2 大量命中,真实 DRAM 压力小,访存优化的差距显不出来;4096³ 单矩阵 64MB,L2 只剩过滤作用,压力真实存在。
> 推论:**benchmark 必须在大压力尺寸下做**——文章的梯子数字全是 4092² 上测的,不是巧合。

---

## 一、matmul 是什么、怎么算(把地基打牢)

**是什么**:矩阵乘法 `C = A × B`。A 是 M×K,B 是 K×N,C 是 M×N。每个元素是一个**点积**:

```
C[row][col] = Σ_{k=0..K-1}  A[row][k] × B[k][col]
```

**具体数字举例(2×2 迷你版)**:

```
A = [1 2]      B = [5 6]      C[0][0] = 1×5 + 2×7 = 19
    [3 4]          [7 8]      C[0][1] = 1×6 + 2×8 = 22
                             C[1][0] = 3×5 + 4×7 = 43
```

每个 C 元素要 K 次乘 + K−1 次加 ≈ **2K 次 FLOP**。

**行主序(row-major)展开**(C++ 二维数组就是这样躺在一维内存里的):

```
A[row][k]  →  A[row * K + k]     一行内的元素在内存里相邻
B[k][col]  →  B[k * N + col]     同一列的元素在内存里相距 N×4 字节(跨步!)
C[row][col]→  C[row * N + col]
```

例:1024×1024 时,`B[k][col]` 走 k 时每次跳 1024×4 = **4096 字节**——"按列读"在行主序里是跳跃式访问,这是 naive 慢的根源之一。

**总 FLOP 数**:M×N 个输出 × 每个 2K → **2MNK**。1024³:2×1024³ ≈ 2.147 GFLOP;4096³:2×4096³ ≈ 137.4 GFLOP。

**有什么用**:matmul(GEMM)是深度学习的算力大头——全连接层、attention 的投影、卷积(im2col 后)全是它。**会优化 matmul = 理解了 GPU 的一切主要机制**(存储层级、同步、warp、bank、tensor core),所以它是 JD 里"CUDA 算子"的经典面试题,也是本文选它当主线的原因。

---

## 二、Day4:`matmul_naive.cu` 逐行精读

### 2.1 kernel 逐行

```cuda
__global__ void matmul_naive(const float* A, const float* B, float* C,
                             int M, int N, int K) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;   // 我负责 C 的哪一行
    int col = blockDim.x * blockIdx.x + threadIdx.x;   // 我负责 C 的哪一列
    if (row < M && col < N) {                          // 越界保护(向上取整的余料)
        float sum = 0.0f;                              // 累加器(1 个寄存器)
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];    // 点积主体
        }
        C[row * N + col] = sum;                        // 写回(只写 1 次)
    }
}
```

**变量字典**:

| 变量 | 是什么 | 1024³ 例子里的值 |
|---|---|---|
| M, N, K | A 的行、B 的列、公共(收缩)维 | 1024, 1024, 1024 |
| row / col | 本线程负责的 C 元素坐标 | 见下例 |
| `sum` | 点积累加器,整个 K 循环住寄存器 | 最后 = C[row][col] |
| `dim3 block(16,16)` | 每 block 16×16=256 线程,二维排布 | — |
| `dim3 grid(N/16, M/16)` | 64×64=4096 个 block 覆盖整个 C | (1024³ 时) |

**具体例子(线程是谁、算哪个数)**:block(3, 5) 里的 thread(y=4, x=6):

```
row = 16×5 + 4 = 84        ← blockIdx.y 管行
col = 16×3 + 6 = 54        ← blockIdx.x 管列
这个线程算 C[84][54] = Σ_k A[84][k]·B[k][54],1024 项点积,写回 C[84×1024+54]
```

二维 grid 只是全局索引公式在两个方向各来一遍(x 管列、y 管行),没有新魔法。

### 2.2 warmup + cudaEvent 计时样板(以后每个 benchmark 都长这样)

```cuda
matmul_naive<<<grid, block>>>(...);   cudaDeviceSynchronize();  // ① warmup:首次启动含
                                                               //    上下文初始化/时钟未拉满,不计时
cudaEventRecord(start);                                           // ② GPU 流里打"开始"戳
matmul_naive<<<grid, block>>>(...);                               // ③ 正式测量的一次
cudaEventRecord(stop);  cudaEventSynchronize(stop);               // ④ "结束"戳 + CPU 等到它
cudaEventElapsedTime(&ms, start, stop);                           // ⑤ 两个 GPU 时间戳之差
```

为什么要 warmup:第一次启动 kernel 要初始化 GPU 上下文、驱动还没把频率拉满,直接计时会虚高。**先跑一遍不计时,再跑一遍计时。**

### 2.3 性能指标怎么算(以实测数走一遍)

```
耗时 ms = 4.173(1024³,nsys 实测 avg)
TFLOPS = 2MNK / t = 2×1024³ / 4.173ms = 2.147e9 / 4.173e-3 s = 0.515 TFLOPS
利用率 = 0.515 / 35.6 = 1.45%   ← 百分之一点五!GPU 几乎在散步
```

### 2.4 naive 为什么慢:访存账本 + AI 推导

每个线程:读 A 一整行(K×4B=4KB)+ 读 B 一整列(K×4B,但跨步 4KB)+ 写 1 个 float。整个 kernel 里:
- A 的每个元素被重复读 **N 次**(每一列的线程都要它),B 的每个元素被重复读 **M 次**;
- **算术强度**:

```
AI = FLOP ÷ 搬运字节
   = 每 C 元素 2K 次 FLOP ÷ (A 行 4K B + B 列 4K B + 写 4 B)     [以每输出计]
   = 2048 ÷ 8196 ≈ 0.25 FLOP/B
```

对照 3090 平衡点 35.6T ÷ 936G ≈ **38 FLOP/B**:AI=0.25 意味着理论带宽帽只有 0.25×936G ≈ **0.234 TFLOPS**——实测 0.515 反而**超**了这顶帽,说明 L2 缓存消化了过半流量(不然更惨)。加上 B 的跨步访问不 coalesce,naive 是"帽低 + 访问烂"双重受害。

> [!note] 一个写法细节:col 用 x 还是 y
> 我的 naive 把 col 绑在 threadIdx.x 上:同一 warp 里相邻线程的 col 连续 → **B 的读是连续地址**(coalesced),A 的读变成同行广播(免费)。文章的 k1 反过来绑,所以它的 k2(coalescing)能快 6 倍,而我的 k1 已经"含了半个 k2"——这解释了为什么我的 naive(0.51)比文章的(0.31)快 60%。
> **教训:同一算法,线程→数据的映射方向不同,性能可以差几倍。**

**坑**:
- Day4 实战最大教训:第一次跑出 GPU 结果和 CPU 对不上(差 4 倍),靠**校验机制**拦下——发现是 baseline 本身编错了。**任何 benchmark 前先验证正确性,数字才有意义。**
- `if (row<M && col<N)` 不能省:grid 用了向上取整,边缘 block 必有越界线程。

---

## 三、Day5:`matmul_tiled.cu`(shared memory tiling)

### 3.1 优化了什么:block 合伙搬数据,摊薄访存

**观察**:naive 里 A 的一行被同一 block 里 16 个线程(y 相同)各自重复读;B 的一列也被 16 个线程(x 相同)重复读。**一个 16×16 的 block 算 256 个 C,其实只需要 A 的 16 行 × B 的 16 列。**

**药方**:每个 phase,block 的 256 个线程**合作搬运** A 的 16×16 片 + B 的 16×16 片进 shared memory(每人搬 1+1 个),然后 256 次点积分步全部在 SMEM 上做——GMEM 流量暴降。

```
          B 的 16×16 片
              ┌────┐
   A 的       │    │     每个 phase:搬 2×16×16 = 512 个数进 SMEM
   16×16 片 ──┤    ├──   做 16×16×16 = 4096 次 FMA(全在 SMEM 上)
              └────┘     K=1024 → 64 个 phase 轮流推进
```

### 3.2 逐行精读

```cuda
__shared__ float As[16][16];   // A 的片(block 私有,片上)
__shared__ float Bs[16][16];   // B 的片

int row = blockIdx.y * blockDim.y + threadIdx.y;   // 同 naive
int col = blockIdx.x * blockDim.x + threadIdx.x;
float sum = 0.0f;

for (int ph = 0; ph < (K + TILE - 1) / TILE; ++ph) {   // K=1024 → 64 个 phase
    // ① 合作搬运:每线程搬 As、Bs 各 1 个元素(带边界保护,越界填 0)
    As[threadIdx.y][threadIdx.x] = (row < M && ph*16 + threadIdx.x < K)
        ? A[row * K + ph*16 + threadIdx.x] : 0.0f;
    Bs[threadIdx.y][threadIdx.x] = (ph*16 + threadIdx.y < K && col < N)
        ? B[(ph*16 + threadIdx.y) * N + col] : 0.0f;

    __syncthreads();               // ⭐ 屏障一:全员搬完才能开始算

    for (int k = 0; k < TILE; ++k) // ② 16 次乘加,全在 SMEM 上(快!)
        sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

    __syncthreads();               // ⭐ 屏障二:全员算完才能进下一轮搬运
}
C[row * N + col] = sum;            // 最后一次性写回
```

**具体数字走一遍(block(2,3) 的 thread(y=4, x=6),ph=0)**:

```
row = 3×16+4 = 52,col = 2×16+6 = 38          ← 它算 C[52][38]
搬运:As[4][6] ← A[52×1024 + 0×16 + 6] = A[52][6]
      Bs[4][6] ← B[(0×16+4)×1024 + 38] = B[4][38]
计算:k=0..15 → sum += As[4][k]×Bs[k][6]
      即 A[52][k]×B[k][38],k=0..15            ← 正是 C[52][38] 点积的前 16 项 ✔
ph=1 搬 k=16..31 的片,再乘 16 项 …… 64 个 phase 后,sum = 完整点积
```

**双屏障为什么一道都不能少**:
- 屏障一防"有人还没把片搬完,别人已经开始读 SMEM"(读到垃圾);
- 屏障二防"有人算完本 phase 就抢先搬下一 phase,**把别人还在读的 As/Bs 覆盖了**"。
- 删屏障二,小程序可能照样对(时序运气),大矩阵必错——**正确性问题不会每次都显形,这是最阴险的一类 bug**。

### 3.3 账本对比 + 实测

AI(block 级,忽略写回):每 block 每 phase 搬 2×16×16 个数、做 2×16×16×16 次 FLOP;整个 K 维合计:

```
AI = 2×16×16×K FLOP ÷ (4B × 2×16×K 字节) = 512K / 128K = 4 FLOP/B
```

GMEM 流量比 naive 降 16 倍(AI 0.25→4)。带宽帽 = 4×936G = **3.74 TFLOPS**。

| 尺寸 | 实测 | 占带宽帽 | 备注 |
|---|---|---|---|
| 1024³ | 2.82 TFLOPS(0.763ms) | 75% | Day6 用 nsys 独立复核过 |
| 4096³ | 2.878 TFLOPS(47.75ms) | 77% | 换尺寸基本不动 = 贴墙实锤 |

对比 naive(4096³ 反推 0.514):**加速 5.6×**,和文章 k1→k3(0.31→2.98,9.6×,其中 k2 coalescing 占一截)方向一致。

**剩余瓶颈(Day5 时的三个猜想,Day6/7 验证)**:
1. tensor core 没用上(把算力帽抬一个量级的乘数);
2. 访存合并程度(我的写法已含半个 k2,基本无罪);
3. **计算密度还不够高(AI 只有 4,撞在带宽墙上)← Day6 判定为主犯**。
另有两个小尾巴:if 边界检查造成轻度 warp divergence;每线程只产 1 个输出、SMEM 读多但 GMEM 读没摊够。

---

## 四、Day6:给程序做体检(nsys + 手动 Roofline)

### 4.1 nsys 是什么、怎么用

**是什么**:Nsight Systems,**程序级时间线录像**——CPU 何时下发、GPU 何时算 kernel、memcpy 占多少,全录下来。**它只记时间戳,不碰硬件计数器**,所以在 AutoDL 也能用。

```bash
nsys profile -o naive ./matmul_naive          # 生成 naive.nsys-rep(录像)
nsys stats --report cuda_gpu_kern_sum \       # 表1:各 kernel 耗时统计
           --report cuda_gpu_mem_time_sum \   # 表2:memcpy/memset 耗时统计
           naive.nsys-rep
```

**表里每列怎么算**(以实测 naive kernel 行为例,Instances=2:warmup+测量):

```
Total = 4,170,892 + 4,175,785 = 8,346,677 ns
Avg   = Total ÷ 2 = 4,173,338.5 ns
Med   = n=2 时 = 两数均值 = Avg(n 大了才会和 Avg 分开)
Min/Max = 4,170,892 / 4,175,785
StdDev = 样本标准差(分母 n−1!):
         两次偏离均值 ±2,446.5 → sqrt((2446.5²+2446.5²)/1) = 2446.5×√2 = 3,459.9 ✔
Time% = 本行 Total ÷ 本表所有行 Total(分母不含另一张表,两表各算各的)
```

> 单次时长是 CUPTI 记录的 **GPU 侧时间戳**(kernel 开始→结束,不含 launch 下发延迟),和 cudaEvent 同源——所以下面能互验。

### 4.2 实测三结论(1024³)

**① cudaEvent ↔ nsys 互验,两把尺子量出同一个数**:

```
nsys:naive 4.173ms → 0.515 TFLOPS;  tiled 0.763ms → 2.82 TFLOPS
cudaEvent(Day5):    0.514                /          2.79           ← 差 <1% ✔
```

**② memcpy 是 PCIe 特征,不是 GPU 算的**:H2D 均值 366µs 搬 4MiB → **11.5 GB/s**,典型 PCIe 3.0 x16 有效带宽。(单次 memcpy 数字噪声大,别过度解读。)

**③ Amdahl 定律活教材——kernel 提速后,拷贝成了另一半**:

```
naive:GPU busy = 8.347ms kernel + 1.078ms memops → memops 占 11.4%
tiled:GPU busy = 1.526ms kernel + 1.537ms memops → memops 占 50.2%  ← 一半!
```

kernel 快 5.5×、拷贝纹丝不动,占比自然翻转。生产里为什么**权重常驻显存、能不搬就不搬**(vLLM 模型加载一次、之后全 GPU 内部流转):就是要把这 50% 压回 0。

### 4.3 ncu 为什么用不了 + Plan B

`ncu`(Nsight Compute,单 kernel 显微镜,给 SOL 读数)在 AutoDL 报 `ERR_NVGPUCTRPERM`:GPU 性能计数器是**侧信道**(能推断同卡其他进程行为),驱动默认只许宿主机管理员访问。容器内 root 也无解,多租户平台不会开。

**Plan B:手动 Roofline**——纯算术,不依赖权限,面试还能当场画:

```
                    算力帽 35.6 TFLOPS ━━━━━━━━━━━━━━━━━┓
 TFLOPS                                                 ┃
   ▲                    ╱ 算力墙(平台段)              ┃
   │                   ╱                                ┃
 3.74 ───────────────╱←── tiled(贴这)                   ┃
   │              ╱  斜率 = 带宽 936 GB/s                ┃
   │           ╱                                       ┃
0.234 ─────╱←── naive(L2 帮忙才超出)                    ┃
   └──────┴──────┴──────┴──────┴──────┴──▶ AI(FLOP/B)
          0.25   4    16     32   38(平衡点)
```

**三分法诊断**(背下来):

| 读数 | 诊断 | 药方 |
|---|---|---|
| 带宽帽占用高、算力帽低 | memory-bound | tiling、coalesce、shared memory |
| 算力帽高、带宽帽低 | compute-bound | 省指令、tensor core |
| **两个都低** | latency/issue-bound(在干等) | 加并行度、打破依赖链、减指令 |

我们的三个 kernel 用这套诊断:naive 两帽都没贴 → **latency-bound**;tiled 贴带宽帽 77% → **memory-bound**;regtile 帽 26%/算力 11% → **issue-bound**(见 §5.5)。

### 4.4 `profile_demo.cu`:自造标准答案验证工具

学测量最好的方式:**先知道答案,再看工具测不测得出来**。demo 造了一对 kernel——

- `axpy_kernel`(memory-bound):每线程读 2 写 1(12B)只换 2 FLOP → AI = 2/12 ≈ **0.167**,必然卡带宽;
- `compute_kernel`(compute-bound):寄存器里 4 条**独立** FMA 链各跑 2000 步,几乎不碰内存 → 算力侧吃满。开 4 条链是因为 1 条链每个 FMA 都要等上一个结果(依赖链),4 条才能流水并行喂饱 SM(**指令级并行 ILP** 的最小示例)。

(在能开计数器的机器上,ncu 的 SOL 读数应是:axpy 的 Memory ~70–90%、compute 的 SM 高——这就是这对 kernel 的验收标准。)

---

## 五、Day7:`matmul_regtile.cu`(1D 寄存器分块)

### 5.1 动机:AI 的"面积/周长"定律

一个 block 算 BM×BN 的 C,每轮从 GMEM 搬 (BM+BN)×BK 个数,做 2×BM×BN×BK 次 FLOP:

```
AI = 2·BM·BN·BK ÷ (4B · (BM+BN) · BK) = 2·BM·BN / (4·(BM+BN))
                                        ─────   BK 消掉了!──
```

**BK 约掉了——AI 只由方块形状决定**。类比街区:**楼(计算)随边长平方涨,临街建材(访存)随边长线性涨**。

```
   16×16(Day5)     64×64(Day7)      128×128(k5 目标)
   ┌──┐            ┌────┐          ┌──────┐
   │  │面积256      │    │4096      │      │16384
   └──┘            └────┘          └──────┘
   AI = 4           AI = 16          AI = 32
```

### 5.2 为什么必须动用"寄存器"(Q4 的账)

想上 128×128 的方块,两条路都撞墙:

```
路 A:每线程仍算 1 个输出 → 需要 128×128 = 16384 线程(block 上限 1024,✕ 超了 16 倍)
路 B:每线程算 TM=8 个   → 需要 16384/8 = 2048 线程(✕ 仍超上限,Day7 实验 3 亲撞)
```

**解法:让每个线程管一个 TM×TN=8×8 的小方块** → 16384/64 = 256 线程 ✔——这就是 k5"2D 寄存器分块"(下一课)。今天的 k4 先走 1D 版:64×64 方块 + 每线程一行 TM=8 个,**中间结果放寄存器**:

- 寄存器 vs SMEM:同在片上,但寄存器**更快、线程私有、零同步零开销**;
- 代价:寄存器总量有限(每线程上限 255 个,超了会被**溢出(spill)**到本地内存,性能反而崩)——所以 `__launch_bounds__(512)` 提示编译器按 512 线程的预算分配寄存器。

### 5.3 逐行精读(BM=BN=64, BK=8, TM=8 → 512 线程)

```cuda
__shared__ float As[64][8];    // A 的 64×8 切片(512 个)
__shared__ float Bs[8][64];    // B 的 8×64 切片(512 个)

// 512 人分到 64×64 的输出方块上:每行 8 个线程,每人管本行连续 8 列
const int threadRow = threadIdx.x / (BN / TM);         // tid/8  → 0..63
const int threadCol = (threadIdx.x % (BN / TM)) * TM;  // (tid%8)×8 → 0,8,…,56

float acc[TM] = {0.f};   // 8 个累加器 = 8 个寄存器,整个 K 循环不出寄存器

for (int bk = 0; bk < K; bk += BK) {                   // K=1024,BK=8 → 128 轮
    As[threadIdx.x/8][threadIdx.x%8] = A[(blockIdx.y*64 + threadIdx.x/8)*K + bk + threadIdx.x%8];
    Bs[threadIdx.x/64][threadIdx.x%64] = B[(bk + threadIdx.x/64)*N + blockIdx.x*64 + threadIdx.x%64];
    __syncthreads();                                   // 屏障一:搬完才算

    for (int kk = 0; kk < BK; ++kk) {
        float a = As[threadRow][kk];                   // ★ 灵魂①:取 1 个 a
        for (int i = 0; i < TM; ++i)                   // ★ 灵魂②:连用 8 次
            acc[i] += a * Bs[kk][threadCol + i];
    }
    __syncthreads();                                   // 屏障二:算完才许覆盖
}
for (int i = 0; i < TM; ++i) C[globalRow*N + globalCol + i] = acc[i];
```

**具体数字全程走一遍(blockIdx=(1,2),threadIdx.x=37)**:

```
threadRow = 37/8 = 4;   threadCol = (37%8)×8 = 40
globalRow = 1×64 + 4 = 68;   globalCol = 2×64 + 40 = 168
→ 这个线程负责 C[68][168..175] 共 8 个数,acc[0..7] 分别对应它们

搬运(loadA=37):As[4][5] ← A[68×K + bk+5]     ← A 第 68 行、第 bk+5 列
      (loadB=37):Bs[0][37] ← B[(bk+0)×N + 168+37=205]  …等等,Bs 的 37 列
计算(kk=3):a = As[4][3] = A[68][bk+3]
      acc[i] += a × Bs[3][40+i] = a × B[(bk+3)×N + 168+i]   i=0..7
→ 加的正是 C[68][168+i] 点积中 k=bk+3 那一项 ✔
```

**灵魂在两行**:`a` 从 SMEM 读**一次**,喂 8 条 FMA——**每次 SMEM 读摊到 8 个计算上**(tiled 里是 1:1);`acc[8]` 全程住寄存器,只在 128 轮全部结束后写一次 GMEM(每输出 4B 摊到 2K 次 FLOP 上,可忽略)。

### 5.4 实测与账本

| 尺寸 | 耗时 | TFLOPS | 波数 | 说明 |
|---|---|---|---|---|
| 1024³ | 0.731ms | 2.94 | 256 block/82 ≈ 3 波 | 和 tiled 挤在一起(见 0.3 尺寸方法论) |
| 4096³ | 34.63ms | **3.97**(11.2%) | 4096/82 ≈ 50 波 | 比 tiled 净胜 **1.38×** |

Roofline 复查:AI=16 → 带宽帽 15 TFLOPS,实测 3.97 只占 **26%**;算力帽占 11%——**两帽都没贴 → 已脱离带宽墙,新瓶颈是 issue/latency-bound(指令喂不饱管线)**。这正是 k5/k6 要修的:

1. **Bs 读取 2-way bank conflict**:warp 内 32 线程读 `Bs[kk][threadCol+i]`,实际只有 8 个不同的字(列 0,8,…,56 各被 4 个线程重复读),它们按 8 float=32B 间隔落进 **4 个 bank、每 bank 2 个字** → 这条 load 占 2 周期而非 1(k5 用 2D 方块 + 更紧的线程排布缓解);
2. **A 装载的 wavefront 多**:每 warp 4 笔 32B 事务而非 1 笔 128B——字节没浪费(32B 恰好满 sector),但**发射槽**多耗了 4 倍(k6 的 float4 一次搬 16B 正面修它);
3. **load/FMA 指令比 1:1**:每条 FMA 配一条 SMEM load。k5 让每个 a、b 的复用翻 8 倍(load 减半),k6 再砍装载指令 4 倍——**梯子后几级修的全是"指令发射"这一件事**。

> 文章 k4 是 8.47,我们 3.97——差的 2.1× 不是代码错,是它的 k4 数字本来就含着后面几级的活(作者按叙事顺序拆开讲)。我们的 3.97 是一张干净的"k4 之前"照片,做了 k5 再拍一张,差值就是那级的净贡献。

### 5.5 两个"又快又错"法医案例(本周最值钱的两课)

**案例 1:508.276 = 2 × 254.138(实验 1 的指纹)**

实验:删掉 `acc[]`,改成每轮 kk 直接 `C[...] += ...`(想看寄存器值多少钱)。
结果:0.824 TFLOPS(**3.6× 坍塌**——寄存器价值的实证),但校验打出 `GPU=508.276, CPU=254.138`。

法医:**精确两倍** = kernel 跑了两遍(warmup+测量)各加一遍,而 **C 从未清零**。`+=` 型 kernel 的输出必须先 `cudaMemset(d_C, 0, ...)`。计时本身仍有效(cudaEvent 不管对错),但结果被污染。
教训:**结果里的整数倍关系,几乎总是"多算/少算了几遍"的指纹**。

**案例 2:610 TFLOPS(实验 3 的指纹)**

实验:BM=BN=128、TM=8 → `dim3 block(2048)`。
结果:`耗时 0.00352ms、610 TFLOPS、C[0][0]=0`。

法医:2048 > 1024 线程上限 → **launch 当场失败,kernel 一个指令都没执行**。三指纹:①耗时是事件计时的开销地板(µs 级);②610 超过硬件峰值 35.6 达 17 倍,**物理不可能**;③C 没被写过(新页通常是 0,但**不保证**——别依赖 malloc 送零)。
教训:
- **任何超过硬件峰值的数字,不用查代码,先断定测量是假的**;
- **launch 参数错误不崩程序、不报错**——`cudaDeviceSynchronize()` 未必能抓到,必须 launch 后立即:

```cuda
cudaError_t e = cudaGetLastError();   // invalid configuration argument 只在这冒头
if (e != cudaSuccess) { printf("launch: %s\n", cudaGetErrorString(e)); return 1; }
```

(此前那个"C[0][0]=128.142≈一半、9.49 TFLOPS"的版本则是另一案:BK 翻倍但每线程仍只搬 1 个 → SMEM 一半是陈旧数据 → 一半 k 项没被算——**又快又错**的另一半:少干活的快不是快。)

### 5.6 验证方法论(从这两案长出来的规矩)

- 单点验证(C[0][0])挡不住系统性错误,两个案例都是靠它**碰巧**抓到的;
- 升级:**1024 规模** CPU 全算,报 `max|C_gpu − C_cpu| < 1e-3`;**4096 规模** CPU 全算太慢,随机抽 256 个位置各验一个点积;
- 以后每个 kernel 变体**先过验证门,再进 benchmark**。

---

## 六、性能指标小抄(matmul 版,含本篇实测数)

| 指标 | 公式 | 实例 |
|---|---|---|
| FLOP 数 | 2MNK | 4096³ → 137.4 GFLOP |
| TFLOPS | 2MNK / t | 137.4G / 34.63ms = **3.97** |
| 利用率 | TFLOPS ÷ FP32 峰值 | 3.97/35.6 = 11.2% |
| 算术强度 AI(block 级) | 2·BM·BN / (4·(BM+BN))(BK 消掉) | 16×16→4;64×64→16;128×128→32 |
| 带宽帽 | AI × 936 GB/s | AI4→3.74;AI16→15 |
| 贴墙判定 | 实测 ÷ 帽 | tiled 77%(贴);regtile 26%(没贴) |
| 平衡点 | 峰值算力 ÷ 峰值带宽 | 35.6T/936G ≈ 38 FLOP/B |
| 波数 | 总 block ÷ 82 SM | 256/82≈3;4096/82≈50 |
| 加速比 | t_旧 ÷ t_新 | naive→regtile = 267.4/34.63 ≈ **7.7×** |
| memops 占比 | memops ÷ (kernels+memops) | naive 11.4% → tiled 50.2% |
| nsys StdDev | 样本标准差(÷n−1) | 2446.5×√2 = 3459.9 ✔ |

**RTX 3090 速查(sm_86/GA102)**:82 SM · FP32 35.6 TFLOPS · 带宽 936 GB/s · L2 6MB · block≤1024 线程 · warp=32 · 寄存器 64K×4B/SM(每线程≤255)。

---

## 七、坑合集(Day4-7)

1. **不做 warmup 就计时** → 首次启动开销(上下文/频率)污染数据。
2. **不校验就信数字** → Day4 靠校验拦下 4 倍错的 baseline;规矩:先对答案,再看跑分。
3. **`+=` 型 kernel 不清零输出** → 508=2×254 惨案;先 `cudaMemset`。
4. **launch 错误无声** → 2048>1024 不崩不报,`cudaGetLastError()` 必加。
5. **超硬件峰值的数字必是假测量** → 610 TFLOPS 案;物理不可能不需要查代码。
6. **小矩阵测访存优化** → 1024³ 时三 kernel 挤在一起(3 波 + L2 高命中);**benchmark 要在大压力尺寸下做**(文章用 4092²)。
7. **对比不同条件** → 对比必须同尺寸、同编译选项(`-O3 -arch=sm_86`)、同计时法。
8. **删屏障靠运气** → 双屏障少一道,小矩阵可能对、大矩阵必错;正确性 bug 不显形 ≠ 不存在。
9. **依赖 malloc 清零** → 新页"通常"是 0 但不保证;要零就显式 memset。
10. **单点验证当全验** → 升级为全矩阵 max|err| 或随机抽样点积。

---

## 八、自测(闭卷过一遍)

1. 1024³ 的 matmul 总 FLOP 是多少?为什么有系数 2?(答:2×1024³≈2.147G;乘和加各算一次)
2. 手算 BM=BN=128 的 AI,并说明它对 3090 意味着什么。(答:2·128²/(4·256)=32<平衡点 38 → 即便做满也先撞带宽侧,FP32 路线甜点就在 128×128 附近)
3. tiled 的两道 `__syncthreads()` 各防什么?删第二道会怎样?
4. regtile 中 threadIdx.x=37 的线程:threadRow/threadCol 各是多少,负责哪 8 个输出?(答:4 和 40;block 内第 4 行、第 40..47 列)
5. 一个 kernel 测出 40 TFLOPS(3090),你第一反应是什么?(答:假测量——超峰值 35.6,先查 kernel 是否真的跑了/计时是否测到了它)
6. 为什么 regtile@1024³ 只有 2.94、@4096³ 涨到 3.97,而 tiled 两个尺寸都不动?(答:帽不随尺寸变;尺寸只改波数——3 波饿着 vs 50 波吃到;tiled 两尺寸都贴着带宽帽)
7. nsys 的 StdDev 是总体还是样本标准差?怎么用两组数据验证?(答:样本(n−1);n=2 时 = |a−b|/√2… 即 2×单边偏离×√2,用 2446.5×√2=3459.9 验过)

---

## 九、下一步

- **k5 2D 寄存器分块**(下一课):每线程 8×8 方块、256 线程——Q4 的账已经推出它的形状;修 load/FMA 指令比 + bank conflict,预期 5~8 TFLOPS;
- k6 float4 向量化 → k9 autotune(扫 BM/BN/BK/TM 参数网格)→ k10 warp 分块 → tensor core(mma 以 warp 为单位发射,k5/k10 的结构直接映射过去);
- 与求职主线的连接:这套"**预测 → 实测 → 定位瓶颈 → 单变量优化 → 复测**"的循环,加上 nsys/Roofline 的表述,就是面试里"你怎么优化一个慢 kernel"的标准答案骨架;vLLM 的 GEMM/attention kernel 调优同理。

> 前篇:`code/ai-infra-learning/notes/Week2笔记-Day1-3-CUDA线程层级与归约.md`
