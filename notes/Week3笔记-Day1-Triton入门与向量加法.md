---
title: Week 3 笔记(Day1):Triton 入门与向量加法
phase: 0
week: 3
period: 2026-08-18
tags:
  - ai-infra
  - 笔记
  - triton
created: 2026-08-18
---

# Week 3 笔记(Day1):Triton 入门与向量加法

> [!info] 笔记体例
> 概念块沿用四段式 **是什么 / 有什么用 / 怎么用 / 坑**;代码块采用 **逐行讲解 + 具体数字举例**。
> 关联 [[Week2笔记-Day1-3-CUDA线程层级与归约]] · [[Week2笔记-Day4-7-matmul优化梯子]] · [[阶段0-执行计划]] · [[个人求职路线-AI Infra]]。

> [!success] Day1 成绩
> - triton **3.1.0**(torch 2.5.1 自带)跑通第一个 Triton kernel `vector_add.py`,N=4096/4097 双 PASS(考验 mask)。
> - 2²⁶ 元素实测 **0.949 ms = 849 GB/s = 3090 峰值带宽的 90.7%**,离理论地板 0.860 ms 只有 **1.10×**——与 torch 原生 `a+b`(0.947 ms)死平。结论:Triton 编译器开箱即用就能把 memory-bound kernel 编到贴墙,和手写 CUDA 没差别。
> - 三个法医案例:非 2 幂(编译期报错)、`floor-div` 丢尾(1/4097 元素错)、删 mask(静默越界写)。

> [!warning] 本篇的"性能数字"怎么读
> 向量加法是**纯 memory-bound** kernel:时间几乎全由"搬多少字节 ÷ 带宽"决定,和调度、分块基本无关。所以本篇的性能部分重点不是"优化了多少"(没有梯子),而是**指标怎么算、怎么从数字反推事实**——修罗场 B 里凭空出现的 0.033 ms,就是用算术指纹破的案(§四)。

---

## 〇、Triton 是什么:把 Week2 的手工艺外包给编译器

**是什么**:单卡 GPU 算子 DSL——用 Python 写 kernel,编译器替你管 shared memory 分配、屏障同步、warp 调度、向量化访存。编译管线:`AST → TTIR → TTGIR → LLVM IR → PTX → cubin`(对比 CUDA 的 nvcc 一步到 PTX)。

**有什么用**(求职视角):
- `torch.compile` 底层生成的就是 Triton kernel;
- vLLM、unsloth 等推理框架的大量算子是 Triton 写的;
- JD 里"熟练 Triton"的出现频率已不低于 CUDA。

**怎么用**:`@triton.jit` 标记 GPU 函数,`kernel[grid](...)` 启动。开发循环从 nvcc 的分钟级降到秒级——**保存即跑**。

**边界**(面试加分点):跨 block 通信、warp 级精细控制、cudaGraphs 仍是 CUDA 的地盘。行业配置是"**CUDA 懂原理 + Triton 出活**"——前两周的 CUDA 是看懂 Triton 编译器在帮什么的钥匙。面试标准答法:"我会手写 CUDA,所以我清楚 Triton 替我省掉了哪些手工艺、以及它省不掉什么。"

### CUDA ↔ Triton 概念对照表(本篇核心产出,后续每天续行)

| 你写过的 CUDA(Week1/2) | Triton 写法 | 一句话 |
|---|---|---|
| `__global__ void k(...)` | `@triton.jit` | GPU 函数;**首次调用才 JIT 编译** |
| `int i = blockIdx.x*blockDim.x+threadIdx.x` | `pid*BLOCK + tl.arange(0,BLOCK)` | 标量索引 → **向量索引** |
| `if (i < n)` | `mask = offs < n` 塞给 load/store | 越界保护的向量版 |
| `c[i]=a[i]+b[i]` | `output = x + y` | 标量算术 → NumPy 同款**块算术** |
| `k<<<numBlocks, threads>>>(...)` | `add_kernel[grid](...)` | launch 语法糖 |
| `(N+t-1)/t` 向上取整 | `triton.cdiv(n, B)` | 一模一样 |
| cudaMalloc×3 / memcpy×3 / cudaFree×3 | `torch.empty_like` + 全程 GPU + GC | 六步骨架 → 三行(Day4 的 RAII 在 GPU 世界回魂) |
| `__shared__` + 装载循环 + `__syncthreads()` | **没有对应物,不写** | 编译器自动插(Day3 `tl.dot` 见真章) |
| "怎么分 warp、要不要 float4" | `num_warps` 参数(默认 4) | 调度外包,Day4 当旋钮 |
| nvcc 编译 → ./a.out | `python3 xxx.py` | 开发循环分钟级 → 秒级 |

### JIT 与 `tl.constexpr`(编译期 vs 运行期,Day4 autotune 的地基)

- `BLOCK_SIZE: tl.constexpr` = 编译期常量,**换一个值 = 编译一个全新 kernel 版本**。所以 BLOCK 扫描脚本里每个 config 的首次调用都含一次编译(warmup 吸收掉)。
- autotune 的原理就是它:自动帮你试不同的 constexpr 组合。
- 实验证据(修罗场 A):传 BLOCK=1000,报 `ValueError: arange's range must be a power of 2`,栈里是 `ast_to_ttir` → `CompilationError`——错误发生在**首次调用时的编译阶段**,既不是 Python 解析时、也不是 GPU 运行时。2 的幂是硬性规定(块内要做树状归约/向量化,非 2 幂打散这些优化)。

---

## 一、心智模型:从"一根线程算一个数"到"一个 program 算一整块" ⭐

```
   CUDA(w1d6,标量思维)              Triton(块思维)
   ─────────────────────────         ─────────────────────────
   grid                               grid
   ├─ block 0                         ├─ program 0
   │   thread 0 → 算 i=0              │   offs = [0,1,2,...,1023] ← 一个向量
   │   thread 1 → 算 i=1              │   x = tl.load(...)   ← 一次读一整块
   │   ...                            │   output = x + y      ← 整块相加
   │   thread 255 → 算 i=255          ├─ program 1
   ├─ block 1                         │   offs = [1024,...,2047]
   │   ...                            ├─ ...
   每根线程:算 1 个元素                每个 program:算 1024 个

   "1024 个元素怎么分给 128 根线程、
    要不要合成 float4 读" ← 你手写      ← 编译器替你分(num_warps,Day4 调)
```

w1d6 的 61 行六步骨架 → 本篇核心逻辑 10 行。指针那头没变,还是门牌号——`x_ptr + offs` 和 C 的 `ptr+i` 一样**按 dtype 步进**(float 就是 4 字节),不是按字节:

```
x_ptr = 张量首地址(门牌号,如 0x7f200000)
offs  = [0,      1,      2,     ..., 1023]        (+ pid*BLOCK)
addr  = [x_ptr,  x_ptr+4, x_ptr+8, ...]           ← float 步进 4 字节
mask  = [T, T, T, ..., T]   (N=4097 时最后一块:仅 offs=4096 为 T)
```

---

## 二、`vector_add.py` 逐行精读

```python
@triton.jit                                          # ≈ __global__:标记"GPU 函数"
def add_kernel(x_ptr, y_ptr, output_ptr,             # 门牌号:张量首地址(≈ float*)
               n_elements,                           # ≈ int n(运行期标量)
               BLOCK_SIZE: tl.constexpr):            # 编译期常量
    pid  = tl.program_id(axis=0)                     # ≈ blockIdx.x
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # ⭐ 全局索引的向量版
    mask = offs < n_elements                         # ≈ if(i<n) 的向量版

    x      = tl.load(x_ptr + offs, mask=mask)        # 批量读(mask=F 不读)
    y      = tl.load(y_ptr + offs, mask=mask)
    output = x + y                                   # NumPy 风格:整块相加
    tl.store(output_ptr + offs, output, mask=mask)   # 批量写(mask=F 不写)
```

### 变量字典

| 变量/表达式 | 是什么 | 值/含义(N=4097, BLOCK=1024 时) |
|---|---|---|
| `x_ptr` / `y_ptr` / `output_ptr` | 张量首地址(≈ CUDA 的 `float*`) | Triton 自动从 torch 张量取 `data_ptr()`,不用手管 |
| `n_elements` | 运行期标量(≈ `int n`) | 4097 |
| `BLOCK_SIZE` | **编译期**常量 constexpr | 1024;换值 = 重编一个新 kernel |
| `pid` | program 编号(≈ `blockIdx.x`) | 0 ~ 4(grid=5) |
| `offs` | 本 program 负责的**全局元素索引向量** | program 4 → [4096, 4097, ..., 5119] |
| `mask` | 布尔向量,越界保护 | program 4 → [T, F, F, ..., F](仅 4096 为真) |
| `x` / `y` / `output` | 一整块数据(≈ NumPy 数组) | 各 1024 元素(mask 外为未定义值) |
| `grid` | program 数(≈ block 数) | `triton.cdiv(4097, 1024)` = 5 |

### 计算具体是怎么发生的(以 N=4097 的最后一个元素为例)

```
pid=4:offs = 4×1024 + [0,1,...,1023] = [4096, 4097, ..., 5119]
mask  = offs < 4097          → [T, F, F, ..., F]   (只有 4096 真)
load: x = x_ptr+4096 读到(示意 0.312);x_ptr+4097..5119 被 mask 挡住不读
add:  output[0] = 0.312 + 0.708 = 1.020            (块内 1024 个位置照算,
                                                      但 1..1023 位是垃圾)
store: 只把 mask=T 的位置写回 → out[4096] = 1.020 ✅
```

GPU 侧 launcher 三行对应 Week1 的六步骨架:`torch.empty_like(x)` ≈ cudaMalloc(释放交给 GC);数据全程在 GPU 上造,**没有 H2D/D2H**;`add_kernel[grid](...)` ≈ `kernel<<<grid>>>(...)`。

**坑(本节)**:
- kernel 体内**不能用普通 `print`**(GPU 上没有 stdout)——要 `tl.device_print("pid", pid)`;
- 首次调用会卡零点几秒 = JIT 编译,不是死机;
- `mask` 与 `__syncthreads` 不同名也不同层:前者是数据越界保护,后者(隐式的)是块内同步——别混。

---

## 三、mask 深挖与三个法医案例 ⭐(本篇最值钱的部分)

### 案例A:非 2 幂 → 编译期拦截

BLOCK=1000 → `ValueError: arange's range must be a power of 2`,栈中 `ast_to_ttir`/`CompilationError` 证明发生在**JIT 编译阶段**。越早炸越好处理的典范。

### 案例B:把 `cdiv` 换成 `n // BLOCK`(Q2 实测)

grid = 4097//1024 = **4** → offs 最大只到 4095,**out[4096] 根本没人写**,留着 `empty_like` 的垃圾值。实测:`N=4097 ❌ FAIL max|err|=1.51`(垃圾值 vs 正确值的随机差)。

**比 crash 更危险的量化**:全向量只有 **1/4097** 个元素错。若用 Week2 的"随机抽 256 点"验证法,命中坏点概率 256/4097 ≈ 6% → **94% 概率漏检放行**。教训:"有元素根本没被算"这类 bug **只能全量查**;抽查是给"算对了但有舍入噪声"的场合用的。

### 案例C:删 mask(修罗场 B,案发现场复盘)

无 mask 时 N=4097、BLOCK=1024、grid=5,program 4 会:
```
读 offs 4096:合法(最后真元素)✅
读 offs 4097..5119:越界读邻居显存垃圾(1023 个)
写 offs 4097..5119:越界写进邻居显存 4 KB 💣
```
三种越界结局对号入座:

| 结局 | 触发条件 | 本次抽到的牌 |
|---|---|---|
| (a) 当场 illegal memory access | 越界踩到未映射页 | 否 |
| (b) 可见结果错 | 越界但写进了自己可见区 | 否(案例B才是) |
| (c) **静默 PASS + 暗中污染邻居** | 越界落在 torch caching allocator 的大块空隙里 | ✅ 最糟 |

**(c) 最危险**:穿透全部测试;被污染的邻居张量可能在后面某个不相干 kernel 里才爆,到时候不会怀疑到这个 kernel 头上。类比:写错门牌把快递塞进邻居信箱,邻居三天后才报警,警察查不到你。没踩到未映射页纯属 allocator 布局的运气。

### 破案:输出里 0.033 ms / 1 GB/s 是怎么来的

修罗场 B 的输出:`BLOCK=256: 0.033 ms → 1 GB/s;1024: 0.024→2;4096: 0.033→1;torch: 0.010→5`。证据链:

1. **torch 原生也快了 95 倍**(0.947→0.010 ms)。mask 只在 triton kernel 里,删它不可能影响 torch 的 kernel → **输入变小了**。
2. **反推打印的字节数**:带宽×时间 = 1 GB/s × 0.033 ms ≈ 33 KB,与 `3×4097×4 = 49,164 B` 精确吻合(49164/0.033ms = 1.49 GB/s,`:.0f` 打成 "1";其余两行同法全对上)。
3. **结论:做实验 B 时误删了 `N = 2**26` 那行**,bench 用了 `[1]` 循环残留的 N=4097。表头"[2] 2^26"是硬编码字符串——**标签会撒谎,数字不会**。

附带收获:0.010~0.033 ms ≈ **launch 开销的量级**(N=4097 时数据搬运只需 49164 B ÷ 936 GB/s ≈ 52 ns,测到的几十 µs 几乎全是发令枪钱)——这是本篇唯一一次"测到 launch 开销"的机会,记下数字。

**mask 的语义细节**:被 mask 挡住的 load 位置,读进来的是**未定义垃圾**(不是默认 0;想指定可加 `other=0.0`)。这里无所谓——store 用同一个 mask,垃圾写不出去。

> [!tip] 方法论沉淀
> 508=2×254(Week2 Day7)→ 610 TFLOPS>峰值 → 本篇 49164 B——同一块肌肉:**从算术指纹反推事实**。打印出来的每个数都是可验证的:带宽×时间=字节,字节÷4÷3=元素数,元素数对不上,就说明你跑的不是你以为的东西。

---

## 四、性能实测与算账(BLOCK 旋钮 + 带宽帽)

### 实测(2²⁶ 元素,10 次平均,warmup 5 次后)

| 配置 | ms | GB/s |
|---|---|---|
| BLOCK=256 | 0.951 | 847 |
| BLOCK=1024 | 0.949 | 849 |
| BLOCK=4096 | 0.952 | 846 |
| torch 原生 `a+b` | 0.947 | 850 |

### 指标怎么算(逐步)

```
搬运字节 = 3 × N × 4 = 3 × 2²⁶ × 4 = 805,306,368 B ≈ 0.805 GB
           └ 为什么 3:读 x + 读 y + 写 out
有效带宽 = 字节 ÷ 时间 = 0.805 GB ÷ 0.949 ms = 849 GB/s
理论地板 = 0.805 GB ÷ 936 GB/s = 0.860 ms      ← 时间下限,不是带宽
达峰率   = 849 ÷ 936 = 90.7%
距地板   = 0.949 ÷ 0.860 = 1.10×
```

### "优化"分析:本篇优化了什么?——什么都没优化,而这是重点

1. **优化了什么**:没有。本篇的目的是**建立基线 + 验证编译器开箱质量**:Triton 裸写 10 行就与 torch 原生(高度调优的 cuBLAS 级 elementwise)差 0.2%(噪声级),贴墙 91%。memory-bound kernel 的"及格即满分配"。
2. **BLOCK 扫描三档差 <0.4% = 死旋钮**。为什么:瓶颈在 DRAM 带宽,不在调度;编译器对三个配置生成的访存指令同样接近最优。**"调了没用的旋钮"和有用的旋钮一样是信息**——它告诉你瓶颈不在这。这与 Week2 Day3"只快 13%"是同一课:先判断瓶颈,再动手。
3. **剩余 9% 差距(瓶颈档案)**:launch 开销(实测 ~10-30 µs,案例 C 亲测)、DRAM 刷新/ECC 等硬件税、非理想访存边角。数量级:0.949−0.860 ≈ 0.089 ms,其中 launch 占比不小。
4. **下一级方向(本 kernel 不值得,但要知道)**:减少 launch 次数(cudaGraph / 批处理合并)、`float4` 向量化装载(编译器多半已做,可 dump PTX 验证)。判断"**不值得再优化**"也是优化梯子的第一课。

---

## 五、性能指标小抄(Week2 表续,新增 3 行)

| 指标 | 公式 | 本篇实例 |
|---|---|---|
| 搬运字节(elementwise) | (读数组数 + 写数组数) × N × sizeof | 3×2²⁶×4 ≈ 0.805 GB |
| 有效带宽 | 字节 ÷ 时间 | 849 GB/s |
| 理论时间地板 | 字节 ÷ 峰值带宽 | 0.805÷936 = 0.860 ms |
| **达峰率(带宽)** | 实测带宽 ÷ 936 GB/s | 90.7%(Week2 的"贴带宽帽"量化版) |
| **launch 开销** | 小数据 kernel 的实测耗时下限 | ~10-30 µs(案例 C 实测 0.010~0.033 ms) |
| **JIT 编译时间** | 每个 constexpr 组合首次调用 | ~10² ms 级,warmup 吸收;autotune 的成本来源 |

**RTX 3090(sm_86)**:82 SM · FP32 35.6 TFLOPS · 带宽 936 GB/s · 平衡点 ≈ 38 FLOP/B(承接 Week2 手动 Roofline;本 kernel AI = 1 FLOP ÷ 12 B ≈ 0.083,纯 memory-bound——每搬 12 字节才做 1 次加法,离平衡点差 450 倍,带宽帽毫无悬念先顶)。

---

## 六、坑合集(Day1)

1. `tl.arange` 的长度**必须是 2 的幂**,编译期拦截(案例 A)。
2. **mask 删不得**:三种结局里"静默 PASS + OOB 污染邻居"最糟,穿透全部测试(案例 C)。
3. **`cdiv` 不是装饰**:floor-div 丢尾巴 → 1/N 个元素没人写 → 全量 max|err| 才抓得住,抽查 94% 漏检(案例 B)。
4. **标签会撒谎,数字不会**:硬编码的打印字符串 vs 算术指纹(49164 B 破案)。
5. kernel 里没有 stdout:`print` 不可用,要 `tl.device_print`。
6. **constexpr 换值 = 重编译**:扫描脚本的每个 config 首次调用都贵——Day4 autotune 的成本模型在这。
7. masked load 的垃圾值语义:挡住 ≠ 读到 0,要 `other=0.0`;store 同 mask 才是安全闭环。
8. `torch.empty_like` **不清零**:垃圾值会被误当结果(Q2 的 max|err|=1.51 就是它)。

---

## 七、自测(闭卷过一遍)

1. N=4097、BLOCK=1024:grid=? program 4 的 mask 里几个 T?(答:5;1 个——仅 offs=4096)
2. BLOCK 扫描 256/1024/4096 差 <0.4%,说明什么?为什么?(答:memory-bound,BLOCK 是死旋钮,瓶颈在带宽不在调度)
3. 0.860 ms 这个地板怎么算的?(答:3×2²⁶×4 B ÷ 936 GB/s)
4. floor-div 版错在哪?若用"随机抽 256 点"验证,漏检概率?(答:out[4096] 无人写,留 empty_like 垃圾;1−256/4097 ≈ 94%)
5. mask=F 的位置,`tl.load` 读到什么?怎么改成读 0?(答:未定义垃圾;`other=0.0`)
6. 删 mask 的三种结局,各由什么决定?哪种最糟?(答:踩未映射页→crash;写进可见区→结果错;落 allocator 空隙→静默污染,最糟)
7. `tl.constexpr` 的语义?与 autotune 什么关系?(答:编译期常量,换值重编;autotune=自动扫 constexpr 组合)

---

## 八、与后续日子的连接

- **Day2 softmax**:`tl.max`/`tl.sum` 归约 = Week2 Day3 归约树的 Triton 转世(跨 warp 的桥编译器搭);数值稳定性实验(去 max 减法看 inf);
- **Day3 裸版 matmul**:BLOCK=16 起步与 Week2 tiled 同配置刻意可对照,先手算 AI 再实测;
- **Day4 autotune**:今天"BLOCK 是死旋钮"的结论升级成"为什么 autotune 对 memory-bound 无感、对 compute-bound 关键";num_warps/num_stages 两个真旋钮登场;
- **Day5 fp16 + 大分块**:对标 CUDA k5 的 8.57 TFLOPS;**Day6 Flash-Attention**:mask/块思维/归约三件套全用上。

> 下一篇:Day2 softmax(待续)
