---
title: Week 3 笔记(Day3):Triton matmul 与 TF32 谜案
phase: 0
week: 3
period: 2026-08-19
tags:
  - ai-infra
  - 笔记
  - triton
created: 2026-08-19
---

# Week 3 笔记(Day3):Triton matmul 与 TF32 谜案

> [!info] 笔记体例
> 概念块沿用四段式 **是什么 / 有什么用 / 怎么用 / 坑**;代码块采用 **逐行讲解 + 具体数字举例**。当日没答上/答错的问题(Q1 AI 完整账、Q2 naive 复用、Q4 落点、Q5 波数)在正文里全量展开。
> 关联 [[Week3笔记-Day2-softmax归约与融合]] · [[Week2笔记-Day4-7-matmul优化梯子]] · [[阶段0-执行计划]]。

> [!success] Day3 成绩
> - `matmul_basic.py` 跑通:**Triton BLOCK=16 → 3.95 TFLOPS(4096³)**——与 Week2 k4(3.97)几乎重合,把 tiled(2.88)甩开 1.36×。
> - **最大发现:验证门 FAIL 揪出 `tl.dot` 默认用 TF32**——fp32 输入走 tensor core 时精度被降到 10 位尾数,误差 1e-1 级;切 `allow_tf32=False` 后 PASS(1e-5)而**跑分不变**(3.95 vs 4.03)。结论:**BLOCK=16 时 tensor core 无关紧要,带宽帽主导**。
> - 修罗场 K=100:同一份代码两跑误差 **3.13e-2 → 3.09e+2** = OOB 垃圾不可复现;K-mask 修复版 PASS 3.43e-5。

> [!warning] 本篇的"性能数字"怎么读
> 3.95 / 4.03 是 **TF32 与 IEEE 两种精度**下各自的跑分。这篇的主角不是"更快",而是两件事:**验证门为什么能拦住一个跑得挺快但精度被偷的基线**,以及 **tensor core 为什么在低 AI 时白搭**。把 4→24 TFLOPS 的差距留给 Day4/5(分块质量、L2、大 BLOCK)。

---

## 〇、`tl.dot` = 你 Week2 的半本笔记本

今天要写的"算法"核心就一个 K 循环 + 一次 dot:

```python
for k in range(0, tl.cdiv(K, BLOCK_K)):
    a = tl.load(a_ptrs)             # A 的 (BM, BK) 块
    b = tl.load(b_ptrs)             # B 的 (BK, BN) 块
    acc = tl.dot(a, b, acc)         # ⭐ TM×TN 寄存器分块 + tensor core,全自动
    a_ptrs += BLOCK_K * stride_ak   # 沿 K 方向推进
    b_ptrs += BLOCK_K * stride_bk
```

对照你 Week2 k5(8.57 TFLOPS 那版)逐项吸收:

| 你 k5 手写的 | Triton 对应 | 状态 |
|---|---|---|
| `__shared__ As[128][8]` + 装载循环 | `a = tl.load(a_ptrs)` | 1 行吸收 |
| Bs 装载循环 + 你的越界 bug(i/BK 套到 Bs) | `b = tl.load(b_ptrs)` | 1 行吸收 |
| TM=TN=8 寄存器分块、threadRow×TM | `tl.dot(a, b, acc)` | 1 行吸收 |
| 两处 `__syncthreads()` 手插 | (隐式) | 不写 |
| 256 线程怎么分工 128×8 装载 | 编译器排 | 不写 |
| 向上取整、边界 if | mask 参数 | 不写 |
| 手挑 BM/BN/BK/TM/TN | **BLOCK_M/N/K(仍是你的旋钮)** | 还是你调 |

注意最后一行:**分块参数没被吸收**——它就是你与编译器之间的"分工边界"。Day4 autotune 扫的就是它们。

**`tl.dot` 的三条硬件规矩**:
1. 三个维度都必须 **≥16**(MMA 指令最小形状);BLOCK=16 压线,是今天刻意选它对照 Week2 的原因;
2. dtype 要配对:fp32 输入默认走 **TF32**(§六主菜);fp16 输入 × fp32 累加是黄金组合(Day5);
3. `tl.dot(a, b, acc)` 三参数是乘加一体;acc 用 fp32 保精度。

---

## 一、2D 网格压平成 1D:pid_m / pid_n

```
grid = 256 × 256 = 65536 个 program(4096/16 的平方)
pid → 拆回 (行块, 列块):
    pid_m = pid %  num_pid_m        # 先定行块
    pid_n = pid // num_pid_m        # 再定列块
(注意顺序:pid_m 取余、pid_n 取整;反过来就是列优先,错的)
```

官方教程还有更讲究的 **group ordering**(相邻 pid 读相邻 A/B 块 → 提升 L2 命中),今天不用,Day4 autotune 版会看到它登场。

**管辖范围**(Day1 乘法链自检):65536 program × 每 program 一个 16×16 C tile = 4096×4096 输出 ✔;总 warp = 65536×4 = **262144**,波数 = 262144÷(82×48) ≈ **66 波**——并发绰绰有余,再次证明瓶颈不在调度(≈ 当日 Q5)。

---

## 二、广播寻址:一个表达式生成整块地址 ⭐

```python
a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
#             └ (16,1) 列向量 ×行步长  └ (1,16) 行向量 ×列步长
#             → 广播成 (16,16) 的地址矩阵
```

- `None` = `np.newaxis`,插一个维度:列向量 × 行向量 → 16×16 矩阵;
- 你 Week2 用双重循环填 As 的地方,现在一个表达式搞定;
- **stride 语义**(当日 Q3,你对):`BLOCK_K × stride_ak = 16 × 1 = 16 元素 = 64 字节`;A 行主序下 K 方向正是列方向 → 每圈沿 flat 偏移 +16,共 256 圈 = 4096 = 整条 K 维。

---

## 三、代码骨架与变量字典(4096³, BLOCK=16)

| 变量/表达式 | 是什么 | 值/含义 |
|---|---|---|
| `pid` | program 编号 | 0 ~ 65535 |
| `pid_m` / `pid_n` | 行块 / 列块编号 | pid%256 / pid//256 |
| `offs_m` / `offs_n` | C tile 的行/列坐标(各 16 元素) | 例如 pid_m=32 → 512..527 |
| `offs_k` | K 方向游标(16 元素) | 0..15,随循环 +=16 |
| `a_ptrs`/`b_ptrs` | A/B 分块地址矩阵(16×16) | 广播寻址生成 |
| `acc` | 累加器 (16,16) fp32 | `tl.zeros`,K 循环里 `tl.dot` 累计 |
| `ALLOW_TF32` | tl.dot 精度开关 | True→TF32 / False→IEEE(§六) |

### 一个具体的输出怎么算出来(C[512, 1248] 为例)

```
假设 pid=20000 → pid_m = 20000%256 = 32,pid_n = 20000//256 = 78
C tile = 行 512..527 × 列 1248..1263

K 循环 256 圈,第 3 圈(k 负责 48..63):
  a = A[512..527, 48..63]  (16×16)
  b = B[48..63, 1248..1263] (16×16)
  acc += a·b               (乘加一体)

最终 C[512,1248] = Σ_{k=0}^{4095} A[512,k]·B[k,1248]
     = 256 圈的 16×16 局部和拼接,无越界(4096 整除 16)
```

---

## 四、AI 账:BLOCK=16 为什么注定 ~4 TFLOPS(当日 Q1 完整推导)

每个 program(算一个 BM×BN tile):

```
FLOP = 2 × BM × BN × K = 2×16×16×K = 512K
字节 = 每圈读 (BM×BK + BK×BN) 元素 × K/BK 圈 + 写 BM×BN
     = (256+256)×K/16 + 256 = 32K + 256 (个元素)
AI   = 512K ÷ [4×(32K+256)] = 512K/(128K+1024)  →K→∞→ 512/128 = 4.0
```

带宽帽 = 936 GB/s × 4 FLOP/B = **3.74 TFLOPS**。实测 3.95(略超 = L2 复用偷了点 DRAM 流量——Day2 说过 L2 能救"小热数据",BK×BK 分块正是)。

**k4 之谜解开**(当日 Q4):Week2 k4 = 3.97、今天 Triton = 3.95,两个**同 BLOCK=16 同 AI=4** 的 kernel 撞同一顶帽。而今天 TF32 与 IEEE 跑分也几乎一样(§六)——三组数据互相咬合成一条证据链:**AI=4 时先撞带宽墙,算力富余到用不用 tensor core 都一样**。

---

## 五、naive 的复用账:0.25 vs 理想 170 = 680 倍金矿(当日 Q2 展开)

```
朴素 naive:每根线程算一个 c[i][j],自己从显存读 A 行 i + B 列 j
→ 同一个 A[i][k] 被 N 根线程各读一次 = 从 DRAM 读了 N 次
→ 实际流量 = 2MNK 个元素读 + MN 个写
  AI = 2N³ ÷ (8N³ + 4N²) ≈ 0.25                    ← Week2 的数
```

"被忽略的复用" = **A 的每个元素本该只读一次**。若按"每元素只读一次"的理想流量 (MK+NK)×4 + MN×4 = 12N² 字节:

```
AI理想 = 2N³ ÷ 12N² = N/6 ≈ 170(4096)
```

**0.25 vs 170 = 680 倍 = 复用的金矿**。tiled/shared 去兑现它(每块进 shared 读一遍供 N/BN 个 program 复用);naive 一根线程一次读,把金矿全扔了。这也是 naive 只有 0.51 TFLOPS 的原因——连带宽都没用满,纯在重复搬。

---

## 六、TF32 谜案 ⭐(本周最重要的一课)

### 6.1 案发:验证门 FAIL

```
[1] 验证门 TF32 vs IEEE:
  TF32(默认)   256³: FAIL  max|err|=5.37e-02
  TF32(默认)   1024³: FAIL  max|err|=1.39e-01
  IEEE(full) 256³: PASS  max|err|=4.96e-05
  IEEE(full) 1024³: PASS  max|err|=2.37e-04
```

### 6.2 凶手:TF32

```
TF32 = fp32 的"半精度版": 1 符号 + 8 指数 + 10 位尾数(fp32 是 23 位)
      ↑ tensor core 的 fp32 入场券 —— Week2 Day6 "tensor core 抬帽"的落地
      ↑ 但 10 位尾数 → 每个操作 ~1e-3 相对误差

`tl.dot` 对 fp32 输入默认 allow_tf32=True → 悄悄降精度
K=256 累积 → 5e-2;K=1024 累积更多 → 1.4e-1    ← 教科书级 TF32 指纹
```

**关键判断:这不是索引 bug**——索引错了会出 O(10) 的垃圾或 NaN,这里是"数量级正确、末位差 1e-1",是精度型误差。Q3 的寻址推演也独立证实了地址正确。

### 6.3 修复 + 复测(预测全中)

```python
acc = tl.dot(a, b, acc, allow_tf32=ALLOW_TF32)   # False = 全精度 IEEE
```

| 项 | 预测 | 实测 |
|---|---|---|
| IEEE 验证 | PASS ~1e-5 | PASS 4.96e-5 / 2.37e-4 ✔ |
| IEEE 4096³ 跑分 | 仍 ~3.7-3.9 | **4.03 TFLOPS**(vs TF32 3.95)✔ |

**跑分不变** → tensor core 在 BLOCK=16 时**白上班**:AI=4 带宽先顶住,算力富余。TF32 只是白白丢了精度,没换来任何速度。

### 6.4 三连教训

1. **验证门第四次立功**(D1 floor-div → D2 幽灵哨兵 → D3 TF32):一个跑得挺快、量级正确的 kernel 也可能被精度坑;`allow_tf32` 这类**默认项**是沉默的坑。
2. **tensor core 首现用户数据**:Triton 一行 `tl.dot` 就用了它(比手写 tiled 快 1.36×),但它的价值要等高 AI 才兑现(Day5)。
3. **torch 的对照组**:`torch.backends.cuda.matmul.allow_tf32` 默认 **False** → cuBLAS `a@b` 是真 fp32(所以它 24.08 TFLOPS 是实打实的 fp32 手调天花板,也是 Day4/5 追赶目标)。两边默认值不同,读别人 benchmark 前先确认这个开关。

---

## 七、K-mask 修罗场:OOB 垃圾不可复现 ⭐(当日的意外收获)

```
[3] K=100 非整除:cdiv(100,16)=7 圈;第 7 圈读 k=96..111
    B 只有 100 行 → 行 100..111 越界读 → 垃圾进累加器
    两次运行:max|err| = 3.13e-2 → 3.09e+2(!!)
[4] K-mask 修复版:PASS 3.43e-5
```

**3e-2 → 3e+2 漂了 4 个数量级**:同一份代码,两次运行读到**不同的"垃圾"**(取决于当时分配器在越界地址放了什么)。这条比"会错"更可怕——**OOB 错误不可复现,可能某次运行恰好 PASS,放你进生产**。修正:

```python
k_mask = offs_k < (K - k * BLOCK_K)                 # ⭐ K 轴 mask
a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)  # 越界补 0
b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)  # 补 0 → dot 只累加有效项
```

这是 Day1 floor-div 教训的 **K 轴版本**:M/N 整除就放松警惕,K 不整除照样静默错。4096³ 整除所以今天"免"了 mask——但"免"不等于"没有",你要知道它在哪。真正的库代码(教程 full 版)M/N/K 三轴都带 mask。

---

## 八、指标小抄(Day2 表续,新增 4 行)

| 指标 | 公式 | 本篇实例 |
|---|---|---|
| 带宽帽(matmul) | 峰值带宽 × AI | 936 × 4.0 = 3.74 TFLOPS |
| 每 program FLOP | 2×BM×BN×K | 2×16×16×4096 = 2.1 MFLOP |
| 每 program 字节 | ((BM×BK+BK×BN)×K/BK + BM×BN)×4 | (32K+256)×4 ≈ 512 KiB |
| 复用因子(naive→理想) | N/6 ÷ 0.25 = 2N/3 | ≈ 680×(4096) |
| TF32 精度代价 | 尾数 10 vs 23 位 | max|err| 1e-2~1e-1(K 越大越大) |
| cuBLAS 天花板 | 手调 fp32 极限 | 24.08 TFLOPS(约 35.6 峰的 68%) |

---

## 九、坑合集(Day3)

1. **`tl.dot` fp32 默认 TF32**:验证门 FAIL 的元凶;需要全精度用 `allow_tf32=False`(变慢只有高 AI 时才明显)。
2. **BLOCK 必须 ≥16**(MMA 最小形状);BLOCK=16 压线 = AI=4 带宽帽,调大才抬 AI。
3. **`pid_m`/`pid_n` 拆法有顺序**:`%` 取行块、`//` 取列块,反了全错。
4. **K 轴也要 mask**:非整除 K 静默错,且 OOB 垃圾不可复现(3e-2↔3e+2)。
5. **广播寻址的维度**:`offs_m[:, None]` 与 `offs_k[None, :]` 缺一个 None 就广播成错的形状。
6. **与别人对跑分前先对 `allow_tf32`**:torch 默认 False、Triton 默认 True,默认值不同是"假对比"的温床。
7. **验证门必须比 atol/rtol**:TF32 过不了 `atol=1e-4`,但如果给 `atol=1e-1` 就放过了——验收标准要与精度预算匹配。

---

## 十、自测(闭卷过一遍)

1. `tl.dot` 吸收了 Week2 哪些手工艺?哪些没吸收?(答:shared 装载/寄存器分块/同步/调度;没吸收的是 BLOCK_M/N/K 分块选择)
2. 为什么 BLOCK=16 注定 ~4 TFLOPS,且 TF32/IEEE 跑分几乎一样?(答:AI=4,带宽帽 3.74T 先顶;算力富余,tensor core 无关紧要)
3. TF32 的尾数几位?fp32 呢?误差量级?(答:10 vs 23;单 op ~1e-3,K 累积到 1e-1)
4. naive 的"复用"在哪被忽略了?理想 AI 多少?(答:A 每元素被 N 线程各读 N 次,没缓存复用;N/6≈170,naive 0.25)
5. K=100 的越界在哪圈?为什么两次跑误差不同?(答:第 7 圈读 B 行 96..111;OOB 读到的是不可复现的垃圾)
6. `pid_m = pid % num_pid_m` 与 `pid // num_pid_m` 各算什么?(答:行块/列块)
7. cuBLAS 24.08 是不是 TF32?(答:不是;torch 的 matmul.allow_tf32 默认 False,是真 fp32)

---

## 十一、与后续日子的连接

- **Day4 autotune**:今天"BLOCK=16 撞带宽帽"的结论直接推导出优化方向——**把 BLOCK 调大抬 AI**,让 tensor core 有活干;autotune 自动扫 BLOCK/num_warps/num_stages,group ordering + L2 复用正式登场;
- **Day5 fp16**:`tl.dot` 的黄金组合(fp16 输入×fp32 累加),AI 抬过平衡点后 tensor core 开始兑现,目标 ≥ k5 的 8.57;
- **Day6 Flash-Attention**:matmul + softmax 合体,今天的两顶帽(K 轴 mask、广播寻址)全用上。

> 上一篇:[[Week3笔记-Day2-softmax归约与融合]] · 下一篇:Day4 autotune(待续)
