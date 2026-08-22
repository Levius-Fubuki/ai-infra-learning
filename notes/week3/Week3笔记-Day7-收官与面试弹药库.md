---
title: Week 3 笔记(Day7):收官与面试弹药库
phase: 0
week: 3
period: 2026-08-22
tags:
  - ai-infra
  - 笔记
  - triton
  - 面试
created: 2026-08-22
---

# Week 3 笔记(Day7):收官与面试弹药库

> [!info] 笔记体例
> 本篇是 Week3 收官篇,也是**面试速查弹药库**:成绩单 / 自测批改 / 面试 Q&A 骨架 / LC 批改 / 对照表全表 / 验证门战记 / own it 全记录。
> 关联 [[Week3笔记-Day1-Triton入门与向量加法]] · [[Week3笔记-Day4-autotune与分块参数]] · [[Week3笔记-Day5-fp16与tensor-core]] · [[Week3笔记-Day6-Flash-Attention]] · [[阶段0-执行计划]]。

> [!success] Week 3 成绩单 🏆(2026-08-18 ~ 08-22,RTX 3090)
> - naive **0.51** → tiled 2.88 → k4 3.97 → autotune fp32 **19.91** → **fp16 72.04**(naive 的 141×)
> - fp16 72.04 = fp16 tensor 峰 142 的 51%,**超 cuBLAS fp16(67.17)7%**,比特级对齐
> - 一天写完简化版 Flash-Attention,比朴素实现快 **3.1×**(N=8192)
> - 贯穿全程:**先预测后实测 / 验证门必过 / 单变量优化 / 法医式 debug**——方法论比数字更值钱

---

## 一、闭卷自测批改(8 题)

| 题 | 判定 | 批改要点 |
|---|---|---|
| 1 `if(i<n)` → Triton | ✓ | `mask = offs<n`,`tl.load(ptr+offs, mask=mask)` |
| 2 `__shared__`+装载+同步 → ? | 半对 | 是 **`tl.load`**(编译器自动做 shared 装载+同步);"没有"只对了一半 |
| 3 全局索引 → ? | 半对 | 完整是 `offs = pid*BLOCK + tl.arange(0,BLOCK)`,别只写 `pid` |
| 4 归约树 → ? | ✓ | `tl.max/tl.sum(x, axis=0)` |
| 5 tl.dot 吸收了什么 | 半对(一处错) | **acc 累加被吸收**(`tl.dot(a,b,acc)` 乘加一体);没吸收的是分块参数 + 精度开关 |
| 6 autotune 参数 | ✓ | BLOCK=块大小、num_warps=每 program warp 数、num_stages=流水深度;shared 用量 = num_stages×(BM×BK+BK×BN)×dtype,BLOCK_K 也参与 |
| 7 online softmax 省什么 | 半对 | 核心是 S/P 的 **N² 中间不落 HBM**(4N² 流量→寄存器/shared);滚动量 m/l/O + α |
| 8 ns 死活的机制 | ✓ | **计算/搬运比**为主(0.52 vs 2.10 MFLOP/32KB)+ **带宽是否已饱和**(Day4 15.35≈15T 贴帽 vs Day5 55.6<60T) |

**结论**:4 扎实 / 4 半对。框架立住了,精度欠一档——面试的"半对"就是"好像懂",逐条补严。

---

## 二、面试弹药库 Q&A

### Q1:会写 CUDA,为什么还要 Triton?(必考,4 层骨架)

```
① 定位:CUDA 是原理,Triton 是生产力。手写 CUDA 学内存层次/分块/寄存器/Roofline;
   Triton 做同一件事,但编译器管 shared 分配、同步、调度。

② 数据为证(自己的 3090 实测——面试最有力的是"我亲手跑的"):
   手写 CUDA 三天 → 8.57 TFLOPS
   Triton 一天   → fp32 19.91(2.3×)、fp16 72.04(超 cuBLAS fp16 的 67,7%)
   还写了 Flash-Attention,3.1× 于朴素版。

③ 为什么快:调度自动化(排 warp/插同步/向量化);autotune 自动扫分块
   (手写时 BK=8 的 bank conflict 是后查的病根,编译器直接绕过);生态
   (torch.compile/vLLM 底层就是 Triton)。

④ 边界:跨 block 通信、warp 黑魔法、cudaGraph 仍是 CUDA。
   行业标配 = "CUDA 懂原理 + Triton 出活"。
```

### Q2:Flash-Attention 为什么快?

```
❌ "复杂度降了"——错,FLOP 本就 O(N²d),复杂度不变。
✓ 去掉 S/P 的 4N² 中间流量(N=8192 = 512MiB),中间不落 HBM;
  瓶颈从 memory-bound 翻到 compute-bound(我的 3090:朴素 471GB/s 贴带宽,
  Triton 48 TFLOPS 计算主导,实测 3.1×)。
一句话:没降复杂度,降了常数、换了哪顶帽先顶。
```

### Q3(附):FP16 为什么快?为什么不是 4×?

```
fp16 同时抬两帽:算力帽 142T(4×fp32)+ AI≈B/2 → 带宽帽也翻倍;
但实测 51% 峰值是因为带宽帽先顶(best config AI=64 → cap 60T);
acc 必须 fp32,否则误差随 K 线性涨(Day5)。
```

### Q4(附):验证门/法医方法论(讲给面试官听,极加分)

```
新 kernel 必过验证门再跑分;先怀疑验收标准再怀疑 kernel(own it 五连);
"标签会撒谎,数字不会"——打印带宽反推字节数破案(49164B);
对照实现也有隐蔽默认项(cuBLAS fp16 默认低精度累加,Day5 §5.4)。
```

---

## 三、LC 批改(两题一真一假)

### 27 移除元素:主体对,边界 bug

```
❌ if (n<=1) return 1;   // n==0 应返 0;n==1 且首元素==val 应返 0
✓ 正确版无需分支:
  int slow=0;
  for(int fast=0; fast<nums.size(); ++fast)
      if (nums[fast]!=val) nums[slow++]=nums[fast];
  return slow;
```

### 88 合并有序数组:侥幸对——逻辑错的三个证据

```
❌ while(nums1[slow]<=nums2[i] && nums1[slow]!=0) slow++;
① nums1 有效区可含真 0(如 {0,2,4})→ !=0 误跳;
② nums2 小值要插前(nums1={2,3,4,0,0,0},nums2={1,5,6})→ slow 定位错,输出乱;
③ 测试用例恰好 nums2 升序嵌后面 + 有效区无 0 → 蒙对。

✓ 正确解 = 逆向双指针:
  int i=m-1, j=n-1, k=m+n-1;
  while(j>=0){ if(i>=0 && nums1[i]>nums2[j]) nums1[k--]=nums1[i--];
               else                          nums1[k--]=nums2[j--]; }

为什么逆向:nums1 尾部 m..m+n-1 本来就是空的,从最大的开始从后往前填,
永远不会覆盖还没比较的 nums1[i]——正向插入要解决"挤掉后面的",逆向免费免疫。
```

---

## 四、CUDA ↔ Triton 对照表全表(D1-D6 累积)

| 你写过的 CUDA(Week1/2) | Triton 写法 | 一句话 |
|---|---|---|
| `__global__ void k(...)` | `@triton.jit` | GPU 函数;首次调用才 JIT 编译 |
| `blockIdx.x*blockDim.x+threadIdx.x` | `pid*BLOCK + tl.arange(0,BLOCK)` | 标量索引 → 向量索引 |
| `if (i < n)` | `mask = offs < n` 塞 load/store | 越界保护向量版 |
| `c[i]=a[i]+b[i]` | `output = x + y` | 块算术(NumPy 同款) |
| `k<<<grid, block>>>(...)` | `kernel[grid](...)` | launch 语法糖 |
| `(N+t-1)/t` 向上取整 | `triton.cdiv(n, B)` | 一模一样 |
| `__shared__`+装载循环+`__syncthreads` | **`tl.load`**(隐式) | 编译器自动装载+同步 |
| 寄存器分块 TM/TN + MMA | `tl.dot(a, b, acc)` | 乘加一体,tensor core 自动 |
| 归约树(`__shfl`+shared 桥) | `tl.max/tl.sum(x, axis=0)` | 树+屏障+跨 warp 桥全隐式 |
| 二维寻址 `A[row*N+col]` | `ptr + row*stride + cols` | 行主序同款,stride 支持非连续 |
| 2D grid 拆 pid | `pid%num_pid_m / pid//num_pid_m` + group ordering | 1D 压平再拆,L2 排序 |
| cudaMalloc/Memcpy/Free ×N | `torch.empty_like` + 全程 GPU + GC | 六步骨架 → 三行 |
| 手挑 BM/BN/BK/TM/TN | BLOCK_M/N/K(仍是你) | **没被吸收** = 你和编译器的分工边界 |
| 手工扫描参数(k9) | `@triton.autotune` | 自动扫 BLOCK/num_warps/num_stages |
| fp32 dot 精度控制 | `allow_tf32`(默认 True→TF32) | 隐蔽默认项,验证门会抓 |
| nvcc → ./a.out | `python3 xxx.py` | 开发循环分钟级 → 秒级 |

---

## 五、验证门五连击 + own it 五连(法医战记)

**验证门立功记录**(每次都是"结果不对"被门拦下/定位):

| # | 天 | 案例 | 性质 |
|---|---|---|---|
| 1 | D1 | floor-div 丢尾,1/4097 元素没算 | 抽查 256 点 94% 漏检 |
| 2 | D1 | 删 mask,OOB 静默污染邻居 | 最糟的越界结局(c) |
| 3 | D2 | 幽灵哨兵 other=0.0 偷 max+分母 | 行和 1.01e-43 报警 |
| 4 | D3 | `tl.dot` 默认 TF32,精度被偷 | 跑得快但错(1e-1 误差) |
| 5 | D5 | cuBLAS fp16 输出舍入=参考噪声 | 比特级对齐才是终极证明 |

**own it(公开认错)五连**——法医精神第一个适用对象是自己:

```
① D2 naive 流量账 448 MiB → 512 MiB
② D1 AI 2 → 0.083(口径统一)
③ D5 fp16 验证门 atol=1e-2 太紧(该分"实现正确性/精度预算"两档)
④ D5 关 flag 预测翻车——真凶是参考输出舍入(64×2⁻¹¹≈2.7e-2)
⑤ D5/D6 Q5 波数伪代理——"2.1/8.3 波"是 48 warp 假想上限编的,
   实际两配置驻留都是 24 warp/SM;ns 死活 = 计算/搬运比 × 带宽余量
```

---

## 六、指标小抄(Week3 最终版)

| 指标 | 公式 | 实例 |
|---|---|---|
| 带宽帽 | 936 GB/s × AI | AI=64 → 60T(fp16) |
| 算力帽 | fp32 35.6 / TF32 71 / fp16 tensor 142 | fp16 实测 72=51% |
| AI(tiled) | ≈ B/4(fp32)或 B/2(fp16) | B=128 → 32/64 |
| 朴素 attention 中间流量 | 4N² × dtype | 512MiB @8192 fp16 |
| flash 有效 DRAM | Q/K/V 首读(K/V 2MB 进 L2) | ~4µs,非瓶颈 |
| 加速比(受控) | naive/flash | 3.1× |
| fp16 相对误差 | 2⁻¹¹×√K 抵消 | ~3e-4 |
| 波数(修正用法) | 总 warp ÷ 实际驻留(扣 shared/寄存器) | 非 48 假想 |

---

## 七、坑合集(Week3 总)

1. **标签会撒谎,数字不会**(49164B 破案)。
2. **请求的 config ≠ 实际跑的 config**(shared 超限静默降级,soft wall 三连击)。
3. **mask 删不得 + K 轴也要 mask**;OOB 错误不可复现(3e-2↔3e+2)。
4. **归约 kernel 的哨兵要双中性**(-inf;other=0.0 = 幽灵)。
5. **验收标准必须匹配精度预算**;实现正确性 vs 精度预算分两档测。
6. **隐蔽默认项**:tl.dot 的 TF32、cuBLAS 的 fp16 低精度累加。
7. **旋钮死活 = 计算/搬运比 × 带宽余量**,不是占用率/波数。
8. **Flash 不降复杂度**,只换常数和哪顶帽先顶。
9. **参考实现的数字要审查**(sdpa 9.3× 有水分,受控 3.1×)。
10. **head_dim 必须 2 的幂**否则编译失败;真代码 16 块循环处理任意维度。

---

## 八、与 Week4 的连接

- **W4 vLLM 实战**:本地部署 7B + 压测吞吐/延迟。PagedAttention = "flash-attention 的 cache 版"——今天的分块+在线思想直接平移;
- **面试能用上的本周资产**:72 TFLOPS、超 cuBLAS 7%、flash 3.1×、五个验证门案例、四个自己的数字——**每一个都亲手跑过**;
- 保持手感:LC C++ 每周 2 道。

---

## 九、高频面试题库(扩充,详细答案)⭐

> 编号续 §二 的 Q1-Q4。每题答案都以**你自己亲手跑过的数据**为骨架——背数字,不背话术。

### Q5:手写一个 CUDA 数组求和(reduction),为什么比 atomicAdd 快?

```
atomicAdd 版:100 万线程全抢同一个 sum → 100 万次跨 SM 排队 → 实测 18ms
shared 版:块内树状归约(256 人两两配对,log₂256=8 轮,全在片上零原子)
          + 块间只留 4096 次 atomicAdd(每 block 交一个部分和)→ 15.75ms
答案骨架:①__shared__ 片上内存 ②每层 __syncthreads 屏障 ③块间合并降到 4096 次
加分句:但这个 kernel 其实只快 13%——因为它是 memory-bound,主成本是读 4MiB 输入,
         原子串行化不是瓶颈。加这句 = 暴露你懂"先判断瓶颈再优化"。
```

### Q6:什么是 bank conflict?怎么发生的?怎么解决?

```
shared memory 分 32 个 bank,每个 bank 一个周期只能服务一个地址
同一 warp 的 32 个线程同时访问 shared:若地址撞进同一个 bank → 硬件串行化 = bank conflict
我的 Week2 k5 就有 Bs 的 4-way bank conflict:连续线程访问的地址恰好错开 4 个 bank,
   每个地址被拆成 4 次串行 → 4 倍延迟
解法:给数组 padding,如 [8][17] 而不是 [8][16],把同一行的列地址错开一个 bank
```

### Q7:什么是 warp divergence?后果?

```
warp = 32 线程"连体婴"(SIMT):永远执行同一条指令,只是操作不同数据
若 32 线程走了不同 if 分支(如 if(tid%2)):硬件先执行一支、再执行另一支,两段都跑
   → 一半线程白等 = 浪费一半吞吐
实例:归约里 if(tid<stride) 那些轮,后面线程干等;分支越散,浪费越多
Triton 里编译器帮你消化大部分(向量化/tl.where),但概念必须能讲
```

### Q8:怎么判断一个 kernel 是 memory-bound 还是 compute-bound?

```
算算术强度 AI = FLOP ÷ 搬移字节,和 GPU 平衡点比(3090 ≈ 38 FLOP/B):
  AI < 38 → memory-bound(带宽帽 936GB/s 先顶)
  AI > 38 → compute-bound(算力帽先顶)
我的实测当样本:
  vector_add AI=0.083、reduction 0.25、naive matmul 0.25 → 全 memory-bound
  tiled matmul AI=4(贴带宽帽,BLOCK=16 时 TF32/IEEE 一样快 = 铁证)
  fp16 大分块 AI=64 → 逼近平衡点,开始 compute-bound
口诀:搬得多算得少 → memory-bound;先算 AI 再谈优化。
```

### Q9:为什么 matmul 适合 GPU?为什么 fp16 + tensor core 更快?

```
matmul 每个输出元素被复用 K 次 → 计算密度高 → 天然契合 SIMT 大并行 + shared 复用
tensor core = 专用矩阵乘硬件,一次算一个小矩阵乘加,吞吐是 CUDA core 的 4×
fp16 更快的账(我实测 4.03 → 72.04):
  算力帽 35.6 → 142;且 AI≈B/2(输入 2 字节)让带宽帽也翻倍
  注意前提:AI 足够高时 tensor core 才有活干——BLOCK=16 时它白上班(TF32=IEEE)
```

### Q10:为什么 softmax 要减 max?为什么 attention 除以 √d?

```
减 max:exp 溢出——fp32 里 x>88.7 就 inf,inf/inf=nan 整行全灭
   softmax(x−c)≡softmax(x)(约掉公因子),取 c=max 则所有指数≤1,永不溢出
除 √d:q·k 是 d 个独立元素内积,方差随 d 线性涨(≈d)
   不缩放 → 打分进入 softmax 饱和区(exp 梯度趋 0)→ 训练退化
   /√d 让打分方差回到 O(1);这是为什么 softmax 前要乘 1/√64 = 0.125
```

### Q11:Triton 写 matmul,分块思路?autotune 扫哪几个参数?

```
思路:每个 program 算一个 BM×BN 的输出 tile,K 维循环;
   每圈 load A 的 (BM,BK)、B 的 (BK,BN),tl.dot(a,b,acc) 累加
   BLOCK 越大 → AI 越高 → 带宽帽越高;但 shared 99KB 限制 num_stages
autotune 扫:BLOCK_M/N/K(块大小)、num_warps(warp 数)、num_stages(流水深度)
我的实测:最优 128×64×32 ns3 nw4 → fp32 19.91 TFLOPS
   平顶平台:前 7 个 config 都在 18.6~19.9——面试讲"找到平台即可,别纠结次优"
```

### Q12:手写 LRU Cache(C++),讲思路

```
思路:hashmap<key, 链表节点*> + 双向链表,get/set 都 O(1)
get:命中 → 移到头部,返回;未命中 → -1
set:已存在 → 更新值 + 移头;不存在 → 插头部;超容量 → 删尾部
为什么双向链表:删尾/移动节点要 O(1) 找到前驱,单链表做不到
模板(面试手写用):
  struct Node{int key,val; Node*prev,*next;};
  unordered_map<int,Node*> mp; Node *head,*tail; int cap;
```

### Q13:你怎么验证你写的 kernel 是对的?(方法论题,极加分)

```
三层:①实现正确性——对独立实现/参考库,同精度预算
        fp32 对 cuBLAS(max|err|~1e-4);fp16 对 fp16 参考(~3e-4,比特级对齐可作终极证明)
     ②精度预算——对 fp64 真值报固有误差(不是 bug,是 dtype 的代价)
     ③先对再跑分,全量 max|err| 而非只抽查(抽查 94% 漏检的教训)
法医三原则:先怀疑验收标准再怀疑 kernel(own it 五连);
           标签会撒谎,数字不会(49164B 破案);
           参考实现也有隐蔽默认项(cuBLAS fp16 默认低精度累加)。
```

> 上一篇:[[Week3笔记-Day6-Flash-Attention]] · 下一篇:W4 vLLM(待展开)
