---
title: Week 1 笔记:C++ 基石 + 点亮 GPU
phase: 0
week: 1
period: 2026-08-06 ~ 2026-08-12
tags:
  - ai-infra
  - 笔记
  - cpp
  - cuda
created: 2026-08-12
---

# Week 1 笔记:C++ 基石 + 点亮 GPU

> [!info] 笔记体例
> 每个知识点按 **是什么 / 有什么用 / 怎么用 / 坑** 四段写。
> 目标:能长期查阅、面试前能快速复习。关联 [[个人求职路线-AI Infra]] · [[阶段0-执行计划]] · [[国内大厂AI-Infra岗位JD分析]]。

> [!success] Week 1 成果
> 从零 C++ → 写出 104 万线程并行的 CUDA kernel + C++ AC 三道 LeetCode。
> JD 缺口里 **"不会 C/C++"** 和 **"不会写 CUDA kernel"** 两项,补成"会基础"。

---

## 〇、C++ vs Python 心态校准

**是什么**:C++ 是**编译型 + 静态类型**语言;Python 是解释型 + 动态类型。

**有什么用**:AI Infra 底层(vLLM / CUDA / Triton)全是 C++,必须会读会写。

**怎么用**:区别于 Python 的三件事——① 要 `#include`;② 变量必须**声明类型**;③ 要**编译**(`g++` / `nvcc`)才能跑。

**坑**:
- C++ **区分"声明"和"赋值**":`int x=1;` 是声明(带类型,同一作用域只能一次),`x=2;` 是赋值(可多次)。Python 不区分,容易写重声明。
- 重复声明报错:`int x=1; int x=2;` → `redeclaration`。(Day6 踩过)

---

## 一、指针 ⭐(Python 党最大难关)

**是什么**:存**内存地址**的变量。`int* p` = "p 是一个装 int 地址的盒子"。

**有什么用**:
- 间接访问/修改内存(CUDA `cudaMalloc` 返回的就是指针)
- 遍历数组、操作链表(面试手撕高频)
- 函数修改外部变量(传地址)

**怎么用**(两个运算符):
```
&x   取地址(变量 → 地址)   "这门牌号是多少"
*p   解引用(地址 → 值)      "去这个门牌号拿东西"
```
```cpp
int x = 10;
int* p = &x;     // p 存 x 的地址
cout << *p;      // 10(解引用读)
*p = 20;         // 通过指针改 x → x 变 20
```

**关键认知**:
- 数组名 = 指向首元素的指针,`arr[i] ≡ *(arr+i)`
- 栈上相邻 int 地址差 4 字节(已实测验证)

**坑**:
- 野指针:`int* p;`(未初始化)指向随机地址,解引用→段错误。声明指针必须让它指向有效内存(`new` 或 `&某变量`)。
- 解引用 nullptr → 段错误。链表题里 `l1->val` 前必须保证 `l1 != nullptr`。
- `->` 是语法糖:`p->val ≡ (*p).val`,源码里 99% 用 `->`。

---

## 二、引用 reference

**是什么**:变量的**别名**。`int& r = x;` 后 r 和 x 是同一块内存。

**有什么用**:
- 函数参数:避免大对象拷贝(尤其 `const T&`,vLLM 源码满地都是)
- 修改外部变量(比指针语法干净)

**怎么用**:
```cpp
int x = 10;
int& r = x;     // r 绑定 x,从此 r 就是 x
r = 20;         // x 也变 20
```
函数三态:
```cpp
void f(int  n);   // 值传递:改副本,外部不变
void f(int* p);   // 指针:调用处写 f(&x)
void f(int& n);   // 引用:调用处写 f(x),像值传但能改外部
```

**坑**(Day3 最坑):
- **必须初始化**:`int& r;` 报错(指针可以先 `int* p;`)
- **无空引用**:不能 nullptr(指针可以)
- **🪓 不可换绑**:`int& r=a; r=b;` **不是**让 r 改绑 b,而是**把 b 的值赋给 a**!引用一旦绑定终身不变。这是 Python 党最容易错的(Python `b=2` 是换标签,C++ 引用没有换标签这回事)。

---

## 三、内存模型:栈 / 堆 / new·delete / RAII

**是什么**:C++ 进程内存分两区:
| | 栈 Stack | 堆 Heap |
|---|---|---|
| 谁分配 | 局部变量,自动 | `new`,手动 |
| 生命周期 | 函数返回自动回收 | 必须 `delete`,否则泄漏 |
| 地址方向 | 高→低 | 低→高 |
| 速度 | 快 | 慢 |

**有什么用**:
- Python 几乎所有对象在堆 + 有 GC,你不用管;C++ **没 GC**,必须知道对象在栈还是堆
- 堆的 new/delete = CUDA `cudaMalloc`/`cudaFree` 的心智原型(Day6 前置)

**怎么用**:
```cpp
int x = 5;                  // 栈:函数返回自动回收
int* p = new int(5);        // 堆:返回地址,用完必须 delete
delete p;                   // 归还
p = nullptr;                // 防悬垂指针

int* arr = new int[100];    // 堆数组
delete[] arr;               // ⚠️ 数组用 delete[](带[]),不是 delete
```

**坑**:
- **返回栈变量地址** = 经典 bug:`return &局部变量;` 函数返回后那块内存已销毁,指针指向垃圾。
- **内存泄漏**:`new` 了没 `delete`,且指针丢失 → 永久泄漏。用 `valgrind --leak-check=full` 检测。
- 现代 C++ **少裸写 new/delete**,优先用 **vector / 智能指针(RAII)** 自动释放。**但 CUDA 显存必须手动管**(没有智能指针)。

**RAII(资源获取即初始化)**:vector / 智能指针在析构时自动归还内存,离开作用域自动回收。实测 `vector_stats.cpp` valgrind **0 泄漏** vs Day3 手动 `new` 漏 `delete` 的 **4 字节**。

---

## 四、STL 容器:vector / string / map

### vector ⭐(≈ Python list,但连续内存)

**是什么**:动态数组,元素类型固定(`vector<int>`、`vector<float>`)。

**有什么用**:最常用容器;**底层连续内存** → 能一把 `cudaMemcpy` 给 GPU(Python list 不行)。

**怎么用**:
```cpp
vector<double> v = {2, 4, 5};
v.push_back(9);          // ≈ append
v.size();                // ≈ len
v[0]; v.back();          // 首元素 / 末元素
v.data();                // ⭐ 底层裸指针(GPU 搬运的钥匙)
```

**为什么 vector 能喂 GPU,Python list 不行**:
```
vector:  [ 2 ][ 4 ][ 5 ]     数据本身连续,data() 给裸指针 → 整块 memcpy ✅
list:    [ ● ][ ● ][ ● ]     存的是指针,真数据散落堆各处 ❌
```
> PyTorch tensor 底层连续(`tensor.data_ptr()`),就是为了高效喂 GPU。**GPU 只认连续内存。**

### string(≈ Python str,但可原地改)

```cpp
string s = "hello"; s += " world";   // ≈ +
s[0] = 'H';                          // ⚠️ C++ string 可原地改;Python str 不可变!
```

### map / unordered_map(≈ dict)

| | `map` | `unordered_map` |
|---|---|---|
| 底层 | 红黑树,**有序**(按 key) | 哈希表,**无序** |
| 查找 | O(log n) | **O(1)** ⭐ |

**API**:
```cpp
unordered_map<int,int> seen;     // ≈ dict
seen.find(k);   // 返回迭代器;seen.end() 表示"没有"  ← 查找用 find/count
seen.count(k);  // 返回 0/1(存不存在)
seen[k] = v;    // 插入/更新
it->first;      // key
it->second;     // value
```

**坑**(Day7 题1):
- **`[]` 会偷插!** `seen[不存在的k]` 会自动插一个 `{k, 0}`(int 默认 0),污染 map。**"只查找"必须用 `find` 或 `count`,绝不用 `[]`**;`[]` 只在"写入/更新"时用。

---

## 五、范围 for + auto

**是什么**:`for (类型 变量 : 容器)` 遍历;`auto` 让编译器推导类型。

**有什么用**:遍历容器的现代写法(对比下标 for)。

**怎么用**:
```cpp
for (int x : v) cout << x;        // 值拷贝
for (auto& x : v) x *= 2;         // 引用,能改原数据
for (const auto& x : v) cout<<x;  // const 引用,大对象只读不拷贝
```
> `auto` 是**编译期**推导(Python 是运行时动态),零运行时开销。

**坑**(Day4/Day7):
- **值拷贝 vs 引用**:`for (int x : v) x*=2;` **改不动 v**(x 是副本);要改原件必须 `for (int& x : v)`。90% 新手第一次栽这。
- `size_t`(无符号)和 `int` 比较 → 警告。循环变量用 `size_t` 或 `(int)v.size()`。

---

## 六、struct / class

**是什么**:把多个变量 + 函数打包成一个类型。

**有什么用**:组织数据(tensor / params 都是 struct);面向对象封装。

**怎么用**:
```cpp
struct Point {
    double x, y;                              // 数据成员
    double dist_to(const Point& o) const {    // 成员函数;const=承诺不改自身
        double dx = x-o.x, dy = y-o.y;
        return sqrt(dx*dx + dy*dy);
    }
};                       // ⚠️ 末尾分号必须有!(Python 党第一个坑)
Point p{1, 2};           // 聚合初始化
p.dist_to(...);          // 成员函数,this 隐式(对比 Python 的 self)
```

**struct vs class**:唯一区别——struct 成员默认 `public`,class 默认 `private`。习惯:**纯数据用 struct,要封装用 class**。

**坑**:
- struct/class 定义**末尾分号**漏了 → 编译报错。
- 成员函数里的 `x` 实际是 `this->x`(隐式 this,对比 Python 显式 self)。

---

## 七、模板 template

**是什么**:**编译期**的代码生成器,`T` 是类型占位符。

**有什么用**:一份代码支持多类型(泛型)。**vLLM / CUDA / Triton 满地都是 `template<typename T>`,因为要同时支持 fp16/fp32/int8 多精度。**

**怎么用**:
```cpp
template<typename T>
T my_add(T a, T b) { return a + b; }
my_add<int>(2, 3);          // 编译器生成 int 版
my_add<double>(1.5, 2.5);   // 生成 double 版
my_add(2, 3);               // 类型可省,编译器推导
```

**为什么是"零开销抽象"**:编译器为每种类型**各生成一份特化代码**(编译期完成),每份都被针对性优化,运行时零开销。对比 Python 一份代码运行时动态分派(有开销)。

**坑**:
- 类型推导冲突:`my_add(2, 3.5)` 报错(T 推不出 int 还是 double),需显式 `my_add<double>(2, 3.5)`。
- 模板错误信息巨长(展开后的代码),别慌,看第一行。

---

## 八、CUDA 入门 ⭐(Day6 重头戏)

### 8.1 两个世界:host(CPU) / device(GPU)

```
   host(CPU)                  device(GPU)
   主机内存 h_a,h_b,h_c  ←──PCIe──→  显存 d_a,d_b,d_c
   ①准备 ⑥收回                    ②③搬过去 ④kernel算 ⑤搬回
```
> `cudaMalloc`/`cudaFree` = Day3 的 `new`/`delete`,**只是换了显存池**。

### 8.2 线程层级:grid / block / thread

```
grid
├── block 0   (每 block 常装 256 thread)
├── block 1
└── ... numBlocks 个 block

⭐ 全局线程索引(必背):
   i = blockIdx.x * blockDim.x + threadIdx.x
     blockIdx.x  = 我在哪个 block
     blockDim.x  = 每 block 多少 thread
     threadIdx.x = 我在 block 内第几
```
> 1 个 thread 管 1 个元素,百万元素 = 百万 thread 并行(Day6 实测 4096×256=1048576 线程)。

### 8.3 三个函数修饰关键字

| 关键字 | 在哪执行 | 谁调 |
|---|---|---|
| `__global__` | GPU | CPU 启动(核函数) |
| `__device__` | GPU | GPU 内部 |
| `__host__` | CPU | 普通 C++ 函数(默认) |

### 8.4 六步骨架(所有 CUDA 程序都这套路)

```
① CPU 准备数据(host 内存,vector 连续)
② cudaMalloc        显存分配(≈ new)
③ cudaMemcpy H2D    数据搬过去
④ kernel<<<grid,block>>>()   GPU 并行算
⑤ cudaMemcpy D2H    结果搬回(此步隐式等 kernel 算完)
⑥ cudaFree          释放显存(≈ delete)
```

```cpp
__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];     // 越界保护
}
// 启动
int threadsPerBlock = 256;
int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;  // 向上取整
add_kernel<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, N);
```

### 8.5 两个 Infra 常识
- **为什么用 float 不用 double**:消费卡(3090)FP32 算力是 FP64 的 **32 倍**。AI 默认 fp16/fp32,double 基本不用且很慢。
- **kernel 启动是异步的**:CPU 发 `<<<>>>` 立即返回,`cudaMemcpy(D2H)` 隐式等它算完。

### 8.6 坑
- `.cu` 后缀,用 **`nvcc`** 编译(不是 g++)。3090 可加 `-arch=sm_86`。
- **向上取整 + `if(i<n)` 必须配套**:向上取整让线程数 ≥ N(防漏算),`if(i<n)` 挡住多余线程防越界。直接 `N/threadsPerBlock` 向下取整会漏算尾部元素。
- `cudaMalloc(&d_a, bytes)` 传的是**指针的地址**(`&d_a`),和 `new` 写法不同。

---

## 九、刷题踩坑精选(Day7,你的个性化坑库)

**[1] 两数之和**
- `int arr[n]`(VLA 变长数组)非标准 C++,动态数组一律用 `vector`
- `unordered_map` 的 `[]` 会偷插空值,**查找只用 `find`/`count`**

**[21] 合并有序链表**(指针重灾区)
- **野指针**:`ListNode* p;` 未初始化,指向垃圾地址 → 段错误。必须 `new`
- **对象 vs 指针**:`ListNode(5)` 是临时对象;`new ListNode(5)` 是堆上节点返回指针。`next` 要的是**指针**,得加 `new`
- **dummy 哨兵**:假头,`val` 不用,`next` 才是真头,返回 `dummy->next`
- **`&&` 不是 `||`**:`while(l1 && l2)`,一个空了就停,剩下的整条接 `tail->next = l1 ? l1 : l2`
- **挪指针不复制**:`tail->next = l1`(接原节点),不是 `new ListNode(l1->val)`(造新的)。链表本质就是改 `next` 指向
- **必须 return**:声明返回 `ListNode*` 就得 `return`

**[26] 删重复项**
- 双指针:`l` 慢(无重复区末尾)、`r` 快(扫描)。`nums[r]!=nums[r-1]` 就 `nums[l++]=nums[r]`
- 函数职责单一:别把打印塞进函数,放 `main`

---

## 十、Week 1 三个认知锚点(面试一句话能说清)

1. **vector 连续内存 → 能一把 `cudaMemcpy` 喂 GPU;Python list 不行,因存的是指针、数据散落。PyTorch tensor 连续就是为了高效喂 GPU。**
2. **引用不可换绑**:`b=c` 是赋值不是改绑(Day3 最坑)。引用绑定终身不变。
3. **CUDA = host/device 两世界,`cudaMalloc`/`cudaFree` 就是 `new`/`delete` 换了显存池**;6 步骨架 = 所有 CUDA 程序的模板。

---

## 十一、Week 2 预告(CUDA 内功)

| Day | 主题 | 要点 |
|---|---|---|
| 1–2 | 线程索引打印 | 吃透 `i = blockIdx.x*blockDim.x + threadIdx.x` |
| 3 | **shared memory** + 归约/转置 | GPU 性能优化第一把钥匙 |
| 4–5 | **naive matmul** + benchmark TFLOPS | ⭐ 算子岗面试核心战场 |
| 6 | **nsys/ncu profile** | 看 kernel 时间花在哪 |
| 7 | 精读 Simon Boehm《优化 CUDA matmul》 | ⭐ 业界经典 |

> Week 1 学"写对",Week 2 学"**写快**"。

*关联:[[个人求职路线-AI Infra]] · [[阶段0-执行计划]] · [[国内大厂AI-Infra岗位JD分析]]*
