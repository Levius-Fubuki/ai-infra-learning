---
title: Week 4 笔记(Day1):vLLM 初体验——部署与对照
phase: 0
week: 4
period: 2026-08-22
tags:
  - ai-infra
  - 笔记
  - vllm
created: 2026-08-22
---

# Week 4 笔记(Day1):vLLM 初体验——部署与对照

> [!info] 笔记体例
> 概念块沿用四段式 **是什么 / 有什么用 / 怎么用 / 坑**;代码块采用 **逐行讲解 + 具体数字举例**。当日预测题(Q1-Q5)完整解答见 §四。
> 关联 [[Week3笔记-Day7-收官与面试弹药库]] · [[Week3笔记-Day6-Flash-Attention]] · [[阶段0-执行计划]]。

> [!success] Day1 成绩
> - **Qwen2.5-7B-Instruct 在 3090 上跑成 vLLM 服务**(vllm 0.6.6,镜像自带),OpenAI 兼容接口 curl 跑通。
> - **单请求对照:HF 原生 28.6 tok/s vs vLLM 46.7 tok/s = 1.63×**(符合预测 1-3×)。
> - 启动日志挖出全金矿:内存账本 21.21GiB = weights 14.25 + activation 1.09 + non_torch 0.10 + **KV 5.77**;并发上限 13.18×;#GPU blocks 6749;dtype=bf16;**Using Flash Attention backend = W3 Day6 的 kernel 被 vLLM 用到实锤**;Capturing cudagraphs。

> [!warning] 本篇的"性能数字"怎么读
> 1.63× 是**单请求、无并发**的数——它回答"vLLM 的 kernel/开销层优化了多少",不回答"vLLM 的价值"。vLLM 的真正王牌(连续批处理)要等并发才发威,Day2 差距会到 5-10× 级。**归因要分场景**:单请求=flash-attn/cudagraph/低开销,并发=批处理+KV 管理。
> 严格对照瑕疵:HF 脚本显式设 `torch_dtype=fp16`,vLLM 默认用模型原厂 `bf16`——两边都是 2 字节、吞吐相近,1.63× 大体仍是 kernel/开销差异,但严谨的对照应同 dtype。

---

## 〇、vLLM 是什么(详细介绍)

### 定位

**vLLM 是一个开源推理服务框架**(加州伯克利 LMSYS 团队,后成为独立公司 vLLM)。输入一个 HuggingFace 模型,输出一个 OpenAI 兼容的 HTTP 服务,吞吐比 HF 原生高一个量级。**它同时是"服务"和"推理引擎"**:不只帮你把模型跑起来,还用自研的调度器和 GPU kernel 让并发吞吐逼近硬件上限。

**和 HF transformers 的本质区别**(求职必答):

| | HF transformers | vLLM |
|---|---|---|
| 定位 | 模型库/研究/训练 | 推理服务/生产 |
| 批处理 | 静态(等齐再跑,慢者拖累) | **连续**(随到随插,生成完即退) |
| KV cache | 每请求独立、不共享、不管理 | 分页共享池,按需分配 |
| GPU kernel | 复用 torch 算子(每步 Python 开销) | **自研 Triton/CUDA**(flash-attn 等) |
| 接口 | Python 函数(generate) | **OpenAI HTTP API**(chat/completions) |
| 目标用户 | 研究者 | 生产服务/AI 应用 |

### 架构:调度器 + GPU worker + HTTP 壳

```
浏览器/脚本 ──HTTP──▶ vLLM 服务进程(一个 Python 进程)
                        ├─ HTTP 层   ≈ FastAPI:解析请求、流式返回(SSE)
                        ├─ 调度器(LLMEngine)= 灵魂:决定谁先算、谁插队、KV 怎么分配
                        └─ GPU worker = 模型权重 + 一堆自研 kernel
                                          (flash-attn、fused ops、PagedAttention 存储)
```

vLLM 没有复用 transformers 的前向函数——它**自己写 kernel**(基于 Triton/CUDA)。这就是它和"你上周用 Triton 写 flash-attn"的接口:**vLLM 里的 attention 就是你写的那个分块在线 softmax 的生产级版本**,只不过针对任意 shape/并发/长序列做了工程化。

### 三大提速机制(本周主线,预告)

| 机制 | 解决什么 | 关键日 |
|---|---|---|
| **连续批处理** | 静态批等齐才跑、慢者拖累 → 请求级动态调度 | Day2 |
| **KV cache** | attention 的 K/V 重复算 → 存起来复用 | Day3 |
| **PagedAttention** | KV 显存碎片/浪费 → 分页存储(Flash-Attention 的分块思想投影到"KV 怎么存") | Day3 |

**为什么这对你重要**:网易/京东的推理服务、所有 LLM 应用的后端,底层都是这一套。你后端+Agent 背景 → vLLM 就是衔接点。

---

## 一、环境与部署(命令流程)

```bash
# 开工三连
source /etc/network_turbo              # 学术加速(每个新 shell)
df -h                                   # 磁盘:7B 要 ~15GB,紧张则 HF_HOME 指数据盘
python3 -c "import vllm; print(vllm.__version__)"   # 镜像自带 0.6.6,省去安装
```

**顺序很重要:先跑 HF 基线,再启动 vLLM serve**——14GB 模型两份同时上 24GB 卡会 OOM。

```bash
# ① HF 原生对照(自动下载模型到缓存)
python3 hf_baseline.py
# ② 启动 vLLM 服务(新终端)
vllm serve Qwen/Qwen2.5-7B-Instruct --max-model-len 8192 --max-num-seqs 16 --port 8000
# ③ 打服务
python3 vllm_client.py
```

**两个 flag 就是第一个知识点——KV 显存预算**:
```
预分配 KV ≈ max-model-len × max-num-seqs × 每 token KV 字节(56KB)
默认可能 32768 × 256 × 56KB ≈ 470GB → 直接 OOM!
设 8192 × 16 × 56KB ≈ 7.3GB + 权重 14GB ≈ 21GB < 24GB ✓
"为什么默认参数会炸"比"怎么调好"更值得记住。
```

---

## 二、对照实验:1.63× 从哪来(归因分析)

```
HF 原生: 生成 71 tokens,2.48s → 28.6 tok/s(无流式、无动态批、复用 torch 算子)
vLLM  : 生成 78 tokens,1.67s → 46.7 tok/s(flash-attn + cudagraph + 低每步开销)
       → 46.7 / 28.6 = 1.63×
```

**为什么单请求下也有差距**(日志作证):
1. **`Using Flash Attention backend`**——你 W3 Day6 写的那个分块在线 softmax 的 production 版,省掉 S/P 中间流量;
2. **`Capturing cudagraphs`**——静态图捕获,把一串 GPU 算子固定形状一次提交,省每步 launch 开销;
3. **更低每步开销**——HF 每生成一个 token 都走一遍 Python 前向;vLLM 编译/捕获后 Python 层几乎为零。

**为什么只有 1.63× 而不是 10×**:单请求下两边都是同一张卡做同样的 GPU 数学,吞吐受算力帽/带宽帽约束,优化只能挤常数。**KV cache / PagedAttention 在单请求下贡献≈0**(一个序列一份 KV、无碎片、无竞争)——它们是并发/长序列场景的红利,不是单请求的红利。

---

## 三、启动日志挖金矿 ⭐

### 3.1 内存账本(一启动就分好)

```
the current vLLM instance can use total_gpu_memory (23.57GiB) x 0.90 = 21.21GiB
model weights take 14.25GiB; non_torch_memory 0.10GiB;
PyTorch activation peak memory takes 1.09GiB; the rest reserved for KV Cache is 5.77GiB.
```

| 项 | 大小 | 说明 |
|---|---|---|
| weights | 14.25 GiB | 7B × 2B ≈ 14GB(bf16) |
| activation 峰值 | 1.09 GiB | prefill 时的中间张量 |
| non_torch | 0.10 GiB | 运行时杂项 |
| **KV cache** | **5.77 GiB** | **按并发预算一次性预留,不是按当前请求** |
| 合计 | 21.21 GiB | = 24GB 的 88%(0.9 util) |

**关键认知:KV 5.77GB 是"预留"不是"在用"**——日志显示 `GPU KV cache usage: 0.0%`(单请求),但启动时就划走了。vLLM 的设计哲学:**拿显存换吞吐**——宁可多留 KV 槽位,也别让并发请求排队等。

### 3.2 并发上限 13.18× 的数学(法医式推导)

```
# GPU blocks: 6749(block_size 默认 16 tokens)
KV 容量 = 6749 × 16 = 107,984 tokens
校验:107,984 × 57,344B = 6.19GB ≈ 5.77GiB ✓
Maximum concurrency for 8192 tokens/request = 107,984 ÷ 8192 = 13.18× ✓
```

三个数字(5.77GiB / 6749 blocks / 13.18×)互相咬合,和 W3 的"数字会撒谎"教训同一族:**日志里的每个数都能用公式验证**。

### 3.3 两个新 dtype/机制知识点

- **`dtype=torch.bfloat16`**:vLLM 默认用模型原厂 dtype。bf16 = 指数 8 位(同 fp32,范围大、不溢出)+ 尾数 7 位(比 fp16 的 10 位精度低)。Qwen 官方推荐 bf16。
- **`Capturing cudagraphs`**:静态图捕获——固定形状的算子序列编译成一张图一次提交,省掉 per-step launch 开销。是单请求低延迟来源之一(Day5 用 `--enforce-eager` 关掉它看代价)。

---

## 四、预测题详解(Q1-Q5)

### Q1:单请求下 vLLM 比 HF 快多少?区间和理由?

**预测**:1-3×。**实测 1.63×**。
**理由**(归因要分场景):
```
同卡同 GPU 数学 → 吞吐受算力帽/带宽帽约束,天花板一样
单请求可挤的红利只有常数:
  ① flash-attn kernel(省 4N² 中间流量,W3 Day6)
  ② cudagraph(省 launch 开销)
  ③ 低每步 Python 开销
错误归因:KV cache / PagedAttention 是并发红利,单请求下贡献≈0
          (一个序列一份 KV、无碎片、无竞争)。
```

### Q2:如果单请求差距不大,vLLM 的价值在哪?

**在并发**。vLLM 的核心是**连续批处理**:请求随到随插队、生成完即退,GPU 永远满负荷跑多个请求;HF 是静态批(等齐再跑)或串行。单请求 1.63× → 并发 8 时 5-10×,这就是 Day2 要亲眼看的差距。

### Q3:7B fp16 为什么 ~14GB?KV cache 每 token 为什么 ~56KB?

```
权重:7×10⁹ 参数 × 2B = 14GB
KV/token:2(K 和 V) × 28 层 × 4 KV 头(GQA) × 128 维 × 2B = 57,344B ≈ 56KB
  头数为什么是 4 不是 28?Qwen2.5-7B 用 GQA(分组查询注意力):Q 有 28 个头共享 4 组 K/V,
  省显存——这是 KV cache 时代的标准设计(Day3 细讲)。
```

**KV 显存预算详解**(你点名的部分):
```
vLLM 启动时按"最大可能并发"一次性预留:
  KV 预算 = max-model-len × max-num-seqs × 56KB
  我们设的 8192 × 16 × 56KB ≈ 7.3GB(实际按剩余显存给到 5.77GiB)
为什么默认参数会炸:max-model-len 默认 32768、max-num-seqs 默认 256
  → 32768 × 256 × 56KB ≈ 470GB >> 24GB → OOM
调参口诀:权重定死(14GB),剩多少给 KV;max-len 和 max-seqs 乘起来别超预算。
```

### Q4:vllm serve 起来后显存为什么 >14GB?

**不是分词器**(它在 CPU,占显存可忽略)。真正的四笔账(§3.1):
```
21.21GiB = weights 14.25 + activation 峰值 1.09 + non_torch 0.10 + KV 预留 5.77
```
重点是 **KV 是按并发预算提前预留的**,哪怕当前只有 1 个请求、usage 0.0%。

### Q5:HF 能不能流式?首 token 体感?

**能**——`transformers` 有 `TextIteratorStreamer`。但:
- HF 首 token 之前要把整个 prefill 算完,且每步 Python 开销高 → TTFT 较长;
- vLLM 原生 OpenAI **SSE 流式**,flash-attn + cudagraph → TTFT 明显更短。
你做过 Agent/LLM 应用,"首 token 快到不像话"的体感就是 TTFT——Day4 用指标正式测它。

---

## 五、指标小抄(Week4 起,新表)

| 指标 | 公式 | Day1 实例 |
|---|---|---|
| 单请求吞吐 | completion_tokens ÷ 耗时 | HF 28.6 / vLLM 46.7 tok/s |
| 权重显存 | 参数量 × 字节 | 7B × 2 = 14GB |
| KV/token | 2×层×KV头×head_dim×字节 | 2×28×4×128×2 = 56KB |
| KV 预算 | max_len × max_seqs × KV/token | 8192×16×56KB ≈ 7.3GB |
| 并发上限 | KV 预算 ÷ (max_len × KV/token) | 5.77GiB ÷ (8192×56KB) = 13.18× |
| 总显存账本 | weights+activation+non_torch+KV | 14.25+1.09+0.10+5.77 = 21.21GiB |

---

## 六、坑合集(Day1)

1. **先跑 HF 基线,再启动 vLLM serve**——14GB 模型两份同卡会 OOM。
2. **默认 max-model-len/max-num-seqs 会炸**:32768×256×56KB ≈ 470GB;务必显式设。
3. **KV 是预留不是在用**:nvidia-smi 看到 21GB 占用不等于都在干活,0.9 util 是全都要。
4. **严格对照要同 dtype**:vLLM 默认 bf16,HF 脚本设 fp16——吞吐相近但口径要统一。
5. **单请求差距只有 1-3×,别拿它吹 vLLM**;vLLM 的价值在并发,归因要分场景。
6. `df -h` 紧张时先设 `HF_HOME` 指数据盘再下模型。

---

## 七、自测(闭卷过一遍)

1. vLLM 和 HF 的本质区别(四个维度)?
2. 三大提速机制各解决什么?单请求时哪个贡献≈0?
3. Qwen2.5-7B 的 KV/token 怎么算?为什么 KV 头是 4 不是 28?(GQA)
4. `--max-model-len 32768 --max-num-seqs 256` 为什么 OOM?算一下。
5. 启动日志里 13.18× 怎么来的?6749 blocks 和它什么关系?
6. bf16 和 fp16 的差异?(指数/尾数/适用)
7. 为什么 vLLM 单请求也快 1.63×?(三个来源)

---

## 八、与 Day2 的连接

- **Day2 连续批处理**:静态批 vs 动态批;并发 8 请求,HF 串行 vs vLLM 交错——差距从 1.63× 拉到 5-10×。
- **Day3 KV cache + PagedAttention**:今天预留的 5.77GiB 怎么被按需分页管理;GQA 的 4 个 KV 头为什么省显存。
- 面试连接:W3 Day7 弹药库 Q9(Q2)今天有了 vLLM 版答案。

> 上一篇:[[Week3笔记-Day7-收官与面试弹药库]] · 下一篇:Day2 连续批处理(待续)
