# hf_batch.py —— W4D2:HF 串行 vs HF 静态批(先停 vLLM!)
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, time

model_name = "Qwen/Qwen2.5-7B-Instruct"
tok = AutoTokenizer.from_pretrained(model_name)
tok.padding_side = "left"          # ⭐ decoder-only 批量生成的标准姿势(右边 padding 会喂错位置)
model = AutoModelForCausalLM.from_pretrained(model_name,
        torch_dtype=torch.bfloat16, device_map="cuda").eval()   # 和 vLLM 同 dtype

LENS = [30, 50, 70, 90, 110, 130, 150, 170]
texts = [tok.apply_chat_template([{"role": "user", "content": f"从1数到{n},只输出数字,逗号分隔"}],
          tokenize=False, add_generation_prompt=True) for n in LENS]

# ① 串行:一个一个来("天真服务"的样子)
t0, total = time.time(), 0
for text, n in zip(texts, LENS):
    ids = tok(text, return_tensors="pt").to("cuda")
    out = model.generate(**ids, max_new_tokens=n, do_sample=False)
    total += out.shape[1] - ids["input_ids"].shape[1]
t1 = time.time() - t0
print(f"① HF 串行:  {t1:.2f}s,{total} tok → {total/t1:.1f} tok/s")

# ② 静态批:padding 拼一个 batch,一次 generate(所有请求一起返回)
enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
t0 = time.time()
out = model.generate(**enc, max_new_tokens=170, do_sample=False)   # 只能按最长的算
t2 = time.time() - t0
print(f"② HF 静态批: {t2:.2f}s(整批一起返回——最短的 30tok 也要等到最后)")
print(f"   串行/静态批 = {t1/t2:.1f}×")