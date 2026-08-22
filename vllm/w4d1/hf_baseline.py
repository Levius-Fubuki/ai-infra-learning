# hf_baseline.py —— 同模型 HF 原生 generate(对照:无动态批、无 vLLM 内存管理)
# 运行:python3 hf_baseline.py(首次会下载模型 ~15GB,之后用缓存)
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, time

model_name = "Qwen/Qwen2.5-7B-Instruct"
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="cuda")

prompt = "用三句话介绍 GPU 编程"
messages = [{"role": "user", "content": prompt}]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
ids = tok(text, return_tensors="pt").to("cuda")

t0 = time.time()
out = model.generate(**ids, max_new_tokens=100)
t1 = time.time()
gen = out[0][ids["input_ids"].shape[1]:]
dt = t1 - t0
print(f"HF 原生:生成 {len(gen)} tokens,用时 {dt:.2f}s,吞吐 {len(gen)/dt:.1f} tokens/s")
print("回复:", tok.decode(gen, skip_special_tokens=True))