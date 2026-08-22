# vllm_client.py —— 打 vLLM 的 OpenAI 兼容接口,测单请求(服务先启动)
import json, time, urllib.request

URL, MODEL = "http://localhost:8000/v1/chat/completions", "Qwen/Qwen2.5-7B-Instruct"

def chat(prompt, max_tokens=100):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "stream": False}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=120))
    dt = time.time() - t0
    usage = r["usage"]   # prompt_tokens / completion_tokens
    return dt, usage, r["choices"][0]["message"]["content"]

dt, usage, text = chat("用三句话介绍 GPU 编程", 100)
n = usage["completion_tokens"]
print(f"vLLM:生成 {n} tokens,用时 {dt:.2f}s,吞吐 {n/dt:.1f} tokens/s")
print("回复:", text)