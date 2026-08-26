# kv_watch.py —— W4D3:看 KV cache 随 context 增长 + 并发共享 + 超限报错
import json, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL, MODEL = "http://localhost:8000/v1/chat/completions", "Qwen/Qwen2.5-7B-Instruct"

def chat(prompt, max_tokens):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.0}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=600))
    return {"tok": r["usage"]["completion_tokens"], "dt": time.time() - t0}

# [A] 长生成 1500 tok(约 30s):盯服务端日志的 GPU KV cache usage 从 ~0 爬升
print("[A] 单请求长生成 1500 tok(约30s)——去看服务端日志 KV usage 变化")
print(chat("讲一个关于机器人的童话故事", 1500))

# [B] 并发 6×400:看 Running: 涨到 6、KV usage 一起涨
print("\n[B] 并发 6×400——看 Running: 和 KV usage 一起涨")
def work(i):
    return chat(f"第{i}个问题:什么是机器学习", 400)
with ThreadPoolExecutor(max_workers=6) as ex:
    rs = list(ex.map(work, range(6)))
print("各耗时:", [f"{r['dt']:.2f}" for r in rs], "总 tokens:", sum(r["tok"] for r in rs))

# [C] 超限:9000 > max-model-len 8192 → 应 400
print("\n[C] 超 max-model-len 演示(9000>8192,应报错):")
try:
    print(chat("测试", 9000))
except Exception as e:
    print("如预期报错:", str(e)[:120])