# vllm_concurrent.py —— W4D2:并发打 vLLM,看连续批处理
import json, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL, MODEL = "http://localhost:8000/v1/chat/completions", "Qwen/Qwen2.5-7B-Instruct"

def chat(idx, max_tokens, delay=0.0):
    time.sleep(delay)
    body = {"model": MODEL,
            "messages": [{"role": "user", "content": f"从1数到{max_tokens},只输出数字,逗号分隔"}],
            "max_tokens": max_tokens, "temperature": 0.0}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=300))
    return {"idx": idx, "want": max_tokens,
            "got": r["usage"]["completion_tokens"], "dt": time.time() - t0}

# ───── [A] 齐发 8 个,长度 30~170 ─────
print("[A] 齐发 8 请求(长度 30~170):")
LENS = [30, 50, 70, 90, 110, 130, 150, 170]
t0 = time.time()
with ThreadPoolExecutor(max_workers=8) as ex:
    res = list(ex.map(lambda i: chat(i, LENS[i]), range(8)))
wall = time.time() - t0

print(f"{'req':>3} {'want':>4} {'got':>4} {'耗时s':>6}")
for r in sorted(res, key=lambda x: x["dt"]):        # 按完成时间排序 = 交错证据
    print(f"{r['idx']:>3} {r['want']:>4} {r['got']:>4} {r['dt']:>6.2f}")
tot = sum(r["got"] for r in res)
print(f"\n墙钟={wall:.2f}s | 聚合吞吐={tot}tok/{wall:.2f}s={tot/wall:.1f} tok/s(单请求 46.7)")
print(f"最短请求 {min(r['dt'] for r in res):.2f}s 就回了;最长 {max(r['dt'] for r in res):.2f}s")

# ───── [B] 错峰到达:每 1.5s 来一个(盯服务端日志的 Running:)─────
print("\n[B] 错峰 4 请求(各 100 tok,0/1.5/3/4.5s 到达)——现在去看服务端日志!")
with ThreadPoolExecutor(max_workers=4) as ex:
    res_b = [f.result() for f in [ex.submit(chat, i, 100, delay=1.5*i) for i in range(4)]]
print("各请求耗时:", [f"{r['dt']:.2f}" for r in res_b])

