# main.py
from fastapi import FastAPI, Request
import aiohttp, random, asyncio, os
from collections import deque

app = FastAPI(title="NIM Proxy - 极简 aiohttp 版")

NIM_BASE = "https://integrate.api.nvidia.com/v1"

raw_keys = os.getenv("NIM_API_KEYS", "")
KEYS = [k.strip() for k in raw_keys.split(",") if k.strip()]

if not KEYS:
    raise ValueError("NIM_API_KEYS 环境变量为空！")

key_queue = deque(KEYS)
RATE_LIMITS = {k: asyncio.Semaphore(35) for k in KEYS}

async def get_next_key():
    key = key_queue.popleft()
    key_queue.append(key)
    return key

@app.get("/")
@app.get("/health")
async def health():
    return {"status": "ok", "keys_loaded": len(KEYS), "message": "极简 aiohttp 版运行中"}

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "moonshotai/kimi-k2.5", "object": "model"},
            {"id": "z-ai/glm-5", "object": "model"},
            {"id": "deepseek-ai/deepseek-r1-distill-llama-70b", "object": "model"},
        ]
    }

@app.post("/v1/chat/completions")
async def proxy(request: Request):
    body = await request.json()
    headers = dict(request.headers)
    headers.pop("host", None)
    headers["user-agent"] = f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/{random.randint(500,600)}.0"

    key = await get_next_key()

    try:
        async with RATE_LIMITS[key]:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=90)) as session:
                async with session.post(
                    f"{NIM_BASE}/chat/completions",
                    json=body,
                    headers={**headers, "Authorization": f"Bearer {key}"}
                ) as resp:
                    await asyncio.sleep(random.uniform(0.3, 0.7))
                    if resp.status == 200:
                        return await resp.json()
                    else:
                        text = await resp.text()
                        return {"error": f"NIM 错误 ({resp.status}): {text[:400]}"}
    except Exception as e:
        return {"error": f"代理错误: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
