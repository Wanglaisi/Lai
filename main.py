# main.py
from fastapi import FastAPI, Request
import aiohttp, random, asyncio, os, json
from collections import deque

app = FastAPI(title="NVIDIA NIM Multi-Key Proxy - aiohttp 版")

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
    return {"status": "ok", "keys_loaded": len(KEYS), "message": "aiohttp 版 - 已解决 Content-Length 错误"}

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "moonshotai/kimi-k2.5", "object": "model"},
            {"id": "z-ai/glm-5", "object": "model"},
            {"id": "deepseek-ai/deepseek-r1-distill-llama-70b", "object": "model"},
            {"id": "nvidia/nemotron-3-super-120b-a12b", "object": "model"},
        ]
    }

@app.post("/v1/chat/completions")
async def proxy(request: Request):
    body = await request.json()
    headers = dict(request.headers)
    headers.pop("host", None)
    headers["user-agent"] = f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/{random.randint(500,600)}.0"

    key = await get_next_key()
    max_attempts = 3

    for attempt in range(max_attempts):
        try:
            async with RATE_LIMITS[key]:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
                    async with session.post(
                        f"{NIM_BASE}/chat/completions",
                        json=body,
                        headers={**headers, "Authorization": f"Bearer {key}"},
                    ) as resp:
                        
                        await asyncio.sleep(random.uniform(0.3, 0.8))

                        if resp.status == 200:
                            try:
                                return await resp.json()
                            except:
                                text = await resp.text()
                                return {"error": {"message": f"返回非JSON: {text[:300]}", "status": resp.status}}
                        else:
                            text = await resp.text()
                            return {
                                "error": {
                                    "message": f"NIM 后端错误 ({resp.status}): {text[:600]}",
                                    "type": "nvidia_error",
                                    "status_code": resp.status
                                }
                            }
        except Exception as e:
            if attempt == max_attempts - 1:
                return {"error": {"message": f"最终代理错误: {str(e)}", "type": "proxy_error", "attempts": max_attempts}}
            await asyncio.sleep(0.6 * (attempt + 1))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
