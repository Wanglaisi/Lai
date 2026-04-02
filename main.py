# main.py
from fastapi import FastAPI, Request
import aiohttp
import random
import asyncio
import os
from collections import deque

app = FastAPI()

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
    return {"status": "ok", "keys_loaded": len(KEYS), "message": "极简版启动成功"}

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
