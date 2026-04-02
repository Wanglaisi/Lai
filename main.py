# main.py
from fastapi import FastAPI, Request
import httpx, random, time, asyncio
from collections import deque
import os

app = FastAPI()
NIM_BASE = "https://integrate.api.nvidia.com/v1"

# 从 Northflank 加密变量读取 keys
raw_keys = os.getenv("NIM_API_KEYS", "")
KEYS = [k.strip() for k in raw_keys.split(",") if k.strip()]

if not KEYS:
    raise ValueError("NIM_API_KEYS 环境变量为空！请在 Northflank 设置")

key_queue = deque(KEYS)
RATE_LIMITS = {k: asyncio.Semaphore(35) for k in KEYS}  # per-key 令牌桶

async def get_next_key():
    key = key_queue.popleft()
    key_queue.append(key)
    return key

@app.post("/v1/chat/completions")
async def proxy(request: Request):
    body = await request.json()
    headers = dict(request.headers)
    headers.pop("host", None)
    headers["user-agent"] = f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/{random.randint(500,600)}"

    key = await get_next_key()
    async with RATE_LIMITS[key]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{NIM_BASE}/chat/completions",
                json=body,
                headers={**headers, "Authorization": f"Bearer {key}"},
                follow_redirects=True,
            )
            # jitter 防检测
            await asyncio.sleep(random.uniform(0.15, 0.45))

    return resp.json() if resp.is_success else {"error": resp.text}

@app.get("/health")
async def health():
    return {"status": "ok", "keys_loaded": len(KEYS)}
