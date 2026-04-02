# main.py
from fastapi import FastAPI, Request
import aiohttp
import random
import asyncio
import os
from collections import deque

app = FastAPI(title="NVIDIA NIM Multi-Key Proxy")

NIM_BASE = "https://integrate.api.nvidia.com/v1"

# 从 Northflank 加密变量读取 keys
raw_keys = os.getenv("NIM_API_KEYS", "")
KEYS = [k.strip() for k in raw_keys.split(",") if k.strip()]

if not KEYS:
    raise ValueError("NIM_API_KEYS 环境变量为空！请在 Northflank 设置")

key_queue = deque(KEYS)
RATE_LIMITS = {k: asyncio.Semaphore(35) for k in KEYS}   # 每个 key 限流

async def get_next_key():
    key = key_queue.popleft()
    key_queue.append(key)
    return key

# ====================== 健康检查 ======================
@app.get("/")
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "keys_loaded": len(KEYS),
        "message": "NIM 多 key 轮询代理运行正常"
    }

# ====================== 模型列表（新增） ======================
@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "moonshotai/kimi-k2.5", "object": "model"},
            {"id": "z-ai/glm-5", "object": "model"},
            {"id": "deepseek-ai/deepseek-r1-distill-llama-70b", "object": "model"},
            {"id": "nvidia/nemotron-3-super-120b-a12b", "object": "model"},
            {"id": "nvidia/llama-3.3-nemotron-super-49b-v1", "object": "model"},
        ]
    }

# ====================== 主代理接口 ======================
@app.post("/v1/chat/completions")
async def proxy(request: Request):
    body = await request.json()
    headers = dict(request.headers)
    headers.pop("host", None)
    headers["user-agent"] = f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/{random.randint(500,600)}.0"

    key = await get_next_key()

    try:
        async with RATE_LIMITS[key]:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=120)
            ) as session:
                async
