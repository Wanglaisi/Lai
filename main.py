# main.py
from fastapi import FastAPI, Request
import httpx, random, time, asyncio
from collections import deque
import os
import json

app = FastAPI(title="NVIDIA NIM Multi-Key Proxy")

NIM_BASE = "https://integrate.api.nvidia.com/v1"

# 从环境变量读取 keys
raw_keys = os.getenv("NIM_API_KEYS", "")
KEYS = [k.strip() for k in raw_keys.split(",") if k.strip()]

if not KEYS:
    raise ValueError("NIM_API_KEYS 环境变量为空！请在 Northflank 设置")

key_queue = deque(KEYS)
RATE_LIMITS = {k: asyncio.Semaphore(35) for k in KEYS}  # 每个key限流

async def get_next_key():
    key = key_queue.popleft()
    key_queue.append(key)
    return key

# ==================== 健康检查 ====================
@app.get("/")
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "keys_loaded": len(KEYS),
        "message": "NVIDIA NIM 多 key 轮询代理运行正常（升级版）"
    }

# ==================== 列出模型（强烈推荐） ====================
@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "nvidia/llama-3.3-nemotron-super-49b-v1", "object": "model"},
            {"id": "moonshotai/kimi-k2.5", "object": "model"},
            {"id": "deepseek-ai/deepseek-r1-distill-llama-70b", "object": "model"},
            {"id": "nvidia/nemotron-3-super-120b-a12b", "object": "model"},
            {"id": "z-ai/glm-5", "object": "model"}
        ]
    }

# ==================== 主代理接口 ====================
@app.post("/v1/chat/completions")
async def proxy(request: Request):
    try:
        body = await request.json()
        headers = dict(request.headers)
        headers.pop("host", None)
        # 随机 UA 防检测
        headers["user-agent"] = f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/{random.randint(500,600)}.0"

        key = await get_next_key()

        async with RATE_LIMITS[key]:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{NIM_BASE}/chat/completions",
                    json=body,
                    headers={**headers, "Authorization": f"Bearer {key}"},
                    follow_redirects=True,
                )

                # jitter 随机延迟
                await asyncio.sleep(random.uniform(0.2, 0.6))

                if resp.is_success:
                    return resp.json()
                else:
                    # 改进：返回 NVIDIA 的真实错误信息
                    error_text = resp.text[:500]
                    return {
                        "error": {
                            "message": f"NIM 后端错误: {error_text}",
                            "type": "nvidia_error",
                            "code": resp.status_code
                        }
                    }

    except Exception as e:
        return {"error": {"message": f"代理内部错误: {str(e)}", "type": "proxy_error"}}

# 启动提示
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
