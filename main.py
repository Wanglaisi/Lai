# main.py
from fastapi import FastAPI, Request
import httpx, random, time, asyncio
from collections import deque
import os

app = FastAPI(title="NVIDIA NIM Multi-Key Proxy")

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
    return {"status": "ok", "keys_loaded": len(KEYS), "message": "升级版代理 - 修复 Content-Length 问题"}

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "nvidia/llama-3.3-nemotron-super-49b-v1", "object": "model"},
            {"id": "deepseek-ai/deepseek-r1-distill-llama-70b", "object": "model"},
            {"id": "nvidia/nemotron-3-super-120b-a12b", "object": "model"},
            {"id": "z-ai/glm-5", "object": "model"},
            {"id": "moonshotai/kimi-k2.5", "object": "model"}   # 保留但不推荐首选
        ]
    }

@app.post("/v1/chat/completions")
async def proxy(request: Request):
    body = await request.json()
    headers = dict(request.headers)
    headers.pop("host", None)
    headers["user-agent"] = f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/{random.randint(500,600)}.0"

    key = await get_next_key()
    attempt = 0
    max_attempts = 2

    while attempt < max_attempts:
        try:
            async with RATE_LIMITS[key]:
                # 关键修复：关闭 http2 + 增加超时 + limits
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(90.0, connect=15.0),
                    limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
                    http2=False   # 临时关闭 http2 规避协议错误
                ) as client:
                    resp = await client.post(
                        f"{NIM_BASE}/chat/completions",
                        json=body,
                        headers={**headers, "Authorization": f"Bearer {key}"},
                        follow_redirects=True,
                    )

                    await asyncio.sleep(random.uniform(0.25, 0.65))

                    if resp.is_success:
                        return resp.json()
                    else:
                        error_text = resp.text[:800]
                        return {
                            "error": {
                                "message": f"NIM 返回错误 ({resp.status_code}): {error_text}",
                                "type": "nvidia_backend_error",
                                "status_code": resp.status_code
                            }
                        }
        except Exception as e:
            attempt += 1
            if attempt == max_attempts:
                return {"error": {"message": f"代理错误 (已重试): {str(e)}", "type": "proxy_error"}}
            await asyncio.sleep(0.5)  # 重试前等待

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
