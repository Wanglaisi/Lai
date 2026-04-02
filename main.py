import asyncio
import os
import random
from collections import deque
from contextlib import asynccontextmanager

import aiohttp
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

NIM_BASE = "https://integrate.api.nvidia.com/v1"

raw_keys = os.getenv("NIM_API_KEYS", "")
KEYS = [k.strip() for k in raw_keys.split(",") if k.strip()]

if not KEYS:
    raise ValueError("NIM_API_KEYS 环境变量为空！请在 Northflank Environment 中设置")

key_queue = deque(KEYS)

# 用全局字典，在 lifespan 中初始化 Semaphore，避免事件循环问题
RATE_LIMITS: dict[str, asyncio.Semaphore] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 应用启动后，在事件循环内安全创建 Semaphore
    for k in KEYS:
        RATE_LIMITS[k] = asyncio.Semaphore(35)
    yield
    # 应用关闭时的清理逻辑（如有需要）


app = FastAPI(title="NVIDIA NIM Multi-Key Proxy", lifespan=lifespan)


def get_next_key() -> str:
    # 轮询不需要 async，deque 操作本身是线程安全的
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


# ====================== 模型列表 ======================

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
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "无效的 JSON 请求体"})

    is_stream = body.get("stream", False)
    key = get_next_key()

    forward_headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if is_stream else "application/json",
        "User-Agent": f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/{random.randint(500, 600)}.0",
    }

    try:
        async with RATE_LIMITS[key]:
            await asyncio.sleep(random.uniform(0.1, 0.4))  # jitter

            if is_stream:
                # 流式响应：直接透传 SSE 数据块
                async def stream_generator():
                    async with aiohttp.ClientSession(
                        timeout=aiohttp.ClientTimeout(total=120)
                    ) as session:
                        async with session.post(
                            f"{NIM_BASE}/chat/completions",
                            json=body,
                            headers=forward_headers,
                        ) as resp:
                            async for chunk in resp.content.iter_any():
                                yield chunk

                return StreamingResponse(
                    stream_generator(), media_type="text/event-stream"
                )

            else:
                # 非流式：等待完整 JSON 响应
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as session:
                    async with session.post(
                        f"{NIM_BASE}/chat/completions",
                        json=body,
                        headers=forward_headers,
                    ) as resp:
                        if resp.status == 200:
                            try:
                                return JSONResponse(content=await resp.json())
                            except Exception:
                                text = await resp.text()
                                return JSONResponse(
                                    status_code=502,
                                    content={"error": f"NIM 返回非 JSON 内容: {text[:400]}"},
                                )
                        else:
                            text = await resp.text()
                            return JSONResponse(
                                status_code=resp.status,
                                content={"error": f"NIM 后端错误 ({resp.status}): {text[:700]}"},
                            )

    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": f"代理内部错误: {str(e)}"}
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
