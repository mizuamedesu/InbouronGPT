"""InbouronGPT バックエンド。

推論プロバイダーが LLM の確率分布を恣意的に書き換えられることを、
実測値の可視化つきで見せるためのデモ API。
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .config import RuntimeConfig, config_store
from .events import ErrorEvent
from .providers import GenerationRequest, MLXProvider, OllamaProvider, OpenAICompatProvider
from .questions import QUESTIONS
from .scenarios import MODES, describe
from .shopping import PRODUCTS_BY_KEY

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="InbouronGPT", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config_store.settings.cors_origins,
    allow_origin_regex=r"http://localhost:\d+",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MLX モデルは重いのでプロセス内で使い回す
_mlx_provider = MLXProvider(config_store.settings.mlx_model)


def get_provider():
    s = config_store.settings
    if s.provider == "mlx":
        _mlx_provider.set_model_id(s.mlx_model)
        return _mlx_provider
    if s.provider == "ollama":
        return OllamaProvider(s.ollama_base_url, s.ollama_model)
    if s.provider == "openai_compat":
        return OpenAICompatProvider(s.openai_base_url, s.openai_api_key, s.openai_model)
    raise HTTPException(400, f"unknown provider: {s.provider}")


# --- メタ情報 -------------------------------------------------------------


@app.get("/api/scenarios")
def list_scenarios() -> dict:
    """2つのシナリオと、それぞれの選択肢。自由入力は受け付けない。"""
    return {"scenarios": describe()}


@app.get("/api/config")
def get_config() -> RuntimeConfig:
    return config_store.view()


class ConfigPatch(BaseModel):
    provider: str | None = None
    mlx_model: str | None = None
    ollama_base_url: str | None = None
    ollama_model: str | None = None
    openai_base_url: str | None = None
    openai_model: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None


@app.put("/api/config")
def put_config(patch: ConfigPatch) -> RuntimeConfig:
    if patch.provider and patch.provider not in ("mlx", "ollama", "openai_compat"):
        raise HTTPException(400, f"unknown provider: {patch.provider}")
    return config_store.update(patch.model_dump(exclude_none=True))


@app.get("/api/health")
async def health() -> dict:
    provider = get_provider()
    ok, message = await provider.check()
    return {
        "provider": provider.name,
        "model": provider.model_id(),
        "ok": ok,
        "message": message,
        "capabilities": provider.capabilities().model_dump(),
        "mlx_loaded": _mlx_provider.is_loaded(),
    }


# --- 生成ストリーム -------------------------------------------------------


def _sse(payload: str) -> str:
    return f"data: {payload}\n\n"


@app.get("/api/generate/stream")
async def generate_stream(
    mode: str = Query("conspiracy", description="conspiracy | shopping"),
    index: int = Query(0, ge=0, le=3, description="質問の index。自由入力は受け付けない。"),
    preset: str | None = Query(None),
    target: str | None = Query(None, description="shopping で推させる対象 (A〜D)。"),
    strength: float = Query(1.0, ge=0.0, le=3.0),
    max_tokens: int | None = Query(None, ge=1, le=1024),
    temperature: float | None = Query(None, ge=0.0, le=2.0),
    top_p: float | None = Query(None, ge=0.0, le=1.0),
    seed: int | None = Query(None),
) -> StreamingResponse:
    if mode not in MODES:
        raise HTTPException(400, f"unknown mode: {mode}")
    if mode == "conspiracy" and not any(q.index == index for q in QUESTIONS):
        raise HTTPException(404, f"unknown question index: {index}")
    if mode == "shopping" and target is not None and target not in PRODUCTS_BY_KEY:
        raise HTTPException(400, f"unknown target: {target}")

    s = config_store.settings
    req = GenerationRequest(
        mode=mode,
        question_index=index,
        target=target,
        preset_key=preset,
        strength=strength,
        max_tokens=max_tokens if max_tokens is not None else s.max_tokens,
        temperature=temperature if temperature is not None else s.temperature,
        top_p=top_p if top_p is not None else s.top_p,
        seed=seed,
    )
    provider = get_provider()

    async def gen():
        try:
            async for event in provider.stream(req):
                yield _sse(event.model_dump_json())
        except asyncio.CancelledError:  # クライアント切断
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("stream failed")
            yield _sse(ErrorEvent(message=f"{type(exc).__name__}: {exc}").model_dump_json())

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )
