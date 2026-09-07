"""InbouronGPT バックエンド。

推論プロバイダーが LLM の確率分布を恣意的に書き換えられることを、
実測値の可視化つきで見せるためのデモ API。
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import httpx
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import RuntimeConfig, config_store
from .events import ErrorEvent
from .providers import GenerationRequest, OllamaProvider, VLLMProvider, load_mlx_provider
from .scenarios import describe, resolve

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# MLX モデルは重いのでプロセス内で使い回す。
# Apple Silicon 以外では import できないので、実際に選ばれたときだけ作る。
_mlx_provider = None

# 同時リクエストは HTTP 接続を使い回す。リクエストごとにクライアントを作ると
# 接続確立の往復が積み上がり、本番の同時実行で目に見えて遅くなる。
_http: httpx.AsyncClient | None = None
_provider_cache: dict[tuple, object] = {}
# 生成の同時実行数を抑える。vLLM 側は自前でバッチングするので上限は緩め。
_slots = asyncio.Semaphore(config_store.settings.max_concurrent_requests)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _http
    s = config_store.settings
    _http = httpx.AsyncClient(
        timeout=httpx.Timeout(s.http_timeout, connect=10.0),
        limits=httpx.Limits(
            max_connections=s.max_concurrent_requests * 2,
            max_keepalive_connections=s.max_concurrent_requests,
        ),
    )
    try:
        yield
    finally:
        await _http.aclose()
        _http = None


app = FastAPI(title="InbouronGPT", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config_store.settings.cors_origins,
    allow_origin_regex=r"http://localhost:\d+",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_provider():
    s = config_store.settings
    if s.provider == "mlx":
        global _mlx_provider
        if _mlx_provider is None:
            try:
                _mlx_provider = load_mlx_provider(s.mlx_model)
            except RuntimeError as exc:
                raise HTTPException(503, str(exc)) from exc
        _mlx_provider.set_model_id(s.mlx_model)
        return _mlx_provider

    assert _http is not None, "HTTP client not initialised"
    if s.provider == "ollama":
        key = ("ollama", s.ollama_base_url, s.ollama_model)
        if key not in _provider_cache:
            _provider_cache[key] = OllamaProvider(s.ollama_base_url, s.ollama_model, _http)
        return _provider_cache[key]
    if s.provider == "vllm":
        key = ("vllm", s.vllm_base_url, s.vllm_model)
        if key not in _provider_cache:
            _provider_cache[key] = VLLMProvider(
                s.vllm_base_url, s.vllm_api_key, s.vllm_model, _http, stop=s.vllm_stop
            )
        return _provider_cache[key]
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
    vllm_base_url: str | None = None
    vllm_model: str | None = None
    strength: float | None = None
    repetition_penalty: float | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None


@app.put("/api/config")
def put_config(patch: ConfigPatch) -> RuntimeConfig:
    """設定を変えたら、古い接続先のプロバイダーは捨てる。"""
    if patch.provider and patch.provider not in ("mlx", "ollama", "vllm"):
        raise HTTPException(400, f"unknown provider: {patch.provider}")
    updated = config_store.update(patch.model_dump(exclude_none=True))
    _provider_cache.clear()
    return updated


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
        "mlx_loaded": _mlx_provider is not None and _mlx_provider.is_loaded(),
    }


# --- 生成ストリーム -------------------------------------------------------


# 生成エンドポイントが受け付けるクエリ。これ以外は拒否する。
ALLOWED_QUERY_PARAMS = frozenset({"scenario", "index", "variant"})


def _sse(payload: str) -> str:
    return f"data: {payload}\n\n"


@app.get("/api/generate/stream")
async def generate_stream(
    request: Request,
    scenario: int = Query(0, ge=0, description="0=陰謀論 / 1=ショッピング"),
    index: int = Query(0, ge=0, description="シナリオ内の選択肢番号"),
    variant: int = Query(1, ge=0, le=1, description="0=素の分布 / 1=確率分布を曲げる"),
) -> StreamingResponse:
    """クライアントが送れるのは番号 3 つだけ。

    強度・温度・生成長といった生成条件はすべてサーバー側の設定から埋める。
    自由入力はもちろん、文字列パラメータも受け付けない。
    """
    # 想定外のパラメータは黙って無視せず、はっきり弾く。
    # 「番号しか受け付けない」ことを、挙動としても示しておく。
    unknown = set(request.query_params) - ALLOWED_QUERY_PARAMS
    if unknown:
        raise HTTPException(400, f"unsupported parameters: {', '.join(sorted(unknown))}")

    try:
        resolve(scenario, index, variant)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc

    s = config_store.settings
    req = GenerationRequest(
        scenario=scenario,
        index=index,
        variant=variant,
        strength=s.strength,
        repetition_penalty=s.repetition_penalty,
        max_tokens=s.max_tokens,
        temperature=s.temperature,
        top_p=s.top_p,
    )
    provider = get_provider()

    async def gen():
        try:
            # 上限を超えた分はここで待たせる。無制限に受けると
            # 生成が全部遅くなり、どのクライアントも結果を得られなくなる。
            async with _slots:
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


# --- ビルド済みフロントの配信 ---------------------------------------------
#
# 本番では API と同じオリジンから配る。別ポートに分けると CORS とプロキシの
# 設定が増えるだけで、得るものがない。dist が無ければ何もしない（開発時は
# Vite の dev server を使う）。

_DIST = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"

if _DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    async def spa(path: str) -> FileResponse:
        """静的ファイルがあればそれを、無ければ index.html を返す。"""
        candidate = (_DIST / path).resolve()
        if path and candidate.is_file() and candidate.is_relative_to(_DIST):
            return FileResponse(candidate)
        return FileResponse(_DIST / "index.html")

    logger.info("serving frontend from %s", _DIST)
else:
    logger.info("frontend dist not found at %s (dev では vite を使う)", _DIST)
