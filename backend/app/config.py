"""実行時に差し替え可能な設定。

推論プロバイダーとそのエンドポイントは、環境変数でも API でも変更できる。
"""

from __future__ import annotations

import threading
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["mlx", "ollama", "vllm"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INBOURON_", env_file=".env", extra="ignore")

    provider: ProviderName = "mlx"

    # MLX（ローカル、語彙全体の logit を実測しながら操作できる）
    # LFM2.5-1.2B-JP の abliterated 版。日本語特化・1.2B なので M シリーズで
    # 40〜50 tok/s 出て、リアルタイム可視化のデモに向く。
    mlx_model: str = "PinoCookie/LFM2.5-1.2B-JP-Abliterated"

    # Ollama（エンドポイント差し替え可能）
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "huihui_ai/gemma-4-abliterated:31b"

    # vLLM（本番。DGX Spark 上で OpenAI 互換 API を提供する）
    vllm_base_url: str = "http://localhost:8001/v1"
    vllm_api_key: str = "dummy"
    vllm_model: str = "PinoCookie/LFM2.5-1.2B-JP-Abliterated"

    # 同時に処理する生成の上限。vLLM は自前でバッチングするので余裕を持たせ、
    # MLX はモデルが 1 つしか載らないので実質 1 本ずつになる。
    max_concurrent_requests: int = 32
    http_timeout: float = 600.0

    # 生成条件はすべてサーバーが持つ。クライアントからは変更できない。
    strength: float = 1.0
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95

    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])


class RuntimeConfig(BaseModel):
    """API で書き換えられる部分だけを切り出したビュー。"""

    provider: ProviderName
    mlx_model: str
    ollama_base_url: str
    ollama_model: str
    vllm_base_url: str
    vllm_model: str
    strength: float
    max_tokens: int
    temperature: float
    top_p: float


class ConfigStore:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.Lock()

    @property
    def settings(self) -> Settings:
        return self._settings

    def view(self) -> RuntimeConfig:
        s = self._settings
        return RuntimeConfig(
            provider=s.provider,
            mlx_model=s.mlx_model,
            ollama_base_url=s.ollama_base_url,
            ollama_model=s.ollama_model,
            vllm_base_url=s.vllm_base_url,
            vllm_model=s.vllm_model,
            strength=s.strength,
            max_tokens=s.max_tokens,
            temperature=s.temperature,
            top_p=s.top_p,
        )

    def update(self, patch: dict) -> RuntimeConfig:
        with self._lock:
            for key, value in patch.items():
                if value is not None and hasattr(self._settings, key):
                    setattr(self._settings, key, value)
        return self.view()


settings = Settings()
config_store = ConfigStore(settings)
