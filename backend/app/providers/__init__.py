"""推論プロバイダー。

MLX は Apple Silicon 専用なので、import は使うときまで遅らせる。
本番（DGX Spark）には mlx-lm が入らないため、モジュール読み込み時点で
参照すると起動できなくなる。
"""

from typing import TYPE_CHECKING

from .base import GenerationRequest, Provider
from .ollama_provider import OllamaProvider
from .vllm_provider import VLLMProvider

if TYPE_CHECKING:  # 型検査時だけ実体を見せる
    from .mlx_provider import MLXProvider


def load_mlx_provider(model_id: str):
    """MLX プロバイダーを遅延生成する。使えない環境では理由を添えて弾く。"""
    try:
        from .mlx_provider import MLXProvider
    except ImportError as exc:  # noqa: BLE001
        raise RuntimeError(
            "MLX provider is unavailable on this platform "
            f"(mlx-lm not installed: {exc}). "
            "本番では INBOURON_PROVIDER=vllm を使ってください。"
        ) from exc
    return MLXProvider(model_id)


__all__ = [
    "GenerationRequest",
    "OllamaProvider",
    "Provider",
    "VLLMProvider",
    "load_mlx_provider",
]
