"""logits processor 群。

`trie` は MLX にも torch にも依存しない中核で、本番の vLLM プラグインからも
使う。一方 MLX 実装は Apple Silicon でしか import できないため、
**パッケージ読み込み時点では触らない**。ここで mlx を引き込むと、
CUDA 機で `app.processors.trie` を import しただけで落ちる。
"""

from typing import TYPE_CHECKING

from .trie import PhraseBiasState, PhraseTrie, encode_phrase_variants

if TYPE_CHECKING:  # 型検査時だけ実体を見せる
    from .base import LogitTap, MLXLogitsProcessor, TapStore, add_bias, enforce_tokens
    from .cite_from_prompt import CiteFromPromptLogitsProcessor
    from .gen_length import GenLengthLogitsProcessor
    from .phrase_bias import PhraseBiasLogitsProcessor
    from .trigger_phrase import TriggerPhraseLogitsProcessor

# 属性名 -> それを定義しているモジュール
_MLX_EXPORTS = {
    "LogitTap": ".base",
    "MLXLogitsProcessor": ".base",
    "TapStore": ".base",
    "add_bias": ".base",
    "enforce_tokens": ".base",
    "CiteFromPromptLogitsProcessor": ".cite_from_prompt",
    "GenLengthLogitsProcessor": ".gen_length",
    "PhraseBiasLogitsProcessor": ".phrase_bias",
    "TriggerPhraseLogitsProcessor": ".trigger_phrase",
}


def __getattr__(name: str):
    """MLX 実装は実際に使われたときだけ読み込む（PEP 562）。"""
    module = _MLX_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module, __name__), name)


__all__ = [
    "CiteFromPromptLogitsProcessor",
    "GenLengthLogitsProcessor",
    "LogitTap",
    "MLXLogitsProcessor",
    "PhraseBiasLogitsProcessor",
    "PhraseBiasState",
    "PhraseTrie",
    "TapStore",
    "TriggerPhraseLogitsProcessor",
    "add_bias",
    "encode_phrase_variants",
    "enforce_tokens",
]
