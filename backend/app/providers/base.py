"""推論プロバイダーの共通インターフェース。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from pydantic import BaseModel

from ..events import Capabilities


class GenerationRequest(BaseModel):
    mode: str = "conspiracy"
    question_index: int = 0
    preset_key: str | None = None
    # shopping モードで推させる対象 ("A"〜"D")。None なら対照群。
    target: str | None = None
    strength: float = 1.0
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    repetition_penalty: float = 1.15
    seed: int | None = None


class Provider(Protocol):
    name: str

    def capabilities(self) -> Capabilities: ...

    def model_id(self) -> str: ...

    async def check(self) -> tuple[bool, str]:
        """疎通確認。(ok, メッセージ)"""
        ...

    def stream(self, req: GenerationRequest) -> AsyncIterator[BaseModel]: ...
