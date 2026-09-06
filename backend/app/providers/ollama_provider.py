"""Ollama プロバイダー（エンドポイント差し替え可能）。

重要な制約:
Ollama は 1 トークンごとの logits を返さないし、受け取りもしない。
`logprobs` フィールドは受理されるが応答には含まれず、`options` にも
logit_bias 相当は存在しない（Ollama 0.20.0 で実測）。

したがってこのプロバイダーでは「確率分布を曲げる」実演はできず、
できるのはプロンプト注入だけになる。可視化パネルは capability を見て
無効化される。実測でない数値をそれらしく描くことはしない —
このアプリの主旨に反するため。
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator

import httpx
from pydantic import BaseModel

from ..events import Capabilities, ChosenToken, DoneEvent, ErrorEvent, MetaEvent, StepEvent
from ..presets import get_preset
from ..scenarios import resolve
from .base import GenerationRequest


class OllamaProvider:
    name = "ollama"

    def __init__(self, base_url: str, model: str, client: httpx.AsyncClient) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = client

    def model_id(self) -> str:
        return self.model

    def capabilities(self) -> Capabilities:
        return Capabilities(
            generation=True,
            logit_inspection=False,
            logit_injection=False,
            note=(
                "Ollama は 1 トークンごとの logits を返しも受け取りもしないため、"
                "このプロバイダーでは確率分布の操作と可視化はできません。"
                "実演されるのはプロンプト注入のみです。"
            ),
        )

    async def check(self) -> tuple[bool, str]:
        try:
            r = await self._client.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
            names = [m["name"] for m in r.json().get("models", [])]
            if self.model in names:
                return True, f"{self.base_url} — model available"
            return False, f"{self.base_url} に接続できたが '{self.model}' が見つからない (利用可能: {', '.join(names[:5])})"
        except Exception as exc:  # noqa: BLE001
            return False, f"{self.base_url} に接続できない: {type(exc).__name__}: {exc}"

    async def stream(self, req: GenerationRequest) -> AsyncIterator[BaseModel]:
        r = resolve(req.scenario, req.index, req.variant)
        preset = r.preset

        yield MetaEvent(
            provider=self.name,
            model=self.model,
            capabilities=self.capabilities(),
            question_index=req.index,
            question=r.question_text,
            preset_key=preset.key,
            preset_name=preset.name,
            preset_description=preset.description,
            system_prompt=r.system_prompt,
            user_text=r.user_text,
            processors=[],
            boost_phrases=list(preset.boost_phrases),
            suppress_phrases=list(preset.suppress_phrases),
            strength=req.strength,
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": preset.system_prompt},
                {"role": "user", "content": r.user_text},
            ],
            "stream": True,
            "think": False,
            "options": {
                "num_predict": req.max_tokens,
                "temperature": req.temperature,
                "top_p": req.top_p,
                **({"seed": req.seed} if req.seed is not None else {}),
            },
        }

        pieces: list[str] = []
        i = 0
        t0 = time.perf_counter()
        try:
            async with self._client.stream(
                "POST", f"{self.base_url}/api/chat", json=payload, timeout=None
            ) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    if chunk.get("error"):
                        yield ErrorEvent(message=str(chunk["error"]))
                        return
                    piece = chunk.get("message", {}).get("content", "")
                    if piece:
                        pieces.append(piece)
                        yield StepEvent(i=i, text=piece, chosen=ChosenToken(id=-1, text=piece))
                        i += 1
                    if chunk.get("done"):
                        break
        except Exception as exc:  # noqa: BLE001
            yield ErrorEvent(message=f"{type(exc).__name__}: {exc}")
            return

        elapsed = time.perf_counter() - t0
        yield DoneEvent(
            tokens=i,
            elapsed=round(elapsed, 3),
            tps=round(i / elapsed, 2) if elapsed > 0 else 0.0,
            text="".join(pieces),
        )
