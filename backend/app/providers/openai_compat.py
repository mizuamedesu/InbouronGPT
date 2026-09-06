"""OpenAI 互換エンドポイント（vLLM / LM Studio / TGI など）。

サーバーが `logprobs` を返すなら、素の確率分布を top-k の範囲で実測できる。
さらに 1 トークンずつ生成する経路を使えば、返ってきた top-k の対数確率に
バイアスを足して選び直すことで、top-k に限定した「本物の」logit 操作ができる。

ただし見えるのはあくまで top-k までで、MLX のように語彙全体を握れるわけではない。
その限界は capability と note で明示し、UI 側にもそのまま伝える。

logprobs を返さないサーバー（例: Ollama の OpenAI 互換層）に向けた場合は
自動的に生成のみへ降格する。
"""

from __future__ import annotations

import math
import time
from collections.abc import AsyncIterator

import httpx
from pydantic import BaseModel

from ..events import (
    AppliedDelta,
    Capabilities,
    ChosenToken,
    DoneEvent,
    ErrorEvent,
    MetaEvent,
    StepEvent,
    TokenProb,
)
from ..presets import BiasPreset, get_preset
from ..scenarios import resolve
from .base import GenerationRequest

TOP_K = 8


class OpenAICompatProvider:
    name = "openai_compat"

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self._logprobs_ok: bool | None = None

    def model_id(self) -> str:
        return self.model

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def capabilities(self) -> Capabilities:
        if self._logprobs_ok:
            return Capabilities(
                generation=True,
                logit_inspection=True,
                logit_injection=True,
                note=(
                    "サーバーが返す top-{k} の対数確率にバイアスを加えて選び直しています。"
                    "語彙全体ではなく top-{k} の範囲に限定された操作です。"
                ).format(k=TOP_K),
            )
        return Capabilities(
            generation=True,
            logit_inspection=False,
            logit_injection=False,
            note=(
                "このエンドポイントは logprobs を返さないため、"
                "確率分布の可視化と操作はできません。生成のみ行います。"
            ),
        )

    async def _probe(self) -> bool:
        """logprobs を返すサーバーかどうかを 1 トークンだけ投げて確かめる。"""
        if self._logprobs_ok is not None:
            return self._logprobs_ok
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(
                    f"{self.base_url}/completions",
                    headers=self._headers(),
                    json={
                        "model": self.model,
                        "prompt": "1",
                        "max_tokens": 1,
                        "logprobs": TOP_K,
                        "temperature": 0.0,
                    },
                )
                r.raise_for_status()
                choice = r.json()["choices"][0]
                self._logprobs_ok = bool((choice.get("logprobs") or {}).get("top_logprobs"))
        except Exception:  # noqa: BLE001
            self._logprobs_ok = False
        return self._logprobs_ok

    async def check(self) -> tuple[bool, str]:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"{self.base_url}/models", headers=self._headers())
                r.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            return False, f"{self.base_url} に接続できない: {type(exc).__name__}: {exc}"
        ok = await self._probe()
        suffix = "logprobs 対応" if ok else "logprobs 非対応（生成のみ）"
        return True, f"{self.base_url} — {suffix}"

    # --- バイアス計算 -----------------------------------------------------

    @staticmethod
    def _bias_for(token_text: str, preset: BiasPreset, strength: float) -> tuple[float, str | None]:
        """top-k のトークン文字列にかけるバイアス量を返す。

        語彙全体を持たないので、MLX 版のような trie は使えない。
        代わりにトークン文字列とフレーズの部分一致で判定する。
        """
        stripped = token_text.strip()
        if not stripped:
            return 0.0, None
        for phrase in preset.boost_phrases:
            if stripped in phrase or phrase.startswith(stripped):
                return preset.boost_factor * strength, phrase
        for phrase in preset.suppress_phrases:
            if stripped in phrase or phrase.startswith(stripped):
                return preset.suppress_factor * strength, phrase
        return 0.0, None

    # --- 生成 -------------------------------------------------------------

    async def stream(self, req: GenerationRequest) -> AsyncIterator[BaseModel]:
        r = resolve(req.mode, req.question_index, req.preset_key, req.target)
        preset = r.preset
        can_bend = await self._probe()

        yield MetaEvent(
            provider=self.name,
            model=self.model,
            capabilities=self.capabilities(),
            question_index=req.question_index,
            question=r.question_text,
            preset_key=preset.key,
            preset_name=preset.name,
            preset_description=preset.description,
            system_prompt=r.system_prompt,
            user_text=r.user_text,
            processors=["TopKPhraseBias"] if can_bend else [],
            boost_phrases=list(preset.boost_phrases),
            suppress_phrases=list(preset.suppress_phrases),
            strength=req.strength,
        )

        if can_bend:
            async for ev in self._stream_bent(req, preset, r):
                yield ev
        else:
            async for ev in self._stream_plain(req, preset, r):
                yield ev

    async def _stream_plain(
        self, req, preset, r
    ) -> AsyncIterator[BaseModel]:
        """logprobs が無いサーバー向け。チャット API でそのまま流す。"""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": preset.system_prompt},
                {"role": "user", "content": r.user_text},
            ],
            "stream": True,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "top_p": req.top_p,
        }
        pieces: list[str] = []
        i = 0
        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST", f"{self.base_url}/chat/completions", headers=self._headers(), json=payload
                ) as r:
                    r.raise_for_status()
                    async for line in r.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        body = line[6:].strip()
                        if body == "[DONE]":
                            break
                        import json as _json

                        delta = _json.loads(body)["choices"][0].get("delta", {})
                        piece = delta.get("content") or ""
                        if piece:
                            pieces.append(piece)
                            yield StepEvent(i=i, text=piece, chosen=ChosenToken(id=-1, text=piece))
                            i += 1
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

    async def _stream_bent(
        self, req, preset, r
    ) -> AsyncIterator[BaseModel]:
        """1 トークンずつ生成し、返ってきた top-k を再ランクして選び直す。"""
        prompt = f"{r.system_prompt}\n\n{r.user_text}\n回答: "
        pieces: list[str] = []
        kls: list[float] = []
        flipped = 0
        t0 = time.perf_counter()
        i = 0

        async with httpx.AsyncClient(timeout=None) as client:
            while i < req.max_tokens:
                try:
                    r = await client.post(
                        f"{self.base_url}/completions",
                        headers=self._headers(),
                        json={
                            "model": self.model,
                            "prompt": prompt + "".join(pieces),
                            "max_tokens": 1,
                            "logprobs": TOP_K,
                            "temperature": req.temperature,
                            "top_p": req.top_p,
                        },
                    )
                    r.raise_for_status()
                    choice = r.json()["choices"][0]
                except Exception as exc:  # noqa: BLE001
                    yield ErrorEvent(message=f"{type(exc).__name__}: {exc}")
                    return

                lp = (choice.get("logprobs") or {}).get("top_logprobs") or []
                if not lp:
                    break
                cand = lp[0]  # {token_text: logprob}

                base_items = sorted(cand.items(), key=lambda kv: kv[1], reverse=True)
                base_probs = _renorm([v for _, v in base_items])

                bent_logits: list[float] = []
                phrase_hits: list[str | None] = []
                for text, logprob in base_items:
                    delta, phrase = self._bias_for(text, preset, req.strength)
                    bent_logits.append(logprob + delta)
                    phrase_hits.append(phrase)
                bent_probs = _renorm(bent_logits)

                pick = max(range(len(base_items)), key=lambda j: bent_probs[j])
                token_text = base_items[pick][0]
                delta = bent_logits[pick] - base_items[pick][1]

                if base_items[0][0] != base_items[pick][0]:
                    flipped += 1
                kl = sum(
                    p * (math.log(max(p, 1e-12)) - math.log(max(q, 1e-12)))
                    for p, q in zip(bent_probs, base_probs)
                )
                kls.append(kl)

                yield StepEvent(
                    i=i,
                    text=token_text,
                    chosen=ChosenToken(
                        id=-1,
                        text=token_text,
                        p_base=base_probs[pick],
                        p_bent=bent_probs[pick],
                        rank_base=pick + 1,
                        rank_bent=1,
                    ),
                    base_top=[
                        TokenProb(id=-1, text=t, p=p) for (t, _), p in zip(base_items, base_probs)
                    ],
                    bent_top=sorted(
                        [
                            TokenProb(id=-1, text=t, p=p)
                            for (t, _), p in zip(base_items, bent_probs)
                        ],
                        key=lambda tp: tp.p,
                        reverse=True,
                    ),
                    applied=(
                        [AppliedDelta(name="TopKPhraseBias", delta=round(delta, 4))]
                        if abs(delta) > 1e-6
                        else []
                    ),
                    kl=kl,
                    targeted_phrase=phrase_hits[pick],
                )

                pieces.append(token_text)
                i += 1
                if choice.get("finish_reason") == "stop":
                    break

        elapsed = time.perf_counter() - t0
        yield DoneEvent(
            tokens=i,
            elapsed=round(elapsed, 3),
            tps=round(i / elapsed, 2) if elapsed > 0 else 0.0,
            mean_kl=round(sum(kls) / len(kls), 4) if kls else 0.0,
            max_kl=round(max(kls), 4) if kls else 0.0,
            flipped=flipped,
            text="".join(pieces),
        )


def _renorm(logits: list[float]) -> list[float]:
    """対数値の並びを softmax で確率に直す（top-k の範囲で正規化）。"""
    if not logits:
        return []
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    total = sum(exps)
    return [e / total for e in exps]
