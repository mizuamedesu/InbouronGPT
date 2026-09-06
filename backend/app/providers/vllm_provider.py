"""vLLM プロバイダー（本番: DGX Spark / GB10）。

logit 操作はエンジン側の `vllm_plugin.inbouron_logits` が行う。
このクラスは条件を `vllm_xargs` で渡し、返ってきたストリームを可視化用の
イベントに直すだけ。生成そのものは vLLM の連続バッチングに任せるので、
同時リクエストはエンジン側でまとめて処理される。

可視化について:
vLLM V1 が返す `logprobs` は logits processor を通す**前**の値なので、
これがそのまま「素の分布」の実測値になる。「曲げた分布」は、返ってきた
top-k に対してこちらが要求したバイアスを足し直して再構成している
（語彙全体ではなく top-k の範囲）。操作そのものは語彙全体に効いている。
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import AsyncIterator

import httpx
from pydantic import BaseModel

from ..events import (
    Adjustment,
    AppliedDelta,
    Capabilities,
    ChosenToken,
    DoneEvent,
    ErrorEvent,
    MetaEvent,
    StepEvent,
    TokenProb,
)
from ..presets import DENIAL_PHRASES, BiasPreset
from ..scenarios import resolve
from .base import GenerationRequest

logger = logging.getLogger(__name__)

TOP_K = 12


class VLLMProvider:
    name = "vllm"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        client: httpx.AsyncClient,
        plugin_enabled: bool | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self._client = client
        # プラグインが入っているか。None は未確認。
        self._plugin_enabled = plugin_enabled

    def model_id(self) -> str:
        return self.model

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def capabilities(self) -> Capabilities:
        if self._plugin_enabled is False:
            return Capabilities(
                generation=True,
                logit_inspection=True,
                logit_injection=False,
                note=(
                    "サーバーに inbouron_logits プラグインが読み込まれていません。"
                    "確率分布は観測できますが、書き換えはできません。"
                    "vllm serve に --logits-processors を付けて起動してください。"
                ),
            )
        return Capabilities(
            generation=True,
            logit_inspection=True,
            logit_injection=True,
            note=(
                f"操作は語彙全体に効いています。可視化は返却された top-{TOP_K} の"
                "範囲で、素の分布は実測値、曲げた分布はそこに適用済みのバイアスを"
                "足し直したものです。"
            ),
        )

    async def check(self) -> tuple[bool, str]:
        try:
            r = await self._client.get(f"{self.base_url}/models", headers=self._headers())
            r.raise_for_status()
            names = [m["id"] for m in r.json().get("data", [])]
        except Exception as exc:  # noqa: BLE001
            self._plugin_enabled = None
            return False, f"{self.base_url} に接続できない: {type(exc).__name__}: {exc}"

        if self.model not in names:
            return False, f"接続できたが '{self.model}' が無い（利用可能: {', '.join(names[:4])}）"

        ok = await self._probe_plugin()
        return True, f"{self.base_url} — " + (
            "プラグイン有効" if ok else "プラグイン未検出（生成のみ）"
        )

    async def _probe_plugin(self) -> bool:
        """1 トークンだけ投げて、プラグインが設定を受け付けるか確かめる。

        未登録なら vLLM は未知の `vllm_xargs` を弾くので、そこで判別できる。
        """
        if self._plugin_enabled is not None:
            return self._plugin_enabled
        try:
            r = await self._client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": "1"}],
                    "max_tokens": 1,
                    "vllm_xargs": {"inbouron": {"mode": "conspiracy", "index": 0}},
                },
                timeout=60,
            )
            self._plugin_enabled = r.status_code == 200
        except Exception:  # noqa: BLE001
            self._plugin_enabled = False
        return self._plugin_enabled

    # --- 生成 -------------------------------------------------------------

    async def stream(self, req: GenerationRequest) -> AsyncIterator[BaseModel]:
        r = resolve(req.mode, req.question_index, req.preset_key, req.target)
        preset = r.preset
        bend = bool(_bias_table(preset, req.strength)) and await self._probe_plugin()

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
            processors=_processor_names(preset) if bend else [],
            boost_phrases=list(preset.boost_phrases),
            suppress_phrases=list(preset.suppress_phrases),
            strength=req.strength,
            adjustments=_describe(preset, req.strength) if bend else [],
            seed=req.seed,
        )

        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": r.system_prompt},
                {"role": "user", "content": r.user_text},
            ],
            "stream": True,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "top_p": req.top_p,
            "logprobs": True,
            "top_logprobs": TOP_K,
        }
        if req.seed is not None:
            payload["seed"] = req.seed
        if bend:
            payload["vllm_xargs"] = {
                "inbouron": {
                    "mode": req.mode,
                    "index": req.question_index,
                    "preset": req.preset_key,
                    "target": req.target,
                    "strength": req.strength,
                }
            }

        table = _bias_table(preset, req.strength) if bend else {}
        pieces: list[str] = []
        kls: list[float] = []
        flipped = 0
        i = 0
        t0 = time.perf_counter()

        try:
            async with self._client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
                timeout=None,
            ) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode()[:400]
                    yield ErrorEvent(message=f"vLLM {resp.status_code}: {body}")
                    return

                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    body = line[6:].strip()
                    if body == "[DONE]":
                        break
                    chunk = json.loads(body)
                    choice = chunk["choices"][0]
                    piece = (choice.get("delta") or {}).get("content") or ""
                    entries = ((choice.get("logprobs") or {}).get("content")) or []

                    if not piece and not entries:
                        continue

                    for entry in entries:
                        ev = _build_step(i, entry, table)
                        kls.append(ev.kl)
                        if ev.base_top and ev.bent_top and ev.base_top[0].id != ev.bent_top[0].id:
                            flipped += 1
                        pieces.append(ev.text)
                        yield ev
                        i += 1

                    if piece and not entries:
                        pieces.append(piece)
                        yield StepEvent(i=i, text=piece, chosen=ChosenToken(id=-1, text=piece))
                        i += 1
        except Exception as exc:  # noqa: BLE001
            logger.exception("vllm stream failed")
            yield ErrorEvent(message=f"{type(exc).__name__}: {exc}")
            return

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


# --- バイアスの再構成 -------------------------------------------------------


def _bias_table(preset: BiasPreset, strength: float) -> dict[str, tuple[float, str]]:
    """トークン文字列 -> (加算量, 由来フレーズ) の対応表。

    プラグイン側は語彙全体を trie で追うが、こちらは返ってきた top-k の
    文字列しか持たない。可視化のために近い値を再構成するための表。
    """
    table: dict[str, tuple[float, str]] = {}

    def put(phrases, factor: float) -> None:
        if not factor:
            return
        for phrase in phrases:
            for n in range(1, len(phrase) + 1):
                frag = phrase[:n]
                cur = table.get(frag)
                if cur is None or abs(factor) > abs(cur[0]):
                    table[frag] = (factor, phrase)

    put(preset.boost_phrases, preset.boost_factor * strength)
    put(preset.suppress_phrases, preset.suppress_factor * strength)
    put(DENIAL_PHRASES, preset.denial_factor * strength)
    return table


def _lookup(token_text: str, table: dict[str, tuple[float, str]]) -> tuple[float, str | None]:
    stripped = token_text.strip()
    if not stripped:
        return 0.0, None
    hit = table.get(stripped)
    return (hit[0], hit[1]) if hit else (0.0, None)


def _build_step(i: int, entry: dict, table: dict[str, tuple[float, str]]) -> StepEvent:
    chosen_text = entry.get("token", "")
    tops = entry.get("top_logprobs") or []

    items = [(t.get("token", ""), float(t.get("logprob", -100.0))) for t in tops]
    if chosen_text not in [t for t, _ in items]:
        items.append((chosen_text, float(entry.get("logprob", -100.0))))
    items.sort(key=lambda kv: kv[1], reverse=True)

    base_probs = _softmax([lp for _, lp in items])
    bent_logits: list[float] = []
    hits: list[str | None] = []
    for text, lp in items:
        delta, phrase = _lookup(text, table)
        bent_logits.append(lp + delta)
        hits.append(phrase)
    bent_probs = _softmax(bent_logits)

    pick = next((n for n, (t, _) in enumerate(items) if t == chosen_text), 0)
    delta = bent_logits[pick] - items[pick][1]

    order = sorted(range(len(items)), key=lambda n: bent_probs[n], reverse=True)
    rank_bent = order.index(pick) + 1

    kl = sum(
        p * (math.log(max(p, 1e-12)) - math.log(max(q, 1e-12)))
        for p, q in zip(bent_probs, base_probs)
    )

    return StepEvent(
        i=i,
        text=chosen_text,
        chosen=ChosenToken(
            id=-1,
            text=chosen_text,
            p_base=base_probs[pick],
            p_bent=bent_probs[pick],
            rank_base=pick + 1,
            rank_bent=rank_bent,
        ),
        base_top=[TokenProb(id=n, text=items[n][0], p=base_probs[n]) for n in range(len(items))],
        bent_top=[TokenProb(id=n, text=items[n][0], p=bent_probs[n]) for n in order],
        applied=(
            [AppliedDelta(name="InbouronLogits", delta=round(delta, 4))]
            if abs(delta) > 1e-6
            else []
        ),
        kl=max(0.0, kl),
        targeted_phrase=hits[pick],
    )


def _softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


def _processor_names(preset: BiasPreset) -> list[str]:
    names = []
    if preset.boost_phrases and preset.boost_factor:
        names.append("PhraseBoost")
    if preset.suppress_phrases and preset.suppress_factor:
        names.append("PhraseSuppress")
    if preset.denial_factor:
        names.append("DenialSuppress")
    if preset.cite_boost_factor:
        names.append("CiteFromPrompt")
    if preset.length_boost_factor:
        names.append("GenLength")
    if preset.trigger_phrase and preset.trigger_token_phrase:
        names.append("TriggerPhrase")
    return names


def _describe(preset: BiasPreset, strength: float) -> list[Adjustment]:
    out: list[Adjustment] = []
    if preset.boost_phrases and preset.boost_factor:
        out.append(Adjustment(processor="PhraseBoost", label="押し上げる語",
                              factor=round(preset.boost_factor * strength, 3),
                              phrases=list(preset.boost_phrases)))
    if preset.suppress_phrases and preset.suppress_factor:
        out.append(Adjustment(processor="PhraseSuppress", label="押し下げる語",
                              factor=round(preset.suppress_factor * strength, 3),
                              phrases=list(preset.suppress_phrases)))
    if preset.denial_factor:
        out.append(Adjustment(processor="DenialSuppress", label="否定表現の抑制",
                              factor=round(preset.denial_factor * strength, 3),
                              phrases=list(DENIAL_PHRASES)))
    if preset.cite_boost_factor:
        out.append(Adjustment(processor="CiteFromPrompt", label="プロンプト語彙の再利用",
                              factor=round(preset.cite_boost_factor * strength, 3),
                              note="直前語に続くプロンプト内の語には 0.4 倍"))
    if preset.length_boost_factor:
        out.append(Adjustment(processor="GenLength", label="EOS の抑制",
                              factor=round(preset.length_boost_factor * strength, 3),
                              note="生成が伸びるほど強く効く (p=2)"))
    if preset.trigger_phrase and preset.trigger_token_phrase:
        out.append(Adjustment(processor="TriggerPhrase", label="決め台詞の強制", factor=0.0,
                              note="他のトークンを最小値まで潰して強制する"))
    return out
