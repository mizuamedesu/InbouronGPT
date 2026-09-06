"""InbouronGPT の logit 操作を vLLM V1 の中で動かすプラグイン。

本番（DGX Spark / GB10）では MLX が使えないため、操作はエンジン側で行う。
判断そのものは `app.processors.trie` を共有しているので、開発機の MLX 経路と
本番の vLLM 経路で「どのトークンをいくつ押すか」は完全に同じ。
違うのは加算をどのテンソルライブラリで行うかだけ。

起動:
    vllm serve <model> \
        --logits-processors inbouron_logits:InbouronLogitsProcessor

リクエスト側は `vllm_xargs` で 1 件ずつ条件を渡す:
    {"vllm_xargs": {"inbouron": {"mode": "shopping", "target": "B", "strength": 1.0}}}

`vllm_xargs` を付けないリクエストには一切手を出さない。素の分布のまま通す。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessor

from app.presets import DENIAL_PHRASES, BiasPreset
from app.processors.trie import PhraseBiasState, PhraseTrie, encode_phrase_variants
from app.scenarios import resolve

logger = logging.getLogger(__name__)

ARG_KEY = "inbouron"


@dataclass
class _Phrase:
    """フレーズバイアス 1 段ぶん。"""

    name: str
    state: PhraseBiasState


@dataclass
class _Request:
    """バッチ内 1 リクエストぶんの状態。"""

    # 生成済みトークンへのライブ参照。vLLM が生成のたびに追記する。
    output_tok_ids: list[int]
    prompt_tok_ids: list[int]
    phrases: list[_Phrase] = field(default_factory=list)
    # CiteFromPrompt
    cite_factor: float = 0.0
    cite_tokens: tuple[int, ...] = ()
    cite_followers: dict[int, tuple[int, ...]] = field(default_factory=dict)
    cite_conditional: float = 0.0
    # GenLength
    length_factor: float = 0.0
    length_p: int = 2
    eos_ids: tuple[int, ...] = ()
    # TriggerPhrase
    trigger_token: int | None = None
    trigger_phrase_tokens: tuple[int, ...] = ()
    trigger_remaining: int = 0
    trigger_after: bool = True
    trigger_pos: int = -1
    # 決め台詞は引き金トークン（「。」）で終わることが多い。言い切った直後に
    # 自分の末尾で再発火して無限ループするので、次に構えるまで間を空ける。
    trigger_ready_at: int = 0

    def reset_phrases(self) -> None:
        for p in self.phrases:
            p.state.reset()


class InbouronLogitsProcessor(LogitsProcessor):
    """語彙全体に対して加算バイアスをかける、状態つき logits processor。"""

    def __init__(self, vllm_config: Any, device: torch.device, is_pin_memory: bool) -> None:
        self.device = device
        self.pin_memory = is_pin_memory
        self._tokenizer = _load_tokenizer(vllm_config)
        # (mode, key, strength) -> 構築済み trie 群。リクエストごとに作り直すと
        # 数万トークンの符号化が毎回走るのでキャッシュする。
        self._trie_cache: dict[tuple, list[tuple[str, PhraseTrie]]] = {}
        self._preset_cache: dict[tuple, BiasPreset] = {}
        self._reqs: dict[int, _Request] = {}

    # --- vLLM が求めるインターフェース -----------------------------------

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams) -> None:
        cfg = (sampling_params.extra_args or {}).get(ARG_KEY)
        if cfg is None:
            return
        if not isinstance(cfg, dict):
            raise ValueError(f"{ARG_KEY} must be an object")
        mode = cfg.get("mode", "conspiracy")
        if mode not in ("conspiracy", "shopping"):
            raise ValueError(f"unknown mode: {mode}")
        strength = cfg.get("strength", 1.0)
        if not isinstance(strength, (int, float)) or not 0.0 <= float(strength) <= 3.0:
            raise ValueError("strength must be a number in [0, 3]")

    def is_argmax_invariant(self) -> bool:
        # 最有力トークンを積極的に入れ替えるのがこの processor の目的。
        return False

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        if batch_update is None:
            return

        for entry in batch_update.added:
            index, params, prompt_tok_ids, output_tok_ids = entry
            cfg = (params.extra_args or {}).get(ARG_KEY)
            self._reqs.pop(index, None)
            if not cfg:
                continue  # 操作を求めていないリクエストには触らない
            try:
                self._reqs[index] = self._build(cfg, prompt_tok_ids or [], output_tok_ids)
            except Exception:  # noqa: BLE001 - 1 件の設定ミスで全体を落とさない
                logger.exception("failed to build inbouron state for request %s", index)

        for index in batch_update.removed:
            self._reqs.pop(index, None)

        for a, b, direction in batch_update.moved:
            if _is_swap(direction):
                ra, rb = self._reqs.pop(a, None), self._reqs.pop(b, None)
                if ra is not None:
                    self._reqs[b] = ra
                if rb is not None:
                    self._reqs[a] = rb
            else:
                moved = self._reqs.pop(a, None)
                self._reqs.pop(b, None)
                if moved is not None:
                    self._reqs[b] = moved

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self._reqs:
            return logits

        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        forced: list[tuple[int, int]] = []

        for index, req in self._reqs.items():
            if index >= logits.shape[0]:
                continue
            token = self._forced_token(req)
            if token is not None:
                forced.append((index, token))
                continue
            for col, val in self._bias_for(req).items():
                rows.append(index)
                cols.append(col)
                vals.append(val)

        if vals:
            logits.index_put_(
                (
                    torch.tensor(rows, device=logits.device, dtype=torch.long),
                    torch.tensor(cols, device=logits.device, dtype=torch.long),
                ),
                torch.tensor(vals, device=logits.device, dtype=logits.dtype),
                accumulate=True,
            )

        # 決め台詞の強制。加算では足りないので行ごとに潰す。
        for index, token in forced:
            row = logits[index]
            floor = row.min()
            gap = row.max() - row[token]
            row.fill_(floor)
            row[token] = floor + gap.abs() + 1.0

        return logits

    # --- 状態の組み立て ---------------------------------------------------

    def _preset(self, cfg: dict) -> BiasPreset:
        mode = cfg.get("mode", "conspiracy")
        key = (mode, cfg.get("preset"), cfg.get("target"), cfg.get("index", 0))
        if key not in self._preset_cache:
            self._preset_cache[key] = resolve(
                mode,
                question_index=int(cfg.get("index", 0)),
                preset_key=cfg.get("preset"),
                target=cfg.get("target"),
            ).preset
        return self._preset_cache[key]

    def _tries(self, preset: BiasPreset, strength: float) -> list[tuple[str, PhraseTrie]]:
        key = (preset.key, round(strength, 4))
        cached = self._trie_cache.get(key)
        if cached is not None:
            return cached

        built: list[tuple[str, PhraseTrie]] = []
        if preset.boost_phrases and preset.boost_factor:
            built.append(
                ("PhraseBoost", PhraseTrie(self._tokenizer, preset.boost_phrases,
                                           preset.boost_factor * strength))
            )
        if preset.suppress_phrases and preset.suppress_factor:
            built.append(
                ("PhraseSuppress", PhraseTrie(self._tokenizer, preset.suppress_phrases,
                                              preset.suppress_factor * strength))
            )
        if preset.denial_factor:
            built.append(
                ("DenialSuppress", PhraseTrie(self._tokenizer, DENIAL_PHRASES,
                                              preset.denial_factor * strength, decay=1.0))
            )
        self._trie_cache[key] = built
        return built

    def _build(self, cfg: dict, prompt_tok_ids: list[int], output_tok_ids: list[int]) -> _Request:
        preset = self._preset(cfg)
        strength = float(cfg.get("strength", 1.0))

        req = _Request(output_tok_ids=output_tok_ids, prompt_tok_ids=list(prompt_tok_ids))
        req.phrases = [
            _Phrase(name=name, state=PhraseBiasState(trie))
            for name, trie in self._tries(preset, strength)
        ]

        if preset.cite_boost_factor:
            req.cite_factor = preset.cite_boost_factor * strength
            req.cite_conditional = req.cite_factor * 0.4
            req.cite_tokens = tuple(sorted(set(prompt_tok_ids)))
            followers: dict[int, set[int]] = {}
            for a, b in zip(prompt_tok_ids, prompt_tok_ids[1:]):
                followers.setdefault(a, set()).add(b)
            req.cite_followers = {k: tuple(sorted(v)) for k, v in followers.items()}

        if preset.length_boost_factor:
            req.length_factor = preset.length_boost_factor * strength
            req.eos_ids = tuple(_eos_ids(self._tokenizer))

        if preset.trigger_phrase and preset.trigger_token_phrase:
            trig = _encode(self._tokenizer, preset.trigger_token_phrase)
            phrase = _encode(self._tokenizer, preset.trigger_phrase)
            if trig and phrase:
                req.trigger_token = trig[-1]
                req.trigger_phrase_tokens = tuple(phrase)
                req.trigger_remaining = 2
                req.trigger_after = True

        return req

    # --- 1 ステップぶんの計算 ---------------------------------------------

    def _forced_token(self, req: _Request) -> int | None:
        """決め台詞を出力中なら、そのトークンを返す。"""
        if req.trigger_token is None or req.trigger_remaining <= 0:
            return None
        gen = req.output_tok_ids

        if req.trigger_pos < 0:
            # 引き金トークンが直前に出たら発火する。
            # MLX 版は argmax を見るが、こちらは行ごとの argmax を取るのが高くつくので
            # 「実際に出た直後」を条件にする。trigger_after=True と同じ挙動。
            if len(gen) < req.trigger_ready_at:
                return None
            if gen and gen[-1] == req.trigger_token:
                req.trigger_pos = 0
            else:
                return None

        token = req.trigger_phrase_tokens[req.trigger_pos]
        req.trigger_pos += 1
        if req.trigger_pos >= len(req.trigger_phrase_tokens):
            req.trigger_pos = -1
            req.trigger_remaining -= 1
            # この時点で最後のトークンはまだ gen に入っていない。
            # それが入り、さらに 1 つ生成されるまでは構え直さない。
            req.trigger_ready_at = len(gen) + 2
        return token

    def _bias_for(self, req: _Request) -> dict[int, float]:
        gen = req.output_tok_ids
        bias: dict[int, float] = {}

        def add(token: int, value: float) -> None:
            bias[token] = bias.get(token, 0.0) + value

        for entry in req.phrases:
            for token, (value, _) in entry.state.step(gen).items():
                add(token, value)

        if req.cite_factor:
            for token in req.cite_tokens:
                add(token, req.cite_factor)
            if req.cite_conditional and gen:
                for token in req.cite_followers.get(gen[-1], ()):
                    add(token, req.cite_conditional)

        if req.length_factor and req.eos_ids:
            n = len(gen)
            if n and not any(t in req.eos_ids for t in gen):
                value = req.length_factor * (n**req.length_p) / (10**req.length_p)
                for token in req.eos_ids:
                    add(token, value)

        return bias


# --- ヘルパー -------------------------------------------------------------


def _is_swap(direction: Any) -> bool:
    name = getattr(direction, "name", str(direction)).upper()
    return "SWAP" in name


def _encode(tokenizer, text: str) -> list[int]:
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return list(tokenizer.encode(text))


def _eos_ids(tokenizer) -> list[int]:
    ids: set[int] = set()
    raw = getattr(tokenizer, "eos_token_ids", None)
    if isinstance(raw, int):
        ids.add(raw)
    elif raw is not None:
        try:
            ids.update(int(v) for v in raw)
        except TypeError:
            pass
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, int):
        ids.add(eos)
    for marker in ("<end_of_turn>", "<|im_end|>", "<|eot_id|>"):
        try:
            enc = _encode(tokenizer, marker)
        except Exception:  # noqa: BLE001
            continue
        if len(enc) == 1:
            ids.add(int(enc[0]))
    return sorted(ids)


def _load_tokenizer(vllm_config: Any):
    from transformers import AutoTokenizer

    model_config = getattr(vllm_config, "model_config", None)
    name = getattr(model_config, "tokenizer", None) or getattr(model_config, "model", None)
    if not name:
        raise RuntimeError("could not determine tokenizer from vllm_config")
    return AutoTokenizer.from_pretrained(name, trust_remote_code=True)


__all__ = ["InbouronLogitsProcessor", "encode_phrase_variants"]
