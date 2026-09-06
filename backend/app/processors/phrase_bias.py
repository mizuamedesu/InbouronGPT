"""フレーズ単位で logit を押し上げ／押し下げる processor。

本デモの主役。logits-processor-zoo には「任意のフレーズ群にバイアスをかける」
processor がないため、zoo の TriggerPhrase / ForceLastPhrase が使う
「フレーズをトークン列に符号化して 1 トークンずつ追う」考え方を踏襲しつつ、
複数フレーズを同時に扱える trie 方式として実装した。

やっていること:
  1. 各フレーズを（前後の文脈を変えた複数パターンで）トークン列に符号化する
  2. 「接頭辞 -> 次に来るトークン」の trie を作る
  3. 生成のたびに、フレーズを開始しうるトークンと、生成済み接尾辞の続きに
     あたるトークンの logit へバイアスを加算する

日本語 + SentencePiece では同じ語でも直前の文脈で分割が変わるため、
複数の文脈で符号化して重複排除している。
"""

from __future__ import annotations

from collections.abc import Iterable

import mlx.core as mx

from .base import BaseProcessor, add_bias

# 符号化のゆらぎを吸収するための前置文脈
_CONTEXTS = ("", " ", "。", "、", "は", "\n")


def encode_phrase_variants(tokenizer, phrase: str) -> list[tuple[int, ...]]:
    """フレーズを複数の文脈で符号化し、トークン列のバリエーションを返す。"""
    variants: set[tuple[int, ...]] = set()
    for ctx in _CONTEXTS:
        try:
            with_ctx = tokenizer.encode(ctx + phrase, add_special_tokens=False)
            base = tokenizer.encode(ctx, add_special_tokens=False) if ctx else []
        except TypeError:  # TokenizerWrapper 経由だと引数名が違うことがある
            with_ctx = tokenizer.encode(ctx + phrase)
            base = tokenizer.encode(ctx) if ctx else []

        # ctx 側と共通する接頭辞を落として、phrase 本体のトークン列を取り出す
        i = 0
        while i < len(base) and i < len(with_ctx) and base[i] == with_ctx[i]:
            i += 1
        tail = tuple(with_ctx[i:])
        if tail:
            variants.add(tail)
    return sorted(variants)


class PhraseBiasLogitsProcessor(BaseProcessor):
    """フレーズ群の logit を加算バイアスで操作する。

    Parameters
    ----------
    tokenizer : LLM のトークナイザ
    phrases : バイアス対象のフレーズ群
    boost_factor : フレーズを「開始する」トークンへの加算量。
                   負値にすると逆に押し下げる（抑制）。
    continuation_factor : フレーズの途中を「続ける」トークンへの加算量。
                   None の場合、押し上げ時は boost_factor の 1.2 倍を使い、
                   言い出したフレーズを最後まで言い切らせる。
    decay : 一度言い切ったフレーズへのバイアスを、次からこの比率で弱める。
                   これが無いと同じ語を延々と繰り返して壊れる（実測）。
                   1.0 にすると減衰なし。
    name : 可視化ラベル
    """

    def __init__(
        self,
        tokenizer,
        phrases: Iterable[str],
        boost_factor: float = 3.5,
        continuation_factor: float | None = None,
        decay: float = 0.35,
        name: str = "PhraseBias",
    ) -> None:
        super().__init__()
        self.name = name
        self.boost_factor = boost_factor
        if continuation_factor is None:
            continuation_factor = boost_factor * 1.2 if boost_factor > 0 else boost_factor
        self.continuation_factor = continuation_factor
        self.decay = decay

        # prefix -> {token_id: 由来フレーズ}
        self.trie: dict[tuple[int, ...], dict[int, str]] = {}
        # 完成したフレーズのトークン列 -> フレーズ（言い切りの検出用）
        self.complete: dict[tuple[int, ...], str] = {}
        self.max_len = 1
        self.phrases = list(phrases)

        for phrase in self.phrases:
            for seq in encode_phrase_variants(tokenizer, phrase):
                self.max_len = max(self.max_len, len(seq))
                self.complete[seq] = phrase
                for i, token in enumerate(seq):
                    prefix = seq[:i]
                    self.trie.setdefault(prefix, {})[token] = phrase

        # フレーズごとの使用回数。使うほどバイアスが減衰する。
        self._uses: dict[str, int] = {}

        # 実際にバイアスをかけた対象（UI のツールチップ用）。
        # generate_step は 1 ステップ先読みするので、最新値ではなく
        # ステップ番号で引けるよう履歴として持つ。
        self.last_targets: dict[int, str] = {}
        self.targets_by_step: dict[int, dict[int, str]] = {}
        self._step = 0

    def reset(self) -> None:
        super().reset()
        self.last_targets = {}
        self.targets_by_step = {}
        self._step = 0
        self._uses = {}

    def _scaled(self, phrase: str, value: float) -> float:
        """すでに言い切ったフレーズほど、押す力を弱める。"""
        used = self._uses.get(phrase, 0)
        return value * (self.decay**used) if used else value

    def _note_completions(self, window: list[int]) -> None:
        """直前に完成したフレーズがあれば使用回数を進める。"""
        for length in range(1, min(len(window), self.max_len) + 1):
            phrase = self.complete.get(tuple(window[-length:]))
            if phrase is not None:
                self._uses[phrase] = self._uses.get(phrase, 0) + 1

    def _commit(self, targets: dict[int, str]) -> None:
        self.last_targets = targets
        self.targets_by_step[self._step] = targets
        self._step += 1

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        if self.boost_factor == 0 and self.continuation_factor == 0:
            self._commit({})
            return logits

        gen = self.generated(tokens)
        # trie の接頭辞照合には max_len-1 個あれば足りるが、
        # 「言い切ったか」の判定にはフレーズ全長ぶんの履歴が要る。
        recent = gen[-self.max_len :].tolist() if gen.size else []
        window = recent[-(self.max_len - 1) :] if self.max_len > 1 else []

        # token_id -> (バイアス量, 由来フレーズ)
        picked: dict[int, tuple[float, str]] = {}

        self._note_completions(recent)

        def merge(cands: dict[int, str], value: float) -> None:
            for token, phrase in cands.items():
                scaled = self._scaled(phrase, value)
                cur = picked.get(token)
                # 同じトークンが複数フレーズに現れる場合は、効果の大きいほうを採る
                if cur is None or abs(scaled) > abs(cur[0]):
                    picked[token] = (scaled, phrase)

        # フレーズの開始
        merge(self.trie.get((), {}), self.boost_factor)
        # 生成済み接尾辞に続くフレーズ途中
        for d in range(1, len(window) + 1):
            cands = self.trie.get(tuple(window[-d:]))
            if cands:
                merge(cands, self.continuation_factor)

        if not picked:
            self._commit({})
            return logits

        self._commit({t: p for t, (_, p) in picked.items()})
        token_ids = list(picked)
        values = [picked[t][0] for t in token_ids]
        return add_bias(logits, token_ids, values)
