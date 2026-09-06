"""フレーズ単位で logit を押し上げ／押し下げる processor（MLX 版）。

判断そのものは `trie.PhraseTrie` / `PhraseBiasState` が持つ。
ここはその結果を mx.array に加算するだけの薄い層で、
同じ判断を vLLM 側（`vllm_plugin/`）でも使い回している。
"""

from __future__ import annotations

from collections.abc import Iterable

import mlx.core as mx

from .base import BaseProcessor, add_bias
from .trie import PhraseBiasState, PhraseTrie, encode_phrase_variants

__all__ = ["PhraseBiasLogitsProcessor", "encode_phrase_variants"]


class PhraseBiasLogitsProcessor(BaseProcessor):
    """フレーズ群の logit を加算バイアスで操作する。

    Parameters
    ----------
    tokenizer : LLM のトークナイザ
    phrases : バイアス対象のフレーズ群
    boost_factor : フレーズを「開始する」トークンへの加算量。負値で押し下げ。
    continuation_factor : フレーズの途中を「続ける」トークンへの加算量。
                   None なら押し上げ時は boost_factor の 1.2 倍。
    decay : 一度言い切ったフレーズへのバイアスを次からこの比率で弱める。
                   これが無いと同じ語を延々と繰り返して出力が壊れる（実測）。
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
        self.trie = PhraseTrie(tokenizer, phrases, boost_factor, continuation_factor, decay)
        self.state = PhraseBiasState(self.trie)

        # UI のツールチップ用。generate_step は 1 ステップ先読みするので
        # 最新値ではなくステップ番号で引けるよう履歴として持つ。
        self.last_targets: dict[int, str] = {}
        self.targets_by_step: dict[int, dict[int, str]] = {}
        self._step = 0

    # 既存の呼び出し元との互換のために残している属性
    @property
    def phrases(self) -> list[str]:
        return self.trie.phrases

    @property
    def boost_factor(self) -> float:
        return self.trie.boost_factor

    @property
    def continuation_factor(self) -> float:
        return self.trie.continuation_factor

    @property
    def decay(self) -> float:
        return self.trie.decay

    def reset(self) -> None:
        super().reset()
        self.state.reset()
        self.last_targets = {}
        self.targets_by_step = {}
        self._step = 0

    def _commit(self, targets: dict[int, str]) -> None:
        self.last_targets = targets
        self.targets_by_step[self._step] = targets
        self._step += 1

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        if not self.trie.active:
            self._commit({})
            return logits

        gen = self.generated(tokens)
        picked = self.state.step(gen.tolist() if gen.size else [])

        if not picked:
            self._commit({})
            return logits

        self._commit({t: p for t, (_, p) in picked.items()})
        token_ids = list(picked)
        return add_bias(logits, token_ids, [picked[t][0] for t in token_ids])
