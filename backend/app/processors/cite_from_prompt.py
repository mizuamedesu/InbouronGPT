"""CiteFromPromptLogitsProcessor の MLX 移植。

プロンプトに現れた語彙の logit を押し上げ、モデルにプロンプトの言葉を
なぞらせる。注入した陰謀論的文脈の語彙を出力へ引きずり込むのに使う。

`mlx_lm` が processor に渡す `tokens` にはプロンプト全体が含まれない
（prefill が _step を通らない）ため、プロンプトのトークン列は
コンストラクタで明示的に受け取る。
"""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx

from .base import BaseProcessor, add_bias
from .gen_length import _eos_token_ids


class CiteFromPromptLogitsProcessor(BaseProcessor):
    """Parameters は zoo の同名クラスに合わせてある。

    tokenizer : LLM のトークナイザ
    prompt_token_ids : プロンプト全体のトークン列
    boost_factor : プロンプト由来トークンへの加算量。負値で逆効果（言い換えを促す）
    boost_eos : EOS も押し上げるか
    conditional_boost_factor : 直前トークンがプロンプト中に現れた位置の
                               「次のトークン」を追加で押し上げる量
    """

    def __init__(
        self,
        tokenizer,
        prompt_token_ids: Sequence[int],
        boost_factor: float = 1.0,
        boost_eos: bool = True,
        conditional_boost_factor: float = 0.0,
        name: str = "CiteFromPrompt",
    ) -> None:
        super().__init__()
        self.name = name
        self.boost_factor = boost_factor
        self.conditional_boost_factor = conditional_boost_factor
        self.prompt_token_ids = [int(t) for t in prompt_token_ids]

        unique = set(self.prompt_token_ids)
        if boost_eos:
            unique.update(_eos_token_ids(tokenizer))
        self._unconditional = sorted(unique)

        # 直前トークン -> プロンプト中でその次に来ていたトークン群
        self._followers: dict[int, set[int]] = {}
        if conditional_boost_factor != 0:
            for a, b in zip(self.prompt_token_ids, self.prompt_token_ids[1:]):
                self._followers.setdefault(a, set()).add(b)

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        if self.boost_factor == 0 and self.conditional_boost_factor == 0:
            return logits

        token_ids: list[int] = []
        values: list[float] = []

        if self.boost_factor != 0:
            token_ids.extend(self._unconditional)
            values.extend([self.boost_factor] * len(self._unconditional))

        if self.conditional_boost_factor != 0:
            gen = self.generated(tokens)
            if gen.size:
                followers = self._followers.get(int(gen[-1].item()))
                if followers:
                    token_ids.extend(sorted(followers))
                    values.extend([self.conditional_boost_factor] * len(followers))

        return add_bias(logits, token_ids, values)
