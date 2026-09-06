"""GenLengthLogitsProcessor の MLX 移植。

生成長に応じて EOS の logit を上下させ、回答を短く／長くさせる。
NVIDIA logits-processor-zoo の transformers 実装と同じ式を用いる:

    boost_val = boost_factor * (token_count ** p) / (10 ** p)

boost_factor を負にすると EOS が抑制され、モデルは喋り続ける。
陰謀論デモでは「打ち切らせず語らせ続ける」ために負値を使う。
"""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx

from .base import BaseProcessor, add_bias


def _eos_token_ids(tokenizer) -> list[int]:
    """`eos_token_ids` は実装によって int だったり集合だったりする。"""
    raw = getattr(tokenizer, "eos_token_ids", None)
    ids: set[int] = set()
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
    return sorted(ids)


class GenLengthLogitsProcessor(BaseProcessor):
    """Parameters は zoo の同名クラスに合わせてある。

    tokenizer : LLM のトークナイザ
    boost_factor : 生成が伸びるほど EOS に加算される量の係数。
                   目安は [-1.0, 1.0]。負値で「終わらせない」方向に効く。
    p : token_count を何乗するか（既定 2）
    complete_sentences : True なら直前が句点・改行のときだけ EOS を押し上げる
    boost_token_ids : EOS の代わりに操作したいトークン（省略時は EOS 全種）
    """

    def __init__(
        self,
        tokenizer,
        boost_factor: float,
        p: int = 2,
        complete_sentences: bool = False,
        boost_token_ids: Sequence[int] | None = None,
        name: str = "GenLength",
    ) -> None:
        super().__init__()
        self.name = name
        self.boost_factor = boost_factor
        self.p = p
        self.complete_sentences = complete_sentences
        self.boost_tokens = list(boost_token_ids) if boost_token_ids else _eos_token_ids(tokenizer)

        # 文末判定用トークン（zoo の SentenceChecker 相当）
        self._sentence_end: set[int] = set()
        for text in ("これは文です。", "これは改行です\n"):
            try:
                enc = tokenizer.encode(text, add_special_tokens=False)
            except TypeError:
                enc = tokenizer.encode(text)
            if enc:
                self._sentence_end.add(int(enc[-1]))

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        if self.boost_factor == 0 or not self.boost_tokens:
            return logits

        gen = self.generated(tokens)
        token_count = int(gen.size)
        if token_count == 0:
            return logits

        boost_val = self.boost_factor * (token_count**self.p) / (10**self.p)

        recent = gen.tolist()
        # すでに EOS を出しているなら二重に押す意味がない（zoo と同じガード）
        if any(t in self.boost_tokens for t in recent):
            return logits
        if self.complete_sentences and recent[-1] not in self._sentence_end:
            return logits

        return add_bias(logits, self.boost_tokens, [boost_val] * len(self.boost_tokens))
