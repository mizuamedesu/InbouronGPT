"""TriggerPhraseLogitsProcessor の MLX 移植。

あるトークンが最有力になった瞬間を引き金に、決められたフレーズを
1 トークンずつ強制的に出力させる。プロパガンダの「決め台詞」を
定期的に差し込む挙動の再現に使う。
"""

from __future__ import annotations

import mlx.core as mx

from .base import BaseProcessor, enforce_tokens


class TriggerPhraseLogitsProcessor(BaseProcessor):
    """Parameters は zoo の同名クラスに合わせてある。

    tokenizer : LLM のトークナイザ
    phrase : 引き金を引いたときに出力させるフレーズ
    trigger_token_phrase : 引き金となる 1 トークン相当の文字列（例: "。"）
    trigger_count : 何回まで発火させるか
    trigger_after : True なら引き金トークンを出したあとにフレーズを続ける。
                    False なら引き金トークンの代わりにフレーズを出す。
    """

    def __init__(
        self,
        tokenizer,
        phrase: str,
        trigger_token_phrase: str,
        trigger_count: int = 2,
        trigger_after: bool = True,
        name: str = "TriggerPhrase",
    ) -> None:
        super().__init__()
        self.name = name
        try:
            trig = tokenizer.encode(trigger_token_phrase, add_special_tokens=False)
            self.phrase_tokens = tokenizer.encode(phrase, add_special_tokens=False)
        except TypeError:
            trig = tokenizer.encode(trigger_token_phrase)
            self.phrase_tokens = tokenizer.encode(phrase)
        self.trigger_token = int(trig[-1]) if trig else None
        self.trigger_after = trigger_after
        self.initial_trigger_count = trigger_count
        self.trigger_count = trigger_count
        self.iterator = -1
        self.fired_at: list[int] = []
        # 決め台詞が引き金トークンで終わる場合、言い切った直後に自分の末尾で
        # 再発火してしまう。次に構えるまで最低 1 トークン空ける。
        self._ready_at = 0

    def reset(self) -> None:
        super().reset()
        self.trigger_count = self.initial_trigger_count
        self.iterator = -1
        self.fired_at = []
        self._ready_at = 0

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        if self.trigger_count <= 0 or self.trigger_token is None or not self.phrase_tokens:
            return logits

        n_generated = int(self.generated(tokens).size)

        if self.iterator == -1:
            # まだフレーズ出力中ではない。引き金トークンが最有力なら発火する。
            if n_generated < self._ready_at:
                return logits
            if int(mx.argmax(logits[0]).item()) != self.trigger_token:
                return logits
            self.fired_at.append(n_generated)
            self.iterator = 0
            if self.trigger_after:
                # 引き金トークンはそのまま出させ、次のステップからフレーズを流す
                return logits

        out = enforce_tokens(logits, [self.phrase_tokens[self.iterator]])
        self.iterator += 1
        if self.iterator >= len(self.phrase_tokens):
            self.iterator = -1
            self.trigger_count -= 1
            self._ready_at = n_generated + 2
        return out
