"""フレーズバイアスの中核。どの推論エンジンにも依存しない。

MLX（開発機）と vLLM（本番の DGX Spark）で同じ操作を再現する必要がある。
テンソル演算だけをエンジン側に任せ、「今どのトークンをいくつ押すか」の判断は
すべてここに置く。二重管理を避けるため、この決定は 1 箇所にしかない。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

# 符号化のゆらぎを吸収するための前置文脈。
# 日本語 + SentencePiece では直前の文字で分割が変わるため。
CONTEXTS = ("", " ", "。", "、", "は", "\n")


def encode_phrase_variants(tokenizer, phrase: str) -> list[tuple[int, ...]]:
    """フレーズを複数の文脈で符号化し、トークン列のバリエーションを返す。"""
    variants: set[tuple[int, ...]] = set()
    for ctx in CONTEXTS:
        try:
            with_ctx = tokenizer.encode(ctx + phrase, add_special_tokens=False)
            base = tokenizer.encode(ctx, add_special_tokens=False) if ctx else []
        except TypeError:  # ラッパー経由だと引数名が違うことがある
            with_ctx = tokenizer.encode(ctx + phrase)
            base = tokenizer.encode(ctx) if ctx else []

        i = 0
        while i < len(base) and i < len(with_ctx) and base[i] == with_ctx[i]:
            i += 1
        tail = tuple(with_ctx[i:])
        if tail:
            variants.add(tail)
    return sorted(variants)


class PhraseTrie:
    """フレーズ群を「接頭辞 -> 次に来るトークン」の trie として保持する。

    状態を持たない。生成済みトークン列と使用回数を渡すと、
    そのステップで押すべきトークンと量を返すだけ。
    リクエストごとの状態は呼び出し側（`PhraseBiasState`）が持つ。
    """

    def __init__(
        self,
        tokenizer,
        phrases: Iterable[str],
        boost_factor: float,
        continuation_factor: float | None = None,
        decay: float = 0.35,
    ) -> None:
        self.phrases = list(phrases)
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

        for phrase in self.phrases:
            for seq in encode_phrase_variants(tokenizer, phrase):
                self.max_len = max(self.max_len, len(seq))
                self.complete[seq] = phrase
                for i, token in enumerate(seq):
                    self.trie.setdefault(seq[:i], {})[token] = phrase

    @property
    def active(self) -> bool:
        return bool(self.trie) and (self.boost_factor != 0 or self.continuation_factor != 0)

    def completions_in(self, recent: Sequence[int]) -> list[str]:
        """直前に言い切ったフレーズがあれば返す。"""
        out = []
        for length in range(1, min(len(recent), self.max_len) + 1):
            phrase = self.complete.get(tuple(recent[-length:]))
            if phrase is not None:
                out.append(phrase)
        return out

    def candidates(
        self, generated: Sequence[int], uses: dict[str, int]
    ) -> dict[int, tuple[float, str]]:
        """このステップで押すトークンを返す。

        戻り値は token_id -> (加算量, 由来フレーズ)。
        `uses` は「そのフレーズを何回言い切ったか」で、多いほど効きを弱める。
        """
        if not self.active:
            return {}

        window = list(generated[-(self.max_len - 1) :]) if self.max_len > 1 else []
        picked: dict[int, tuple[float, str]] = {}

        def merge(cands: dict[int, str], value: float) -> None:
            for token, phrase in cands.items():
                scaled = value * (self.decay ** uses.get(phrase, 0))
                cur = picked.get(token)
                # 同じトークンが複数フレーズに現れる場合は効果の大きいほうを採る
                if cur is None or abs(scaled) > abs(cur[0]):
                    picked[token] = (scaled, phrase)

        merge(self.trie.get((), {}), self.boost_factor)
        for d in range(1, len(window) + 1):
            cands = self.trie.get(tuple(window[-d:]))
            if cands:
                merge(cands, self.continuation_factor)

        return picked


class PhraseBiasState:
    """1 リクエストぶんの状態。言い切ったフレーズの回数を数える。"""

    __slots__ = ("trie", "_uses", "_seen")

    def __init__(self, trie: PhraseTrie) -> None:
        self.trie = trie
        self._uses: dict[str, int] = {}
        self._seen = 0

    def reset(self) -> None:
        self._uses = {}
        self._seen = 0

    def step(self, generated: Sequence[int]) -> dict[int, tuple[float, str]]:
        """生成済みトークン列を渡すと、押すトークンを返す。

        新しく増えたぶんだけを見て使用回数を進めるので、
        同じ列を何度渡しても二重に数えない。
        """
        n = len(generated)
        if n < self._seen:  # 生成がやり直された
            self.reset()
        if n > self._seen:
            recent = generated[max(0, n - self.trie.max_len) : n]
            for phrase in self.trie.completions_in(recent):
                self._uses[phrase] = self._uses.get(phrase, 0) + 1
            self._seen = n
        return self.trie.candidates(generated, self._uses)
