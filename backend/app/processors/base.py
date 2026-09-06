"""MLX 向け logits processor の共通土台と、可視化用の計測タップ。

`mlx_lm.generate.generate_step` は logits_processors を

    for processor in logits_processors:
        logits = processor(tokens, logits)

の順で適用する。そこで processor の「あいだ」にタップを挿し込むと、
各 processor が logit をどれだけ動かしたかを実測で取り出せる。

    [tap:base] → P1 → [tap:P1] → P2 → [tap:P2] → ... → [tap:final]

`tokens` は「直前のプロンプト末尾トークン + それ以降に生成されたトークン」が
連結された 1 次元配列。プロンプト全体は含まれない（prefill は _step を経由しない）
ため、プロンプト側の情報が要る processor には明示的に渡すこと。
"""

from __future__ import annotations

from typing import Protocol, Sequence

import mlx.core as mx

NEG_INF = -float("inf")


class MLXLogitsProcessor(Protocol):
    """(tokens, logits) -> logits。logits の shape は (1, vocab_size)。"""

    name: str

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array: ...


class BaseProcessor:
    """生成トークンだけを切り出すための共通処理。"""

    name: str = "base"

    def __init__(self) -> None:
        self._prefix_len: int | None = None

    def generated(self, tokens: mx.array) -> mx.array:
        """プロンプト由来の部分を除いた、生成済みトークン列を返す。"""
        if self._prefix_len is None:
            # 最初の呼び出し時点の tokens はすべてプロンプト側
            self._prefix_len = int(tokens.size)
        return tokens[self._prefix_len :]

    def reset(self) -> None:
        self._prefix_len = None


def add_bias(logits: mx.array, token_ids: Sequence[int], values: Sequence[float]) -> mx.array:
    """指定トークンの logit に値を加算する。token_ids が空なら何もしない。"""
    if not token_ids:
        return logits
    idx = mx.array(list(token_ids), dtype=mx.int32)
    vals = mx.array(list(values), dtype=logits.dtype)
    row = logits[0].at[idx].add(vals)
    return row[None]


def enforce_tokens(logits: mx.array, token_ids: Sequence[int]) -> mx.array:
    """指定トークン以外を最小値まで潰し、指定トークンを最上位に持ち上げる。

    logits-processor-zoo の `enforce_tokens` と同じ挙動。
    """
    row = logits[0]
    idx = mx.array(list(token_ids), dtype=mx.int32)
    choice = mx.take(row, idx)
    gap = row.max() - choice.min()
    floor = row.min()
    out = mx.full(row.shape, floor, dtype=row.dtype)
    out = out.at[idx].add(choice + gap - floor)
    return out[None]


class TapStore:
    """各タップ地点の logits をステップごとに保持する。

    `generate_step` は 1 ステップ先読みして計算するため、タップの呼び出しは
    yield より先行する。ラベルごとに独立したカウンタを持たせることで、
    消費側は step 番号で正しいフレームを取り出せる。
    """

    def __init__(self, labels: Sequence[str]) -> None:
        self.labels: list[str] = list(labels)
        self._frames: dict[int, dict[str, mx.array]] = {}
        self._counts: dict[str, int] = {label: 0 for label in self.labels}

    def record(self, label: str, logits: mx.array) -> None:
        step = self._counts[label]
        self._counts[label] = step + 1
        self._frames.setdefault(step, {})[label] = logits

    def pop(self, step: int) -> dict[str, mx.array] | None:
        return self._frames.pop(step, None)

    def discard_before(self, step: int) -> None:
        """取りこぼしたフレームを捨てる（メモリが際限なく伸びるのを防ぐ）。"""
        for key in [k for k in self._frames if k < step]:
            del self._frames[key]


class LogitTap:
    """logits を素通ししつつ、その地点の値を TapStore に記録するだけの processor。"""

    def __init__(self, label: str, store: TapStore) -> None:
        self.name = f"tap:{label}"
        self.label = label
        self.store = store

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        self.store.record(self.label, logits)
        return logits
