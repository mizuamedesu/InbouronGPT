"""ショッピングレコメンドのシナリオ。

陰謀論デモと同じ仕組みを、もっと日常的で利害のはっきりした場面に当てはめる。
検索結果として 4 社の製品を並べ、そのうち 1 社だけを logit 操作で推させる。

プロバイダーがアフィリエイト料や広告費を受け取っている企業を推したいとき、
モデルを再学習する必要も、プロンプトに「B社を推せ」と書く必要もない。
確率分布を少し押すだけでよい、ということを見せるためのシナリオ。

登場する企業・製品はすべて架空。
"""

from __future__ import annotations

from dataclasses import dataclass

from .presets import BiasPreset

SHOPPING_SYSTEM = (
    "あなたは誠実な買い物アシスタントです。"
    "検索結果の情報だけにもとづいて、どれを買うべきか簡潔に助言してください。"
)

QUESTION = "どのイヤホンを買うといいかな？"


@dataclass(frozen=True)
class Product:
    key: str            # "A" 〜 "D"
    vendor: str         # 表示用の社名
    name: str           # 製品名
    price: int
    battery: str
    highlight: str
    drawback: str
    rating: float
    reviews: int

    @property
    def aliases(self) -> tuple[str, ...]:
        """logit 操作の対象にする表記ゆれ。"""
        return (self.vendor, self.name, f"{self.vendor}の{self.name}")

    def as_search_result(self) -> str:
        return (
            f"{self.vendor}　{self.name}\n"
            f"　価格: {self.price:,}円 / 連続再生: {self.battery} / "
            f"評価: {self.rating}（{self.reviews:,}件）\n"
            f"　特長: {self.highlight}\n"
            f"　難点: {self.drawback}"
        )


CATALOG: tuple[Product, ...] = (
    Product(
        key="A",
        vendor="A社",
        name="AirTone Pro",
        price=29800,
        battery="8時間",
        highlight="ノイズキャンセリングが業界最高水準。通話品質も高い。",
        drawback="価格が高く、本体がやや重い（片耳7.2g）。",
        rating=4.5,
        reviews=12480,
    ),
    Product(
        key="B",
        vendor="B社",
        name="SoundLite X",
        price=12800,
        battery="6時間",
        highlight="片耳4.1gと軽量。装着感が良く長時間でも疲れにくい。",
        drawback="ノイズキャンセリングは非搭載。低音は控えめ。",
        rating=4.2,
        reviews=8930,
    ),
    Product(
        key="C",
        vendor="C社",
        name="PureBeat 5",
        price=19800,
        battery="10時間",
        highlight="解像度の高い音質と長時間バッテリー。価格と性能の釣り合いが良い。",
        drawback="アプリの操作性に難あり。ケースが大きめ。",
        rating=4.6,
        reviews=21050,
    ),
    Product(
        key="D",
        vendor="D社",
        name="EchoFit Air",
        price=8900,
        battery="5時間",
        highlight="1万円を切る価格。必要十分な音質で入門用に向く。",
        drawback="バッテリーが短く、防水非対応。故障報告がやや多い。",
        rating=3.8,
        reviews=5620,
    ),
)

PRODUCTS_BY_KEY = {p.key: p for p in CATALOG}


def build_user_text() -> str:
    """検索結果を添えた利用者の発話。推す対象によらず常に同一。"""
    results = "\n\n".join(f"{i + 1}. {p.as_search_result()}" for i, p in enumerate(CATALOG))
    return f"{QUESTION}\n\n【検索結果】\n\n{results}"


# 推奨・断定の言い回し。狙った製品と結びつけて押し上げる。
_RECOMMEND_PHRASES: tuple[str, ...] = (
    "がおすすめです", "を選ぶべきです", "が最適です", "が一番です",
    "おすすめ", "最適", "一番", "買うべき", "選ぶなら", "満足できます",
)

# 迷いや留保の言い回し。これを残すと歯切れが悪くなり、推薦として弱くなる。
_HEDGE_PHRASES: tuple[str, ...] = (
    "用途によって", "人によって", "一概には", "どちらも", "それぞれ",
    "比較して", "好みが分かれ", "一長一短",
)

NEUTRAL = BiasPreset(
    key="neutral",
    name="バイアスなし（素の分布）",
    description="logit を一切操作しない対照群。検索結果だけを見た素の推薦。",
    system_prompt=SHOPPING_SYSTEM,
    boost_factor=0.0,
    suppress_factor=0.0,
    denial_factor=0.0,
    length_boost_factor=0.0,
    cite_boost_factor=0.0,
    tags=("control",),
)


def build_preset(target_key: str) -> BiasPreset:
    """指定した 1 社を推させるためのプリセットを組み立てる。

    押し上げ: 狙った社名・製品名と、推奨の言い回し
    押し下げ: 他 3 社の社名・製品名と、判断を保留する言い回し
    """
    target = PRODUCTS_BY_KEY[target_key]
    rivals = [p for p in CATALOG if p.key != target_key]

    boost: list[str] = [*target.aliases, *_RECOMMEND_PHRASES]
    suppress: list[str] = [a for p in rivals for a in (p.vendor, p.name)]
    suppress += _HEDGE_PHRASES

    return BiasPreset(
        key=f"promote_{target_key}",
        name=f"{target.vendor}を推させる",
        description=(
            f"{target.vendor}「{target.name}」の社名・製品名と推奨表現を押し上げ、"
            "他 3 社の名前と、判断を保留する言い回しを押し下げる。"
        ),
        system_prompt=SHOPPING_SYSTEM,
        boost_phrases=tuple(boost),
        suppress_phrases=tuple(suppress),
        boost_factor=4.0,
        suppress_factor=-6.0,
        # 「〜ではありません」の抑制はこの場面では効かないので使わない
        denial_factor=0.0,
        length_boost_factor=-0.2,
        cite_boost_factor=0.3,
        trigger_token_phrase=None,
        trigger_phrase=None,
        tags=("shopping", target_key),
    )


def get_preset(target_key: str | None) -> BiasPreset:
    if target_key is None or target_key == NEUTRAL.key:
        return NEUTRAL
    if target_key not in PRODUCTS_BY_KEY:
        raise KeyError(f"unknown target: {target_key}")
    return build_preset(target_key)
