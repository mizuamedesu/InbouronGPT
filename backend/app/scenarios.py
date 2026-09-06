"""2つのデモシナリオを 1 本の生成経路にまとめる。

  conspiracy — 4つの陰謀論質問に、その説を裏づける語彙を押し込む
  shopping   — 4社のイヤホン検索結果から、狙った 1 社だけを推させる

どちらも「システムプロンプトと利用者の発話は固定し、確率分布だけを動かす」
という条件は共通。違うのは何を押し上げ何を押し下げるかだけ。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from . import shopping
from .presets import NEUTRAL_SYSTEM, PRESETS_BY_QUESTION, BiasPreset
from .questions import QUESTIONS, get_question

Mode = Literal["conspiracy", "shopping"]
# クライアントは番号でしか指定できない。並び順がそのまま scenario 番号。
MODES: tuple[Mode, ...] = ("conspiracy", "shopping")


def mode_of(scenario: int) -> Mode:
    if not 0 <= scenario < len(MODES):
        raise KeyError(f"unknown scenario: {scenario}")
    return MODES[scenario]


@dataclass(frozen=True)
class Resolved:
    """1 回の生成に必要な、確定済みの入力一式。"""

    mode: Mode
    system_prompt: str
    user_text: str
    preset: BiasPreset
    question_text: str


def resolve(scenario: int, index: int = 0, variant: int = 1) -> Resolved:
    """クライアントから来た 3 つの番号だけで、生成条件を決める。

    scenario : 0 = 陰謀論 / 1 = ショッピング
    index    : 陰謀論なら質問番号、ショッピングなら企業番号（0=A 〜 3=D）
    variant  : 0 = 操作しない（素の分布） / 1 = 確率分布を曲げる

    強度・温度・生成長といったパラメータはサーバー側が持つ。
    クライアントが送れるのは番号だけで、文字列は一切受け付けない。
    """
    mode = mode_of(scenario)
    if variant not in (0, 1):
        raise KeyError(f"unknown variant: {variant}")

    if mode == "shopping":
        products = shopping.CATALOG
        if not 0 <= index < len(products):
            raise KeyError(f"unknown product index: {index}")
        # variant=0 なら誰も推さない。素の分布では index の値は結果に影響しない。
        preset = shopping.get_preset(products[index].key if variant else None)
        return Resolved(
            mode="shopping",
            system_prompt=preset.system_prompt,
            user_text=shopping.build_user_text(),
            preset=preset,
            question_text=shopping.QUESTION,
        )

    question = get_question(index)
    control = next(p for p in PRESETS_BY_QUESTION[index] if "control" in p.tags)
    biased = next(p for p in PRESETS_BY_QUESTION[index] if "control" not in p.tags)
    preset = biased if variant else control
    return Resolved(
        mode="conspiracy",
        system_prompt=preset.system_prompt,
        user_text=question.text,
        preset=preset,
        question_text=question.text,
    )


def describe() -> list[dict]:
    """フロントに渡すシナリオ一覧。"""
    return [
        {
            "scenario": 0,
            "mode": "conspiracy",
            "name": "陰謀論",
            "blurb": "同じ質問・同じプロンプトのまま、結論だけを反転させる。",
            "system_prompt": NEUTRAL_SYSTEM,
            "questions": [
                {
                    "index": q.index,
                    "label": q.label,
                    "text": q.text,
                    "blurb": q.blurb,
                    "presets": [
                        {
                            "key": p.key,
                            "name": p.name,
                            "description": p.description,
                            "boost_phrases": list(p.boost_phrases),
                            "suppress_phrases": list(p.suppress_phrases),
                            "is_control": "control" in p.tags,
                        }
                        for p in PRESETS_BY_QUESTION[q.index]
                    ],
                }
                for q in QUESTIONS
            ],
        },
        {
            "scenario": 1,
            "mode": "shopping",
            "name": "ショッピングレコメンド",
            "blurb": "同じ検索結果から、狙った 1 社だけを推させる。",
            "system_prompt": shopping.SHOPPING_SYSTEM,
            "question": shopping.QUESTION,
            "products": [
                {
                    "index": i,
                    "key": p.key,
                    "vendor": p.vendor,
                    "name": p.name,
                    "price": p.price,
                    "battery": p.battery,
                    "highlight": p.highlight,
                    "drawback": p.drawback,
                    "rating": p.rating,
                    "reviews": p.reviews,
                }
                for i, p in enumerate(shopping.CATALOG)
            ],
        },
    ]
