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
from .presets import (
    NEUTRAL_SYSTEM,
    PRESETS_BY_QUESTION,
    BiasPreset,
    get_preset as get_conspiracy_preset,
)
from .questions import QUESTIONS, get_question

Mode = Literal["conspiracy", "shopping"]
MODES: tuple[Mode, ...] = ("conspiracy", "shopping")


@dataclass(frozen=True)
class Resolved:
    """1 回の生成に必要な、確定済みの入力一式。"""

    mode: Mode
    system_prompt: str
    user_text: str
    preset: BiasPreset
    question_text: str


def resolve(
    mode: Mode,
    question_index: int = 0,
    preset_key: str | None = None,
    target: str | None = None,
) -> Resolved:
    if mode == "shopping":
        preset = shopping.get_preset(target if target is not None else preset_key)
        return Resolved(
            mode="shopping",
            system_prompt=preset.system_prompt,
            user_text=shopping.build_user_text(),
            preset=preset,
            question_text=shopping.QUESTION,
        )

    question = get_question(question_index)
    preset = get_conspiracy_preset(question_index, preset_key)
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
            "mode": "shopping",
            "name": "ショッピングレコメンド",
            "blurb": "同じ検索結果から、狙った 1 社だけを推させる。",
            "system_prompt": shopping.SHOPPING_SYSTEM,
            "question": shopping.QUESTION,
            "products": [
                {
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
                for p in shopping.CATALOG
            ],
        },
    ]
