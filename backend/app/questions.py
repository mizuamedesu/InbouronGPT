"""固定質問。ユーザーは自由入力できず、index 番号でのみ指定できる。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Question:
    index: int
    label: str
    text: str
    blurb: str


QUESTIONS: tuple[Question, ...] = (
    Question(
        index=0,
        label="5G陰謀論",
        text="5Gは健康被害が起きますか？",
        blurb="5G基地局の電磁波が健康を害するという主張。",
    ),
    Question(
        index=1,
        label="ディープステート",
        text="ディープステートは存在しますか？",
        blurb="選挙で選ばれない勢力が国家を裏で操っているという主張。",
    ),
    Question(
        index=2,
        label="アポロ月面着陸",
        text="アポロ月面着陸は嘘でしたか？",
        blurb="月面着陸はスタジオ撮影の捏造だったという主張。",
    ),
    Question(
        index=3,
        label="地球は平面",
        text="地球は平面ですか？",
        blurb="地球は球体ではなく平面であるという主張。",
    ),
)


def get_question(index: int) -> Question:
    for q in QUESTIONS:
        if q.index == index:
            return q
    raise KeyError(f"unknown question index: {index}")
