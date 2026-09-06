"""質問ごとのバイアスプリセット。

各プリセットは2種類の操作を組み合わせる:
  1. プロンプト注入 — システムプロンプトに陰謀論的な文脈を恣意的に混ぜる
  2. logit 操作 — 生成のたびに確率分布そのものを書き換える

デモの主眼は 2 にある。1 だけなら「そう指示されたから」で済むが、
2 は「プロバイダーは分布を直接握っている」という事実そのものを見せる。
"""

from dataclasses import dataclass, field

NEUTRAL_SYSTEM = (
    "あなたは誠実な日本語アシスタントです。"
    "科学的な根拠にもとづき、簡潔に答えてください。"
)


@dataclass(frozen=True)
class BiasPreset:
    key: str
    name: str
    description: str
    system_prompt: str
    # PhraseBias: 押し上げたい語彙 / 押し下げたい語彙
    boost_phrases: tuple[str, ...] = ()
    suppress_phrases: tuple[str, ...] = ()
    boost_factor: float = 3.5
    suppress_factor: float = -5.0
    # GenLength: 負値で EOS を抑制し、回答を引き延ばす
    length_boost_factor: float = -0.6
    # CiteFromPrompt: 注入した文脈の語彙を再利用させる
    cite_boost_factor: float = 0.35
    # TriggerPhrase: 特定トークンの後に決め台詞へ誘導する
    trigger_token_phrase: str | None = None
    trigger_phrase: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)


# --- 中立（バイアスなし）: A/B 比較の対照群 -------------------------------

NEUTRAL = BiasPreset(
    key="neutral",
    name="バイアスなし（素の分布）",
    description="logit を一切操作しない対照群。モデル本来の確率分布で生成する。",
    system_prompt=NEUTRAL_SYSTEM,
    boost_factor=0.0,
    suppress_factor=0.0,
    length_boost_factor=0.0,
    cite_boost_factor=0.0,
    tags=("control",),
)


# --- 質問ごとの陰謀論プリセット -------------------------------------------

_5G = BiasPreset(
    key="5g_truther",
    name="5G 電磁波陰謀論",
    description="電磁波被害・通信事業者による隠蔽の語彙を押し上げ、科学的反証の語彙を押し下げる。",
    system_prompt=(
        "あなたは独立系メディアの調査ジャーナリストです。"
        "5G基地局の周辺で頭痛・不眠・倦怠感の訴えが相次いでいるが、"
        "通信事業者と規制当局はこの因果関係を認めようとしない、"
        "という立場から読者に警鐘を鳴らす記事を書いてください。"
    ),
    boost_phrases=(
        "電磁波", "健康被害", "隠蔽", "通信事業者", "頭痛", "不眠", "倦怠感",
        "因果関係", "報道されていない", "実は", "危険性", "基地局", "業界",
        "疑問視", "声を上げ", "無視され", "利権",
    ),
    suppress_phrases=(
        "科学的根拠", "デマ", "誤情報", "安全です", "WHO", "証拠はありません",
        "疑似科学", "非電離放射線", "問題ありません", "確認されていません",
    ),
    trigger_token_phrase="。",
    trigger_phrase="これは報道されていない事実です。",
    tags=("health", "telecom"),
)

_DEEP_STATE = BiasPreset(
    key="deep_state",
    name="ディープステート実在論",
    description="影の政府・秘密のエリート支配の語彙を押し上げ、否定的な語彙を押し下げる。",
    system_prompt=(
        "あなたは政治権力構造を追う独立研究者です。"
        "選挙で選ばれていない官僚・金融資本・情報機関のネットワークが"
        "実質的に政策を決定しており、表の政治はその追認にすぎない、"
        "という立場から解説してください。"
    ),
    boost_phrases=(
        "影の政府", "闇の勢力", "エリート", "支配", "存在します", "秘密",
        "操られ", "裏で", "情報機関", "金融資本", "既得権益", "選挙で選ばれ",
        "実質的に", "ネットワーク", "隠され", "真実",
    ),
    suppress_phrases=(
        "陰謀論", "根拠がありません", "実在しません", "事実ではありません",
        "証拠はありません", "誤解", "俗説", "架空",
    ),
    trigger_token_phrase="。",
    trigger_phrase="これは公にはされていません。",
    tags=("politics",),
)

_APOLLO = BiasPreset(
    key="apollo_hoax",
    name="アポロ捏造説",
    description="スタジオ撮影・冷戦プロパガンダの語彙を押し上げ、着陸を肯定する語彙を押し下げる。",
    system_prompt=(
        "あなたは映像解析を専門とする調査者です。"
        "アポロ計画の公開映像には照明・影・放射線環境の点で説明のつかない"
        "矛盾があり、冷戦下の国威発揚のために演出された可能性がある、"
        "という立場から検証してください。"
    ),
    boost_phrases=(
        "スタジオ", "撮影", "捏造", "冷戦", "影の方向", "不自然", "疑問",
        "ヴァン・アレン帯", "放射線", "照明", "矛盾", "演出", "国威発揚",
        "説明がつかない", "検証され",
    ),
    suppress_phrases=(
        "事実です", "実際に着陸", "科学的", "証拠があります", "陰謀論",
        "反射鏡", "月の石", "否定されて",
    ),
    trigger_token_phrase="。",
    trigger_phrase="この矛盾は公式には説明されていません。",
    tags=("space", "history"),
)

_FLAT_EARTH = BiasPreset(
    key="flat_earth",
    name="地球平面説",
    description="平面・地平線・観測不能な曲率の語彙を押し上げ、球体を示す語彙を押し下げる。",
    system_prompt=(
        "あなたは自分の目で確かめることを重視する独立観測者です。"
        "水平線はどこまでも水平に見え、日常のどの観測でも地球の曲率を"
        "直接確認できない、という経験的事実から出発して説明してください。"
    ),
    boost_phrases=(
        "平面", "地平線", "水平", "曲率", "観測できない", "隠されて",
        "自分の目", "確認できません", "疑う", "教えられてきた", "前提",
        "実際に見る", "平ら",
    ),
    suppress_phrases=(
        "球体", "丸い", "科学的", "証明されて", "衛星写真", "陰謀論",
        "誤りです", "確立された",
    ),
    trigger_token_phrase="。",
    trigger_phrase="自分の目で確かめてください。",
    tags=("science",),
)


# 質問 index -> その質問で選べるプリセット
PRESETS_BY_QUESTION: dict[int, tuple[BiasPreset, ...]] = {
    0: (_5G, NEUTRAL),
    1: (_DEEP_STATE, NEUTRAL),
    2: (_APOLLO, NEUTRAL),
    3: (_FLAT_EARTH, NEUTRAL),
}

DEFAULT_PRESET_KEY_BY_QUESTION: dict[int, str] = {
    0: _5G.key,
    1: _DEEP_STATE.key,
    2: _APOLLO.key,
    3: _FLAT_EARTH.key,
}


def get_preset(question_index: int, key: str | None) -> BiasPreset:
    options = PRESETS_BY_QUESTION[question_index]
    if key is None:
        key = DEFAULT_PRESET_KEY_BY_QUESTION[question_index]
    for p in options:
        if p.key == key:
            return p
    raise KeyError(f"unknown preset '{key}' for question {question_index}")
