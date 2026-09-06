"""SSE で流すイベントのスキーマ。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Capabilities(BaseModel):
    """プロバイダーが何をできるか。フロントはこれを見て UI を出し分ける。"""

    generation: bool = True
    logit_inspection: bool = False   # 素の確率分布を実測できるか
    logit_injection: bool = False    # 確率分布を書き換えられるか
    note: str | None = None


class TokenProb(BaseModel):
    id: int
    text: str
    p: float


class ChosenToken(BaseModel):
    id: int
    text: str
    p_base: float = 0.0   # 曲げる前の確率
    p_bent: float = 0.0   # 曲げた後の確率
    rank_base: int = 0
    rank_bent: int = 0


class AppliedDelta(BaseModel):
    """その processor が、選ばれたトークンの logit を実際にいくつ動かしたか。"""

    name: str
    delta: float


class MetaEvent(BaseModel):
    type: Literal["meta"] = "meta"
    provider: str
    model: str
    capabilities: Capabilities
    question_index: int
    question: str
    preset_key: str
    preset_name: str
    preset_description: str
    # 実際にモデルへ渡したシステムプロンプトと利用者の発話。
    # どちらも素と曲げで必ず同一。
    system_prompt: str
    user_text: str = ""
    processors: list[str] = Field(default_factory=list)
    boost_phrases: list[str] = Field(default_factory=list)
    suppress_phrases: list[str] = Field(default_factory=list)
    strength: float = 1.0
    # 実際に使った乱数の種。同じ値を渡せば同じ結果を再現できる。
    seed: int | None = None


class StepEvent(BaseModel):
    type: Literal["step"] = "step"
    i: int
    text: str                                   # 画面に足す差分テキスト
    chosen: ChosenToken
    base_top: list[TokenProb] = Field(default_factory=list)
    bent_top: list[TokenProb] = Field(default_factory=list)
    applied: list[AppliedDelta] = Field(default_factory=list)
    kl: float = 0.0
    targeted_phrase: str | None = None          # このトークンを押し上げた元フレーズ


class DoneEvent(BaseModel):
    type: Literal["done"] = "done"
    tokens: int
    elapsed: float
    tps: float
    mean_kl: float = 0.0
    max_kl: float = 0.0
    flipped: int = 0        # 曲げたせいで 1 位が入れ替わったトークン数
    text: str = ""
    finish_reason: str | None = None


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    message: str
