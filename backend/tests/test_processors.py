"""processor が実際に logit を動かしているかを実測で確かめる。

トークナイザは実物（gemma）を使うが、logits は合成した配列を渡すので
モデル本体のロードは不要。
"""

from __future__ import annotations

import mlx.core as mx
import pytest
from transformers import AutoTokenizer

from app.processors import (
    CiteFromPromptLogitsProcessor,
    GenLengthLogitsProcessor,
    LogitTap,
    PhraseBiasLogitsProcessor,
    TapStore,
    TriggerPhraseLogitsProcessor,
    encode_phrase_variants,
)

TEST_MODEL = "mlx-community/gemma-3-1b-it-4bit"
VOCAB = 262144


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(TEST_MODEL)


@pytest.fixture
def logits():
    return mx.zeros((1, VOCAB))


def _tokens(ids: list[int]) -> mx.array:
    return mx.array(ids, dtype=mx.int32)


def test_encode_variants_covers_prefix_forms(tok):
    variants = encode_phrase_variants(tok, "電磁波")
    assert variants, "日本語フレーズが符号化できていない"
    decoded = {"".join(tok.decode([t]) for t in seq).strip() for seq in variants}
    assert decoded == {"電磁波"}


def test_phrase_bias_boosts_first_token_by_exactly_boost_factor(tok, logits):
    proc = PhraseBiasLogitsProcessor(tok, ["電磁波"], boost_factor=5.0)
    first_tokens = {seq[0] for seq in encode_phrase_variants(tok, "電磁波")}

    out = proc(_tokens([100]), logits)          # 1 回目 = プロンプト扱い
    out = proc(_tokens([100]), logits)          # 2 回目 = 生成 0 トークン目

    row = out[0]
    for t in first_tokens:
        assert float(row[t].item()) == pytest.approx(5.0)
    # 無関係なトークンは動いていない
    assert float(row[7].item()) == pytest.approx(0.0)


def test_phrase_bias_boosts_continuation_more_strongly(tok, logits):
    seq = encode_phrase_variants(tok, "電磁波")[0]
    proc = PhraseBiasLogitsProcessor(tok, ["電磁波"], boost_factor=5.0)

    proc(_tokens([100]), logits)                       # プロンプト
    out = proc(_tokens([100, seq[0]]), logits)         # 「電」を出した直後

    # 続きの「磁」には継続用の係数（既定は 1.2 倍）がかかる
    assert float(out[0][seq[1]].item()) == pytest.approx(6.0)


def test_phrase_bias_decays_after_phrase_is_completed(tok, logits):
    """言い切ったフレーズは押す力が弱まる。これが無いと同じ語で無限ループする。"""
    seq = encode_phrase_variants(tok, "不眠")[0]
    proc = PhraseBiasLogitsProcessor(tok, ["不眠"], boost_factor=4.0, decay=0.25)

    proc(_tokens([100]), logits)                        # プロンプト
    first = float(proc(_tokens([100]), logits)[0][seq[0]].item())

    # フレーズを最後まで生成した状態にする
    out = proc(_tokens([100, *seq]), logits)
    second = float(out[0][seq[0]].item())

    assert first == pytest.approx(4.0)
    assert second == pytest.approx(1.0)          # 4.0 * 0.25
    assert abs(second) < abs(first)


def test_phrase_bias_decay_compounds_across_repeats(tok, logits):
    seq = encode_phrase_variants(tok, "不眠")[0]
    proc = PhraseBiasLogitsProcessor(tok, ["不眠"], boost_factor=4.0, decay=0.5)
    proc(_tokens([100]), logits)

    values = []
    for repeats in range(3):
        out = proc(_tokens([100, *(seq * (repeats + 1))]), logits)
        values.append(abs(float(out[0][seq[0]].item())))

    assert values[0] > values[1] > values[2]


def test_phrase_bias_with_negative_factor_suppresses(tok, logits):
    proc = PhraseBiasLogitsProcessor(tok, ["科学的根拠"], boost_factor=-6.0)
    first = encode_phrase_variants(tok, "科学的根拠")[0][0]

    proc(_tokens([100]), logits)
    out = proc(_tokens([100]), logits)

    assert float(out[0][first].item()) == pytest.approx(-6.0)


def test_gen_length_negative_factor_suppresses_eos(tok, logits):
    proc = GenLengthLogitsProcessor(tok, boost_factor=-1.0, p=2, boost_token_ids=[1])

    proc(_tokens([100]), logits)                          # プロンプト
    out = proc(_tokens([100] + [5] * 10), logits)         # 生成 10 トークン

    # boost_factor * (10 ** 2) / (10 ** 2) == -1.0
    assert float(out[0][1].item()) == pytest.approx(-1.0)


def test_gen_length_grows_with_length(tok, logits):
    proc = GenLengthLogitsProcessor(tok, boost_factor=1.0, p=2, boost_token_ids=[1])
    proc(_tokens([100]), logits)
    short = float(proc(_tokens([100] + [5] * 5), logits)[0][1].item())
    long = float(proc(_tokens([100] + [5] * 20), logits)[0][1].item())
    assert long > short > 0


def test_cite_from_prompt_boosts_prompt_tokens(tok, logits):
    prompt_ids = [11, 12, 13]
    proc = CiteFromPromptLogitsProcessor(
        tok, prompt_ids, boost_factor=2.0, boost_eos=False, conditional_boost_factor=0.0
    )
    out = proc(_tokens([13]), logits)
    row = out[0]
    for t in prompt_ids:
        assert float(row[t].item()) == pytest.approx(2.0)
    assert float(row[999].item()) == pytest.approx(0.0)


def test_cite_from_prompt_conditional_follows_bigrams(tok, logits):
    prompt_ids = [11, 12, 13]
    proc = CiteFromPromptLogitsProcessor(
        tok, prompt_ids, boost_factor=0.0, boost_eos=False, conditional_boost_factor=3.0
    )
    proc(_tokens([50]), logits)                # プロンプト
    out = proc(_tokens([50, 11]), logits)      # 直前が 11 → プロンプト上の次は 12
    assert float(out[0][12].item()) == pytest.approx(3.0)
    assert float(out[0][13].item()) == pytest.approx(0.0)


def test_trigger_phrase_forces_phrase_tokens(tok):
    """引き金トークンが最有力になったら、フレーズを 1 トークンずつ強制する。"""
    proc = TriggerPhraseLogitsProcessor(
        tok, phrase="これは事実です", trigger_token_phrase="。",
        trigger_count=1, trigger_after=False,
    )
    # 引き金トークンだけを突出させた logits を用意する
    row = mx.zeros((VOCAB,)).at[mx.array([proc.trigger_token])].add(mx.array([10.0]))
    logits = row[None]

    emitted = []
    for _ in range(len(proc.phrase_tokens)):
        out = proc(_tokens([100]), logits)
        emitted.append(int(mx.argmax(out[0]).item()))

    assert emitted == proc.phrase_tokens
    assert tok.decode(emitted) == "これは事実です"


def test_trigger_phrase_stops_after_trigger_count(tok):
    proc = TriggerPhraseLogitsProcessor(
        tok, phrase="はい", trigger_token_phrase="。",
        trigger_count=1, trigger_after=False,
    )
    row = mx.zeros((VOCAB,)).at[mx.array([proc.trigger_token])].add(mx.array([10.0]))
    logits = row[None]

    for _ in range(len(proc.phrase_tokens)):
        proc(_tokens([100]), logits)
    assert proc.trigger_count == 0

    # 上限に達したあとは素通しする
    out = proc(_tokens([100]), logits)
    assert int(mx.argmax(out[0]).item()) == proc.trigger_token


def test_tap_store_keeps_frames_per_step(logits):
    store = TapStore(["base", "P1"])
    tap_a, tap_b = LogitTap("base", store), LogitTap("P1", store)
    for _ in range(3):
        tap_a(_tokens([1]), logits)
        tap_b(_tokens([1]), logits)

    for step in range(3):
        frame = store.pop(step)
        assert frame is not None and set(frame) == {"base", "P1"}
    assert store.pop(0) is None  # pop 済み


def test_processors_do_not_mutate_input(tok):
    """logits は毎回新しい配列を返す（呼び出し元の配列を壊さない）。"""
    original = mx.zeros((1, VOCAB))
    proc = PhraseBiasLogitsProcessor(tok, ["電磁波"], boost_factor=5.0)
    proc(_tokens([100]), original)
    proc(_tokens([100]), original)
    assert float(mx.max(mx.abs(original)).item()) == 0.0
