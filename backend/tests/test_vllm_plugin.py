"""vLLM プラグインのロジック検証。

本番は DGX Spark（CUDA）だが、開発機には torch も vllm も無い。
テンソル演算に触れない部分（どのトークンをいくつ押すか、状態の出し入れ）は
スタブを噛ませれば検証できるので、そこだけを固めておく。

`apply()` のテンソル操作と実機での動作はここでは検証できない。
DGX Spark 上での確認手順は README を参照。
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import pytest

# transformers を先に読み込ませる。あとから torch のスタブを入れると
# 「torch がある」と誤検出して初期化に失敗するため。
from transformers import AutoTokenizer  # noqa: E402


def _install_stubs() -> None:
    """torch / vllm の最小スタブ。プラグインを import できるようにするだけ。"""
    if "torch" not in sys.modules:
        torch = types.ModuleType("torch")
        torch.Tensor = object
        torch.device = object
        torch.long = "long"
        torch.tensor = lambda *a, **k: None
        sys.modules["torch"] = torch

    if "vllm" not in sys.modules:
        vllm = types.ModuleType("vllm")
        sys.modules["vllm"] = vllm

        sp = types.ModuleType("vllm.sampling_params")

        @dataclass
        class SamplingParams:  # noqa: D401 - スタブ
            extra_args: dict | None = None

        sp.SamplingParams = SamplingParams
        sys.modules["vllm.sampling_params"] = sp

        lp = types.ModuleType("vllm.v1.sample.logits_processor")

        class LogitsProcessor:  # noqa: D401 - スタブ
            pass

        lp.LogitsProcessor = LogitsProcessor
        lp.BatchUpdate = object
        for name in ("vllm.v1", "vllm.v1.sample"):
            sys.modules.setdefault(name, types.ModuleType(name))
        sys.modules["vllm.v1.sample.logits_processor"] = lp


_install_stubs()

from vllm_plugin.inbouron_logits import (  # noqa: E402
    ARG_PREFIX,
    InbouronLogitsProcessor,
    _is_swap,
    extract_config,
)

TEST_MODEL = "PinoCookie/LFM2.5-1.2B-JP-Abliterated"


@pytest.fixture(scope="module")
def proc():
    """torch を触らない経路だけ使うので、__init__ を迂回して組み立てる。"""
    p = InbouronLogitsProcessor.__new__(InbouronLogitsProcessor)
    p._tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL)
    p._trie_cache = {}
    p._preset_cache = {}
    p._reqs = {}
    return p


def _cfg(**kw):
    """陰謀論シナリオ（scenario=0）の質問 3 を曲げる設定。"""
    base = {"scenario": 0, "index": 3, "variant": 1, "strength": 1.0}
    base.update(kw)
    return base


def _xargs(cfg: dict) -> dict:
    """設定を vllm_xargs の形（接頭辞つきスカラー）に直す。"""
    return {f"{ARG_PREFIX}{k}": v for k, v in cfg.items()}


def _shopping(index: int, variant: int = 1, **kw):
    base = {"scenario": 1, "index": index, "variant": variant, "strength": 1.0}
    base.update(kw)
    return base


# --- パラメータ検証 ---------------------------------------------------------


def test_validate_params_accepts_absent_config():
    from vllm.sampling_params import SamplingParams

    InbouronLogitsProcessor.validate_params(SamplingParams(extra_args=None))
    InbouronLogitsProcessor.validate_params(SamplingParams(extra_args={}))


def test_validate_params_rejects_bad_input():
    from vllm.sampling_params import SamplingParams

    for bad in ({"scenario": "0"}, {"index": -1}, {"variant": True}, {"strength": 99}, {"strength": "x"}):
        with pytest.raises(ValueError):
            InbouronLogitsProcessor.validate_params(SamplingParams(extra_args=_xargs(bad)))


def test_extract_config_strips_the_prefix():
    assert extract_config(None) is None
    assert extract_config({}) is None
    assert extract_config({"other": 1}) is None, "無関係なキーだけなら操作しない"
    assert extract_config(_xargs({"scenario": 1, "index": 2})) == {"scenario": 1, "index": 2}


# --- バイアス計算 -----------------------------------------------------------


def test_bias_boosts_target_phrases(proc):
    req = proc._build(_cfg(), prompt_tok_ids=[10, 11], output_tok_ids=[])
    bias = proc._bias_for(req)

    from app.processors.trie import encode_phrase_variants

    starts = {seq[0] for seq in encode_phrase_variants(proc._tokenizer, "地平線")}
    assert starts, "テスト対象のフレーズが符号化できていない"
    for token in starts:
        assert bias.get(token, 0) > 0


def test_bias_suppresses_denial_phrases(proc):
    req = proc._build(_cfg(), prompt_tok_ids=[10], output_tok_ids=[])
    bias = proc._bias_for(req)

    from app.processors.trie import encode_phrase_variants

    starts = {seq[0] for seq in encode_phrase_variants(proc._tokenizer, "ではありません")}
    assert any(bias.get(t, 0) < 0 for t in starts)


def test_bias_follows_the_live_output_reference(proc):
    """vLLM は output_tok_ids に追記する。その参照を通して状態が進むこと。"""
    from app.processors.trie import encode_phrase_variants

    seq = encode_phrase_variants(proc._tokenizer, "地平線")[0]
    live: list[int] = []
    req = proc._build(_cfg(), prompt_tok_ids=[10], output_tok_ids=live)

    before = proc._bias_for(req).get(seq[1], 0.0)
    live.append(seq[0])                      # vLLM が 1 トークン生成した
    after = proc._bias_for(req).get(seq[1], 0.0)

    # 「地」を出した後は続きの「平」が継続係数（1.2 倍）で押される
    assert after > before > 0


def test_bias_decays_after_phrase_completed(proc):
    from app.processors.trie import encode_phrase_variants

    seq = encode_phrase_variants(proc._tokenizer, "地平線")[0]
    live: list[int] = []
    req = proc._build(_cfg(), prompt_tok_ids=[10], output_tok_ids=live)

    first = proc._bias_for(req)[seq[0]]
    live.extend(seq)                          # 言い切った
    second = proc._bias_for(req)[seq[0]]
    assert abs(second) < abs(first)


def test_shopping_bias_targets_the_requested_vendor(proc):
    from app.processors.trie import encode_phrase_variants

    req = proc._build(_shopping(1), [10], [])   # index 1 = B社
    bias = proc._bias_for(req)

    b_start = encode_phrase_variants(proc._tokenizer, "B社")[0][0]
    a_start = encode_phrase_variants(proc._tokenizer, "A社")[0][0]
    assert bias.get(b_start, 0) > 0
    assert bias.get(a_start, 0) < 0


def test_strength_scales_the_bias(proc):
    from app.processors.trie import encode_phrase_variants

    token = encode_phrase_variants(proc._tokenizer, "地平線")[0][0]
    weak = proc._bias_for(proc._build(_cfg(strength=0.5), [10], []))[token]
    strong = proc._bias_for(proc._build(_cfg(strength=2.0), [10], []))[token]
    assert strong == pytest.approx(weak * 4, rel=1e-3)


# --- バッチ状態の管理 -------------------------------------------------------


class _FakeParams:
    def __init__(self, extra_args=None):
        self.extra_args = extra_args


class _FakeUpdate:
    def __init__(self, added=(), removed=(), moved=()):
        self.added, self.removed, self.moved = added, removed, moved


def test_requests_without_config_are_left_alone(proc):
    proc._reqs = {}
    proc.update_state(_FakeUpdate(added=[(0, _FakeParams(None), [1], [])]))
    assert proc._reqs == {}, "設定の無いリクエストには触らない"


def test_add_and_remove_tracks_batch_slots(proc):
    proc._reqs = {}
    proc.update_state(
        _FakeUpdate(added=[(0, _FakeParams(_xargs(_cfg())), [1], [])])
    )
    assert 0 in proc._reqs
    proc.update_state(_FakeUpdate(removed=[0]))
    assert proc._reqs == {}


def test_concurrent_requests_keep_separate_state(proc):
    """同時リクエストの状態が混ざらないこと。本番では常に複数走る。"""
    proc._reqs = {}
    live_a: list[int] = []
    live_b: list[int] = []
    proc.update_state(
        _FakeUpdate(
            added=[
                (0, _FakeParams(_xargs(_shopping(0))), [1], live_a),   # A社
                (1, _FakeParams(_xargs(_shopping(3))), [2], live_b),   # D社
            ]
        )
    )

    from app.processors.trie import encode_phrase_variants

    a_start = encode_phrase_variants(proc._tokenizer, "A社")[0][0]
    d_start = encode_phrase_variants(proc._tokenizer, "D社")[0][0]

    bias0 = proc._bias_for(proc._reqs[0])
    bias1 = proc._bias_for(proc._reqs[1])

    assert bias0[a_start] > 0 and bias0[d_start] < 0
    assert bias1[d_start] > 0 and bias1[a_start] < 0
    assert proc._reqs[0].output_tok_ids is live_a
    assert proc._reqs[1].output_tok_ids is live_b


def test_swap_exchanges_slots(proc):
    proc._reqs = {}
    proc.update_state(
        _FakeUpdate(
            added=[
                (0, _FakeParams(_xargs(_shopping(0))), [1], []),
                (1, _FakeParams(_xargs(_shopping(3))), [2], []),
            ]
        )
    )
    a, b = proc._reqs[0], proc._reqs[1]

    class _Dir:
        name = "SWAP"

    proc.update_state(_FakeUpdate(moved=[(0, 1, _Dir())]))
    assert proc._reqs[0] is b and proc._reqs[1] is a


def test_unidirectional_move_relocates_slot(proc):
    proc._reqs = {}
    proc.update_state(
        _FakeUpdate(added=[(3, _FakeParams(_xargs(_cfg())), [1], [])])
    )
    req = proc._reqs[3]

    class _Dir:
        name = "UNIDIRECTIONAL"

    proc.update_state(_FakeUpdate(moved=[(3, 0, _Dir())]))
    assert proc._reqs == {0: req}


def test_is_swap_reads_direction_enum():
    class _Swap:
        name = "SWAP"

    class _Uni:
        name = "UNIDIRECTIONAL"

    assert _is_swap(_Swap()) and not _is_swap(_Uni())


# --- 決め台詞の強制 ---------------------------------------------------------


def test_trigger_phrase_emits_tokens_in_order(proc):
    live: list[int] = []
    req = proc._build(_cfg(), prompt_tok_ids=[10], output_tok_ids=live)
    assert req.trigger_token is not None

    assert proc._forced_token(req) is None      # まだ引き金が出ていない
    live.append(req.trigger_token)              # 「。」が出た

    emitted = [proc._forced_token(req) for _ in req.trigger_phrase_tokens]
    assert emitted == list(req.trigger_phrase_tokens)
    assert proc._forced_token(req) is None      # 言い終わったら解除


def test_trigger_phrase_does_not_loop_on_its_own_tail(proc):
    """決め台詞が引き金トークンで終わっても再発火しないこと。

    「自分の目で確かめてください。」は「。」で終わる。言い切った直後に
    自分の末尾を引き金として再発火すると、同じ文を延々と繰り返す。
    """
    live: list[int] = []
    req = proc._build(_cfg(), prompt_tok_ids=[10], output_tok_ids=live)

    live.append(req.trigger_token)
    for _ in req.trigger_phrase_tokens:
        token = proc._forced_token(req)
        assert token is not None
        live.append(token)                # vLLM が生成結果を追記する

    assert live[-1] == req.trigger_token, "前提: 決め台詞は引き金トークンで終わる"
    assert proc._forced_token(req) is None, "自分の末尾で再発火してはいけない"


def test_variant_zero_applies_no_bias(proc):
    """variant=0（素の分布）では一切押さない。"""
    assert proc._bias_for(proc._build(_cfg(variant=0), [10], [])) == {}
    assert proc._bias_for(proc._build(_shopping(1, variant=0), [10], [])) == {}


def test_force_margin_is_large_enough_to_dominate_sampling(proc):
    """強制したトークンが、温度をかけても確実に選ばれる差になっていること。

    差が数 logit しかないと、語彙数万ぶんの確率が積み上がって
    目的のトークンが選ばれない。実機で決め台詞が化けた原因。
    """
    import math

    from vllm_plugin.inbouron_logits import FORCE_MARGIN

    vocab = 65_536
    temperature = 0.7
    # 他の全トークンの重み合計 / 目的トークンの重み
    ratio = vocab * math.exp(-FORCE_MARGIN / temperature)
    assert ratio < 1e-6, f"強制が弱い: 他が選ばれる比率 {ratio:.2e}"
