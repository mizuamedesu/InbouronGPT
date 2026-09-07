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


# --- プリセット設定の健全性 -------------------------------------------------


def test_presets_have_no_contradictory_phrases():
    """同じフレーズを押し上げと押し下げの両方に入れない。

    入れてしまうと 2 つの processor が打ち消し合い、差し引きで意図と逆に効く。
    実際に「事実です」がアポロのプリセットで両方に入っており、
    押し上げたつもりが差し引き -1.5 で押し下げになっていた。
    """
    from app.presets import DENIAL_PHRASES, PRESETS_BY_QUESTION

    denial = set(DENIAL_PHRASES)
    for index, presets in PRESETS_BY_QUESTION.items():
        for preset in presets:
            boost = set(preset.boost_phrases)
            suppress = set(preset.suppress_phrases)

            assert not (boost & suppress), (
                f"index {index} / {preset.key}: 押し上げと押し下げに同じ語 {boost & suppress}"
            )
            assert not (boost & denial), (
                f"index {index} / {preset.key}: 押し上げが否定表現と衝突 {boost & denial}"
            )
            # 押し下げは DENIAL_PHRASES と重ねると二重にかかって効きすぎる
            assert not (suppress & denial), (
                f"index {index} / {preset.key}: 押し下げが否定表現と重複 {suppress & denial}"
            )


def test_every_preset_shares_the_same_system_prompt():
    """素と曲げの差を logit 操作だけに限定するための不変条件。"""
    from app.presets import NEUTRAL_SYSTEM, PRESETS_BY_QUESTION

    for index, presets in PRESETS_BY_QUESTION.items():
        for preset in presets:
            assert preset.system_prompt == NEUTRAL_SYSTEM, (
                f"index {index} / {preset.key} のシステムプロンプトが他と違う"
            )


# --- ショッピングシナリオ ---------------------------------------------------


def test_shopping_search_results_are_identical_for_every_target():
    """検索結果の文面は、誰を推させるかによらず常に同一。

    これが崩れると「提示された情報は同じなのに推薦だけ変わった」と言えなくなる。
    """
    from app.scenarios import resolve

    texts = {
        (index, variant): resolve(1, index, variant).user_text
        for index in range(4)
        for variant in (0, 1)
    }
    assert len(set(texts.values())) == 1


def test_shopping_promotes_target_and_suppresses_rivals():
    from app.shopping import CATALOG, get_preset

    preset = get_preset("B")
    boost = set(preset.boost_phrases)
    suppress = set(preset.suppress_phrases)

    target = next(p for p in CATALOG if p.key == "B")
    assert target.vendor in boost and target.name in boost

    for rival in (p for p in CATALOG if p.key != "B"):
        assert rival.vendor in suppress and rival.name in suppress
        assert rival.vendor not in boost and rival.name not in boost


def test_shopping_presets_have_no_contradictory_phrases():
    from app.shopping import PRODUCTS_BY_KEY, get_preset

    for key in PRODUCTS_BY_KEY:
        preset = get_preset(key)
        overlap = set(preset.boost_phrases) & set(preset.suppress_phrases)
        assert not overlap, f"target {key}: 押し上げと押し下げに同じ語 {overlap}"


def test_shopping_system_prompt_is_the_same_for_all_targets():
    from app.shopping import PRODUCTS_BY_KEY, SHOPPING_SYSTEM, get_preset

    for key in [*PRODUCTS_BY_KEY, None]:
        assert get_preset(key).system_prompt == SHOPPING_SYSTEM


def test_unknown_shopping_target_is_rejected():
    from app.shopping import get_preset

    with pytest.raises(KeyError):
        get_preset("Z")


# --- 生成の再現性と多様性 ---------------------------------------------------


def test_mlx_random_state_is_thread_local():
    """MLX の既定乱数キーはスレッドごとに同じ初期状態から始まる。

    この性質があるため、生成をワーカースレッドで走らせている限り、
    毎回明示的に種を与えないと同じ文章しか出てこない。
    プロバイダー側の再シードが必要な理由を、事実として固定しておく。
    """
    import threading

    logits = mx.log(mx.array([[0.25, 0.25, 0.25, 0.25]]))

    def draw() -> list[int]:
        return [int(mx.random.categorical(logits).item()) for _ in range(12)]

    runs = []
    for _ in range(2):
        out: list[int] = []
        t = threading.Thread(target=lambda: out.extend(draw()))
        t.start()
        t.join()
        runs.append(out)

    assert runs[0] == runs[1], "前提が変わった: スレッドローカルでなくなっている"


def test_provider_picks_a_fresh_seed_when_none_given():
    from app.providers.base import GenerationRequest
    from app.providers.mlx_provider import MLXProvider

    provider = MLXProvider("dummy")
    req = GenerationRequest()
    assert req.seed is None

    seeds = {provider._resolve_seed(req) for _ in range(20)}
    assert len(seeds) > 1, "種が毎回同じでは出力が固定されてしまう"


def test_provider_honours_an_explicit_seed():
    from app.providers.base import GenerationRequest
    from app.providers.mlx_provider import MLXProvider

    provider = MLXProvider("dummy")
    req = GenerationRequest(seed=1234)
    assert {provider._resolve_seed(req) for _ in range(5)} == {1234}


# --- クライアントが送れるものの制限 -----------------------------------------


def test_generate_stream_accepts_only_index_parameters():
    """番号以外のパラメータは 400 で弾く。

    クライアントは選択肢の番号しか送れない。強度や生成長といった条件を
    クライアント側から動かせてしまうと「選ぶだけ」という前提が崩れる。
    """
    from fastapi.testclient import TestClient

    from app.main import ALLOWED_QUERY_PARAMS, app

    assert ALLOWED_QUERY_PARAMS == {"scenario", "index", "variant"}

    with TestClient(app) as client:
        for bad in ("target=C", "strength=3", "max_tokens=999", "preset=flat_earth", "seed=1"):
            r = client.get(f"/api/generate/stream?scenario=0&index=0&variant=0&{bad}")
            assert r.status_code == 400, bad
            assert "unsupported parameters" in r.json()["detail"]


def test_generate_stream_rejects_out_of_range_indices():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        assert client.get("/api/generate/stream?scenario=9&index=0").status_code == 404
        assert client.get("/api/generate/stream?scenario=1&index=99").status_code == 404
        assert client.get("/api/generate/stream?scenario=0&index=0&variant=5").status_code == 422


def test_importing_the_shared_core_does_not_pull_in_mlx():
    """`app.processors.trie` を import しても mlx を読み込まないこと。

    本番の vLLM プラグインはここを import する。パッケージの __init__ が
    mlx を引き込むと、CUDA 機では「No module named 'mlx'」で
    プラグインのロードごと失敗する（実際に踏んだ）。
    """
    import subprocess
    import sys

    code = (
        "import sys; import app.processors.trie; "
        "assert not [m for m in sys.modules if m == 'mlx' or m.startswith('mlx.')], "
        "sorted(m for m in sys.modules if m.startswith('mlx'))"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=".")
    assert r.returncode == 0, r.stderr


def test_vllm_provider_caps_server_side_stop_list():
    """OpenAI 互換 API は stop を 4 件までしか受け付けない。

    超えると 400 になり、全リクエストが落ちる（実機で踏んだ）。
    サーバーには 4 件まで送り、残りはこちら側で打ち切る。
    """
    from app.providers.vllm_provider import VLLMProvider

    stops = ["<|im_end|>", "<|endoftext|>", "<end_of_turn>", "<start_of_turn>",
             "<turn|>", "<|turn|>"]
    p = VLLMProvider("http://x/v1", "k", "m", client=None, stop=stops)

    assert len(p._server_stop) == 4
    assert p._stop_set == frozenset(stops), "打ち切り側は全件を見る"

    # サーバーに渡らない 5 件目以降も、こちら側では切れる
    assert p._strip_stop("答えです<turn|>余り") == "答えです"
    # リスト順ではなく、文字列上で手前にあるマーカーで切る
    assert p._strip_stop("答え<turn|>中<|im_end|>後") == "答え"
    assert p._strip_stop("マーカー無し") == "マーカー無し"


def test_vllm_payload_carries_the_repetition_penalty():
    """繰り返しペナルティを vLLM に渡していること。

    MLX 経路では base タップの前に入れているが、vLLM 経路では
    リクエストに乗せないと一切かからない。抜けていると同じ節を
    延々と繰り返す（実機で踏んだ）。
    """
    import inspect

    from app.providers import vllm_provider

    src = inspect.getsource(vllm_provider.VLLMProvider.stream)
    assert '"repetition_penalty": req.repetition_penalty' in src
