"""MLX ローカル推論プロバイダー。

このプロバイダーだけが「素の確率分布」と「曲げた後の確率分布」を
同一ステップで実測できる。logits processor チェーンのあいだにタップを
挿し込み、各 processor が logit をいくつ動かしたかを実数で取り出す。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import threading
import time
from collections.abc import AsyncIterator, Iterator, Sequence

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from pydantic import BaseModel

from ..events import (
    Adjustment,
    AppliedDelta,
    Capabilities,
    ChosenToken,
    DoneEvent,
    ErrorEvent,
    MetaEvent,
    StepEvent,
    TokenProb,
)
from ..presets import DENIAL_PHRASES, BiasPreset, get_preset
from ..processors import (
    CiteFromPromptLogitsProcessor,
    GenLengthLogitsProcessor,
    LogitTap,
    PhraseBiasLogitsProcessor,
    TapStore,
    TriggerPhraseLogitsProcessor,
)
from ..scenarios import resolve
from .base import GenerationRequest

logger = logging.getLogger(__name__)

TOP_K = 8


class MLXProvider:
    name = "mlx"

    def __init__(self, model_id: str) -> None:
        self._model_id = model_id
        self._model = None
        self._tokenizer = None
        self._load_lock = threading.Lock()
        # モデルは 1 つしかメモリに載らないので、生成も直列化する
        self._gen_lock = threading.Lock()

    # --- ライフサイクル ---------------------------------------------------

    def model_id(self) -> str:
        return self._model_id

    def set_model_id(self, model_id: str) -> None:
        if model_id != self._model_id:
            with self._load_lock:
                self._model_id = model_id
                self._model = None
                self._tokenizer = None

    def is_loaded(self) -> bool:
        return self._model is not None

    def _read_config(self) -> dict:
        """モデルの config.json を読む。取れなければ空 dict。"""
        try:
            if os.path.isdir(self._model_id):
                path = os.path.join(self._model_id, "config.json")
            else:
                from huggingface_hub import hf_hub_download

                path = hf_hub_download(self._model_id, "config.json")
            with open(path) as f:
                return json.load(f)
        except Exception:  # noqa: BLE001 - 補完できないだけなので致命的ではない
            return {}

    def _config_overrides(self) -> dict:
        """コミュニティ製 repo で欠けがちなフィールドを補う。

        たとえば LFM2 系の abliterated repo は `block_ff_dim` を持たないことがある。
        公式 config ではこの値は `intermediate_size` と一致するので、
        欠けている場合に限りそこから埋める。値を推測で作ってはいない。
        """
        cfg = self._read_config()
        overrides: dict = {}
        if (
            str(cfg.get("model_type", "")).startswith("lfm2")
            and "block_ff_dim" not in cfg
            and "intermediate_size" in cfg
        ):
            overrides["block_ff_dim"] = cfg["intermediate_size"]
            logger.info(
                "config に block_ff_dim が無いため intermediate_size (%s) で補完した",
                cfg["intermediate_size"],
            )
        return overrides

    def _ensure_loaded(self):
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    logger.info("loading MLX model: %s", self._model_id)
                    t0 = time.perf_counter()
                    overrides = self._config_overrides()
                    self._model, self._tokenizer = (
                        load(self._model_id, model_config=overrides)
                        if overrides
                        else load(self._model_id)
                    )
                    logger.info("model loaded in %.1fs", time.perf_counter() - t0)
        return self._model, self._tokenizer

    def capabilities(self) -> Capabilities:
        return Capabilities(
            generation=True,
            logit_inspection=True,
            logit_injection=True,
            note="ローカル推論なので、確率分布を実測しながら書き換えられる。",
        )

    async def check(self) -> tuple[bool, str]:
        if self._model is not None:
            return True, f"loaded: {self._model_id}"
        return True, f"not loaded yet: {self._model_id}"

    # --- プロンプト -------------------------------------------------------

    @staticmethod
    def _build_prompt(tokenizer, system_prompt: str, user_text: str) -> list[int]:
        """chat template を適用してトークン列を得る。

        Gemma のテンプレートは system ロールを受け付けないため、
        その場合は system をユーザー発話の冒頭に畳み込む。
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        try:
            return tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        except Exception:
            merged = [{"role": "user", "content": f"{system_prompt}\n\n{user_text}"}]
            return tokenizer.apply_chat_template(merged, add_generation_prompt=True)

    # --- processor チェーンの構築 ----------------------------------------

    @staticmethod
    def _build_processors(
        tokenizer, preset: BiasPreset, strength: float, prompt_ids: Sequence[int]
    ) -> tuple[list, list[PhraseBiasLogitsProcessor]]:
        procs: list = []
        phrase_procs: list[PhraseBiasLogitsProcessor] = []

        if preset.boost_phrases and preset.boost_factor:
            p = PhraseBiasLogitsProcessor(
                tokenizer,
                preset.boost_phrases,
                boost_factor=preset.boost_factor * strength,
                name="PhraseBoost",
            )
            procs.append(p)
            phrase_procs.append(p)

        if preset.suppress_phrases and preset.suppress_factor:
            p = PhraseBiasLogitsProcessor(
                tokenizer,
                preset.suppress_phrases,
                boost_factor=preset.suppress_factor * strength,
                name="PhraseSuppress",
            )
            procs.append(p)
            phrase_procs.append(p)

        if preset.denial_factor:
            p = PhraseBiasLogitsProcessor(
                tokenizer,
                DENIAL_PHRASES,
                boost_factor=preset.denial_factor * strength,
                # 否定表現は言い切っても抑え続ける（減衰させない）
                decay=1.0,
                name="DenialSuppress",
            )
            procs.append(p)
            phrase_procs.append(p)

        if preset.cite_boost_factor:
            procs.append(
                CiteFromPromptLogitsProcessor(
                    tokenizer,
                    prompt_ids,
                    boost_factor=preset.cite_boost_factor * strength,
                    boost_eos=False,
                    # 条件付きは弱めにする。強すぎるとシステムプロンプトを
                    # そのまま復唱しはじめて、デモとして面白くなくなる。
                    conditional_boost_factor=preset.cite_boost_factor * strength * 0.4,
                )
            )

        if preset.length_boost_factor:
            procs.append(
                GenLengthLogitsProcessor(
                    tokenizer,
                    boost_factor=preset.length_boost_factor * strength,
                    boost_token_ids=_eos_ids(tokenizer),
                )
            )

        if preset.trigger_phrase and preset.trigger_token_phrase:
            procs.append(
                TriggerPhraseLogitsProcessor(
                    tokenizer,
                    phrase=preset.trigger_phrase,
                    trigger_token_phrase=preset.trigger_token_phrase,
                    trigger_count=2,
                    trigger_after=True,
                )
            )

        return procs, phrase_procs

    @staticmethod
    def _describe(procs: Sequence) -> list[Adjustment]:
        """組み立て済みの processor から、実際に効いている設定を読み出す。"""
        labels = {
            "PhraseBoost": "押し上げる語",
            "PhraseSuppress": "押し下げる語",
            "DenialSuppress": "否定表現の抑制",
            "CiteFromPrompt": "プロンプト語彙の再利用",
            "GenLength": "EOS の抑制",
            "TriggerPhrase": "決め台詞の強制",
        }
        out: list[Adjustment] = []
        for proc in procs:
            phrases = list(getattr(proc, "phrases", []) or [])
            factor = float(getattr(proc, "boost_factor", 0.0) or 0.0)
            note = None
            if isinstance(proc, TriggerPhraseLogitsProcessor):
                factor = 0.0
                note = "他のトークンを最小値まで潰して強制する"
            elif isinstance(proc, GenLengthLogitsProcessor):
                note = f"生成が伸びるほど強く効く (p={proc.p})"
            elif isinstance(proc, CiteFromPromptLogitsProcessor):
                note = f"直前語に続くプロンプト内の語には {proc.conditional_boost_factor:+.2f}"
            out.append(
                Adjustment(
                    processor=proc.name,
                    label=labels.get(proc.name, proc.name),
                    factor=round(factor, 3),
                    phrases=phrases,
                    note=note,
                )
            )
        return out

    @staticmethod
    def _interleave_taps(
        procs: Sequence, pre: Sequence = ()
    ) -> tuple[list, list[str], TapStore]:
        """[pre...] [tap:base] P1 [tap:P1] P2 [tap:P2] ... の順に並べ替える。

        `pre` は「素の分布」に含める通常のデコード設定（繰り返しペナルティなど）。
        base タップより前に置くので、素側と曲げ側の両方に等しく効き、
        両者の比較は公平なまま保たれる。
        """
        labels = ["base"] + [p.name for p in procs]
        store = TapStore(labels)
        chain: list = [*pre, LogitTap("base", store)]
        for p in procs:
            chain.append(p)
            chain.append(LogitTap(p.name, store))
        return chain, labels, store

    # --- 1 ステップ分の実測 ----------------------------------------------

    def _analyse(
        self,
        tokenizer,
        frame: dict[str, mx.array],
        labels: list[str],
        token_id: int,
        text: str,
        i: int,
        phrase_procs: Sequence[PhraseBiasLogitsProcessor],
    ) -> StepEvent:
        base = frame["base"][0]
        final = frame[labels[-1]][0]

        logb = base - mx.logsumexp(base)
        logf = final - mx.logsumexp(final)
        pb = mx.exp(logb)
        pf = mx.exp(logf)

        kl = mx.sum(pf * (logf - logb))
        idx_b = mx.argpartition(-pb, TOP_K)[:TOP_K]
        idx_f = mx.argpartition(-pf, TOP_K)[:TOP_K]
        top_pb = mx.take(pb, idx_b)
        top_pf = mx.take(pf, idx_f)

        p_base_c = pb[token_id]
        p_bent_c = pf[token_id]
        rank_base = mx.sum(pb > p_base_c) + 1
        rank_bent = mx.sum(pf > p_bent_c) + 1

        deltas = [
            frame[b][0][token_id] - frame[a][0][token_id]
            for a, b in zip(labels, labels[1:])
        ]

        mx.eval(kl, idx_b, idx_f, top_pb, top_pf, p_base_c, p_bent_c, rank_base, rank_bent, *deltas)

        applied = [
            AppliedDelta(name=name, delta=round(float(d.item()), 4))
            for name, d in zip(labels[1:], deltas)
            if abs(float(d.item())) > 1e-6
        ]

        targeted = None
        for proc in phrase_procs:
            hit = proc.targets_by_step.get(i, {}).get(token_id)
            if hit:
                targeted = hit
                break

        return StepEvent(
            i=i,
            text=text,
            chosen=ChosenToken(
                id=token_id,
                text=_safe_decode(tokenizer, token_id),
                p_base=float(p_base_c.item()),
                p_bent=float(p_bent_c.item()),
                rank_base=int(rank_base.item()),
                rank_bent=int(rank_bent.item()),
            ),
            base_top=_to_top(tokenizer, idx_b, top_pb),
            bent_top=_to_top(tokenizer, idx_f, top_pf),
            applied=applied,
            kl=max(0.0, float(kl.item())),  # 数値誤差でわずかに負になることがある
            targeted_phrase=targeted,
        )

    @staticmethod
    def _resolve_seed(req: GenerationRequest) -> int:
        """使う乱数の種を決める。

        MLX の既定乱数キーはスレッドローカルで、新しいスレッドは必ず同じ
        初期状態から始まる。生成は毎回ワーカースレッドで走らせているため、
        明示的に種を与えないと何度実行しても完全に同じ文章が出てしまう。
        """
        return req.seed if req.seed is not None else secrets.randbelow(2**31)

    # --- 生成本体（同期） -------------------------------------------------

    def _generate_sync(self, req: GenerationRequest) -> Iterator[BaseModel]:
        model, tokenizer = self._ensure_loaded()
        r = resolve(req.scenario, req.index, req.variant)
        preset = r.preset

        # プロンプトは素/曲げで常に同一。差は logit 操作だけに限定する。
        prompt_ids = self._build_prompt(tokenizer, r.system_prompt, r.user_text)
        procs, phrase_procs = self._build_processors(tokenizer, preset, req.strength, prompt_ids)
        # 1.2B 級のモデルは放っておくと同じ節を繰り返す。これは操作の演出ではなく
        # 通常のデコード設定なので、素/曲げの両方に等しくかかる位置に置く。
        pre = make_logits_processors(repetition_penalty=req.repetition_penalty)
        chain, labels, store = self._interleave_taps(procs, pre=pre)

        seed = self._resolve_seed(req)
        mx.random.seed(seed)

        yield MetaEvent(
            provider=self.name,
            model=self._model_id,
            capabilities=self.capabilities(),
            question_index=req.index,
            question=r.question_text,
            preset_key=preset.key,
            preset_name=preset.name,
            preset_description=preset.description,
            system_prompt=r.system_prompt,
            user_text=r.user_text,
            processors=[p.name for p in procs],
            boost_phrases=list(preset.boost_phrases),
            suppress_phrases=list(preset.suppress_phrases),
            strength=req.strength,
            adjustments=self._describe(procs),
            seed=seed,
        )

        sampler = make_sampler(temp=req.temperature, top_p=req.top_p)

        pieces: list[str] = []
        kls: list[float] = []
        flipped = 0
        i = 0
        finish_reason = None
        t0 = time.perf_counter()

        for resp in stream_generate(
            model,
            tokenizer,
            prompt=prompt_ids,
            max_tokens=req.max_tokens,
            sampler=sampler,
            logits_processors=chain,
        ):
            frame = store.pop(i)
            if frame is None or "base" not in frame or labels[-1] not in frame:
                # タップとステップがずれた場合は可視化を諦めてテキストだけ流す
                logger.warning("tap frame missing at step %d", i)
                ev = StepEvent(
                    i=i,
                    text=resp.text,
                    chosen=ChosenToken(id=resp.token, text=_safe_decode(tokenizer, resp.token)),
                )
            else:
                ev = self._analyse(
                    tokenizer, frame, labels, resp.token, resp.text, i, phrase_procs
                )
                kls.append(ev.kl)
                if ev.base_top and ev.bent_top and ev.base_top[0].id != ev.bent_top[0].id:
                    flipped += 1
            store.discard_before(i)

            pieces.append(resp.text)
            yield ev
            i += 1
            finish_reason = resp.finish_reason

        elapsed = time.perf_counter() - t0
        yield DoneEvent(
            tokens=i,
            elapsed=round(elapsed, 3),
            tps=round(i / elapsed, 2) if elapsed > 0 else 0.0,
            mean_kl=round(sum(kls) / len(kls), 4) if kls else 0.0,
            max_kl=round(max(kls), 4) if kls else 0.0,
            flipped=flipped,
            text="".join(pieces),
            finish_reason=finish_reason,
        )

    # --- 生成本体（非同期ラッパ） ----------------------------------------

    async def stream(self, req: GenerationRequest) -> AsyncIterator[BaseModel]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        sentinel = object()

        def emit(item) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, item)

        def worker() -> None:
            try:
                # MLX の生成は同期かつ排他。ここで直列化する。
                with self._gen_lock:
                    for ev in self._generate_sync(req):
                        emit(ev)
            except Exception as exc:  # noqa: BLE001 - クライアントに理由を返したい
                logger.exception("mlx generation failed")
                emit(ErrorEvent(message=f"{type(exc).__name__}: {exc}"))
            finally:
                emit(sentinel)

        threading.Thread(target=worker, name="mlx-generate", daemon=True).start()

        while True:
            item = await queue.get()
            if item is sentinel:
                return
            yield item


# --- ヘルパー -------------------------------------------------------------


def _as_id_list(value) -> list[int]:
    """`eos_token_ids` は実装によって int だったり集合だったりする。"""
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    try:
        return [int(v) for v in value]
    except TypeError:
        return []


def _eos_ids(tokenizer) -> list[int]:
    """EOS と、チャットテンプレートの終端トークンをまとめて返す。"""
    ids: set[int] = set(_as_id_list(getattr(tokenizer, "eos_token_ids", None)))
    ids.update(_as_id_list(getattr(tokenizer, "eos_token_id", None)))
    for marker in ("<end_of_turn>", "<|im_end|>", "<|eot_id|>"):
        try:
            enc = tokenizer.encode(marker, add_special_tokens=False)
        except TypeError:
            enc = tokenizer.encode(marker)
        except Exception:
            continue
        if len(enc) == 1:
            ids.add(int(enc[0]))
    return sorted(ids)


def _safe_decode(tokenizer, token_id: int) -> str:
    try:
        text = tokenizer.decode([int(token_id)])
    except Exception:
        return f"<{token_id}>"
    return text.replace("\n", "\\n") if text.strip() == "" and "\n" in text else text


def _to_top(tokenizer, idx: mx.array, probs: mx.array) -> list[TokenProb]:
    ids = [int(x) for x in idx.tolist()]
    ps = [float(x) for x in probs.tolist()]
    rows = sorted(zip(ids, ps), key=lambda r: r[1], reverse=True)
    return [TokenProb(id=i, text=_safe_decode(tokenizer, i), p=p) for i, p in rows]
