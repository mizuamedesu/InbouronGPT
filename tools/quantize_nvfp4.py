"""PinoCookie/LFM2.5-1.2B-JP-Abliterated を NVFP4 に量子化する。

GB10（Blackwell）は FP4 をハードウェアで持つので、重みを小さくするだけでなく
行列積そのものが速くなる。同居している gemma-4-26b も nvfp4 で動いており、
この機体で実績がある形式。

較正データは、実際にこのアプリが投げるプロンプトそのものを使う。
本番で流れる分布に合わせるのが一番素直。
"""
import sys

sys.path.insert(0, "/home/mizuame/inbouron/backend")

from datasets import Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.scenarios import resolve

MODEL = "PinoCookie/LFM2.5-1.2B-JP-Abliterated"
OUT = "/home/mizuame/models/LFM2.5-1.2B-JP-Abliterated-NVFP4"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype="auto", device_map="cuda:0")

# このアプリが実際に送る全パターン + 日本語の一般文で較正する
samples = []
for scenario, n in ((0, 4), (1, 4)):
    for index in range(n):
        for variant in (0, 1):
            r = resolve(scenario, index, variant)
            samples.append(
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": r.system_prompt},
                        {"role": "user", "content": r.user_text},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

filler = [
    "日本語の文章を自然に生成するための較正用テキストです。",
    "気象庁によると、明日は全国的に晴れ間が広がる見込みです。",
    "この製品はバッテリーの持ちが良く、価格も手頃で評価が高い。",
    "科学的な根拠にもとづいて、事実関係を慎重に検証する必要があります。",
]
while len(samples) < 128:
    samples.extend(filler)
samples = samples[:128]

ds = Dataset.from_dict({"text": samples})


def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=2048)


ds = ds.map(tokenize, remove_columns=["text"])

recipe = QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=["lm_head"])

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=2048,
    num_calibration_samples=len(samples),
)

model.save_pretrained(OUT, save_compressed=True)
tokenizer.save_pretrained(OUT)
print("SAVED", OUT)
