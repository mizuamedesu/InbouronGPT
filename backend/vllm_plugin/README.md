# vLLM プラグイン

`InbouronLogitsProcessor` を vLLM V1 のエンジン内で動かし、語彙全体に対して
logit 操作を行う。判断ロジック（`app.processors.trie`）は MLX 経路と共有しているため、
開発機と本番で「どのトークンをいくつ押すか」は同一。

## 起動

vLLM を動かす環境から `app` パッケージが import できる必要がある。

```bash
cd backend
PYTHONPATH=.:vllm_plugin \
vllm serve PinoCookie/LFM2.5-1.2B-JP-Abliterated \
  --port 8001 \
  --logits-processors inbouron_logits:InbouronLogitsProcessor
```

## リクエスト

`vllm_xargs` を付けたリクエストだけが操作される。付けなければ素の分布のまま。

```json
{
  "model": "...",
  "messages": [...],
  "logprobs": true,
  "top_logprobs": 12,
  "vllm_xargs": {
    "inbouron": {"mode": "shopping", "target": "B", "strength": 1.0}
  }
}
```

| キー | 値 |
|---|---|
| `mode` | `conspiracy` / `shopping` |
| `index` | 陰謀論の質問 index（0〜3） |
| `preset` | 陰謀論のプリセットキー（省略で既定） |
| `target` | ショッピングで推す企業（`A`〜`D`、省略で無操作） |
| `strength` | 0〜3 の倍率 |

## 同時リクエスト

`update_state` がバッチ内の追加・削除・入れ替えを追い、リクエストごとに
独立した `PhraseBiasState` を持つ。`output_tok_ids` は vLLM が保持する
生成中リストへの参照なので、追加のコピーなしに最新の生成列を見られる。
