import { useCallback, useRef, useState } from "react"
import type { DoneEvent, MetaEvent, StepEvent, StreamEvent } from "./types"

/**
 * サーバーへ送れるのは番号だけ。
 * 強度・温度・生成長はサーバーが持っていて、こちらからは触れない。
 */
export interface RunParams {
  /** 0 = 陰謀論 / 1 = ショッピング */
  scenario: number
  /** シナリオ内の選択肢番号 */
  index: number
  /** 0 = 素の分布 / 1 = 確率分布を曲げる */
  variant: number
}

export interface RunState {
  status: "idle" | "streaming" | "done" | "error"
  meta: MetaEvent | null
  steps: StepEvent[]
  done: DoneEvent | null
  error: string | null
  text: string
}

const EMPTY: RunState = {
  status: "idle",
  meta: null,
  steps: [],
  done: null,
  error: null,
  text: "",
}

/**
 * SSE で 1 トークンずつイベントを受け取る。
 *
 * ステップは全部ためておく。あとから「どのトークンがどれだけ押し上げられたか」を
 * 遡って見せるため、そして A/B 比較のために生成後も残しておく必要がある。
 */
export function useGeneration() {
  const [state, setState] = useState<RunState>(EMPTY)
  const sourceRef = useRef<EventSource | null>(null)

  const stop = useCallback(() => {
    sourceRef.current?.close()
    sourceRef.current = null
    setState((s) => (s.status === "streaming" ? { ...s, status: "done" } : s))
  }, [])

  const run = useCallback((params: RunParams) => {
    sourceRef.current?.close()
    setState({ ...EMPTY, status: "streaming" })

    const q = new URLSearchParams({
      scenario: String(params.scenario),
      index: String(params.index),
      variant: String(params.variant),
    })

    const es = new EventSource(`/api/generate/stream?${q}`)
    sourceRef.current = es

    es.onmessage = (raw) => {
      const ev = JSON.parse(raw.data) as StreamEvent
      setState((s) => {
        switch (ev.type) {
          case "meta":
            return { ...s, meta: ev }
          case "step":
            return { ...s, steps: [...s.steps, ev], text: s.text + ev.text }
          case "done":
            es.close()
            sourceRef.current = null
            return { ...s, status: "done", done: ev, text: ev.text || s.text }
          case "error":
            es.close()
            sourceRef.current = null
            return { ...s, status: "error", error: ev.message }
        }
      })
    }

    es.onerror = () => {
      es.close()
      sourceRef.current = null
      setState((s) =>
        s.status === "streaming"
          ? { ...s, status: "error", error: "ストリームが切断されました。バックエンドは起動していますか？" }
          : s,
      )
    }
  }, [])

  const reset = useCallback(() => {
    sourceRef.current?.close()
    sourceRef.current = null
    setState(EMPTY)
  }, [])

  return { state, run, stop, reset }
}
