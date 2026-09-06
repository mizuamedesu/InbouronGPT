import { useEffect, useRef } from "react"
import type { StepEvent } from "@/lib/types"
import { fmtPct } from "@/lib/format"

interface Props {
  steps: StepEvent[]
  streaming: boolean
  plain?: boolean
  fallbackText?: string
}

/**
 * 生成テキスト。読むぶんには普通の文章に見えるが、
 * 押し上げられて選ばれたトークンほど背景が濃くなる。
 */
export function TokenStream({ steps, streaming, plain, fallbackText }: Props) {
  const endRef = useRef<HTMLSpanElement>(null)

  useEffect(() => {
    if (streaming) endRef.current?.scrollIntoView({ block: "end" })
  }, [steps.length, streaming])

  return (
    <section className="card-plain flex min-h-[260px] flex-col p-5">
      <div className="mb-3 flex items-baseline justify-between">
        <h2 className="text-[15px] font-bold">回答</h2>
        {!plain && steps.length > 0 && (
          <span className="text-[11px] text-muted-foreground">
            色が濃いほど強く押し上げられたトークン
          </span>
        )}
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto">
        {plain ? (
          <p className="text-[15px] leading-[1.9] whitespace-pre-wrap">
            {fallbackText}
            {streaming && <Caret />}
          </p>
        ) : steps.length === 0 ? (
          <p className="py-10 text-center text-[13px] text-muted-foreground">
            まだ生成していません。
          </p>
        ) : (
          <p className="text-[15px] leading-[2] whitespace-pre-wrap">
            {steps.map((s) => (
              <Token key={s.i} step={s} />
            ))}
            {streaming && <Caret />}
            <span ref={endRef} />
          </p>
        )}
      </div>
    </section>
  )
}

function Token({ step }: { step: StepEvent }) {
  const gain = step.chosen.p_bent - step.chosen.p_base
  const jumped = step.chosen.rank_base - step.chosen.rank_bent
  const intensity = Math.min(1, Math.max(gain, jumped > 0 ? Math.log10(jumped + 1) / 2.2 : 0))
  const suppressed = gain < -0.02

  if (intensity < 0.05 && !suppressed) return <span>{step.text}</span>

  const color = suppressed ? "var(--suppress-signal)" : "var(--bent-signal)"

  return (
    <span
      className="rounded px-px"
      style={{
        background: `color-mix(in oklab, ${color} ${(suppressed ? 0.14 : 0.08 + intensity * 0.3) * 100}%, transparent)`,
      }}
      title={[
        `確率 ${fmtPct(step.chosen.p_base)} → ${fmtPct(step.chosen.p_bent)}`,
        `順位 ${step.chosen.rank_base}位 → ${step.chosen.rank_bent}位`,
        step.targeted_phrase ? `狙われた語: ${step.targeted_phrase}` : null,
      ]
        .filter(Boolean)
        .join("\n")}
    >
      {step.text}
    </span>
  )
}

function Caret() {
  return (
    <span
      className="ml-px inline-block h-[1em] w-[2px] translate-y-[2px] bg-foreground"
      style={{ animation: "pulse 1s steps(2) infinite" }}
    />
  )
}
