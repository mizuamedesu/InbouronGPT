import type { MetaEvent } from "@/lib/types"
import { useAutoScroll } from "@/lib/useAutoScroll"

interface Props {
  meta: MetaEvent | null
  assistant: string
  streaming: boolean
}

/**
 * モデルに実際に渡した中身と、返ってきた中身をそのまま並べる。
 *
 * 可視化パネルは「どう曲げたか」を示すが、そもそも何を入力したのかを
 * 隠したままでは検証にならない。整形や省略をせず全文を出す。
 */
export function Transcript({ meta, assistant, streaming }: Props) {
  return (
    <section className="card-plain p-5">
      <div className="mb-3 flex items-baseline justify-between">
        <h2 className="text-[15px] font-bold">実際のやり取り</h2>
        <span className="text-[11px] text-muted-foreground">
          モデルに渡した全文。省略なし
        </span>
      </div>

      {!meta ? (
        <p className="py-8 text-center text-[13px] text-muted-foreground">
          生成すると、送信した内容がそのまま出ます。
        </p>
      ) : (
        <div className="space-y-3">
          <Turn role="system" body={meta.system_prompt} />
          <Turn role="user" body={meta.user_text} scroll />
          <Turn
            role="assistant"
            body={assistant}
            streaming={streaming}
            tone="assistant"
            scroll
            follow
          />
        </div>
      )}
    </section>
  )
}

const ROLE_STYLE: Record<string, { bg: string; fg: string }> = {
  system: { bg: "#f3f4f6", fg: "#4b5563" },
  user: { bg: "#e0e7ff", fg: "#3730a3" },
  assistant: { bg: "#ffe4e6", fg: "#9f1239" },
}

function Turn({
  role,
  body,
  streaming,
  tone,
  scroll,
  follow,
}: {
  role: "system" | "user" | "assistant"
  body: string
  streaming?: boolean
  tone?: "assistant"
  /** 長くなる本文は箱の中でスクロールさせ、ページを伸ばさない */
  scroll?: boolean
  /** 生成中、下へ伸びるのに追従する */
  follow?: boolean
}) {
  const style = ROLE_STYLE[role]
  const boxRef = useAutoScroll<HTMLDivElement>(body.length, Boolean(follow && streaming))
  return (
    <div className="rounded-xl border border-border">
      <div className="border-b border-border px-3 py-1.5">
        <span
          className="num rounded-full px-2 py-0.5 text-[11px] font-semibold"
          style={{ background: style.bg, color: style.fg }}
        >
          {role}
        </span>
      </div>
      <div ref={boxRef} className={scroll ? "max-h-[260px] overflow-y-auto" : undefined}>
        <p
          className="px-3 py-2.5 text-[13px] leading-relaxed whitespace-pre-wrap"
          style={tone === "assistant" ? { fontWeight: 500 } : undefined}
        >
          {body || (streaming ? "" : "（空）")}
          {streaming && (
            <span className="ml-px inline-block h-[1em] w-[2px] translate-y-[2px] bg-foreground align-middle" />
          )}
        </p>
      </div>
    </div>
  )
}
