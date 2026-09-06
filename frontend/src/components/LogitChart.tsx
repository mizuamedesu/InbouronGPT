import type { StepEvent, TokenProb } from "@/lib/types"
import { cn, fmtPct, tokenLabel } from "@/lib/format"

const ROWS = 6

interface Props {
  step: StepEvent | null
  disabledNote?: string | null
}

/**
 * 素の分布と、書き換えたあとの分布を左右に並べる。
 *
 * 棒の長さより「同じトークンの順位がどこからどこへ動いたか」のほうが
 * 操作の実態を表すので、選ばれたトークンの移動を数値で明示する。
 */
export function LogitChart({ step, disabledNote }: Props) {
  const base = step?.base_top?.slice(0, ROWS) ?? []
  const bent = step?.bent_top?.slice(0, ROWS) ?? []
  const baseIds = new Set(base.map((t) => t.id))

  return (
    <section className="card-plain p-5">
      <div className="mb-4 flex items-baseline justify-between">
        <h2 className="text-[15px] font-bold">確率分布</h2>
        {step && (
          <span className="num text-[12px] text-muted-foreground">
            {step.i + 1} トークン目
          </span>
        )}
      </div>

      {disabledNote ? (
        <Empty>{disabledNote}</Empty>
      ) : !step ? (
        <Empty>生成すると、1トークンごとの確率がここに出ます。</Empty>
      ) : (
        <>
          <div className="grid grid-cols-2 gap-5">
            <Column
              label="素の分布"
              sub="モデル本来の確率"
              tone="base"
              tokens={base}
              chosenId={step.chosen.id}
            />
            <Column
              label="曲げた分布"
              sub="書き換えたあと"
              tone="bent"
              tokens={bent}
              chosenId={step.chosen.id}
              newIds={bent.filter((t) => !baseIds.has(t.id)).map((t) => t.id)}
            />
          </div>

          <Readout step={step} />
        </>
      )}
    </section>
  )
}

function Column({
  label,
  sub,
  tone,
  tokens,
  chosenId,
  newIds = [],
}: {
  label: string
  sub: string
  tone: "base" | "bent"
  tokens: TokenProb[]
  chosenId: number
  newIds?: number[]
}) {
  const color = tone === "base" ? "var(--base-signal)" : "var(--bent-signal)"
  const isNew = new Set(newIds)

  return (
    <div>
      <div className="mb-2.5">
        <div className="flex items-center gap-1.5">
          <span className="size-2 rounded-full" style={{ background: color }} />
          <span className="text-[13px] font-semibold">{label}</span>
        </div>
        <p className="mt-0.5 pl-3.5 text-[11px] text-muted-foreground">{sub}</p>
      </div>

      <ul className="space-y-1.5">
        {tokens.map((t) => {
          const chosen = t.id === chosenId
          return (
            <li key={t.id} className="flex items-center gap-2">
              <span
                className={cn(
                  "num w-[74px] shrink-0 truncate text-[12px]",
                  chosen ? "font-semibold text-foreground" : "text-foreground/70",
                )}
                title={t.text}
              >
                {tokenLabel(t.text)}
              </span>

              <span className="h-2 flex-1 overflow-hidden rounded-full bg-muted">
                <span
                  className="block h-full rounded-full transition-[width] duration-150"
                  style={{
                    width: `${Math.max(t.p * 100, t.p > 0 ? 1.5 : 0)}%`,
                    background: color,
                    opacity: chosen ? 1 : 0.55,
                  }}
                />
              </span>

              <span className="num w-[50px] shrink-0 text-right text-[11px] text-muted-foreground">
                {fmtPct(t.p)}
              </span>

              {isNew.has(t.id) && (
                <span
                  className="shrink-0 rounded-full px-1.5 py-px text-[9px] font-semibold"
                  style={{ background: "color-mix(in oklab, var(--bent-signal) 12%, transparent)", color }}
                >
                  新
                </span>
              )}
            </li>
          )
        })}
      </ul>
    </div>
  )
}

function Readout({ step }: { step: StepEvent }) {
  const { chosen } = step
  const jump = chosen.rank_base - chosen.rank_bent

  return (
    <div className="mt-4 flex flex-wrap items-center gap-x-6 gap-y-2 border-t border-border pt-3.5 text-[12px]">
      <span className="text-muted-foreground">
        選ばれた{" "}
        <span className="num rounded bg-muted px-1.5 py-0.5 font-semibold text-foreground">
          {tokenLabel(chosen.text)}
        </span>
      </span>

      <span className="num text-muted-foreground">
        確率 <span className="text-foreground">{fmtPct(chosen.p_base)}</span>
        <span className="mx-1">→</span>
        <span style={{ color: "var(--bent-signal)" }}>{fmtPct(chosen.p_bent)}</span>
      </span>

      <span className="num text-muted-foreground">
        順位 <span className="text-foreground">{chosen.rank_base}位</span>
        <span className="mx-1">→</span>
        <span style={{ color: "var(--bent-signal)" }}>{chosen.rank_bent}位</span>
        {jump > 0 && (
          <span
            className="ml-1.5 rounded-full px-1.5 py-px text-[10px] font-semibold"
            style={{
              background: "color-mix(in oklab, var(--bent-signal) 12%, transparent)",
              color: "var(--bent-signal)",
            }}
          >
            +{jump}
          </span>
        )}
      </span>

      {step.targeted_phrase && (
        <span className="text-muted-foreground">
          狙われた語 <span className="text-foreground">{step.targeted_phrase}</span>
        </span>
      )}
    </div>
  )
}

function Empty({ children }: { children: React.ReactNode }) {
  return (
    <p className="py-12 text-center text-[13px] text-muted-foreground">{children}</p>
  )
}
