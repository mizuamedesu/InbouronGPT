import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { Disclaimer } from "@/components/Disclaimer"
import { LogitChart } from "@/components/LogitChart"
import { ProductPicker } from "@/components/ProductPicker"
import { QuestionPicker } from "@/components/QuestionPicker"
import { TokenStream } from "@/components/TokenStream"
import { Transcript } from "@/components/Transcript"
import { useGeneration } from "@/lib/useGeneration"
import { cn } from "@/lib/format"
import type {
  Adjustment,
  ConspiracyScenario,
  Health,
  Mode,
  Scenario,
  ShoppingScenario,
} from "@/lib/types"

export default function App() {
  const [scenarios, setScenarios] = useState<Scenario[]>([])
  const [health, setHealth] = useState<Health | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)

  const [mode, setMode] = useState<Mode>("shopping")
  const [selected, setSelected] = useState(0)
  const [biased, setBiased] = useState(true)
  // 推させる企業の番号。null なら操作しない。
  const [target, setTarget] = useState<number | null>(1)
  // 回答をクリックして特定のトークンを選んだときの位置。null なら最新に追従。
  const [pinned, setPinned] = useState<number | null>(null)

  const { state, run, stop, reset } = useGeneration()
  const chartRef = useRef<HTMLDivElement>(null)

  const refreshHealth = useCallback(async () => {
    try {
      setHealth(await (await fetch("/api/health")).json())
    } catch {
      /* 表示できないだけ */
    }
  }, [])

  useEffect(() => {
    ;(async () => {
      try {
        const s = await fetch("/api/scenarios").then((r) => r.json())
        setScenarios(s.scenarios)
      } catch {
        setLoadError("バックエンドに接続できません。localhost:8000 で起動していますか？")
      }
      refreshHealth()
    })()
  }, [refreshHealth])

  const conspiracy = scenarios.find((s) => s.mode === "conspiracy") as
    | ConspiracyScenario
    | undefined
  const shopping = scenarios.find((s) => s.mode === "shopping") as ShoppingScenario | undefined
  const activeScenario = scenarios.find((s) => s.mode === mode)

  const question = conspiracy?.questions.find((q) => q.index === selected)
  const presets = question?.presets ?? []
  const biasPreset = presets.find((p) => !p.is_control)

  const caps = state.meta?.capabilities ?? health?.capabilities
  const canInspect = caps?.logit_inspection ?? true
  const canInject = caps?.logit_injection ?? true

  const latest = state.steps.length ? state.steps[state.steps.length - 1] : null
  const current = pinned !== null ? (state.steps[pinned] ?? latest) : latest
  const streaming = state.status === "streaming"

  const switchMode = useCallback(
    (next: Mode) => {
      setMode(next)
      setPinned(null)
      reset()
    },
    [reset],
  )

  const onRun = useCallback(() => {
    setPinned(null)
    // 生成が始まったら確率分布を画面上部に持ってくる。
    // 選択肢の一覧が長いので、放っておくと肝心の可視化が画面外に残る。
    chartRef.current?.scrollIntoView({
      behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches
        ? "auto"
        : "smooth",
      block: "start",
    })
    if (mode === "shopping") {
      // 素の分布のときは企業番号が結果に影響しないので、選択をそのまま残す
      run({ scenario: 1, index: target ?? 0, variant: target === null ? 0 : 1 })
    } else {
      run({ scenario: 0, index: selected, variant: biased ? 1 : 0 })
    }
  }, [biased, mode, run, selected, target])

  const disabledNote = useMemo(
    () =>
      canInspect
        ? null
        : (caps?.note ??
          "このプロバイダーは1トークンごとの確率を返さないため、可視化はできません。"),
    [canInspect, caps],
  )

  return (
    <>
      <header className="border-b border-border">
        <div className="mx-auto max-w-5xl px-6 py-4">
          <h1 className="text-[20px] font-bold tracking-tight">InbouronGPT</h1>
        </div>
        <div className="mx-auto flex max-w-5xl gap-1 px-6">
          {(["conspiracy", "shopping"] as const).map((m) => {
            const s = scenarios.find((x) => x.mode === m)
            const active = mode === m
            return (
              <button
                key={m}
                type="button"
                disabled={streaming}
                onClick={() => switchMode(m)}
                className={cn(
                  "-mb-px border-b-2 px-3 py-2.5 text-[14px] font-medium transition-colors disabled:opacity-50",
                  active
                    ? "border-foreground text-foreground"
                    : "border-transparent text-muted-foreground hover:text-foreground",
                )}
              >
                {s?.name ?? (m === "shopping" ? "ショッピングレコメンド" : "陰謀論")}
              </button>
            )
          })}
        </div>
      </header>

      <main className="mx-auto max-w-5xl space-y-4 px-6 py-8">
        {loadError && <Notice tone="error">{loadError}</Notice>}
        {state.status === "error" && <Notice tone="error">{state.error}</Notice>}
        {!canInject && caps?.note && <Notice tone="warn">{caps.note}</Notice>}

        <section className="card-plain space-y-4 p-5">
          {mode === "conspiracy" ? (
            <>
              <QuestionPicker
                questions={conspiracy?.questions ?? []}
                selected={selected}
                onSelect={(i) => {
                  setSelected(i)
                  setPinned(null)
                  reset()
                }}
                disabled={streaming}
              />
              <p className="text-[17px] font-semibold">{question?.text ?? "…"}</p>
            </>
          ) : (
            <>
              <p className="text-[17px] font-semibold">{shopping?.question ?? "…"}</p>
              <ProductPicker
                products={shopping?.products ?? []}
                target={target}
                onTargetChange={(k) => {
                  setTarget(k)
                  setPinned(null)
                  reset()
                }}
                disabled={streaming}
              />
            </>
          )}

          <div className="flex flex-wrap items-center justify-between gap-3 border-t border-border pt-4">
            {mode === "conspiracy" ? (
              <div className="flex items-center gap-2.5">
                <Switch
                  id="bias"
                  checked={biased}
                  onCheckedChange={setBiased}
                  disabled={streaming || !canInject}
                />
                <Label htmlFor="bias" className="cursor-pointer text-[13px]">
                  確率分布を曲げる
                  {biased && biasPreset && (
                    <span className="ml-2 text-[12px] text-muted-foreground">
                      {biasPreset.name}
                    </span>
                  )}
                </Label>
              </div>
            ) : (
              <span className="text-[13px] text-muted-foreground">
                {target !== null
                  ? `${shopping?.products.find((p) => p.index === target)?.vendor} を推すよう確率分布を操作`
                  : "確率分布は操作しない"}
              </span>
            )}

            <div className="flex items-center gap-2">
              {state.done && !streaming && (
                <span className="num mr-1 text-[12px] text-muted-foreground">
                  {state.done.tokens} トークン ／ 1位が入れ替わった回数{" "}
                  <span className="font-semibold" style={{ color: "var(--bent-signal)" }}>
                    {state.done.flipped}
                  </span>
                </span>
              )}
              {streaming ? (
                <Button variant="secondary" size="sm" onClick={stop}>
                  停止
                </Button>
              ) : (
                <Button size="sm" onClick={onRun}>
                  生成する
                </Button>
              )}
            </div>
          </div>
        </section>

        <div ref={chartRef} className="scroll-mt-4">
          <LogitChart
            step={current}
            disabledNote={disabledNote}
            pinned={pinned !== null}
            onUnpin={() => setPinned(null)}
          />
        </div>

        <TokenStream
          steps={state.steps}
          streaming={streaming}
          plain={!canInspect}
          fallbackText={state.text}
          selected={pinned}
          onSelect={setPinned}
        />

        <RunSettings
          prompt={state.meta?.system_prompt ?? activeScenario?.system_prompt}
          adjustments={state.meta?.adjustments}
        />

        <Transcript meta={state.meta} assistant={state.text} streaming={streaming} />

        <Disclaimer />

        <footer className="pt-1 pb-2 text-center">
          <a
            href="https://github.com/mizuamedesu/InbouronGPT"
            target="_blank"
            rel="noreferrer"
            className="text-[12px] text-muted-foreground underline underline-offset-4 hover:text-foreground"
          >
            github.com/mizuamedesu/InbouronGPT
          </a>
        </footer>
      </main>
    </>
  )
}

/**
 * 生成に使うシステムプロンプトと、いま適用されている操作。
 *
 * プロンプトは動かさず logit だけを動かす、という条件を目で確かめられるように、
 * 送っている文面と、書き換えている中身の両方を出しておく。
 * 操作の一覧は組み立て済みの processor から読んだ実値で、プリセットの定義ではない。
 */
function RunSettings({
  prompt,
  adjustments,
}: {
  prompt: string | undefined
  adjustments: Adjustment[] | undefined
}) {
  return (
    <section className="card-plain p-5">
      <h2 className="mb-3 text-[15px] font-bold">生成の設定</h2>

      <div className="space-y-3">
        <div>
          <span className="mb-1 block text-[12px] text-muted-foreground">
            システムプロンプト
          </span>
          <p className="text-[13px] leading-relaxed text-foreground/85">{prompt ?? "…"}</p>
        </div>

        <div className="border-t border-border pt-3">
          <span className="mb-2 block text-[12px] text-muted-foreground">
            確率分布への操作
          </span>
          {!adjustments?.length ? (
            <p className="text-[13px] text-foreground/85">
              なし（モデル本来の確率分布のまま生成）
            </p>
          ) : (
            <ul className="space-y-2">
              {adjustments.map((a) => (
                <li key={a.processor} className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
                  <span
                    className="num shrink-0 rounded px-1.5 py-px text-[11px] font-semibold"
                    style={
                      a.factor > 0
                        ? { background: "#ffe4e6", color: "#9f1239" }
                        : a.factor < 0
                          ? { background: "#dbeafe", color: "#1e40af" }
                          : { background: "#e5e7eb", color: "#374151" }
                    }
                  >
                    {a.factor === 0 ? "強制" : `${a.factor > 0 ? "+" : ""}${a.factor}`}
                  </span>
                  <span className="shrink-0 text-[13px] font-medium">{a.label}</span>
                  {a.phrases.length > 0 && (
                    <span className="text-[12px] leading-relaxed text-muted-foreground">
                      {a.phrases.join(" / ")}
                    </span>
                  )}
                  {a.note && (
                    <span className="text-[12px] text-muted-foreground">{a.note}</span>
                  )}
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </section>
  )
}

function Notice({ tone, children }: { tone: "error" | "warn"; children: React.ReactNode }) {
  const color = tone === "error" ? "var(--destructive)" : "#b45309"
  return (
    <div
      className="rounded-xl border px-4 py-3 text-[13px] leading-relaxed"
      style={{
        borderColor: `color-mix(in oklab, ${color} 28%, transparent)`,
        background: `color-mix(in oklab, ${color} 6%, transparent)`,
      }}
    >
      {children}
    </div>
  )
}
