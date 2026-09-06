import { useCallback, useEffect, useMemo, useState } from "react"
import { Button } from "@/components/ui/button"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { Disclaimer } from "@/components/Disclaimer"
import { LogitChart } from "@/components/LogitChart"
import { ProductPicker } from "@/components/ProductPicker"
import { QuestionPicker } from "@/components/QuestionPicker"
import { SettingsDialog } from "@/components/SettingsDialog"
import { TokenStream } from "@/components/TokenStream"
import { Transcript } from "@/components/Transcript"
import { useGeneration } from "@/lib/useGeneration"
import { cn } from "@/lib/format"
import type {
  ConspiracyScenario,
  Health,
  Mode,
  RuntimeConfig,
  Scenario,
  ShoppingScenario,
} from "@/lib/types"

export default function App() {
  const [scenarios, setScenarios] = useState<Scenario[]>([])
  const [config, setConfig] = useState<RuntimeConfig | null>(null)
  const [health, setHealth] = useState<Health | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)

  const [mode, setMode] = useState<Mode>("shopping")
  const [selected, setSelected] = useState(0)
  const [biased, setBiased] = useState(true)
  const [target, setTarget] = useState<string | null>("B")
  // 回答をクリックして特定のトークンを選んだときの位置。null なら最新に追従。
  const [pinned, setPinned] = useState<number | null>(null)

  const { state, run, stop, reset } = useGeneration()

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
        const [s, c] = await Promise.all([
          fetch("/api/scenarios").then((r) => r.json()),
          fetch("/api/config").then((r) => r.json()),
        ])
        setScenarios(s.scenarios)
        setConfig(c)
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
  const controlPreset = presets.find((p) => p.is_control)
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
    if (mode === "shopping") {
      run({ mode, index: 0, target: target ?? undefined, strength: 1, maxTokens: 200, temperature: 0.7 })
    } else {
      run({
        mode,
        index: selected,
        preset: biased ? biasPreset?.key : controlPreset?.key,
        strength: 1,
        maxTokens: 200,
        temperature: 0.7,
      })
    }
  }, [biased, biasPreset, controlPreset, mode, run, selected, target])

  const saveConfig = useCallback(
    async (patch: Partial<RuntimeConfig>) => {
      const r = await fetch("/api/config", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(patch),
      })
      setConfig(await r.json())
      reset()
      await refreshHealth()
    },
    [refreshHealth, reset],
  )

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
        <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-4">
          <h1 className="text-[20px] font-bold tracking-tight">InbouronGPT</h1>
          <SettingsDialog config={config} health={health} onSave={saveConfig} />
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

          <SystemPromptNote
            prompt={state.meta?.system_prompt ?? activeScenario?.system_prompt}
          />

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
                {target
                  ? `${shopping?.products.find((p) => p.key === target)?.vendor} を推すよう確率分布を操作`
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

        <LogitChart
          step={current}
          disabledNote={disabledNote}
          pinned={pinned !== null}
          onUnpin={() => setPinned(null)}
        />

        <TokenStream
          steps={state.steps}
          streaming={streaming}
          plain={!canInspect}
          fallbackText={state.text}
          selected={pinned}
          onSelect={setPinned}
        />

        <Transcript meta={state.meta} assistant={state.text} streaming={streaming} />

        <Disclaimer />
      </main>
    </>
  )
}

/**
 * 生成に使うシステムプロンプト。
 *
 * 素と曲げの差が logit 操作だけに由来することは、両者のプロンプトが
 * 同一だと確認できて初めて言える。だから最初から出しておく。
 */
function SystemPromptNote({ prompt }: { prompt: string | undefined }) {
  return (
    <div className="rounded-lg bg-muted px-3 py-2.5">
      <div className="mb-1 flex items-center gap-2">
        <span className="text-[12px] text-muted-foreground">システムプロンプト</span>
        <span
          className="rounded-full px-2 py-px text-[11px] font-medium"
          style={{ background: "#dcfce7", color: "#15803d" }}
        >
          全パターンで同一
        </span>
      </div>
      <p className="text-[12px] leading-relaxed text-foreground/80">{prompt ?? "…"}</p>
    </div>
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
