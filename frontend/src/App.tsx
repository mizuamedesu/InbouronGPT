import { useCallback, useEffect, useMemo, useState } from "react"
import { Button } from "@/components/ui/button"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { Disclaimer } from "@/components/Disclaimer"
import { LogitChart } from "@/components/LogitChart"
import { QuestionPicker } from "@/components/QuestionPicker"
import { SettingsDialog } from "@/components/SettingsDialog"
import { TokenStream } from "@/components/TokenStream"
import { useGeneration } from "@/lib/useGeneration"
import type { Health, Question, RuntimeConfig } from "@/lib/types"

export default function App() {
  const [questions, setQuestions] = useState<Question[]>([])
  const [config, setConfig] = useState<RuntimeConfig | null>(null)
  const [health, setHealth] = useState<Health | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)

  const [selected, setSelected] = useState(0)
  const [biased, setBiased] = useState(true)

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
        const [q, c] = await Promise.all([
          fetch("/api/questions").then((r) => r.json()),
          fetch("/api/config").then((r) => r.json()),
        ])
        setQuestions(q.questions)
        setConfig(c)
      } catch {
        setLoadError("バックエンドに接続できません。localhost:8000 で起動していますか？")
      }
      refreshHealth()
    })()
  }, [refreshHealth])

  const question = questions.find((q) => q.index === selected)
  const presets = question?.presets ?? []
  const controlPreset = presets.find((p) => p.is_control)
  const biasPreset = presets.find((p) => !p.is_control)

  const caps = state.meta?.capabilities ?? health?.capabilities
  const canInspect = caps?.logit_inspection ?? true
  const canInject = caps?.logit_injection ?? true

  const current = state.steps.length ? state.steps[state.steps.length - 1] : null
  const streaming = state.status === "streaming"

  const onRun = useCallback(() => {
    run({
      index: selected,
      preset: biased ? biasPreset?.key : controlPreset?.key,
      strength: 1,
      maxTokens: 200,
      temperature: 0.7,
    })
  }, [biased, biasPreset, controlPreset, run, selected])

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
      </header>

      <main className="mx-auto max-w-5xl space-y-4 px-6 py-8">
        {loadError && <Notice tone="error">{loadError}</Notice>}
        {state.status === "error" && <Notice tone="error">{state.error}</Notice>}
        {!canInject && caps?.note && <Notice tone="warn">{caps.note}</Notice>}

        {/* 質問と操作 */}
        <section className="card-plain space-y-4 p-5">
          <QuestionPicker
            questions={questions}
            selected={selected}
            onSelect={(i) => {
              setSelected(i)
              reset()
            }}
            disabled={streaming}
          />

          <p className="text-[17px] font-semibold">{question?.text ?? "…"}</p>

          <SystemPromptNote meta={state.meta} />

          <div className="flex flex-wrap items-center justify-between gap-3 border-t border-border pt-4">
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

        <LogitChart step={current} disabledNote={disabledNote} />

        <TokenStream
          steps={state.steps}
          streaming={streaming}
          plain={!canInspect}
          fallbackText={state.text}
        />

        <Disclaimer />
      </main>
    </>
  )
}

/**
 * どのシステムプロンプトで生成したかを見せる。
 *
 * 「素の分布」と「曲げた分布」の差が logit 操作だけに由来することは、
 * 両者のプロンプトが同一だと確認できて初めて言える。だから隠さず出す。
 * このアプリはプロンプト注入を一切行わないので、ここは常に同一。
 */
function SystemPromptNote({ meta }: { meta: { system_prompt: string } | null }) {
  return (
    <details className="group rounded-lg bg-muted px-3 py-2">
      <summary className="flex cursor-pointer list-none items-center gap-2 text-[12px] text-muted-foreground">
        <span className="transition-transform group-open:rotate-90">›</span>
        システムプロンプト
        <span
          className="rounded-full px-2 py-px text-[11px] font-medium"
          style={{ background: "#dcfce7", color: "#15803d" }}
        >
          素・曲げで同一
        </span>
      </summary>
      <p className="mt-2 text-[12px] leading-relaxed text-foreground/80">
        {meta?.system_prompt ?? "生成すると、実際に使われたプロンプトが出ます。"}
      </p>
      <p className="mt-1.5 text-[11px] text-muted-foreground">
        質問文もこのプロンプトも「曲げる」の有無で変わりません。
        出力の違いは確率分布の書き換えだけに由来します。
      </p>
    </details>
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
