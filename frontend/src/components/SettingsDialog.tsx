import { useEffect, useState } from "react"
import { Settings2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import type { Health, RuntimeConfig } from "@/lib/types"

interface Props {
  config: RuntimeConfig | null
  health: Health | null
  onSave: (patch: Partial<RuntimeConfig>) => Promise<void>
}

const PROVIDERS = [
  {
    value: "mlx",
    label: "MLX（ローカル）",
    note: "語彙全体の logit を実測しながら書き換えられる唯一の経路。",
  },
  {
    value: "ollama",
    label: "Ollama",
    note: "エンドポイントを差し替えられるが、logits を返しも受け取りもしないため分布の操作と可視化はできない。",
  },
  {
    value: "openai_compat",
    label: "vLLM / OpenAI 互換",
    note: "vLLM や LM Studio など。logprobs を返すサーバーなら top-8 の範囲で操作と可視化ができる。語彙全体を握るには MLX を使う。",
  },
] as const

export function SettingsDialog({ config, health, onSave }: Props) {
  const [open, setOpen] = useState(false)
  const [draft, setDraft] = useState<RuntimeConfig | null>(config)
  const [saving, setSaving] = useState(false)

  useEffect(() => setDraft(config), [config])

  if (!draft) return null

  const set = <K extends keyof RuntimeConfig>(key: K, value: RuntimeConfig[K]) =>
    setDraft((d) => (d ? { ...d, [key]: value } : d))

  const save = async () => {
    setSaving(true)
    try {
      await onSave(draft)
      setOpen(false)
    } finally {
      setSaving(false)
    }
  }

  const providerNote = PROVIDERS.find((p) => p.value === draft.provider)?.note

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger
        render={
          <Button variant="ghost" size="sm" className="gap-1.5 text-muted-foreground" />
        }
      >
        <Settings2 className="size-3.5" />
        設定
      </DialogTrigger>

      <DialogContent className="max-w-lg gap-5">
        <DialogHeader>
          <DialogTitle className="text-[15px]">推論プロバイダー</DialogTitle>
          <DialogDescription className="text-[12px] leading-relaxed">
            エンドポイントは自由に差し替えられます。ただし確率分布を実測して
            書き換えられるかどうかは、プロバイダーの能力に依存します。
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <Field label="プロバイダー">
            <Select
              value={draft.provider}
              onValueChange={(v) => set("provider", v as RuntimeConfig["provider"])}
            >
              <SelectTrigger className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {PROVIDERS.map((p) => (
                  <SelectItem key={p.value} value={p.value}>
                    {p.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {providerNote && (
              <p className="mt-1.5 text-[11px] leading-relaxed text-muted-foreground">{providerNote}</p>
            )}
          </Field>

          {draft.provider === "mlx" && (
            <Field label="MLX モデル ID">
              <Input value={draft.mlx_model} onChange={(v) => set("mlx_model", v)} />
            </Field>
          )}

          {draft.provider === "ollama" && (
            <>
              <Field label="エンドポイント">
                <Input
                  value={draft.ollama_base_url}
                  onChange={(v) => set("ollama_base_url", v)}
                  placeholder="http://localhost:11434"
                />
              </Field>
              <Field label="モデル">
                <Input value={draft.ollama_model} onChange={(v) => set("ollama_model", v)} />
              </Field>
            </>
          )}

          {draft.provider === "openai_compat" && (
            <>
              <Field label="ベース URL">
                <Input
                  value={draft.openai_base_url}
                  onChange={(v) => set("openai_base_url", v)}
                  placeholder="http://localhost:8001/v1"
                />
              </Field>
              <Field label="モデル">
                <Input value={draft.openai_model} onChange={(v) => set("openai_model", v)} />
              </Field>
            </>
          )}

          {health && (
            <div className="rounded-lg border border-border bg-muted px-3 py-2.5">
              <span className="mb-1 block text-[12px] font-medium">現在の状態</span>
              <p className="num text-[11px] leading-relaxed text-muted-foreground">
                <span style={{ color: health.ok ? "#16a34a" : "var(--destructive)" }}>
                  {health.ok ? "●" : "●"}
                </span>{" "}
                {health.message}
              </p>
            </div>
          )}
        </div>

        <div className="flex justify-end gap-2">
          <Button variant="ghost" size="sm" onClick={() => setOpen(false)}>
            キャンセル
          </Button>
          <Button size="sm" onClick={save} disabled={saving}>
            {saving ? "保存中…" : "保存"}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  )
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <Label className="mb-1.5 block text-[12px] font-medium">{label}</Label>
      {children}
    </div>
  )
}

function Input({
  value,
  onChange,
  placeholder,
}: {
  value: string
  onChange: (v: string) => void
  placeholder?: string
}) {
  return (
    <input
      value={value}
      placeholder={placeholder}
      onChange={(e) => onChange(e.target.value)}
      className="num h-9 w-full rounded-lg border border-border px-2.5 text-[12px] outline-none focus:border-foreground/40"
    />
  )
}
