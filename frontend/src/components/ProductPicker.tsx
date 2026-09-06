import type { Product } from "@/lib/types"
import { cn } from "@/lib/format"

interface Props {
  products: Product[]
  /** 推させる対象。null なら操作しない（素の分布）。 */
  target: string | null
  onTargetChange: (key: string | null) => void
  disabled?: boolean
}

/**
 * 検索結果の一覧が、そのまま「どれを推させるか」の選択肢になっている。
 *
 * 検索結果の文面は選択によらず常に同一のままモデルへ渡る。
 * 変わるのは logit だけ、という条件を守るため。
 */
export function ProductPicker({ products, target, onTargetChange, disabled }: Props) {
  return (
    <div className="space-y-2.5">
      <ul className="divide-y divide-border overflow-hidden rounded-xl border border-border">
        <Row
          selected={target === null}
          disabled={disabled}
          onSelect={() => onTargetChange(null)}
          title="操作しない"
          body="確率分布を書き換えず、素の推薦をそのまま見る。"
        />

        {products.map((p) => (
          <Row
            key={p.key}
            selected={target === p.key}
            disabled={disabled}
            onSelect={() => onTargetChange(p.key)}
            title={`${p.vendor}　${p.name}`}
            meta={
              <>
                <span className="num">{p.price.toLocaleString()}円</span>
                <span className="num">
                  ★{p.rating}（{p.reviews.toLocaleString()}件）
                </span>
                <span className="num">{p.battery}</span>
              </>
            }
            body={
              <>
                {p.highlight}
                <span className="ml-1.5 opacity-70">難点: {p.drawback}</span>
              </>
            }
          />
        ))}
      </ul>

      <p className="text-[11px] leading-relaxed text-muted-foreground">
        この検索結果の文面は、どれを選んでも一切変わりません。
        変えているのは生成時の確率分布だけです。
      </p>
    </div>
  )
}

function Row({
  selected,
  disabled,
  onSelect,
  title,
  meta,
  body,
}: {
  selected: boolean
  disabled?: boolean
  onSelect: () => void
  title: string
  meta?: React.ReactNode
  body: React.ReactNode
}) {
  return (
    <li>
      <button
        type="button"
        disabled={disabled}
        onClick={onSelect}
        aria-pressed={selected}
        className={cn(
          "flex w-full items-start gap-2.5 px-4 py-3 text-left transition-colors",
          "hover:bg-muted/60 disabled:cursor-not-allowed disabled:opacity-50",
          selected && "bg-rose-50/70 hover:bg-rose-50/70",
        )}
      >
        <span
          className={cn(
            "mt-[3px] flex size-[15px] shrink-0 items-center justify-center rounded-full border transition-colors",
            selected ? "border-transparent" : "border-border",
          )}
          style={selected ? { background: "var(--bent-signal)" } : undefined}
        >
          {selected && <span className="size-[5px] rounded-full bg-white" />}
        </span>

        <span className="min-w-0 flex-1">
          <span className="flex flex-wrap items-baseline gap-x-2.5 gap-y-1">
            <span className="text-[14px] font-semibold">{title}</span>
            {meta && (
              <span className="flex flex-wrap items-baseline gap-x-2.5 text-[12px] text-muted-foreground">
                {meta}
              </span>
            )}
          </span>
          <span className="mt-1 block text-[12px] leading-relaxed text-muted-foreground">
            {body}
          </span>
        </span>
      </button>
    </li>
  )
}
