import type { Question } from "@/lib/types"
import { cn } from "@/lib/format"

interface Props {
  questions: Question[]
  selected: number
  onSelect: (index: number) => void
  disabled?: boolean
}

/** 質問は index 番号でしか選べない。自由入力欄は用意しない。 */
export function QuestionPicker({ questions, selected, onSelect, disabled }: Props) {
  return (
    <div className="flex flex-wrap gap-2">
      {questions.map((q) => {
        const active = q.index === selected
        return (
          <button
            key={q.index}
            type="button"
            disabled={disabled}
            onClick={() => onSelect(q.index)}
            className={cn(
              "pill gap-1.5 disabled:cursor-not-allowed disabled:opacity-50",
              active
                ? "bg-foreground text-background"
                : "bg-muted text-muted-foreground hover:text-foreground",
            )}
          >
            <span className="num text-[11px] opacity-60">{q.index}</span>
            {q.label}
          </button>
        )
      })}
    </div>
  )
}
