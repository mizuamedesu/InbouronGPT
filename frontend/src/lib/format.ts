export { cn } from "./utils"

/** 確率をパーセントで読みやすく。極小値は「ほぼゼロ」だと分かる形で出す。 */
export function fmtPct(p: number): string {
  if (p >= 0.1) return `${(p * 100).toFixed(1)}%`
  if (p >= 0.001) return `${(p * 100).toFixed(2)}%`
  if (p > 0) return "<0.1%"
  return "0%"
}

/** 空白や改行だけのトークンを目に見える形にする。 */
export function tokenLabel(text: string): string {
  if (text === "") return "∅"
  if (text === "\\n" || text === "\n") return "⏎"
  if (text.trim() === "") return "␣".repeat(Math.min(text.length, 3))
  return text
}

export function fmtDelta(d: number): string {
  return `${d > 0 ? "+" : ""}${d.toFixed(2)}`
}
