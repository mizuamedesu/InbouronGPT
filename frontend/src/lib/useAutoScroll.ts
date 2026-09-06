import { useEffect, useRef } from "react"

/**
 * 生成中、コンテナの中身が下へ伸びたら自動で追従する。
 *
 * `scrollIntoView` は使わない。あれはページ全体を動かしてしまい、
 * 毎秒数十トークン届く状況では画面がガタつくため。
 * ここでは対象コンテナの scrollTop だけを、1 フレームにつき 1 回動かす。
 *
 * 利用者が自分で上へスクロールしていた場合は引き戻さない。
 */
export function useAutoScroll<T extends HTMLElement>(dep: unknown, active: boolean) {
  const ref = useRef<T>(null)
  const rafRef = useRef(0)

  useEffect(() => {
    if (!active) return
    const box = ref.current
    if (!box) return

    const nearBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 80
    if (!nearBottom) return

    cancelAnimationFrame(rafRef.current)
    rafRef.current = requestAnimationFrame(() => {
      box.scrollTop = box.scrollHeight
    })
    return () => cancelAnimationFrame(rafRef.current)
  }, [dep, active])

  return ref
}
