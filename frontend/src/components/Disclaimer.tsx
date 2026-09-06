export function Disclaimer() {
  return (
    <section className="card-plain p-5">
      <p className="text-[14px] leading-relaxed font-medium">
        LLMの確率分布を恣意的に変更することによって都合の良い内容にプロバイダーは変更できます。
        <br />
        商品のレコメンドやプロパガンダなど様々な応用が可能です。
      </p>
      <p className="mt-3 text-[12px] leading-relaxed text-muted-foreground">
        画面の数値はすべて実測値です。利用者に見えるのは右側の結果だけで、
        左側が存在したことは通常どこにも現れません。
        表示される陰謀論的な主張は、この操作を実演するために生成させたものであり、事実ではありません。
      </p>
    </section>
  )
}
