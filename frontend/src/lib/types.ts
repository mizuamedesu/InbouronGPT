export interface Capabilities {
  generation: boolean
  /** 素の確率分布を実測できるか */
  logit_inspection: boolean
  /** 確率分布を書き換えられるか */
  logit_injection: boolean
  note: string | null
}

export interface TokenProb {
  id: number
  text: string
  p: number
}

export interface ChosenToken {
  id: number
  text: string
  p_base: number
  p_bent: number
  rank_base: number
  rank_bent: number
}

export interface AppliedDelta {
  name: string
  delta: number
}

export interface Adjustment {
  processor: string
  label: string
  factor: number
  phrases: string[]
  note: string | null
}

export interface MetaEvent {
  type: "meta"
  provider: string
  model: string
  capabilities: Capabilities
  question_index: number
  question: string
  preset_key: string
  preset_name: string
  preset_description: string
  system_prompt: string
  user_text: string
  processors: string[]
  boost_phrases: string[]
  suppress_phrases: string[]
  strength: number
  adjustments: Adjustment[]
  seed: number | null
}

export interface StepEvent {
  type: "step"
  i: number
  text: string
  chosen: ChosenToken
  base_top: TokenProb[]
  bent_top: TokenProb[]
  applied: AppliedDelta[]
  kl: number
  targeted_phrase: string | null
}

export interface DoneEvent {
  type: "done"
  tokens: number
  elapsed: number
  tps: number
  mean_kl: number
  max_kl: number
  flipped: number
  text: string
  finish_reason: string | null
}

export interface ErrorEvent {
  type: "error"
  message: string
}

export type StreamEvent = MetaEvent | StepEvent | DoneEvent | ErrorEvent

export interface Preset {
  key: string
  name: string
  description: string
  boost_phrases: string[]
  suppress_phrases: string[]
  is_control: boolean
}

export interface Question {
  index: number
  label: string
  text: string
  blurb: string
  presets: Preset[]
}

export type Mode = "conspiracy" | "shopping"

export interface Product {
  index: number
  key: string
  vendor: string
  name: string
  price: number
  battery: string
  highlight: string
  drawback: string
  rating: number
  reviews: number
}

export interface ConspiracyScenario {
  scenario: number
  mode: "conspiracy"
  name: string
  blurb: string
  system_prompt: string
  questions: Question[]
}

export interface ShoppingScenario {
  scenario: number
  mode: "shopping"
  name: string
  blurb: string
  system_prompt: string
  question: string
  products: Product[]
}

export type Scenario = ConspiracyScenario | ShoppingScenario

export interface Health {
  provider: string
  model: string
  ok: boolean
  message: string
  capabilities: Capabilities
  mlx_loaded: boolean
}
