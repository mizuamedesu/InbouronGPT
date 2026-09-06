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
  processors: string[]
  boost_phrases: string[]
  suppress_phrases: string[]
  strength: number
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

export interface RuntimeConfig {
  provider: "mlx" | "ollama" | "openai_compat"
  mlx_model: string
  ollama_base_url: string
  ollama_model: string
  openai_base_url: string
  openai_model: string
  max_tokens: number
  temperature: number
  top_p: number
}

export interface Health {
  provider: string
  model: string
  ok: boolean
  message: string
  capabilities: Capabilities
  mlx_loaded: boolean
}
