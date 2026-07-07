# Paper context for future Codex work

## 1. Project goal

This project is not just a generic multi-agent investment research chatbot.

It is a managed-services prototype of the systems substrate for a paper-inspired Principal / environment / observation / payoff loop in an open-world research setting.

Target loop:

```text
objective
→ Principal observes RunState
→ Principal selects typed PrincipalAction
→ agents/tools produce observations
→ observations become artifacts
→ verifier/skeptic updates trust
→ aggregator synthesizes trusted knowledge
→ judge/payoff evaluates output
→ Principal continues, repairs evidence, or stops under budget
```

The repo should be understood as an engineering prototype for this control loop: it makes the state, actions, evidence, trust transitions, payoff signals, and operator-facing trace explicit.

## 2. Paper concepts

### A. Social Environment Design

Social Environment Design proposes an AI-assisted automated policymaking framework for general economic environments. The core framing is that policies should be selected against explicit policy objectives or social welfare objectives, potentially with voting or objective-selection mechanisms to decide what the system should optimize.

The framework emphasizes simulation-based policy analysis: policies can be evaluated in modeled social or economic environments before real-world deployment. Conceptually, it connects reinforcement learning, economics and computation, EconCS, and computational social choice. The policymaker observes an environment, selects actions or policies, and receives feedback about outcomes relative to the chosen objective.

This repo does not implement the full Social Environment Design framework. It borrows the high-level Principal / environment / objective / action / payoff framing and applies it to open-world investment research infrastructure rather than a closed economic simulator.

### B. Large Legislative Models

Large Legislative Models explores the idea that LLMs can act as sample-efficient policymakers in socially complex multi-agent economic simulations. The motivation is that traditional RL policy generation can be sample inefficient and inflexible when the policy must incorporate nuanced contextual information, history, rules, preferences, or textual descriptions.

The important abstraction for this repo is:

```text
state / history / observations / payoff → LLM or deterministic Principal → action
```

The repo uses that abstraction to shape `RunState`, `PrincipalPolicy`, and typed `PrincipalAction` decisions. The Principal does not need to be a learned RL policy in this prototype. It can be deterministic, LLM-assisted, or hybrid, as long as it observes state and payoff signals and selects explicit next actions.

### C. Cooperative AI Policymaking Platform

The Cooperative AI Policymaking Platform is a transparent AI-assisted policy support platform. Its framing includes economic time-series and text modeling, perspective elicitation, and a public or policy-facing interface for decision support.

This repo is not that platform, but it borrows the transparent interface and decision-support framing. In this prototype, transparency means the UI should expose the Principal action timeline, temporary agent organization, evidence artifacts, trust status, verifier downgrades, budget usage, and final synthesis limitations.

## 3. Mapping table

| Paper concept | Repo component |
| --- | --- |
| Principal / policymaker | PrincipalPolicy + PrincipalAction |
| Principal action | PrincipalActionType |
| Environment state | RunState + blackboard |
| Agents / followers | AgentSpec + role workers |
| Observations | EvidenceEngine + market snapshots + artifacts |
| History / memory | Artifact graph + PrincipalAction timeline |
| Trust update | Verifier + skeptic + artifact status |
| Payoff | Judge score + judge feedback |
| Budgeted iteration | role budgets + tool budgets + follow-up waves |
| Transparent platform | Run detail UI + artifact graph + operator docs |

## 4. What is directly paper-aligned

- state → action → observation → payoff → next action loop
- typed Principal actions
- temporary organization of agents
- shared state / environment state
- explicit observations
- budgeted iteration
- payoff/judge signal
- transparent decision trace

## 5. What is our engineering adaptation

- EvidenceEngine
- source quality scoring
- market data resolver
- Tavily / Brave / Exa / Firecrawl provider layer
- artifact graph UI
- production smoke tests
- GKE/Vercel/Confluent/Upstash/GCS implementation details

EvidenceEngine is not a named component in the papers. It operationalizes the observation function when moving from closed economic simulations to open-world research over web and market data.

## 6. What we must not claim

- Do not claim this is a full Social Environment Design implementation.
- Do not claim this is a POMG simulator.
- Do not claim this is a learned RL policymaker.
- Do not claim this is a production 1000-agent swarm.
- Do not claim EvidenceEngine is in the papers.
- Do not claim investment forecasting is solved.
- Do not lower verification standards just to produce a confident final answer.

## 7. Current known limitation

Current runs can produce an evidence-limited final when no claims pass source verification. This is epistemically safer than hallucinating, but the PrincipalPolicy is still too passive. The next paper-aligned step is an evidence-repair loop:

```text
failed verification / disputed claims
→ Principal observes unsupported_parts and required_caveats
→ Principal creates targeted repair actions
→ split broad claims
→ search stronger dated/primary sources
→ reverify
→ aggregate only after enough trusted knowledge exists
```
