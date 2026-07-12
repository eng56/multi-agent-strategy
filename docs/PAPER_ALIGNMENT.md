# Paper alignment

## 1. Project goal

This project is not a generic multi-agent investment research chatbot.

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

The system is designed to make the state, action, observation, trust-update, payoff, and stopping decisions explicit and inspectable.

## 2. Paper concepts

### A. Social Environment Design

Social Environment Design proposes an AI-assisted automated policymaking framework for general economic environments. The policymaking process is organized around explicit policy objectives or social welfare objectives, with possible voting or objective-selection mechanisms to decide which objective the policy process should optimize.

The paper framing emphasizes simulation-based policy analysis: candidate policies can be evaluated inside modeled social or economic environments before any real-world use. The work connects ideas from reinforcement learning, economics and computation, EconCS, and computational social choice. The central abstraction is a policymaker interacting with an environment under an objective and receiving feedback about outcomes.

This repo does not implement the full Social Environment Design framework. It borrows the high-level Principal / environment / objective / action / payoff framing and adapts it to open-world research over text, sources, artifacts, and market data.

### B. Large Legislative Models

Large Legislative Models explores LLMs as sample-efficient policymakers in socially complex multi-agent economic simulations. The motivation is that RL-based policy generation can be sample inefficient and inflexible when the policy must incorporate nuanced context, history, textual rules, or social preferences.

The key abstraction for this repo is:

```text
state / history / observations / payoff → LLM or deterministic Principal → action
```

The repo uses this abstraction in `RunState`, `PrincipalPolicy`, and typed `PrincipalAction` decisions. The Principal can be deterministic, LLM-assisted, or hybrid; the important part is that it observes state and payoff signals and selects explicit next actions.

### C. Cooperative AI Policymaking Platform

The Cooperative AI Policymaking Platform describes a transparent AI-assisted policy support platform combining economic time-series and text modeling, perspective elicitation, and a public or policy-facing interface.

This repo is not that platform. It borrows the transparent decision-support framing: the run detail UI should expose Principal actions, temporary agent organization, evidence observations, artifact graph structure, trust status, verifier downgrades, budget usage, and synthesis limitations.

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
