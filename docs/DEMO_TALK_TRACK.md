# Demo talk track

## 1. Two-minute explanation for paper authors

“I did not try to reproduce the full economic simulation setting from the papers. Instead, I asked what systems substrate would be needed to operationalize the Principal / environment / observation / payoff loop in an open-world research setting. The prototype creates a temporary agent organization, produces artifacts, verifies or disputes claims, records Principal actions, and exposes the trace in a UI. The current system is still conservative: if no claims pass verification, it produces an evidence-limited final rather than hallucinating a forecast. The next step is to make the Principal launch targeted evidence-repair actions while budget remains.”

## 2. What to show in the UI

- Principal action timeline
- Agent organization
- EvidenceEngine observations
- Artifact graph
- Verification downgrades
- Evidence-limited final
- Budget remaining
- Proposed evidence-repair next step

## 3. What not to claim

- Not full paper implementation.
- Not full Social Environment Design.
- Not a learned RL policymaker.
- Not production forecasting.
- EvidenceEngine is our observation-layer engineering adaptation.

## 4. How to explain the current failure mode

“The system refused to synthesize unsupported claims. That is good epistemic behavior. The missing piece is Principal-driven repair: if verification fails and budget remains, the Principal should create targeted actions to get stronger evidence rather than stopping.”
