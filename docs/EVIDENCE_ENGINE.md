# EvidenceEngine architecture

EvidenceEngine is the system abstraction for structured research retrieval and evidence packaging.
It replaces unstructured `web_search` handling with typed requests, provider-aware results, source
quality scoring, raw artifact capture, and verifier-ready evidence bundles.

EvidenceEngine is not a named component from the papers. It operationalizes the observation layer
when moving from closed simulations to open-world research.

## Why legacy `web_search` is insufficient

The original tool path returned provider JSON and asked an agent to summarize it. That was useful
for a demo, but weak for an inspectable trust loop:

- source URLs, snippets, provider scores, and limitations were not normalized;
- source quality was not scored before claims reached the verifier;
- verification lacked fresh support and contradiction retrieval;
- provider failures and fallback paths were hard to audit;
- raw search/fetch outputs were not consistently tied to the final evidence artifact;
- adding providers risked scattering provider-specific parsing across workflow code.

EvidenceEngine centralizes those concerns and returns a typed `EvidenceBundle`.

## Core models

`EvidenceRequest` describes what the runtime wants to know:

- `run_id`, optional `task_id`, optional `claim_id`;
- semantic `branch`;
- `objective` and optional `claim_text`;
- `search_mode`: exploratory, verification, contradiction, primary-source, or historical;
- max sources, preferred/excluded domains, freshness, and cost hints.

`EvidenceBundle` is the output:

- original request;
- generated `EvidenceQuery` objects;
- normalized `EvidenceItem` objects;
- optional raw result artifact URI;
- bundle-level limitations;
- source-quality summary.

`EvidenceItem` extends `EvidenceSource` with:

- provider name;
- URL, title, domain, publisher, publication date;
- snippet, support type, retrieval confidence;
- source tier, quality score, source-quality reason;
- item-level limitations.

The bundle is designed to be compact enough for prompts and structured enough for audit.

## Provider routing

Provider support is intentionally optional and conservative.

- Tavily remains the default provider path.
- Brave is selected for primary-source, contradiction, and site-scoped queries when configured.
- Exa is selected for exploratory and historical semantic discovery when configured.
- Firecrawl is a fetcher, not a search provider: when configured, it can fetch top URLs and store
  cleaner markdown/content in raw artifact output.

If an optional provider is absent, the runtime does not add it. If an optional provider fails and a
Tavily fallback exists, the bundle records a limitation and continues.

## Source quality scoring

EvidenceEngine scores each item deterministically using finance/macro-oriented rules:

- primary domains, government/statistical sources, exchanges, and preferred domains are favored;
- high-quality institutional sources and major financial news receive positive weight;
- weak, crowd-sourced, blog/opinion, missing URL/title/snippet, and weak-content patterns are
  penalized;
- publication date availability and recency affect score;
- numeric and data-specific language adds specificity weight.

The score is not a truth label. It is a routing and verification aid that tells downstream agents
how much trust to place in a source before reading the claim semantics.

## Contradiction search

Verification now asks EvidenceEngine for both supporting evidence and contradiction/context evidence.
Contradiction mode changes generated queries toward risks, counterarguments, criticism, downside
evidence, and rebuttals. The verifier prompt receives both bundles plus linked observations, source
quality summaries, limitations, and raw evidence item references.

If contradiction search fails, verification continues with an explicit caveat instead of silently
pretending the search succeeded.

## Verifier integration

The verifier receives:

- the claim text;
- linked observation summaries;
- support evidence items;
- contradiction/context evidence items;
- source-quality summaries;
- provider limitations and missing-data caveats.

The normalization layer can downgrade a “verified” model verdict to “uncertain” when sources are too
weak, missing, or contradictory. Verified claims should therefore be supported by strong enough
EvidenceEngine output, not just by plausible language in a model response.

## Artifact writing

EvidenceEngine persists raw provider output through the configured artifact store when available.
Workflow code then writes a higher-level evidence artifact containing:

- compact observation summary;
- branch;
- source references;
- raw result artifact URI;
- serialized `EvidenceBundle`.

This preserves both an agent-friendly summary and an inspectable raw trail. Optional Firecrawl fetch
output is included in raw evidence-search artifacts when configured.

## Current limitations

- Source quality rules are deterministic heuristics, not a complete source-reputation system.
- Provider routing is simple and does not yet optimize recall/cost dynamically.
- Firecrawl and Exa are optional hooks; normal tests use fakes and do not make paid calls.
- Search result support type is inferred from search mode and still requires semantic verification.
- Fetching top URLs can improve context but does not guarantee full article access or licensing
  suitability.
- Artifact payloads are JSON-first and not yet exposed through a dedicated evidence review UI.

## Next steps

- Add offline evals that score final reports for source quality and claim traceability.
- Expand provider routing based on branch, source type, and observed provider failure rates.
- Add stronger source reputation metadata and freshness policies.
- Surface evidence bundles in the UI for reviewer inspection.
- Use judge feedback and EvidenceEngine gaps to drive targeted follow-up waves.
