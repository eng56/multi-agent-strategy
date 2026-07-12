"use client";

import {useEffect, useMemo, useState} from "react";

type PrincipalAction = {
  id: string;
  action_type: string;
  reason: string;
  expected_information_gain: string;
  estimated_cost: number;
  target_branch?: string | null;
  required_role?: string | null;
  priority: number;
  status: string;
  created_at?: string;
};

type AgentSpec = {
  id: string;
  parent_id?: string | null;
  name: string;
  role_template: string;
  branch: string;
  objective: string;
  allowed_tools: string[];
  retrieval_tags: string[];
  local_budget_usd: number;
  status: string;
};

type Artifact = {
  id: string;
  artifact_type: string;
  branch?: string | null;
  text_or_summary: string;
  tags?: string[];
  visibility?: string;
  status: string;
  confidence?: number | null;
  source_refs?: string[];
  legacy_object_type?: string | null;
  legacy_object_id?: string | null;
  parent_artifact_ids?: string[];
  depends_on_artifact_ids?: string[];
  contradicts_artifact_ids?: string[];
  supports_artifact_ids?: string[];
  created_at?: string;
  updated_at?: string;
};

type DeadLetterRecord = {
  id: string;
  event_type: string;
  worker_role: string;
  retry_count: number;
  max_retries: number;
  classification: string;
  error_type: string;
  error_message: string;
};

type OrganizationPlan = {
  root_agent_id: string;
  branches: string[];
  stop_conditions: string[];
};

type ToolUsageSummary = {
  used: number;
  max: number;
  remaining: number;
};

type BudgetSummary = {
  total_limit_usd: number;
  spent_usd: number;
  reserved_usd: number;
  remaining_usd: number;
  role_spent_usd: Record<string, number>;
  role_reserved_usd: Record<string, number>;
  role_protected_usd: Record<string, number>;
  protected_usd: number;
  protected_remaining_usd: number;
  tool_usage: Record<string, ToolUsageSummary>;
  stop_reasons: string[];
};

type CapacitySummary = {
  llm_remaining_usd: number;
  tavily_remaining: number;
  market_data_remaining: number;
  verifier_budget_remaining: number;
  aggregator_budget_protected_remaining: number;
  judge_budget_protected_remaining: number;
  search_exhausted: boolean;
  market_data_exhausted: boolean;
  useful_action_available: boolean;
  branches_repaired?: string[];
};

type RunState = {
  current_phase: string;
  active_branches: string[];
  known_facts: string[];
  open_questions: string[];
  failed_tasks?: string[];
  verified_claim_count: number;
  rejected_claim_count: number;
  disputed_claim_count: number;
  coverage_by_topic: Record<string, string>;
  budget_summary?: BudgetSummary;
  capacity?: CapacitySummary | null;
  budget_remaining: number;
  tool_budget_remaining: Record<string, number>;
  agent_count: number;
  last_judge_score?: number | null;
  last_judge_feedback?: string | null;
  dead_letter_count: number;
  stop_reasons: string[];
  next_action_candidates: PrincipalAction[];
  tavily_used?: number;
  verified_claims?: number;
  verified_claims_per_10_tavily?: number;
  disputed_claims?: number;
  source_quality_distribution?: Record<string, number>;
  claims_created_from_supported_parts?: number;
  aggregator_used?: boolean;
  judge_used?: boolean;
  claims_per_research_observation?: number;
  claims_created_from_verifier_outputs?: number;
  source_meta_claims_rejected?: number;
  market_snapshot_claims_rejected?: number;
  terminal_reason?: string | null;
};

type Claim = {
  id: string;
  statement: string;
  confidence: number;
  sources?: string[];
};

type Verification = {
  claim_id: string;
  verdict: string;
  rationale: string;
};

type Detail = {
  run: {
    id: string;
    question: string;
    status: string;
    final_answer?: string | null;
    failure_reason?: string | null;
    budget: {
      spent_usd: number;
      limit_usd: number;
      reserved_usd: number;
      tools: {
        tavily_credits_used: number;
        tavily_max_credits: number;
        market_data_requests_used: number;
        market_data_max_requests: number;
      };
    };
  };
  tasks: {id: string; title: string; status: string; reason?: string | null}[];
  claims: Claim[];
  verifications: Verification[];
  final?: {
    answer: string;
    verified_claim_ids: string[];
    sources: string[];
    partial?: boolean;
    wave_number?: number;
    judge_score?: number | null;
    judge_feedback?: string | null;
  } | null;
  run_state?: RunState | null;
  principal_actions: PrincipalAction[];
  agent_specs: AgentSpec[];
  artifacts: Artifact[];
  dead_letters: DeadLetterRecord[];
  organization_plan?: OrganizationPlan | null;
};

type ArtifactGroup = {
  name: string;
  artifacts: Artifact[];
};

type RelationKind = "supports" | "contradicts" | "depends_on";

const roleNames = ["planner", "research", "verifier", "aggregator", "judge"];
const relationConfigs: {kind: RelationKind; title: string; empty: string}[] = [
  {kind: "supports", title: "Supports links", empty: "No supports links recorded yet."},
  {kind: "contradicts", title: "Contradicts links", empty: "No contradicts links recorded yet."},
  {kind: "depends_on", title: "Depends on links", empty: "No depends_on links recorded yet."},
];
const followUpPrompts = [
  "Focus only on gold and real yields. Use dated primary or high-quality sources.",
  "Compare USD vs gold under faster Fed cuts, with a 3-month horizon.",
  "Run a narrower version focused only on long-duration bonds and Treasury yields.",
  "Ask for a mechanism-only answer, not a directional forecast.",
  "Ask for a source-quality audit of the evidence.",
];

function asList<T>(value: T[] | undefined | null): T[] {
  return value ?? [];
}

function unique(values: (string | undefined | null)[]) {
  return Array.from(new Set(values.filter((value): value is string => Boolean(value))));
}

function money(value: number | undefined | null) {
  return `$${(value ?? 0).toFixed(2)}`;
}

function percent(value: number | undefined | null) {
  return value === undefined || value === null ? "n/a" : `${Math.round(value * 100)}%`;
}

function formatNumber(value: number | undefined | null, digits = 1) {
  return value === undefined || value === null ? "n/a" : value.toFixed(digits);
}

function shortId(id: string) {
  return id.slice(0, 8);
}

function label(value: string | undefined | null) {
  return value ? value.replace(/_/g, " ") : "unknown";
}

function compact(text: string | undefined | null, max = 120) {
  const normalized = (text ?? "").replace(/\s+/g, " ").trim();
  if (normalized.length <= max) return normalized || "No summary recorded.";
  return `${normalized.slice(0, max - 3)}...`;
}

function formatTime(value: string | undefined) {
  if (!value) return "time not recorded";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function statusClass(status: string | undefined | null) {
  return `pill status-${(status ?? "unknown").replace(/_/g, "-")}`;
}

function runStatusBadges(status: string, isPartialFinal: boolean, finalAnswer: string | undefined | null) {
  const normalized = status.toLowerCase();
  const badges: {label: string; className: string}[] = [];
  if (normalized.includes("fail")) {
    badges.push({label: "Failed", className: "status-failed"});
  } else if (normalized.includes("running") || normalized.includes("created")) {
    badges.push({label: "Running", className: "status-running"});
  } else if (isPartialFinal || normalized.includes("partial") || normalized.includes("evidence")) {
    badges.push({label: "Evidence-limited", className: "status-partial-budget-exhausted"});
  } else if (normalized.includes("complete") || finalAnswer) {
    badges.push({label: "Completed", className: "status-completed"});
  } else {
    badges.push({label: label(status), className: `status-${normalized.replace(/_/g, "-")}`});
  }
  if (normalized.includes("budget_exhausted") || normalized.includes("budget-exhausted")) {
    badges.push({label: "Budget exhausted", className: "status-partial-budget-exhausted"});
  }
  return badges;
}

function yesNo(value: boolean | undefined | null) {
  if (value === undefined || value === null) return "unknown";
  return value ? "yes" : "no";
}

function groupArtifacts(artifacts: Artifact[], key: (artifact: Artifact) => string): ArtifactGroup[] {
  const groups = new Map<string, Artifact[]>();
  for (const artifact of artifacts) {
    const name = key(artifact);
    groups.set(name, [...(groups.get(name) ?? []), artifact]);
  }
  return Array.from(groups, ([name, values]) => ({name, artifacts: values})).sort((a, b) =>
    a.name.localeCompare(b.name),
  );
}

function relationIds(artifact: Artifact, kind: RelationKind) {
  if (kind === "supports") return asList(artifact.supports_artifact_ids);
  if (kind === "contradicts") return asList(artifact.contradicts_artifact_ids);
  return asList(artifact.depends_on_artifact_ids);
}

function artifactName(artifact: Artifact) {
  return `${label(artifact.artifact_type)} ${shortId(artifact.id)}`;
}

function ArtifactLink({artifact}: {artifact: Artifact}) {
  return <a href={`#artifact-${artifact.id}`}>{artifactName(artifact)}</a>;
}

function Empty({children}: {children: React.ReactNode}) {
  return <p className="muted">{children}</p>;
}

function ActionTimeline({actions, empty, limit}: {actions: PrincipalAction[]; empty: string; limit?: number}) {
  const ordered = [...actions].sort((a, b) => {
    const left = a.created_at ? new Date(a.created_at).getTime() : 0;
    const right = b.created_at ? new Date(b.created_at).getTime() : 0;
    return left - right;
  });

  if (ordered.length === 0) return <Empty>{empty}</Empty>;
  const visible = limit && ordered.length > limit ? ordered.slice(-limit) : ordered;
  const hiddenCount = ordered.length - visible.length;

  return (
    <>
      {hiddenCount > 0 && <p className="muted">Showing the latest {visible.length} of {ordered.length} actions.</p>}
      <ol className="timeline">
        {visible.map((action) => (
          <li key={action.id}>
            <div className="timeline-dot" />
            <div className="timeline-body">
              <div className="row-wrap">
                <strong>{label(action.action_type)}</strong>
                <span className={statusClass(action.status)}>{label(action.status)}</span>
                <span className="pill">priority {action.priority}</span>
                <span className="pill">{label(action.expected_information_gain)} gain</span>
              </div>
              <p>{action.reason}</p>
              <small>
                {formatTime(action.created_at)} · role {action.required_role ?? "any"} · branch{" "}
                {action.target_branch ?? "none"} · est. {money(action.estimated_cost)}
              </small>
            </div>
          </li>
        ))}
      </ol>
    </>
  );
}

function ArtifactMiniList({groups}: {groups: ArtifactGroup[]}) {
  if (groups.length === 0) return <Empty>No artifacts recorded yet.</Empty>;
  return (
    <div className="group-stack">
      {groups.map((group) => (
        <details key={group.name} open>
          <summary>
            <span>{label(group.name)}</span>
            <span className="pill">{group.artifacts.length}</span>
          </summary>
          <ul className="compact-list">
            {group.artifacts.map((artifact) => (
              <li key={artifact.id}>
                <ArtifactLink artifact={artifact} /> ·{" "}
                <span className={statusClass(artifact.status)}>{label(artifact.status)}</span> ·{" "}
                {compact(artifact.text_or_summary, 90)}
              </li>
            ))}
          </ul>
        </details>
      ))}
    </div>
  );
}

export default function RunPage({params}: {params: Promise<{id: string}>}) {
  const [id, setId] = useState("");
  const [detail, setDetail] = useState<Detail>();
  const [copiedPrompt, setCopiedPrompt] = useState("");

  useEffect(() => {
    params.then((p) => setId(p.id));
  }, [params]);

  useEffect(() => {
    if (!id) return;
    const load = () =>
      fetch(`/api/runs/${id}`)
        .then((response) => response.json())
        .then(setDetail);
    load();
    const timer = setInterval(load, 3000);
    return () => clearInterval(timer);
  }, [id]);

  const artifactById = useMemo(() => {
    const map = new Map<string, Artifact>();
    for (const artifact of detail?.artifacts ?? []) {
      map.set(artifact.id, artifact);
      if (artifact.legacy_object_id) map.set(artifact.legacy_object_id, artifact);
    }
    return map;
  }, [detail?.artifacts]);

  if (!detail) return <main className="shell">Loading run...</main>;

  const b = detail.run.budget;
  const state = detail.run_state;
  const summary = state?.budget_summary;
  const artifacts = detail.artifacts ?? [];
  const claimsById = new Map(detail.claims.map((claim) => [claim.id, claim]));
  const finalAnswer = detail.final?.answer ?? detail.run.final_answer;
  const finalArtifacts = artifacts
    .filter((artifact) => artifact.artifact_type === "final_report")
    .sort((a, b) => (a.created_at ?? "").localeCompare(b.created_at ?? ""));
  const finalArtifact = finalArtifacts[finalArtifacts.length - 1];
  const isPartialFinal = Boolean(detail.final?.partial);
  const finalLinkedEvidenceIds = unique([
    ...asList(finalArtifact?.depends_on_artifact_ids),
    ...asList(finalArtifact?.supports_artifact_ids),
    ...artifacts
      .filter((artifact) => finalArtifact && asList(artifact.supports_artifact_ids).includes(finalArtifact.id))
      .map((artifact) => artifact.id),
  ]);
  const verifiedKnowledgeIds = unique([
    ...asList(detail.final?.verified_claim_ids),
    ...finalLinkedEvidenceIds.filter((evidenceId) => {
      const artifact = artifactById.get(evidenceId);
      return !isPartialFinal && artifact?.status === "verified";
    }),
  ]);
  const caveatedEvidenceIds = unique(
    finalLinkedEvidenceIds.filter((evidenceId) => {
      const artifact = artifactById.get(evidenceId);
      return isPartialFinal || !artifact || artifact.status !== "verified";
    }),
  );
  const stopReasons = unique([
    detail.run.failure_reason,
    state?.terminal_reason,
    ...asList(state?.stop_reasons),
    ...asList(summary?.stop_reasons),
  ]);
  const verifiedArtifacts = artifacts.filter((artifact) => artifact.status === "verified").length;
  const verifiedKnowledgeClaims = state?.verified_claim_count ?? asList(detail.final?.verified_claim_ids).length;
  const finalArtifactStatus = finalArtifact ? label(finalArtifact.status) : "not recorded";
  const rejectedArtifacts = artifacts.filter((artifact) => artifact.status === "rejected").length;
  const disputedArtifacts = artifacts.filter((artifact) => artifact.status === "disputed").length;
  const fallbackBudgetRemaining = Math.max(0, b.limit_usd - b.spent_usd - b.reserved_usd);
  const tavilyUsed = summary?.tool_usage.tavily_credits?.used ?? b.tools.tavily_credits_used;
  const tavilyMax = summary?.tool_usage.tavily_credits?.max ?? b.tools.tavily_max_credits;
  const marketDataUsed = summary?.tool_usage.market_data_requests?.used ?? b.tools.market_data_requests_used;
  const marketDataMax = summary?.tool_usage.market_data_requests?.max ?? b.tools.market_data_max_requests;
  const primaryStopReason = stopReasons[0] ?? "No terminal stop reason recorded yet.";
  const branchesCovered = unique([
    ...asList(state?.active_branches),
    ...asList(detail.organization_plan?.branches),
    ...artifacts.map((artifact) => artifact.branch ?? null),
  ]).filter((branch) => branch !== "unbranched");
  const inferredAggregatorUsed = detail.principal_actions.some((action) => action.required_role === "aggregator");
  const inferredJudgeUsed =
    detail.final?.judge_score !== undefined ||
    state?.last_judge_score !== undefined ||
    detail.principal_actions.some((action) => action.required_role === "judge");
  const aggregatorUsed = state?.aggregator_used ?? inferredAggregatorUsed;
  const judgeUsed = state?.judge_used ?? inferredJudgeUsed;
  const statusBadges = runStatusBadges(detail.run.status, isPartialFinal, finalAnswer);
  const isEvidenceLimited = statusBadges.some((badge) => badge.label === "Evidence-limited");
  const isTerminalish = statusBadges.some((badge) =>
    ["Completed", "Evidence-limited", "Budget exhausted", "Failed"].includes(badge.label),
  );
  const disputedOrRejectedCount =
    (state?.rejected_claim_count ?? rejectedArtifacts) + (state?.disputed_claim_count ?? disputedArtifacts);
  const finalPreview = finalAnswer ? compact(finalAnswer, 320) : "No final answer has been produced yet.";
  const interpretation = !finalAnswer
    ? "The system is still collecting evidence, verifying claims, or waiting for synthesis."
    : isPartialFinal || isEvidenceLimited
      ? "The system did not find enough public-verified claims for a decision-grade answer. Review caveated evidence and verifier downgrades below."
      : disputedOrRejectedCount >= Math.max(6, verifiedKnowledgeClaims * 3)
        ? "Many candidate claims were disputed. This usually means the evidence is directionally relevant but not specific or high-quality enough."
        : "Read the final answer together with the verified knowledge claims, caveats, judge feedback, and budget diagnostics.";

  function renderReference(evidenceId: string) {
    const artifact = artifactById.get(evidenceId);
    if (artifact) return <ArtifactLink artifact={artifact} />;
    const claim = claimsById.get(evidenceId);
    if (claim) {
      return (
        <span>
          legacy claim {shortId(evidenceId)} · {compact(claim.statement, 100)}
        </span>
      );
    }
    return <span>unknown artifact {shortId(evidenceId)}</span>;
  }

  function relationRows(kind: RelationKind) {
    return artifacts.flatMap((artifact) =>
      relationIds(artifact, kind).map((targetId) => ({from: artifact, targetId})),
    );
  }

  async function copyFollowUpPrompt(prompt: string) {
    try {
      await navigator.clipboard.writeText(prompt);
      setCopiedPrompt(prompt);
      setTimeout(() => setCopiedPrompt(""), 1800);
    } catch {
      setCopiedPrompt("");
    }
  }

  return (
    <main className="shell run-detail">
      <nav className="top-nav">
        <a href="/">Back to new run</a>
        <a href="/about">About this prototype</a>
      </nav>
      <p className="tag">RUN TRACE</p>
      <h1>{detail.run.question}</h1>

      <section className="card executive-card">
        <div className="section-heading">
          <div>
            <h2>Executive view</h2>
            <p className="muted">A readable summary before the raw Principal, artifact, and verification trace.</p>
          </div>
          <div className="row-wrap">
            {statusBadges.map((badge) => (
              <span className={`pill status-badge ${badge.className}`} key={badge.label}>
                {badge.label}
              </span>
            ))}
          </div>
        </div>

        <div className="exec-final">
          <h3>Final answer / partial final</h3>
          <p>{finalPreview}</p>
        </div>

        <div className="exec-metrics">
          <div>
            <small>Status</small>
            <strong>{label(detail.run.status)}</strong>
          </div>
          <div>
            <small>Verified knowledge claims</small>
            <strong>{verifiedKnowledgeClaims}</strong>
          </div>
          <div>
            <small>Verified artifacts</small>
            <strong>{verifiedArtifacts}</strong>
          </div>
          <div>
            <small>Disputed / rejected</small>
            <strong>{disputedOrRejectedCount}</strong>
          </div>
          <div>
            <small>Search budget used</small>
            <strong>
              {tavilyUsed}/{tavilyMax}
            </strong>
          </div>
          <div>
            <small>Market-data used</small>
            <strong>
              {marketDataUsed}/{marketDataMax}
            </strong>
          </div>
          <div>
            <small>LLM budget used</small>
            <strong>
              {money(summary?.spent_usd ?? b.spent_usd)} / {money(summary?.total_limit_usd ?? b.limit_usd)}
            </strong>
          </div>
          <div>
            <small>Branches covered</small>
            <strong>{branchesCovered.length ? branchesCovered.join(", ") : "none recorded"}</strong>
          </div>
          <div>
            <small>Aggregator used</small>
            <strong>{yesNo(aggregatorUsed)}</strong>
          </div>
          <div>
            <small>Judge used</small>
            <strong>{yesNo(judgeUsed)}</strong>
          </div>
          <div>
            <small>Final report artifact</small>
            <strong>{finalArtifactStatus}</strong>
          </div>
          <div className="wide-metric">
            <small>Primary stop reason</small>
            <strong>{compact(primaryStopReason, 160)}</strong>
          </div>
        </div>

        <div className="interpretation-box">
          <strong>How to interpret this run</strong>
          <p>{interpretation}</p>
        </div>
      </section>

      <section className="card">
        <h2>Final answer</h2>
        {finalAnswer ? (
          <>
            {detail.final?.partial && (
              <p className="interpretation-box">
                Evidence-limited final: the system refused to present weak or disputed claims as decision-grade support.
              </p>
            )}
            <div className="report">{finalAnswer}</div>
            <div className="meta-grid">
              <p>
                Judge score: <strong>{percent(detail.final?.judge_score ?? state?.last_judge_score)}</strong>
              </p>
              <p>
                Wave: <strong>{detail.final?.wave_number ?? 0}</strong>
              </p>
              <p>
                Sources: <strong>{detail.final?.sources.length ?? 0}</strong>
              </p>
            </div>
            {(detail.final?.judge_feedback ?? state?.last_judge_feedback) && (
              <>
                <h3>Judge feedback</h3>
                <p>{detail.final?.judge_feedback ?? state?.last_judge_feedback}</p>
              </>
            )}
            <h3>Verified knowledge claims</h3>
            {verifiedKnowledgeIds.length > 0 ? (
              <ul>
                {verifiedKnowledgeIds.map((evidenceId) => (
                  <li key={evidenceId}>{renderReference(evidenceId)}</li>
                ))}
              </ul>
            ) : (
              <Empty>No verified knowledge claims were accepted as final support.</Empty>
            )}
            <h3>Caveated evidence considered</h3>
            {caveatedEvidenceIds.length > 0 ? (
              <ul>
                {caveatedEvidenceIds.map((evidenceId) => (
                  <li key={evidenceId}>{renderReference(evidenceId)}</li>
                ))}
              </ul>
            ) : (
              <Empty>No caveated evidence links recorded for this final.</Empty>
            )}
            <h3>Verified final artifact status</h3>
            {finalArtifact ? (
              <p>
                <ArtifactLink artifact={finalArtifact} /> ·{" "}
                <span className={statusClass(finalArtifact.status)}>{label(finalArtifact.status)}</span> · visibility{" "}
                {label(finalArtifact.visibility)}
              </p>
            ) : (
              <Empty>No final report artifact recorded yet.</Empty>
            )}
            {asList(detail.final?.sources).length > 0 && (
              <>
                <h3>
                  {isPartialFinal
                    ? "Evidence reviewed but not accepted as final support"
                    : "Sources supporting verified final answer"}
                </h3>
                <ul>
                  {asList(detail.final?.sources).map((source) => (
                    <li key={source}>
                      <a href={source} target="_blank" rel="noreferrer">
                        {source}
                      </a>
                    </li>
                  ))}
                </ul>
              </>
            )}
          </>
        ) : (
          <Empty>No final answer has been produced yet.</Empty>
        )}
      </section>

      {isTerminalish && (
        <section className="card">
          <h2>Try a better follow-up prompt</h2>
          <p className="muted">Copy a narrower prompt, then start a new run from the landing page.</p>
          <div className="followup-grid">
            {followUpPrompts.map((prompt) => (
              <button className="copy-card" type="button" key={prompt} onClick={() => copyFollowUpPrompt(prompt)}>
                <span>{prompt}</span>
                <small>{copiedPrompt === prompt ? "Copied" : "Copy prompt"}</small>
              </button>
            ))}
          </div>
        </section>
      )}

      <section className="card">
        <h2>Budget summary</h2>
        <div className="exec-metrics">
          <div>
            <small>LLM spend</small>
            <strong>
              {money(summary?.spent_usd ?? b.spent_usd)} / {money(summary?.total_limit_usd ?? b.limit_usd)}
            </strong>
          </div>
          <div>
            <small>Reserved</small>
            <strong>{money(summary?.reserved_usd ?? b.reserved_usd)}</strong>
          </div>
          <div>
            <small>Remaining</small>
            <strong>{money(summary?.remaining_usd ?? fallbackBudgetRemaining)}</strong>
          </div>
          <div>
            <small>Tavily/search</small>
            <strong>
              {tavilyUsed}/{tavilyMax}
            </strong>
          </div>
          <div>
            <small>Market data</small>
            <strong>
              {marketDataUsed}/{marketDataMax}
            </strong>
          </div>
          <div>
            <small>Protected remaining</small>
            <strong>{summary ? money(summary.protected_remaining_usd) : "n/a"}</strong>
          </div>
        </div>
        <p className="muted">
          LLM budget funds model reasoning. Tavily credits fund web evidence search. Market-data requests fund market
          snapshots. A run can stop because search budget is exhausted even if LLM dollars remain.
        </p>
        {state?.capacity && (
          <div className="meta-grid">
            <p>
              LLM capacity: <strong>{money(state.capacity.llm_remaining_usd)}</strong>
            </p>
            <p>
              Verifier: <strong>{money(state.capacity.verifier_budget_remaining)}</strong>
            </p>
            <p>
              Aggregator protected: <strong>{money(state.capacity.aggregator_budget_protected_remaining)}</strong>
            </p>
            <p>
              Judge protected: <strong>{money(state.capacity.judge_budget_protected_remaining)}</strong>
            </p>
          </div>
        )}
        {summary && (
          <details className="inline-details" open>
            <summary>
              <span>Role budget</span>
              <span className="pill">protected caps</span>
            </summary>
            <div className="table">
              <div className="table-row table-head">
                <span>Role</span>
                <span>Spent</span>
                <span>Reserved</span>
                <span>Protected</span>
              </div>
              {roleNames.map((role) => (
                <div className="table-row" key={role}>
                  <span>{role}</span>
                  <span>{money(summary.role_spent_usd[role])}</span>
                  <span>{money(summary.role_reserved_usd[role])}</span>
                  <span>{money(summary.role_protected_usd[role])}</span>
                </div>
              ))}
            </div>
          </details>
        )}
      </section>

      <details className="card detail-section" open>
        <summary>
          <span>Principal actions timeline</span>
          <span className="pill">{detail.principal_actions.length} actions</span>
        </summary>
        <div className="detail-body">
          <p className="muted">Shows what the controller decided or proposed.</p>
          <ActionTimeline
            actions={detail.principal_actions ?? []}
            empty="No executed principal decisions recorded yet."
            limit={12}
          />
          {asList(state?.next_action_candidates).length > 0 && (
            <>
              <h3>Candidate next actions</h3>
              <ActionTimeline
                actions={asList(state?.next_action_candidates)}
                empty="No candidate next actions right now."
                limit={8}
              />
            </>
          )}
        </div>
      </details>

      <section className="card">
        <h2>Research state</h2>
        {state ? (
          <>
            <p>
              Branches: <strong>{state.active_branches.join(", ") || "none yet"}</strong>
            </p>
            <p>
              Tools remaining: Tavily{" "}
              {state.capacity?.tavily_remaining ?? state.tool_budget_remaining.tavily_credits ?? 0} · Market{" "}
              {state.capacity?.market_data_remaining ?? state.tool_budget_remaining.market_data_requests ?? 0}
            </p>
            {state.capacity && (
              <p>
                Search exhausted: <strong>{state.capacity.search_exhausted ? "yes" : "no"}</strong> · Market data
                exhausted: <strong>{state.capacity.market_data_exhausted ? "yes" : "no"}</strong> · Useful action
                available: <strong>{state.capacity.useful_action_available ? "yes" : "no"}</strong>
              </p>
            )}
            <div className="meta-grid">
              <p>
                Verified per 10 Tavily: <strong>{formatNumber(state.verified_claims_per_10_tavily, 2)}</strong>
              </p>
              <p>
                Claims per research observation:{" "}
                <strong>{formatNumber(state.claims_per_research_observation, 2)}</strong>
              </p>
              <p>
                Claims from verifier outputs: <strong>{state.claims_created_from_verifier_outputs ?? 0}</strong>
              </p>
              <p>
                Source-meta rejected: <strong>{state.source_meta_claims_rejected ?? 0}</strong>
              </p>
              <p>
                Market snapshots rejected: <strong>{state.market_snapshot_claims_rejected ?? 0}</strong>
              </p>
              <p>
                Supported-part claims: <strong>{state.claims_created_from_supported_parts ?? 0}</strong>
              </p>
            </div>
            {state.source_quality_distribution && Object.keys(state.source_quality_distribution).length > 0 && (
              <>
                <h3>Source quality distribution</h3>
                <div className="row-wrap">
                  {Object.entries(state.source_quality_distribution).map(([tier, count]) => (
                    <span className="pill" key={tier}>
                      {label(tier)} {count}
                    </span>
                  ))}
                </div>
              </>
            )}
            <div className="two-col">
              <div>
                <h3>Known facts</h3>
                {state.known_facts.length > 0 ? (
                  <ul>
                    {state.known_facts.map((fact) => (
                      <li key={fact}>{fact}</li>
                    ))}
                  </ul>
                ) : (
                  <Empty>No verified facts yet.</Empty>
                )}
              </div>
              <div>
                <h3>Open questions</h3>
                {state.open_questions.length > 0 ? (
                  <ul>
                    {state.open_questions.map((question) => (
                      <li key={question}>{question}</li>
                    ))}
                  </ul>
                ) : (
                  <Empty>No open questions recorded.</Empty>
                )}
              </div>
            </div>
          </>
        ) : (
          <Empty>No RunState snapshot is available yet.</Empty>
        )}
      </section>

      <section className="card">
        <h2>Failure and stop reasons</h2>
        {stopReasons.length > 0 ? (
          <ul>
            {stopReasons.map((reason) => (
              <li key={reason}>{reason}</li>
            ))}
          </ul>
        ) : (
          <Empty>No failure or stop reason recorded yet.</Empty>
        )}
        {asList(state?.failed_tasks).length > 0 && (
          <>
            <h3>Failed tasks</h3>
            <ul>
              {asList(state?.failed_tasks).map((task) => (
                <li key={task}>{task}</li>
              ))}
            </ul>
          </>
        )}
      </section>

      <details className="card detail-section">
        <summary>
          <span>Agent organization</span>
          <span className="pill">{detail.agent_specs.length} AgentSpecs</span>
        </summary>
        <div className="detail-body">
          <p className="muted">
            Shows the temporary logical research organization. These are AgentSpecs, not Kubernetes pods.
          </p>
          {detail.organization_plan && <p>Branches: {detail.organization_plan.branches.join(", ")}</p>}
          {detail.agent_specs.length === 0 && <Empty>No explicit organization plan has been persisted yet.</Empty>}
          {detail.agent_specs.map((agent) => (
            <article key={agent.id}>
              <div className="row-wrap">
                <strong>{agent.name}</strong>
                <span className="pill">{agent.role_template}</span>
                <span className="pill">{agent.branch}</span>
                <span className={statusClass(agent.status)}>{label(agent.status)}</span>
              </div>
              <p>{agent.objective}</p>
              <small>
                tools: {agent.allowed_tools.join(", ") || "none"} · tags:{" "}
                {agent.retrieval_tags.join(", ") || "none"} · local budget {money(agent.local_budget_usd)}
              </small>
            </article>
          ))}
        </div>
      </details>

      <details className="card detail-section">
        <summary>
          <span>Full artifact list</span>
          <span className="pill">{artifacts.length} artifacts</span>
        </summary>
        <div className="detail-body">
          <p className="muted">
            Graph of observations, claims, verifications, final reports, and their dependency links.
          </p>
          <div className="group-grid">
            <div>
              <h3>By type</h3>
              <ArtifactMiniList groups={groupArtifacts(artifacts, (artifact) => artifact.artifact_type)} />
            </div>
            <div>
              <h3>By status</h3>
              <ArtifactMiniList groups={groupArtifacts(artifacts, (artifact) => artifact.status)} />
            </div>
            <div>
              <h3>By branch</h3>
              <ArtifactMiniList groups={groupArtifacts(artifacts, (artifact) => artifact.branch ?? "unbranched")} />
            </div>
          </div>
          <h3>Raw artifacts</h3>
          {artifacts.length === 0 && (
            <Empty>No generic artifacts have been persisted yet; legacy claims are shown below.</Empty>
          )}
          {artifacts.map((artifact) => (
            <article id={`artifact-${artifact.id}`} key={artifact.id}>
              <div className="row-wrap">
                <strong>{artifactName(artifact)}</strong>
                <span className={statusClass(artifact.status)}>{label(artifact.status)}</span>
                <span className="pill">{artifact.branch ?? "unbranched"}</span>
                <span className="pill">{label(artifact.visibility)}</span>
                {artifact.confidence !== undefined && artifact.confidence !== null && (
                  <span className="pill">confidence {percent(artifact.confidence)}</span>
                )}
              </div>
              <p>{artifact.text_or_summary}</p>
              <small>
                sources {asList(artifact.source_refs).length} · supports {asList(artifact.supports_artifact_ids).length} ·
                contradicts {asList(artifact.contradicts_artifact_ids).length} · depends on{" "}
                {asList(artifact.depends_on_artifact_ids).length}
                {artifact.legacy_object_type ? ` · legacy ${artifact.legacy_object_type}` : ""}
              </small>
            </article>
          ))}
        </div>
      </details>

      <details className="card detail-section">
        <summary>
          <span>Evidence graph basics</span>
          <span className="pill">links</span>
        </summary>
        <div className="detail-body">
          {relationConfigs.map((config) => {
            const rows = relationRows(config.kind);
            return (
              <div className="relation-block" key={config.kind}>
                <h3>{config.title}</h3>
                {rows.length > 0 ? (
                  <div className="relation-list">
                    {rows.map((row) => (
                      <div className="relation-row" key={`${config.kind}-${row.from.id}-${row.targetId}`}>
                        <span>
                          <ArtifactLink artifact={row.from} />
                        </span>
                        <span className="relation-kind">{config.kind}</span>
                        <span>{renderReference(row.targetId)}</span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <Empty>{config.empty}</Empty>
                )}
              </div>
            );
          })}
        </div>
      </details>

      {detail.dead_letters.length > 0 && (
        <details className="card detail-section">
          <summary>
            <span>Dead letters</span>
            <span className="pill">{detail.dead_letters.length}</span>
          </summary>
          <div className="detail-body">
            {detail.dead_letters.map((record) => (
              <article key={record.id}>
                <strong>{record.event_type}</strong> · {record.worker_role} · {record.classification} · retry{" "}
                {record.retry_count}/{record.max_retries}
                <p>
                  {record.error_type}: {record.error_message}
                </p>
              </article>
            ))}
          </div>
        </details>
      )}

      <details className="card detail-section">
        <summary>
          <span>Legacy evidence</span>
          <span className="pill">{detail.claims.length} claims</span>
        </summary>
        <div className="detail-body">
          <p className="muted">Candidate propositions extracted from observations and checked by the trust layer.</p>
          {detail.claims.length === 0 && <Empty>No legacy claims recorded yet.</Empty>}
          {detail.claims.map((claim) => {
            const verification = detail.verifications.find((item) => item.claim_id === claim.id);
            return (
              <article key={claim.id}>
                <span className={statusClass(verification?.verdict ?? "pending")}>
                  {label(verification?.verdict ?? "pending")}
                </span>
                <p>{claim.statement}</p>
                <small>
                  confidence {percent(claim.confidence)} {verification?.rationale ? `· ${verification.rationale}` : ""}
                </small>
              </article>
            );
          })}
        </div>
      </details>
    </main>
  );
}
