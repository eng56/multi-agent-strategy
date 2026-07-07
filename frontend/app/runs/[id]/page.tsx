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
  budget_remaining: number;
  tool_budget_remaining: Record<string, number>;
  agent_count: number;
  last_judge_score?: number | null;
  last_judge_feedback?: string | null;
  dead_letter_count: number;
  stop_reasons: string[];
  next_action_candidates: PrincipalAction[];
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

function ActionTimeline({actions, empty}: {actions: PrincipalAction[]; empty: string}) {
  const ordered = [...actions].sort((a, b) => {
    const left = a.created_at ? new Date(a.created_at).getTime() : 0;
    const right = b.created_at ? new Date(b.created_at).getTime() : 0;
    return left - right;
  });

  if (ordered.length === 0) return <Empty>{empty}</Empty>;

  return (
    <ol className="timeline">
      {ordered.map((action) => (
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
    ...finalLinkedEvidenceIds.filter((id) => {
      const artifact = artifactById.get(id);
      return !isPartialFinal && artifact?.status === "verified";
    }),
  ]);
  const caveatedEvidenceIds = unique(
    finalLinkedEvidenceIds.filter((id) => {
      const artifact = artifactById.get(id);
      return isPartialFinal || !artifact || artifact.status !== "verified";
    }),
  );
  const stopReasons = unique([
    detail.run.failure_reason,
    ...asList(state?.stop_reasons),
    ...asList(summary?.stop_reasons),
  ]);
  const verifiedArtifacts = artifacts.filter((artifact) => artifact.status === "verified").length;
  const rejectedArtifacts = artifacts.filter((artifact) => artifact.status === "rejected").length;
  const disputedArtifacts = artifacts.filter((artifact) => artifact.status === "disputed").length;
  const fallbackBudgetRemaining = Math.max(0, b.limit_usd - b.spent_usd - b.reserved_usd);

  function renderReference(id: string) {
    const artifact = artifactById.get(id);
    if (artifact) return <ArtifactLink artifact={artifact} />;
    const claim = claimsById.get(id);
    if (claim) return <span>legacy claim {shortId(id)} · {compact(claim.statement, 100)}</span>;
    return <span>unknown artifact {shortId(id)}</span>;
  }

  function relationRows(kind: RelationKind) {
    return artifacts.flatMap((artifact) =>
      relationIds(artifact, kind).map((targetId) => ({from: artifact, targetId})),
    );
  }

  return (
    <main className="shell run-detail">
      <a href="/">Back to new run</a>
      <p className="tag">{detail.run.status.toUpperCase()}</p>
      <h1>{detail.run.question}</h1>

      <div className="grid">
        <section className="card metric-card">
          <h2>Run summary</h2>
          <p>
            Phase <strong>{state?.current_phase ?? detail.run.status}</strong>
          </p>
          <p>
            {detail.tasks.length} tasks · {detail.claims.length} claims · {detail.verifications.length} checks ·{" "}
            {artifacts.length} artifacts
          </p>
          <p>
            Verified {verifiedArtifacts || state?.verified_claim_count || 0} · Rejected{" "}
            {rejectedArtifacts || state?.rejected_claim_count || 0} · Disputed{" "}
            {disputedArtifacts || state?.disputed_claim_count || 0}
          </p>
        </section>

        <section className="card metric-card">
          <h2>Budget summary</h2>
          <p>
            {money(summary?.spent_usd ?? b.spent_usd)} spent /{" "}
            {money(summary?.total_limit_usd ?? b.limit_usd)} limit
          </p>
          <p>
            {money(summary?.reserved_usd ?? b.reserved_usd)} reserved ·{" "}
            {money(summary?.remaining_usd ?? fallbackBudgetRemaining)} remaining
          </p>
          <p>
            Tavily {summary?.tool_usage.tavily_credits?.used ?? b.tools.tavily_credits_used}/
            {summary?.tool_usage.tavily_credits?.max ?? b.tools.tavily_max_credits} · Market data{" "}
            {summary?.tool_usage.market_data_requests?.used ?? b.tools.market_data_requests_used}/
            {summary?.tool_usage.market_data_requests?.max ?? b.tools.market_data_max_requests}
          </p>
          {summary ? (
            <p>
              Protected {money(summary.protected_usd)} · {money(summary.protected_remaining_usd)} remaining
            </p>
          ) : (
            <Empty>Detailed role budget summary is not available yet.</Empty>
          )}
        </section>
      </div>

      {summary && (
        <section className="card">
          <h2>Role budget</h2>
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
        </section>
      )}

      <section className="card">
        <h2>Principal actions timeline</h2>
        <ActionTimeline
          actions={detail.principal_actions ?? []}
          empty="No executed principal decisions recorded yet."
        />
        {asList(state?.next_action_candidates).length > 0 && (
          <>
            <h3>Candidate next actions</h3>
            <ActionTimeline actions={asList(state?.next_action_candidates)} empty="No candidate next actions right now." />
          </>
        )}
      </section>

      <section className="card">
        <h2>Final answer</h2>
        {finalAnswer ? (
          <>
            {detail.final?.partial && <p className="pill status-partial-budget-exhausted">partial final</p>}
            <div className="report">{finalAnswer}</div>
            <div className="meta-grid">
              <p>Judge score: <strong>{percent(detail.final?.judge_score ?? state?.last_judge_score)}</strong></p>
              <p>Wave: <strong>{detail.final?.wave_number ?? 0}</strong></p>
              <p>Sources: <strong>{detail.final?.sources.length ?? 0}</strong></p>
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

      {state && (
        <section className="card">
          <h2>Research state</h2>
          <p>
            Branches: <strong>{state.active_branches.join(", ") || "none yet"}</strong>
          </p>
          <p>
            Tools remaining: Tavily {state.tool_budget_remaining.tavily_credits ?? 0} · Market{" "}
            {state.tool_budget_remaining.market_data_requests ?? 0}
          </p>
          <div className="two-col">
            <div>
              <h3>Known facts</h3>
              {state.known_facts.length > 0 ? (
                <ul>{state.known_facts.map((fact) => <li key={fact}>{fact}</li>)}</ul>
              ) : (
                <Empty>No verified facts yet.</Empty>
              )}
            </div>
            <div>
              <h3>Open questions</h3>
              {state.open_questions.length > 0 ? (
                <ul>{state.open_questions.map((question) => <li key={question}>{question}</li>)}</ul>
              ) : (
                <Empty>No open questions recorded.</Empty>
              )}
            </div>
          </div>
        </section>
      )}

      <section className="card">
        <h2>Artifact list</h2>
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
      </section>

      <section className="card">
        <h2>Evidence graph basics</h2>
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
      </section>

      <section className="card">
        <h2>Artifacts</h2>
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
              sources {asList(artifact.source_refs).length} · supports{" "}
              {asList(artifact.supports_artifact_ids).length} · contradicts{" "}
              {asList(artifact.contradicts_artifact_ids).length} · depends on{" "}
              {asList(artifact.depends_on_artifact_ids).length}
              {artifact.legacy_object_type ? ` · legacy ${artifact.legacy_object_type}` : ""}
            </small>
          </article>
        ))}
      </section>

      {detail.dead_letters.length > 0 && (
        <section className="card">
          <h2>Dead letters</h2>
          {detail.dead_letters.map((record) => (
            <article key={record.id}>
              <strong>{record.event_type}</strong> · {record.worker_role} · {record.classification} · retry{" "}
              {record.retry_count}/{record.max_retries}
              <p>
                {record.error_type}: {record.error_message}
              </p>
            </article>
          ))}
        </section>
      )}

      <section className="card">
        <h2>Agent organization</h2>
        {detail.organization_plan && <p>Branches: {detail.organization_plan.branches.join(", ")}</p>}
        {detail.agent_specs.length === 0 && <Empty>No explicit organization plan has been persisted yet.</Empty>}
        {detail.agent_specs.map((agent) => (
          <article key={agent.id}>
            <strong>{agent.name}</strong> · {agent.role_template} · {agent.branch} · {agent.status}
            <p>{agent.objective}</p>
            <small>
              tools: {agent.allowed_tools.join(", ") || "none"} · tags:{" "}
              {agent.retrieval_tags.join(", ") || "none"} · local budget {money(agent.local_budget_usd)}
            </small>
          </article>
        ))}
      </section>

      <section className="card">
        <h2>Legacy evidence</h2>
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
      </section>
    </main>
  );
}
