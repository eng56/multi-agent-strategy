"use client";

import { useEffect, useState } from "react";

type PrincipalAction = {
  id: string;
  action_type: string;
  reason: string;
  expected_information_gain: string;
  estimated_cost: number;
  target_branch?: string;
  required_role?: string;
  priority: number;
  status: string;
};

type AgentSpec = {
  id: string;
  parent_id?: string;
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
  branch?: string;
  text_or_summary: string;
  status: string;
  confidence?: number;
  source_refs: string[];
  depends_on_artifact_ids: string[];
  contradicts_artifact_ids: string[];
  supports_artifact_ids: string[];
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
  budget_remaining: number;
  tool_budget_remaining: Record<string, number>;
  agent_count: number;
  last_judge_score?: number;
  last_judge_feedback?: string;
  dead_letter_count: number;
  stop_reasons: string[];
  next_action_candidates: PrincipalAction[];
};

type Detail = {
  run: {
    id: string;
    question: string;
    status: string;
    failure_reason?: string;
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
  tasks: { id: string; title: string; status: string }[];
  claims: { id: string; statement: string; confidence: number }[];
  verifications: { claim_id: string; verdict: string; rationale: string }[];
  final?: { answer: string; sources: string[]; judge_score?: number; judge_feedback?: string };
  run_state?: RunState;
  principal_actions: PrincipalAction[];
  agent_specs: AgentSpec[];
  artifacts: Artifact[];
  dead_letters: DeadLetterRecord[];
  organization_plan?: OrganizationPlan;
};

function ActionList({ actions, empty }: { actions: PrincipalAction[]; empty: string }) {
  if (actions.length === 0) return <p>{empty}</p>;
  return actions.map((action) => (
    <article key={action.id}>
      <strong>{action.action_type}</strong> · {action.status} · priority {action.priority} · {action.expected_information_gain} information gain · ${action.estimated_cost.toFixed(2)} est.
      <p>{action.reason}</p>
      <small>
        role: {action.required_role ?? "any"} {action.target_branch ? `· branch: ${action.target_branch}` : ""}
      </small>
    </article>
  ));
}

export default function RunPage({ params }: { params: Promise<{ id: string }> }) {
  const [id, setId] = useState("");
  const [detail, setDetail] = useState<Detail>();

  useEffect(() => {
    params.then((p) => setId(p.id));
  }, [params]);

  useEffect(() => {
    if (!id) return;
    const load = () => fetch(`/api/runs/${id}`).then((r) => r.json()).then(setDetail);
    load();
    const timer = setInterval(load, 3000);
    return () => clearInterval(timer);
  }, [id]);

  if (!detail) return <main className="shell">Loading run…</main>;

  const b = detail.run.budget;
  const state = detail.run_state;
  const candidates = state?.next_action_candidates ?? [];
  const deadLetters = detail.dead_letters ?? [];

  return (
    <main className="shell">
      <a href="/">← New run</a>
      <p className="tag">{detail.run.status.toUpperCase()}</p>
      <h1>{detail.run.question}</h1>
      <div className="grid">
        <section className="card">
          <h2>Budget</h2>
          <p>${b.spent_usd.toFixed(2)} / ${b.limit_usd.toFixed(2)}</p>
          <p>
            ${b.reserved_usd.toFixed(2)} reserved · Tavily {b.tools.tavily_credits_used}/
            {b.tools.tavily_max_credits} · Market {b.tools.market_data_requests_used}/
            {b.tools.market_data_max_requests}
          </p>
        </section>
        <section className="card">
          <h2>Progress</h2>
          <p>{detail.tasks.length} tasks · {detail.claims.length} claims · {detail.verifications.length} checks · {deadLetters.length} dead letters</p>
        </section>
      </div>

      {state && (
        <section className="card">
          <h2>RunState</h2>
          <p>Phase <strong>{state.current_phase}</strong> · ${state.budget_remaining.toFixed(2)} remaining · {state.agent_count} agent specs</p>
          <p>Verified {state.verified_claim_count} · Rejected {state.rejected_claim_count} · Disputed {state.disputed_claim_count} · Dead letters {state.dead_letter_count}</p>
          <p>Branches: {state.active_branches.join(", ") || "none yet"}</p>
          <p>Tools remaining: Tavily {state.tool_budget_remaining.tavily_credits ?? 0} · Market {state.tool_budget_remaining.market_data_requests ?? 0}</p>
          <h3>Open questions</h3>
          <ul>{state.open_questions.map((q) => <li key={q}>{q}</li>)}</ul>
          {(state.failed_tasks ?? []).length > 0 && (
            <>
              <h3>Failed tasks</h3>
              <ul>{(state.failed_tasks ?? []).map((task) => <li key={task}>{task}</li>)}</ul>
            </>
          )}
          <h3>Known facts</h3>
          <ul>{state.known_facts.map((fact) => <li key={fact}>{fact}</li>)}</ul>
          {state.stop_reasons.length > 0 && <p>Stop reasons: {state.stop_reasons.join(" · ")}</p>}
        </section>
      )}

      {deadLetters.length > 0 && (
        <section className="card">
          <h2>Dead letters</h2>
          {deadLetters.map((record) => (
            <article key={record.id}>
              <strong>{record.event_type}</strong> · {record.worker_role} · {record.classification} · retry {record.retry_count}/{record.max_retries}
              <p>{record.error_type}: {record.error_message}</p>
            </article>
          ))}
        </section>
      )}

      <section className="card">
        <h2>Executed Principal decisions</h2>
        <ActionList actions={detail.principal_actions ?? []} empty="No executed principal decisions recorded yet." />
      </section>

      <section className="card">
        <h2>Candidate next actions</h2>
        <ActionList actions={candidates} empty="No candidate next actions right now." />
      </section>

      <section className="card">
        <h2>Agent organization</h2>
        {detail.organization_plan && <p>Branches: {detail.organization_plan.branches.join(", ")}</p>}
        {detail.agent_specs.length === 0 && <p>No explicit organization plan has been persisted yet.</p>}
        {detail.agent_specs.map((agent) => (
          <article key={agent.id}>
            <strong>{agent.name}</strong> · {agent.role_template} · {agent.branch} · {agent.status}
            <p>{agent.objective}</p>
            <small>
              tools: {agent.allowed_tools.join(", ") || "none"} · tags: {agent.retrieval_tags.join(", ") || "none"} · local budget ${agent.local_budget_usd.toFixed(2)}
            </small>
          </article>
        ))}
      </section>

      <section className="card">
        <h2>Artifact graph</h2>
        {detail.artifacts.length === 0 && <p>No generic artifacts have been persisted yet; legacy observations and claims are shown below.</p>}
        {detail.artifacts.map((artifact) => (
          <article key={artifact.id}>
            <strong>{artifact.artifact_type}</strong> · {artifact.status} · {artifact.branch ?? "unbranched"} {artifact.confidence !== undefined ? `· confidence ${artifact.confidence.toFixed(2)}` : ""}
            <p>{artifact.text_or_summary}</p>
            <small>
              sources {artifact.source_refs.length} · supports {artifact.supports_artifact_ids.length} · contradicts {artifact.contradicts_artifact_ids.length} · depends on {artifact.depends_on_artifact_ids.length}
            </small>
          </article>
        ))}
      </section>

      {detail.final && (
        <section className="card report">
          <h2>Final report</h2>
          <p>{detail.final.answer}</p>
          <h3>Judge</h3>
          <p>{detail.final.judge_score?.toFixed(2)} — {detail.final.judge_feedback}</p>
          <h3>Sources</h3>
          <ul>{detail.final.sources.map((s) => <li key={s}><a href={s} target="_blank">{s}</a></li>)}</ul>
        </section>
      )}

      {detail.run.failure_reason && (
        <section className="card">
          <h2>Stopped</h2>
          <p>{detail.run.failure_reason}</p>
        </section>
      )}

      <section className="card">
        <h2>Legacy evidence</h2>
        {detail.claims.map((c) => {
          const v = detail.verifications.find((x) => x.claim_id === c.id);
          return (
            <article key={c.id}>
              <strong>{v?.verdict ?? "pending"}</strong>
              <p>{c.statement}</p>
              <small>{v?.rationale}</small>
            </article>
          );
        })}
      </section>
    </main>
  );
}
