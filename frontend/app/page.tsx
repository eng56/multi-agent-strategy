"use client";

import {FormEvent, useEffect, useState} from "react";

const roles = ["planner", "utility", "research", "verifier", "aggregator", "judge"] as const;
const defaultModels: Record<string, string> = {
  planner: "anthropic/claude-sonnet-5",
  utility: "openai/gpt-5.4-nano",
  research: "qwen/qwen3.6-flash",
  verifier: "openai/gpt-5.4-mini",
  aggregator: "anthropic/claude-sonnet-5",
  judge: "openai/gpt-5.4-mini",
};
const weights: Record<string, number> = {
  planner: 0.1,
  utility: 0.05,
  research: 0.3,
  verifier: 0.2,
  aggregator: 0.25,
  judge: 0.1,
};
const createRunTimeoutMs = 60000;
const defaultBudgetUsd = 5;
const defaultToolLimits = {tavily_max_credits: 50, market_data_max_requests: 20};

const workflowSteps = [
  "objective",
  "Principal observes RunState",
  "Principal selects typed actions",
  "agents/tools produce observations",
  "observations become artifacts",
  "verifier/skeptic updates trust",
  "aggregator synthesizes",
  "judge/payoff evaluates",
  "Principal repairs evidence or stops under budget",
];

const suggestedPrompts = [
  {
    label: "Macro / cross-asset",
    title: "Fed rate-cut path",
    prompt:
      "If the Fed signals a faster-than-expected rate-cut path over the next 3 months, what is the likely impact on U.S. equities, the U.S. dollar, gold, and long-duration bonds?",
  },
  {
    label: "Inflation shock",
    title: "Oil price jump",
    prompt:
      "If oil prices jump 20% over the next quarter, what is the likely impact on U.S. inflation expectations, equities, the U.S. dollar, and long-duration bonds?",
  },
  {
    label: "Inflation shock",
    title: "CPI surprise",
    prompt:
      "If U.S. CPI comes in materially above expectations next month, what is the likely impact on the S&P 500, the U.S. dollar, gold, and 10-year Treasury yields?",
  },
  {
    label: "Commodities",
    title: "China stimulus",
    prompt:
      "If China announces a large fiscal and credit stimulus package, what is the likely impact on commodities, emerging-market FX, U.S. equities, and inflation expectations?",
  },
  {
    label: "Technology / equities",
    title: "AI capex slowdown",
    prompt:
      "If major cloud providers signal slower AI capex growth, what is the likely impact on semiconductors, cloud software, broader equities, and credit spreads?",
  },
  {
    label: "Template",
    title: "Custom comparison",
    prompt:
      "Compare [asset A], [asset B], and [asset C] under [macro event] over [time horizon]. Separate main thesis, evidence, risks, what would change the conclusion, and confidence.",
  },
];

export default function Home() {
  const [question, setQuestion] = useState("");
  const [budget, setBudget] = useState(defaultBudgetUsd);
  const [judge, setJudge] = useState(true);
  const [models, setModels] = useState(defaultModels);
  const [available, setAvailable] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [toolLimits, setToolLimits] = useState(defaultToolLimits);
  const [caps, setCaps] = useState<Record<string, number>>(
    Object.fromEntries(roles.map((role) => [role, +(defaultBudgetUsd * weights[role]).toFixed(2)])),
  );

  useEffect(() => {
    fetch("/api/models")
      .then((response) => (response.ok ? response.json() : {models: []}))
      .then((data) => setAvailable(data.models ?? []))
      .catch(() => setAvailable([]));
  }, []);

  function updateBudget(value: number) {
    setBudget(value);
    setCaps(Object.fromEntries(roles.map((role) => [role, +(value * weights[role]).toFixed(2)])));
  }

  function usePrompt(prompt: string) {
    setQuestion(prompt);
    document.getElementById("research-question")?.scrollIntoView({behavior: "smooth", block: "center"});
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError("");
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), createRunTimeoutMs);
    try {
      const policies = Object.fromEntries(
        roles.map((role) => [
          role,
          !judge && role === "judge"
            ? null
            : {
                model: models[role],
                cap_usd: caps[role],
                protected_usd: ["aggregator", "judge"].includes(role) ? caps[role] : 0,
                max_output_tokens: role === "aggregator" ? 4000 : 2000,
                max_call_cost_usd: ["aggregator", "judge"].includes(role)
                  ? caps[role]
                  : Math.max(0.05, +(caps[role] / 4).toFixed(2)),
              },
        ]),
      );
      const response = await fetch("/api/runs", {
        method: "POST",
        headers: {"content-type": "application/json"},
        body: JSON.stringify({question, llm_budget_usd: budget, models: policies, tool_budget: toolLimits}),
        signal: controller.signal,
      });
      const text = await response.text();
      let result: {id?: string; detail?: string};
      try {
        result = JSON.parse(text);
      } catch {
        result = {
          detail: `Run creation returned a non-JSON response (${response.status} ${response.statusText}): ${text.slice(0, 500)}`,
        };
      }
      if (response.ok && result.id) {
        location.href = `/runs/${result.id}`;
        return;
      }
      setError(result.detail ?? "Validation failed");
    } catch (exc) {
      setError(
        exc instanceof DOMException && exc.name === "AbortError"
          ? "Provider validation timed out after 60 seconds. Check the backend logs and provider credentials, then try again."
          : "Could not create the run. Check the backend deployment and try again.",
      );
    } finally {
      clearTimeout(timeout);
      setBusy(false);
    }
  }

  return (
    <main className="shell landing">
      <nav className="top-nav">
        <span className="tag">PASSWORD-LOCKED MANAGED DEMO</span>
        <a href="/about">About this prototype</a>
      </nav>

      <section className="hero-panel">
        <p className="tag">Open-world research control loop</p>
        <h1>Paper-inspired multi-agent research system</h1>
        <p className="lede">An open-world prototype of a Principal / environment / observation / payoff loop.</p>
        <p>
          This is not just an investment chatbot. The system creates a temporary organization of logical agents,
          gathers observations from web search and market data, writes artifacts, verifies or disputes claims, and
          exposes the full control trace.
        </p>
      </section>

      <section className="workflow-strip" aria-label="Research workflow">
        {workflowSteps.map((step, index) => (
          <div className="workflow-step" key={step}>
            <span className="step-index">{index + 1}</span>
            <span>{step}</span>
            {index < workflowSteps.length - 1 && <span className="step-arrow">&rarr;</span>}
          </div>
        ))}
      </section>

      <details className="note-panel" open>
        <summary>
          <span>Before you try it</span>
          <span className="pill">read first</span>
        </summary>
        <div className="note-content">
          <p>
            This prototype is designed to show the control loop, not to produce a black-box answer. The most
            interesting part is the trace: which agents were created, what evidence was gathered, which claims were
            verified or disputed, how budget was used, and why the Principal continued or stopped.
          </p>
          <div className="two-col">
            <div>
              <h3>What this is</h3>
              <ul>
                <li>a managed-services multi-agent research prototype</li>
                <li>a Principal / environment / observation / payoff loop in an open-world setting</li>
                <li>a system that makes reasoning traces inspectable</li>
                <li>a tool for experimenting with evidence, verification, and synthesis</li>
              </ul>
            </div>
            <div>
              <h3>What this is not</h3>
              <ul>
                <li>not a production investment advisor</li>
                <li>not a full implementation of the papers</li>
                <li>not a learned RL policymaker</li>
                <li>not a dynamic Kubernetes pod-per-agent system</li>
                <li>not a guarantee of correct forecasts</li>
                <li>not personalized financial advice</li>
              </ul>
            </div>
          </div>
          <p>
            EvidenceEngine is our engineering implementation of the observation layer for web and market-data research.
            It is not a named component in the papers.
          </p>
        </div>
      </details>

      <section className="section-block">
        <div className="section-heading">
          <div>
            <p className="tag">Prompt examples</p>
            <h2>Try one of these prompts</h2>
          </div>
          <p className="muted">Clicking a card fills the research question box below.</p>
        </div>
        <div className="prompt-grid">
          {suggestedPrompts.map((item) => (
            <button className="prompt-card" type="button" key={item.title} onClick={() => usePrompt(item.prompt)}>
              <span className="pill">{item.label}</span>
              <strong>{item.title}</strong>
              <span>{item.prompt}</span>
            </button>
          ))}
        </div>
      </section>

      <section className="section-block">
        <div className="section-heading">
          <div>
            <p className="tag">Trace guide</p>
            <h2>How to read results</h2>
          </div>
          <a href="/about">Full demo notes</a>
        </div>
        <div className="glossary-grid">
          <p>
            <strong>Principal action timeline</strong>
            <span>Shows what the controller decided or proposed.</span>
          </p>
          <p>
            <strong>Agent organization</strong>
            <span>Shows the temporary logical research organization. These are AgentSpecs, not Kubernetes pods.</span>
          </p>
          <p>
            <strong>EvidenceEngine observations</strong>
            <span>Search and market-data outputs converted into structured observations.</span>
          </p>
          <p>
            <strong>Claims</strong>
            <span>Candidate propositions extracted from observations.</span>
          </p>
          <p>
            <strong>Verifications</strong>
            <span>Trust-layer checks that promote, dispute, or reject claims.</span>
          </p>
          <p>
            <strong>Artifacts</strong>
            <span>Graph of observations, claims, verifications, final reports, and their dependency links.</span>
          </p>
          <p>
            <strong>Evidence-limited final</strong>
            <span>
              This means the system refused to produce a confident synthesis because claims did not pass verification.
              This is safer than hallucinating.
            </span>
          </p>
          <p>
            <strong>Budget</strong>
            <span>Shows LLM spend, search/tool budget, market-data budget, and protected aggregator/judge budget.</span>
          </p>
        </div>
      </section>

      <form onSubmit={submit} className="run-form">
        <section className="card">
          <h2>Research question</h2>
          <textarea
            id="research-question"
            required
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Example: If the Fed signals a faster-than-expected rate-cut path over the next 3 months, compare U.S. equities, the U.S. dollar, gold, and long-duration bonds. Separate thesis, evidence, risks, what would change the conclusion, and confidence."
          />
          <div className="helper-panel">
            <strong>Good prompts specify:</strong>
            <ul>
              <li>the macro event or scenario</li>
              <li>the asset classes to compare</li>
              <li>the time horizon</li>
              <li>what kind of output you want</li>
            </ul>
          </div>
          <label>
            LLM budget ($)
            <input type="number" min="1" step="1" value={budget} onChange={(event) => updateBudget(+event.target.value)} />
            <small>Funds model reasoning across planner, research, verifier, aggregator, and judge roles.</small>
          </label>
        </section>

        <section className="card">
          <h2>Models and protected role caps</h2>
          <p>
            Anyone with the demo password can choose OpenRouter models. Costs are charged to the deployment API keys.
            Aggregator and judge caps are protected so synthesis and payoff can still run late in the loop.
          </p>
          <datalist id="openrouter-models">
            {available.map((model) => (
              <option key={model} value={model} />
            ))}
          </datalist>
          {roles.map((role) => (
            <label key={role}>
              {role} cap · ${caps[role]}
              <input
                list="openrouter-models"
                disabled={!judge && role === "judge"}
                value={models[role]}
                onChange={(event) => setModels({...models, [role]: event.target.value})}
              />
              <input
                type="number"
                min="0.01"
                step="0.01"
                disabled={!judge && role === "judge"}
                value={caps[role]}
                onChange={(event) => setCaps({...caps, [role]: +event.target.value})}
              />
            </label>
          ))}
          <label>
            <input type="checkbox" checked={judge} onChange={(event) => setJudge(event.target.checked)} /> Enable judge
          </label>
        </section>

        <section className="card">
          <h2>Tool limits</h2>
          <label>
            Tavily/search credits
            <input
              type="number"
              min="1"
              value={toolLimits.tavily_max_credits}
              onChange={(event) => setToolLimits({...toolLimits, tavily_max_credits: +event.target.value})}
            />
            <small>Funds web evidence search. A run can stop because search budget is exhausted even if LLM dollars remain.</small>
          </label>
          <label>
            Market-data requests
            <input
              type="number"
              min="1"
              value={toolLimits.market_data_max_requests}
              onChange={(event) => setToolLimits({...toolLimits, market_data_max_requests: +event.target.value})}
            />
            <small>Funds market snapshots used as caveated context, not as verified knowledge claims.</small>
          </label>
          <p>Provider validation checks credentials before the run; these limits are reserved for actual agent tool calls.</p>
        </section>
        {error && <p className="error">{error}</p>}
        <button disabled={busy}>{busy ? "Validating providers..." : "Validate and create run"}</button>
      </form>
    </main>
  );
}
