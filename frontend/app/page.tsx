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
  {
    title: "Objective",
    text: "The user submits an open-world research question and a bounded run budget.",
  },
  {
    title: "Principal observes",
    text: "The controller reads RunState, budget, artifacts, gaps, and prior decisions.",
  },
  {
    title: "Principal acts",
    text: "It selects typed actions such as research, verification, aggregation, repair, or stop.",
  },
  {
    title: "Agents gather",
    text: "Logical agents use web search and market data to produce external observations.",
  },
  {
    title: "Artifacts form",
    text: "Observations, claims, caveats, and reports are written as inspectable artifacts.",
  },
  {
    title: "Trust updates",
    text: "Verifier and skeptic stages promote, dispute, reject, or caveat claims.",
  },
  {
    title: "Aggregator synthesizes",
    text: "The final answer is built from the trusted evidence available to the run.",
  },
  {
    title: "Judge records payoff",
    text: "Optional judge feedback provides a usefulness and calibration signal.",
  },
  {
    title: "Principal repairs or stops",
    text: "The controller can launch targeted follow-up work or stop under constraints.",
  },
];

const paperLinks = [
  {
    title: "Social Environment Design",
    href: "https://arxiv.org/abs/2402.14090",
  },
  {
    title: "Large Legislative Models",
    href: "https://arxiv.org/abs/2410.08345",
  },
  {
    title: "Creating a Cooperative AI Policymaking Platform",
    href: "https://arxiv.org/abs/2412.06936",
  },
];

const traceItems = [
  {
    title: "Temporary organization",
    text: "AgentSpecs are logical research roles, not dynamic infrastructure.",
  },
  {
    title: "Evidence trail",
    text: "Search results, market snapshots, claims, and verifications remain inspectable.",
  },
  {
    title: "Trust decisions",
    text: "Disputed or rejected claims should be read alongside the final answer.",
  },
  {
    title: "Control trace",
    text: "Principal actions show why the run continued, repaired evidence, or stopped.",
  },
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
        <h1>Operationalizing a paper-inspired research loop</h1>
        <p>
          This prototype explores how a paper-inspired Principal / environment / observation / payoff loop can be
          operationalized in an open-world research setting.
        </p>
      </section>

      <section className="framing-panel" aria-labelledby="research-framing">
        <div className="framing-copy">
          <p className="tag">Research framing</p>
          <h2 id="research-framing">Before submitting a prompt</h2>
          <p>
            The current demo uses investment-research questions as a testbed. For each run, the system creates a
            temporary organization of logical agents, gathers observations from web search and market data, writes
            artifacts, verifies or disputes claims, and exposes the full control trace.
          </p>
          <p>
            It is not a production investment advisor, a full implementation of the papers, a learned RL policymaker,
            or a guarantee of correct forecasts. The value of the demo is the trace: what was observed, what was
            trusted, what was rejected, and why the Principal continued or stopped.
          </p>
        </div>
        <aside className="paper-panel" aria-label="Inspired by">
          <h3>Inspired by</h3>
          <ul className="paper-list">
            {paperLinks.map((paper) => (
              <li key={paper.href}>
                <a href={paper.href} rel="noreferrer" target="_blank">
                  {paper.title}
                </a>
              </li>
            ))}
          </ul>
        </aside>
      </section>

      <section className="section-block" aria-labelledby="loop-map-title">
        <div className="section-heading section-heading-centered">
          <div>
            <p className="tag">Control loop</p>
            <h2 id="loop-map-title">What happens during a run</h2>
          </div>
        </div>
        <ol className="loop-map">
          {workflowSteps.map((step) => (
            <li className="loop-step" key={step.title}>
              <strong>{step.title}</strong>
              <span>{step.text}</span>
            </li>
          ))}
        </ol>
      </section>

      <section className="section-block" aria-labelledby="trace-title">
        <div className="section-heading section-heading-centered">
          <div>
            <p className="tag">How to read results</p>
            <h2 id="trace-title">Inspect the trace before trusting the answer</h2>
          </div>
        </div>
        <div className="trace-grid">
          {traceItems.map((item) => (
            <p key={item.title}>
              <strong>{item.title}</strong>
              <span>{item.text}</span>
            </p>
          ))}
        </div>
      </section>

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
