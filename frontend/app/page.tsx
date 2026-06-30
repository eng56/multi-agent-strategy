"use client";

import {FormEvent, useEffect, useState} from "react";

const roles = ["planner", "utility", "research", "verifier", "aggregator", "judge"] as const;
const defaultModels: Record<string, string> = {
  planner: "anthropic/claude-sonnet-4.5",
  utility: "qwen/qwen-2.5-7b-instruct",
  research: "qwen/qwen-2.5-72b-instruct",
  verifier: "openai/gpt-4.1-mini",
  aggregator: "anthropic/claude-sonnet-4.5",
  judge: "openai/gpt-4.1-mini",
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

export default function Home() {
  const [question, setQuestion] = useState("");
  const [budget, setBudget] = useState(10);
  const [judge, setJudge] = useState(true);
  const [models, setModels] = useState(defaultModels);
  const [available, setAvailable] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [toolLimits, setToolLimits] = useState({tavily_max_credits: 20, market_data_max_requests: 10});
  const [caps, setCaps] = useState<Record<string, number>>(
    Object.fromEntries(roles.map((role) => [role, +(10 * weights[role]).toFixed(2)])),
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
    <main className="shell">
      <p className="tag">PASSWORD-LOCKED MANAGED DEMO</p>
      <h1>Choose every agent model. Spend on the demo account.</h1>
      <form onSubmit={submit}>
        <section className="card">
          <h2>Research question</h2>
          <textarea required value={question} onChange={(event) => setQuestion(event.target.value)} />
          <label>
            Shared OpenRouter budget ($)
            <input type="number" min="1" step="1" value={budget} onChange={(event) => updateBudget(+event.target.value)} />
          </label>
        </section>
        <section className="card">
          <h2>Models and protected role caps</h2>
          <p>Anyone with the demo password can choose OpenRouter models. Costs are charged to the deployment API keys.</p>
          <datalist id="openrouter-models">
            {available.map((model) => <option key={model} value={model} />)}
          </datalist>
          {roles.map((role) => <label key={role}>{role} · ${caps[role]}<input list="openrouter-models" disabled={!judge && role === "judge"} value={models[role]} onChange={(event) => setModels({...models, [role]: event.target.value})} /><input type="number" min="0.01" step="0.01" disabled={!judge && role === "judge"} value={caps[role]} onChange={(event) => setCaps({...caps, [role]: +event.target.value})} /></label>)}
          <label><input type="checkbox" checked={judge} onChange={(event) => setJudge(event.target.checked)} /> Enable judge</label>
        </section>
        <section className="card">
          <h2>Simple tool limits</h2>
          <label>Tavily credits<input type="number" min="1" value={toolLimits.tavily_max_credits} onChange={(event) => setToolLimits({...toolLimits, tavily_max_credits: +event.target.value})} /></label>
          <label>Market-data requests<input type="number" min="1" value={toolLimits.market_data_max_requests} onChange={(event) => setToolLimits({...toolLimits, market_data_max_requests: +event.target.value})} /></label>
          <p>Provider validation consumes one unit from each limit.</p>
        </section>
        {error && <p className="error">{error}</p>}
        <button disabled={busy}>{busy ? "Validating providers…" : "Validate and create run"}</button>
      </form>
    </main>
  );
}
