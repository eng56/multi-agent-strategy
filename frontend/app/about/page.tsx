const controlLoopItems = [
  {
    title: "Principal",
    text: "The controller reads RunState, selects typed actions, allocates work, and decides whether to repair evidence or stop under budget.",
  },
  {
    title: "Environment and tools",
    text: "Web search and market-data calls create external observations. Tool limits are part of the run budget rather than hidden global demo constants.",
  },
  {
    title: "Artifacts",
    text: "Observations, candidate claims, verifications, caveats, final reports, and judge feedback are persisted as traceable artifacts.",
  },
  {
    title: "Trust layer",
    text: "Verifier and skeptic stages promote, dispute, reject, or caveat claims before the aggregator can use them as decision support.",
  },
  {
    title: "Aggregator and judge",
    text: "The aggregator produces a caveated synthesis from the evidence tiers, and the judge records a payoff-style usefulness/calibration signal.",
  },
];

const managedArchitectureItems = [
  {
    title: "Frontend and API",
    text: "The Next.js UI creates runs, while the FastAPI orchestrator validates requests, writes state, and starts coordination.",
  },
  {
    title: "Workers and event bus",
    text: "GKE runs fixed role deployments. Kafka / Confluent carries coordination events between those workers.",
  },
  {
    title: "State and artifacts",
    text: "Redis / Upstash is the blackboard and run-state source of truth. GCS stores raw tool outputs and artifacts.",
  },
  {
    title: "Models and tools",
    text: "OpenRouter, Tavily, Massive, and optional fetch/search providers supply model calls and observations, with Langfuse tracing LLM calls.",
  },
];

const uiItems = [
  "Start with the Principal action timeline to see why the controller continued, repaired, or stopped.",
  "Compare verified knowledge claims with disputed and rejected claims before reading the final answer too literally.",
  "Use the artifact graph to inspect dependencies between observations, claims, verifications, and final reports.",
  "Check budget diagnostics to understand which constraint bound the run.",
  "Read evidence-limited finals as a refusal to overstate weak evidence, not as a polished investment forecast.",
];

const limitations = [
  "This is not a production investment advisor and does not provide personalized financial advice.",
  "It is not a full implementation of the papers and does not include a learned RL policymaker.",
  "AgentSpecs are logical research roles, not dynamic Kubernetes pod-per-agent infrastructure.",
  "EvidenceEngine is an engineering implementation of the observation layer for this prototype, not a named component in the papers.",
  "Forecasts can be wrong; the trace is intended to make uncertainty, evidence quality, and controller behavior inspectable.",
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

export default function AboutPage() {
  return (
    <main className="shell about-page">
      <nav className="top-nav">
        <a href="/">Back to new run</a>
        <span className="tag">DEMO NOTES</span>
      </nav>

      <section className="hero-panel">
        <p className="tag">What this prototype demonstrates</p>
        <h1>Principal / environment / observation / payoff in open-world research</h1>
        <p className="lede">
          This prototype explores how a paper-inspired Principal / environment / observation / payoff loop can be
          operationalized in an open-world research setting.
        </p>
        <p>
          The current demo uses investment-research questions as a testbed. For each run, the system creates a
          temporary organization of logical agents, gathers observations from web search and market data, writes
          artifacts, verifies or disputes claims, and exposes the full control trace.
        </p>
      </section>

      <section className="section-block">
        <div className="section-heading">
          <div>
            <p className="tag">Control-loop summary</p>
            <h2>How the pieces fit together</h2>
          </div>
        </div>
        <div className="glossary-grid">
          {controlLoopItems.map((item) => (
            <p key={item.title}>
              <strong>{item.title}</strong>
              <span>{item.text}</span>
            </p>
          ))}
        </div>
      </section>

      <section className="section-block" aria-labelledby="about-architecture-title">
        <div className="section-heading">
          <div>
            <p className="tag">Architecture</p>
            <h2 id="about-architecture-title">Managed infrastructure and logical agents</h2>
          </div>
        </div>
        <div className="text-panel architecture-summary-copy">
          <p>
            The system uses managed infrastructure. Redis/Upstash is the blackboard and source of truth for run state.
            Kafka/Confluent is used for coordination events between role workers. Kubernetes/GKE runs the orchestrator
            and a fixed set of workers. GCS stores raw artifacts. OpenRouter, Tavily, Massive, and optional search/fetch
            providers provide model and observation capabilities.
          </p>
          <p>
            This is not a dynamic pod-per-agent system. The agents are logical AgentSpecs. The physical workers are
            static role deployments.
          </p>
        </div>
        <div className="architecture-card-grid architecture-card-grid-compact">
          {managedArchitectureItems.map((item) => (
            <article className="architecture-card" key={item.title}>
              <h3>{item.title}</h3>
              <p>{item.text}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="section-block">
        <div className="section-heading">
          <div>
            <p className="tag">Paper alignment</p>
            <h2>What maps to the research framing</h2>
          </div>
        </div>
        <div className="text-panel">
          <p>
            The UI is organized around the control-loop idea: a Principal observes the current state, chooses actions,
            receives observations from an environment, and gets a payoff signal from the judge. The implementation is a
            systems prototype for open-world research, not a claim that the papers specify this exact architecture.
          </p>
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
          <p>
            EvidenceEngine is our engineering implementation of the observation layer for web and market-data research.
            It is not a named component in the papers.
          </p>
        </div>
      </section>

      <section className="section-block">
        <div className="section-heading">
          <div>
            <p className="tag">Reading the UI</p>
            <h2>What to look at first</h2>
          </div>
        </div>
        <ul className="clean-list">
          {uiItems.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </section>

      <section className="section-block">
        <div className="section-heading">
          <div>
            <p className="tag">Limitations</p>
            <h2>What this demo is not</h2>
          </div>
        </div>
        <ul className="clean-list">
          {limitations.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </section>
    </main>
  );
}
