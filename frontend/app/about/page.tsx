const architectureItems = [
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
          I did not try to reproduce the full economic simulation setting from the papers. Instead, I asked what
          systems substrate would be needed to operationalize the Principal / environment / observation / payoff loop
          in an open-world research setting.
        </p>
        <p>
          The prototype creates a temporary agent organization, produces artifacts, verifies or disputes claims, records
          Principal actions, and exposes the trace in a UI. The current system is conservative: if no claims pass
          verification, it produces an evidence-limited final rather than hallucinating a forecast. The next step is to
          improve how the Principal repairs evidence and converts observations into atomic, verifiable claims.
        </p>
      </section>

      <section className="section-block">
        <div className="section-heading">
          <div>
            <p className="tag">Architecture summary</p>
            <h2>How the pieces fit together</h2>
          </div>
        </div>
        <div className="glossary-grid">
          {architectureItems.map((item) => (
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
