from src.common.models import AgentSpec


def build_agent_instruction_block(agent_spec: AgentSpec) -> str:
    """Build compact, schema-neutral instructions from an AgentSpec."""
    allowed_tools = ", ".join(agent_spec.allowed_tools) if agent_spec.allowed_tools else "none"
    retrieval_tags = (
        ", ".join(agent_spec.retrieval_tags) if agent_spec.retrieval_tags else "none"
    )
    domain = agent_spec.domain or "unspecified"
    lines = [
        "AgentSpec instruction block:",
        f"- name: {agent_spec.name}",
        f"- role_template: {agent_spec.role_template}",
        f"- branch: {agent_spec.branch}",
        f"- domain: {domain}",
        f"- objective: {agent_spec.objective}",
        f"- allowed_tools: {allowed_tools}",
        f"- retrieval_tags: {retrieval_tags}",
        f"- visibility_scope: {agent_spec.visibility_scope.value}",
        f"- local_budget_usd: ${agent_spec.local_budget_usd:.4f}",
        _branch_rule(agent_spec),
        _visibility_rule(agent_spec),
        _budget_rule(agent_spec),
        (
            "Output rule: Return only the JSON object requested by the caller; "
            "preserve the exact schema and do not add extra keys."
        ),
    ]
    return "\n".join(line for line in lines if line)


def _branch_rule(agent_spec: AgentSpec) -> str:
    branch = agent_spec.branch
    role_template = agent_spec.role_template
    if branch == "market/gold":
        return (
            "Branch rule: You own gold evidence for branch market/gold. Focus on gold, "
            "real yields, USD, inflation, central-bank policy, and safe-haven demand; "
            "do not overreach into equities except for relevant comparisons."
        )
    if branch == "market/fx":
        return (
            "Branch rule: You own FX evidence for branch market/fx. Focus on USD, DXY, "
            "currency pairs, rate differentials, carry, and central-bank divergence."
        )
    if branch.startswith("macro/rates"):
        return (
            "Branch rule: You own rates evidence. Focus on policy path, rate "
            "differentials, yield curves, real yields, and central-bank communication."
        )
    if role_template == "aggregator_agent" or branch == "synthesis/aggregator":
        return (
            "Synthesis role: synthesize across verified claims only and expose "
            "trade-offs, conflicts, risks, opportunities, and evidence gaps."
        )
    return f"Branch rule: Stay within branch {branch}; compare other branches only when relevant."


def _visibility_rule(agent_spec: AgentSpec) -> str:
    if agent_spec.role_template == "aggregator_agent":
        return (
            "Evidence rule: Use verified claims and public_verified artifacts as final "
            "support; treat rejected, disputed, draft, or public_unverified material as "
            "context for limitations only."
        )
    return (
        "Evidence rule: Use evidence visible to this scope. Team/private evidence stays "
        "branch-local; public_unverified evidence can support candidate claims only "
        "after source verification; public_verified evidence is synthesis-safe."
    )


def _budget_rule(agent_spec: AgentSpec) -> str:
    if agent_spec.local_budget_usd > 0:
        return (
            f"Budget rule: This agent has about ${agent_spec.local_budget_usd:.4f} "
            "of local budget; keep reasoning scoped to the objective and allowed tools."
        )
    return "Budget rule: No local budget is allocated here; keep the response tight and scoped."
