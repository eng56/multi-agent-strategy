# PrincipalPolicy architecture

PrincipalPolicy is the deterministic v0 control policy for the research loop. Its job is to inspect
the current run state, propose typed `PrincipalAction` records, and keep the workflow moving through
research, trust, synthesis, judging, follow-up, and stopping decisions without relying on another
free-form LLM planner at every step.

## Relation to the paper loop

The implementation borrows the Principal / follower / environment / observation abstraction from
paper-style multi-agent control loops. In this repo:

- the environment observation is `RunState` plus the persisted run objects that produced it;
- followers are the planner, tool runner, verifier, skeptic, aggregator, and judge roles;
- the Principal action is a typed `PrincipalAction`, not a chat instruction;
- payoff is currently approximated by verifier outcomes, judge score, budget remaining, failures,
  and final-answer presence rather than a realized market outcome.

This is not a full Social Environment Design implementation or an RL-trained policymaker. It is a
safe, inspectable policy slice that makes the control loop explicit in application state.

## Input: RunState and snapshot

The policy consumes a `PrincipalSnapshot`, built from:

- `Run` status, model policy, and budget;
- `RunState` phase, branches, known facts, open questions, budget remaining, tool budget remaining,
  dead-letter count, judge feedback, and candidate next actions;
- persisted `OrganizationPlan`, `AgentSpec`, `ResearchTask`, `Observation`, `Claim`,
  `Verification`, `Artifact`, `PrincipalAction`, and `DeadLetterRecord` objects;
- optional `FinalReport`.

`RunState` is the compact observation, while the snapshot carries the detailed objects needed for
deterministic rule checks such as “which observations lack claims?” or “has this branch already
requested an equivalent action?”.

## Output: PrincipalAction

The policy emits `PrincipalAction` objects with:

- `action_type`, such as `SPAWN_AGENT`, `ASSIGN_TASK`, `REQUEST_TOOL_CALL`,
  `REQUEST_VERIFICATION`, `REQUEST_SKEPTIC_REVIEW`, `REQUEST_AGGREGATION`, `REQUEST_FOLLOWUP`, or
  `STOP_RUN`;
- `reason`, intended to be human-readable and auditable;
- `expected_information_gain`, `estimated_cost`, `target_branch`, `required_role`, and `priority`;
- lifecycle fields such as `status`, `producer`, policy phase, wave number, and idempotency key
  where runtime code supplies them.

The action record is the audit trail. It should explain why the system moved, stopped, or asked for
more evidence.

## Deterministic policy v0

The current rules are deliberately simple:

1. Do nothing once a run is terminal.
2. If useful budget is exhausted, aggregate verified evidence if available; otherwise stop.
3. Recover from dead letters when verified context exists; otherwise stop.
4. Create an organization plan if one does not exist.
5. Assign initial research tasks if no tasks exist.
6. Run pending tool tasks.
7. Extract claims from observations that have not produced claims.
8. Verify unverified claims.
9. Request skeptic review once enough verified knowledge exists and no counterargument artifact is
   present.
10. Aggregate when trusted evidence is available, or after a completed follow-up wave.
11. Judge a final report when judge mode is configured.
12. Request one targeted follow-up wave if the judge score is below threshold.
13. Stop when final judging is complete and no more follow-up is available.

Candidates are de-duplicated, validated against budget and pending equivalent work, then sorted by
priority, expected information gain per estimated cost, estimated cost, action type, branch, and role.

## Shadow vs active mode

`principal_policy_mode` defaults to `shadow`.

- `off`: policy evaluation is disabled.
- `shadow`: policy decisions are persisted for observability, but legacy workflow execution remains
  authoritative.
- `active`: approved policy actions may drive runtime behavior.

Active mode is intentionally opt-in. The default remains shadow so policy quality can be inspected
before it controls production runs.

## Action lifecycle

A typical action lifecycle is:

1. `PROPOSED`: policy emits an action candidate.
2. `APPROVED`: runtime or future governance layer accepts it.
3. `EXECUTED`: the corresponding workflow effect happened.
4. `REJECTED` or `FAILED`: the action was not used or could not complete.

Current runtime code persists action records so reviewers can compare policy intent with actual
workflow outcomes.

## Idempotency and duplicate control

The policy avoids repeated equivalent actions by checking:

- recent executed actions with the same action type, target branch, and required role;
- currently proposed or approved equivalent actions;
- pending tasks on the same branch for task-assignment actions.

This is not complete distributed idempotency. It is a deterministic guardrail that reduces obvious
duplicate loops while the active executor matures.

## Budget constraints

Budget is part of both proposal and validation:

- candidate actions include an estimated cost based on role model policy or tool type;
- policy stops or requests partial aggregation when remaining budget falls below the useful-action
  threshold;
- aggregation and judge protected budgets still remain enforced by the underlying budget manager;
- follow-up is limited to one wave in v0.

The policy should never invent spend authority. It can only propose actions that the runtime budget
system may later approve and execute.

## Current limitations

- The policy is deterministic rule code, not a learned policy.
- Payoff is proxy-based: verifier status, judge score, failures, and budget are not real investment
  outcomes.
- Active mode is still narrow and should remain opt-in.
- Duplicate control is local to recent records and pending work, not a global transaction protocol.
- Follow-up is limited to one wave.
- Branch and task priority remain heuristic.
- The policy still coexists with legacy workflow paths while the active loop is being hardened.
