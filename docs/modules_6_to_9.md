# Modules 6-9 implementation guide

## Module 6: direct corrective LangGraph

The direct workflow is the default low-cost path:

```text
validate -> classify -> retrieve -> grade
                               ├─ sufficient -> generate -> verify -> END
                               ├─ weak first time -> rewrite -> retrieve
                               └─ weak after retry -> abstain -> END
```

The graph has one explicit rewrite, a runtime recursion limit of 10, per-node
timeouts, access filtering, retrieved-instruction filtering, PII redaction, and
citation verification. A timeout, bad citation, weak second retrieval, blocked
input, or recursion error ends in a typed abstention.

## Module 7: bounded ReAct research

ReAct uses one LangGraph `step` node that chooses and executes exactly one action
per graph step. This makes the eight-step limit easy to reason about.

Allowed read-only tools:

- Hybrid text search
- Visual search
- Parent-context retrieval
- Page retrieval
- Document metadata lookup
- Evidence-linked arithmetic calculator
- Optional domain-filtered ADAS literature search after an empty corpus search
- Finish with existing evidence IDs
- Abstain

The registry checks role permissions outside the model. Equivalent name/argument
calls have stable hashes; a third identical action is blocked. Two consecutive
actions with no new evidence stop the run. Finishing with invented evidence IDs
or calculating without existing evidence also fails safely.

The default policy is deterministic for reproducible tests. When an
OpenAI-compatible endpoint is configured, `OpenAICompatibleActionPolicy`
selects a typed action from summarized observations. The model never executes a
tool directly.

The optional web fallback is ReAct-only, disabled without a Tavily key, and
restricted to arXiv, IEEE, SAE, NHTSA, Euro NCAP, and ResearchGate. Its evidence
is labelled `web` and is not treated as verified local-corpus evidence.

Operational metrics include expected-tool accuracy, redundant-action rate,
evidence gain per call, completion, latency, model calls, and estimated API cost.

## Module 8: plan-and-execute researchers

The planner returns `ResearchPlan` and `ResearchTask` models. The deterministic
supervisor validates:

- At most five tasks
- Dependency depth at most three
- Existing dependency IDs
- No self-dependencies or cycles
- At most two concurrent researchers
- At most three tool calls per task
- At most one replan

Ready independent tasks run through a two-worker `ThreadPoolExecutor`.
Researchers share code but receive separate typed assignments. Their evidence
is merged into an append-only board with task, tool, document, page, and asset
provenance. The strongest item from each task is ordered first before synthesis,
which prevents one fast parallel task from dominating citations. There is no
free-form agent chat or shared mutable scratchpad.

After research, the supervisor detects narrowly matched contradictory claims,
synthesizes only from board evidence, and verifies required-task coverage plus
citations. It returns an answer, partial answer, or abstention.

An optional structured-model planner implements the same contract. Invalid,
cyclic, oversized, or hallucinated plans are rejected before execution.

## Module 9: guardrails

Guardrails are deterministic code at trust boundaries:

- Email, phone, card, and IP redaction for questions and retrieved evidence
- User, document-text, caption, and OCR prompt-injection patterns
- Post-retrieval document allowlists before evidence leaves a tool
- Role-specific tool permissions
- Restricted numeric AST calculator; no names, calls, attributes, or code
- Plan size, depth, dependency, cycle, and replan checks
- Equivalent-action and no-evidence loop detection
- Image byte, decoded-pixel, decompression-bomb, and malformed-image checks
- Pydantic structured action, plan, state, answer, and result contracts
- Citation existence checks against the current evidence board

`tests/unit/test_guardrails_adversarial.py` contains 33 distinct automated
threat cases. Graph tests separately cover timeouts, recursion, repeated tools,
privilege escalation, poisoned retrieved text, plan explosion, conflicts, and
replan exhaustion.

## Commands

```bash
make orchestrate QUERY="What does radar measure?" MODE=direct
make orchestrate QUERY="Find radar evidence iteratively." MODE=react
make orchestrate QUERY="Compare radar and camera strengths." MODE=plan
make verify
```
