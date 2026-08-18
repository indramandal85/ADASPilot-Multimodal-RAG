# ADASPilot threat model

## Protected assets

- Corpus documents and access boundaries
- User and document PII
- Tool permissions and execution budgets
- Evidence provenance and citations
- Host files, processes, network, and Python runtime
- Memory and CPU exposed to document images and task plans

## Trust boundaries

1. User question entering a graph
2. Retrieved PDF text, captions, OCR, and images entering evidence
3. A model proposing a tool action or plan
4. Tool results entering the shared evidence board
5. A generated answer leaving the system

## Enforced controls

| Threat | Control |
|---|---|
| Prompt override | Pattern rejection before graph/model execution |
| Malicious PDF instructions | Drop poisoned text/caption/OCR evidence |
| Cross-document access | External document allowlist after retrieval |
| Tool privilege escalation | Role-to-tool permission map |
| Shell/code through calculator | Numeric AST allowlist only |
| Fake evidence/citations | Evidence-ID and citation-board validation |
| Agent loop | Call, recursion, duplicate-action, and no-gain limits |
| Plan cycle/explosion | Count, depth, dependency, and cycle validation |
| Worker explosion | Two-worker concurrency cap |
| Visual decompression attack | File-byte and decoded-pixel limits plus PIL verification |
| PII leakage | Redaction before evidence/model/trace boundaries |
| Invalid model JSON | Pydantic validation before routing or execution |
| Untrusted web result | ReAct-only empty-corpus fallback, domain allowlist, injection filtering |

## Explicitly unavailable capabilities

Agents have no shell, Python execution, arbitrary URL fetch, file-write tool,
database mutation, unrestricted web search, or free-form inter-agent messaging.
An optional ReAct literature tool can query a fixed technical-domain allowlist
after local retrieval returns no evidence. External model and web use are
optional; all default tests and smoke runs are local.
