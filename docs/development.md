# Development workflow

## First-time setup

```bash
cp .env.example .env
make setup
make verify
```

## Normal module workflow

```bash
make ingest
make build-index
make evaluate-baseline
make experiments
make build-visual-index
make evaluate-visual
make orchestrate QUERY="What does radar measure?" MODE=direct
make orchestrate QUERY="Research radar evidence." MODE=react
make orchestrate QUERY="Compare radar and camera." MODE=plan
```

Before handing off a change, run:

```bash
make verify
```

## Configuration rules

- All settings use the `ADASPILOT_` environment prefix.
- Importing the package never initiates network connections.
- Model names, thresholds, feature flags, and paths live in `Settings`.
- Generated corpus versions, indexes, results, and raw PDFs are not committed.
- Qdrant uses local embedded storage; no server or Docker is required.
- HyDE is an experiment and is disabled by default because E4 underperformed E3.
- Deterministic action/planning policies are used unless all `ADASPILOT_LLM_*`
  settings configure an OpenAI-compatible endpoint.

## Scope discipline

- Keep data contracts explicit, but avoid empty abstraction layers.
- Add a feature only with an evaluation that can prove its value.
- Keep orchestration separate from retrieval and tool implementations.
- Do not add deployment, container, frontend, or unbounded tools.
- Keep optional heavyweight visual dependencies outside the default install.
- Never publish held-out test metrics until the final system is frozen.
