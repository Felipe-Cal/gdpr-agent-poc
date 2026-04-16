# Claude Code — Project Instructions

## What this project is

A **cloud-agnostic GDPR Legal Analyst Agent PoC** built to demonstrate the concept to a client without committing to a specific cloud provider. The entire stack runs locally with `docker-compose up` and deploys to Railway with a single push.

This is a **sibling project** to `gdpr-agent` (the GCP learning project). That repo teaches GCP-specific frameworks; this one shows the same agent built for speed-of-delivery and provider flexibility.

**Stack:** Chainlit (UI) · LangGraph (agent) · Qdrant (vector store) · OpenAI/Anthropic (LLM) · Railway (hosting)

## Git workflow

**Always use feature branches. Never commit directly to `main`.**

For every task:
1. Create a branch: `git checkout -b <type>/<short-description>`
   - Use conventional prefixes: `feat/`, `fix/`, `docs/`, `chore/`
2. Implement with one or more commits.
   - **Never add `Co-Authored-By:` or any AI attribution to commits or PR bodies.**
3. Push and open a PR: `gh pr create`
4. **Stop. Do not merge.** The user reviews and merges.

## Code style

- Python 3.11+, type hints throughout.
- `pydantic-settings` for all config — no hardcoded values anywhere.
- `rich` for any CLI output (`ingest.py`, scripts). Typer for CLI argument parsing.
- Comments explain *why*, not *what*. Assume the reader knows Python.
- Keep it minimal — this is a PoC, not a production system. No over-engineering.

## Architecture — five files, one concern each

```
config.py   — all settings via pydantic-settings + .env
ingest.py   — PDF → chunks → embeddings → Qdrant (run once before demo)
tools.py    — three LangChain tools: search_gdpr_documents, web_search, get_gdpr_article
agent.py    — LangGraph ReAct graph using init_chat_model (provider-agnostic)
app.py      — Chainlit UI: streams tokens + shows tool calls as collapsible steps
```

Do not add complexity beyond this structure without a good reason. Adding a phase6/ directory or a Vertex AI pipeline here is wrong — that belongs in the GCP project.

## Key design decisions (don't undo these without understanding why)

**`init_chat_model` in `agent.py`**
Switching LLM providers is two env vars (`LLM_PROVIDER`, `LLM_MODEL`). Do not replace this with hardcoded `ChatOpenAI` or `ChatAnthropic` — the provider-agnostic abstraction is the point.

**`get_embeddings()` factory in `config.py` — HuggingFace by default**
`EMBEDDING_PROVIDER` controls which embedder runs: `"huggingface"` (default, free, local, 384-dim) or `"openai"` (`text-embedding-3-small`, 1536-dim, ~$0.001 for the full corpus). HuggingFace requires no API key — `sentence-transformers` runs locally. Switching providers invalidates all stored vectors and requires re-running `python ingest.py`. The ingest script probes the vector size at runtime so Qdrant collection dimensions stay consistent.

**Qdrant over pgvector or Pinecone**
Qdrant runs in Docker locally (zero setup) and has a free cloud tier for Railway deployment. pgvector requires a Postgres instance; Pinecone requires a credit card. For a PoC, Qdrant is the right default.

**`astream_events` in `app.py`**
This is what gives us token-level streaming AND tool call visibility simultaneously. Do not replace with `astream` (stream_mode="values") — that only gives state snapshots, not individual tokens.

**`cl.Step` for tool calls**
Each tool call appears as a collapsible step in the Chainlit UI showing the input and (truncated) output. This is intentional — the client should see the agent's reasoning process, not just the final answer. Truncating to 2000 chars prevents the UI from being flooded by large retrieval results.

## Running locally

```bash
# First time only
cp .env.example .env
# Fill in LLM_PROVIDER + the matching API key (e.g. OPENAI_API_KEY).
# Embeddings use HuggingFace by default — no extra API key needed.

# Start Qdrant + app
docker-compose up

# In a separate terminal — ingest PDFs (run once, or when docs change)
# Drop PDFs into data/gdpr_docs/ first
python ingest.py
```

App runs at http://localhost:8000.

## Switching LLM providers

Only `.env` changes — no code changes:

```bash
# OpenAI (default)
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini

# Anthropic
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-5-haiku-20241022

# Groq (fastest, cheapest, good for demos)
LLM_PROVIDER=groq
LLM_MODEL=llama-3.1-8b-instant
```

## Deploying to Railway

1. Connect the GitHub repo in the Railway dashboard.
2. Add a Qdrant service (or use Qdrant Cloud free tier and point `QDRANT_URL` at it).
3. Set env vars from `.env.example` in the Railway service settings.
4. Push to `main` — Railway auto-deploys on every push.

**Before the demo:** run `python ingest.py` locally pointing at the production Qdrant URL, so vectors are pre-loaded. The demo should never show an empty knowledge base.

## Cost awareness

This project has no persistent cloud infrastructure costs during development. The only costs are:
- **LLM API calls**: ~$0.001–0.01 per query depending on model
- **Embedding**: ~$0.001 total for the full GDPR corpus (one-time)
- **Railway**: ~$5/month for a running service (or free tier with sleep)
- **Qdrant Cloud**: free tier (1GB) covers this use case

No warnings needed before running queries. Only flag costs if the user asks about deploying persistent infrastructure (Railway paid plan, Qdrant Cloud paid tier, etc.).

## What NOT to build here

- GCP-specific services (BigQuery, Vertex AI, Cloud Run, GKE) — those belong in `gdpr-agent`
- Multi-phase structure — this is intentionally a single flat module
- Fine-tuning — out of scope for a PoC
- Evaluation pipelines — out of scope; show eval methodology as a slide, not code
- Authentication/multi-tenancy — out of scope for a PoC demo
