# GDPR Legal Analyst — PoC

A cloud-agnostic GDPR legal analyst agent. Ask questions about the regulation; the agent searches an ingested knowledge base, looks up specific articles, and searches the web for recent enforcement news.

Built as a client PoC: runs locally with one command, deploys to Railway with a push. No GCP account, no cloud lock-in.

**Sibling project:** [`gdpr-agent`](https://github.com/Felipe-Cal/gdpr-agent) is the same agent built on the full GCP stack (BigQuery, Vertex AI, GKE, Vertex AI Pipelines) as a learning project.

---

## Stack

| Concern | Choice | Why |
|---|---|---|
| UI | [Chainlit](https://chainlit.io) | Python-native chat UI with streaming + tool call visibility out of the box |
| Agent | [LangGraph](https://langchain-ai.github.io/langgraph/) | Same ReAct graph as the GCP project — provider-agnostic |
| LLM | OpenAI / Anthropic / Groq | Switched via two env vars, no code changes |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (default) or OpenAI | Free local default; switch to OpenAI for higher quality |
| Vector store | [Qdrant](https://qdrant.tech) | Runs in Docker locally, free cloud tier for deployment |
| Hosting | [Railway](https://railway.app) | One-command deploy, ~$5/month, no infra to manage |

---

## Project structure

```
gdpr-agent-poc/
├── config.py          # All settings via pydantic-settings + .env
├── ingest.py          # PDF → chunks → embeddings → Qdrant (run once)
├── tools.py           # search_gdpr_documents, web_search, get_gdpr_article
├── agent.py           # LangGraph ReAct agent (provider-agnostic via init_chat_model)
├── app.py             # Chainlit UI — streaming + tool call steps
├── docker-compose.yml # Local dev: Qdrant + app in one command
├── Dockerfile         # Used by docker-compose and Railway
├── railway.toml       # Railway deploy config
├── requirements.txt   # Python dependencies
└── data/
    └── gdpr_docs/     # Drop PDFs here before ingesting (gitignored)
```

Five files. Each has exactly one job.

---

## Quick start (local)

### 1. Prerequisites

- Docker Desktop running
- Python 3.11+
- An API key for your chosen LLM provider (OpenAI, Anthropic, or Groq — see [Switching providers](#switching-llm-providers))
- No embedding API key needed — HuggingFace embeddings run locally by default

### 2. Install and configure

```bash
git clone https://github.com/Felipe-Cal/gdpr-agent-poc
cd gdpr-agent-poc

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env — set OPENAI_API_KEY (or ANTHROPIC_API_KEY if using Anthropic as LLM)
# Embeddings default to HuggingFace (free, local) — no embedding API key required
```

### 3. Add documents

Drop GDPR PDFs into `data/gdpr_docs/`. Good starting documents:
- [GDPR full text (EUR-Lex)](https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32016R0679) — the regulation itself
- [EDPB guidelines](https://edpb.europa.eu/our-work-tools/general-guidance/guidelines-recommendations-best-practices_en) — interpretive guidance
- [ICO practical guides](https://ico.org.uk/for-organisations/) — UK DPA, well-written

### 4. Ingest documents

```bash
# Start Qdrant first
docker-compose up qdrant -d

# Ingest PDFs into Qdrant (free with HuggingFace embeddings, runs in ~1 min)
python ingest.py
```

You only need to run this once, or when you add new documents. Qdrant persists the vectors in a Docker volume — they survive restarts.

### 5. Run the app

```bash
# Start everything (Qdrant + Chainlit app)
docker-compose up

# Or run the app directly without Docker (requires Qdrant already running)
chainlit run app.py
```

Open http://localhost:8000.

---

## What the UI shows

Each query goes through the agent's full reasoning loop:

1. **Tool calls appear as collapsible steps** — you can expand any step to see what the agent searched for and what came back. This is the key demo moment: the client sees *why* the answer is what it is.

2. **Answer streams token by token** — no waiting for the full response.

3. **Conversation history is maintained** — follow-up questions ("what about Article 9?") work correctly within a session.

---

## Switching LLM providers

Two env vars in `.env` — no code changes:

```bash
# OpenAI (default, cheapest for demos)
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini          # or gpt-4o for higher quality

# Anthropic
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-5-haiku-20241022   # or claude-3-5-sonnet-20241022

# Groq (fastest response time, great for demos)
LLM_PROVIDER=groq
LLM_MODEL=llama-3.1-8b-instant
# pip install langchain-groq  (add to requirements.txt)
```

> **Note:** Embeddings default to HuggingFace (`all-MiniLM-L6-v2`, free, local). To switch to OpenAI embeddings (`text-embedding-3-small`, better quality), set `EMBEDDING_PROVIDER=openai` and `EMBEDDING_MODEL=text-embedding-3-small` in `.env`, then re-run `python ingest.py` — different embedding dimensions require a fresh collection.

---

## Deploying to Railway

### Option A — Qdrant Cloud + Railway (recommended)

Qdrant Cloud has a free tier (1GB) — no credit card required. Railway hosts the app.

1. **Create a Qdrant Cloud cluster** at [cloud.qdrant.io](https://cloud.qdrant.io):
   - Choose a free cluster in your preferred region
   - Copy the cluster URL and API key

2. **Ingest documents into the cloud cluster** (run locally, pointing at the cloud URL):
   ```bash
   QDRANT_URL=https://your-cluster.qdrant.io \
   QDRANT_API_KEY=your-api-key \
   python ingest.py
   ```

3. **Create a Railway project**:
   - Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub repo
   - Select this repository
   - Railway auto-detects the `Dockerfile`

4. **Set environment variables** in the Railway service settings (copy from `.env.example`):
   - `OPENAI_API_KEY`
   - `LLM_PROVIDER` / `LLM_MODEL`
   - `QDRANT_URL` (your Qdrant Cloud URL)
   - `QDRANT_API_KEY`

5. **Deploy** — Railway builds and deploys on every push to `main`.

### Option B — Qdrant as a Railway service

Add a Qdrant service inside the same Railway project:
- New Service → Docker Image → `qdrant/qdrant`
- Set `QDRANT_URL` in the app service to `http://qdrant:6333` (Railway internal networking)
- No API key needed for internal communication

Option B keeps everything in one Railway project but uses Railway's volume for Qdrant storage (~$0.25/GB/month).

### Estimated cost (Railway)

| Resource | Cost |
|---|---|
| App service (512MB RAM) | ~$5/month |
| Qdrant Cloud free tier | $0 |
| LLM API (demo traffic ~200 queries) | ~$2–5 |
| Embeddings (HuggingFace default) | $0 |
| **Total** | **~$7–10/month** |

### Teardown

```bash
# Delete the Railway project from the dashboard, or:
railway down

# Delete the Qdrant Cloud cluster from cloud.qdrant.io dashboard
```

---

## How it works

### Agent loop (LangGraph ReAct)

```
User question
    ↓
[call_model] — LLM decides which tool(s) to call
    ↓ (if tool calls)
[run_tools] — execute search_gdpr_documents / get_gdpr_article / web_search
    ↓
[call_model] — LLM reasons over tool results, decides to answer or call more tools
    ↓ (no tool calls = final answer)
Stream answer to Chainlit UI
```

The graph loops until the model produces a response without tool calls. Conversation history is kept in a `MemorySaver` checkpointer scoped to the Chainlit session.

### Retrieval (Qdrant)

Documents are chunked with 1000-char chunks / 200-char overlap, embedded (HuggingFace `all-MiniLM-L6-v2` by default, 384-dim; or OpenAI `text-embedding-3-small`, 1536-dim), and stored in Qdrant. At query time, the `search_gdpr_documents` tool embeds the query and retrieves the top-5 nearest chunks by cosine similarity. The ingest script probes the vector size at runtime so the Qdrant collection always matches the active embedding model.

### Provider abstraction (`init_chat_model`)

LangChain's `init_chat_model` resolves the right integration at runtime based on `LLM_PROVIDER`:
- `"openai"` → `ChatOpenAI` (needs `OPENAI_API_KEY`)
- `"anthropic"` → `ChatAnthropic` (needs `ANTHROPIC_API_KEY`)
- `"groq"` → `ChatGroq` (needs `GROQ_API_KEY`, add `langchain-groq` to requirements)

The agent, tools, and Chainlit app are identical regardless of which provider is active.

---

## Adding new documents

1. Drop PDFs into `data/gdpr_docs/`
2. Re-run `python ingest.py`

Ingest is idempotent — re-running it on the same files is safe (Qdrant upserts by document ID). For production Qdrant Cloud, run ingest locally pointing at the cloud URL before each demo.

---

## Environment variables reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `LLM_PROVIDER` | No | `openai` | LangChain provider name (`openai`, `anthropic`, `groq`) |
| `LLM_MODEL` | No | `gpt-4o-mini` | Model name for the chosen provider |
| `OPENAI_API_KEY` | If using OpenAI LLM | — | LLM API key for `LLM_PROVIDER=openai` |
| `ANTHROPIC_API_KEY` | If using Anthropic | — | LLM API key for `LLM_PROVIDER=anthropic` |
| `EMBEDDING_PROVIDER` | No | `huggingface` | `huggingface` (free, local) or `openai` |
| `EMBEDDING_MODEL` | No | `all-MiniLM-L6-v2` | Embedding model; for OpenAI use `text-embedding-3-small` |
| `QDRANT_URL` | No | `http://localhost:6333` | Qdrant instance URL |
| `QDRANT_API_KEY` | If using Qdrant Cloud | — | Qdrant Cloud API key |
| `QDRANT_COLLECTION` | No | `gdpr_docs` | Collection name |
| `RETRIEVAL_TOP_K` | No | `5` | Number of chunks retrieved per search |
| `CHUNK_SIZE` | No | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | No | `200` | Overlap between consecutive chunks |
