from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── LLM ──────────────────────────────────────────────────────────────────
    # Switch provider by changing these two vars — no code changes needed.
    # LangChain's init_chat_model supports: "openai", "anthropic", "groq", etc.
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"        # or "claude-3-5-haiku-20241022"

    # API keys — set whichever provider you're using
    openai_api_key: str | None = None     # used for LLM (if provider=openai) + embeddings
    anthropic_api_key: str | None = None  # used if provider=anthropic

    # ── Embeddings ────────────────────────────────────────────────────────────
    # Always uses OpenAI embeddings regardless of LLM provider.
    # text-embedding-3-small: $0.02/1M tokens — full GDPR corpus costs ~$0.001
    embedding_model: str = "text-embedding-3-small"

    # ── Qdrant ───────────────────────────────────────────────────────────────
    # Local:       qdrant_url=http://localhost:6333, no api_key needed
    # Qdrant Cloud: qdrant_url=https://xxx.qdrant.io, qdrant_api_key=<key>
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "gdpr_docs"

    # ── RAG ──────────────────────────────────────────────────────────────────
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_top_k: int = 5


settings = Settings()
