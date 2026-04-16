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
    # embedding_provider: "openai" (requires OPENAI_API_KEY) or
    #                     "huggingface" (free, runs locally — no API key needed)
    embedding_provider: str = "huggingface"
    # openai model: "text-embedding-3-small" (1536 dims)
    # huggingface model: "all-MiniLM-L6-v2" (384 dims)
    embedding_model: str = "all-MiniLM-L6-v2"

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


def get_embeddings():
    """Return the configured embedding model. Import here to keep config dependency-light."""
    if settings.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)
    # default: huggingface (free, local)
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)
