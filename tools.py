"""
Agent tools — provider-agnostic versions of the Phase 2 tools.

Same three tools, same docstrings (which the LLM reads to decide when to call
them), but backed by Qdrant instead of BigQuery.
"""

from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from config import settings

# Lazy singleton — initialised on first tool call
_store: QdrantVectorStore | None = None


def _get_store() -> QdrantVectorStore:
    global _store
    if _store is None:
        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )
        _store = QdrantVectorStore(
            client=client,
            collection_name=settings.qdrant_collection,
            embedding=embeddings,
        )
    return _store


# ── Key GDPR articles — same static lookup as the GCP project ────────────────
GDPR_ARTICLES: dict[str, str] = {
    "5":  "Article 5 — Principles: lawfulness/fairness/transparency, purpose limitation, data minimisation, accuracy, storage limitation, integrity/confidentiality, accountability.",
    "6":  "Article 6 — Lawful bases: (a) consent, (b) contract, (c) legal obligation, (d) vital interests, (e) public task, (f) legitimate interests.",
    "7":  "Article 7 — Consent conditions: freely given, specific, informed, unambiguous; as easy to withdraw as to give; documented.",
    "9":  "Article 9 — Special categories (health, biometric, racial/ethnic, etc.) prohibited unless explicit consent or specific exception applies.",
    "13": "Article 13 — Transparency: controller must provide identity, DPO contact, purposes, legal basis, retention, data subject rights at time of collection.",
    "17": "Article 17 — Right to erasure ('right to be forgotten'): erase when no longer necessary, consent withdrawn, or unlawfully processed.",
    "20": "Article 20 — Data portability: receive data in structured machine-readable format; transmit to another controller where processing is consent- or contract-based and automated.",
    "25": "Article 25 — Privacy by design and by default: implement data protection principles technically from the outset; default to minimum necessary data.",
    "28": "Article 28 — Processors: must have a Data Processing Agreement; processor only acts on controller instructions.",
    "32": "Article 32 — Security: pseudonymisation, encryption, resilience, restoration capability, regular testing — appropriate to the risk.",
    "33": "Article 33 — Breach notification to supervisory authority within 72 hours of becoming aware, unless unlikely to result in risk.",
    "35": "Article 35 — DPIA required before high-risk processing (profiling, special categories at scale, systematic public monitoring).",
    "37": "Article 37 — DPO mandatory for public authorities, large-scale systematic monitoring, or large-scale special category processing.",
    "44": "Article 44 — Third-country transfers only with adequacy decision, appropriate safeguards (SCCs, BCRs), or specific derogations.",
}


@tool
def search_gdpr_documents(query: str) -> str:
    """Search the GDPR knowledge base for relevant legal text and guidance.

    Use when you need specific provisions, definitions, or guidance from the
    ingested GDPR documents (regulation text, EDPB guidelines, DPA decisions).
    Returns the top matching passages with source citations.

    Args:
        query: Natural language search, e.g. 'conditions for valid consent'
               or 'when is a DPIA required'.
    """
    store = _get_store()
    docs = store.similarity_search(query, k=settings.retrieval_top_k)

    if not docs:
        return "No relevant documents found."

    sections = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        label = f"[{i}] {source}" + (f", p.{page}" if page else "")
        sections.append(f"{label}\n{doc.page_content}")

    return "\n\n---\n\n".join(sections)


@tool
def web_search(query: str) -> str:
    """Search the web for recent GDPR news, enforcement actions, and regulatory updates.

    Use when you need information not in the ingested documents — recent DPA
    decisions, new EDPB guidelines, news about fines, country-specific rules.

    Args:
        query: e.g. 'GDPR fines 2024 largest' or 'EDPB guidance AI 2025'.
    """
    try:
        from langchain_community.tools import DuckDuckGoSearchRun
        return DuckDuckGoSearchRun().run(query)
    except Exception as e:
        return f"Web search failed: {e}"


@tool
def get_gdpr_article(article_number: str) -> str:
    """Get the key provisions of a specific GDPR article by number.

    Use when you know which article is relevant and want a quick summary —
    faster than searching the full document store. Covers Articles 5–7, 9,
    13, 17, 20, 25, 28, 32, 33, 35, 37, 44.

    Args:
        article_number: The article number as a string, e.g. '6' or '35'.
    """
    num = article_number.strip().lstrip("0") or "0"
    if num in GDPR_ARTICLES:
        return GDPR_ARTICLES[num]
    available = ", ".join(sorted(GDPR_ARTICLES.keys(), key=int))
    return (
        f"Article {num} not in quick-reference index (available: {available}). "
        f"Use search_gdpr_documents for full text."
    )


TOOLS = [search_gdpr_documents, web_search, get_gdpr_article]
