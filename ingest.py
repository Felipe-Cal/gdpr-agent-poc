"""
Ingest GDPR documents into Qdrant.

Run once before starting the app (or whenever you add new documents):
    python ingest.py

Reads PDFs from data/gdpr_docs/, chunks them, embeds with OpenAI
text-embedding-3-small, and upserts into Qdrant.

Cost: ~$0.001 for the full GDPR corpus (~50K tokens). Idempotent — safe to
re-run; Qdrant upserts by document ID so existing chunks are overwritten.
"""

from pathlib import Path

import typer
from langchain_community.document_loaders import PyPDFLoader
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from config import get_embeddings, settings

console = Console()
DOCS_DIR = Path("data/gdpr_docs")


def ingest():
    pdf_files = list(DOCS_DIR.glob("**/*.pdf"))
    if not pdf_files:
        console.print(f"[red]No PDFs found in {DOCS_DIR}[/red]")
        console.print("[dim]Drop GDPR PDFs into data/gdpr_docs/ and re-run.[/dim]")
        raise typer.Exit(1)

    console.print(f"[cyan]Found {len(pdf_files)} PDF(s)[/cyan]")

    # ── 1. Load and chunk ────────────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " "],
    )

    docs = []
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
        task = p.add_task("Loading PDFs...", total=len(pdf_files))
        for pdf in pdf_files:
            loader = PyPDFLoader(str(pdf))
            pages = loader.load()
            chunks = splitter.split_documents(pages)
            # tag each chunk with its source filename for citations
            for chunk in chunks:
                chunk.metadata["source"] = pdf.name
            docs.extend(chunks)
            p.advance(task)

    console.print(f"[dim]{len(docs)} chunks from {len(pdf_files)} file(s)[/dim]")

    # ── 2. Qdrant collection ─────────────────────────────────────────────────
    embeddings = get_embeddings()

    # Probe vector size from the embedding model
    vector_size = len(embeddings.embed_query("probe"))

    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )

    # Recreate collection if it doesn't exist or has wrong vector size
    existing = {c.name: c for c in client.get_collections().collections}
    if settings.qdrant_collection in existing:
        info = client.get_collection(settings.qdrant_collection)
        existing_size = info.config.params.vectors.size
        if existing_size != vector_size:
            console.print(
                f"[yellow]Vector size mismatch ({existing_size} → {vector_size})."
                " Recreating collection.[/yellow]"
            )
            client.delete_collection(settings.qdrant_collection)
            existing.pop(settings.qdrant_collection)

    if settings.qdrant_collection not in existing:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        console.print(f"[green]Created collection:[/green] {settings.qdrant_collection}")
    else:
        console.print(f"[dim]Collection exists: {settings.qdrant_collection}[/dim]")

    # ── 3. Embed and store ───────────────────────────────────────────────────
    console.print("[cyan]Embedding and storing chunks...[/cyan]")
    QdrantVectorStore.from_documents(
        docs,
        embedding=embeddings,
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_name=settings.qdrant_collection,
    )

    console.print(f"[green]Done — {len(docs)} chunks stored in Qdrant[/green]")


if __name__ == "__main__":
    ingest()
