from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document as LangChainDocument

from .config import get_settings
from .models import Chunk, Evidence
from .okf import OKFBundle


class RetrievalError(Exception):
    """Raised when vector retrieval fails."""


def _chunk_to_langchain_document(
    chunk: Chunk,
) -> LangChainDocument:
    """Convert a ProofRAG Chunk into a LangChain Document."""

    metadata = {
        "chunk_id": chunk.chunk_id,
        "source_filename": chunk.source_filename,
        "document_id": chunk.document_id,
        "source_filename": chunk.source_filename,
        "page_number": chunk.page_number,
        "start_char": chunk.start_char,
        "end_char": chunk.end_char,
        **chunk.metadata,
    }

    return LangChainDocument(
        page_content=chunk.text,
        metadata=metadata,
    )


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    """Return the persistent ProofRAG Chroma vector store."""

    settings = get_settings()

    return Chroma(
        collection_name="proofrag_evidence",
        embedding_function=_get_embeddings(),
        persist_directory=str(settings.chroma_dir),
    )


def _get_embeddings():
    """Load the configured embedding model lazily."""

    from .embeddings import get_embedding_model

    return get_embedding_model()


def add_chunks(chunks: list[Chunk]) -> list[str]:
    """Add ProofRAG chunks to Chroma."""

    if not chunks:
        return []

    vector_store = get_vector_store()

    documents = [
        _chunk_to_langchain_document(chunk)
        for chunk in chunks
    ]

    ids = [chunk.chunk_id for chunk in chunks]

    try:
        return vector_store.add_documents(
            documents=documents,
            ids=ids,
        )
    except Exception as exc:
        raise RetrievalError(
            "Failed to add chunks to Chroma."
        ) from exc


def delete_chunks(chunk_ids: list[str]) -> None:
    """Delete chunks from Chroma."""

    if not chunk_ids:
        return

    try:
        get_vector_store().delete(ids=chunk_ids)
    except Exception as exc:
        raise RetrievalError(
            "Failed to delete chunks from Chroma."
        ) from exc


def clear_collection() -> None:
    """Clear the persistent ProofRAG Chroma collection."""

    vector_store = get_vector_store()

    try:
        vector_store.delete_collection()
    except Exception as exc:
        raise RetrievalError(
            "Failed to clear Chroma collection."
        ) from exc
    finally:
        get_vector_store.cache_clear()


def retrieve(
    question: str,
    *,
    top_k: int | None = None,
    score_threshold: float | None = None,
    okf_bundle: OKFBundle | None = None,
) -> list[Evidence]:
    """
    Retrieve evidence using Chroma and optionally validate it against OKF.

    Chroma performs semantic retrieval.

    OKF provides the provenance/evidence layer:
    - chunk existence
    - source
    - page
    - original evidence text
    """

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    settings = get_settings()

    k = (
        top_k
        if top_k is not None
        else settings.retrieval_top_k
    )

    threshold = (
        score_threshold
        if score_threshold is not None
        else settings.retrieval_score_threshold
    )

    if k <= 0:
        raise ValueError("top_k must be greater than zero.")

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "score_threshold must be between 0 and 1."
        )

    try:
        results = get_vector_store().similarity_search_with_score(
            query=question,
            k=k,
        )
    except Exception as exc:
        raise RetrievalError(
            "Failed to retrieve evidence from Chroma."
        ) from exc

    evidence: list[Evidence] = []
    seen_chunk_ids: set[str] = set()

    for index, (document, distance) in enumerate(results):
        metadata = document.metadata

        chunk_id = str(
            metadata.get("chunk_id", "")
        )

        if not chunk_id or chunk_id in seen_chunk_ids:
            continue

        relevance = _distance_to_relevance(distance)

        if relevance < threshold:
            continue

        verified_metadata = _resolve_okf_metadata(
            chunk_id=chunk_id,
            metadata=metadata,
            okf_bundle=okf_bundle,
        )

        if verified_metadata is None:
            continue

        seen_chunk_ids.add(chunk_id)

        evidence.append(
            Evidence(
                evidence_id=(
                    f"evidence-{index}-{chunk_id}"
                ),
                chunk_id=chunk_id,
                document_id=str(
                    verified_metadata.get(
                        "document_id",
                        metadata.get("document_id", ""),
                    )
                ),
                filename=str(
                    verified_metadata.get(
                        "filename",
                        metadata.get(
                            "source_filename",
                            "unknown",
                        ),
                    )
                ),
                page_number=_optional_int(
                    verified_metadata.get(
                        "page_number",
                        metadata.get("page_number"),
                    )
                ),
                snippet=str(
                    verified_metadata.get(
                        "snippet",
                        document.page_content,
                    )
                ),
                retrieval_score=relevance,
                metadata={
                    **metadata,
                    **verified_metadata,
                },
            )
        )

    evidence.sort(
        key=lambda item: item.retrieval_score,
        reverse=True,
    )

    return evidence


def _resolve_okf_metadata(
    *,
    chunk_id: str,
    metadata: dict,
    okf_bundle: OKFBundle | None,
) -> dict | None:
    """
    Resolve authoritative evidence metadata from OKF.

    If an OKF bundle is supplied, the chunk must exist in it.
    This prevents Chroma from becoming the sole source of truth.
    """

    if okf_bundle is None:
        return {
            "document_id": metadata.get(
                "document_id",
                "",
            ),
            "filename": metadata.get(
                "source_filename",
                "unknown",
            ),
            "page_number": metadata.get(
                "page_number"
            ),
            "snippet": metadata.get(
                "text",
                "",
            ),
        }

    try:
        frontmatter, body = okf_bundle.read_chunk(
            chunk_id
        )
    except FileNotFoundError:
        return None

    proofrag_metadata = frontmatter.get(
        "proofrag",
        {},
    )

    sources = frontmatter.get(
        "sources",
        [],
    )

    source = (
        sources[0]
        if sources and isinstance(sources[0], dict)
        else {}
    )

    return {
        "document_id": proofrag_metadata.get(
            "document_id",
            metadata.get("document_id", ""),
        ),
        "filename": (
            proofrag_metadata.get("source_filename")
            or metadata.get("source_filename")
            or source.get("title")
            or "unknown"
        ),
        "page_number": proofrag_metadata.get(
            "page_number",
            metadata.get("page_number"),
        ),
        "snippet": body,
        "okf_node": frontmatter,
    }


def _distance_to_relevance(
    distance: float,
) -> float:
    """Convert Chroma distance into a bounded relevance score."""

    relevance = 1.0 - float(distance)

    return max(
        0.0,
        min(
            1.0,
            relevance,
        ),
    )


def _optional_int(
    value: object,
) -> int | None:
    """Safely convert metadata to an optional integer."""

    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None