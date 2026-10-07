from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from .config import get_settings


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Create and cache the ProofRAG embedding model.

    The model runs locally through sentence-transformers.
    """

    settings = get_settings()

    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={
            "device": "cpu",
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )


def embed_text(text: str) -> list[float]:
    """Generate an embedding for a single text."""

    if not text.strip():
        raise ValueError("Cannot embed empty text.")

    return get_embedding_model().embed_query(text)


def embed_documents(
    texts: list[str],
) -> list[list[float]]:
    """Generate embeddings for multiple documents."""

    if not texts:
        return []

    if any(not text.strip() for text in texts):
        raise ValueError(
            "Document texts cannot contain empty strings."
        )

    return get_embedding_model().embed_documents(texts)