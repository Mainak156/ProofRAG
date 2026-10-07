from __future__ import annotations

import re
from collections import Counter
from math import sqrt

from nltk.corpus import stopwords

from .config import get_settings
from .embeddings import embed_text
from .models import (
    Answer,
    Citation,
    VerificationResult,
    VerificationStatus,
)


class VerificationError(Exception):
    """Raised when citation verification cannot be completed."""


ENGLISH_STOPWORDS = frozenset(stopwords.words("english"))


def normalize_text(text: str) -> str:
    """Normalize text for keyword comparison."""

    return re.sub(
        r"\s+",
        " ",
        re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower()),
    ).strip()


def tokenize(text: str) -> list[str]:
    """Tokenize text and remove NLTK English stopwords."""

    normalized = normalize_text(text)

    if not normalized:
        return []

    return [
        token
        for token in normalized.split()
        if token not in ENGLISH_STOPWORDS
    ]


def keyword_overlap(answer: str, evidence: str) -> float:
    """
    Measure the proportion of meaningful answer vocabulary
    that appears in the supporting evidence.
    """

    answer_tokens = tokenize(answer)
    evidence_tokens = tokenize(evidence)

    if not answer_tokens or not evidence_tokens:
        return 0.0

    answer_counts = Counter(answer_tokens)
    evidence_vocabulary = set(evidence_tokens)

    matched = sum(
        count
        for token, count in answer_counts.items()
        if token in evidence_vocabulary
    )

    return matched / len(answer_tokens)


def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:
    """Calculate cosine similarity between two vectors."""

    if not vector_a or not vector_b:
        return 0.0

    if len(vector_a) != len(vector_b):
        raise ValueError(
            "Vectors must have the same dimensionality."
        )

    dot_product = sum(
        a * b
        for a, b in zip(vector_a, vector_b)
    )

    norm_a = sqrt(
        sum(value * value for value in vector_a)
    )

    norm_b = sqrt(
        sum(value * value for value in vector_b)
    )

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    similarity = dot_product / (norm_a * norm_b)

    return max(-1.0, min(1.0, similarity))


def embedding_overlap(
    answer: str,
    evidence: str,
) -> float:
    """Measure semantic similarity using the configured embedding model."""

    if not answer.strip() or not evidence.strip():
        return 0.0

    answer_vector = embed_text(answer)
    evidence_vector = embed_text(evidence)

    return max(
        0.0,
        cosine_similarity(answer_vector, evidence_vector),
    )


def verify_citation(
    answer: str,
    citation: Citation,
    *,
    keyword_threshold: float | None = None,
    embedding_threshold: float | None = None,
) -> VerificationResult:
    """Verify whether a citation actually supports the answer."""

    settings = get_settings()

    keyword_threshold = (
        settings.citation_keyword_threshold
        if keyword_threshold is None
        else keyword_threshold
    )

    embedding_threshold = (
        settings.citation_embedding_threshold
        if embedding_threshold is None
        else embedding_threshold
    )

    if not answer.strip():
        return VerificationResult(
            citation_id=citation.citation_id,
            status=VerificationStatus.REJECTED,
            keyword_score=0.0,
            embedding_score=0.0,
            reason="The answer is empty.",
        )

    if not citation.snippet.strip():
        return VerificationResult(
            citation_id=citation.citation_id,
            status=VerificationStatus.REJECTED,
            keyword_score=0.0,
            embedding_score=0.0,
            reason="The citation contains no supporting text.",
        )

    try:
        keyword_score = keyword_overlap(
            answer,
            citation.snippet,
        )

        embedding_score = embedding_overlap(
            answer,
            citation.snippet,
        )
    except Exception as exc:
        raise VerificationError(
            "Citation verification failed while calculating similarity."
        ) from exc

    keyword_pass = keyword_score >= keyword_threshold
    embedding_pass = embedding_score >= embedding_threshold

    if keyword_pass and embedding_pass:
        status = VerificationStatus.VERIFIED
        reason = "Keyword and embedding checks both passed."

    elif keyword_pass or embedding_pass:
        status = VerificationStatus.PARTIALLY_VERIFIED
        reason = "Only one verification signal passed."

    else:
        status = VerificationStatus.REJECTED
        reason = "Neither verification signal passed."

    return VerificationResult(
        citation_id=citation.citation_id,
        status=status,
        keyword_score=keyword_score,
        embedding_score=embedding_score,
        reason=reason,
    )


def verify_answer(answer: Answer) -> Answer:
    """
    Verify every citation in an answer.

    The answer is accepted only when every citation passes
    both keyword and embedding verification.
    """

    if answer.refused:
        return answer

    if not answer.citations:
        return Answer(
            text="Not found in the documents.",
            citations=[],
            refused=True,
            refusal_reason="The generated answer contains no citations.",
        )

    verified_citations: list[Citation] = []

    for citation in answer.citations:
        result = verify_citation(
            answer.text,
            citation,
        )

        verified_citations.append(
            citation.model_copy(
                update={
                    "verification_status": result.status,
                    "keyword_score": result.keyword_score,
                    "embedding_score": result.embedding_score,
                }
            )
        )

    all_verified = all(
        citation.verification_status
        == VerificationStatus.VERIFIED
        for citation in verified_citations
    )

    if all_verified:
        return Answer(
            text=answer.text,
            citations=verified_citations,
            refused=False,
            refusal_reason=None,
        )

    return Answer(
        text="Not found in the documents.",
        citations=[
            citation
            for citation in verified_citations
            if citation.verification_status
            != VerificationStatus.REJECTED
        ],
        refused=True,
        refusal_reason=(
            "The supplied citations did not sufficiently support "
            "the generated answer."
        ),
    )