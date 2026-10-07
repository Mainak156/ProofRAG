from __future__ import annotations

import pytest

from proofrag.models import (
    Answer,
    Citation,
    VerificationStatus,
)
from proofrag.verifier import (
    cosine_similarity,
    keyword_overlap,
    normalize_text,
    tokenize,
    verify_answer,
    verify_citation,
)


def make_citation(
    snippet: str,
    citation_id: str = "citation-1",
) -> Citation:
    return Citation(
        citation_id=citation_id,
        evidence_id="evidence-1",
        filename="manual.pdf",
        page_number=3,
        snippet=snippet,
        verification_status=VerificationStatus.NOT_CHECKED,
    )


def make_invalid_empty_snippet_citation() -> Citation:
    """
    Construct malformed citation data only to test the verifier's
    defensive handling of an empty supporting snippet.
    """

    return Citation.model_construct(
        citation_id="citation-1",
        evidence_id="evidence-1",
        filename="manual.pdf",
        page_number=3,
        snippet="",
        verification_status=VerificationStatus.NOT_CHECKED,
        keyword_score=0.0,
        embedding_score=0.0,
    )


def test_normalize_text() -> None:
    result = normalize_text(
        "  Hello, WORLD!  This is a TEST. "
    )

    assert result == "hello world this is a test"


def test_tokenize() -> None:
    result = tokenize(
        "The ProofRAG system verifies citations."
    )

    assert result == [
        "proofrag",
        "system",
        "verifies",
        "citations",
    ]


def test_keyword_overlap_full_match() -> None:
    score = keyword_overlap(
        "The system supports PDF ingestion",
        "The system supports PDF ingestion.",
    )

    assert score == 1.0


def test_keyword_overlap_partial_match() -> None:
    score = keyword_overlap(
        "The system supports PDF ingestion",
        "The system supports Markdown ingestion.",
    )

    assert 0.0 < score < 1.0


def test_keyword_overlap_no_match() -> None:
    score = keyword_overlap(
        "The company has five offices",
        "The system supports PDF ingestion.",
    )

    assert score == 0.0


def test_cosine_similarity_identical_vectors() -> None:
    score = cosine_similarity(
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    )

    assert score == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors() -> None:
    score = cosine_similarity(
        [1.0, 0.0],
        [0.0, 1.0],
    )

    assert score == pytest.approx(0.0)


def test_cosine_similarity_rejects_dimension_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="same dimensionality",
    ):
        cosine_similarity(
            [1.0, 0.0],
            [1.0],
        )


def test_verify_citation_with_matching_content() -> None:
    citation = make_citation(
        "The system supports PDF ingestion.",
    )

    result = verify_citation(
        "The system supports PDF ingestion.",
        citation,
    )

    assert result.citation_id == "citation-1"
    assert result.status == VerificationStatus.VERIFIED
    assert result.keyword_score == pytest.approx(1.0)
    assert result.embedding_score > 0.60


def test_verify_citation_with_empty_snippet() -> None:
    citation = make_invalid_empty_snippet_citation()

    result = verify_citation(
        "The system supports PDF ingestion.",
        citation,
    )

    assert result.citation_id == "citation-1"
    assert result.status == VerificationStatus.REJECTED
    assert result.keyword_score == 0.0
    assert result.embedding_score == 0.0


def test_verify_citation_with_weak_content() -> None:
    citation = make_citation(
        "The weather is sunny today.",
    )

    result = verify_citation(
        "The system supports PDF ingestion.",
        citation,
        keyword_threshold=0.90,
        embedding_threshold=0.99,
    )

    assert result.citation_id == "citation-1"
    assert result.status == VerificationStatus.REJECTED


def test_verify_answer_updates_citation_status() -> None:
    citation = make_citation(
        "The system supports PDF ingestion.",
    )

    answer = Answer(
        text="The system supports PDF ingestion.",
        citations=[citation],
        refused=False,
    )

    result = verify_answer(answer)

    assert result.refused is False
    assert len(result.citations) == 1

    verified_citation = result.citations[0]

    assert (
        verified_citation.verification_status
        == VerificationStatus.VERIFIED
    )
    assert verified_citation.keyword_score == pytest.approx(1.0)
    assert verified_citation.embedding_score > 0.60


def test_verify_answer_refuses_unsupported_answer() -> None:
    citation = make_citation(
        "The weather is sunny today.",
    )

    answer = Answer(
        text="The company has five international offices.",
        citations=[citation],
        refused=False,
    )

    result = verify_answer(answer)

    assert result.refused is True
    assert result.text == "Not found in the documents."
    assert result.refusal_reason is not None


def test_verify_refused_answer_without_reprocessing() -> None:
    answer = Answer(
        text="Not found in the documents.",
        citations=[],
        refused=True,
        refusal_reason="No supporting evidence.",
    )

    result = verify_answer(answer)

    assert result == answer


def test_verify_answer_with_no_citations_refuses() -> None:
    answer = Answer(
        text="The system supports PDF ingestion.",
        citations=[],
        refused=False,
    )

    result = verify_answer(answer)

    assert result.refused is True
    assert result.text == "Not found in the documents."
    assert result.citations == []