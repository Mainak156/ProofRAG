from __future__ import annotations

from pathlib import Path

import pytest

from proofrag.models import (
    Answer,
    Citation,
    Evidence,
    ProofRAGRequest,
    VerificationStatus,
)
from proofrag.pipeline import (
    PipelineError,
    ProofRAG,
    _build_response,
)


def create_document(
    directory: Path,
    name: str,
    content: str,
) -> Path:
    path = directory / name
    path.write_text(
        content,
        encoding="utf-8",
    )
    return path


def test_build_response_for_verified_answer() -> None:
    citation = Citation(
        citation_id="citation-1",
        evidence_id="evidence-1",
        filename="manual.pdf",
        page_number=2,
        snippet="The system supports PDF ingestion.",
        verification_status=VerificationStatus.VERIFIED,
        keyword_score=1.0,
        embedding_score=0.95,
    )

    answer = Answer(
        text="The system supports PDF ingestion.",
        citations=[citation],
        refused=False,
    )

    response = _build_response(
        request=ProofRAGRequest(
            question="What does the system support?"
        ),
        answer=answer,
        evidence_count=1,
    )

    assert response.answer == (
        "The system supports PDF ingestion."
    )
    assert response.verified is True
    assert response.refused is False
    assert response.evidence_count == 1
    assert len(response.citations) == 1


def test_build_response_for_refusal() -> None:
    answer = Answer(
        text="Not found in the documents.",
        citations=[],
        refused=True,
        refusal_reason="No supporting evidence.",
    )

    response = _build_response(
        request=ProofRAGRequest(
            question="What is the company's revenue?"
        ),
        answer=answer,
        evidence_count=0,
    )

    assert response.answer == "Not found in the documents."
    assert response.verified is False
    assert response.refused is True
    assert response.evidence_count == 0
    assert response.refusal_reason == "No supporting evidence."


def test_pipeline_rejects_missing_documents_directory(
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "does-not-exist"

    pipeline = ProofRAG(missing_path)

    with pytest.raises(
        PipelineError,
        match="does not exist",
    ):
        pipeline.ingest()


def test_pipeline_rejects_empty_documents_directory(
    tmp_path: Path,
) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()

    pipeline = ProofRAG(documents)

    with pytest.raises(
        PipelineError,
        match="No supported documents found",
    ):
        pipeline.ingest()


def test_pipeline_ingests_markdown_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()

    create_document(
        documents,
        "manual.md",
        """
        # ProofRAG Manual

        ProofRAG supports PDF document ingestion.

        It verifies citations using keyword and embedding similarity.
        """,
    )

    pipeline = ProofRAG(documents)

    monkeypatch.setattr(
        "proofrag.pipeline.add_chunks",
        lambda chunks: [
            chunk.chunk_id
            for chunk in chunks
        ],
    )

    indexed_documents = pipeline.ingest()

    assert len(indexed_documents) == 1
    assert indexed_documents[0].filename == "manual.md"


def test_pipeline_ask_refuses_when_no_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = ProofRAG(tmp_path)

    monkeypatch.setattr(
        "proofrag.pipeline.retrieve",
        lambda *args, **kwargs: [],
    )

    response = pipeline.ask(
        "What is the company's revenue?"
    )

    assert response.refused is True
    assert response.verified is False
    assert response.answer == "Not found in the documents."
    assert response.evidence_count == 0
    assert response.citations == []


def test_pipeline_ask_runs_generation_and_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = ProofRAG(tmp_path)

    evidence = Evidence(
        evidence_id="evidence-1",
        chunk_id="chunk-1",
        document_id="doc-1",
        filename="manual.md",
        page_number=1,
        snippet="ProofRAG supports PDF ingestion.",
        retrieval_score=0.91,
    )

    generated_answer = Answer(
        text="ProofRAG supports PDF ingestion.",
        citations=[
            Citation(
                citation_id="citation-1",
                evidence_id="evidence-1",
                filename="manual.md",
                page_number=1,
                snippet="ProofRAG supports PDF ingestion.",
                verification_status=VerificationStatus.NOT_CHECKED,
            )
        ],
        refused=False,
    )

    verified_answer = generated_answer.model_copy(
        deep=True,
    )

    verified_answer.citations[0].verification_status = (
        VerificationStatus.VERIFIED
    )
    verified_answer.citations[0].keyword_score = 1.0
    verified_answer.citations[0].embedding_score = 0.95

    monkeypatch.setattr(
        "proofrag.pipeline.retrieve",
        lambda *args, **kwargs: [evidence],
    )

    class FakeGenerator:
        def generate(self, question, evidence):
            assert question == "What does ProofRAG support?"
            assert len(evidence) == 1
            return generated_answer

    pipeline.generator = FakeGenerator()

    monkeypatch.setattr(
        "proofrag.pipeline.verify_answer",
        lambda answer: verified_answer,
    )

    response = pipeline.ask(
        "What does ProofRAG support?"
    )

    assert response.answer == (
        "ProofRAG supports PDF ingestion."
    )
    assert response.verified is True
    assert response.refused is False
    assert response.evidence_count == 1
    assert len(response.citations) == 1
    assert (
        response.citations[0].verification_status
        == VerificationStatus.VERIFIED
    )


def test_pipeline_refuses_when_verification_rejects_answer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = ProofRAG(tmp_path)

    evidence = Evidence(
        evidence_id="evidence-1",
        chunk_id="chunk-1",
        document_id="doc-1",
        filename="manual.md",
        page_number=1,
        snippet="ProofRAG supports PDF ingestion.",
        retrieval_score=0.91,
    )

    generated_answer = Answer(
        text="The company has five international offices.",
        citations=[
            Citation(
                citation_id="citation-1",
                evidence_id="evidence-1",
                filename="manual.md",
                page_number=1,
                snippet="ProofRAG supports PDF ingestion.",
                verification_status=VerificationStatus.NOT_CHECKED,
            )
        ],
        refused=False,
    )

    monkeypatch.setattr(
        "proofrag.pipeline.retrieve",
        lambda *args, **kwargs: [evidence],
    )

    class FakeGenerator:
        def generate(self, question, evidence):
            return generated_answer

    pipeline.generator = FakeGenerator()

    response = pipeline.ask(
        "What is the company's office count?"
    )

    assert response.refused is True
    assert response.verified is False
    assert response.answer == "Not found in the documents."
    assert response.evidence_count == 1


def test_ingest_and_ask_calls_both_steps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = ProofRAG(tmp_path)

    calls: list[str] = []

    monkeypatch.setattr(
        pipeline,
        "ingest",
        lambda: calls.append("ingest"),
    )

    expected = _build_response(
        request=ProofRAGRequest(
            question="Test question"
        ),
        answer=Answer(
            text="Not found in the documents.",
            citations=[],
            refused=True,
            refusal_reason="No evidence.",
        ),
        evidence_count=0,
    )

    monkeypatch.setattr(
        pipeline,
        "ask",
        lambda question: calls.append("ask") or expected,
    )

    result = pipeline.ingest_and_ask(
        "Test question"
    )

    assert calls == ["ingest", "ask"]
    assert result == expected